# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from enum import auto, Enum
from pathlib import Path
import math

import rustworkx as rx
import shapely
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import EdgeInfo, NodeInfo
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring, get_empty_geodataframe
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class Algorithm(Enum):
    dijkstra = auto()
    astar = auto()


class MultilayerRouteEngine:
    def __init__(
        self,
        cost_surface_graph: rx.PyGraph,
        osm_graph: rx.PyGraph,
        gdf_cost_surface_nodes: gpd.GeoDataFrame,
        hexagon_size: float,
        algorithm: Algorithm = Algorithm.dijkstra,
        prefix: str = "",
        write_output: bool = False,
        out: Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.hexagon_size = hexagon_size
        self.write_output = write_output
        self.algorithm = algorithm
        self.prefix = prefix
        self.out = out

        self.result_route_node_indices: rx.NodeIndices = rx.NodeIndices()
        self.result_route_guideline: shapely.geometry.LineString = shapely.geometry.LineString()
        self.result_route_edges: gpd.GeoDataFrame = get_empty_geodataframe()
        self.result_route_nodes: gpd.GeoDataFrame = get_empty_geodataframe()
        self.result_route_linestring: shapely.LineString = shapely.LineString()
        self.result_route_straightened: shapely.LineString = shapely.geometry.LineString()
        self.result_route_straightened_node_indices: list = []

    @time_function
    def find_route(self, start_end: shapely.LineString):
        source, target = self.get_source_and_target_nodes(start_end)

        straight_line = self.get_linestring(source, target)
        # Offset to avoid it being exactly on top of the nodes, causes issues with distance calculations during routing.
        self.result_route_guideline = shapely.offset_curve(straight_line, self.hexagon_size / 4)

        match self.algorithm:
            case Algorithm.dijkstra:
                path_node_indices = rx.dijkstra_shortest_paths(
                    self.cost_surface_graph, source, target, self.get_weight_dijkstra
                )
                path_node_indices = path_node_indices[target]
            case Algorithm.astar:
                path_node_indices = rx.astar_shortest_path(
                    self.cost_surface_graph,
                    node=source,
                    goal_fn=lambda x: x.node_id == target,
                    edge_cost_fn=self.get_weight_astar,
                    estimate_cost_fn=self.get_estimate_astar,
                )
            case _:
                raise ValueError("Unsupported algorithm type.")

        gdf_path_nodes = gpd.GeoDataFrame(
            data=[self.cost_surface_graph.get_node_data(i) for i in path_node_indices], crs=Config.CRS
        )
        # TODO replace with new edge retrieval after merge
        edges = []
        for current, next_ in zip(path_node_indices, path_node_indices[1:]):
            edges.append(self.cost_surface_graph.get_edge_data(current, next_))
        gdf_path_edges = gpd.GeoDataFrame(data=edges, crs=Config.CRS)

        self.result_route_edges = gdf_path_edges
        self.result_route_nodes = gdf_path_nodes
        self.result_route_node_indices = path_node_indices

        self.result_route_linestring = shapely.LineString(
            [self.cost_surface_graph.get_node_data(i).geometry for i in path_node_indices]
        )
        self.result_route_straightened, self.result_route_straightened_node_indices = self.straighten_linestring()

        if self.write_output:
            write_results_to_geopackage(
                self.out,
                self.result_route_nodes,
                f"{self.prefix}multilayer_route_nodes",
            )
            write_results_to_geopackage(
                self.out,
                self.result_route_edges,
                f"{self.prefix}multilayer_route_edges",
            )
            write_results_to_geopackage(self.out, self.result_route_guideline, f"{self.prefix}guideline")
            write_results_to_geopackage(
                self.out,
                self.result_route_linestring,
                f"{self.prefix}result_route",
            )
            write_results_to_geopackage(
                self.out,
                self.result_route_straightened,
                f"{self.prefix}result_route_straightened",
            )
            write_results_to_geopackage(
                self.out,
                self.gdf_cost_surface_nodes[
                    self.gdf_cost_surface_nodes["node_id"].isin(self.result_route_straightened_node_indices)
                ],
                f"{self.prefix}result_route_shortcut_nodes",
            )

    def get_source_and_target_nodes(self, start_end: shapely.LineString) -> tuple[int, int]:
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.distance(start).idxmin()
        target = self.gdf_cost_surface_nodes.distance(end).idxmin()
        if source == target:
            raise ValueError("Source and target node are the same. Provide a linestring with points further apart.")
        return source, target

    def get_result_route_length(self) -> float:
        return self.result_route_edges["length"].sum()

    def get_result_route_cost(self) -> float:
        return self.result_route_edges["weight"].sum()

    def get_weight_dijkstra(self, edge: EdgeInfo, modifier: float = 0.01) -> float:
        """
        Weight is leading for edges (MCDA), but we want to add a small distance-based cost to prefer routes that are
        closer to the straight line between start and end.
        """
        weight = self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight
        node_1, node_2 = self.cost_surface_graph.get_edge_endpoints_by_index(edge.edge_id)
        edge_line = self.get_linestring(node_1, node_2)
        distance = edge_line.distance(self.result_route_guideline) * modifier
        if distance > weight:
            logger.warning("Unexpected situation during routing.")
        return weight + distance

    def get_weight_astar(self, edge: EdgeInfo) -> float:
        return self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight

    def get_estimate_astar(self, node: NodeInfo) -> float:
        node_point = self.cost_surface_graph.get_node_data(node.node_id).geometry
        guideline = shapely.LineString([node_point, shapely.get_point(self.result_route_guideline, 1)])

        return guideline.length

    def get_linestring(self, node_1: int, node_2: int) -> shapely.LineString:
        # TODO replace after merge
        edge_line = shapely.LineString(
            [
                self.cost_surface_graph.get_node_data(node_1).geometry,
                self.cost_surface_graph.get_node_data(node_2).geometry,
            ]
        )
        return edge_line

    def straighten_linestring(self, debug: bool = False) -> tuple[shapely.LineString, list[int]]:
        """
        The idea is to create shortcuts in the route by skipping nodes if the cost does not change. This is done by
        creating a linestring from the current node to the next node in the route and checking if the suitability values
        of the cost surface nodes that are within a certain distance (depending on hexagon size used) from this
        linestring are all the same as the suitability value of the current node. If they are, we can skip the nodes in
        between and continue from the forwarded node. If they are not, we add the last node that we could skip to the
        shortcut order and continue from there.

        This is effective when the route crosses an area that is homogenous in suitability value and reduces the
        zigzag effect.

        Illustrations: https://steamcdn-a.akamaihd.net/apps/valve/2009/ai_systems_of_l4d_mike_booth.pdf
        - AKA collapsed path.

        This results in a "straightened" linestring.

        """
        n_skip = 1
        start_idx = 0
        start_node = self.result_route_node_indices[start_idx]
        shortcut_order: list = [start_node]
        forwarded_node = self.result_route_node_indices[start_idx + 1]
        basic_cost = self.cost_surface_graph.get_edge_data(start_node, forwarded_node).weight
        shortcut_costs = basic_cost
        # center to center distance from a neighbouring hexagon
        dwithin_threshold = math.sqrt(3) * self.hexagon_size

        while basic_cost == shortcut_costs:
            n_skip += 1
            if start_idx + n_skip >= len(self.result_route_node_indices):
                shortcut_order.append(self.result_route_node_indices[-1])
                break

            forwarded_node = self.result_route_node_indices[start_idx + n_skip]
            forwarded_linestring = self.get_linestring(start_node, forwarded_node)
            # TODO this does not work with height levels, add after merge
            # TODO we route from hexagon center to center. dwithin might include nodes which are not crossed? Train of thoughts:
            #  If we use a hexagon_size of 0.5. the center-to-center distance == 0.865. If a straightened line is within half of that value, it might cross a different terrain type.
            nearby_nodes = self.gdf_cost_surface_nodes[
                self.gdf_cost_surface_nodes.dwithin(forwarded_linestring, dwithin_threshold)
            ]
            shortcut_costs = nearby_nodes.suitability_value.unique().tolist()

            if len(shortcut_costs) != 1:
                # Check if the forwarded linestring actually passes through the inradius of a node with higher cost
                intersected = nearby_nodes[nearby_nodes.buffer(dwithin_threshold / 2).intersects(forwarded_linestring)]
                shortcut_costs = intersected.suitability_value.unique().tolist()
                if debug:
                    write_results_to_geopackage(
                        self.out, forwarded_linestring, "pytest_forwarded_linestring", overwrite=True
                    )
                    write_results_to_geopackage(self.out, intersected, "pytest_intersected_nodes", overwrite=True)

                if len(shortcut_costs) == 1 and shortcut_costs[0] == basic_cost:
                    shortcut_costs = shortcut_costs[0]
                    continue

                shortcut_order.append(self.result_route_node_indices[start_idx + n_skip - 1])

                # reset from new point
                start_idx = start_idx + n_skip - 1
                start_node = self.result_route_node_indices[start_idx]
                n_skip = 1

                if start_idx + n_skip >= len(self.result_route_node_indices):
                    if start_node != self.result_route_node_indices[-1]:
                        shortcut_order.append(self.result_route_node_indices[-1])
                    break

                forwarded_node = self.result_route_node_indices[start_idx + n_skip]
                basic_cost = self.cost_surface_graph.get_edge_data(start_node, forwarded_node).weight
                shortcut_costs = basic_cost
            else:
                shortcut_costs = shortcut_costs[0]

        # Note we need to preserve order of shortcut nodes to create a valid linestring
        gdf_shortcut_nodes = self.gdf_cost_surface_nodes[self.gdf_cost_surface_nodes["node_id"].isin(shortcut_order)]
        gdf_shortcut_nodes = gdf_shortcut_nodes.set_index("node_id").loc[shortcut_order].reset_index()
        shortcut_linestring = shapely.LineString(gdf_shortcut_nodes.geometry.to_list())

        logger.info(
            f"Input LineString: {self.result_route_linestring.length}. Shortcut LineString: {shortcut_linestring.length}."
        )

        return shortcut_linestring, shortcut_order
