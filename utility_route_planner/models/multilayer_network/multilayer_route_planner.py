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
from utility_route_planner.models.multilayer_network.graph_datastructures import BaseWeightedEdgeInfo, NodeInfo
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    get_hexagon_edge_geometries_for_path,
)
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

        gdf_path_nodes = self.gdf_cost_surface_nodes.loc[self.gdf_cost_surface_nodes["node_id"].isin(path_node_indices)]
        gdf_path_edges = get_hexagon_edge_geometries_for_path(
            self.cost_surface_graph, path_node_indices, gdf_path_nodes
        )

        self.result_route_edges = gdf_path_edges
        self.result_route_nodes = gdf_path_nodes
        self.result_route_node_indices = path_node_indices

        self.result_route_linestring = shapely.LineString(
            [
                self.gdf_cost_surface_nodes.loc[self.gdf_cost_surface_nodes["node_id"] == i].geometry.values[0]
                for i in path_node_indices
            ]
        )
        # self.result_route_straightened, self.result_route_straightened_node_indices = self.straighten_linestring()

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
        source = self.gdf_cost_surface_nodes.loc[self.gdf_cost_surface_nodes.distance(start).idxmin(), "node_id"]
        target = self.gdf_cost_surface_nodes.loc[self.gdf_cost_surface_nodes.distance(end).idxmin(), "node_id"]
        if source == target:
            raise ValueError("Source and target node are the same. Provide a linestring with points further apart.")
        return source, target

    def get_result_route_length(self) -> float:
        return self.result_route_edges.geometry.length.sum()

    def get_result_route_cost(self) -> float:
        """
        For now, divide total route cost by 2 as the edge weight is now computed as the sum of the weights of the source
        and target nodes.
        """
        return self.result_route_edges["weight"].sum() / 2

    def get_weight_dijkstra(self, edge: BaseWeightedEdgeInfo, modifier: float = 0.01) -> float:
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

    def get_weight_astar(self, edge: BaseWeightedEdgeInfo) -> float:
        return self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight

    def get_estimate_astar(self, node: NodeInfo) -> float:
        node_point = self.gdf_cost_surface_nodes.loc[
            self.gdf_cost_surface_nodes["node_id"] == node.node_id
        ].geometry.values[0]
        guideline = shapely.LineString([node_point, shapely.get_point(self.result_route_guideline, 1)])

        return guideline.length

    def get_linestring(self, node_1: int, node_2: int) -> shapely.LineString:
        nodes = self.gdf_cost_surface_nodes
        edge_line = shapely.LineString(
            [
                nodes.loc[nodes["node_id"] == node_1].geometry.values[0],
                nodes.loc[nodes["node_id"] == node_2].geometry.values[0],
            ]
        )
        return edge_line

    def _get_shortcut_costs(self, line: shapely.LineString, inradius: int) -> tuple[float, float]:
        nearby = self.gdf_cost_surface_nodes[self.gdf_cost_surface_nodes.dwithin(line, inradius)]
        costs = nearby.suitability_value.unique().tolist()
        if len(costs) != 1:
            intersected = nearby[nearby.buffer(inradius / 2).intersects(line)]
            costs = intersected.suitability_value.unique().tolist()
        return costs

    def straighten_linestring(self) -> tuple[shapely.LineString, list[int]]:
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
        # center to center distance from a neighbouring hexagon
        inradius = math.sqrt(3) * self.hexagon_size
        shortcut_order: list = [self.result_route_node_indices[0]]

        # TODO add height
        # create dataframe with: node_order, suit value, height level. use to get segments.
        gdf_crossed_nodes = self.gdf_cost_surface_nodes[
            self.gdf_cost_surface_nodes["node_id"].isin(self.result_route_node_indices)
        ]
        gdf_crossed_nodes = gdf_crossed_nodes.set_index("node_id").loc[self.result_route_node_indices].reset_index()
        gdf_crossed_nodes["segment"] = (
            gdf_crossed_nodes["suitability_value"] != gdf_crossed_nodes["suitability_value"].shift()
        ).cumsum()

        # Note segments do not encapsulate pipe rammings as the costs are not on the node.
        for segment in gdf_crossed_nodes["segment"].unique():
            gdf_active_mask = gdf_crossed_nodes[gdf_crossed_nodes["segment"] == segment]
            start_node = int(gdf_active_mask.iloc[0]["node_id"])
            forwarded_node = int(gdf_active_mask.iloc[1]["node_id"]) if len(gdf_active_mask) > 0 else None
            end_node = int(gdf_active_mask.iloc[-1]["node_id"]) if len(gdf_active_mask) > 0 else None
            while start_node != end_node and end_node is not None:
                if len(gdf_active_mask) == 1:
                    # Only one node in this segment / remaining, no need to check for shortcuts.
                    # TODO not sure if this works, we cant come here because of the none check
                    shortcut_order.append(gdf_crossed_nodes[gdf_crossed_nodes["segment"] == segment].iloc[0]["node_id"])
                    continue

                # For each node in the active segment, create a line from start_node and compute shortcut costs.
                # Pick the last node (most skipped) with still the same suitability costs
                basic_cost = self.cost_surface_graph.get_edge_data(start_node, forwarded_node).weight
                start_node_geom = self.gdf_cost_surface_nodes.loc[
                    self.gdf_cost_surface_nodes["node_id"] == start_node
                ].geometry.values[0]
                # Create lines from start_node to all nodes in the active segment
                series_forwarded = gpd.GeoSeries(shapely.shortest_line(start_node_geom, gdf_active_mask["geometry"]))
                # Compute shortcut costs for each line
                series_shortcut_costs = series_forwarded.apply(self._get_shortcut_costs, inradius=inradius)

                # Filter nodes where shortcut costs equal basic_cost
                valid_nodes = gdf_active_mask[series_shortcut_costs.apply(lambda costs: costs == [basic_cost])]

                if valid_nodes.empty:
                    # No valid shortcut found for this part of the segment, move to the next node
                    shortcut_order.append(forwarded_node)
                    gdf_active_mask = gdf_active_mask[1:]
                    start_node = int(gdf_active_mask.iloc[0]["node_id"])
                    forwarded_node = gdf_active_mask.iloc[1]["node_id"] if len(gdf_active_mask) > 0 else end_node

                else:
                    # Pick the last valid node (most nodes skipped)
                    start_node = int(valid_nodes.iloc[-1]["node_id"])
                    shortcut_order.append(start_node)
                    gdf_active_mask = gdf_active_mask[gdf_active_mask.index > valid_nodes.iloc[-1].name]
                    forwarded_node = gdf_active_mask.iloc[0]["node_id"] if len(gdf_active_mask) > 0 else end_node

        shortcut_linestring = shapely.LineString(
            gdf_crossed_nodes[gdf_crossed_nodes["node_id"].isin(shortcut_order)].geometry.to_list()
        )

        logger.info(
            f"Input LineString: {self.result_route_linestring.length}. Shortcut LineString: {shortcut_linestring.length}."
        )

        return shortcut_linestring, shortcut_order

    def apply_bezier_curves(self):
        # TODO make sure the cost of the route remains valid when a curve passes through a node with different cell size
        pass
