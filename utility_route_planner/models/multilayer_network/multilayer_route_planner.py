# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import rustworkx as rx
import shapely
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import EdgeInfo
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring, get_empty_geodataframe
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class MultilayerRouteEngine:
    def __init__(
        self,
        cost_surface_graph: rx.PyGraph,
        osm_graph: rx.PyGraph,
        gdf_cost_surface_nodes: gpd.GeoDataFrame,
        hexagon_size: float,
        prefix: str = "",
        write_output: bool = False,
        out: Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.hexagon_size = hexagon_size
        self.write_output = write_output
        self.prefix = prefix
        self.out = out

        self.line_target_source: shapely.geometry.LineString = shapely.geometry.LineString()
        self.result_route_node_indices: list[rx.NodeIndices] = []
        self.result_route_edges: gpd.GeoDataFrame = get_empty_geodataframe()
        self.result_route_nodes: gpd.GeoDataFrame = get_empty_geodataframe()
        self.result_route_linestring: shapely.LineString = shapely.LineString()
        self.result_route_smoothed: shapely.LineString = shapely.geometry.LineString()

    @time_function
    def find_route(self, start_end: shapely.LineString):
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.distance(start).idxmin()
        target = self.gdf_cost_surface_nodes.distance(end).idxmin()

        straight_line = self.get_linestring(source, target)
        # Offset to avoid it being exactly on top of the nodes, causes issues with distance calculations during routing.
        self.line_target_source = shapely.offset_curve(straight_line, Config.HEXAGON_SIZE / 4)

        path_node_indices = rx.dijkstra_shortest_paths(self.cost_surface_graph, source, target, self.get_weight)
        path_node_indices = path_node_indices[target]

        gdf_path_nodes = gpd.GeoDataFrame(
            data=[self.cost_surface_graph.get_node_data(i) for i in path_node_indices], crs=Config.CRS
        )
        # TODO i think this can be done more efficient
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
        self.smooth_linestring(self.result_route_linestring)

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
            write_results_to_geopackage(
                self.out, self.line_target_source, f"{self.prefix}straight_line"
            )
            write_results_to_geopackage(
                self.out,
                self.result_route_linestring,
                f"{self.prefix}result_route",
            )
            write_results_to_geopackage(
                self.out,
                self.result_route_smoothed,
                f"{self.prefix}result_route_smoothed",
            )

    def get_result_route_length(self) -> float:
        return self.result_route_edges["length"].sum()

    def get_result_route_cost(self) -> float:
        return self.result_route_edges["weight"].sum()

    def get_weight(self, edge: EdgeInfo, modifier: float = 0.01) -> float:
        """
        Weight is leading for edges (MCDA), but we want to add a small distance-based cost to prefer routes that are closer to the straight line between start and end.

        # TODO add logic for prioritizing special edges from piperamming?
        """
        weight = self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight
        node_1, node_2 = self.cost_surface_graph.get_edge_endpoints_by_index(edge.edge_id)
        edge_line = self.get_linestring(node_1, node_2)
        distance = edge_line.distance(self.line_target_source) * modifier
        if distance > weight:
            logger.warning("Unexpected situation during routing.")
        return weight + distance

    def get_linestring(self, node_1, node_2):
        edge_line = shapely.LineString(
            [
                self.cost_surface_graph.get_node_data(node_1).geometry,
                self.cost_surface_graph.get_node_data(node_2).geometry,
            ]
        )
        return edge_line

    def smooth_linestring(self, linestring: shapely.LineString):
        self.result_route_smoothed = linestring.simplify(self.hexagon_size)

        # Thoughts. Loop over the node indices. Check if we can skip a node if we create
        # a linestring from the next node in line without intersecting with a different
        # weight in the cost_surface. Keep trying till it is at the end node or continue
        # trying from the first node which does intersect with a different value.
        shortcut_order = []
        n_skip = 1
        for idx, node in enumerate(self.result_route_node_indices):
            next_node = self.result_route_node_indices[idx+1]
            basic_cost = self.cost_surface_graph.get_edge_data(node, next_node).weight

            shortcut_costs = basic_cost
            while shortcut_costs == basic_cost:
                n_skip += 1
                linestring = self.get_linestring(node, self.result_route_node_indices[idx+n_skip])
                # Note this does not work with height levels, we have to keep track of that
                shortcut_costs = self.gdf_cost_surface_nodes[self.gdf_cost_surface_nodes.dwithin(linestring, self.hexagon_size * 0.50)].suitability_value.unique().tolist()
                if len(shortcut_costs) != 1:
                    # TODO do something here, continue with the previous node and try to skip from there
                    break
                else:
                    shortcut_costs = shortcut_costs[0]

        write_results_to_geopackage(self.out, linestring, "pytest_linestring_skip")




