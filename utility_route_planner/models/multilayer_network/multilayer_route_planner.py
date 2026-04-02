# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
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
        write_output: bool = True,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.write_output = write_output
        self.prefix = prefix

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

        straight_line = shapely.LineString(
            [
                self.cost_surface_graph.get_node_data(source).geometry,
                self.cost_surface_graph.get_node_data(target).geometry,
            ]
        )
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
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
                self.result_route_nodes,
                f"{self.prefix}multilayer_route_nodes",
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
                self.result_route_edges,
                f"{self.prefix}multilayer_route_edges",
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, self.line_target_source, f"{self.prefix}straight_line"
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
                self.result_route_linestring,
                f"{self.prefix}result_route",
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
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
        edge_line = shapely.geometry.LineString(
            [
                self.cost_surface_graph.get_node_data(node_1).geometry,
                self.cost_surface_graph.get_node_data(node_2).geometry,
            ]
        )
        distance = edge_line.distance(self.line_target_source) * modifier
        if distance > weight:
            logger.warning("Unexpected situation during routing.")
        return weight + distance

    def smooth_linestring(self, linestring: shapely.LineString):
        self.result_route_smoothed = linestring.simplify(15)
