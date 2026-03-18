# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import rustworkx as rx
import shapely
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import get_hexagon_edge_weight
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
        prefix: str = "",
        write_output: bool = True,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.write_output = write_output
        self.prefix = prefix

        self.result_route_node_indices: list[rx.NodeIndices] = []
        self.result_route_edges: gpd.GeoDataFrame = get_empty_geodataframe()
        self.result_route_nodes: gpd.GeoDataFrame = get_empty_geodataframe()

    @time_function
    def find_route(self, start_end: shapely.LineString):
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.distance(start).idxmin()
        target = self.gdf_cost_surface_nodes.distance(end).idxmin()

        # HexagonEdgeInfo.weight is used as edge weight for dijkstra
        path_node_indices = rx.dijkstra_shortest_paths(self.cost_surface_graph, source, target, get_hexagon_edge_weight)
        path_node_indices = path_node_indices[target]
        gdf_path_nodes = gpd.GeoDataFrame(
            data=[self.cost_surface_graph.get_node_data(i) for i in path_node_indices], crs=Config.CRS
        )

        edges = []
        for current, next_ in zip(path_node_indices, path_node_indices[1:]):
            edges.append(self.cost_surface_graph.get_edge_data(current, next_))
        gdf_path_edges = gpd.GeoDataFrame(data=edges, crs=Config.CRS)

        self.result_route_edges = gdf_path_edges
        self.result_route_nodes = gdf_path_nodes
        self.result_route_node_indices = path_node_indices

        # self.validate_connectivity()

        if self.write_output:
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, gdf_path_nodes, f"{self.prefix}multilayer_route_nodes"
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, gdf_path_edges, f"{self.prefix}multilayer_route_edges"
            )

    def get_result_route_length(self) -> float:
        return self.result_route_edges["length"].sum()

    def get_result_route_cost(self) -> float:
        return self.result_route_edges["weight"].sum()

    def validate_connectivity(self):
        merged = shapely.line_merge(self.result_route_edges.union_all())
        if not isinstance(merged, shapely.LineString):
            logger.warning("The route is not a single connected LineString, this can occur when it crosses itself.")
            # TODO check if parts of the multilinestring intersect each other
