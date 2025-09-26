# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import rustworkx as rx
import shapely
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring
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
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.prefix = prefix

    @time_function
    def find_route(self, start_end: shapely.LineString):
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.distance(start).idxmin()
        target = self.gdf_cost_surface_nodes.distance(end).idxmin()

        path = rx.dijkstra_shortest_paths(self.cost_surface_graph, source, target, lambda x: x.weight)
        path = path[target]
        path_points = shapely.MultiPoint([self.cost_surface_graph.get_node_data(i).geometry for i in path])
        edges = []
        for current, next_ in zip(path, path[1:]):
            edges.append(self.cost_surface_graph.get_edge_data(current, next_).geometry)

        result_linestring = shapely.MultiLineString(edges)

        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, result_linestring, f"{self.prefix}multilayer_route_edges"
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, path_points, f"{self.prefix}multilayer_route_points"
        )
