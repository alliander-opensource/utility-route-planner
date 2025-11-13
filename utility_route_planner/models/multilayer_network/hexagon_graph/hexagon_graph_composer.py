# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import shapely
import geopandas as gpd
import rustworkx as rx
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeHeightLevelInfo
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.util.geo_utilities import get_empty_geodataframe
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class HexagonGraphComposer:
    def __init__(
        self,
        processed_criteria_per_height_level: dict[int, list[str]],
        processed_graphs_per_height_level: dict[int, rx.PyGraph],
        hexagon_size: float,
        gdf_osm_edges: gpd.GeoDataFrame = get_empty_geodataframe(),
        debug: bool = False,
    ):
        self.processed_criteria_per_height_level = processed_criteria_per_height_level
        self.processed_graphs_per_height_level = processed_graphs_per_height_level
        self.hexagon_size = hexagon_size
        self.gdf_osm_edges = gdf_osm_edges
        self.debug = debug

        self.gdf_main_nodes = convert_hexagon_graph_to_gdfs(processed_graphs_per_height_level[0], edges=False)
        self.validate_input()

    def validate_input(self):
        """Doublecheck the main height level is 0. This is the expected value for the BGT."""
        node_count = {height: graph.num_nodes() for height, graph in self.processed_graphs_per_height_level.items()}
        main_height_level = max(node_count, key=node_count.get)
        if main_height_level != 0:
            raise ValueError(f"Main height level should be 0, but found {main_height_level} instead.")

    def compose(self):
        for height, graph in self.processed_graphs_per_height_level.items():
            if height == 0:
                continue
            gdf_nodes_height = convert_hexagon_graph_to_gdfs(graph, edges=False)
            logger.info(
                f"Connecting {rx.number_connected_components(self.processed_graphs_per_height_level[0])} subgraphs to the main graph."
            )

            # Determine which nodes to connect to each other
            nodes_to_add = {}  # TODO fill iteratively
            for component in rx.connected_components(graph):
                gdf_component = gdf_nodes_height[gdf_nodes_height["node_id"].isin(component)]
                # Get the outer nodes (nodes to join to the main graph) of the component.
                component_area = gdf_component.buffer(self.hexagon_size).union_all(grid_size=0.1)
                assert isinstance(component_area, shapely.Polygon)
                gdf_component_outer = gdf_component[
                    gdf_component.geometry.dwithin(component_area.boundary, self.hexagon_size)
                ]
                pairs = gdf_component_outer.sjoin(
                    self.gdf_main_nodes[~self.gdf_main_nodes.intersects(component_area)],
                    distance=self.hexagon_size * 2,
                    how="left",
                    predicate="dwithin",
                )

                # TODO filter pairs based on osm road (if available)

                nodes_to_add = {
                    row.node_id_right: (
                        row.node_id_left,
                        HexagonEdgeHeightLevelInfo(
                            edge_id=0,
                            weight=(row.suitability_value_left + row.suitability_value_right) / 2,
                            height_level=height,
                            length=1,
                            geometry=shapely.LineString(
                                [
                                    row.geometry,
                                    self.gdf_main_nodes.loc[
                                        self.gdf_main_nodes["node_id"] == row.node_id_right, "geometry"
                                    ].iloc[0],
                                ]
                            ),
                        ),
                    )
                    for idx, row in pairs.iterrows()
                }

                # connect nodes between height levels
                _ = self.processed_graphs_per_height_level[0].compose(graph, nodes_to_add)
                # TODO remap/reindex nodes/edges?

        if self.debug:
            out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
            write_results_to_geopackage(out, component_area, "pytest_component_area")
            write_results_to_geopackage(out, component_area.boundary, "pytest_component_area_boundary")
            write_results_to_geopackage(out, gdf_component_outer, "pytest_component_outer_nodes")
            # visualize the pairs / edges to be
            linestrings = pairs.apply(
                lambda x: shapely.LineString(
                    [
                        x.geometry,
                        self.gdf_main_nodes.loc[self.gdf_main_nodes["node_id"] == x.node_id_right, "geometry"].iloc[0],
                    ]
                ),
                axis=1,
            )
            write_results_to_geopackage(out, linestrings, "pytest_component_connection_lines", overwrite=True)
