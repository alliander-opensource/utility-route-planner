# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
import pygeoops
import shapely
import geopandas as gpd
import rustworkx as rx
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeInfo
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.util.geo_utilities import get_empty_geodataframe
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class HexagonGraphComposer:
    def __init__(
        self,
        processed_criteria_per_height_level: dict[int, list[str]],
        processed_graphs_per_height_level: dict[int, rx.PyGraph],
        hexagon_size: float,
        debug: bool = False,
        out: pathlib.Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.processed_criteria_per_height_level = processed_criteria_per_height_level
        self.processed_graphs_per_height_level = processed_graphs_per_height_level
        self.hexagon_size = hexagon_size
        self.gdf_main_nodes: gpd.GeoDataFrame = get_empty_geodataframe()

        self.debug = debug
        self.out = out

    def compose(self) -> rx.PyGraph:
        n_height_levels = len(self.processed_graphs_per_height_level)
        if n_height_levels == 1:
            logger.info("Only a single height level is present, no merging is required.")
            return self.processed_graphs_per_height_level[next(iter(self.processed_graphs_per_height_level))]
        else:
            logger.info(f"Connecting {n_height_levels - 1} height level(s) to the main graph.")

        main_height_level = self.get_main_height_level()
        self.gdf_main_nodes = convert_hexagon_graph_to_gdfs(
            self.processed_graphs_per_height_level[main_height_level], edges=False
        )
        self.merge_graphs(main_height_level)

        return self.processed_graphs_per_height_level[main_height_level]

    def get_main_height_level(self):
        """Doublecheck the main height level is 0. This is the expected value for the BGT."""
        node_count = {height: graph.num_nodes() for height, graph in self.processed_graphs_per_height_level.items()}
        main_height_level = max(node_count, key=node_count.get)
        if main_height_level != 0:
            logger.warning(f"Main height level is expected to be 0, but found {main_height_level} instead.")

        return main_height_level

    @time_function
    def merge_graphs(self, main_height_level: int):
        """Merge the different height level graphs into a single graph by connecting the subgraphs to the main graph."""
        for height, height_graph in self.processed_graphs_per_height_level.items():
            if height == main_height_level:
                continue
            gdf_nodes_height = convert_hexagon_graph_to_gdfs(height_graph, edges=False)

            logger.info(
                f"Height level: {height} contains {rx.number_connected_components(height_graph)} subgraph(s) to connect the main graph."
            )

            height_mapping = self.get_height_mapping(height_graph, main_height_level)

            # Determine which nodes to connect to each other
            for component in rx.connected_components(height_graph):
                gdf_component_nodes = gdf_nodes_height[gdf_nodes_height["node_id"].isin(component)]
                # Get the outer nodes (nodes to join to the main graph) of the component.
                component_area = gdf_component_nodes.buffer(self.hexagon_size).union_all(grid_size=0.1)
                if not isinstance(component_area, shapely.Polygon):
                    logger.warning("Component area is not a polygon, this is unexpected. Skipping.")
                    continue

                gdf_component_outer_nodes = self.filter_component_nodes(component_area, gdf_component_nodes)
                # Outer component nodes are duplicated for each node to connect to in the main graph
                gdf_main_nodes_to_outer_component_nodes = gdf_component_outer_nodes.sjoin(
                    self.gdf_main_nodes[~self.gdf_main_nodes.intersects(component_area)],
                    distance=self.hexagon_size * 2,
                    how="left",
                    predicate="dwithin",
                )
                gdf_main_nodes_to_outer_component_nodes = self.validate_main_to_subgraph_pairs(
                    gdf_main_nodes_to_outer_component_nodes
                )
                self.add_edges_between_height_levels(
                    gdf_main_nodes_to_outer_component_nodes, height, height_mapping, main_height_level
                )

        if self.debug:
            nodes, edges = convert_hexagon_graph_to_gdfs(self.processed_graphs_per_height_level[main_height_level])
            write_results_to_geopackage(self.out, nodes, "pytest_merged_graph_nodes", overwrite=True)
            write_results_to_geopackage(self.out, edges, "pytest_merged_graph_edges", overwrite=True)

    def add_edges_between_height_levels(
        self,
        gdf_main_nodes_to_outer_component_nodes: gpd.GeoDataFrame,
        height: int,
        height_mapping: dict[int, int],
        main_height_level: int,
    ):
        """Add the edges which connect height levels between the main graph and the component/subgraph."""
        edges_to_add = [
            (
                node_pair.node_id_right,
                height_mapping[node_pair.node_id_left],
                HexagonEdgeInfo(
                    weight=(node_pair.suitability_value_left + node_pair.suitability_value_right) / 2,
                    height_level=height,
                    connects_height_levels=True,
                    geometry=shapely.LineString(
                        [
                            node_pair.geometry,
                            self.gdf_main_nodes.loc[
                                self.gdf_main_nodes["node_id"] == node_pair.node_id_right, "geometry"
                            ].iloc[0],
                        ]
                    ),
                ),
            )
            for node_pair in gdf_main_nodes_to_outer_component_nodes.itertuples(index=False)
        ]
        edge_indices = self.processed_graphs_per_height_level[main_height_level].add_edges_from(edges_to_add)
        [
            self.processed_graphs_per_height_level[main_height_level].get_edge_data_by_index(i).set_edge_id(i)
            for i in edge_indices
        ]

        if self.debug:
            # visualize the pairs / edges to be
            linestrings = gdf_main_nodes_to_outer_component_nodes.apply(
                lambda x: shapely.LineString(
                    [
                        x.geometry,
                        self.gdf_main_nodes.loc[self.gdf_main_nodes["node_id"] == x.node_id_right, "geometry"].iloc[0],
                    ]
                ),
                axis=1,
            )
            write_results_to_geopackage(self.out, linestrings, "pytest_component_connection_lines", overwrite=True)

    def filter_component_nodes(
        self, component_area: shapely.Polygon, gdf_component: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Filter the nodes of the component to connect so we do not connect halfway a bridge/tunnel, but only at the
        start and end.
        """
        # TODO validate pairs based on osm road (if available)
        # TODO try to find the counterpart at the other height level through shared boundary
        #  - expand node model first with bgt id's?
        #  - Create extended line perpendicular on the endpoints of the centerline with a width equal to a road (8m)
        component_area_centerline = pygeoops.centerline(component_area, extend=True)
        if isinstance(component_area_centerline, shapely.LineString):
            entrypoints = shapely.MultiPoint(
                [shapely.get_point(component_area_centerline, 0), shapely.get_point(component_area_centerline, -1)]
            )
            gdf_component = gdf_component[gdf_component.dwithin(entrypoints, distance=3)]
        else:
            logger.warning(f"Unhandled situation: {type(component_area_centerline)}")

        gdf_component_outer_nodes = gdf_component[
            gdf_component.geometry.dwithin(component_area.boundary, self.hexagon_size)
        ]

        if self.debug:
            write_results_to_geopackage(self.out, component_area, "pytest_component_area")
            write_results_to_geopackage(self.out, component_area_centerline, "pytest_component_area_centerline")
            write_results_to_geopackage(self.out, component_area.boundary, "pytest_component_area_boundary")
        return gdf_component_outer_nodes

    def validate_main_to_subgraph_pairs(self, gdf_main_nodes_to_outer_subgraph_nodes):
        """This can occur on nodes that are at the edge of the main graph."""
        if gdf_main_nodes_to_outer_subgraph_nodes["node_id_right"].isna().any():
            logger.warning("Some outer subgraph nodes could not be connected to the main graph nodes.")
            na_rows = gdf_main_nodes_to_outer_subgraph_nodes[
                gdf_main_nodes_to_outer_subgraph_nodes["node_id_right"].isna()
            ]
            gdf_main_nodes_to_outer_subgraph_nodes.dropna(subset=["node_id_right"], inplace=True)
            gdf_main_nodes_to_outer_subgraph_nodes["node_id_right"] = gdf_main_nodes_to_outer_subgraph_nodes[
                "node_id_right"
            ].astype(int)
            gdf_main_nodes_to_outer_subgraph_nodes["node_id_left"] = gdf_main_nodes_to_outer_subgraph_nodes[
                "node_id_left"
            ].astype(int)
            if self.debug:
                write_results_to_geopackage(
                    Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, na_rows, "pytest_invalid_nodes"
                )
        return gdf_main_nodes_to_outer_subgraph_nodes

    def get_height_mapping(self, height_graph: rx.PyGraph, main_height_level: int) -> dict[int, int]:
        """Add the complete subgraph to the main graph first."""
        mapping = {}  # idx_height_graph → idx_main_graph mapping for graph merge

        # Add nodes from the subgraph to the main graph
        for old_idx, node_data in enumerate(height_graph.nodes()):
            # Always add as new node (even if many map to same "right" node)
            new_idx = self.processed_graphs_per_height_level[main_height_level].add_node(node_data)
            self.processed_graphs_per_height_level[main_height_level][new_idx].node_id = new_idx
            mapping[old_idx] = new_idx

        # Add subgraph edges to the main graph
        for u, v, weight in height_graph.weighted_edge_list():
            new_idx = self.processed_graphs_per_height_level[main_height_level].add_edge(mapping[u], mapping[v], weight)
            self.processed_graphs_per_height_level[main_height_level].get_edge_data_by_index(new_idx).set_edge_id(
                new_idx
            )

        return mapping
