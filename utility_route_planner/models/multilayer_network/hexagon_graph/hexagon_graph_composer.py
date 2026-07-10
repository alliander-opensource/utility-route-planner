# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import pathlib
import pandas as pd
import pygeoops
import shapely
import geopandas as gpd
import rustworkx as rx
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonConnectionEdgeInfo
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import HexagonGridBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    convert_hexagon_edges_to_gdf,
    update_edge_id,
    get_hexagon_node_geometry,
)
from utility_route_planner.models.multilayer_network.pipe_ramming import (
    GetPotentialPipeRammingCrossings,
    PipeRammingSettings,
)
from utility_route_planner.util.geo_utilities import (
    get_empty_geodataframe,
    get_perpendicular_line,
    get_endlines_of_multilinestring,
)
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


@dataclass
class HeightLevelGraph:
    graph: rx.PyGraph
    nodes_gdf: gpd.GeoDataFrame


class HexagonGraphComposer:
    def __init__(
        self,
        processed_criteria_per_height_level: dict[int, list[str]],
        processed_graphs_and_nodes_per_height_level: dict[int, HeightLevelGraph],
        hexagon_size: float,
        debug: bool = False,
        out: pathlib.Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.processed_criteria_per_height_level = processed_criteria_per_height_level
        self.processed_graphs_and_nodes_per_height_level = processed_graphs_and_nodes_per_height_level
        self.hexagon_size = hexagon_size
        self.gdf_main_nodes: gpd.GeoDataFrame = get_empty_geodataframe()

        self.debug = debug
        self.out = out

    def compose(self) -> tuple[rx.PyGraph, gpd.GeoDataFrame]:
        n_height_levels = len(self.processed_graphs_and_nodes_per_height_level)
        main_height_level = self.get_main_height_level()
        self.gdf_main_nodes = self.processed_graphs_and_nodes_per_height_level[main_height_level].nodes_gdf
        self.gdf_main_nodes["height_level"] = main_height_level

        if self.debug:
            self.write_graph_per_height_level()

        if n_height_levels == 1:
            logger.info("Only a single height level is present, no merging is required.")
        else:
            logger.info(f"Connecting {n_height_levels - 1} height level(s) to the main graph.")
            self.merge_graphs(main_height_level)

        return (
            self.processed_graphs_and_nodes_per_height_level[main_height_level].graph,
            self.processed_graphs_and_nodes_per_height_level[main_height_level].nodes_gdf,
        )

    def get_main_height_level(self):
        """Doublecheck the main height level is 0. This is the expected value for the BGT."""
        node_count = {
            height: graph.graph.num_nodes()
            for height, graph in self.processed_graphs_and_nodes_per_height_level.items()
        }
        main_height_level = max(node_count, key=node_count.get)
        if main_height_level != 0:
            logger.warning(f"Main height level is expected to be 0, but found {main_height_level} instead.")

        return main_height_level

    @time_function
    def merge_graphs(self, main_height_level: int):
        """
        Merge the different height level graphs into a single graph by connecting the subgraphs to the main graph.

        Note that height levels are joined directly to the main graph. Different height levels are not joined to other
        height levels.
        """
        for height, height_graph in self.processed_graphs_and_nodes_per_height_level.items():
            if height == main_height_level:
                continue

            logger.info(
                f"Height level: {height} contains {rx.number_connected_components(height_graph.graph)} subgraph(s) to connect the main graph."
            )

            height_mapping = self.add_height_graph_to_main_graph(
                height_graph.graph, height_graph.nodes_gdf, main_height_level
            )
            self.processed_graphs_and_nodes_per_height_level[main_height_level].nodes_gdf.loc[
                list(height_mapping.values()), "height_level"
            ] = height

            # Determine which nodes to connect to each other
            for component in rx.connected_components(height_graph.graph):
                gdf_component_nodes = height_graph.nodes_gdf.loc[list(component)]
                # Get the outer nodes (nodes to join to the main graph) of the component.
                component_area = gdf_component_nodes.buffer(self.hexagon_size).union_all(grid_size=0.1)
                if not isinstance(component_area, shapely.Polygon):
                    logger.warning("Component area is not a polygon, this is unexpected. Skipping.")
                    continue
                if shapely.get_num_interior_rings(component_area) > 0:
                    component_area = shapely.Polygon(component_area.exterior)

                gdf_component_outer_nodes = self.filter_component_nodes(component_area, gdf_component_nodes)
                # Outer component nodes are duplicated for each node to connect to in the main graph
                gdf_main_nodes_to_outer_component_nodes = gdf_component_outer_nodes.sjoin(
                    self.gdf_main_nodes[~self.gdf_main_nodes.intersects(component_area)],
                    distance=self.hexagon_size * 2,
                    how="left",
                    predicate="dwithin",
                ).reset_index(drop=False)
                gdf_main_nodes_to_outer_component_nodes = self.validate_main_to_subgraph_pairs(
                    gdf_main_nodes_to_outer_component_nodes
                )
                self.add_edges_between_height_levels(
                    gdf_main_nodes_to_outer_component_nodes, height, height_mapping, main_height_level
                )

        if self.debug:
            main_height_level_graph = self.processed_graphs_and_nodes_per_height_level[main_height_level]
            nodes = main_height_level_graph.nodes_gdf
            edges = convert_hexagon_edges_to_gdf(main_height_level_graph.graph, nodes)

            write_results_to_geopackage(self.out, nodes, "pytest_merged_graph_nodes", overwrite=True)
            write_results_to_geopackage(self.out, edges, "pytest_merged_graph_edges", overwrite=True)

    def add_height_graph_to_main_graph(
        self, height_graph: rx.PyGraph, height_node_df: gpd.GeoDataFrame, main_height_level: int
    ) -> dict[int, int]:
        """Add the complete height graph to the main graph. Note it is still not connected at this point."""
        mapping = {}  # idx_height_graph -> idx_main_graph mapping for graph merge

        # Add nodes from the subgraph to the main graph
        for old_idx, node_data in enumerate(height_graph.nodes()):
            # Always add as new node (even if many map to same "right" node)
            new_idx = self.processed_graphs_and_nodes_per_height_level[main_height_level].graph.add_node(node_data)
            self.processed_graphs_and_nodes_per_height_level[main_height_level].graph[new_idx].node_id = new_idx
            mapping[old_idx] = new_idx

        # Reassign nodes to a copied height node df, as the "original" node ids are required to properly connect height
        # levels in the next step
        height_node_df_copy = height_node_df.copy()
        height_node_df_copy.index = height_node_df_copy.index.map(mapping)
        self.processed_graphs_and_nodes_per_height_level[main_height_level].nodes_gdf = gpd.GeoDataFrame(
            pd.concat(
                [self.processed_graphs_and_nodes_per_height_level[main_height_level].nodes_gdf, height_node_df_copy]
            )
        )

        # Add subgraph edges to the main graph
        for u, v, weight in height_graph.weighted_edge_list():
            new_edge_id = self.processed_graphs_and_nodes_per_height_level[main_height_level].graph.add_edge(
                mapping[u], mapping[v], weight
            )
            new_edge_data = self.processed_graphs_and_nodes_per_height_level[
                main_height_level
            ].graph.get_edge_data_by_index(new_edge_id)
            self.processed_graphs_and_nodes_per_height_level[main_height_level].graph.update_edge_by_index(
                new_edge_id, update_edge_id(new_edge_id, new_edge_data)
            )

        return mapping

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
                HexagonConnectionEdgeInfo(
                    weight=node_pair.suitability_value_left + node_pair.suitability_value_right,
                    height_level=height,
                    connects_height_levels=True,
                    geometry=shapely.LineString(
                        [
                            node_pair.geometry,
                            get_hexagon_node_geometry(self.gdf_main_nodes, node_pair.node_id_right),
                        ]
                    ),
                ),
            )
            for node_pair in gdf_main_nodes_to_outer_component_nodes.itertuples(index=False)
        ]
        edge_indices = self.processed_graphs_and_nodes_per_height_level[main_height_level].graph.add_edges_from(
            edges_to_add
        )
        for new_edge_id in edge_indices:
            new_edge_data = self.processed_graphs_and_nodes_per_height_level[
                main_height_level
            ].graph.get_edge_data_by_index(new_edge_id)
            self.processed_graphs_and_nodes_per_height_level[main_height_level].graph.update_edge_by_index(
                new_edge_id, update_edge_id(new_edge_id, new_edge_data)
            )

        if self.debug:
            # visualize the pairs / edges to be
            linestrings = gdf_main_nodes_to_outer_component_nodes.apply(
                lambda x: shapely.LineString(
                    [
                        x.geometry,
                        get_hexagon_node_geometry(self.gdf_main_nodes, x.node_id_right),
                    ]
                ),
                axis=1,
            )
            write_results_to_geopackage(self.out, linestrings, "pytest_component_connection_lines")

    def filter_component_nodes(
        self, component_area: shapely.Polygon, gdf_component: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Filter the nodes of the component to connect so we do not connect halfway a bridge/tunnel, but only at the
        start and end.

        Some thoughts and notes on improvements:
            - This does not work well when a bridge/tunnel is wider than it is long due to the centerline approach.
            - Validation using OSM does not work well as it is not perfectly aligned with the BGT. This is especially
            apparent for smaller pedestrian or cycling tunnels.
            - Expand node model so we can retrieve the BGT polygons which can be used for cleaner centerlines. This can
            also be used for finding the shared boundary of BGT polygons for determining the entrypoints. E.g., finding
            the road which perfectly connects through height levels.
        """

        gdf_component_outer_nodes = gdf_component[
            gdf_component.geometry.dwithin(component_area.boundary, self.hexagon_size)
        ]
        component_area_simplified = component_area.simplify(self.hexagon_size * 1.6)
        component_area_centerline = pygeoops.centerline(component_area_simplified, extend=True)
        if isinstance(component_area_centerline, shapely.LineString):
            start, end = (
                shapely.get_point(component_area_centerline, 0),
                shapely.get_point(component_area_centerline, 1),
            )
            line_1 = get_perpendicular_line(start, end, 40)
            start, end = (
                shapely.get_point(component_area_centerline, -1),
                shapely.get_point(component_area_centerline, -2),
            )
            line_2 = get_perpendicular_line(start, end, 40)
            entrypoint_lines = shapely.MultiLineString([line_1, line_2])
        elif isinstance(component_area_centerline, shapely.MultiLineString):
            lines = get_endlines_of_multilinestring(component_area_centerline)
            entrypoint_lines = shapely.MultiLineString(
                [get_perpendicular_line(shapely.get_point(i, 1), shapely.get_point(i, 0), 40) for i in lines]
            )
        else:
            entrypoint_lines = shapely.MultiLineString()
            logger.warning(f"Unhandled situation: {type(component_area_centerline)}")

        if not entrypoint_lines.is_empty:
            gdf_component_outer_nodes = gdf_component_outer_nodes[
                gdf_component_outer_nodes.dwithin(entrypoint_lines, distance=self.hexagon_size * 2)
            ]
        else:
            logger.warning(
                "Unable to create proper connections between height levels. All outer nodes are now connected"
            )

        if self.debug:
            write_results_to_geopackage(self.out, gdf_component_outer_nodes, "pytest_component_area_outer_nodes")
            write_results_to_geopackage(self.out, component_area, "pytest_component_area")
            write_results_to_geopackage(self.out, component_area_simplified, "pytest_component_area_simplified")
            write_results_to_geopackage(self.out, component_area_centerline, "pytest_component_area_centerline")
            write_results_to_geopackage(self.out, component_area.boundary, "pytest_component_area_boundary")
            write_results_to_geopackage(self.out, entrypoint_lines, "pytest_component_entrypoint_lines")
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
            write_results_to_geopackage(self.out, na_rows, "outer_subgraph_invalid_nodes")
        return gdf_main_nodes_to_outer_subgraph_nodes

    def write_graph_per_height_level(self):
        for height_level, height_level_graph in self.processed_graphs_and_nodes_per_height_level.items():
            edges_gdf = convert_hexagon_edges_to_gdf(height_level_graph.graph, height_level_graph.nodes_gdf)
            write_results_to_geopackage(
                self.out,
                height_level_graph.nodes_gdf,
                f"pytest_graph_nodes_height_level_{height_level}",
                overwrite=True,
            )
            write_results_to_geopackage(
                self.out, edges_gdf, f"pytest_graph_edges_height_level_{height_level}", overwrite=True
            )


def build_and_compose_graph(
    processed_criteria_per_height_level: dict[int, list[str]],
    processed_criteria_vectors: dict[str, gpd.GeoDataFrame],
    raster_groups: dict,
    project_area: shapely.Polygon | shapely.MultiPolygon,
    debug: bool = Config.DEBUG,
    hexagon_size: float = Config.HEXAGON_SIZE,
    block_size: int = Config.HEXAGON_BLOCK_SIZE,
    apply_pipe_ramming: bool = Config.APPLY_PIPE_RAMMING,
    osm_graph_preprocessed: rx.PyGraph = rx.PyGraph(),
    pipe_ramming_settings: PipeRammingSettings = PipeRammingSettings(),
) -> tuple[rx.PyGraph, gpd.GeoDataFrame, list]:

    grid_constructor = HexagonGridBuilder(hexagon_size=hexagon_size, block_size=block_size)
    hexagon_edge_generator = HexagonEdgeGenerator()
    hexagon_graph_builder = HexagonGraphBuilder(grid_builder=grid_constructor, edge_generator=hexagon_edge_generator)

    # TODO Cache the project area node grid so it is not recomputed each time
    graphs_per_height: dict[int, HeightLevelGraph] = {}
    pipe_ramming_crossings = []
    for height_level, criteria in processed_criteria_per_height_level.items():
        criteria_for_height_level = {}
        for criterion in criteria:
            gdf = processed_criteria_vectors[criterion][
                processed_criteria_vectors[criterion]["relatieveHoogteligging"] == height_level
            ]
            criteria_for_height_level[criterion] = gdf  # type: ignore

        graph, nodes_gdf = hexagon_graph_builder.build_graph(
            project_area.buffer(0.01), raster_groups, criteria_for_height_level
        )

        # Only try to get pipe rammings for the "ground" level which is 0 when using the BGT.
        if apply_pipe_ramming and height_level == 0:
            pipe_ramming_settings.hexagon_size = hexagon_size
            pipe_ramming = GetPotentialPipeRammingCrossings(
                osm_graph_preprocessed,
                graph,
                nodes_gdf,
                settings=pipe_ramming_settings,
                debug=debug,
            )
            pipe_ramming.get_crossings()
            graph = pipe_ramming.cost_surface_graph
            nodes_gdf = pipe_ramming.cost_surface_nodes
            pipe_ramming_crossings = pipe_ramming.crossing_collection

        graphs_per_height[height_level] = HeightLevelGraph(graph, nodes_gdf)

    hexagon_graph_composer = HexagonGraphComposer(
        processed_criteria_per_height_level,
        graphs_per_height,
        hexagon_size=hexagon_graph_builder.hexagon_size,
        debug=debug,
    )
    graph, gdf_nodes = hexagon_graph_composer.compose()
    return graph, gdf_nodes, pipe_ramming_crossings
