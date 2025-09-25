# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import rustworkx as rx
import osmnx as ox
import structlog
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import OSMEdgeInfo, OSMNodeInfo
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class OSMGraphPreprocessor:
    def __init__(
        self,
        nx_graph: nx.MultiDiGraph,
        polygon_for_clip: shapely.Polygon | shapely.MultiPolygon | None = None,
        debug: bool = False,
    ):
        self.nx_graph = nx_graph
        self.polygon_for_clip = polygon_for_clip
        self.debug = debug

    def preprocess_graph(self) -> rx.PyGraph:
        self.validate_input()
        logger.info(
            f"Preprocessing NetworkX OSM graph n_nodes: {self.nx_graph.number_of_nodes()} n_edges: {self.nx_graph.number_of_edges()} to Rustworkx."
        )
        self.nx_graph = ox.convert.to_undirected(self.nx_graph)
        self._remove_duplicate_edges_and_add_edge_id_and_length_properties()
        if self.polygon_for_clip:
            self.clip_graph_to_polygon()
        rx_graph = self._convert_to_rustworkx(self.nx_graph)

        return rx_graph

    def _remove_duplicate_edges_and_add_edge_id_and_length_properties(self):
        """
        OSM graphs occasionally contain duplicate edges. These duplicates are removed.
        In addition, a unique edge is assigned to each edge for further processing. Given the geometry of an edge,
        the length is set as an edge attribute.

        :return: the number of edges, which is used as the maximum edge id to assign new edges later on
        """
        edges_to_remove = []
        initial_edge_count = len(self.nx_graph.edges)
        for u, v, key in self.nx_graph.edges(keys=True):
            if key > 0:
                edges_to_remove.append((u, v, key))
            self.nx_graph[u][v][0]["length"] = self.nx_graph[u][v][0]["geometry"].length
        self.nx_graph.remove_edges_from(edges_to_remove)
        logger.info(f"Removed {initial_edge_count - len(self.nx_graph.edges)} duplicate edges from the graph")

    @staticmethod
    def _convert_to_rustworkx(nx_graph) -> rx.PyGraph:
        rx_graph = rx.PyGraph(multigraph=False)

        nodes = list(nx_graph.nodes)
        nx_rx_node_mapping = dict(zip(nodes, rx_graph.add_nodes_from(nodes)))
        edges = [
            (
                nx_rx_node_mapping[u],
                nx_rx_node_mapping[v],
                OSMEdgeInfo(edge.get("length", 0), edge.get("geometry", shapely.LineString()), edge.get("osmid", 0)),
            )
            for u, v, edge in nx_graph.edges(data=True)
        ]
        edge_ids = rx_graph.add_edges_from(edges)
        for (_, _, edge), edge_id in zip(edges, edge_ids):
            edge.edge_id = edge_id

        for node, node_index in nx_rx_node_mapping.items():
            data = nx_graph.nodes[node]
            info = OSMNodeInfo(shapely.Point(data.get("x", 0), data.get("y", 0)), node)
            info.node_id = node_index
            rx_graph[node_index] = info

        return rx_graph

    def validate_input(self):
        if not isinstance(self.nx_graph, nx.MultiDiGraph):
            raise TypeError("Input graph must be a NetworkX MultiDiGraph.")
        if self.nx_graph.number_of_edges() == 0:
            raise ValueError("Graph should have edges.")
        if self.nx_graph.number_of_nodes() == 0:
            raise ValueError("Graph should have nodes.")

    def clip_graph_to_polygon(self) -> None:
        """Clips the graph to the given polygon. Nodes and edges outside the polygon are removed."""
        nodes = ox.graph_to_gdfs(self.nx_graph, edges=False)
        nodes_to_remove = nodes[~nodes.intersects(self.polygon_for_clip)]
        self.nx_graph.remove_nodes_from(nodes_to_remove.index)
        logger.info(f"Clipped graph to polygon, removed {len(nodes_to_remove)} nodes.")

        if self.debug:
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, self.polygon_for_clip, "pytest_polygon_for_clip"
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, nodes, "pytest_original_nodes"
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, nodes_to_remove, "pytest_nodes_to_remove"
            )
