# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pandas as pd

from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonNodeInfo


class HexagonGraph:
    def __init__(self):
        self.nodes: dict[tuple[int, int], HexagonNodeInfo] = {}

    def add_nodes(self, nodes: pd.DataFrame):
        nodes_dict = {
            (node.axial_q, node.axial_r): HexagonNodeInfo(node.suitability_value, node.x, node.y)
            for node in nodes.itertuples(index=False)
        }
        self.nodes = self.nodes | nodes_dict

    def get_node(self, q: int, r: int) -> HexagonNodeInfo | None:
        """
        Given axial coordinates, get the node if present

        :param q: axial q coordinate of node
        :param r: axial r coordinate of node
        :return: neighbour if present in the graph, else None
        """
        return self.nodes.get((q, r), None)
