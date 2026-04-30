# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import rustworkx as rx


class KShortestPathRoutePlanner:
    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int):
        rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)
        # Expand graph and apply pruning using queuing strategy as proposed
        # in https://link.springer.com/article/10.1007/s00778-020-00604-x
