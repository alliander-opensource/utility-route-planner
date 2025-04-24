# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utility_route_planner.util.raster_to_graph import raster_to_graph_nx, GraphNetwork


def test_array_to_graph_small_example():
    array_to_convert = np.array(
        [[1, 64, 64, 64, 64], [64, 1, 64, 64, 64], [-127, 5, 64, 64, 64], [-127, 6, 20, 64, 64], [-127, 64, 64, 30, 1]]
    )
    ## GraphNetwork - this does almost what I want. I need to convert the function so it adds nodes directly to the
    # directed graph rather than adding it to a dictionary.
    graphobj = GraphNetwork(array_to_convert)
    graphobj.create_graph_from_cost_array()

    ## No succes
    G = raster_to_graph_nx(array_to_convert)

    # Draw the graph
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw(
        G,
        pos,
        with_labels=True,
        font_weight="bold",
        node_size=700,
        node_color="skyblue",
        font_color="black",
        font_size=8,
        edge_color="gray",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def test_small_directed_graph_10_nodes_hexagonal():
    G = nx.DiGraph()

    G.add_edge("a", "b", weight=2)
    G.add_edge("a", "d", weight=15)
    G.add_edge("a", "c", weight=10)

    G.add_edge("b", "a", weight=20)
    G.add_edge("b", "d", weight=15)
    G.add_edge("b", "e", weight=2)

    G.add_edge("d", "a", weight=20)
    G.add_edge("d", "b", weight=2)
    G.add_edge("d", "c", weight=10)
    G.add_edge("d", "g", weight=15)
    G.add_edge("d", "f", weight=1)
    G.add_edge("d", "e", weight=2)

    G.add_edge("c", "a", weight=20)
    G.add_edge("c", "d", weight=15)
    G.add_edge("c", "f", weight=1)

    G.add_edge("e", "b", weight=2)
    G.add_edge("e", "d", weight=15)
    G.add_edge("e", "g", weight=15)
    G.add_edge("e", "h", weight=8)

    G.add_edge("f", "c", weight=10)
    G.add_edge("f", "d", weight=15)
    G.add_edge("f", "g", weight=15)
    G.add_edge("f", "i", weight=1)

    G.add_edge("g", "d", weight=15)
    G.add_edge("g", "e", weight=2)
    G.add_edge("g", "h", weight=8)
    G.add_edge("g", "j", weight=8)
    G.add_edge("g", "i", weight=1)
    G.add_edge("g", "f", weight=1)

    G.add_edge("h", "e", weight=2)
    G.add_edge("h", "g", weight=15)
    G.add_edge("h", "j", weight=8)

    G.add_edge("j", "h", weight=8)
    G.add_edge("j", "g", weight=15)
    G.add_edge("j", "i", weight=1)

    G.add_edge("i", "f", weight=1)
    G.add_edge("i", "g", weight=15)
    G.add_edge("i", "j", weight=8)

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    edge_labels = dict(
        [
            (
                (
                    u,
                    v,
                ),
                d["weight"],
            )
            for u, v, d in G.edges(data=True)
        ]
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)

    plt.show()


def test_small_undirected_graph_10_nodes_hexagonal():
    G = nx.Graph()

    G.add_edge("a", "b", weight=11)
    G.add_edge("a", "d", weight=35 / 2)
    G.add_edge("a", "c", weight=15)

    # G.add_edge("b", "a", weight=20)
    G.add_edge("b", "d", weight=17 / 2)
    G.add_edge("b", "e", weight=2)

    # G.add_edge("d", "a", weight=20)
    # G.add_edge("d", "b", weight=2)
    G.add_edge("d", "c", weight=35 / 2)
    G.add_edge("d", "g", weight=15)
    G.add_edge("d", "f", weight=8)
    # G.add_edge("d", "e", weight=17/2)

    # G.add_edge("c", "a", weight=20)
    # G.add_edge("c", "d", weight=15)
    G.add_edge("c", "f", weight=11 / 2)

    # G.add_edge("e", "b", weight=)
    G.add_edge("e", "d", weight=17 / 2)
    G.add_edge("e", "g", weight=17 / 2)
    G.add_edge("e", "h", weight=5)

    # G.add_edge("f", "c", weight=11/2)
    # G.add_edge("f", "d", weight=15)
    G.add_edge("f", "g", weight=8)
    G.add_edge("f", "i", weight=1)

    # G.add_edge("g", "d", weight=15)
    # G.add_edge("g", "e", weight=2)
    G.add_edge("g", "h", weight=23 / 2)
    G.add_edge("g", "j", weight=23 / 2)
    G.add_edge("g", "i", weight=8)
    # G.add_edge("g", "f", weight=1)

    # G.add_edge("h", "e", weight=2)
    # G.add_edge("h", "g", weight=15)
    G.add_edge("h", "j", weight=8)

    # G.add_edge("j", "h", weight=8)
    # G.add_edge("j", "g", weight=15)
    G.add_edge("j", "i", weight=9 / 2)

    # G.add_edge("i", "f", weight=1)
    # G.add_edge("i", "g", weight=15)
    # G.add_edge("i", "j", weight=8)

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    edge_labels = dict(
        [
            (
                (
                    u,
                    v,
                ),
                d["weight"],
            )
            for u, v, d in G.edges(data=True)
        ]
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)

    plt.show()
