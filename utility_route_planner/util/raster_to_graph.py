# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for auxiliary data structures used in MultiPass algorithm.
"""

import networkx as nx
import numpy as np
from scipy.spatial import distance

from settings import Config


class Path:
    """
    This class creates a path object.
    """

    def __init__(self):
        """
        length: The length of the path
        weight: The weight of the path
        nodes: Set of nodes that compose the path (not in indices format)
        """
        self.weight = 0
        self.length = 0
        self.nodes = []


class GraphNetwork:
    """
    This class creates a graph from the cost raster array.
    """

    def __init__(self, cost_raster_array):
        """
        Creates a GraphNetwork object.

        adj_list_in: Adjacency list dictionary: key (node) -> value (dict(node: (weight, length)))  - all edges that
            go into that node + (weight, euclidean length)
        adj_list_out: Adjacency list dictionary: key (node) -> value (dict(node: (weight, length))) - all edges that
            come out of that node + (weight, euclidean length)
        nodes: Array with all the nodes that form the graph

        Parameters
        -----
        cost_raster_array: The cost raster array
        """
        self.adj_list_in = {}
        self.adj_list_out = {}
        self.nodes = []
        self.cost_raster = cost_raster_array

    def create_graph_from_cost_array(self):
        """
        This method converts the cost_raster array into 2 adjacency lists.
        The adjacency list is a dict: key (node) -> value (dict(node: weight)).
        """
        number_nodes = self.cost_raster.size

        self.nodes = np.linspace(0, number_nodes, number_nodes).astype(int)
        self.nodes = np.reshape(self.nodes, (-1, self.cost_raster.shape[1]))

        for i in range(0, self.cost_raster.shape[0]):
            for j in range(0, self.cost_raster.shape[1]):
                # If the node is in the project area, create links to all the surrounding nodes (8 directions)
                # We only want to connect the nodes that are part of the project area (so not with the ones that
                # have the cost of 65534 which is the value assigned to the cropped section

                if self.cost_raster[i][j] != Config.FINAL_RASTER_NO_DATA:
                    # For each direction, check if the indexes are valid and if the other end of the edge is also
                    # part of the project area
                    if i - 1 >= 0 and j - 1 >= 0 and self.cost_raster[i - 1][j - 1] != Config.FINAL_RASTER_NO_DATA:
                        # For the diagonal movement, the weight is sightly bigger than just moving horiz/vertically
                        # In this way we avoid going in diagonal when is not necessary
                        weight = (self.cost_raster[i][j] + self.cost_raster[i - 1][j - 1]) / np.sqrt(2)
                        length = np.sqrt(2) / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i - 1][j - 1])

                    if j - 1 >= 0 and self.cost_raster[i][j - 1] != Config.FINAL_RASTER_NO_DATA:
                        weight = self.cost_raster[i][j] / 2 + self.cost_raster[i][j - 1] / 2
                        length = 1 / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i][j - 1])

                    if i - 1 >= 0 and self.cost_raster[i - 1][j] != Config.FINAL_RASTER_NO_DATA:
                        weight = self.cost_raster[i][j] / 2 + self.cost_raster[i - 1][j] / 2
                        length = 1 / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i - 1][j])

                    if (
                        i + 1 < self.cost_raster.shape[0]
                        and j + 1 < self.cost_raster.shape[1]
                        and self.cost_raster[i + 1][j + 1] != Config.FINAL_RASTER_NO_DATA
                    ):
                        weight = (self.cost_raster[i][j] + self.cost_raster[i + 1][j + 1]) / np.sqrt(2)
                        length = np.sqrt(2) / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i + 1][j + 1])

                    if i + 1 < self.cost_raster.shape[0] and self.cost_raster[i + 1][j] != Config.FINAL_RASTER_NO_DATA:
                        weight = self.cost_raster[i][j] / 2 + self.cost_raster[i + 1][j] / 2
                        length = 1 / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i + 1][j])

                    if j + 1 < self.cost_raster.shape[1] and self.cost_raster[i][j + 1] != Config.FINAL_RASTER_NO_DATA:
                        weight = self.cost_raster[i][j] / 2 + self.cost_raster[i][j + 1] / 2
                        length = 1 / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i][j + 1])

                    if (
                        i - 1 >= 0
                        and j + 1 < self.cost_raster.shape[1]
                        and self.cost_raster[i - 1][j + 1] != Config.FINAL_RASTER_NO_DATA
                    ):
                        weight = (self.cost_raster[i][j] + self.cost_raster[i - 1][j + 1]) / np.sqrt(2)
                        length = np.sqrt(2) / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i - 1][j + 1])

                    if (
                        i + 1 < self.cost_raster.shape[0]
                        and j - 1 >= 0
                        and self.cost_raster[i + 1][j - 1] != Config.FINAL_RASTER_NO_DATA
                    ):
                        weight = (self.cost_raster[i][j] + self.cost_raster[i + 1][j - 1]) / np.sqrt(2)
                        length = np.sqrt(2) / 2
                        self.add_to_adj_list(weight, length, self.nodes[i][j], self.nodes[i + 1][j - 1])

    def add_to_adj_list(self, weight, length, node_in, node_out):
        """
        This is a helper method to add the new key-value pairs to the adj lists.
        Add the nodes into the adjacency lists. The adj_list is a dictionary that has the keys the node ids
        and as the values, it has another dictionary with keys as the node ids they are connected  with
        and the value the weight of the edge.

        Parameters
        ----------
        weight (float): The weight corresponding to the edge
        length (float): The length
        node_in (int): In node of the edge
        node_out (int): Out node of the edge
        """
        # pylint: disable= consider-iterating-dictionary
        if node_out not in self.adj_list_out.keys():
            self.adj_list_out[node_out] = {node_in: (weight, length)}
        else:
            self.adj_list_out[node_out].update({node_in: (weight, length)})
        if node_in not in self.adj_list_in.keys():
            self.adj_list_in[node_in] = {node_out: (weight, length)}
        else:
            self.adj_list_in[node_in].update({node_out: (weight, length)})

    def calculate_distance_between_edges(self, edge_1: tuple, edge_2: tuple):
        """
        Method to calculate the distance between to edges, used for making sure that the alternative routes are
        different enough.
        """
        # edge = (n1, n2)
        coordinates_e_1 = np.where(self.nodes == edge_1[0])
        coordinates_e_2 = np.where(self.nodes == edge_1[1])
        edge_1_coords = [
            list(zip(coordinates_e_1[0], coordinates_e_1[1]))[0],
            list(zip(coordinates_e_2[0], coordinates_e_2[1]))[0],
        ]
        coord_e_1 = np.where(self.nodes == edge_2[0])
        coord_e_2 = np.where(self.nodes == edge_2[1])
        edge_2_coords = [list(zip(coord_e_1[0], coord_e_1[1]))[0], list(zip(coord_e_2[0], coord_e_2[1]))[0]]
        dst_1 = distance.euclidean(edge_1_coords[0], edge_2_coords[0])
        dst_2 = distance.euclidean(edge_1_coords[1], edge_2_coords[1])

        return min(dst_1, dst_2)


# def simple_graph_example():
#     import igraph as ig
#     g = ig.Graph()
#     g = ig.Graph(n=10, edges=[[0, 1], [0, 5]])
#     layout = g.layout_kamada_kawai()
#     ig.plot(g, layout=layout)
#
# simple_graph_example()


def raster_to_graph_nx(array_to_convert):
    # Create a graph
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    rows, cols = array_to_convert.shape
    for i in range(rows):
        for j in range(cols):
            node_value = array_to_convert[i, j]
            G.add_node(node_value)

            # Add directed edges to neighbors (8 adjacency) with weighted edges
            if i > 0:
                G.add_edge(node_value, array_to_convert[i - 1, j], weight=array_to_convert[i - 1, j])
            if i < rows - 1:
                G.add_edge(node_value, array_to_convert[i + 1, j], weight=array_to_convert[i + 1, j])
            if j > 0:
                G.add_edge(node_value, array_to_convert[i, j - 1], weight=array_to_convert[i, j - 1])
            if j < cols - 1:
                G.add_edge(node_value, array_to_convert[i, j + 1], weight=array_to_convert[i, j + 1])
            if i > 0 and j > 0:
                G.add_edge(node_value, array_to_convert[i - 1, j - 1], weight=array_to_convert[i - 1, j - 1])
            if i > 0 and j < cols - 1:
                G.add_edge(node_value, array_to_convert[i - 1, j + 1], weight=array_to_convert[i - 1, j + 1])
            if i < rows - 1 and j > 0:
                G.add_edge(node_value, array_to_convert[i + 1, j - 1], weight=array_to_convert[i + 1, j - 1])
            if i < rows - 1 and j < cols - 1:
                G.add_edge(node_value, array_to_convert[i + 1, j + 1], weight=array_to_convert[i + 1, j + 1])

    return G
