#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator
import polars as pl

from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeInfo


class HexagonEdgeGenerator:
    def generate(
        self, hexagonal_grid: pl.DataFrame, all_nodes: pl.DataFrame
    ) -> Iterator[list[tuple[int, int, HexagonEdgeInfo]]]:
        """
        We need to generate edges for four directions instead of three, as we need to deal with cross-block edges.

        TODO: can we do all crossings at once to prevent duplicate edge generation
        """
        # Left (-1, 0)
        left_neighbour_candidates = hexagonal_grid.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") - 1, pl.col("r")]
        )

        # Right (+1, 0)
        right_neighbour_candidates = hexagonal_grid.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") + 1, pl.col("r")]
        )

        # Bottom-right (0, +1)
        bottom_right_neighbour_candidates = hexagonal_grid.select(
            [pl.col("node_id").alias("source_node"), pl.col("q"), pl.col("r") + 1]
        )

        # Bottom-left (-1, +1)
        bottom_left_neighbour_candidates = hexagonal_grid.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") - 1, pl.col("r") + 1]
        )

        for candidate in [
            left_neighbour_candidates,
            right_neighbour_candidates,
            bottom_right_neighbour_candidates,
            bottom_left_neighbour_candidates,
        ]:
            yield self._get_neighbouring_edges(all_nodes, candidate)

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: pl.DataFrame, neighbour_candidates: pl.DataFrame
    ) -> list[tuple[int, int, HexagonEdgeInfo]]:
        # TODO: can this be combined into a single polars expression?
        # Which neighbours do exist?
        neighbours = neighbour_candidates.join(
            all_nodes.select(pl.col("node_id").alias("target_node"), pl.col("q"), pl.col("r")),
            on=["q", "r"],
            how="inner",
        )
        # Compute weights
        neighbours = neighbours.with_columns(
            (
                (
                    all_nodes.filter(all_nodes["node_id"].is_in(neighbours["source_node"]))["suitability_value"]
                    + all_nodes.filter(all_nodes["node_id"].is_in(neighbours["target_node"]))["suitability_value"]
                )
                / 2
            ).alias("weight")
        )
        edges = [
            (edge[0], edge[1], edge[2])
            for edge in neighbours.select("source_node", "target_node", "weight").iter_rows()
        ]
        return edges
