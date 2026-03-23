#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator
import polars as pl

from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeInfo


class HexagonEdgeGenerator:
    def generate(
        self,
        block_coordinates: pl.DataFrame,
        previous_row_edge_coordinates: pl.DataFrame,
    ) -> Iterator[list[tuple[int, int, HexagonEdgeInfo]]]:
        """ """
        inner_block_edges = self._get_edge_candidates(block_coordinates)

        for candidate in inner_block_edges:
            yield self._get_neighbouring_edges(previous_row_edge_coordinates, candidate)

    @staticmethod
    def _get_edge_candidates(
        block_coordinates: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        top = block_coordinates.select([pl.col("node_id").alias("source_node"), pl.col("q"), pl.col("r") + 1])
        left_top = block_coordinates.select([pl.col("node_id").alias("source_node"), pl.col("q") + 1, pl.col("r")])
        left_bottom = block_coordinates.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") + 1, pl.col("r") - 1]
        )

        # right_top = block_coordinates.select([pl.col("node_id").alias("source_node"), pl.col("q") -1, pl.col("r")])
        right_bottom = block_coordinates.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") - 1, pl.col("r") + 1]
        )

        return (top, left_top, left_bottom, right_bottom)

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: pl.DataFrame, neighbour_candidates: pl.DataFrame
    ) -> list[tuple[int, int, HexagonEdgeInfo]]:
        neighbours = (
            neighbour_candidates.join(
                all_nodes.select(
                    pl.col("node_id").alias("target_node"),
                    pl.col("q"),
                    pl.col("r"),
                    pl.col("suitability_value").alias("target_suitability"),
                ),
                on=["q", "r"],
                how="inner",
            )
            .unique(subset=["source_node", "target_node"])
            .join(
                all_nodes.select(
                    pl.col("node_id"),
                    pl.col("suitability_value").alias("source_suitability"),
                ),
                left_on="source_node",
                right_on="node_id",
                how="inner",
            )
            .with_columns(((pl.col("source_suitability") + pl.col("target_suitability")) / 2).alias("weight"))
        )
        edges = [
            (edge[0], edge[1], edge[2])
            for edge in neighbours.select("source_node", "target_node", "weight").iter_rows()
        ]
        return edges
