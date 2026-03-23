#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import polars as pl

from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeInfo


class HexagonEdgeGenerator:
    def generate(
        self,
        block_coordinates: pl.DataFrame,
        previous_row_edge_coordinates: pl.DataFrame,
    ) -> list[tuple[int, int, HexagonEdgeInfo]]:
        """ """
        inner_block_edges = self._get_edge_candidates(block_coordinates)
        candidates = pl.concat(inner_block_edges)

        return self._get_neighbouring_edges(previous_row_edge_coordinates.lazy(), candidates.lazy())

    @staticmethod
    def _get_edge_candidates(
        block_coordinates: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        top = block_coordinates.select([pl.col("node_id").alias("source_node"), pl.col("q"), pl.col("r") + 1])
        left_top = block_coordinates.select([pl.col("node_id").alias("source_node"), pl.col("q") + 1, pl.col("r")])
        left_bottom = block_coordinates.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") + 1, pl.col("r") - 1]
        )

        right_bottom = block_coordinates.select(
            [pl.col("node_id").alias("source_node"), pl.col("q") - 1, pl.col("r") + 1]
        )

        return (top, left_top, left_bottom, right_bottom)

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: pl.LazyFrame, neighbour_candidates: pl.LazyFrame
    ) -> list[tuple[int, int, HexagonEdgeInfo]]:
        neighbours = (
            # First, for join the candidates with nodes that are actually present in the graph. The neighbour
            # candidate computation does not take the existence of nodes into account, which is resolved by dropping
            # candidate neighbours that cannot be joined
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
            # Remove duplicate edges. Due to adding candidates for four directions (instead of three), duplicate
            # edges occur in the query. By normalizing by taking the lowest and highest node id horizontally,
            # duplicates can be identified and removed in the query.
            .with_columns(
                pl.min_horizontal("source_node", "target_node").alias("edge_low"),
                pl.max_horizontal("source_node", "target_node").alias("edge_high"),
            )
            .unique(subset=["edge_low", "edge_high"])
            .select("source_node", "target_node", "weight")
            .collect()
        )
        return neighbours.rows()
