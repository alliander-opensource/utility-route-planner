#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import polars as pl


class HexagonEdgeGenerator:
    def generate(
        self,
        block_coordinates: pl.DataFrame,
        previous_row_edge_coordinates: pl.DataFrame,
    ) -> list[tuple[int, int, float]]:
        edge_candidates = self._get_edge_candidates(block_coordinates)

        # Use lazy dataframe to allow the query to make use of Polars query optimization.
        return self._get_neighbouring_edges(previous_row_edge_coordinates.lazy(), edge_candidates.lazy())

    @staticmethod
    def _get_edge_candidates(
        block_coordinates: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        For each node in the block, compute its potential edges by considering the neighbours based on the
        axial coordinates. By using axial coordinates, all potential edges can be computed at once by
        cross-joining an offset dataframe.

        Note: as the grid is constructed in block-wise fashion, we need to compute in four instead of three
        directions which causes duplicates. Using only three directions results in missing cross-block edges.
        Duplicates are handled when generating the actual edges.

        Information on the offsets can be found here: https://www.redblobgames.com/grids/hexagons/#neighbors-axial

        :param block_coordinates: all axial coordinates in a single block to compute potential neighbours for
        :return: axial coordinates of all potential neighbours per node in the block
        """

        offsets = pl.DataFrame(
            data=[
                # Vertical neighbour candidates (q, r+1)
                [0, 1],
                # Top-right neighbour candidates (q+1, r)
                [-1, +1],
                # Bottom-right neighbour candidates (q+1, r-1)
                [-1, 0],
                # Top-left neighbour candidates (q-1, r+1)
                [+1, 0],
            ],
            schema={"dq": pl.Int8, "dr": pl.Int8},
            orient="row",
        )

        neighbour_candidates = block_coordinates.join(offsets, how="cross").select(
            pl.col("node_id").alias("source_node"),
            (pl.col("q") + pl.col("dq")).alias("q"),
            (pl.col("r") + pl.col("dr")).alias("r"),
        )

        return neighbour_candidates

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: pl.LazyFrame, neighbour_candidates: pl.LazyFrame
    ) -> list[tuple[int, int, float]]:
        """
        For each node, determine which neighbours candidates are valid. For each valid neighbour, the suitability
        value is computed by (source_node_suitability + target_node_suitability) / 2. All valid neighbours are
        transformed into edges

        :param all_nodes: lazy dataframe which contains all nodes in the current block + previous nodes in the current
        or previous row which could be cross-edge candidates.
        :param neighbour_candidates: lazy dataframe which contains the potential neighbours per node in the current block
        :return: list of tuples containing all valid edges with suitability values in the current block.
        """
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
            .with_columns((pl.col("source_suitability") + pl.col("target_suitability")).alias("weight"))
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
