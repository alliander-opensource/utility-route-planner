#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator
import pandas as pd

from utility_route_planner.util.timer import time_function


class HexagonEdgeGenerator:
    def generate(self, hexagonal_grid: pd.DataFrame, all_nodes: pd.DataFrame) -> Iterator[list[tuple[int, int, float]]]:
        q, r = hexagonal_grid["axial_q"], hexagonal_grid["axial_r"]

        vertical_q, vertical_r = q, r + 1
        left_q, left_r = q - 1, r
        right_q, right_r = q + 1, r - 1

        for neighbour_q, neighbour_r in [
            (vertical_q, vertical_r),
            (left_q, left_r),
            (right_q, right_r),
        ]:
            yield self._get_neighbouring_edges(all_nodes, neighbour_q, neighbour_r)

    @staticmethod
    @time_function
    def _get_neighbouring_edges(
        all_nodes: pd.DataFrame,
        neighbour_q: pd.Series,
        neighbour_r: pd.Series,
    ) -> list[tuple[int, int, float]]:
        all_nodes_copy = all_nodes.copy()

        # Which neighbours do exist?
        all_nodes_copy = all_nodes_copy.reset_index(names="target_node")
        neighbour_candidates = pd.concat([neighbour_q, neighbour_r], axis=1)
        neighbour_candidates = neighbour_candidates.reset_index(names=["source_node"])

        neighbours = neighbour_candidates.loc[:, ["axial_q", "axial_r", "source_node"]].merge(
            all_nodes_copy.loc[:, ["axial_q", "axial_r", "target_node"]], on=["axial_q", "axial_r"], how="inner"
        )
        neighbours["weight"] = (
            all_nodes.loc[neighbours["source_node"], "suitability_value"].values
            + all_nodes.loc[neighbours["target_node"], "suitability_value"].values
        ) / 2

        edges = list(neighbours.loc[:, ["source_node", "target_node", "weight"]].itertuples(index=False))
        return edges
