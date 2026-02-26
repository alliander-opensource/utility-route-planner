#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator
import pandas as pd


class HexagonEdgeGenerator:
    def generate(self, hexagonal_grid: pd.DataFrame, all_nodes: pd.DataFrame) -> Iterator[list[tuple[int, int, float]]]:
        vertical_neighbour_candidates = hexagonal_grid.loc[:, ["axial_q", "axial_r"]] + (0, 1)
        left_neighbour_candidates = hexagonal_grid.loc[:, ["axial_q", "axial_r"]] - (1, 0)
        right_neighbour_candidates = hexagonal_grid.loc[:, ["axial_q", "axial_r"]] + (1, -1)

        for candidate in [vertical_neighbour_candidates, left_neighbour_candidates, right_neighbour_candidates]:
            yield self._get_neighbouring_edges(all_nodes, candidate)

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: pd.DataFrame, neighbour_candidates: pd.DataFrame
    ) -> list[tuple[int, int, float]]:
        # Which neighbours do exist?
        neighbour_candidates = neighbour_candidates.reset_index(names=["source_node"])

        neighbours = neighbour_candidates.loc[:, ["axial_q", "axial_r", "source_node"]].merge(
            all_nodes.loc[:, ["axial_q", "axial_r"]].reset_index(names=["target_node"]),
            on=["axial_q", "axial_r"],
            how="inner",
        )

        neighbours["weight"] = (
            all_nodes.loc[neighbours["source_node"], "suitability_value"].values
            + all_nodes.loc[neighbours["target_node"], "suitability_value"].values
        ) / 2

        edges = [
            (int(edge[0]), int(edge[1]), edge[2])
            for edge in neighbours.loc[:, ["source_node", "target_node", "weight"]].values
        ]
        return edges
