#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import geopandas as gpd
import pandas as pd

from utility_route_planner.models.multilayer_network.graph_datastructures import TempNode


class HexagonEdgeGenerator:
    def generate(
        self, hexagonal_grid: gpd.GeoDataFrame, all_nodes: dict[tuple[int, int], TempNode]
    ) -> Iterator[gpd.GeoDataFrame]:
        q, r = hexagonal_grid["axial_q"], hexagonal_grid["axial_r"]

        vertical_q, vertical_r = q, r + 1
        left_q, left_r = q - 1, r
        right_q, right_r = q + 1, r - 1

        for neighbour_q, neighbour_r in [
            (vertical_q, vertical_r),
            (left_q, left_r),
            (right_q, right_r),
        ]:
            yield self._get_neighbouring_edges(all_nodes, hexagonal_grid, neighbour_q, neighbour_r)

    @staticmethod
    def _get_neighbouring_edges(
        all_nodes: dict[tuple[int, int], TempNode],
        hexagonal_grid: pd.DataFrame,
        neighbour_q: pd.Series,
        neighbour_r: pd.Series,
    ):
        # TODO: convert to pandas approach for speed
        neighbour_candidates = pd.concat([neighbour_q, neighbour_r], axis=1)
        candidate_coordinates = list(neighbour_candidates.loc[:, ["axial_q", "axial_r"]].itertuples(index=False))

        neighbour_nodes = [all_nodes.get(candidate, None) for candidate in candidate_coordinates]

        edges = [
            (source_node_id, neighbour.node_id, (neighbour.suitability_value + source_suitability_value) / 2)
            for (source_node_id, source_suitability_value), neighbour in zip(
                hexagonal_grid.loc[:, "suitability_value"].items(), neighbour_nodes
            )
            if neighbour is not None
        ]

        return edges
