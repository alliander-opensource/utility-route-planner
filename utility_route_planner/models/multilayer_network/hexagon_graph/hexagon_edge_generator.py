#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import geopandas as gpd
import pandas as pd


class HexagonEdgeGenerator:
    def generate(
        self, hexagonal_grid: gpd.GeoDataFrame, all_nodes: dict[tuple[int, int], int]
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
        all_nodes: dict[tuple[int, int], int],
        hexagonal_grid: pd.DataFrame,
        neighbour_q: pd.Series,
        neighbour_r: pd.Series,
    ):
        neighbour_candidates = pd.concat([neighbour_q, neighbour_r], axis=1)
        candidate_coordinates = list(neighbour_candidates.loc[:, ["axial_q", "axial_r"]].itertuples(index=False))

        neighbour_nodes = [all_nodes.get(candidate, None) for candidate in candidate_coordinates]

        edge_attr = list(range(len(hexagonal_grid.index)))
        edges = list(zip(hexagonal_grid.index, neighbour_nodes, edge_attr))
        edges_filtered = [edge for edge in edges if edge[1] is not None]

        # neighbours = pd.merge(
        #     hexagonal_grid,
        #     neighbour_candidates,
        #     how="inner",
        #     on=["axial_q", "axial_r"],
        # )
        #
        # neighbours["weight"] = (
        #     all_blocks.loc[neighbours["node_id_source"], "suitability_value"].values
        #     + all_blocks.loc[neighbours["node_id_target"], "suitability_value"].values
        # ) / 2
        #
        # line_string_coords = np.stack(
        #     [
        #         all_blocks.loc[neighbours["node_id_source"], ["x", "y"]].values,
        #         all_blocks.loc[neighbours["node_id_target"], ["x", "y"]].values,
        #     ],
        #     axis=1,
        # )
        # edge_line_strings = shapely.linestrings(line_string_coords)
        # neighbours = gpd.GeoDataFrame(neighbours, geometry=edge_line_strings, crs=Config.CRS)
        # neighbours["length"] = neighbours.geometry.length

        return edges_filtered
