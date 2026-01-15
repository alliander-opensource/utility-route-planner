#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from settings import Config


class HexagonEdgeGenerator:
    def generate(self, hexagonal_grid: gpd.GeoDataFrame, all_blocks: pd.DataFrame) -> Iterator[gpd.GeoDataFrame]:
        q, r = hexagonal_grid["axial_q"], hexagonal_grid["axial_r"]

        vertical_q, vertical_r = q, r + 1
        left_q, left_r = q - 1, r
        right_q, right_r = q + 1, r - 1

        for neighbour_q, neighbour_r in [
            (vertical_q, vertical_r),
            (left_q, left_r),
            (right_q, right_r),
        ]:
            yield self._get_neighbouring_edges(all_blocks, neighbour_q, neighbour_r)

    @staticmethod
    def _get_neighbouring_edges(
        all_blocks: pd.DataFrame, neighbour_q: pd.Series, neighbour_r: pd.Series
    ) -> gpd.GeoDataFrame:
        neighbour_candidates = pd.concat([neighbour_q, neighbour_r], axis=1)
        neighbour_candidates["node_id_source"] = neighbour_candidates.index
        all_blocks["node_id_target"] = all_blocks.index

        neighbours = pd.merge(
            neighbour_candidates,
            all_blocks[["axial_q", "axial_r", "node_id_target"]],
            how="inner",
            on=["axial_q", "axial_r"],
        )

        neighbours["weight"] = (
            all_blocks.loc[neighbours["node_id_source"], "suitability_value"].values
            + all_blocks.loc[neighbours["node_id_target"], "suitability_value"].values
        ) / 2

        line_string_coords = np.stack(
            [
                all_blocks.loc[neighbours["node_id_source"], ["x", "y"]].values,
                all_blocks.loc[neighbours["node_id_target"], ["x", "y"]].values,
            ],
            axis=1,
        )
        edge_line_strings = shapely.linestrings(line_string_coords)
        neighbours = gpd.GeoDataFrame(neighbours, geometry=edge_line_strings, crs=Config.CRS)
        neighbours["length"] = neighbours.geometry.length

        return neighbours[["node_id_source", "node_id_target", "length", "weight", "geometry"]]
