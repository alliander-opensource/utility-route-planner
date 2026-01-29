#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely

from settings import Config


class HexagonEdgeGenerator:
    def generate(self, grid_block: pl.DataFrame, all_blocks: pl.DataFrame) -> Iterator[gpd.GeoDataFrame]:
        q, r = grid_block.select("node_id", "axial_q"), grid_block.select("node_id", "axial_r")

        vertical_q, vertical_r = q.clone(), r.clone()
        vertical_r = vertical_r.with_columns(pl.col("axial_r") + 1)

        left_q, left_r = q.clone(), r.clone()
        left_q = left_q.with_columns(pl.col("axial_q") - 1)

        right_q, right_r = q.clone(), r.clone()
        right_q = right_q.with_columns(pl.col("axial_q") + 1)
        right_r = right_r.with_columns(pl.col("axial_r") - 1)

        for neighbour_q, neighbour_r in [
            (vertical_q, vertical_r),
            (left_q, left_r),
            (right_q, right_r),
        ]:
            yield self._get_neighbouring_edges(all_blocks, neighbour_q, neighbour_r)

    @staticmethod
    def _get_neighbouring_edges(
        all_blocks: pl.DataFrame, neighbour_q: pl.DataFrame, neighbour_r: pl.DataFrame
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
