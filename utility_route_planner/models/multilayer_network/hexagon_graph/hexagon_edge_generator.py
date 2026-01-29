#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
from typing import Iterator

import geopandas as gpd
import numpy as np
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
        neighbour_candidates = neighbour_q.join(neighbour_r, on="node_id").rename({"node_id": "node_id_source"})
        all_blocks_clone = all_blocks.clone().rename({"node_id": "node_id_target"})

        neighbours = neighbour_candidates.join(
            all_blocks_clone.select("axial_q", "axial_r", "node_id_target"), how="inner", on=["axial_q", "axial_r"]
        )

        edge_weight = (
            all_blocks_clone.filter(pl.col("node_id_target").is_in(neighbours["node_id_source"]))["suitability_value"]
            + all_blocks_clone.filter(pl.col("node_id_target").is_in(neighbours["node_id_target"]))["suitability_value"]
        ) / 2
        neighbours = neighbours.with_columns(edge_weight.alias("weight"))

        line_string_coords = np.stack(
            [
                all_blocks_clone.loc[neighbours["node_id_source"], ["x", "y"]].values,
                all_blocks_clone.loc[neighbours["node_id_target"], ["x", "y"]].values,
            ],
            axis=1,
        )
        edge_line_strings = shapely.linestrings(line_string_coords)
        neighbours = gpd.GeoDataFrame(neighbours, geometry=edge_line_strings, crs=Config.CRS)
        neighbours["length"] = neighbours.geometry.length

        return neighbours[["node_id_source", "node_id_target", "length", "weight", "geometry"]]
