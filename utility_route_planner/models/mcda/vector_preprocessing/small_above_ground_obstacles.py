# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import pandas as pd
import typing

from utility_route_planner.models.mcda.vector_preprocessing.validation import validate_values_to_reclassify

if typing.TYPE_CHECKING:
    from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class SmallAboveGroundObstacles(VectorPreprocessorBase):
    criterion = "small_above_ground_obstacles"

    def specific_preprocess(
        self, input_gdf: list[gpd.GeoDataFrame], criterion: RasterPresetCriteria
    ) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf, criterion.weight_values)
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: list[gpd.GeoDataFrame], weight_values: dict) -> gpd.GeoDataFrame:
        # bgt_scheiding has different fields which we process first (first two gdfs).
        logger.info("Merging bgt scheiding tables.")
        # identify bgt-type geodataframes, process them first here. do not index
        bgt_scheiding = []
        bgt_others = []
        for gdf in input_gdf:
            if "bgt-type" in gdf.columns:
                bgt_scheiding.append(gdf)
            else:
                bgt_others.append(gdf)
        gdf_bgt_scheiding = pd.concat(bgt_scheiding)
        validate_values_to_reclassify(gdf_bgt_scheiding["bgt-type"].unique().tolist(), weight_values)
        logger.info("Setting suitability values.")
        # Function is always filled in.
        gdf_bgt_scheiding["sv_1"] = gdf_bgt_scheiding["bgt-type"]
        gdf_bgt_scheiding["sv_1"] = gdf_bgt_scheiding["sv_1"].case_when(
            [(gdf_bgt_scheiding["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        gdf_bgt_scheiding = gdf_bgt_scheiding[gdf_bgt_scheiding["bgt-type"] != "niet-bgt"]
        gdf_bgt_scheiding["suitability_value"] = gdf_bgt_scheiding["sv_1"]

        logger.info("Merging remaining obstacles.")
        gdf_remaining_obstacles = pd.concat(bgt_others)
        gdf_remaining_obstacles = gdf_remaining_obstacles.dropna(subset=["plus-type", "function"], how="all")
        gdf_remaining_obstacles = gdf_remaining_obstacles[
            ~(gdf_remaining_obstacles["function"].isin(["niet-bgt"]) & gdf_remaining_obstacles["plus-type"].isna())
        ]
        gdf_remaining_obstacles = gdf_remaining_obstacles[gdf_remaining_obstacles["function"] != "waardeOnbekend"]
        validate_values_to_reclassify(gdf_remaining_obstacles["plus-type"].unique().tolist(), weight_values)
        # plus-type is not always filled in.
        gdf_remaining_obstacles["sv_1"] = gdf_remaining_obstacles["plus-type"]
        gdf_remaining_obstacles["sv_1"] = gdf_remaining_obstacles["sv_1"].case_when(
            [(gdf_remaining_obstacles["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        gdf_remaining_obstacles = gdf_remaining_obstacles[gdf_remaining_obstacles["plus-type"] != "waardeOnbekend"]
        gdf_remaining_obstacles["suitability_value"] = gdf_remaining_obstacles["sv_1"]

        # Merge dfs
        gdf_merged = pd.concat([gdf_bgt_scheiding, gdf_remaining_obstacles])

        return gdf_merged
