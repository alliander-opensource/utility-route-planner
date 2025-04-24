# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import pandas as pd
import numpy as np
import typing

from utility_route_planner.models.mcda.vector_preprocessing.validation import validate_values_to_reclassify

if typing.TYPE_CHECKING:
    from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class VegetationObject(VectorPreprocessorBase):
    criterion = "vegetation_object"

    def specific_preprocess(
        self, input_gdf: list[gpd.GeoDataFrame], criterion: RasterPresetCriteria
    ) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf, criterion.weight_values)
        input_gdf = self._update_geometry_values(input_gdf, criterion.geometry_values)  # type: ignore
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: list[gpd.GeoDataFrame], weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        gdf_vegetation = pd.concat([*input_gdf])
        validate_values_to_reclassify(gdf_vegetation["plus-type"].unique().tolist(), weight_values)

        # Class is always filled in.
        gdf_vegetation["sv_1"] = gdf_vegetation["plus-type"]
        gdf_vegetation["sv_1"] = gdf_vegetation["sv_1"].case_when(
            [(gdf_vegetation["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        gdf_vegetation = gdf_vegetation[gdf_vegetation["plus-type"] != "waardeOnbekend"].copy()

        gdf_vegetation["suitability_value"] = gdf_vegetation["sv_1"]

        return gdf_vegetation

    @staticmethod
    def _update_geometry_values(input_gdf: gpd.GeoDataFrame, buffer_values: dict):
        logger.info("Updating geometry values.")

        for key, value in buffer_values.items():
            input_gdf["geometry"] = np.where(
                input_gdf["plus-type"].eq(key), input_gdf["geometry"].buffer(value), input_gdf["geometry"]
            )

        return input_gdf
