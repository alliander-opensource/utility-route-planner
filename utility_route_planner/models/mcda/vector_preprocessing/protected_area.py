# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import typing

from utility_route_planner.util.geo_utilities import get_empty_geodataframe

if typing.TYPE_CHECKING:
    from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class ProtectedArea(VectorPreprocessorBase):
    criterion = "protected_area"

    def specific_preprocess(self, input_gdf: list, criterion: RasterPresetCriteria) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf, criterion.weight_values)  # we only have 1 layer.
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: list[gpd.GeoDataFrame], weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        gdf_kering, gdf_natura2000 = get_empty_geodataframe(), get_empty_geodataframe()
        for gdf in input_gdf:
            if "bgt-type" in gdf.columns:
                gdf_kering = gdf.copy()
                # Class is always filled in.
                gdf_kering["sv_1"] = gdf_kering["bgt-type"]
                gdf_kering["sv_1"] = gdf_kering["sv_1"].case_when(
                    [(gdf_kering["sv_1"].eq(i), weight_values[i]) for i in weight_values]
                )
                gdf_kering = gdf_kering[gdf_kering["bgt-type"] == "kering"].copy()

                gdf_kering["suitability_value"] = gdf_kering["sv_1"]
                gdf_kering = gdf_kering[["suitability_value", "geometry"]]
                gdf_kering["type"] = "kering"
            else:
                gdf_natura2000 = gdf.copy()
                gdf_natura2000["suitability_value"] = weight_values.get("natura2000")
                gdf_natura2000 = gdf_natura2000[["suitability_value", "geometry"]]
                gdf_natura2000["type"] = "natura2000"

        gdf_merged = pd.concat([gdf_kering, gdf_natura2000])

        return gdf_merged
