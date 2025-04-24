# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import typing

from utility_route_planner.models.mcda.vector_preprocessing.validation import validate_values_to_reclassify

if typing.TYPE_CHECKING:
    from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class OverigBouwwerk(VectorPreprocessorBase):
    criterion = "overig_bouwwerk"

    def specific_preprocess(self, input_gdf: list, criterion: RasterPresetCriteria) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf[0], criterion.weight_values)  # we only have 1 layer.
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: gpd.GeoDataFrame, weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        validate_values_to_reclassify(input_gdf["bgt-type"].unique().tolist(), weight_values)

        # Class is always filled in.
        input_gdf["sv_1"] = input_gdf["bgt-type"]
        input_gdf["sv_1"] = input_gdf["sv_1"].case_when(
            [(input_gdf["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        input_gdf = input_gdf[input_gdf["bgt-type"] != "niet-bgt"].copy()

        input_gdf.loc[:, "suitability_value"] = input_gdf["sv_1"]

        return input_gdf
