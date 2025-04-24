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


class Wegdeel(VectorPreprocessorBase):
    criterion = "wegdeel"

    def specific_preprocess(self, input_gdf: list, criterion: RasterPresetCriteria) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf[0], criterion.weight_values)  # we only have 1 layer.
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: gpd.GeoDataFrame, weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        validate_values_to_reclassify(input_gdf["function"].unique().tolist(), weight_values)

        # Function is always filled in.
        input_gdf["sv_1"] = input_gdf["function"]
        input_gdf["sv_1"] = input_gdf["sv_1"].case_when(
            [(input_gdf["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        # surfaceMaterial is always filled in.
        input_gdf["sv_2"] = input_gdf["surfaceMaterial"]
        input_gdf["sv_2"] = input_gdf["sv_2"].case_when(
            [(input_gdf["sv_2"].eq(i), weight_values[i]) for i in weight_values]
        )
        input_gdf["suitability_value"] = input_gdf["sv_1"] + input_gdf["sv_2"]

        return input_gdf
