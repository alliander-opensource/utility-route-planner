from __future__ import annotations

from src.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import typing

from src.models.mcda.vector_preprocessing.validation import validate_values_to_reclassify

if typing.TYPE_CHECKING:
    from src.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class BegroeidTerreindeel(VectorPreprocessorBase):
    criterion = "begroeid_terreindeel"

    def specific_preprocess(self, input_gdf: list, criterion: RasterPresetCriteria) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf[0], criterion.weight_values)  # we only have 1 layer.
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: gpd.GeoDataFrame, weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        validate_values_to_reclassify(input_gdf["class"].unique().tolist(), weight_values)

        # Class is always filled in.
        input_gdf["sv_1"] = input_gdf["class"]
        input_gdf["sv_1"] = input_gdf["sv_1"].case_when(
            [(input_gdf["sv_1"].eq(i), weight_values[i]) for i in weight_values]
        )
        # plus-fysiekVoorkomen is optionally filled in, complementary to class.
        input_gdf["sv_2"] = input_gdf["plus-fysiekVoorkomen"]
        input_gdf["sv_2"] = input_gdf["sv_2"].case_when(
            [(input_gdf["sv_2"].eq(i), weight_values[i]) for i in weight_values]
        )
        input_gdf["suitability_value"] = input_gdf["sv_1"]
        # Overwrite suitability_value if sv_2 is filled in with a valid integer
        mask = input_gdf["sv_2"].astype(str).str.isnumeric()
        input_gdf.loc[mask, "suitability_value"] = input_gdf.loc[mask, "sv_2"]

        return input_gdf
