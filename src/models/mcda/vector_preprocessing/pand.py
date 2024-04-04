from __future__ import annotations

from src.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
import structlog
import geopandas as gpd
import typing

if typing.TYPE_CHECKING:
    from src.models.mcda.load_mcda_preset import RasterPresetCriteria

logger = structlog.get_logger(__name__)


class Pand(VectorPreprocessorBase):
    criterion = "pand"

    def specific_preprocess(self, input_gdf: list, criterion: RasterPresetCriteria) -> gpd.GeoDataFrame:
        input_gdf = self._set_suitability_values(input_gdf[0], criterion.weight_values)  # we only have 1 layer.
        return input_gdf

    @staticmethod
    def _set_suitability_values(input_gdf: gpd.GeoDataFrame, weight_values: dict) -> gpd.GeoDataFrame:
        logger.info("Setting suitability values.")

        input_gdf["suitability_value"] = weight_values.get("pand")

        return input_gdf
