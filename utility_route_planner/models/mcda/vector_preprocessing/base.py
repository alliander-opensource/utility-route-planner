# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import typing

import fiona
import pandas
import shapely
import geopandas as gpd
import structlog
from pandas.errors import IntCastingNaNError

from settings import Config
from utility_route_planner.models.mcda.exceptions import InvalidSuitabilityValue
from utility_route_planner.util.geo_utilities import get_empty_geodataframe
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

if typing.TYPE_CHECKING:
    from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria, RasterPresetGeneral


logger = structlog.get_logger(__name__)


class VectorPreprocessorBase(abc.ABC):
    @property
    @abc.abstractmethod
    def criterion(self) -> str:
        """Name of the criterion"""

    @time_function
    def execute(
        self, general: RasterPresetGeneral, criterion: RasterPresetCriteria
    ) -> tuple[bool, gpd.GeoDataFrame, list]:
        """Run all methods in order for a criteria returning the processed geodataframe with suitability values."""
        logger.info(f"Start preprocessing: {self.criterion}.")

        prepared_gdfs = self.prepare_input_data(general.project_area_geometry, criterion, general.path_input_geopackage)
        if len(prepared_gdfs) == 1 and prepared_gdfs[0].empty:
            return False, get_empty_geodataframe(), []  # Nothing to process when there is no data available, return.
        processed_gdf = self.specific_preprocess(prepared_gdfs, criterion)
        if not self.is_valid_result(processed_gdf):
            return False, get_empty_geodataframe(), []
        height_levels = self.get_height_levels(processed_gdf)
        self.write_to_file(general.prefix, processed_gdf)

        return True, processed_gdf, height_levels

    @staticmethod
    def prepare_input_data(
        project_area: shapely.MultiPolygon, criterion: RasterPresetCriteria, path_geopackage_mcda_input
    ) -> list[gpd.GeoDataFrame]:
        """Check existing layers in geopackage / clip data / check if gdf is empty / filter historic BGT data"""
        prepared_input = []
        available_layers = fiona.listlayers(path_geopackage_mcda_input)
        for layer_name in criterion.layer_names:
            if layer_name not in available_layers:
                logger.warning(f"Layer name: {layer_name} is not available in geopackage, skipping.")
                gdf = get_empty_geodataframe()
            else:
                gdf = gpd.read_file(
                    path_geopackage_mcda_input, layer=layer_name, engine="pyogrio", bbox=project_area.bounds
                ).clip(project_area)
            if "eindRegistratie" in gdf.columns:  # BGT data has this attribute, filter historic items.
                gdf = gdf.loc[gdf["eindRegistratie"].isna()]
            if "terminationDate" in gdf.columns:  # BGT data has this attribute, filter historic items.
                gdf = gdf.loc[gdf["terminationDate"].isna()]
            gdf["suitability_value"] = pandas.NA  # Placeholder value
            if not gdf.empty:
                prepared_input.append(gdf)

        if all([i.empty for i in prepared_input]) or len(prepared_input) == 0:
            logger.warning("No data available in project area for criterion.")
            prepared_input = [get_empty_geodataframe()]

        return prepared_input

    @abc.abstractmethod
    def specific_preprocess(self, prepared_data, criterion) -> gpd.GeoDataFrame:
        """Subclasses must implement this abstract method which contains logic for handling the criteria."""

    def is_valid_result(self, processed_gdf: gpd.GeoDataFrame) -> bool:
        """Validate the result if all features were assigned a valid suitability value."""
        if processed_gdf.empty:
            logger.warning(f"No data available for criterion: {self.criterion}.")
            return False
        try:
            processed_gdf.astype({"suitability_value": int}, errors="raise")
        except (ValueError, IntCastingNaNError, TypeError):
            logger.error(
                f"Suitability value is invalid rows: {processed_gdf.loc[~processed_gdf['suitability_value'].astype(str).str.isnumeric()]}. Check mcda_presets.yaml or preprocessing function for criteria: {self.criterion}."
            )
            raise InvalidSuitabilityValue
        return True

    def get_height_levels(self, processed_gdf: gpd.GeoDataFrame, default_height: int = 0) -> list:
        """Get unique height levels from the processed geodataframe, if available."""
        if "relatieveHoogteligging" not in processed_gdf.columns:
            logger.info(f"No height levels found for criterion: {self.criterion}. Defaulting to: {default_height}.")
            processed_gdf["relatieveHoogteligging"] = default_height
            return [default_height]
        height_levels = processed_gdf["relatieveHoogteligging"].unique().tolist()
        logger.info(f"Found height levels: {height_levels} for criterion: {self.criterion}.")
        return height_levels

    def write_to_file(self, prefix, validated_gdf: gpd.GeoDataFrame) -> None:
        """Write to the geopackage for debugging and rasterizing."""
        write_results_to_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, validated_gdf, prefix + self.criterion)
