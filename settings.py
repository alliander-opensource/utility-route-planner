# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import logging


class Config:
    # General
    BASEDIR = Path(__file__).parent
    LOG_LEVEL = int(os.environ.get("LOG_LEVEL", logging.INFO))
    DEBUG = False
    CRS = 28992  # https://epsg.io/28992

    # MCDA
    RASTER_PRESET_NAME_BENCHMARK = "preset_benchmark_raw"
    RASTER_CELL_SIZE = 0.5
    MAX_BLOCK_SIZE = 2048
    # No data is ignored during creation of the raster.
    INTERMEDIATE_RASTER_NO_DATA = -32768
    # To prevent unwanted rounding/capping at the intermediate steps, allow larger values as int16 datatype.
    INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER = -32767
    INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER = 32767
    # No data is set for areas: outside the project area, manually set, invalid data which are ignored during LCPA.
    FINAL_RASTER_NO_DATA = 0
    # Cap final data to the int8 datatype.
    FINAL_RASTER_VALUE_LIMIT_LOWER = 1
    FINAL_RASTER_VALUE_LIMIT_UPPER = 126

    # input/output paths.
    PATH_RESULTS = BASEDIR / "data/processed"
    PATH_GEOPACKAGE_MCDA_OUTPUT = BASEDIR / "data/processed/mcda_output.gpkg"
    PATH_GEOPACKAGE_LCPA_OUTPUT = BASEDIR / "data/processed/lcpa_results.gpkg"

    # Testing paths.
    PATH_EXAMPLE_RASTER = BASEDIR / "data/examples/pytest_example_suitability_raster.tif"
    PYTEST_PATH_GEOPACKAGE_MCDA = BASEDIR / "data/examples/pytest_data.gpkg"
    PYTEST_LAYER_NAME_PROJECT_AREA = "project_area_ede"

    # Research question 1 data paths.
    PATH_GEOPACKAGE_CASE_01 = BASEDIR / "data/examples/case_01.gpkg"
    PATH_GEOPACKAGE_CASE_02 = BASEDIR / "data/examples/case_02.gpkg"
    PATH_GEOPACKAGE_CASE_03 = BASEDIR / "data/examples/case_03.gpkg"
    PATH_GEOPACKAGE_CASE_04 = BASEDIR / "data/examples/case_04.gpkg"
    PATH_GEOPACKAGE_CASE_05 = BASEDIR / "data/examples/case_05.gpkg"

    LAYER_NAME_PROJECT_AREA_CASE_01 = "ps_case_01_project_area"
    LAYER_NAME_PROJECT_AREA_CASE_02 = "ps_case_02_project_area"
    LAYER_NAME_PROJECT_AREA_CASE_03 = "ps_case_03_project_area"
    LAYER_NAME_PROJECT_AREA_CASE_04 = "ps_case_04_project_area"
    LAYER_NAME_PROJECT_AREA_CASE_05 = "ps_case_05_project_area"

    LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_01 = "ps_case_01_route_human_designed"
    LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_02 = "ps_case_02_route_human_designed"
    LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_03 = "ps_case_03_route_human_designed"
    LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_04 = "ps_case_04_route_human_designed"
    LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_05 = "ps_case_05_route_human_designed"
