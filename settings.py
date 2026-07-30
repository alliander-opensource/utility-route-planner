# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv(override=True)  # reads variables from a .env file and sets them in os.environ


class Config:
    # General
    BASEDIR = Path(__file__).parent
    LOG_LEVEL = int(os.environ.get("LOG_LEVEL", logging.INFO))
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
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

    # Multilayer network
    OSM_API_TIMEOUT_IN_SECONDS = 20
    MIN_NODE_SUITABILITY_VALUE = 1
    MAX_NODE_SUITABILITY_VALUE = 126
    HEXAGON_SIZE = 0.5
    HEXAGON_BLOCK_SIZE = 512
    THRESHOLD_EDGE_LENGTH_CROSSING_M: float = 30
    APPLY_PIPE_RAMMING: bool = True
    MAX_PIPE_RAMMING_LENGTH_M: float = 15
    MIN_PIPE_RAMMING_LENGTH_M: float = 3
    SUITABILITY_VALUE_CROSSING_THRESHOLD: float = 10
    SUITABILITY_VALUE_OBSTACLES_THRESHOLD: float = 76

    # input/output paths.
    PATH_RESULTS = BASEDIR / "data/processed"
    PATH_GEOPACKAGE_MCDA_OUTPUT = BASEDIR / "data/processed/mcda_output.gpkg"
    PATH_GEOPACKAGE_LCPA_OUTPUT = BASEDIR / "data/processed/lcpa_results.gpkg"
    PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT = BASEDIR / "data/processed/multilayer_network.gpkg"
    PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT = BASEDIR / "data/processed/hexagon_graph.gpkg"

    # Testing paths.
    PATH_EXAMPLE_RASTER = BASEDIR / "data/examples/pytest_example_suitability_raster.tif"
    PYTEST_PATH_GEOPACKAGE_MCDA = BASEDIR / "data/examples/pytest_data.gpkg"
    PYTEST_LAYER_NAME_PROJECT_AREA = "project_area_ede"
    PYTEST_OSM_GRAPH_PICKLE = BASEDIR / "data/examples/pytest_osm_graph.pkl"
