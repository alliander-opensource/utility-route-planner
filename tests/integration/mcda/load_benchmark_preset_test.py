# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import geopandas as gpd

from settings import Config
from utility_route_planner.models.mcda.exceptions import InvalidSuitabilityValue, InvalidLayerName, InvalidGroupValue
from utility_route_planner.models.mcda.load_mcda_preset import load_preset, validate_layer_names
from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria


def test_load_benchmark_with_default_settings_as_str():
    # Pydantic validates the values in the model.
    load_preset(
        Config.RASTER_PRESET_NAME_BENCHMARK,
        Config.PYTEST_PATH_GEOPACKAGE_MCDA,
        gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry,
    )


def test_invalid_preset_name():
    with pytest.raises(ValueError):
        load_preset(
            "this_preset_does_not_exist_and_should_raise_an_error",
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )


@pytest.mark.parametrize("invalid_input", [1, False, None, [1, 2, 3]])
def test_invalid_input(invalid_input):
    with pytest.raises(ValueError):
        load_preset(
            invalid_input,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )


@pytest.fixture
def setup_raster_preset_dummy():
    dummy_preset = {
        "general": {
            "description": "Dummy preset.",
            "prefix": "b_",
            "final_raster_name": "benchmark_suitability_raster",
            "project_area_geometry": gpd.read_file(
                Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA
            )
            .iloc[0]
            .geometry,
        },
        "criteria": {
            "test_criteria": {
                "description": "Information on something.",
                "layer_names": ["my_layer_name"],
                "preprocessing_function": MagicMock(spec=VectorPreprocessorBase),
                "constraint": False,
                "group": "b",
                "weight_values": {
                    "w_dummy": 50,
                    "w_dummy2": 1,
                },
                "geometry_values": {"buffer_m_dummy": 20},
            }
        },
    }
    yield dummy_preset


class TestCriteriaInput:
    def test_invalid_group(self):
        with pytest.raises(InvalidGroupValue):
            RasterPresetCriteria.validate_group("d")

    def test_correct_group(self):
        RasterPresetCriteria.validate_group("a")
        RasterPresetCriteria.validate_group("b")
        RasterPresetCriteria.validate_group("c")

    @pytest.mark.parametrize(
        "valid_input",
        [
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER,
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER + 1,
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER,
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER - 1,
        ],
    )
    def test_correct_weight_values(self, valid_input):
        weight_to_verify = {"w_dummy": valid_input}
        RasterPresetCriteria.validate_weights(weight_to_verify)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER - 1,
            Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER + 1,
            1.02,
            "str",
            True,
            False,
        ],
    )
    def test_invalid_weight_values(self, invalid_input):
        with pytest.raises(InvalidSuitabilityValue):
            weight_to_verify = {"w_dummy": invalid_input}
            RasterPresetCriteria.validate_weights(weight_to_verify)

    @pytest.mark.parametrize(
        "invalid_input",
        [[1, 2], [1], [False], ["layer_name_invalid_too"], ["layer_name_invalid", False]],
    )
    def test_invalid_layer_name_values(self, invalid_input):
        existing_layers = ["valid_layer1", "valid_layer2"]
        with pytest.raises(InvalidLayerName):
            validate_layer_names(existing_layers, invalid_input)

    @pytest.mark.parametrize(
        "valid_input",
        [["single_layer"], ["layer1", "layer2"]],
    )
    def test_valid_layer_name_values(self, valid_input, setup_raster_preset_dummy):
        existing_layers = ["single_layer", "layer1", "layer2"]
        validate_layer_names(existing_layers, valid_input)
