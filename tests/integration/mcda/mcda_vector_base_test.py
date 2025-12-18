# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd

import pytest
import shapely

from settings import Config
from utility_route_planner.models.mcda.exceptions import InvalidSuitabilityValue, UnassignedValueFoundDuringReclassify

from utility_route_planner.models.mcda.load_mcda_preset import RasterPresetCriteria
from utility_route_planner.models.mcda.vector_preprocessing.base import VectorPreprocessorBase
from utility_route_planner.models.mcda.vector_preprocessing.validation import validate_values_to_reclassify
from utility_route_planner.util.geo_utilities import get_empty_geodataframe


@pytest.fixture
def setup_base_class():
    class MyVectorPreprocessor(VectorPreprocessorBase):
        criterion = "test_criterion"

        def specific_preprocess(self, **kwargs):
            pass

    return MyVectorPreprocessor()


@pytest.fixture
def setup_mock_criterion():
    description = "Test"
    layer_names = ["bgt_begroeidterreindeel_V", "bgt_kast_P", "bgt_spoor_L"]
    constraint = False
    group = "a"
    weight_values = {"value1": 10, "value2": 20}

    test_criterion = RasterPresetCriteria(
        description=description,
        layer_names=layer_names,
        constraint=constraint,
        preprocessing_function=Mock(spec=VectorPreprocessorBase),
        group=group,
        weight_values=weight_values,
    )

    return test_criterion


class TestBaseVectorPreprocessing:
    def test_base_prepare_input_happy(self, setup_base_class, setup_mock_criterion):
        base_instance = setup_base_class
        criterion = setup_mock_criterion
        project_area = (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )

        result = base_instance.prepare_input_data(project_area, criterion, Config.PYTEST_PATH_GEOPACKAGE_MCDA)
        assert len(result) == len(criterion.layer_names) - 1  # bgt_spoor is not inside the project area
        for gdf in result:
            assert gdf.columns.__contains__("suitability_value")
        raw = gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer="bgt_begroeidterreindeel_V")
        assert len(raw) > len(result[0])  # Check that the filtering worked for historic features.

    def test_base_prepare_input_data_not_in_project_area(self, setup_base_class, setup_mock_criterion):
        base_instance = setup_base_class
        criterion = setup_mock_criterion
        project_area = (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )

        criterion.layer_names = ["bgt_spoor_L"]  # Outside the project area.
        result = base_instance.prepare_input_data(project_area, criterion, Config.PYTEST_PATH_GEOPACKAGE_MCDA)
        assert len(result) == 1
        assert result[0].empty

    @pytest.mark.parametrize(
        "invalid_input", [[1, 2, 3, "invalid"], [1, 2, 3, np.nan], [1, 2, 3, [1, 2]], [1, 2, 3, None]]
    )
    def test_base_validate_result_unhappy(self, setup_base_class, invalid_input):
        base_instance = setup_base_class
        with pytest.raises(InvalidSuitabilityValue):
            base_instance.is_valid_result(gpd.GeoDataFrame({"suitability_value": invalid_input}))

    @pytest.mark.parametrize("valid_input", [[1, 211, 33], [20, 2, 300.123]])
    def test_base_validate_result_happy(self, setup_base_class, valid_input):
        base_instance = setup_base_class
        assert base_instance.is_valid_result(gpd.GeoDataFrame({"suitability_value": valid_input}))

    def test_base_validate_result_empty(self, setup_base_class):
        base_instance = setup_base_class
        assert not base_instance.is_valid_result(get_empty_geodataframe())
        assert not base_instance.is_valid_result(gpd.GeoDataFrame())


def test_all_values_present():
    values_to_reclassify = [1, 2, 3]
    assigned_values = {1: "A", 2: "B", 3: "C"}
    validate_values_to_reclassify(values_to_reclassify, assigned_values)


@pytest.mark.parametrize("input_which_should_raise", [{1: "A", 2: "B", 3: "C"}, {1: "A", 4: 0}])
def test_missing_values(input_which_should_raise):
    values_to_reclassify = [1, 2, 4]
    wrong_input = input_which_should_raise
    with pytest.raises(UnassignedValueFoundDuringReclassify):
        validate_values_to_reclassify(values_to_reclassify, wrong_input)


class DummyPreprocessor(VectorPreprocessorBase):
    criterion = "dummy_criterion"

    def specific_preprocess(self, prepared_data, criterion):
        pass


class TestGetMetrics:
    @pytest.fixture
    def get_dummy_criterion(self):
        return MagicMock(spec=RasterPresetCriteria)

    def test_get_statistics_invalid_column(self, get_dummy_criterion):
        dummy_criterion = get_dummy_criterion
        dummy_criterion.columns_to_reclassify = ["missing_col"]
        dummy_criterion.layer_names = ["layer1"]

        gdf = gpd.GeoDataFrame({"geometry": [None], "col1": ["A"], "suitability_value": [1]})
        preprocessor = DummyPreprocessor()
        with pytest.raises(ValueError):
            preprocessor.get_statistics(get_dummy_criterion, gdf)

    def test_get_statistics_no_reclassify(self, get_dummy_criterion):
        mock_criterion = get_dummy_criterion
        mock_criterion.columns_to_reclassify = []
        mock_criterion.layer_names = ["layer1"]
        gdf = gpd.GeoDataFrame({"geometry": [], "suitability_value": []})

        preprocessor = DummyPreprocessor()
        result, present_weights = preprocessor.get_statistics(mock_criterion, gdf)
        assert "criterion" in result.columns
        assert "weight_key" in result.columns
        assert "area_m2" in result.columns
        assert "suitability_value" in result.columns
        assert present_weights == ["dummy_criterion"]

    def test_get_statistics_one_reclassify_column(self, get_dummy_criterion):
        dummy_criterion = get_dummy_criterion
        dummy_criterion.columns_to_reclassify = ["col1"]
        dummy_criterion.layer_names = ["layer1"]
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [shapely.Point(0, 0).buffer(10), shapely.Point(0, 0).buffer(10)],
                "col1": ["A", "B"],
                "suitability_value": [1, 2],
            }
        )
        preprocessor = DummyPreprocessor()
        result, present_weights = preprocessor.get_statistics(dummy_criterion, gdf)
        expected_df = pd.DataFrame(
            {
                "criterion": ["dummy_criterion", "dummy_criterion"],
                "weight_key": ["A", "B"],
                "suitability_value": [1, 2],
                "area_m2": [313.6548490545941, 313.6548490545941],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df, check_dtype=False)

    def test_get_statistics_two_reclassify_columns(self, get_dummy_criterion):
        dummy_criterion = get_dummy_criterion
        get_dummy_criterion.columns_to_reclassify = ["col1", "col2"]
        get_dummy_criterion.layer_names = ["layer1"]
        example_polygon1 = shapely.Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
        example_polygon2 = shapely.Polygon([(5, 5), (5, 15), (15, 15), (15, 5), (5, 5)])
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [example_polygon1, example_polygon1, example_polygon2],
                "col1": ["A", "B", "B"],
                "col2": ["X", "Y", "Y"],
                "suitability_value": [1, 2, 2],
            }
        )
        preprocessor = DummyPreprocessor()
        result, present_weights = preprocessor.get_statistics(dummy_criterion, gdf)

        expected_df = pd.DataFrame(
            {
                "criterion": ["dummy_criterion", "dummy_criterion"],
                "weight_key": ["A: X", "B: Y"],
                "suitability_value": [1, 2],
                "area_m2": [100, 175.0],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df, check_dtype=False)

    def test_get_statistics_three_reclassify_columns_raises(self, get_dummy_criterion):
        dummy_criterion = get_dummy_criterion
        dummy_criterion.columns_to_reclassify = ["col1", "col2", "col3"]
        dummy_criterion.layer_names = ["layer1"]
        gdf = gpd.GeoDataFrame(
            {"geometry": [None], "col1": ["A"], "col2": ["B"], "col3": ["C"], "suitability_value": [1]}
        )
        preprocessor = DummyPreprocessor()
        with pytest.raises(ValueError):
            preprocessor.get_statistics(dummy_criterion, gdf)
