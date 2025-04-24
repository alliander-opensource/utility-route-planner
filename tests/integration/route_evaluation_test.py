# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import shapely
import geopandas as gpd

from settings import Config
from utility_route_planner.models.route_evaluation_metrics import RouteEvaluationMetrics


class TestRouteEvaluationMetrics:
    def test_route_costs_single_input_with_project_area(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        route = shapely.LineString([[174877.07, 451050.52], [174978.55, 451105.11]])

        route_evaluation_metrics = RouteEvaluationMetrics(route, path_raster, project_area=route.buffer(50))
        route_evaluation_metrics.get_route_evaluation_metrics()

        assert round(route_evaluation_metrics.route_relative_cost_sota) == 9969
        assert route_evaluation_metrics.route_sota.length == route.length

    def test_route_costs_two_inputs(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        route_sota = shapely.LineString([[174877.07, 451050.52], [174978.55, 451105.11]])
        route_human = shapely.LineString([[174967.92, 450902.59], [175283.58, 450783.57]])

        route_evaluation_metrics = RouteEvaluationMetrics(
            route_sota, path_raster, route_human, similarity_threshold_m=2
        )
        route_evaluation_metrics.get_route_evaluation_metrics()

        assert round(route_evaluation_metrics.route_relative_cost_sota) == 9969
        assert round(route_evaluation_metrics.route_relative_cost_human) == 30991
        assert route_evaluation_metrics.route_sota.length == route_sota.length
        assert route_evaluation_metrics.route_human.length == route_human.length
        assert route_evaluation_metrics.route_similarity_sota == 0

    def test_route_similarity(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        route_sota = shapely.LineString([[0, 0], [0, 5], [0, 10]])
        route_human = shapely.LineString([[0, 0], [1, 5], [3, 5], [3, 10], [0, 10]])

        route_evaluation_metrics = RouteEvaluationMetrics(route_sota, path_raster, route_human)
        sota_overlap_percentage, human_overlap_percentage = route_evaluation_metrics.get_route_similarity(
            route_sota, route_human, 2
        )

        assert sota_overlap_percentage == 87.3
        assert human_overlap_percentage == 53.64

        sota_overlap_percentage, human_overlap_percentage = route_evaluation_metrics.get_route_similarity(
            route_sota, route_human, 10
        )
        assert sota_overlap_percentage, human_overlap_percentage == 100

    def test_invalid_supplied_route(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        route = shapely.LineString([[0, 0], [1, 1]])  # Not within raster

        with pytest.raises(ValueError):
            route_evaluation_metrics = RouteEvaluationMetrics(route, path_raster)
            route_evaluation_metrics.get_route_evaluation_metrics()

    def test_route_costs_of_linestring_in_single_cell(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        route = shapely.LineString(
            [
                [174962.5653, 451096.9245],
                [174962.8651, 451096.8886],
                [174962.9070, 451096.6188],
                [174962.5594, 451096.5829],
            ]
        )

        route_evaluation_metrics = RouteEvaluationMetrics(route, path_raster)
        route_evaluation_metrics.get_route_evaluation_metrics()

        cost_of_single_cell = 126  # As viewed in QGIS.

        assert round(route_evaluation_metrics.route_relative_cost_sota) == round(cost_of_single_cell * route.length)

    def test_get_number_of_nodes_edges_for_a_raster(self):
        path_raster = Config.PATH_EXAMPLE_RASTER
        project_area = (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )
        route = shapely.LineString([[174877.07, 451050.52], [174978.55, 451105.11]])

        route_evaluation_metrics = RouteEvaluationMetrics(route, path_raster)
        nodes, edges = route_evaluation_metrics.get_number_of_nodes_edges(path_raster, project_area)
        assert edges / nodes <= 8
        assert nodes == 2182753
        assert edges == 17435116

    def test_get_number_of_nodes_edges_small_array(self):
        pytest_array = np.array(
            [
                [1, 2],
                [4, 5],
            ]
        )
        route_evaluation_metrics = RouteEvaluationMetrics(shapely.LineString(), "")
        n_nodes, n_edges = route_evaluation_metrics.count_cells(pytest_array, 0)
        assert n_nodes == 4
        assert n_edges == 12

    def test_get_number_of_nodes_edges_small_array_with_no_data(self):
        pytest_array = np.array(
            [
                [1, 2, 0],
                [4, 5, 0],
            ]
        )
        route_evaluation_metrics = RouteEvaluationMetrics(shapely.LineString(), "")
        n_nodes, n_edges = route_evaluation_metrics.count_cells(pytest_array, 0)
        assert n_nodes == 4
        assert n_edges == 12

    def test_get_number_of_nodes_edges_no_data(self):
        pytest_array = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        route_evaluation_metrics = RouteEvaluationMetrics(shapely.LineString(), "")
        n_nodes, n_edges = route_evaluation_metrics.count_cells(pytest_array, 0)
        assert n_nodes == 0
        assert n_edges == 0

    def test_get_number_of_nodes_edges_larger_area(self):
        pytest_array = np.array(
            [
                [0, 0, 3],  # 2
                [0, 1, 1],  # 5, 4
                [1, 1, 4],  # 2, 4, 3
            ]
        )
        route_evaluation_metrics = RouteEvaluationMetrics(shapely.LineString(), "")
        n_nodes, n_edges = route_evaluation_metrics.count_cells(pytest_array, 0)
        assert n_nodes == 6
        assert n_edges == 20
