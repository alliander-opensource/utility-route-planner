# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
from dataclasses import dataclass

from settings import Config


@dataclass
class BenchmarkRoute:
    path_geopackage: pathlib.Path
    layer_name_project_area: str
    layer_name_human_designed_route: str
    raster_name_prefix: str
    stops: list[list[float]]


@dataclass
class BenchmarkRouteCollection:
    benchmark_routes = {
        1: BenchmarkRoute(
            Config.PATH_GEOPACKAGE_CASE_01,
            Config.LAYER_NAME_PROJECT_AREA_CASE_01,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_01,
            "route_1_",
            [],
        ),
        2: BenchmarkRoute(
            Config.PATH_GEOPACKAGE_CASE_02,
            Config.LAYER_NAME_PROJECT_AREA_CASE_02,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_02,
            "route_2_",
            [],
        ),
        3: BenchmarkRoute(
            Config.PATH_GEOPACKAGE_CASE_03,
            Config.LAYER_NAME_PROJECT_AREA_CASE_03,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_03,
            "route_3_",
            [],
        ),
        4: BenchmarkRoute(
            Config.PATH_GEOPACKAGE_CASE_04,
            Config.LAYER_NAME_PROJECT_AREA_CASE_04,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_04,
            "route_4_",
            [],
        ),
        5: BenchmarkRoute(
            Config.PATH_GEOPACKAGE_CASE_05,
            Config.LAYER_NAME_PROJECT_AREA_CASE_05,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_05,
            "route_5_",
            [[121462.8, 487153.4]],
        ),
    }

    def get_benchmark_route(self, benchmark_id: int):
        if benchmark_id not in self.benchmark_routes.keys():
            raise ValueError(f"No benchmark route found for id: {benchmark_id}")
        return self.benchmark_routes[benchmark_id]

    def get_all_benchmarks(self) -> list[BenchmarkRoute]:
        return list(self.benchmark_routes.values())
