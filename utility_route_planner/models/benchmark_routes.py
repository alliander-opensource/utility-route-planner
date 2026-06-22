# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
from dataclasses import dataclass
from datetime import datetime

from settings import Config


@dataclass
class BenchmarkRoute:
    path_geopackage: pathlib.Path
    layer_name_project_area: str
    layer_name_human_designed_route: str
    raster_name_prefix: str
    stops: list[list[float]]
    construction_date: datetime


@dataclass
class BenchmarkRouteCollection:
    benchmark_routes = {
        1: BenchmarkRoute(
            Config.BASEDIR / "data/examples/case_01.gpkg",
            "ps_case_01_project_area",
            "ps_case_01_route_human_designed",
            "route_1_",
            [],
            datetime(year=2022, month=3, day=25),
        ),
        2: BenchmarkRoute(
            Config.BASEDIR / "data/examples/case_02.gpkg",
            "ps_case_02_project_area",
            "ps_case_02_route_human_designed",
            "route_2_",
            [],
            datetime(year=2018, month=12, day=10),
        ),
        3: BenchmarkRoute(
            Config.BASEDIR / "data/examples/case_03.gpkg",
            "ps_case_03_project_area",
            "ps_case_03_route_human_designed",
            "route_3_",
            [],
            datetime(year=2020, month=9, day=16),
        ),
        4: BenchmarkRoute(
            Config.BASEDIR / "data/examples/case_04.gpkg",
            "ps_case_04_project_area",
            "ps_case_04_route_human_designed",
            "route_4_",
            [],
            datetime(year=2021, month=2, day=3),
        ),
        5: BenchmarkRoute(
            Config.BASEDIR / "data/examples/case_05.gpkg",
            "ps_case_05_project_area",
            "ps_case_05_route_human_designed",
            "route_5_",
            [[121462.8, 487153.4]],
            datetime(year=2021, month=5, day=9),
        ),
    }

    def get_case(self, benchmark_id: int):
        if benchmark_id not in self.benchmark_routes.keys():
            raise ValueError(f"No benchmark route found for id: {benchmark_id}")
        return self.benchmark_routes[benchmark_id]

    def get_all_benchmarks(self) -> list[BenchmarkRoute]:
        return list(self.benchmark_routes.values())

    def get_all_case_ids(self) -> list[int]:
        return list(self.benchmark_routes.keys())
