# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
from dataclasses import dataclass
from datetime import datetime

import shapely
from settings import Config


@dataclass
class BenchmarkRoute:
    path_geopackage: pathlib.Path
    layer_name_project_area: str
    layer_name_human_designed_route: str
    raster_name_prefix: str
    stops: list[list[float]]
    construction_date: datetime
    # Optional for running smaller bits of the project
    custom_project_area: shapely.Polygon = shapely.Polygon()


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
            custom_project_area=shapely.Polygon(
                [
                    (202639.41, 495198.44),
                    (202711.57, 498061.02),
                    (202243.66, 498975.08),
                    (201188.14, 499377.71),
                    (200894.33, 499834.73),
                    (200099.97, 501586.68),
                    (199120.62, 502206.94),
                    (196264.78449903356, 499988.4143900737),
                    (196793.55272657756, 499167.7661009254),
                    (196721.64024763159, 498981.6396848299),
                    (197051.59162161904, 498609.3868526389),
                    (197656.13233617018, 497836.9622258426),
                    (198059.31810967252, 497315.59675348416),
                    (198905.77028832512, 496326.16564610373),
                    (198927.5555393, 496158.01734974474),
                    (201224.10170516933, 495934.66565043014),
                    (202639.41, 495198.44),
                ]
            ),
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
