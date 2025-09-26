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


class BenchmarkRouteCollection:
    route_1: BenchmarkRoute = BenchmarkRoute(
        Config.PATH_GEOPACKAGE_CASE_01,
        Config.LAYER_NAME_PROJECT_AREA_CASE_01,
        Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_01,
        "route_1_",
        [],
    )
    route_2: BenchmarkRoute = BenchmarkRoute(
        Config.PATH_GEOPACKAGE_CASE_02,
        Config.LAYER_NAME_PROJECT_AREA_CASE_02,
        Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_02,
        "route_2_",
        [],
    )
    route_3: BenchmarkRoute = BenchmarkRoute(
        Config.PATH_GEOPACKAGE_CASE_03,
        Config.LAYER_NAME_PROJECT_AREA_CASE_03,
        Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_03,
        "route_3_",
        [],
    )
    route_4: BenchmarkRoute = BenchmarkRoute(
        Config.PATH_GEOPACKAGE_CASE_04,
        Config.LAYER_NAME_PROJECT_AREA_CASE_04,
        Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_04,
        "route_4_",
        [],
    )
    route_5: BenchmarkRoute = BenchmarkRoute(
        Config.PATH_GEOPACKAGE_CASE_05,
        Config.LAYER_NAME_PROJECT_AREA_CASE_05,
        Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_05,
        "route_5_",
        [[121462.8, 487153.4]],
    )

    def get_routes(self, route_numbers: list = []) -> list[BenchmarkRoute]:
        route_map = {
            1: self.route_1,
            2: self.route_2,
            3: self.route_3,
            4: self.route_4,
            5: self.route_5,
        }
        if route_numbers:
            return list(route_map.values())
        return [route_map[n] for n in route_numbers if n in route_map]
