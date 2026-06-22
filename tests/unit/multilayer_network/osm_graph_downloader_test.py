# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
import osmnx as ox
import pytest
import geopandas as gpd
import shapely

from settings import Config
from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.multilayer_network.exceptions import NoGraphDataForProjectArea
from utility_route_planner.models.multilayer_network.osm_graph_downloader import OSMGraphDownloader


class TestOSMGraphDownloader:
    @pytest.fixture
    def get_example_polygon(self) -> shapely.Polygon:
        return gpd.read_file(
            Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA
        ).geometry.iloc[0]

    @pytest.mark.skip(reason="For manual testing only, requires internet connection.")
    def test_download_valid_graph(self, get_example_polygon: shapely.Polygon):
        osm_graph_io = OSMGraphDownloader(project_area_geometry=get_example_polygon)
        project_area_graph = osm_graph_io.download_graph()

        assert project_area_graph.number_of_edges() > 0
        assert project_area_graph.number_of_nodes() > 0
        assert project_area_graph.graph["crs"].srs == "EPSG:28992"

    @pytest.mark.skip(reason="For manual testing only, requires internet connection.")
    def test_invalid_project_area_geometry_raises_no_graph_for_project_area(self):
        # Choose a point that is located in the North Sea (and therefore does not have a graph)
        northsea_polygon = shapely.Point(40466, 594514).buffer(5)
        osm_graph_downloader = OSMGraphDownloader(project_area_geometry=northsea_polygon)

        with pytest.raises(NoGraphDataForProjectArea):
            osm_graph_downloader.download_graph()

    @pytest.mark.skip(reason="For manual testing only, requires internet connection.")
    def test_download_benchmark_case_with_date(self):
        benchmarks = BenchmarkRouteCollection()
        benchmark_case = benchmarks.get_case(4)
        osm_graph_downloader = OSMGraphDownloader(
            project_area_geometry=gpd.read_file(
                benchmark_case.project_area_geometry, layer=benchmark_case.layer_name_project_area
            ).geometry.iloc[0],
            graph_date=benchmark_case.construction_date,
        )
        project_area_graph = osm_graph_downloader.download_graph()

        assert project_area_graph.number_of_edges() > 0
        assert project_area_graph.number_of_nodes() > 0
        assert project_area_graph.graph["crs"].srs == "EPSG:28992"

    def test_no_graph_date_returns_default_settings(self, get_example_polygon: shapely.Polygon):
        downloader = OSMGraphDownloader(get_example_polygon, graph_date=None)
        default_settings = ox.settings.overpass_settings

        result = downloader.build_overpass_settings()

        assert result == default_settings

    def test_with_graph_date_appends_date_clause(self, get_example_polygon: shapely.Polygon):
        graph_date = datetime(2023, 5, 15, 14, 30, 0)
        downloader = OSMGraphDownloader(get_example_polygon, graph_date=graph_date)

        result = downloader.build_overpass_settings()

        expected = '[out:json][timeout:{timeout}]{maxsize}[date:"2023-05-15T00:00:00Z"]'
        assert result == expected
