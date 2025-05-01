# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import geopandas as gpd
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.exceptions import NoGraphDataForProjectArea
from utility_route_planner.util.osm_graph_downloader import OSMGraphDownloader


class TestOSMGraphDownloader:
    @pytest.fixture
    def osm_district_setup(self) -> OSMGraphDownloader:
        project_area = gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
        max_cable_length = 50
        osm_graph_downloader = OSMGraphDownloader(project_area, max_cable_length)

        return osm_graph_downloader

    def test_download_valid_graph(self, osm_district_setup: OSMGraphDownloader):
        osm_graph_io = osm_district_setup
        project_area_graph = osm_graph_io.download_graph()

        assert project_area_graph.number_of_edges() > 0
        assert project_area_graph.number_of_nodes() > 0
        assert project_area_graph.graph["crs"].srs == "EPSG:28992"

    def test_invalid_project_area_geometry_raises_no_graph_for_project_area(
        self, osm_district_setup: OSMGraphDownloader
    ):
        # Choose a point that is located in the North Sea (and therefore does not have a graph)
        northsea_polygon = gpd.GeoDataFrame(geometry=[shapely.Point(40466, 594514)], crs=Config.CRS)
        osm_graph_downloader = OSMGraphDownloader(project_area_geometry=northsea_polygon, max_cable_length=1)

        with pytest.raises(NoGraphDataForProjectArea):
            osm_graph_downloader.download_graph()
