# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import geopandas as gpd
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.exceptions import NoGraphDataForProjectArea
from utility_route_planner.models.multilayer_network.osm_graph_downloader import OSMGraphDownloader


@pytest.mark.skip(reason="For manual testing only, requires internet connection.")
class TestOSMGraphDownloader:
    @pytest.fixture
    def osm_district_setup(self) -> OSMGraphDownloader:
        project_area = gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
        osm_graph_downloader = OSMGraphDownloader(project_area)

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
        northsea_polygon = shapely.Point(40466, 594514).buffer(5)
        osm_graph_downloader = OSMGraphDownloader(project_area_geometry=northsea_polygon)

        with pytest.raises(NoGraphDataForProjectArea):
            osm_graph_downloader.download_graph()
