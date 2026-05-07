# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pickle

import pytest
from networkx import MultiDiGraph

from settings import Config


@pytest.fixture
def load_osm_graph_pickle(refresh_example_graph=False) -> MultiDiGraph:
    # Option to refresh to example osm graph.
    if refresh_example_graph:
        import geopandas as gpd
        from utility_route_planner.models.multilayer_network.osm_graph_downloader import OSMGraphDownloader

        project_area = (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )
        osm_graph_downloader = OSMGraphDownloader(project_area)
        project_area_graph = osm_graph_downloader.download_graph()

        with open(Config.PYTEST_OSM_GRAPH_PICKLE, "wb") as file:
            pickle.dump(project_area_graph, file)

    with open(Config.PYTEST_OSM_GRAPH_PICKLE, "rb") as file:
        osm_graph = pickle.load(file)
    return osm_graph
