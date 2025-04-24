# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import osmnx
import pytest

from utility_route_planner.models.multilayer_network.reachability_design import get_network_for_project_area
import geopandas
from settings import Config


@pytest.mark.skip(reason="Part of RQ2.")
def test_get_piece_of_network_small_and_get_two_paths():
    project_area_graph = get_network_for_project_area(
        geopandas.read_file(Config.PATH_PROJECT_AREA_APELDOORN_ROAD_CROSSING), 50
    )
    osmnx.save_graph_geopackage(project_area_graph, str(Config.PATH_RESULTS / "mygraph.gpkg"))

    # Plot the two shortest paths
    shortest_paths = list(osmnx.routing.k_shortest_paths(project_area_graph, 45606512, 45603937, 2))
    for idx, path in enumerate(shortest_paths):
        subgraph = osmnx.utils_graph.route_to_gdf(project_area_graph, path)
        subgraph.to_file(str(Config.PATH_RESULTS / f"k_shortest_path_n_{idx}.geojson"))
