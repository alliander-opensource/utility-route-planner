# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import geopandas as gpd
import osmnx as ox

from settings import Config


def get_network_for_project_area(project_area: gpd.GeoDataFrame, cable_length):
    xmin, ymin, xmax, ymax = project_area.buffer(cable_length * 2).to_crs("4326").total_bounds
    graph_wgs84 = ox.graph_from_bbox(north=ymax, south=ymin, east=xmax, west=xmin, network_type="all", simplify=False)
    graph_rdnew = ox.project_graph(graph_wgs84, Config.CRS)
    return graph_rdnew
