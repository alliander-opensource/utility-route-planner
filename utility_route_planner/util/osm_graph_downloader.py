# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from shapely.geometry import MultiPolygon, Polygon
import structlog
from settings import Config
import geopandas as gpd
import networkx as nx
import osmnx as ox
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from utility_route_planner.models.multilayer_network.exceptions import NoGraphDataForProjectArea

logger = structlog.get_logger(__name__)


class OSMGraphDownloader:
    def __init__(self, project_area_geometry: Polygon | MultiPolygon, max_cable_length: int):
        self.project_area_geometry = project_area_geometry
        self.max_cable_length = max_cable_length

        ox.settings.log_console = False
        ox.settings.overpass_rate_limit = True
        ox.settings.requests_timeout = Config.OSM_API_TIMEOUT_IN_SECONDS
        ox.settings.use_cache = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=16),
        retry=retry_if_not_exception_type(NoGraphDataForProjectArea),
    )
    def download_graph(self) -> nx.MultiGraph:
        logger.info("Start downloading graph from OSM")
        buffered_project_area = self.project_area_geometry.buffer(self.max_cable_length)
        reprojected_project_area = gpd.GeoSeries(buffered_project_area, crs=Config.CRS).to_crs(4326).iloc[0]

        try:
            graph_wgs84 = ox.graph_from_polygon(reprojected_project_area, network_type="all", simplify=False)
        except ValueError as e:
            logger.error("Error while downloading graph", error=str(e))
            raise NoGraphDataForProjectArea("No graph could be created for the project area.")

        graph_rdnew = ox.project_graph(graph_wgs84, to_crs=Config.CRS)
        logger.info("Successfully downloaded graph")
        return graph_rdnew
