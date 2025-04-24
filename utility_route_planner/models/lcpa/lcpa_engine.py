# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import structlog
import numpy as np
import shapely
from skimage.graph import route_through_array

from settings import Config
from utility_route_planner.models.lcpa.lcpa_datastructures import LcpaInputModel
from utility_route_planner.util.geo_utilities import (
    array_indices_to_linestring,
    align_linestring,
    load_suitability_raster_data,
)
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class LcpaUtilityRouteEngine:
    route_model: LcpaInputModel
    lcpa_result: shapely.LineString

    @time_function
    def get_lcpa_route(
        self,
        path_raster: str,
        utility_route_sketch: shapely.LineString,
        project_area: shapely.Polygon = shapely.Polygon(),
    ) -> shapely.LineString:
        # Set a default project area if not provided, this is a bad idea most of the time.
        if shapely.is_empty(project_area):
            project_area = utility_route_sketch.buffer(utility_route_sketch.length / 2)

        # Creates a numpy array from cost surface raster and saves the metadata for further usage.
        raster_array, raster_geotransform = load_suitability_raster_data(path_raster, project_area)
        # Preprocess input linestring geometry to a structured datamodel.
        self.preprocess_input_linestring(raster_geotransform, utility_route_sketch)
        # Creates path array and the respective sequence as numpy array indices.
        cost_path, cost_path_indices = self.calculate_least_cost_path(raster_array, self.route_model)
        # Converts path array to raster and linestring.
        linestring = array_indices_to_linestring(raster_geotransform, cost_path_indices)
        # The linestring is the result of a vectorized raster, which results in a jagged shape. Smooth this.
        linestring_aligned = align_linestring(linestring, Config.RASTER_CELL_SIZE)

        self.lcpa_result = linestring_aligned
        write_results_to_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT, self.lcpa_result, "utility_route_result")

        return self.lcpa_result

    def preprocess_input_linestring(self, geotransform: tuple, utility_route_sketch: shapely.LineString):
        """
        Convert input to a dictionary for further processing and check if we have optional stops. The current input is
        a tuple, this might be changed to a shapely / GeoJSON linestring geometry later on depending on the GUI.

        :param geotransform: metadata of the raster from gdal.
        :param utility_route_sketch: input linestring sketch for which to compute a utility route.
        :return route_model: input converted to a route_model.
        """

        self.route_model = LcpaInputModel(utility_route_sketch, geotransform)
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_LCPA_OUTPUT, self.route_model.input_linestring, "utility_sketch_route"
        )
        write_results_to_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT, self.route_model.route_points, "route_points")

    @staticmethod
    def calculate_least_cost_path(suit_raster_array: np.ndarray, utility_route_model) -> tuple:
        """
        Calculates the least cost path in the given suitability raster. Handle one or multiple stops if present.

        :param suit_raster_array: numpy array containing the values of the suitability raster.
        :param utility_route_model: input as lcpa data structure.
        :return: numpy array containing the least cost path.
        """

        # Check if we have to account for intermediate stops in the path calculations.
        if len(utility_route_model.idx_stops) == 0:
            logger.info("There are no intermediate stops to account for in determining the cable route.")
            indices, weight = route_through_array(
                suit_raster_array,
                utility_route_model.idx_start,
                utility_route_model.idx_end,
                geometric=True,
                fully_connected=True,
            )
        else:
            # Call the route finding function multiple times.
            logger.info(f"There are {len(utility_route_model.idx_stops)} intermediate stop(s) in the utility route.")
            indices = []
            weight = []
            for idx, item in enumerate(utility_route_model.idx_stops):
                # For the first call, we take the start point and the first stop.
                if idx == 0:
                    tmp_indices, tmp_weight = route_through_array(
                        suit_raster_array, utility_route_model.idx_start, item, geometric=True, fully_connected=True
                    )
                # Check if there are more stops to account for. Take the current stop and the previous one.
                else:
                    tmp_indices, tmp_weight = route_through_array(
                        suit_raster_array,
                        utility_route_model.idx_stops[idx - 1],
                        item,
                        geometric=True,
                        fully_connected=True,
                    )
                # Add route part to the complete cable route.
                indices += tmp_indices
                weight += tmp_weight

            # Finally, add the ending route segment. Use the last stop point in combination with the end point.
            tmp_indices, tmp_weight = route_through_array(
                suit_raster_array,
                utility_route_model.idx_stops[-1],
                utility_route_model.idx_end,
                geometric=True,
                fully_connected=True,
            )
            indices += tmp_indices
            weight += tmp_weight

        # Gather all indices and paths, merge them. Create a new array where 1 = cable route, 0 = not cable route.
        indices_np = np.array(indices).T
        path = np.zeros_like(suit_raster_array)
        path[indices_np[0], indices_np[1]] = 1

        return path, indices
