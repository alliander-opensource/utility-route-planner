# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import asdict

import affine
import shapely
import structlog
import rasterio
import rasterio.merge
import rasterio.mask
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize, geometry_mask

from utility_route_planner.models.mcda.mcda_datastructures import McdaRasterSettings, RasterizedCriterion
from settings import Config
from utility_route_planner.models.mcda.exceptions import (
    InvalidGroupValue,
    InvalidSuitabilityRasterInput,
    RasterCellSizeTooSmall,
)

logger = structlog.get_logger(__name__)


def get_raster_settings(
    project_area: shapely.MultiPolygon | shapely.Polygon, cell_size: float = Config.RASTER_CELL_SIZE
) -> McdaRasterSettings:
    minx, miny, maxx, maxy = project_area.bounds

    # In order to fit the given cell size to the project area bounds, we slightly extend the maxx and maxy accordingly.
    if cell_size > maxx - minx or cell_size > maxy - miny:
        raise RasterCellSizeTooSmall("Given raster cell size is too large for the project area.")

    raster_settings = McdaRasterSettings(
        width=math.ceil((maxx - minx) / cell_size),
        height=math.ceil((maxy - miny) / cell_size),
        nodata=Config.INTERMEDIATE_RASTER_NO_DATA,
        transform=affine.Affine(cell_size, 0.0, round(minx), 0.0, -cell_size, round(maxy)),
    )
    return raster_settings


def rasterize_vector_data(
    criterion: str,
    gdf_to_rasterize: gpd.GeoDataFrame,
    raster_settings: McdaRasterSettings,
) -> np.ndarray:
    """
    Burns the vector data to the project area in the desired raster cell size.
    If values overlap in the geodataframe, pick the highest value.
    """
    logger.debug(f"Rasterizing layer: {criterion} in cell size: {Config.RASTER_CELL_SIZE} meters")
    # Highest value is leading within a criteria, using sorting we create the reverse painters algorithm effect.
    gdf_to_rasterize.sort_values("suitability_value", ascending=True, inplace=True)
    # Bump values which would be no-data prior to rasterizing to avoid marking them as no-data unwanted.
    gdf_to_rasterize.suitability_value = gdf_to_rasterize.suitability_value.replace(
        raster_settings.nodata, raster_settings.nodata + 1
    )
    # Reset values exceeding the min/max.
    gdf_to_rasterize.loc[
        gdf_to_rasterize.suitability_value < Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER, "suitability_value"
    ] = Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER
    gdf_to_rasterize.loc[
        gdf_to_rasterize.suitability_value > Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER, "suitability_value"
    ] = Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER

    out_array = np.full(
        (raster_settings.height, raster_settings.width), Config.INTERMEDIATE_RASTER_NO_DATA, dtype="int16"
    )
    shapes = ((geom, value) for geom, value in zip(gdf_to_rasterize.geometry, gdf_to_rasterize.suitability_value))
    rasterized_vector = rasterize(shapes=shapes, out=out_array, transform=raster_settings.transform, all_touched=False)

    return rasterized_vector


def merge_criteria_rasters(
    rasters_to_process: list[RasterizedCriterion],
    raster_height: int,
    raster_width: int,
) -> np.ma.MaskedArray:
    """
    List of rasters to combine and their respective group.

    Criteria in group a: highest value in group a is leading.
    Criteria in group b: values in group b are added or subtracted to group a if present.
    Criteria in group c: mark as no data if present, overruling group a and b.
    """
    logger.debug(f"Starting summing {len(rasters_to_process)} rasters into the final cost surface.")

    # Split groups and process accordingly prior to summing all together.
    group_a, group_b, group_c = [], [], []
    for rasterized_vector in rasters_to_process:
        match rasterized_vector.group:
            case "a":
                group_a.append(rasterized_vector)
            case "b":
                group_b.append(rasterized_vector)
            case "c":
                group_c.append(rasterized_vector)
            case _:
                raise InvalidGroupValue(
                    f"Invalid group value encountered during raster processing: {rasterized_vector.group}"
                )

    merged_group_a = np.ma.empty(shape=(raster_height, raster_width))
    merged_group_b = np.ma.empty(shape=(raster_height, raster_width))
    merged_group_c = np.ma.empty(shape=(raster_height, raster_width))
    if len(group_a) > 0:
        merged_group_a = process_raster_groups(group_a, "max", raster_height, raster_width)
    if len(group_b) > 0:
        merged_group_b = process_raster_groups(group_b, "sum", raster_height, raster_width)
    if len(group_c) > 0:
        merged_group_c = process_raster_groups(group_c, "sum", raster_height, raster_width)

    if len(group_b) > 0 and len(group_a) > 0:
        summed_raster = np.ma.sum([merged_group_a, merged_group_b], axis=0)

    elif len(group_b) > 0 and len(group_a) == 0:
        summed_raster = merged_group_b
    elif len(group_a) > 0 and len(group_b) == 0:
        summed_raster = merged_group_a
    else:
        raise InvalidSuitabilityRasterInput("No rasters to sum, exiting.")

    # Force values to fit in the int8 datatype
    summed_raster = np.ma.clip(
        summed_raster, Config.FINAL_RASTER_VALUE_LIMIT_LOWER, Config.FINAL_RASTER_VALUE_LIMIT_UPPER
    )

    # Update the mask of the summed_raster so that every cell intersecting with group c is set to no data.
    if len(group_c) > 0:
        summed_raster.mask = np.ma.mask_or(summed_raster.mask, ~merged_group_c.mask)

    return summed_raster


def clip_raster_mask_to_project_area(
    raster: np.ma.MaskedArray, project_area: shapely.Polygon, transform: affine.Affine
):
    """
    Update the raster mask such that all values outside the project area are masked and set to no data later on
    """
    project_area_mask = geometry_mask([project_area], transform=transform, invert=True, out_shape=raster.shape)
    raster.mask = np.ma.mask_or(raster.mask, ~project_area_mask)
    return raster


def process_raster_groups(
    group: list[RasterizedCriterion],
    method: str,
    height: int,
    width: int,
) -> np.ma.MaskedArray:
    """
    Iterate over all raster groups and perform the desired method to combine the different groups.
    :param group: list of criterion groups to process.
    :param method: mathematical operation to perform on the rasters. Currently, "sum" and "max" are supported.
    :param height: height of the raster
    :param width: width of the raster

    :return: masked numpy array of processed raster groups
    """
    # Use numpy masks to ignore the nodata values in the computations.
    stacked_raster_groups = np.ma.empty(shape=(len(group), height, width))
    for idx, criterion in enumerate(group):
        stacked_raster_groups[idx] = np.ma.masked_equal(criterion.raster, Config.INTERMEDIATE_RASTER_NO_DATA)

    match method:
        case "sum":
            processed_raster = np.ma.sum(stacked_raster_groups, axis=0)
        case "max":
            processed_raster = np.ma.max(stacked_raster_groups, axis=0)
        case _:
            raise InvalidSuitabilityRasterInput(
                f"Invalid method for processing raster group: {method}. Expected 'sum' or 'max'."
            )
    return processed_raster


def write_raster_block(
    complete_raster: np.ma.MaskedArray, raster_settings: McdaRasterSettings, final_raster_name
) -> tuple[str, list[float]]:
    raster_settings.nodata = Config.FINAL_RASTER_NO_DATA
    final_raster_path = Config.PATH_RESULTS / (final_raster_name + ".tif")
    with rasterio.open(final_raster_path, "w", **asdict(raster_settings)) as dest:
        dest.write(np.ma.filled(complete_raster, Config.FINAL_RASTER_NO_DATA), 1)
        bbox = list(dest.bounds)

    return final_raster_path.__str__(), bbox
