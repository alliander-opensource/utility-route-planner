# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib

import geopandas
import geopandas as gpd
import shapely
import structlog
import os
import fiona

from settings import Config

logger = structlog.get_logger(__name__)


def write_to_file(geometry: gpd.GeoSeries | gpd.GeoDataFrame | shapely.Geometry, name: str):
    if isinstance(geometry, shapely.Geometry):
        geometry = gpd.GeoSeries(geometry, crs=28992)

    if isinstance(geometry, gpd.GeoSeries | gpd.GeoDataFrame):
        pass

    geometry.to_file(Config.PATH_RESULTS / name)


def reset_geopackage(path_geopackage: pathlib.Path, truncate=True) -> None:
    """
    Clean start, delete or truncate result geopackage to write to.
    """
    logger.info(f"Resetting geopackage {path_geopackage} for a clean start.")
    if os.path.exists(path_geopackage):
        if truncate:
            existing_layers = [layer_name for layer_name in fiona.listlayers(path_geopackage)]
            for layer_name in existing_layers:
                gdf = gpd.read_file(path_geopackage, layer=layer_name)
                if gdf.empty:
                    return
                gdf = gdf.iloc[0:0]
                gdf.to_file(path_geopackage, layer=layer_name, driver="GPKG")
        else:
            os.remove(path_geopackage)
    else:
        logger.info("Geopackage does not exists, continuing as normal.")


def write_results_to_geopackage(
    path_geopackage: pathlib.Path,
    item_to_write: shapely.Geometry | gpd.GeoDataFrame | gpd.GeoSeries,
    layer_name: str,
    overwrite=False,
) -> None:
    """
    Write results to a geopackage file which is handy for debugging in QGIS and intermediate storage.

    :param path_geopackage: path to the geopackage as pathlib.Path.
    :param item_to_write: feature to write to the geopackage.
    :param layer_name: name of the feature in the geopackage.
    :param overwrite: force overwriting the layer in the geopackage if it exists.
    """
    logger.info(f"Writing features to geopackage: {layer_name}")
    if isinstance(item_to_write, shapely.Geometry):
        item_to_write = geopandas.GeoSeries(item_to_write, crs=Config.CRS)
    if overwrite:
        item_to_write.to_file(path_geopackage, layer=layer_name, driver="GPKG", OVERWRITE="YES")
    else:
        mode = _get_writing_mode_geopackage(layer_name, path_geopackage)
        item_to_write.to_file(path_geopackage, layer=layer_name, driver="GPKG", mode=mode)


def _get_writing_mode_geopackage(filename, path_geopackage):
    """
    Function for determining the mode used for writing the results to file, specific for geopackage.
    - We want to create a new geopackage if it does not exist -> w
    - If the layer in the geopackage exists, append to it -> a
    - If the layer in the geopackage does not exist, create it -> w
    """
    # if geopackage does not exist, create a new one.
    if not os.path.exists(path_geopackage):
        mode = "w"
    else:
        # if layer does not exist, create a new one. Otherwise, append.
        existing_layers = [layer_name for layer_name in fiona.listlayers(path_geopackage)]
        if filename in existing_layers:
            mode = "a"
        else:
            mode = "w"
    return mode
