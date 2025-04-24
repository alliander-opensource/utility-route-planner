# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import math

import geopandas as gpd
import numpy as np
from shapely.geometry.geo import box

from settings import Config


def create_project_area_grid(min_x: float, min_y: float, max_x: float, max_y: float, max_block_size: int):
    """
    Creates a grid given the bounding box of the project area. The block size is computed based on the max block
    size and the project area bounding box. In case width or height % max_block_size != 0, the block size is decreased
    such that each block is equally sized and the remainder on the project area boundaries is minimized.
    """
    project_area_width = max_x - min_x
    project_area_heigth = max_y - min_y

    number_blocks_width = math.ceil(project_area_width / max_block_size)
    block_width = math.ceil(project_area_width / number_blocks_width)

    number_blocks_height = math.ceil(project_area_heigth / max_block_size)
    block_height = math.ceil(project_area_heigth / number_blocks_height)

    x_coords = np.arange(min_x, max_x, block_width)
    y_coords = np.arange(min_y, max_y, block_height)
    grid_cells = [box(x, y, x + block_width, y + block_height) for x in x_coords for y in y_coords]
    grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=Config.CRS)
    return grid
