# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.windows import Window

from settings import Config


@dataclass
class McdaRasterSettings:
    width: int
    height: int
    nodata: int
    transform: Affine
    driver: str = "GTiff"
    compress: str = "lzw"
    tiled: bool = True
    dtype: str = "int8"
    count: int = 1
    crs: CRS = CRS.from_epsg(code=Config.CRS)


@dataclass
class RasterizedCriterion:
    criterion: str
    raster: np.ndarray
    group: str


@dataclass
class RasterBlock:
    array: np.ma.MaskedArray
    window: Window
