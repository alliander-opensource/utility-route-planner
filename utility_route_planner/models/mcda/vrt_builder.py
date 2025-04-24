# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

# Based on the script generously provided by 12rambau on https://github.com/12rambau/rio-vrt
from os.path import relpath
from pathlib import Path
import xml.etree.cElementTree as et
from xml.dom import minidom

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import ColorInterp


class VRTBuilder:
    def __init__(
        self,
        block_files: tuple[str],
        block_bboxes: tuple[float],
        crs: CRS,
        resolution,
        vrt_path: Path,
    ):
        self.block_files = block_files
        self.crs = crs
        self.resolution = resolution
        self.vrt_path = vrt_path
        self.xml_datatype = "Int8"
        self.min_x, self.min_y, self.max_x, self.max_y = self.get_raster_extends(block_bboxes)

    @staticmethod
    def get_raster_extends(block_bboxes: tuple[float]) -> tuple[float, float, float, float]:
        block_bboxes_matrix = np.array(block_bboxes)
        min_x = block_bboxes_matrix[:, 0].min()
        min_y = block_bboxes_matrix[:, 1].min()
        max_x = block_bboxes_matrix[:, 2].max()
        max_y = block_bboxes_matrix[:, 3].max()

        return min_x, min_y, max_x, max_y

    def build_and_write_to_disk(self):
        vrt_tree, raster_band = self.setup_tree()
        self.add_blocks_to_band(vrt_band=raster_band)

        self.vrt_path.resolve().write_text(
            minidom.parseString(et.tostring(vrt_tree).decode("utf-8")).toprettyxml(indent="  ").replace("&quot;", '"')
        )

    def setup_tree(self) -> tuple[et.Element, et.Element]:
        # Construct the transformation based on the bounding box of the grid
        transform = rasterio.Affine.from_gdal(self.min_x, self.resolution, 0, self.max_y, 0, -self.resolution)
        total_width = round((self.max_x - self.min_x) / self.resolution)
        total_height = round((self.max_y - self.min_y) / self.resolution)

        # Initialize the VRT tree
        vrt_tree = et.Element("VRTDataset", {"rasterXSize": str(total_width), "rasterYSize": str(total_height)})
        et.SubElement(vrt_tree, "SRS").text = self.crs.to_wkt()

        transform_as_string = ", ".join([str(i) for i in transform.to_gdal()])
        et.SubElement(vrt_tree, "GeoTransform").text = transform_as_string
        et.SubElement(vrt_tree, "OverviewList", {"resampling": "nearest"}).text = "2 4 8"

        # Initialize the band on which all blocks will be added
        vrt_band = et.SubElement(vrt_tree, "VRTRasterBand", {"dataType": "Int8", "band": "1"})
        et.SubElement(vrt_band, "Offset").text = "0.0"
        et.SubElement(vrt_band, "Scale").text = "1.0"
        et.SubElement(vrt_band, "ColorInterp").text = ColorInterp.gray.name.capitalize()
        et.SubElement(vrt_band, "NoDataValue").text = "0.0"

        return vrt_tree, vrt_band

    def add_blocks_to_band(self, vrt_band: et.Element):
        relative_to_vrt = "1"
        for f in self.block_files:
            with rasterio.open(f) as src:
                source = et.SubElement(vrt_band, "ComplexSource")
                transform_as_string = relpath(f, self.vrt_path.parent)
                et.SubElement(source, "SourceFilename", {"relativeToVRT": relative_to_vrt}).text = transform_as_string
                et.SubElement(source, "SourceBand").text = "1"

                self.add_source_content(
                    source=source,
                    src=src,
                    x_off=str(abs(round((src.bounds.left - self.min_x) / self.resolution))),
                    y_off=str(abs(round((src.bounds.top - self.max_y) / self.resolution))),
                )

                et.SubElement(source, "NODATA").text = "0.0"

    def add_source_content(self, source: et.Element, src: rasterio.DatasetReader, x_off: str, y_off: str):
        """
        Given a tiff file, add its properties to the source element of the raster band row

        :param source: source element for this raster block
        :param src: reader instance which holds the tiff of the current raster block
        :param x_off: x offset wrt the total raster size
        :param y_off: y offset wrt the total raster size
        """
        width, height = str(src.width), str(src.height)
        block_x = str(src.profile.get("blockxsize", ""))
        block_y = str(src.profile.get("blockysize", ""))

        et.SubElement(
            source,
            "SourceProperties",
            {
                "RasterXSize": width,
                "RasterYSize": height,
                "DataType": self.xml_datatype,
                "BlockXSize": block_x,
                "BlockYSize": block_y,
            },
        )

        et.SubElement(source, "SrcRect", {"xOff": "0", "yOff": "0", "xSize": width, "ySize": height})
        et.SubElement(source, "DstRect", {"xOff": x_off, "yOff": y_off, "xSize": width, "ySize": height})
