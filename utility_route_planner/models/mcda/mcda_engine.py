# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from functools import cached_property

import shapely

from utility_route_planner.models.mcda.mcda_datastructures import McdaRasterSettings, RasterizedCriterion
from utility_route_planner.models.mcda.mcda_utils import create_project_area_grid
from utility_route_planner.models.mcda.vrt_builder import VRTBuilder
from settings import Config
from utility_route_planner.util.geo_utilities import get_empty_geodataframe
from utility_route_planner.models.mcda.load_mcda_preset import RasterPreset, load_preset
import structlog
import geopandas as gpd

from utility_route_planner.models.mcda.mcda_rasterizing import (
    rasterize_vector_data,
    get_raster_settings,
    merge_criteria_rasters,
    write_raster_block,
    clip_raster_mask_to_project_area,
)
from utility_route_planner.util.timer import time_function

logger = structlog.get_logger(__name__)


class McdaCostSurfaceEngine:
    raster_preset: RasterPreset
    path_geopackage_input: pathlib.Path

    def __init__(
        self,
        preset_to_load: str,
        path_geopackage_mcda_input: pathlib.Path,
        project_area_geometry: shapely.Polygon,
        raster_name_prefix: str = "",
    ):
        self.raster_preset = load_preset(preset_to_load, path_geopackage_mcda_input, project_area_geometry)
        self.processed_vectors: dict = {}
        self.unprocessed_criteria_names: set = set()
        self.processed_criteria_names: set = set()
        self.raster_name_prefix: str = raster_name_prefix
        self.project_area_geometry = project_area_geometry
        self.project_area_grid = get_empty_geodataframe()

    @cached_property
    def number_of_criteria(self):
        return len(self.raster_preset.criteria)

    @cached_property
    def number_of_criteria_to_rasterize(self):
        return len(self.processed_vectors)

    @time_function
    def preprocess_vectors(self):
        logger.info(
            f"Processing {self.number_of_criteria} criteria using geopackage: {self.raster_preset.general.path_input_geopackage}"
        )
        for idx, criterion in enumerate(self.raster_preset.criteria):
            logger.info(f"Processing criteria number {idx + 1} of {self.number_of_criteria}.")
            is_processed, processed_gdf = self.raster_preset.criteria[criterion].preprocessing_function.execute(
                self.raster_preset.general, self.raster_preset.criteria[criterion]
            )
            if is_processed:
                self.processed_vectors[criterion] = processed_gdf
            else:
                assert processed_gdf.empty
                self.unprocessed_criteria_names.add(criterion)

        self.processed_criteria_names = set(self.raster_preset.criteria.keys()).difference(
            set(self.unprocessed_criteria_names)
        )

    @time_function
    def preprocess_rasters(
        self,
        vector_to_convert: dict[str, gpd.GeoDataFrame],
        cell_size: float,
        max_block_size: int,
        run_in_parallel: bool,
    ) -> str:
        logger.info(f"Starting rasterizing for {self.number_of_criteria_to_rasterize} criteria.")
        min_x, min_y, max_x, max_y = self.project_area_geometry.bounds
        self.project_area_grid = create_project_area_grid(min_x, min_y, max_x, max_y, max_block_size)
        self.assign_vector_groups_to_grid()
        block_ids = list(self.project_area_grid.index)

        logger.info(f"Rasterizing vector using {len(block_ids)} blocks")
        if run_in_parallel:
            rasters = self.compute_raster_blocks_in_parallel(block_ids, vector_to_convert, cell_size)
        else:
            rasters = self.compute_raster_blocks_sequentially(block_ids, vector_to_convert, cell_size)

        block_paths, block_bboxes = zip(*rasters)
        vrt_path = Config.PATH_RESULTS / f"{self.raster_name_prefix}{self.raster_preset.general.final_raster_name}.vrt"
        raster_settings = get_raster_settings(self.project_area_geometry)

        vrt_builder = VRTBuilder(
            block_files=block_paths,
            block_bboxes=block_bboxes,
            crs=raster_settings.crs,
            resolution=Config.RASTER_CELL_SIZE,
            vrt_path=vrt_path,
        )
        vrt_builder.build_and_write_to_disk()
        return str(vrt_path)

    def compute_raster_blocks_sequentially(
        self,
        block_ids: list[int],
        vector_to_convert: dict[str, gpd.GeoDataFrame],
        cell_size: float = Config.RASTER_CELL_SIZE,
    ) -> list[tuple[str, list[float]]]:
        rasters = [self.compute_and_write_raster(block_id, cell_size, vector_to_convert) for block_id in block_ids]
        return rasters

    def compute_raster_blocks_in_parallel(
        self,
        block_ids: list[int],
        vector_to_convert: dict[str, gpd.GeoDataFrame],
        cell_size: float = Config.RASTER_CELL_SIZE,
    ) -> list[tuple[str, list[float]]]:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.compute_and_write_raster, block_id, cell_size, vector_to_convert)
                for block_id in block_ids
            ]
            rasters = [future.result() for future in as_completed(futures)]
        return rasters

    def assign_vector_groups_to_grid(self):
        """
        For each processed vector, assign the vector to the intersecting project area grid block based on intersection.
        """
        for processed_group_name, vector in self.processed_vectors.items():
            vector_with_grid = gpd.sjoin(vector, self.project_area_grid, how="left", predicate="intersects")
            vector_with_grid = vector_with_grid.rename(columns={"index_right": "block_id"})
            vector_with_grid = vector_with_grid.set_index("block_id", drop=True)
            self.processed_vectors[processed_group_name] = vector_with_grid

    def compute_and_write_raster(
        self, block_id: int, cell_size: float, vector_to_convert: dict[str, gpd.GeoDataFrame]
    ) -> tuple[str, list[float]]:
        block_geometry = self.project_area_grid.iloc[block_id].values[0]
        raster_settings = get_raster_settings(block_geometry, cell_size)
        rasters_to_sum = [
            self.rasterize_vector(idx, criterion, gdf, raster_settings)
            for idx, (criterion, gdf) in enumerate(vector_to_convert.items())
        ]

        complete_raster = merge_criteria_rasters(rasters_to_sum, raster_settings.height, raster_settings.width)
        complete_raster = clip_raster_mask_to_project_area(
            complete_raster, self.project_area_geometry, raster_settings.transform
        )

        return write_raster_block(
            complete_raster,
            raster_settings,
            f"{self.raster_name_prefix}{self.raster_preset.general.final_raster_name}-{block_id}",
        )

    def rasterize_vector(
        self, idx: int, criterion: str, gdf: gpd.GeoDataFrame, raster_settings: McdaRasterSettings
    ) -> RasterizedCriterion:
        logger.info(f"Processing criteria number {idx + 1} of {self.number_of_criteria_to_rasterize}.")
        rasterized_vector = rasterize_vector_data(criterion, gdf, raster_settings)
        raster_criteria = self.raster_preset.criteria[criterion]
        return RasterizedCriterion(criterion, rasterized_vector, raster_criteria.group)
