# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Generator
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely
import structlog

from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import get_hexagon_width_and_height
from settings import Config
from utility_route_planner.util.timer import time_function

logger = structlog.get_logger(__name__)


class HexagonGridBuilder:
    """
    Class that is used to build a spatial grid with a hexagonal structure given a set of preprocessed vectors and
    raster preset. Each point has as assigned suitabililty value that is used to construct a spatial graph in the next
    step.
    """

    def __init__(
        self,
        raster_groups: dict[str, str],
        preprocessed_vectors: dict[str, gpd.GeoDataFrame],
        hexagon_size: float,
        block_size: int,
    ):
        self.raster_groups = raster_groups
        self.preprocessed_vectors = preprocessed_vectors
        self.hexagon_size = hexagon_size
        self.hexagon_width, self.hexagon_height = get_hexagon_width_and_height(hexagon_size)
        self.block_size = block_size

    def construct_grid(self, project_area: shapely.Polygon) -> Generator[gpd.GeoDataFrame, None, None]:
        x_matrix, y_matrix = self.construct_hexagonal_grid_for_bounding_box(project_area)

        for block in self.divide_matrices_into_blocks(x_matrix, y_matrix):
            hexagonal_grid_for_block = self.filter_block_to_project_area(block)

            # A block can be empty in case it does not intersect with any vector
            if not hexagonal_grid_for_block.empty:
                weighted_hexagonal_block = self.assign_suitability_values_to_block(hexagonal_grid_for_block)
                weighted_hexagonal_block["axial_q"], weighted_hexagonal_block["axial_r"] = (
                    self.convert_cartesian_coordinates_to_axial(weighted_hexagonal_block)
                )
                weighted_hexagonal_block = gpd.GeoDataFrame(
                    pd.concat([weighted_hexagonal_block, weighted_hexagonal_block.get_coordinates()], axis=1),
                    geometry="geometry",
                )

                # Reset index of grid to align with node ids generated using rustworkx
                weighted_hexagonal_block = weighted_hexagonal_block.reset_index(drop=True)
                yield weighted_hexagonal_block

    @time_function
    def construct_hexagonal_grid_for_bounding_box(self, project_area: shapely.Polygon) -> gpd.GeoDataFrame:
        """
        Given the bounding box of the project area, create a hexagonal grid in flat-top orientation.

        :return: GeoDataFrame where each point represents a location on the grid
        """
        x_min, y_min, x_max, y_max = project_area.bounds

        # 0.75 is used to correctly set the offset of the x coordinate of the center, as each hexagon is partially covered
        # by the surrounding tiles
        x_coordinates = np.arange(x_min, x_max, self.hexagon_width * 0.75)
        y_coordinates = np.arange(y_min, y_max, self.hexagon_height)
        x_matrix, y_matrix = np.meshgrid(x_coordinates, y_coordinates)

        # Reverse order of matrices, as the coordinate system should be ordered using decreasing coordinates instead
        # of increasing coordinates which is the numpy default
        x_matrix = np.flip(x_matrix)
        y_matrix = np.flip(y_matrix)

        # Every even column must be offset by half of the hexagon height to properly determine the vertical
        # position of the hexagon.
        y_matrix[:, ::2] += self.hexagon_height / 2

        return x_matrix, y_matrix

    def divide_matrices_into_blocks(
        self, x_matrix: np.ndarray, y_matrix: np.ndarray
    ) -> Generator[gpd.GeoDataFrame, None, None]:
        """
        Generator which yields indexed blocks from the x- and y-matrix given the desired block size

        :param x_matrix: x_matrix to divide into blocks
        :type x_matrix: np.ndarray
        :param y_matrix: y_matrix to divide into blocks
        :type y_matrix: np.ndarray
        :return: row_start, row_end, column_start, column_end, x_block, y_block
        :rtype: Generator[gpd.GeoDataFrame, None, None]
        """
        # Determine number of columns and columns given the desired block size. Round up to prevent
        # losing data
        n_rows_blocks = math.ceil(x_matrix.shape[0] / self.block_size)
        n_columns_blocks = math.ceil(y_matrix.shape[1] / self.block_size)
        logger.info(f"Total number of blocks: {n_rows_blocks * n_columns_blocks}")

        if n_rows_blocks == 1 and n_columns_blocks == 1:
            grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_matrix.ravel(), y_matrix.ravel()), crs=Config.CRS)
            grid = grid.reset_index(names="node_id")
            yield grid
            return

        row_splits = np.linspace(0, x_matrix.shape[0], n_rows_blocks + 1, dtype=int)
        column_splits = np.linspace(0, y_matrix.shape[1], n_columns_blocks + 1, dtype=int)

        # Iterate over the split indexes to extract the blocks from the matrices. Convert each block
        # to a GeoDataFrame.
        for row_start, row_end in zip(row_splits[:-1], row_splits[1:]):
            for column_start, column_end in zip(column_splits[:-1], column_splits[1:]):
                x_block = x_matrix[row_start:row_end, column_start:column_end]
                y_block = y_matrix[row_start:row_end, column_start:column_end]

                block_grid = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(x_block.ravel(), y_block.ravel()), crs=Config.CRS
                )
                block_grid = block_grid.reset_index(names="node_id")

                yield block_grid

    def filter_block_to_project_area(self, bounding_box_grid: gpd.GeoDataFrame):
        """
        Concatenate all preprocessed vectors into a single geodataframe. Use this concatenated dataframe
        filter all points from the bounding box that do not intersect with any of the vectors.
        """
        for criterion, vector_gdf in self.preprocessed_vectors.items():
            vector_gdf["criterion"] = criterion
            vector_gdf["group"] = self.raster_groups[criterion]
        concatenated_vectors = gpd.GeoDataFrame(pd.concat(self.preprocessed_vectors.values()), crs=Config.CRS)

        points_within_project_area = gpd.sjoin(
            bounding_box_grid,
            concatenated_vectors[["group", "suitability_value", "geometry"]],
            predicate="within",
            how="inner",
        ).set_index("node_id")

        return points_within_project_area

    @time_function
    def assign_suitability_values_to_block(self, points_within_project_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Given the group the vector of a suitability value belongs to, a specific aggregation functions is applied for overlapping
        points within this group:
        - group a: take max suitability value of overlapping points
        - group b: sum overlapping suitability values
        - group c: sum overlapping suitability values

        In case points that intersect with group a and b are overlapping, they are summed after aggregation. Finally, all points
        that intersect with group c are set to the max possible suitability value.

        :return: GeoDataFrame containing all points within the project area in combination with aggregated suitability
        values for every point.
        """

        polars_df = pl.from_pandas(points_within_project_area.loc[:, ["group", "suitability_value"]].reset_index())

        # Aggregate dataframe with overlapping values for the same node id. Max for group a, sum for
        # group b and c.
        aggregated_suit_values_per_group_and_node = (
            polars_df.lazy()
            .group_by("group", "node_id")
            .agg(
                pl.when(pl.first("group") == "a")
                .then(pl.col("suitability_value").max())
                .otherwise(pl.col("suitability_value").sum())
                .alias("agg_suit_val"),
            )
        ).collect()

        aggregated_suit_values_per_node = (
            aggregated_suit_values_per_group_and_node.lazy()
            .group_by("node_id")
            .agg(
                has_a=(pl.col("group") == "a").any(),
                has_b=(pl.col("group") == "b").any(),
                has_c=(pl.col("group") == "c").any(),
                summed=pl.col("agg_suit_val").sum(),
                first_val=pl.col("agg_suit_val").first(),
            )
            .with_columns(
                # In case a node is in both group a and b but not c, use the summed total as suitability value
                pl.when(pl.col("has_a") & pl.col("has_b") & ~pl.col("has_c"))
                .then(pl.col("summed"))
                # When a group is in node c, use the max node suitability value
                .when(pl.col("has_c"))
                .then(Config.MAX_NODE_SUITABILITY_VALUE)
                # In all other cases there is no overlap of groups, simply pick the first value (there is always only
                # 1) as suitability value
                .otherwise(pl.col("first_val"))
                .alias("suitability_value")
            )
            .select("node_id", "suitability_value")
        ).collect()

        # Make sure all suitability values are within boundaries
        aggregated_suit_values_per_node["suitability_value"] = aggregated_suit_values_per_node[
            "suitability_value"
        ].clip(Config.MIN_NODE_SUITABILITY_VALUE, Config.MAX_NODE_SUITABILITY_VALUE)

        # Join location afterwards, as this is faster than picking the first one within the aggregation step
        hexagon_points = gpd.GeoDataFrame(
            aggregated_suit_values_per_node.join(
                points_within_project_area["geometry"], how="left", lsuffix="l", rsuffix="r"
            ),
            geometry="geometry",
        )
        # Remove duplicate points, as a point could have joined multiple vector which results in duplicate rows within
        # the right dataframe.
        hexagon_points = hexagon_points[~hexagon_points.index.duplicated()]

        return hexagon_points

    def convert_cartesian_coordinates_to_axial(
        self, hexagon_center_points: gpd.GeoDataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        To efficiently determine neighbours to construct a hexagonal graph later on, convert all cartesian coordinates
        to axial coordinates.

        Used algorithms as provided by:
        - coordinate to hex: https://www.redblobgames.com/grids/hexagons/#pixel-to-hex
        - rounding hex correctly: https://observablehq.com/@jrus/hexround (via redblobgames)

        :return: tuple containing q- and r-values as integers in numpy ndarray format
        """
        x, y = np.split(hexagon_center_points.get_coordinates().values, 2, axis=1)

        # Convert x- and y-coordinates to axial
        q = (-2 / 3 * x) / self.hexagon_size
        r = (1 / 3 * x + np.sqrt(3) / 3 * y) / self.hexagon_size

        # Convert coordinates to integers and correct rounding errors
        xgrid = np.round(q).astype(np.int32)
        ygrid = np.round(r).astype(np.int32)

        q_diff = q - xgrid
        r_diff = r - ygrid

        mask = np.abs(q_diff) > np.abs(r_diff)
        xgrid[mask] = xgrid[mask] + np.round(q_diff[mask] + 0.5 * r_diff[mask])
        ygrid[~mask] = ygrid[~mask] + np.round(r_diff[~mask] + 0.5 * q_diff[~mask])

        return xgrid, ygrid
