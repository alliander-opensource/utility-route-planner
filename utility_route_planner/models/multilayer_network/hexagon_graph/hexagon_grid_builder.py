# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Generator
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import shapely
import structlog

from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    get_hexagon_width_and_height,
    build_flat_top_hexagons,
)
from settings import Config

logger = structlog.get_logger(__name__)


class HexagonGridBuilder:
    """
    Class that is used to build a spatial grid with a hexagonal structure given a set of preprocessed vectors and
    raster preset. Each point has as assigned suitabililty value that is used to construct a spatial graph in the next
    step.
    """

    def __init__(
        self,
        hexagon_size: float,
        block_size: int,
    ):
        self.hexagon_size = hexagon_size
        self.hexagon_width, self.hexagon_height = get_hexagon_width_and_height(hexagon_size)
        self.block_size = block_size

    def construct_grid_blocks(
        self,
        x_matrix: np.ndarray,
        y_matrix: np.ndarray,
        preprocessed_vectors: dict[str, gpd.GeoDataFrame],
        raster_groups: dict[str, str],
    ) -> Generator[tuple[pl.DataFrame | None, bool], None, None]:
        concatenated_vectors = self.concatenate_preprocessed_vectors(preprocessed_vectors, raster_groups)

        for block, final_column in self.divide_matrices_into_blocks(x_matrix, y_matrix):
            hexagonal_grid_for_block = self.filter_block_to_project_area(block, concatenated_vectors)

            # A block can be empty in case it does not intersect with any vector. We still yield it (as None) so the
            # caller can advance its row/column bookkeeping. Skipping it entirely would drop the final-column signal,
            # leaving the next row unable to connect upward and fragmenting the graph into disconnected bands.
            if hexagonal_grid_for_block.empty:
                yield None, final_column
                continue

            weighted_hexagonal_block = self.assign_suitability_values_to_block(hexagonal_grid_for_block)
            weighted_hexagonal_block = self.convert_cartesian_coordinates_to_axial(weighted_hexagonal_block)

            yield weighted_hexagonal_block, final_column

    def construct_hexagonal_grid_for_bounding_box(self, project_area: shapely.Polygon) -> tuple[np.ndarray, np.ndarray]:
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

        # Reverse order of matrices twice, to make sure y-coordinates are decreasing going south and x-coordinates are
        # increasing going east. This is required since the numpy default initialization does not respect the same order
        # as the coordinate system being used.
        x_matrix = np.flip(x_matrix)
        y_matrix = np.flip(y_matrix)

        # Every even column must be offset by half of the hexagon height to properly determine the vertical
        # position of the hexagon.
        y_matrix[:, ::2] += self.hexagon_height / 2

        return x_matrix, y_matrix

    def divide_matrices_into_blocks(
        self, x_matrix: np.ndarray, y_matrix: np.ndarray
    ) -> Generator[tuple[gpd.GeoDataFrame, bool], None, None]:
        """
        Generator which yields indexed blocks from the x- and y-matrix given the desired block size

        :param x_matrix: x_matrix to divide into blocks
        :type x_matrix: np.ndarray
        :param y_matrix: y_matrix to divide into blocks
        :type y_matrix: np.ndarray
        :return: block and indication whether the final column of the current row is reached
        :rtype: Generator[tuple[gpd.GeoDataFrame, bool], None, None]
        """
        # Determine number of columns and columns given the desired block size. Round up to prevent
        # losing data
        n_rows_blocks = math.ceil(x_matrix.shape[0] / self.block_size)
        n_columns_blocks = math.ceil(y_matrix.shape[1] / self.block_size)
        total_nr_of_blocks = n_rows_blocks * n_columns_blocks
        logger.info(f"Total number of blocks: {n_rows_blocks * n_columns_blocks}")

        if n_rows_blocks == 1 and n_columns_blocks == 1:
            grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_matrix.ravel(), y_matrix.ravel()), crs=Config.CRS)
            grid = grid.reset_index(names="node_id")
            yield grid, True
            return

        row_splits = np.linspace(0, x_matrix.shape[0], n_rows_blocks + 1, dtype=int)
        column_splits = np.linspace(0, y_matrix.shape[1], n_columns_blocks + 1, dtype=int)

        # Iterate over the split indexes to extract the blocks from the matrices. Each block is yielded as a polars
        # dataframe. In addition, it is yielded whether the final column of the current row is reached. This is required
        # for edge construction later on.
        with tqdm(total=total_nr_of_blocks, desc="Constructing hexagonal suitability grid") as pbar:
            for row_start, row_end in zip(row_splits[:-1], row_splits[1:]):
                for column_start, column_end in zip(column_splits[:-1], column_splits[1:]):
                    x_block = x_matrix[row_start:row_end, column_start:column_end]
                    y_block = y_matrix[row_start:row_end, column_start:column_end]

                    block_grid = gpd.GeoDataFrame(
                        geometry=gpd.points_from_xy(x_block.ravel(), y_block.ravel()), crs=Config.CRS
                    )
                    block_grid = block_grid.reset_index(names="node_id")

                    final_column = column_end == column_splits[-1]
                    pbar.update()
                    yield block_grid, final_column

    @staticmethod
    def concatenate_preprocessed_vectors(
        preprocessed_vectors: dict[str, gpd.GeoDataFrame], raster_groups: dict[str, str]
    ):
        """
        Concatenate all preprocessed vectors into a single geodataframe.
        """
        for criterion, vector_gdf in preprocessed_vectors.items():
            vector_gdf["criterion"] = criterion
            vector_gdf["group"] = raster_groups[criterion]
        concatenated_vectors = gpd.GeoDataFrame(pd.concat(preprocessed_vectors.values()), crs=Config.CRS)

        return concatenated_vectors

    def filter_block_to_project_area(
        self, bounding_box_grid: gpd.GeoDataFrame, concatenated_vectors
    ) -> gpd.GeoDataFrame:
        """
        Create temporarily hexagons from the centroid grid for the spatial join. Filter to project area and discard
        hexagon geometry afterward, keep only centroids.
        """
        hexagons = build_flat_top_hexagons(bounding_box_grid, self.hexagon_size)
        hexagon_grid = gpd.GeoDataFrame(
            {
                "node_id": bounding_box_grid["node_id"].to_numpy(),
                "centroid": bounding_box_grid.geometry.to_numpy(),
            },
            geometry=gpd.GeoSeries(hexagons, index=bounding_box_grid.index, crs=Config.CRS),
            crs=Config.CRS,
        )

        joined = gpd.sjoin(
            hexagon_grid,
            concatenated_vectors[["group", "suitability_value", "geometry"]],
            predicate="intersects",
            how="inner",
        )

        # Discard the hexagon polygons and restore the centroid point as the active geometry.
        points_within_project_area = (
            joined.drop(columns="geometry")
            .rename(columns={"centroid": "geometry"})
            .set_geometry("geometry")
            .set_index("node_id")
        )

        return points_within_project_area

    def assign_suitability_values_to_block(self, points_within_project_area: gpd.GeoDataFrame) -> pl.DataFrame:
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
                # Make sure suitability value is always within allowed bounds
                .alias("suitability_value")
                .cast(pl.Int16)
                .clip(Config.MIN_NODE_SUITABILITY_VALUE, Config.MAX_NODE_SUITABILITY_VALUE)
            )
            .select("node_id", "suitability_value")
        ).collect()

        # Rejoin coordinates to the dataframe based on node_id
        coordinates = pl.from_pandas(points_within_project_area["geometry"].get_coordinates(), include_index=True)
        hexagon_points = aggregated_suit_values_per_node.join(coordinates, on="node_id", how="left")

        # Remove duplicate points, as a point could have joined multiple vector which results in duplicate rows within
        # the right dataframe.
        hexagon_points = hexagon_points.unique(subset=["node_id"])
        hexagon_points = hexagon_points.cast({pl.Int64: pl.Int32, pl.Float64: pl.Float32})

        return hexagon_points

    def convert_cartesian_coordinates_to_axial(self, hexagon_center_points: pl.DataFrame) -> pl.DataFrame:
        """
        To efficiently determine neighbours to construct a hexagonal graph later on, convert all cartesian coordinates
        to axial coordinates.

        Used algorithms as provided by:
        - coordinate to hex: https://www.redblobgames.com/grids/hexagons/#pixel-to-hex
        - rounding hex correctly: https://observablehq.com/@jrus/hexround (via redblobgames)

        :return: tuple containing q- and r-values as integers in numpy ndarray format
        """
        x, y = hexagon_center_points["x"], hexagon_center_points["y"]

        # Convert x- and y-coordinates to axial
        q = pl.Series((-2 / 3 * x) / self.hexagon_size).rename("q")
        r = pl.Series((1 / 3 * x + math.sqrt(3) / 3 * y) / self.hexagon_size).rename("r")

        # Convert coordinates to integers and correct rounding errors
        xgrid = q.round().cast(pl.Int32)
        ygrid = r.round().cast(pl.Int32)

        q_diff = q - xgrid
        r_diff = r - ygrid

        # Update x and y grid where q difference is larger than r difference
        mask = q_diff.abs() > r_diff.abs()
        updated_x_grid = xgrid + (q_diff + 0.5 * r_diff).round().cast(pl.Int32)
        xgrid = updated_x_grid.zip_with(mask, xgrid)

        updated_y_grid = ygrid + (r_diff + 0.5 * q_diff).round().cast(pl.Int32)
        ygrid = updated_y_grid.zip_with(~mask, ygrid)

        hexagon_center_points = hexagon_center_points.with_columns(xgrid, ygrid)

        return hexagon_center_points
