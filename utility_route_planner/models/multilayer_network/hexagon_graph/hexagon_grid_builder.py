# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Generator
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import get_hexagon_width_and_height
from settings import Config


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

    def construct_grid(self, project_area: shapely.Polygon) -> gpd.GeoDataFrame:
        hexagonal_grid_bounding_box = self.construct_hexagonal_grid_for_bounding_box(project_area)
        hexagonal_grid_for_project_area = self.filter_grid_to_project_area(hexagonal_grid_bounding_box)

        weighted_hexagonal_grid = self.assign_suitability_values_to_grid(hexagonal_grid_for_project_area)
        weighted_hexagonal_grid["axial_q"], weighted_hexagonal_grid["axial_r"] = (
            self.convert_cartesian_coordinates_to_axial(weighted_hexagonal_grid)
        )
        weighted_hexagonal_grid = gpd.GeoDataFrame(
            pd.concat([weighted_hexagonal_grid, weighted_hexagonal_grid.get_coordinates()], axis=1), geometry="geometry"
        )

        # Reset index of grid to align with node ids generated using rustworkx
        weighted_hexagonal_grid = weighted_hexagonal_grid.reset_index(drop=True)
        return weighted_hexagonal_grid

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

        self.divide_matrices_into_blocks(x_matrix, y_matrix)

        bounding_box_grid = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_matrix.ravel(), y_matrix.ravel()), crs=Config.CRS
        )
        return bounding_box_grid.reset_index(names="node_id")

    def divide_matrices_into_blocks(
        self, x_matrix: np.ndarray, y_matrix: np.ndarray
    ) -> Generator[tuple[int, int, int, int, np.ndarray, np.ndarray]]:
        """
        Generator which yields indexed blocks from the x- and y-matrix given the desired block size

        :param x_matrix: x_matrix to divide into blocks
        :type x_matrix: np.ndarray
        :param y_matrix: y_matrix to divide into blocks
        :type y_matrix: np.ndarray
        :return: row_start, row_end, column_start, column_end, x_block, y_block
        :rtype: Generator[tuple[int, int, int, int, np.ndarray, np.ndarray]]
        """
        # Determine number of columns and columns given the desired block size. Round up to prevent
        # losing data
        n_rows_blocks = math.ceil(x_matrix.shape[0] // self.block_size)
        n_columns_blocks = math.ceil(y_matrix.shape[1] // self.block_size)

        row_splits = np.linspace(0, x_matrix.shape[0], n_rows_blocks, dtype=int)
        column_splits = np.linspace(0, y_matrix.shape[1], n_columns_blocks, dtype=int)

        # Iterate over the split indexes to extract the blocks from the matrices.
        for row_start, row_end in zip(row_splits[:-1], row_splits[1:]):
            for column_start, column_end in zip(column_splits[:-1], column_splits[1:]):
                print(f"Row: {row_start}:{row_end} - Column: {column_start}:{column_end}")
                x_block = x_matrix[row_start:row_end, column_start:column_end]
                y_block = y_matrix[row_start:row_end, column_start:column_end]

                yield row_start, row_end, column_start, column_end, x_block, y_block

    def filter_grid_to_project_area(self, bounding_box_grid: gpd.GeoDataFrame):
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

    def assign_suitability_values_to_grid(self, points_within_project_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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

        group_keys = points_within_project_area["group"].unique()
        aggregated_group_a = pd.DataFrame()
        aggregated_group_b = pd.DataFrame()
        aggregated_group_c = pd.DataFrame()
        points_grouped_by_group = points_within_project_area.groupby("group")
        for group_key in group_keys:
            group = points_grouped_by_group.get_group(group_key)

            match group_key:
                case "a":
                    aggregated_group_a = group.groupby("node_id").agg(
                        a=pd.NamedAgg(column="suitability_value", aggfunc="max")
                    )
                case "b":
                    aggregated_group_b = group.groupby("node_id").agg(
                        b=pd.NamedAgg(column="suitability_value", aggfunc="sum")
                    )
                case "c":
                    aggregated_group_c = group.groupby("node_id").agg(
                        c=pd.NamedAgg(column="suitability_value", aggfunc="sum")
                    )
                case _:
                    print("Invalid group value")

        aggregated_suitability_values = pd.DataFrame()
        if len(aggregated_group_a) > 0 and len(aggregated_group_b) > 0:
            aggregated_suitability_values = pd.concat([aggregated_group_a, aggregated_group_b], axis=1)
            aggregated_suitability_values = aggregated_suitability_values.fillna(0)
            aggregated_suitability_values["suitability_value"] = (
                aggregated_suitability_values.a + aggregated_suitability_values.b
            )
            aggregated_suitability_values = aggregated_suitability_values.drop(columns=["a", "b"])
        elif len(aggregated_group_a) > 0 and len(aggregated_group_b) == 0:
            aggregated_suitability_values["suitability_value"] = aggregated_group_a.a
        elif len(aggregated_group_b) > 0 and len(aggregated_group_a) == 0:
            aggregated_suitability_values["suitability_value"] = aggregated_group_b.b

        if len(aggregated_group_c) > 0:
            aggregated_suitability_values = pd.concat([aggregated_suitability_values, aggregated_group_c], axis=1)
            aggregated_suitability_values.loc[aggregated_suitability_values.c.notna(), "suitability_value"] = (
                Config.MAX_NODE_SUITABILITY_VALUE
            )
            aggregated_suitability_values = aggregated_suitability_values.drop(columns=["c"])

        # Make sure all suitability values are within boundaries
        aggregated_suitability_values["suitability_value"] = aggregated_suitability_values["suitability_value"].clip(
            Config.MIN_NODE_SUITABILITY_VALUE, Config.MAX_NODE_SUITABILITY_VALUE
        )

        # Join location afterwards, as this is faster than picking the first one within the aggregation step
        hexagon_points = gpd.GeoDataFrame(
            aggregated_suitability_values.join(
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
