import math

import numpy as np
import shapely

from utility_route_planner.models.mcda.mcda_utils import create_project_area_grid


class TestProjectAreaGridCreation:
    def test_grid_block_size_smaller_than_max_block_size(self):
        project_area_geometry = shapely.Point(187542.57, 428280.47).buffer(1000)
        project_area_grid = create_project_area_grid(*project_area_geometry.bounds, 2048)

        # The number of blocks must be equal to 1, as the MAX_BLOCK_SIZE exceeds the project area size
        assert len(project_area_grid) == 1

        # The bounds of the project area must be equal to the bounding box of the grid
        assert np.array_equal(project_area_grid.total_bounds, np.array(project_area_geometry.bounds))

    def test_create_project_area_grid_block_size_exceeds_max_block_size(self):
        project_area_geometry = shapely.Point(187542.57, 428280.47).buffer(5000)
        max_block_size = 2048
        project_area_grid = create_project_area_grid(*project_area_geometry.bounds, max_block_size)

        # The number of blocks on the y- and x-axis should be >= the ceiled number of blocks that can be placed given
        # the width and height and the max block size
        min_x, min_y, max_x, max_y = project_area_geometry.bounds
        expected_blocks_width = math.ceil((max_x - min_x) / max_block_size)
        expected_blocks_height = math.ceil((max_y - min_y) / max_block_size)
        assert len(project_area_grid) >= expected_blocks_width * expected_blocks_height

        # Verify that the size of each block is equal to the max block size
        for block_geom in project_area_grid.geometry:
            x_min, y_min, x_max, y_max = block_geom.bounds
            assert x_max - x_min <= max_block_size
            assert y_max - y_min <= max_block_size

        # Verify that the project area is completely covered by the project area grid
        assert project_area_geometry.within(project_area_grid.unary_union)
