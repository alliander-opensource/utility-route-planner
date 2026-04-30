#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0

import math

import pytest

from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import get_hexagon_width_and_height


@pytest.mark.parametrize("size", [0.5, 1.0, 1.5, 2.0])
def test_get_hexagon_width_and_height(size: float):
    result_width, result_height = get_hexagon_width_and_height(size)
    expected_width = 2 * size
    expected_heigth = math.sqrt(3) * size

    assert expected_width == result_width
    assert expected_heigth == result_height
