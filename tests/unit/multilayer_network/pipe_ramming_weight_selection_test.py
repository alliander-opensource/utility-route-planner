#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import math
import pytest

from utility_route_planner.models.multilayer_network.pipe_ramming import (
    GetPotentialPipeRammingCrossings,
    PipeRammingSettings,
)

ASPHALT_VALUE = 36  # wegdeel: voetpad (3) + gesloten verharding (33), cheapest closed-paved street cell


@pytest.mark.parametrize("hexagon_size", [0.5, 1.0])  # coarser than 1m does not make sense for ramming
def test_pipe_ramming_preferred_over_asphalt_trench(hexagon_size):
    """A ramming across the narrowest asphalt band beats trenching it (graph units, sum of node values)."""
    settings = PipeRammingSettings(hexagon_size=hexagon_size)

    step = math.sqrt(3) * hexagon_size
    n_asphalt = max(1, math.floor(settings.min_pipe_ramming_length_m / step))  # lower bound on asphalt cells crossed
    length = (n_asphalt + 1) * step  # costliest ramming over those cells

    # Each extra asphalt cell adds more to the trench than to the ramming, so the narrowest band binds.
    assert ASPHALT_VALUE > settings.ramming_impedance_per_m * step
    # The via must exist for the narrowest band, otherwise the route trenches regardless of cost.
    assert length >= settings.min_pipe_ramming_length_m

    ramming = int(GetPotentialPipeRammingCrossings.get_pipe_ramming_weight(length, settings))
    trench = 2 * n_asphalt * ASPHALT_VALUE  # regular edge cost
    assert ramming < trench, f"h={hexagon_size} m: ramming {ramming} >= trench {trench}"
