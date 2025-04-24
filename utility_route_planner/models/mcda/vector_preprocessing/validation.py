# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from utility_route_planner.models.mcda.exceptions import UnassignedValueFoundDuringReclassify


def validate_values_to_reclassify(values_to_reclassify: list, assigned_values: dict):
    for value in values_to_reclassify:
        if not assigned_values.get(value):
            raise UnassignedValueFoundDuringReclassify(
                f"Value: {value} is not present in the weight values dictionary as valid non-zero integer: {assigned_values.keys()}. Add it to the mcda_presets.py."
            )
