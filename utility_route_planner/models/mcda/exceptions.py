# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0


class InvalidGroupValue(Exception):
    pass


class InvalidConstraint(Exception):
    pass


class InvalidLayerName(Exception):
    pass


class InvalidSuitabilityValue(Exception):
    pass


class RasterCellSizeTooSmall(Exception):
    pass


class UnassignedValueFoundDuringReclassify(Exception):
    pass


class InvalidRasterValues(Exception):
    pass


class InvalidSuitabilityRasterInput(Exception):
    pass
