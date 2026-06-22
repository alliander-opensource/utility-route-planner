# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import math

import shapely


def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Deflection angle between two 2D vectors, in [0, pi]."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.acos(cos_a)


def _point_along(origin: shapely.Point, towards: shapely.Point, distance: float) -> shapely.Point:
    dx, dy = towards.x - origin.x, towards.y - origin.y
    n = math.hypot(dx, dy)
    if n == 0:
        return origin
    return shapely.Point(origin.x + dx / n * distance, origin.y + dy / n * distance)


def _quadratic_bezier(p0: shapely.Point, p1: shapely.Point, p2: shapely.Point, samples: int) -> shapely.LineString:
    """
    Create the quadratic Bézier curve.

    :param p0: starting point
    :param p1: control point
    :param p2: end point
    :param samples: number of points to create the curve with.
    :return: quadratic Bézier curve as linestring.
    """
    bezier_points = []
    for sample in range(samples):
        rel_1 = sample / (samples - 1)
        rel_2 = 1 - rel_1
        x = rel_2 * rel_2 * p0.x + 2 * rel_2 * rel_1 * p1.x + rel_1 * rel_1 * p2.x
        y = rel_2 * rel_2 * p0.y + 2 * rel_2 * rel_1 * p1.y + rel_1 * rel_1 * p2.y
        bezier_points.append((x, y))
    return shapely.LineString(bezier_points)


def get_inradius(hexagon_size: float) -> float:
    """Get the inradius of a hexagon for checking intersections with cost surface nodes during straightening."""
    return math.sqrt(3) * hexagon_size / 2
