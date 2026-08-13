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


def get_quadratic_bezier(p0: shapely.Point, p1: shapely.Point, p2: shapely.Point, samples: int) -> shapely.LineString:
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


def get_tangent_arc_fillet(
    p_prev: shapely.Point,
    p_curr: shapely.Point,
    p_next: shapely.Point,
    radius: float,
    offset: float | None = None,
    samples: int = 30,
) -> shapely.LineString:
    """Create a circular arc fillet of the given radius, tangent to both legs of a corner."""
    v_in = (p_curr.x - p_prev.x, p_curr.y - p_prev.y)
    v_out = (p_next.x - p_curr.x, p_next.y - p_curr.y)
    norm_in = math.hypot(*v_in)
    norm_out = math.hypot(*v_out)
    if norm_in == 0 or norm_out == 0:
        return shapely.LineString([p_prev, p_next])

    # Deflection angle of the corner (0 == straight, pi == full reversal).
    alpha = _angle_between(v_in, v_out)
    if alpha < 1e-9:
        # Essentially straight, no fillet needed.
        return shapely.LineString([p_prev, p_next])

    # Tangent offset for an arc of this radius to touch both legs. Clamp to the available leg
    # length so the tangent points never overshoot the corner.
    if offset is None:
        offset = radius * math.tan(alpha / 2)
    offset = min(offset, norm_in, norm_out)

    a = _point_along(p_curr, p_prev, offset)
    b = _point_along(p_curr, p_next, offset)

    v_in_hat = (v_in[0] / norm_in, v_in[1] / norm_in)

    # Turn direction: a positive cross product is a left (counter-clockwise) turn. The
    # arc centre lies on the inside of the corner, i.e. along the inward normal of the
    # incoming leg at tangent point A.
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    if cross >= 0:
        normal = (-v_in_hat[1], v_in_hat[0])
        counter_clockwise = True
    else:
        normal = (v_in_hat[1], -v_in_hat[0])
        counter_clockwise = False

    center = shapely.Point(a.x + radius * normal[0], a.y + radius * normal[1])

    ang_a = math.atan2(a.y - center.y, a.x - center.x)
    ang_b = math.atan2(b.y - center.y, b.x - center.x)
    sweep = ang_b - ang_a
    # Normalise the swept angle to match the turn direction (minor arc).
    if counter_clockwise:
        while sweep <= 0:
            sweep += 2 * math.pi
    else:
        while sweep >= 0:
            sweep -= 2 * math.pi

    arc_points = []
    for sample in range(samples):
        rel = sample / (samples - 1)
        ang = ang_a + sweep * rel
        arc_points.append((center.x + radius * math.cos(ang), center.y + radius * math.sin(ang)))
    return shapely.LineString(arc_points)


def create_arc_fillets(line: shapely.LineString, radius: float, samples: int = 30) -> shapely.LineString:
    """
    Round every interior corner of a linestring with tangent arc fillets of the given radius.

    Straight pieces connect the tangent points of consecutive fillets, so the result is a
    smooth path whose curvature never exceeds ``1 / radius``.

    :param line: the polyline to fillet (e.g. a collapsed/straightened route).
    :param radius: fillet radius, typically the hexagon inradius.
    :param samples: number of points used to discretise each arc.
    :return: the filleted linestring.
    """
    coords = list(line.coords)
    if len(coords) < 3:
        return line

    points = [shapely.Point(c) for c in coords]
    stitched: list[tuple[float, float]] = [coords[0]]
    for i in range(1, len(points) - 1):
        arc = get_tangent_arc_fillet(points[i - 1], points[i], points[i + 1], radius, samples=samples)
        # Straight run up to the arc start, then the arc itself.
        stitched.append(arc.coords[0])
        stitched.extend(arc.coords[1:])
    stitched.append(coords[-1])

    return shapely.remove_repeated_points(shapely.LineString(stitched), tolerance=0)
