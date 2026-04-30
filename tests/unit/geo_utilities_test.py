# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import shapely
from shapely.testing import assert_geometries_equal

from utility_route_planner.util.geo_utilities import (
    get_angle_between_points,
    extrapolate_point_to_target,
    extend_linestring_both_ends,
    split_polygon_by_linestrings,
    get_endlines_of_multilinestring,
)


class TestGetAngleBetweenPoints:
    @pytest.mark.parametrize(
        "a,b,c,expected",
        [
            (shapely.Point(1, 1), shapely.Point(1, 1), shapely.Point(0, 0), 0),
            (shapely.Point(0, 10), shapely.Point(0, 1), shapely.Point(0, 0), 0),
            (shapely.Point(1, 0), shapely.Point(1, 1), shapely.Point(0, 0), 45),
            (shapely.Point(1, 0), shapely.Point(0, 1), shapely.Point(0, 0), 90),
            (shapely.Point(1, 0), shapely.Point(-1, 0), shapely.Point(0, 0), 180),
            (shapely.Point(0, 1), shapely.Point(1, 0), shapely.Point(0, 0), 270),
        ],
    )
    def test_some_angles(self, a, b, c, expected):
        assert get_angle_between_points(a, b, c) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "point_a, point_b, center_point, expected_exception",
        [
            (shapely.Point(1, 1), shapely.Point(2, 2), shapely.Point(1, 1), ValueError),
            (shapely.Point(2, 2), shapely.Point(1, 1), shapely.Point(1, 1), ValueError),
        ],
    )
    def test_get_angle_between_points_raises(self, point_a, point_b, center_point, expected_exception):
        with pytest.raises(expected_exception):
            get_angle_between_points(point_a, point_b, center_point)


class TestExtendLinestring:
    @pytest.mark.parametrize(
        "point_a, direction, distance, expected",
        [
            (shapely.Point(0, 0), shapely.Point(0, 1), 100, shapely.LineString([(0, 0), (0, 100)])),
            (shapely.Point(0, 0), shapely.Point(0, 1), 10, shapely.LineString([(0, 0), (0, 10)])),
            (shapely.Point(0, 0), shapely.Point(0, 100), 10, shapely.LineString([(0, 0), (0, 10)])),
            (shapely.Point(0, 0), shapely.Point(1, 1), 10, shapely.LineString([(0, 0), (7.07, 7.07)])),
            (shapely.Point(0, 0), shapely.Point(-10, 0), 10, shapely.LineString([(0, 0), (-10, 0)])),
        ],
    )
    def test_extrapolate_point_to_target(self, point_a, direction, distance, expected):
        result = extrapolate_point_to_target(point_a, direction, distance)
        assert result.equals_exact(expected, tolerance=0.01, normalize=True)
        assert result.length >= distance
        assert shapely.get_num_points(result) == 2

    @pytest.mark.parametrize(
        "point_a, direction, distance",
        [
            (shapely.Point(0, 0), shapely.Point(0, 0), 10),
            (shapely.Point(1, 1), shapely.Point(1, 1), 10),
        ],
    )
    def test_extrapolate_point_to_target_raises(self, point_a, direction, distance):
        with pytest.raises(ValueError):
            extrapolate_point_to_target(point_a, direction, distance)


class TestExtendLinestringBothEnds:
    @pytest.mark.parametrize(
        "linestring, distance, expected",
        [
            (shapely.LineString([(0, 5), (0, 10)]), 5, shapely.LineString([(0, 0), (0, 15)])),
            (shapely.LineString([(0, 5), (0, 6), (0, 10)]), 5, shapely.LineString([(0, 0), (0, 6), (0, 15)])),
            (
                shapely.LineString([(0, 0), (5, 5), (0, 10)]),
                5,
                shapely.LineString([(-3.536, -3.536), (5, 5), (-3.536, 13.536)]),
            ),
            (shapely.LineString([(0, 0), (0, 5), (5, 5)]), 5, shapely.LineString([(0, -5), (0, 5), (10, 5)])),
            (
                shapely.LineString([(0, 5), (0, 6), (99, 1), (0, 10), (0, 11)]),
                5,
                shapely.LineString([(0, 0), (0, 6), (99, 1), (0, 10), (0, 16)]),
            ),
        ],
    )
    def test_extend_linestring_both_ends(self, linestring, distance, expected, debug=False):
        result = extend_linestring_both_ends(linestring, 5, debug)
        assert result.length == linestring.length + 2 * distance
        assert result.equals_exact(expected, tolerance=0.01, normalize=True)


class TestSplitPolygonByMultiLineString:
    @pytest.mark.parametrize(
        "polygon, linestrings, expected_polygons_count",
        [
            (shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), [shapely.LineString([(5, -5), (5, 15)])], 2),
            (
                shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                [shapely.LineString([(5, -5), (5, 15)]), shapely.LineString([(-5, 5), (15, 5)])],
                4,
            ),
            (
                shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                [
                    shapely.LineString([(5, -5), (5, 15)]),
                    shapely.LineString([(-5, 5), (15, 5)]),
                    shapely.LineString([(0, 0), (10, 10)]),
                ],
                6,
            ),
            (
                shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                [
                    shapely.LineString([(5, -5), (5, 15)]),
                    shapely.LineString([(-5, 5), (15, 5)]),
                    shapely.LineString([(0, 0), (10, 10)]),
                    shapely.LineString([(0, 10), (10, 0)]),
                ],
                8,
            ),
            (
                shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                [shapely.LineString([(15, -5), (15, 15)]), shapely.LineString([(5, -5), (5, 15)])],
                2,
            ),
            (
                shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                [
                    shapely.LineString([(5, -5), (5, -1)]),
                    shapely.LineString([(1, 1), (5, 8)]),
                    shapely.LineString([(6, 9), (6, -1)]),
                ],
                1,
            ),
        ],
    )
    def test_split_polygon(self, polygon, linestrings, expected_polygons_count, debug=False):
        result = split_polygon_by_linestrings(polygon, linestrings, debug)
        assert len(result) == expected_polygons_count
        for polygon in result:
            assert polygon.is_valid
            assert polygon.within(polygon)
            assert polygon.area > 0


class TestMultiLineStringEndparts:
    def test_get_lines_star_shape(self):
        linestring = shapely.MultiLineString(
            [
                shapely.LineString([(5, 5), (5, 10)]),
                shapely.LineString([(5, 10), (5, 5)]),  # swapped version of the above, should not matter
                shapely.LineString([(5, 5), (10, 0)]),
                shapely.LineString([(5, 5), (0, 0)]),
                shapely.LineString([(0, 5), (5, 5)]),
                shapely.LineString([(5, 5), (1, 0), (0, -1)]),
            ]
        )
        # "hub" point should always be present first.
        expected_lines = [
            shapely.LineString([(5, 5), (5, 10)]),
            shapely.LineString([(5, 5), (5, 10)]),
            shapely.LineString([(5, 5), (10, 0)]),
            shapely.LineString([(5, 5), (0, 0)]),
            shapely.LineString([(5, 5), (0, 5)]),
            shapely.LineString([(1, 0), (0, -1)]),
        ]

        lines = get_endlines_of_multilinestring(linestring)
        assert_geometries_equal(lines, expected_lines)
