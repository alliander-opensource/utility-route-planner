# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from pathlib import Path
import math
from typing import Any

import pandas as pd
import rustworkx as rx
import shapely
import shapely.ops
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import (
    BaseWeightedEdgeInfo,
    NodeInfo,
    HexagonConnectionEdgeInfo,
    PipeRammingEdgeInfo,
)
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    get_hexagon_edge_geometries_for_path,
    get_hexagon_node_geometry,
)
from utility_route_planner.models.multilayer_network.multilayer_route_helpers import (
    _angle_between,
    _point_along,
    get_quadratic_bezier,
    get_inradius,
    get_tangent_arc_fillet,
)
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring, get_empty_geodataframe
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class Algorithm(Enum):
    dijkstra = auto()
    astar = auto()


@dataclass
class MultiLayerRouteResults:
    node_indices: rx.NodeIndices = field(default_factory=rx.NodeIndices)
    guideline: shapely.LineString = field(default_factory=shapely.LineString)
    unprocessed_edges: gpd.GeoDataFrame = field(default_factory=get_empty_geodataframe)
    unprocessed_nodes: gpd.GeoDataFrame = field(default_factory=get_empty_geodataframe)
    unprocessed_linestring: shapely.LineString = field(default_factory=shapely.LineString)
    collapsed_linestring: shapely.LineString = field(default_factory=shapely.LineString)
    collapsed_node_indices: list = field(default_factory=list)
    quadratic_bezier_linestring: shapely.LineString = field(default_factory=shapely.LineString)
    string_pulled_linestring: shapely.LineString = field(default_factory=shapely.LineString)

    def write_to_geopackage(self, out: Path, prefix: str = "") -> None:
        """Write results containing a geometry to file using the dataclass field name."""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, gpd.GeoDataFrame | gpd.GeoSeries):
                if value.empty:
                    continue
            elif isinstance(value, shapely.geometry.base.BaseGeometry):
                if value.is_empty:
                    continue
            elif isinstance(value, list) and len(value) > 0:
                # This is the node indices list, subset the unprocessed nodes
                value = self.unprocessed_nodes.loc[value]
            else:
                continue
            write_results_to_geopackage(out, value, f"{prefix}result_route_{f.name}")


class MultilayerRouteEngine:
    def __init__(
        self,
        cost_surface_graph: rx.PyGraph,
        osm_graph: rx.PyGraph,
        gdf_cost_surface_nodes: gpd.GeoDataFrame,
        hexagon_size: float,
        algorithm: Algorithm = Algorithm.astar,
        prefix: str = "",
        write_output: bool = False,
        experimental_smoothing: bool = False,
        out: Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.hexagon_size = hexagon_size

        self.algorithm = algorithm
        self.prefix = prefix
        self.write_output = write_output
        self.experimental_smoothing = experimental_smoothing
        self.out = out

        self.minimum_bending_radius = get_inradius(self.hexagon_size)
        self.results = MultiLayerRouteResults()

    def find_route(self, start_end: shapely.LineString):
        source, target = self.get_source_and_target_nodes(start_end)

        straight_line = self.get_linestring(source, target)
        # Offset to avoid it being exactly on top of the nodes, causes issues with distance calculations during routing.
        self.results.guideline = shapely.offset_curve(straight_line, self.hexagon_size / 4)

        path_node_indices = self.find_path_node_indices(source, target)

        gdf_path_nodes = self.gdf_cost_surface_nodes.loc[path_node_indices].copy()
        gdf_path_edges = get_hexagon_edge_geometries_for_path(
            self.cost_surface_graph, path_node_indices, gdf_path_nodes
        )

        self.results.unprocessed_edges = gdf_path_edges
        self.results.unprocessed_nodes = gdf_path_nodes
        self.results.node_indices = path_node_indices

        self.results.unprocessed_linestring = shapely.LineString(
            [get_hexagon_node_geometry(self.gdf_cost_surface_nodes, node_id=i) for i in path_node_indices]
        )
        self.results.collapsed_linestring, self.results.collapsed_node_indices = self.get_collapsed_route()

        if self.experimental_smoothing:
            self.apply_bezier_curves(min_bend_radius=self.minimum_bending_radius)
            self.apply_string_pulling(min_bend_radius=self.minimum_bending_radius)

        if self.write_output:
            self.results.write_to_geopackage(self.out, self.prefix)

    def get_source_and_target_nodes(self, start_end: shapely.LineString) -> tuple[int, int]:
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.iloc[self.gdf_cost_surface_nodes.distance(start).idxmin()].name
        target = self.gdf_cost_surface_nodes.iloc[self.gdf_cost_surface_nodes.distance(end).idxmin()].name
        if source == target:
            raise ValueError("Source and target node are the same. Provide a linestring with points further apart.")
        return source, target

    def get_result_route_length_unprocessed(self) -> float:
        return self.results.unprocessed_edges.geometry.length.sum()

    def get_result_route_cost(self) -> float:
        """
        For now, divide total route cost by 2 as the edge weight is now computed as the sum of the weights of the source
        and target nodes.
        """
        return self.results.unprocessed_edges["weight"].sum() / 2

    def get_weight_dijkstra(self, edge: BaseWeightedEdgeInfo, modifier: float = 0.01) -> float:
        """
        Weight is leading for edges (MCDA), but we want to add a small distance-based cost to prefer routes that are
        closer to the straight line between start and end.
        """
        weight = self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight
        node_1, node_2 = self.cost_surface_graph.get_edge_endpoints_by_index(edge.edge_id)
        edge_line = self.get_linestring(node_1, node_2)
        distance = edge_line.distance(self.results.guideline) * modifier
        if distance > weight:
            logger.warning("Unexpected situation during routing.")
        return weight + distance

    def get_weight_astar(self, edge: BaseWeightedEdgeInfo) -> float:
        return self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight

    def get_estimate_astar(self, node: NodeInfo) -> float:
        node_point = get_hexagon_node_geometry(self.gdf_cost_surface_nodes, node.node_id)
        guideline = shapely.LineString([node_point, shapely.get_point(self.results.guideline, 1)])
        # TODO i think this can be improved, the guideline is not always leading

        return guideline.length

    def get_linestring(self, node_1: int, node_2: int) -> shapely.LineString:
        nodes = self.gdf_cost_surface_nodes
        edge_line = shapely.LineString(
            [
                get_hexagon_node_geometry(nodes, node_1),
                get_hexagon_node_geometry(nodes, node_2),
            ]
        )
        return edge_line

    @time_function
    def find_path_node_indices(self, source, target):
        logger.info("Starting route finding.")
        match self.algorithm:
            case Algorithm.dijkstra:
                path_node_indices = rx.dijkstra_shortest_paths(
                    self.cost_surface_graph, source, target, self.get_weight_dijkstra
                )
                path_node_indices = path_node_indices[target]
            case Algorithm.astar:
                path_node_indices = rx.astar_shortest_path(
                    self.cost_surface_graph,
                    node=source,
                    goal_fn=lambda x: x.node_id == target,
                    edge_cost_fn=self.get_weight_astar,
                    estimate_cost_fn=self.get_estimate_astar,
                )
            case _:
                raise ValueError(f"Unsupported algorithm type. Expected one of: {[a for a in Algorithm]}")
        return path_node_indices

    def _get_shortcut_costs(self, line: shapely.LineString, inradius: float, height: int) -> list[float]:
        # Take some margin because of rounding differences to ensure proper selection.
        nearby = self.gdf_cost_surface_nodes[
            (self.gdf_cost_surface_nodes["height_level"] == height)
            & (self.gdf_cost_surface_nodes.dwithin(line, inradius * 1.02))
        ]
        costs = nearby.suitability_value.unique()

        # Only look at the intersection when necessary to save resources
        if len(costs) != 1:
            within_inner = nearby[nearby.dwithin(line, inradius * 0.98)]
            inner_unique = within_inner.suitability_value.unique()

            intersected_left_line = line.buffer(inradius * 1.02, single_sided=True)
            intersected_right_line = line.buffer(-inradius * 1.02, single_sided=True)

            intersected_left = nearby[nearby.intersects(intersected_left_line)]
            intersected_right = nearby[nearby.intersects(intersected_right_line)]

            if len(intersected_left.suitability_value.unique()) == 1 and len(inner_unique) == 1:
                costs = intersected_left.suitability_value.unique()
            elif len(intersected_right.suitability_value.unique()) == 1 and len(inner_unique) == 1:
                costs = intersected_right.suitability_value.unique()
            else:
                costs = nearby.suitability_value.unique()

        # Multiply each node suitability value by 2, as edge weights are set as the sum of two node suitability values.
        return (costs * 2).tolist()

    @time_function
    def get_collapsed_route(self) -> tuple[shapely.LineString, list[int]]:
        """
        The idea is to create shortcuts in the route by skipping nodes if the cost does not change. This is done by
        creating a linestring from the current node to the next node in the route and checking if the suitability values
        of the cost surface nodes that are within a certain distance (depending on hexagon size used) from this
        linestring are all the same as the suitability value of the current node. If they are, we can skip the nodes in
        between and continue from the forwarded node. If they are not, we add the last node that we could skip to the
        shortcut order and continue from there.

        This is effective when the route crosses an area that is homogenous in suitability value and reduces the
        zigzag effect.

        Illustrations: https://steamcdn-a.akamaihd.net/apps/valve/2009/ai_systems_of_l4d_mike_booth.pdf
        - AKA collapsed path.

        This results in a "straightened" linestring.

        """
        logger.info("Starting collapsed route calculation of the found route.")
        # center to center distance from a neighbouring hexagon
        inradius = get_inradius(self.hexagon_size)
        shortcut_order: list = [self.results.node_indices[0]]

        # Get segments to consider for shortcuts
        gdf_crossed_nodes = self.get_segments()

        for segment in gdf_crossed_nodes["segment"].unique():
            gdf_active_mask = gdf_crossed_nodes[gdf_crossed_nodes["segment"] == segment]
            current_height_level = gdf_active_mask["height_level"].iloc[0]

            if gdf_active_mask.empty:
                continue
            if len(gdf_active_mask) == 1:
                node_id = int(gdf_active_mask.iloc[0]["node_id"])
                if shortcut_order[-1] != node_id:
                    shortcut_order.append(node_id)
                continue

            start_node = int(gdf_active_mask.iloc[0]["node_id"])
            forwarded_node = int(gdf_active_mask.iloc[1]["node_id"])
            end_node = int(gdf_active_mask.iloc[-1]["node_id"])

            while start_node != end_node:
                # For each node in the active segment, create a line from start_node and compute shortcut costs.
                # Pick the last node (most skipped) with still the same suitability costs
                basic_cost = self.cost_surface_graph.get_edge_data(start_node, forwarded_node).weight
                start_node_geom = get_hexagon_node_geometry(self.gdf_cost_surface_nodes, node_id=start_node)
                # Create lines from start_node to all nodes in the active segment
                series_forwarded = gpd.GeoSeries(shapely.shortest_line(start_node_geom, gdf_active_mask["geometry"]))
                if current_height_level != 0:
                    # Select only those nodes within a reasonable perimeter of the route
                    height_nodes = self.gdf_cost_surface_nodes[
                        self.gdf_cost_surface_nodes["height_level"] == current_height_level
                    ]
                    minx, miny, maxx, maxy = series_forwarded.buffer(2).total_bounds
                    subset_height_nodes = height_nodes.cx[minx:maxx, miny:maxy]
                    valid_area = (
                        # Ensure we create a single polygon with the union.
                        subset_height_nodes.buffer(inradius * 2, resolution=4)
                        .union_all()
                        # Give it a bit of slack to prevent it being too strict at the edges.
                        .buffer(-inradius * 1.6, resolution=4)
                    )
                    series_forwarded = series_forwarded.intersection(valid_area)
                    series_forwarded = series_forwarded.where(
                        (series_forwarded.geom_type == "LineString")
                        & (~series_forwarded.is_empty)
                        & (series_forwarded.intersects(start_node_geom)),
                        other=shapely.LineString(),
                    )
                # Compute shortcut costs for each line
                series_shortcut_costs = series_forwarded.apply(
                    self._get_shortcut_costs, inradius=inradius, height=current_height_level
                )

                # Filter nodes where shortcut costs equal basic_cost
                valid_nodes = gdf_active_mask[series_shortcut_costs.apply(lambda costs: costs == [basic_cost])]

                if valid_nodes.empty:
                    # No valid shortcut found for this part of the segment, move to the next node
                    shortcut_order.append(int(forwarded_node))
                    gdf_active_mask = gdf_active_mask[1:]
                    start_node = int(gdf_active_mask.iloc[0]["node_id"])
                    forwarded_node = int(gdf_active_mask.iloc[1]["node_id"]) if len(gdf_active_mask) > 1 else end_node
                else:
                    # Pick the last valid node (most nodes skipped)
                    start_node = int(valid_nodes.iloc[-1]["node_id"])
                    shortcut_order.append(start_node)
                    gdf_active_mask = gdf_active_mask[gdf_active_mask.index > valid_nodes.iloc[-1].name]
                    forwarded_node = int(gdf_active_mask.iloc[0]["node_id"]) if len(gdf_active_mask) > 0 else end_node

        shortcut_linestring = shapely.LineString(
            gdf_crossed_nodes[gdf_crossed_nodes["node_id"].isin(shortcut_order)].geometry.to_list()
        )

        logger.info(
            f"Input LineString: {self.results.unprocessed_linestring.length}. Collapsed LineString: {shortcut_linestring.length}."
        )

        return shortcut_linestring, shortcut_order

    def get_segments(self) -> Any:
        gdf_crossed_nodes = self.gdf_cost_surface_nodes.loc[self.results.node_indices].reset_index()
        edges = self.results.unprocessed_edges.reset_index(drop=True)

        is_height = edges["connects_height_levels"].fillna(False)
        is_junction = edges["origin"].notna() if "origin" in edges.columns else pd.Series(False, index=edges.index)
        is_special = is_height | is_junction

        # An edge starts a new segment when:
        #  - it is special (height transition or junction crossing)
        #  - the previous edge was special (so the node after a special edge starts fresh)
        #  - its weight differs from the previous edge.
        edge_break = is_special | is_special.shift(fill_value=False) | (edges["weight"] != edges["weight"].shift())
        edge_break.iloc[0] = True
        edge_segment = edge_break.cumsum()

        # Map edge segments onto nodes. Node i takes the segment of the edge that arrives at it
        # (edge i-1). The first node takes the first edge's segment.
        node_segment = pd.Series(index=gdf_crossed_nodes.index, dtype=int)
        node_segment.iloc[0] = edge_segment.iloc[0]
        node_segment.iloc[1:] = edge_segment.values

        gdf_crossed_nodes["segment"] = node_segment.values.astype(int)
        return gdf_crossed_nodes

    @time_function
    def apply_bezier_curves(
        self,
        min_bend_radius: float,
        samples_per_curve: int = 30,
    ):
        """
        Replace corners in straightened route with quadratic Bezier arcs.

        At each interior vertex P_i the corner formed by legs (P_{i-1} -> P_i) and
        (P_i -> P_{i+1}) is replaced by:
          - a straight piece up to point A on the incoming leg, distance d back from P_i
          - a quadratic Bezier with control point P_i, ending at point B on the outgoing
            leg, distance d forward from P_i
        The offset d is chosen so that:
          1. The minimum radius of curvature of the Bezier >= min_bend_radius.
             For a symmetric quadratic Bezier with offsets d and deflection angle alpha,
             r_min = d * tan(alpha / 2)  ->  d_min = min_bend_radius * tan(alpha / 2).
          2. The Bezier does not enter hexagon cells whose suitability_value differs
             from the cells covered by the two legs being joined.
        Adjacent corners share legs, so each corner can use at most half of a leg's length.

        """
        logger.info("Starting route smoothing through application of bezier curves.")

        # TODO retrieve height during edge transition
        height = 0

        # -- part 1: sanity check
        coords = list(self.results.collapsed_linestring.coords)
        if len(self.results.collapsed_node_indices) != len(coords):
            raise ValueError("Results seem to be desynced, exiting.")

        if len(self.results.collapsed_node_indices) < 3:
            logger.info("The resulting collapsed route has less than 3 points, unable to create bezier curve.")
            self.results.quadratic_bezier_linestring = self.results.collapsed_linestring
            return

        # -- part 2: prepare
        inradius = get_inradius(self.hexagon_size)

        segments_to_smooth = []
        for i in range(len(coords) - 1):
            node_a, node_b = (self.results.collapsed_node_indices[i], self.results.collapsed_node_indices[i + 1])
            collapsed_linestring = shapely.LineString(
                [
                    shapely.get_point(self.results.collapsed_linestring, i),
                    shapely.get_point(self.results.collapsed_linestring, i + 1),
                ]
            )
            shortcut_costs = self._get_shortcut_costs(collapsed_linestring, inradius, 0)
            if self.cost_surface_graph.has_edge(node_a, node_b):
                edge_data = self.cost_surface_graph.get_edge_data(node_a, node_b)
            else:
                edge_data = None
            segments_to_smooth.append(
                SmootherHelper(
                    collapsed_node_indices=(node_a, node_b),
                    collapsed_linestring=collapsed_linestring,
                    shortcut_cost=shortcut_costs,
                    special_edge=edge_data,
                )
            )

        # part 3: apply curves.
        # - for "normal" segment pairs, apply quadratic beziers # TODO do something about the arbitrary shrinking. it should "hug" the inradius of adjacent higher value hexagons
        # - for height transitions / pipe ramming we can possibly get a 180 degree reversal, so we need to apply a circular arc with the largest radius that still fits the inradius corridor.

        new_pieces: list[shapely.LineString] = []
        cursor = shapely.get_point(segments_to_smooth[0].collapsed_linestring, 0)

        # Skip first/last point
        for segment_1, segment_2 in zip(segments_to_smooth, segments_to_smooth[1:]):
            p_prev = shapely.get_point(segment_1.collapsed_linestring, 0)
            p_curr = shapely.get_point(segment_1.collapsed_linestring, 1)  # same as 0, segment_2
            p_next = shapely.get_point(segment_2.collapsed_linestring, 1)

            v_in = (p_curr.x - p_prev.x, p_curr.y - p_prev.y)
            v_out = (p_next.x - p_curr.x, p_next.y - p_curr.y)
            alpha = _angle_between(v_in, v_out)

            # Essentially straight, no curve needed.
            if alpha < 1e-3:
                continue

            d_min = min_bend_radius * math.tan(alpha / 2)
            d_max = 0.5 * min(segment_1.length, segment_2.length)

            if d_min > d_max:
                logger.warning(
                    "Cannot satisfy minimum bend radius at vertex.",
                    vertex=i,
                    d_required=d_min,
                    d_available=d_max,
                )
                # Fall back: keep the sharp corner.
                new_pieces.append(shapely.LineString([cursor, p_curr]))
                cursor = p_curr
                continue

            # TODO if not the same cost, how to handle?
            allowed_costs = set(segment_1.shortcut_cost) | set(segment_2.shortcut_cost)

            # Iteratively shrink d until the curve stays inside allowed cells.
            d = d_max
            bezier_line = shapely.LineString()
            for _ in range(10):
                a = _point_along(p_curr, p_prev, d)
                b = _point_along(p_curr, p_next, d)
                bezier_line = get_quadratic_bezier(a, p_curr, b, samples_per_curve)

                if self._curve_stays_in_cells(bezier_line, allowed_costs, inradius, height):
                    break
                if d <= d_min + 1e-6:
                    break
                d = max(d_min, d * 0.5)

            new_pieces.append(shapely.LineString([cursor, shapely.Point(bezier_line.coords[0])]))
            new_pieces.append(bezier_line)
            cursor = shapely.Point(bezier_line.coords[-1])

        new_pieces.append(shapely.LineString([cursor, shapely.Point(coords[-1])]))

        # Concatenate raw coordinates
        bezier_coordinates_merged = [coord for piece in new_pieces for coord in piece.coords]
        merged = shapely.remove_repeated_points(shapely.LineString(bezier_coordinates_merged), tolerance=0)

        self.results.quadratic_bezier_linestring = merged

    def _curve_stays_in_cells(self, curve: shapely.LineString, allowed: set, inradius: float, height: int) -> bool:
        costs = self._get_shortcut_costs(curve, int(inradius), height)
        return set(costs).issubset(allowed)

    @time_function
    def apply_string_pulling(
        self,
        min_bend_radius: float,
        samples_per_curve: int = 30,
    ):
        """Apply string pulling"""
        logger.info("Starting route smoothing through application of string pulling / tangent arc fillets.")

        # TODO change MCDA so it resolves on "touches", that way we can use innradius.
        # TODO retrieve height during edge transition
        height = 0

        # -- part 1: sanity check
        coords = list(self.results.collapsed_linestring.coords)
        if len(self.results.collapsed_node_indices) != len(coords):
            raise ValueError("Results seem to be desynced, exiting.")

        if len(self.results.collapsed_node_indices) < 3:
            logger.info("The resulting collapsed route has less than 3 points, unable to create a fillet.")
            self.results.string_pulled_linestring = self.results.collapsed_linestring
            return

        # -- part 2: prepare
        inradius = get_inradius(self.hexagon_size)

        segments_to_smooth = []
        for i in range(len(coords) - 1):
            node_a, node_b = (self.results.collapsed_node_indices[i], self.results.collapsed_node_indices[i + 1])
            collapsed_linestring = shapely.LineString(
                [
                    shapely.get_point(self.results.collapsed_linestring, i),
                    shapely.get_point(self.results.collapsed_linestring, i + 1),
                ]
            )
            shortcut_costs = self._get_shortcut_costs(collapsed_linestring, inradius, 0)
            if self.cost_surface_graph.has_edge(node_a, node_b):
                edge_data = self.cost_surface_graph.get_edge_data(node_a, node_b)
            else:
                edge_data = None
            segments_to_smooth.append(
                SmootherHelper(
                    collapsed_node_indices=(node_a, node_b),
                    collapsed_linestring=collapsed_linestring,
                    shortcut_cost=shortcut_costs,
                    special_edge=edge_data,
                )
            )

        # part 3: apply corner cutting
        new_pieces: list = []
        for segment_1, segment_2 in zip(segments_to_smooth, segments_to_smooth[1:]):
            p_prev = (
                shapely.Point(new_pieces[-1].coords[-1])
                if new_pieces
                else shapely.get_point(segment_1.collapsed_linestring, 0)
            )
            p_curr = shapely.get_point(segment_1.collapsed_linestring, 1)  # same as 0, segment_2
            p_next = shapely.get_point(segment_2.collapsed_linestring, 1)

            if isinstance(segment_2.special_edge, (HexagonConnectionEdgeInfo, PipeRammingEdgeInfo)):
                # do not shortcut these segments
                nice_curve = get_tangent_arc_fillet(p_prev, p_curr, p_next, inradius)
                new_pieces.append(nice_curve)
                continue
            elif (
                isinstance(segment_1.special_edge, (HexagonConnectionEdgeInfo, PipeRammingEdgeInfo))
                and len(new_pieces) > 0
            ):
                # previous segment was special, add arcfillet to the start prior to smoothing
                nice_curve = get_tangent_arc_fillet(p_prev, p_curr, p_next, inradius)
                new_pieces.append(nice_curve)
                p_prev = shapely.get_point(nice_curve, -1)
                shortcut_costs = [segment_2.shortcut_cost[0] / 2]
            else:
                # TODO how to pick this value when having transitions?
                shortcut_costs = [segment_2.shortcut_cost[0] / 2]

            # Should this approach give anymore troubles, we can just select the nearest hexagon from the collapsed path and define that as obstacle.
            line_with_obstacle = shapely.LineString([p_prev, p_next])
            obstacle_area = shapely.convex_hull(
                shapely.GeometryCollection([segment_2.collapsed_linestring, line_with_obstacle])
            )
            obstacle_hexagons = self.gdf_cost_surface_nodes[
                (self.gdf_cost_surface_nodes["height_level"] == height)
                & (self.gdf_cost_surface_nodes.dwithin(obstacle_area, inradius * 1.01))
                & (~self.gdf_cost_surface_nodes.suitability_value.isin(shortcut_costs))
            ]
            if obstacle_hexagons.empty:
                continue
            obstacle = obstacle_hexagons.buffer(inradius * 1.01).union_all().intersection(obstacle_area)

            convex_hull = shapely.convex_hull(shapely.GeometryCollection([obstacle, line_with_obstacle]))
            if not isinstance(convex_hull, shapely.Polygon):
                continue

            hull_coords = list(convex_hull.exterior.coords[:-1])
            taut_points = [
                shapely.Point(c)
                for c in hull_coords
                if not shapely.Point(c).intersects(line_with_obstacle.buffer(0.01))
            ]
            taut_points = shapely.MultiPoint(taut_points)
            # Keep the convex hull's natural ring order (no crossings), but rotate it so the
            # chain starts at the taut point closest to where we currently are, and walks toward p_next.
            hull_points = list(taut_points.geoms)
            if len(hull_points) > 1:
                # anchor = shapely.Point(new_pieces[-1].coords[-1]) if new_pieces else p_prev

                # 1. Rotate the ring so it starts at the vertex closest to the anchor.
                start_idx = min(range(len(hull_points)), key=lambda i: hull_points[i].distance(p_prev))
                rotated = hull_points[start_idx:] + hull_points[:start_idx]

                # 2. Pick the walking direction whose end finishes closest to p_next.
                forward_end_dist = rotated[-1].distance(p_next)
                reversed_chain = [rotated[0]] + rotated[1:][::-1]
                reversed_end_dist = reversed_chain[-1].distance(p_next)
                if reversed_end_dist < forward_end_dist:
                    rotated = reversed_chain

                hull_points = rotated

            taut_points = shapely.MultiPoint(hull_points)
            if taut_points.is_empty or len(taut_points.geoms) < 2:
                print("stahp")
                continue

            new_pieces.append(shapely.LineString([*taut_points.geoms]))

            prefix = "pytest_a_"
            write_results_to_geopackage(
                self.out,
                shapely.MultiLineString([segment_1.collapsed_linestring, segment_2.collapsed_linestring]),
                f"{prefix}segment_12",
                overwrite=True,
            )
            write_results_to_geopackage(self.out, line_with_obstacle, f"{prefix}line_with_obstacle", overwrite=True)
            write_results_to_geopackage(self.out, obstacle, f"{prefix}obstacle", overwrite=True)
            write_results_to_geopackage(self.out, obstacle_area, f"{prefix}obstacle_area", overwrite=True)
            write_results_to_geopackage(self.out, obstacle_hexagons, f"{prefix}obstacle_hexagons", overwrite=True)
            # write_results_to_geopackage(self.out, leading_obstacle, f'{prefix}leading_obstacle', overwrite=True)
            write_results_to_geopackage(self.out, convex_hull, f"{prefix}convex_hull", overwrite=True)
            write_results_to_geopackage(
                self.out, shapely.MultiLineString(new_pieces), f"{prefix}new_piece", overwrite=True
            )
            write_results_to_geopackage(self.out, taut_points, f"{prefix}taut_points", overwrite=True)

        # Merge to a single linestring
        coordinates_merged = [coord for piece in new_pieces for coord in piece.coords]
        coordinates_merged.insert(0, shapely.get_point(self.results.collapsed_linestring, 0))
        coordinates_merged.append(shapely.get_point(self.results.collapsed_linestring, -1))
        merged = shapely.remove_repeated_points(shapely.LineString(coordinates_merged), tolerance=0)

        self.results.string_pulled_linestring = merged


@dataclass
class SmootherHelper:
    collapsed_node_indices: tuple[int, int]
    collapsed_linestring: shapely.LineString
    shortcut_cost: list[float]
    special_edge: BaseWeightedEdgeInfo | HexagonConnectionEdgeInfo | PipeRammingEdgeInfo | None = None
    length: float = field(init=False)

    def __post_init__(self):
        self.length = self.collapsed_linestring.length
