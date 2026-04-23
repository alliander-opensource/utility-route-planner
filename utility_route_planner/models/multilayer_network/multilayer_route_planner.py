# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from pathlib import Path
import math

import rustworkx as rx
import shapely
import shapely.ops
import geopandas as gpd
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import BaseWeightedEdgeInfo, NodeInfo
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    get_hexagon_edge_geometries_for_path,
    get_hexagon_node_geometry,
)
from utility_route_planner.models.multilayer_network.multilayer_route_helpers import (
    _angle_between,
    _point_along,
    _quadratic_bezier,
    get_inradius,
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
    result_route_node_indices: rx.NodeIndices = field(default_factory=rx.NodeIndices)
    result_route_guideline: shapely.LineString = field(default_factory=shapely.LineString)
    result_route_edges: gpd.GeoDataFrame = field(default_factory=get_empty_geodataframe)
    result_route_nodes: gpd.GeoDataFrame = field(default_factory=get_empty_geodataframe)
    result_route_linestring: shapely.LineString = field(default_factory=shapely.LineString)
    result_route_straightened: shapely.LineString = field(default_factory=shapely.LineString)
    result_route_straightened_node_indices: list = field(default_factory=list)
    result_route_quadratic_bezier: shapely.LineString = field(default_factory=shapely.LineString)

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
            else:
                continue
            write_results_to_geopackage(out, value, f"{prefix}{f.name}")


class MultilayerRouteEngine:
    def __init__(
        self,
        cost_surface_graph: rx.PyGraph,
        osm_graph: rx.PyGraph,
        gdf_cost_surface_nodes: gpd.GeoDataFrame,
        hexagon_size: float,
        # minimum_bending_radius: float = 0,
        algorithm: Algorithm = Algorithm.dijkstra,
        prefix: str = "",
        write_output: bool = False,
        out: Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
    ):
        self.cost_surface_graph = cost_surface_graph
        self.gdf_cost_surface_nodes = gdf_cost_surface_nodes
        self.osm_graph = osm_graph
        self.hexagon_size = hexagon_size
        self.minimum_bending_radius = get_inradius(self.hexagon_size) / 2
        self.write_output = write_output
        self.algorithm = algorithm
        self.prefix = prefix
        self.out = out

        self.results = MultiLayerRouteResults()

    @time_function
    def find_route(self, start_end: shapely.LineString):
        source, target = self.get_source_and_target_nodes(start_end)

        straight_line = self.get_linestring(source, target)
        # Offset to avoid it being exactly on top of the nodes, causes issues with distance calculations during routing.
        self.results.result_route_guideline = shapely.offset_curve(straight_line, self.hexagon_size / 4)

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
                raise ValueError("Unsupported algorithm type.")

        gdf_path_nodes = self.gdf_cost_surface_nodes.loc[path_node_indices].copy()
        gdf_path_edges = get_hexagon_edge_geometries_for_path(
            self.cost_surface_graph, path_node_indices, gdf_path_nodes
        )

        self.results.result_route_edges = gdf_path_edges
        self.results.result_route_nodes = gdf_path_nodes
        self.results.result_route_node_indices = path_node_indices

        self.results.result_route_linestring = shapely.LineString(
            [get_hexagon_node_geometry(self.gdf_cost_surface_nodes, node_id=i) for i in path_node_indices]
        )
        self.results.result_route_straightened, self.results.result_route_straightened_node_indices = (
            self.straighten_linestring()
        )
        if self.minimum_bending_radius:
            self.apply_bezier_curves(min_bend_radius=self.minimum_bending_radius)

        if self.write_output:
            self.results.write_to_geopackage(self.out, self.prefix)
            write_results_to_geopackage(
                self.out,
                self.gdf_cost_surface_nodes.loc[self.results.result_route_straightened_node_indices],
                f"{self.prefix}result_route_shortcut_nodes",
            )

    def get_source_and_target_nodes(self, start_end: shapely.LineString) -> tuple[int, int]:
        start, end = get_first_last_point_from_linestring(start_end)
        source = self.gdf_cost_surface_nodes.iloc[self.gdf_cost_surface_nodes.distance(start).idxmin()].name
        target = self.gdf_cost_surface_nodes.iloc[self.gdf_cost_surface_nodes.distance(end).idxmin()].name
        if source == target:
            raise ValueError("Source and target node are the same. Provide a linestring with points further apart.")
        return source, target

    def get_result_route_length(self) -> float:
        return self.results.result_route_edges.geometry.length.sum()

    def get_result_route_cost(self) -> float:
        """
        For now, divide total route cost by 2 as the edge weight is now computed as the sum of the weights of the source
        and target nodes.
        """
        return self.results.result_route_edges["weight"].sum() / 2

    def get_weight_dijkstra(self, edge: BaseWeightedEdgeInfo, modifier: float = 0.01) -> float:
        """
        Weight is leading for edges (MCDA), but we want to add a small distance-based cost to prefer routes that are
        closer to the straight line between start and end.
        """
        weight = self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight
        node_1, node_2 = self.cost_surface_graph.get_edge_endpoints_by_index(edge.edge_id)
        edge_line = self.get_linestring(node_1, node_2)
        distance = edge_line.distance(self.results.result_route_guideline) * modifier
        if distance > weight:
            logger.warning("Unexpected situation during routing.")
        return weight + distance

    def get_weight_astar(self, edge: BaseWeightedEdgeInfo) -> float:
        return self.cost_surface_graph.get_edge_data_by_index(edge.edge_id).weight

    def get_estimate_astar(self, node: NodeInfo) -> float:
        node_point = get_hexagon_node_geometry(self.gdf_cost_surface_nodes, node.node_id)
        guideline = shapely.LineString([node_point, shapely.get_point(self.results.result_route_guideline, 1)])

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

    def _get_shortcut_costs(self, line: shapely.LineString, inradius: int) -> list[float]:
        # Multiply each node suitability value by 2, as edge weights are set as the sum of two node suitability values.
        nearby = self.gdf_cost_surface_nodes[self.gdf_cost_surface_nodes.dwithin(line, inradius)]
        costs = nearby.suitability_value.unique()
        # Only look at the intersection when necessary to save resources
        if len(costs) != 1:
            intersected = nearby[nearby.buffer(inradius / 2).intersects(line)]
            costs = intersected.suitability_value.unique()
        return (costs * 2).tolist()

    def straighten_linestring(self) -> tuple[shapely.LineString, list[int]]:
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
        # center to center distance from a neighbouring hexagon
        inradius = get_inradius(self.hexagon_size)
        shortcut_order: list = [self.results.result_route_node_indices[0]]

        # TODO add height
        # create dataframe with: node_order, suit value, height level. use to get segments.
        gdf_crossed_nodes = self.gdf_cost_surface_nodes.loc[self.results.result_route_node_indices].reset_index()
        gdf_crossed_nodes["segment"] = (
            gdf_crossed_nodes["suitability_value"] != gdf_crossed_nodes["suitability_value"].shift()
        ).cumsum()

        # Note segments do not encapsulate pipe rammings as the costs are not on the node.
        for segment in gdf_crossed_nodes["segment"].unique():
            gdf_active_mask = gdf_crossed_nodes[gdf_crossed_nodes["segment"] == segment]
            if len(gdf_active_mask) == 1:
                shortcut_order.append(int(gdf_active_mask.iloc[0]["node_id"]))
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
                # Compute shortcut costs for each line
                series_shortcut_costs = series_forwarded.apply(self._get_shortcut_costs, inradius=inradius)

                # Filter nodes where shortcut costs equal basic_cost
                valid_nodes = gdf_active_mask[series_shortcut_costs.apply(lambda costs: costs == [basic_cost])]

                if valid_nodes.empty:
                    # No valid shortcut found for this part of the segment, move to the next node
                    shortcut_order.append(forwarded_node)
                    gdf_active_mask = gdf_active_mask[1:]
                    start_node = int(gdf_active_mask.iloc[0]["node_id"])
                    forwarded_node = gdf_active_mask.iloc[1]["node_id"] if len(gdf_active_mask) > 1 else end_node

                else:
                    # Pick the last valid node (most nodes skipped)
                    start_node = int(valid_nodes.iloc[-1]["node_id"])
                    shortcut_order.append(start_node)
                    gdf_active_mask = gdf_active_mask[gdf_active_mask.index > valid_nodes.iloc[-1].name]
                    forwarded_node = gdf_active_mask.iloc[0]["node_id"] if len(gdf_active_mask) > 0 else end_node

        shortcut_linestring = shapely.LineString(
            gdf_crossed_nodes[gdf_crossed_nodes["node_id"].isin(shortcut_order)].geometry.to_list()
        )

        logger.info(
            f"Input LineString: {self.results.result_route_linestring.length}. Shortcut LineString: {shortcut_linestring.length}."
        )

        return shortcut_linestring, shortcut_order

    def apply_bezier_curves(
        self,
        min_bend_radius: float,
        samples_per_curve: int = 30,
    ) -> shapely.LineString:
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
        # TODO split and cleanup
        coords = list(self.results.result_route_straightened.coords)
        if len(coords) < 3:
            return self.results.result_route_straightened

        inradius = get_inradius(self.hexagon_size)

        legs = [shapely.LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
        leg_lengths = [leg.length for leg in legs]
        leg_costs = [self._get_shortcut_costs(leg, int(inradius)) for leg in legs]

        new_pieces: list[shapely.LineString] = []
        cursor = shapely.Point(coords[0])

        for i in range(1, len(coords) - 1):
            p_prev = shapely.Point(coords[i - 1])
            p_curr = shapely.Point(coords[i])
            p_next = shapely.Point(coords[i + 1])

            v_in = (p_curr.x - p_prev.x, p_curr.y - p_prev.y)
            v_out = (p_next.x - p_curr.x, p_next.y - p_curr.y)
            alpha = _angle_between(v_in, v_out)

            # Essentially straight, no curve needed.
            if alpha < 1e-3:
                continue

            d_min = min_bend_radius * math.tan(alpha / 2)
            d_max = 0.5 * min(leg_lengths[i - 1], leg_lengths[i])

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

            allowed_costs = set(leg_costs[i - 1]) | set(leg_costs[i])

            # Iteratively shrink d until the curve stays inside allowed cells.
            d = d_max
            bezier_line = None
            for _ in range(10):
                a = _point_along(p_curr, p_prev, d)
                b = _point_along(p_curr, p_next, d)
                bezier_line = _quadratic_bezier(a, p_curr, b, samples_per_curve)
                if self._curve_stays_in_cells(bezier_line, allowed_costs, inradius):
                    break
                if d <= d_min + 1e-6:
                    break
                d = max(d_min, d * 0.5)

            assert bezier_line is not None
            new_pieces.append(shapely.LineString([cursor, shapely.Point(bezier_line.coords[0])]))
            new_pieces.append(bezier_line)
            cursor = shapely.Point(bezier_line.coords[-1])

        new_pieces.append(shapely.LineString([cursor, shapely.Point(coords[-1])]))

        merged = shapely.ops.linemerge(shapely.MultiLineString(new_pieces))
        if isinstance(merged, shapely.MultiLineString):
            # Fallback: concatenate raw coordinates if linemerge could not produce a single line.
            all_coords: list[tuple[float, float]] = []
            for piece in new_pieces:
                pc = list(piece.coords)
                if all_coords and all_coords[-1] == pc[0]:
                    all_coords.extend(pc[1:])
                else:
                    all_coords.extend(pc)
            merged = shapely.LineString(all_coords)

        self.results.result_route_quadratic_bezier = merged
        return merged

    def _curve_stays_in_cells(self, curve: shapely.LineString, allowed: set, inradius: float) -> bool:
        costs = self._get_shortcut_costs(curve, int(inradius))
        return set(costs).issubset(allowed)
