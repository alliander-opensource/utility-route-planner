# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import shapely
import structlog
import rustworkx as rx
import itertools

from settings import Config
import geopandas as gpd

from utility_route_planner.models.multilayer_network.graph_datastructures import PipeRammingEdgeInfo, PipeRammingOrigin
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.util.geo_utilities import (
    osm_graph_to_gdfs,
    get_empty_geodataframe,
    get_angle_between_points,
    extend_linestring_towards_point,
    extend_linestring_from_centroid,
    split_polygon_by_linestrings,
    extend_linestring_both_ends,
)
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


class GetPotentialPipeRammingCrossings:
    def __init__(
        self,
        osm_graph: rx.PyGraph,
        cost_surface_graph: rx.PyGraph,
        debug: bool = False,
    ):
        self.osm_graph = osm_graph
        self.osm_nodes, self.osm_edges = osm_graph_to_gdfs(osm_graph)
        self.cost_surface_graph = cost_surface_graph
        # TODO discuss adding the properties (BGT elements) to the hexagon nodes. That way you can force it to only include sidewalks, ignoring the suitability value.
        self.cost_surface_nodes = convert_hexagon_graph_to_gdfs(self.cost_surface_graph, edges=False)
        self.junctions_of_interests = get_empty_geodataframe()
        # Minimum length of a street segment to be considered for adding pipe ramming crossings.
        self.threshold_edge_length_crossing_m = 30
        # Maximum/minimum length possible of a pipe ramming crossings.
        self.max_pipe_ramming_length_m = 15
        self.min_pipe_ramming_length_m = 3
        # Cost surface value below which we consider a crossing suitable.
        self.suitability_value_crossing_threshold = 10
        # Cost surface value above which we consider unsuitable for crossing.
        self.suitability_value_obstacles_threshold = 76
        self.hexagon_size = Config.HEXAGON_SIZE
        # Debugging options
        self.debug = debug
        self.out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
        self.plot_crossings = False

        # Do some monkey checks on input.
        if self.threshold_edge_length_crossing_m <= self.max_pipe_ramming_length_m:
            raise ValueError(
                "The threshold_edge_length_crossing_m should be larger than the max_pipe_ramming_length_m for segment pipe ramming."
            )
        if not self.max_pipe_ramming_length_m > self.min_pipe_ramming_length_m > 0:
            raise ValueError("The max_pipe_ramming_length_m should be larger than the min_pipe_ramming_length_m > 0.")
        if not self.cost_surface_graph.num_nodes() and self.cost_surface_graph.num_edges():
            raise ValueError("The cost surface graph appears to be empty.")
        if not self.osm_graph.num_nodes() and self.osm_graph.num_edges():
            raise ValueError("The OSM graph appears to be empty.")

    def get_crossings(self):
        """
        Get the crossings for the pipe ramming process.

        We want a road crossing for each segment longer than a certain length.
        For each street segment (set of edges connected by nodes which only have 2 edges) check:
        - Only if crossing the road is expensive (asphalt).
        - If we have enough space between houses somewhere along that edge.
        - Find two cells on the cost-surface which have a low enough cost (sidewalk / berm).
        - Add edges between the two cells with a cost that is lower than just crossing the road.
        Start by checking junctions (nodes with more than 2 edges).
        After this, check the remaining street segments and split them if they are long enough.
        """
        logger.info("Finding road crossings.")

        # Group the edges into street segments between junctions (node degree > 2).
        self.create_street_segment_groups()

        crossing_collection = []
        # Finds crossings (parallel to the edge!) for junctions.
        # self.prepare_junction_crossings()
        # for idx, (node_id, junction) in enumerate(self.junctions_of_interests.iterrows(), 1):
        #     if idx % 10 == 0 or idx == 1:
        #         logger.info(f"Processing junction {idx}/{len(self.junctions_of_interests)}: node_id={node_id}")
        #     crossing = self.get_crossing_for_junction(node_id, junction.osm_id, junction.geometry, junction.degree)
        #     if len(crossing):
        #         crossing_collection.extend(crossing)
        #     else:
        #         logger.warning(f"No crossings found for junction {node_id}.")

        # Find crossings (perpendicular to the edge!) for larger street segments.
        merged_segments_of_interest = self.prepare_segment_crossings()
        for idx, (segment_group, segment) in enumerate(merged_segments_of_interest.iterrows(), 1):
            if idx % 10 == 0 or idx == 1:
                logger.info(
                    f"Processing segment {idx}/{len(merged_segments_of_interest)}: segment group={segment_group}"
                )
            try:
                crossing = self.get_crossings_per_segment(segment_group, segment.geometry)
                if len(crossing):
                    crossing_collection.extend(crossing)
                else:
                    logger.warning(f"No crossings found for segment group {segment_group}.")
            except Exception as e:
                logger.error(f"An error occurred for segment group {segment_group}: {e}")
        # Add all crossings
        self.add_crossings_to_graph(crossing_collection)

        logger.info(f"Found and added {len(crossing_collection)} crossings.")
        return crossing_collection

    def add_crossings_to_graph(self, crossing_collection: list):
        """Add the crossings to the cost surface graph and set the edge ids."""
        edge_ids = self.cost_surface_graph.add_edges_from(crossing_collection)
        [edge_info[2].set_edge_id(edge_id) for edge_id, edge_info in zip(edge_ids, crossing_collection)]
        if self.plot_crossings:
            print("stahp")

    def create_street_segment_groups(self):
        """
        Similar function: https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.simplification.simplify_graph
        Publication: https://onlinelibrary.wiley.com/doi/10.1111/tgis.70037
        """
        # Group the edges which are connected by nodes with only 2 edges. We refer to this as a street segment.
        node_degree = {i: self.osm_graph.degree(i) for i in self.osm_graph.node_indices()}
        # Initialize all edges with a unique group number, then start merging adjacent edges.
        self.osm_edges["group"] = pd.Series(range(len(self.osm_edges)), index=self.osm_edges.index)

        seen_nodes = set()
        for edge_group_nr, (node_id, degree) in enumerate(node_degree.items(), start=len(self.osm_edges) + 1):
            if degree != 2 and node_id not in seen_nodes:
                continue

            # Get the complete street segment to merge.
            edges_to_group = []
            nodes_to_check = [node_id]
            while nodes_to_check:
                for node_id_2 in nodes_to_check:
                    if node_degree[node_id_2] == 2 and node_id_2 not in seen_nodes:
                        adjacent = self.osm_graph.adj(node_id_2)
                        edges_to_group.extend([edge.edge_id for edge in adjacent.values()])
                        nodes_to_check.extend(list(adjacent.keys()))

                    nodes_to_check.remove(node_id_2)
                    seen_nodes.add(node_id_2)

            self.osm_edges.loc[edges_to_group, "group"] = edge_group_nr

        logger.info(f"{len(self.osm_edges)} edges were grouped into {self.osm_edges['group'].nunique()} segments.")
        # Optionally, we can simplify the graph object by removing the nodes and merging the edges to 1 EdgeInfo.

        if self.debug:
            # Plot the basics, OSM + cost surface
            prefix = "pytest_1_"

            write_results_to_geopackage(self.out, self.osm_nodes, f"{prefix}osm_nodes")
            write_results_to_geopackage(self.out, self.osm_edges, f"{prefix}osm_edges")

            write_results_to_geopackage(self.out, self.cost_surface_nodes, f"{prefix}cost_surface_nodes")
            cost_surface_nodes_filtered = self.cost_surface_nodes[
                self.cost_surface_nodes["suitability_value"] < self.suitability_value_crossing_threshold
            ]
            write_results_to_geopackage(self.out, cost_surface_nodes_filtered, f"{prefix}cost_surface_filtered")

    def prepare_junction_crossings(self):
        node_degree = {i: self.osm_graph.degree(i) for i in self.osm_graph.node_indices()}
        self.osm_nodes["degree"] = pd.Series(node_degree, index=self.osm_nodes.index, dtype=int)
        junctions = self.osm_nodes[self.osm_nodes["degree"] > 2]
        if len(junctions) == 0:
            logger.warning("No junctions found to consider for pipe ramming.")
            return get_empty_geodataframe()
        else:
            logger.info(f"Found {len(junctions)} junctions to consider for pipe ramming.")

        # Determine the area around the junctions where we can ram pipes.
        junctions["geometry"] = junctions.buffer(self.max_pipe_ramming_length_m + 1)

        self.junctions_of_interests = junctions

        if self.debug:
            prefix = "pytest_2_"
            write_results_to_geopackage(self.out, junctions, f"{prefix}osm_junction_areas")

    def prepare_segment_crossings(self) -> gpd.GeoDataFrame:
        """
        Identify the segments which are potentially interesting for adding crossings.

        # TODO-improvement: check for shorter segments where there was no junction crossing found.
        """
        # Get road crossings for only long segments.
        merged_segments = self.osm_edges.dissolve(by="group")
        merged_segments["length"] = merged_segments.geometry.length
        merged_segments["is_suitable"] = merged_segments["length"] > self.threshold_edge_length_crossing_m * 2
        merged_segments["geometry"] = merged_segments.line_merge()
        if not merged_segments.geom_type.unique() == np.array("LineString"):
            logger.warning(
                "Some segments are still MultiLineStrings, this is unexpected as a street should always be "
                "topologically connected."
            )
        logger.info(
            f"Found {len(merged_segments[merged_segments['is_suitable']])} segments longer than "
            f"{self.threshold_edge_length_crossing_m}m to consider for pipe ramming."
        )

        if self.debug:
            prefix = "pytest_4_"
            write_results_to_geopackage(
                self.out,
                merged_segments,
                f"{prefix}merged_segments_of_interest",
            )

        return merged_segments[merged_segments["is_suitable"]]

    def get_crossing_for_junction(
        self, node_id: int, osm_id: int, junction_area: shapely.Polygon, degree: int, prefix: str = "pytest_3_"
    ):
        # TODO discuss what can be done in bulk and what needs to be done per junction.
        # Create rectangles which simulate potential crossings.
        minx, miny, maxx, maxy = junction_area.bounds
        boxes = [
            shapely.box(x, miny, min(x + self.hexagon_size, maxx), maxy)
            for x in np.arange(minx, maxx, self.hexagon_size)
        ]
        center_outer_point = shapely.Point(self.osm_nodes.loc[node_id].geometry.x, maxy)
        all_rammings = gpd.GeoDataFrame(data=boxes, columns=["geometry"], crs=Config.CRS)
        # TODO filter the two crossings/rectangles closest to the center?
        all_rammings["distance_to_junction_center"] = all_rammings.distance(self.osm_nodes.loc[node_id].geometry)

        # Check for edges which are almost 180 degrees apart, create straight crossings for those.
        adjacent_edges = self.osm_edges.loc[self.osm_graph.incident_edges(node_id)]
        adjacent_edges = adjacent_edges.clip(junction_area)
        adjacent_edges["point_a"] = adjacent_edges["geometry"].apply(lambda line: shapely.Point(line.coords[0]))
        adjacent_edges["point_b"] = adjacent_edges["geometry"].apply(lambda line: shapely.Point(line.coords[1]))
        adjacent_edges["point_inner"] = self.osm_graph.get_node_data(node_id).geometry
        adjacent_edges["point_outer"] = adjacent_edges["geometry"].apply(
            lambda line: shapely.Point(line.coords[1])
            if shapely.Point(line.coords[0]).equals(self.osm_graph.get_node_data(node_id).geometry)
            else shapely.Point(line.coords[0])
        )
        adjacent_edges["group"] = range(len(adjacent_edges))
        # Get angle to the center point of the junction.
        adjacent_edges["degree_grid"] = adjacent_edges.apply(
            lambda row: get_angle_between_points(
                row["point_outer"], center_outer_point, self.osm_nodes.loc[node_id].geometry
            ),
            axis=1,
        )
        adjacent_edges["group_angle"] = adjacent_edges["degree_grid"]
        # Compare angles to each other and group edges which are almost 180 degrees apart (straight lines/streets).
        for idx_edge_1, idx_edge_2 in itertools.combinations(adjacent_edges.index, 2):
            angle_degree = abs(adjacent_edges.loc[idx_edge_1].degree_grid - adjacent_edges.loc[idx_edge_2].degree_grid)
            if 170 <= angle_degree <= 190:
                # We do not cover the scenario that three or more edges are almost 180 degrees apart.
                opposite1 = adjacent_edges.loc[idx_edge_1].degree_grid + 180
                if opposite1 > 360:
                    opposite1 -= 360
                angles_rad = np.deg2rad([adjacent_edges.loc[idx_edge_2].degree_grid, opposite1])
                mean_angle_rad = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
                mean_angle_deg = np.rad2deg(mean_angle_rad) % 360

                adjacent_edges.at[idx_edge_2, "group"] = adjacent_edges.at[idx_edge_1, "group"]
                adjacent_edges.loc[idx_edge_1, "group_angle"] = mean_angle_deg
                adjacent_edges.loc[idx_edge_2, "group_angle"] = mean_angle_deg

        # Extend the linestrings of the edges outwards from the junction node prior to splitting
        adjacent_edges["extended"] = adjacent_edges.apply(
            lambda row: extend_linestring_towards_point(
                row["point_inner"],
                row["point_outer"],
                distance=self.max_pipe_ramming_length_m
                * 3,  # Has to be long enough to cross/split the junction area polygon.
            ),
            axis=1,
        )

        # First, split the buffered junction by the osm_edges to create the sides to connect.
        street_sides = split_polygon_by_linestrings(junction_area, adjacent_edges["extended"].to_list())

        # Second, intersect the hexagon_nodes eligible for creating crossings to each created side.
        street_sides = gpd.GeoDataFrame(street_sides, columns=["geometry"], crs=Config.CRS)
        cost_surface_nodes_junction = self.cost_surface_nodes.sjoin(street_sides, how="inner", predicate="intersects")
        cost_surface_nodes_junction.rename(columns={"index_right": "idx_street_side"}, inplace=True)

        # Third, check number of sides created and if there are nodes in each side to connect.
        if not len(street_sides) == degree:
            logger.warning(f"Node {osm_id} has {degree} edges, but {len(street_sides)} polygons were created.")
        if not cost_surface_nodes_junction["idx_street_side"].nunique() == degree:
            logger.warning("Not all street sides have nodes to connect to.")

        unpassable_area, unpassable_area_polygon = self._get_unpassable_area(cost_surface_nodes_junction, prefix)
        cost_surface_nodes_junction["distance_to_junction_center"] = cost_surface_nodes_junction.distance(
            self.osm_nodes.loc[node_id].geometry
        )

        seen = set()
        crossing_collection_polygons = []
        crossing_collection = []
        for idx_edge, edge in adjacent_edges.iterrows():
            if edge.group in seen:
                continue
            all_rammings_rotated_to_edge = all_rammings.copy()
            grid_rotated = all_rammings_rotated_to_edge.rotate(
                360 - edge.group_angle, origin=self.osm_nodes.loc[node_id].geometry
            )
            all_rammings_rotated_to_edge["geometry"] = grid_rotated

            # Clip rotated grid with obstacles to remove unsuitable crossings.
            potential_rammings, grid_with_cost_surface = self._filter_all_rammings(
                cost_surface_nodes_junction, all_rammings_rotated_to_edge, unpassable_area, prefix
            )

            closest_node_pairs = (
                grid_with_cost_surface[grid_with_cost_surface["index_right"].isin(potential_rammings.index)]
                .groupby(["index_right", "idx_street_side"])["distance_to_junction_center_left"]
                .idxmin()
            )
            closest_node_linestrings_filtered, valid_rammings = self._filter_rammings_on_execution_space(
                potential_rammings, closest_node_pairs, grid_with_cost_surface, unpassable_area_polygon, prefix
            )

            # Get the one closest to the center point of the junction
            best_crossings = valid_rammings.sort_values("distance_to_junction_center").drop_duplicates(
                subset="combinations", keep="first"
            )

            # Used only for plotting the debug polygons
            best_crossings["group"] = edge.group
            crossing_collection_polygons.append(best_crossings)

            crossing_collection.extend(
                self._create_crossing_selection_to_add(
                    best_crossings,
                    closest_node_linestrings_filtered,
                    closest_node_pairs,
                    PipeRammingOrigin.STREET_SEGMENT,
                    edge.group,
                    node_id=node_id,
                    prefix=prefix,
                )
            )

            seen.add(edge.group)

        if self.debug:
            write_results_to_geopackage(self.out, street_sides, f"{prefix}street_sides")
            write_results_to_geopackage(
                self.out,
                adjacent_edges[["osm_id", "length", "group", "degree_grid", "geometry"]],
                f"{prefix}junction_adjacent_edges",
            )
            write_results_to_geopackage(
                self.out,
                adjacent_edges[["osm_id", "length", "group", "point_outer"]].set_geometry("point_outer"),
                f"{prefix}junction_adjacent_outer_point",
            )
            write_results_to_geopackage(
                self.out, pd.concat(crossing_collection_polygons, ignore_index=True), f"{prefix}best_crossings_polygons"
            )

        return crossing_collection

    def get_crossings_per_segment(
        self,
        segment_group: int,
        segment_geometry: shapely.LineString,
        prefix: str = "pytest_5_",
    ):
        """
        Create perpendicular crossings for long street segments when there are no obstacles in the way.
        """
        logger.info(f"Finding crossings in street segment: {segment_group}.")

        # Determine points per segment where crossings can be added. Skip the first and last meters as this is near a
        # junction and handled separately.
        intervals = np.linspace(
            self.max_pipe_ramming_length_m,
            segment_geometry.length - self.max_pipe_ramming_length_m,
            int(
                (segment_geometry.length - self.max_pipe_ramming_length_m - self.max_pipe_ramming_length_m)
                // self.threshold_edge_length_crossing_m
            )
            + 1,
            endpoint=True,
        )
        crossing_points = shapely.MultiPoint([segment_geometry.interpolate(dist) for dist in intervals])

        # Subset the cost surface nodes to those within the area of interest (buffered street).
        segment_geometry_area = segment_geometry.buffer(self.max_pipe_ramming_length_m, cap_style="flat")
        sides = shapely.ops.split(segment_geometry_area, segment_geometry)
        if not int(shapely.get_num_geometries(sides)) == 2:
            logger.warning(f"Splitting the segment area for {segment_group} did not result in exactly two sides.")
            # TODO cleanup
            sides = shapely.ops.split(segment_geometry_area, extend_linestring_both_ends(segment_geometry, 1))
            if not int(shapely.get_num_geometries(sides)) == 2:
                return []

        gdf_street_sides = gpd.GeoDataFrame(geometry=[i for i in sides.geoms], crs=Config.CRS)
        cost_surface_nodes_segment = self.cost_surface_nodes.sjoin(
            gdf_street_sides, how="inner", predicate="intersects"
        )
        cost_surface_nodes_segment.rename(columns={"index_right": "idx_street_side"}, inplace=True)
        # Remove nodes near junctions to prevent overlapping crossings.
        cost_surface_nodes_segment = cost_surface_nodes_segment.drop(
            cost_surface_nodes_segment.sjoin(self.junctions_of_interests, how="inner", predicate="intersects").index
        )

        # Buffer each linestring in the larger linestring separately, rotated them 90 degrees and create rectangles.
        all_ramming_rectangles = []
        for street in list(self.osm_edges[self.osm_edges["group"] == segment_group].geometry):
            street_copy = street
            if street.length <= self.hexagon_size:
                logger.warning(f"Street segment {segment_group} appears to be very short, skipping.")
                continue
            if street.length < self.max_pipe_ramming_length_m * 2:
                street = extend_linestring_from_centroid(street, self.max_pipe_ramming_length_m * 2 + 1)
            street_rotated = shapely.affinity.rotate(street, 90, origin="centroid")
            interval = np.linspace(0, street_rotated.length / 2, int((street_rotated.length / 2) // 1), endpoint=False)[
                1:
            ]
            linestrings = (
                [(street_rotated.offset_curve(i)) for i in interval]
                + [(street_rotated.offset_curve(-i)) for i in interval]
                + [street_rotated]
            )
            # Split the buffered street side with the linestrings to create potential crossing rectangles.
            all_ramming_rectangles.extend(
                split_polygon_by_linestrings(
                    street_copy.buffer(self.max_pipe_ramming_length_m, cap_style="flat"), linestrings
                )
            )

        # Assign potential crossings to the crossing points.
        all_rammings = gpd.GeoDataFrame(geometry=all_ramming_rectangles, crs=Config.CRS)
        gdf_crossing_points = (
            gpd.GeoDataFrame(geometry=gpd.GeoSeries(crossing_points), crs=Config.CRS).explode().reset_index(drop=True)
        )
        all_rammings = all_rammings.sjoin_nearest(
            gdf_crossing_points, rsuffix="nearest_crossing", distance_col="distance_to_crossing"
        )

        # Clip the potential crossings with obstacles to remove unsuitable crossings.
        unpassable_area, unpassable_area_polygon = self._get_unpassable_area(cost_surface_nodes_segment, prefix)
        potential_rammings, grid_with_cost_surface = self._filter_all_rammings(
            cost_surface_nodes_segment, all_rammings, unpassable_area, prefix
        )

        grid_with_cost_surface = grid_with_cost_surface[
            grid_with_cost_surface["index_right"].isin(potential_rammings.index)
        ]
        grid_with_cost_surface["distance_to_street"] = grid_with_cost_surface.distance(segment_geometry)
        closest_node_pairs = (
            grid_with_cost_surface[grid_with_cost_surface["index_right"].isin(potential_rammings.index)]
            .groupby(["index_right", "idx_street_side"])["distance_to_street"]
            .idxmin()
        )
        closest_node_linestrings_filtered, valid_rammings = self._filter_rammings_on_execution_space(
            potential_rammings, closest_node_pairs, grid_with_cost_surface, unpassable_area_polygon, prefix
        )

        selected_rammings = valid_rammings.loc[
            valid_rammings.groupby("index_nearest_crossing")["distance_to_crossing"].idxmin()
        ]

        crossings_to_add = self._create_crossing_selection_to_add(
            selected_rammings,
            closest_node_linestrings_filtered,
            closest_node_pairs,
            segment_group,
            PipeRammingOrigin.STREET_SEGMENT,
            prefix=prefix,
        )

        if self.debug:
            write_results_to_geopackage(self.out, segment_geometry, f"{prefix}segment_of_interest")
            write_results_to_geopackage(self.out, gdf_street_sides, f"{prefix}street_sides")
            write_results_to_geopackage(self.out, crossing_points, f"{prefix}crossing_points")
            write_results_to_geopackage(self.out, cost_surface_nodes_segment, f"{prefix}cost_surface_nodes_segment")
            write_results_to_geopackage(self.out, selected_rammings, f"{prefix}best_rammings")

        return crossings_to_add

    def _filter_all_rammings(
        self,
        cost_surface_nodes_junction: gpd.GeoDataFrame,
        all_rammings: gpd.GeoDataFrame,
        unpassable_area: gpd.GeoSeries,
        prefix: str = "",
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        rammings_without_obstacles = all_rammings.overlay(  # TODO this needs to be sped up
            gpd.GeoDataFrame(geometry=unpassable_area, crs=28992), how="difference"
        )
        rammings_without_obstacles = rammings_without_obstacles.explode().reset_index(drop=True)
        # Filter rammings which intersect with at least 1 pair of suitable nodes on either side of the street.
        rammings_intersecting_suitable_nodes = cost_surface_nodes_junction[
            cost_surface_nodes_junction["suitability_value"] <= self.suitability_value_crossing_threshold
        ].sjoin(rammings_without_obstacles, predicate="intersects", how="left")
        potential = rammings_intersecting_suitable_nodes.groupby(by="index_right")["idx_street_side"].nunique()
        combinations = (
            rammings_intersecting_suitable_nodes.groupby(by="index_right")["idx_street_side"]
            .unique()
            .apply(lambda x: tuple(sorted(x)))
        )
        filters = pd.DataFrame({"potential": potential, "combinations": combinations})
        potential_rammings = rammings_without_obstacles.join(filters[filters.potential > 1], how="right")

        if self.debug:
            write_results_to_geopackage(self.out, all_rammings, f"{prefix}rammings_all")
            write_results_to_geopackage(self.out, rammings_without_obstacles, f"{prefix}rammings_without_obstacles")
            write_results_to_geopackage(self.out, potential_rammings, f"{prefix}rammings_potential")

        return potential_rammings, rammings_intersecting_suitable_nodes

    def _filter_rammings_on_execution_space(
        self,
        all_crossings: gpd.GeoDataFrame,
        closest_node_pairs: pd.Series,
        grid_with_cost_surface: gpd.GeoDataFrame,
        unpassable_area_polygon: shapely.MultiPolygon | shapely.Polygon,
        prefix: str = "",
    ) -> tuple[gpd.GeoSeries, gpd.GeoDataFrame]:
        pairs = grid_with_cost_surface.loc[closest_node_pairs]
        # TODO it can crash here
        #  Check minimal distance, discard short and long crossings.
        closest_node_linestrings = pairs.groupby("index_right")["geometry"].apply(
            lambda points: shapely.LineString(points)
        )
        closest_node_linestrings_filtered = closest_node_linestrings[
            (closest_node_linestrings.length >= self.min_pipe_ramming_length_m)
            & (closest_node_linestrings.length <= self.max_pipe_ramming_length_m)
        ]
        #  Check if there is enough space in either direction for a ramming.
        side_1 = closest_node_linestrings_filtered.apply(
            lambda line: shapely.affinity.rotate(line, 180, origin=line.coords[0])
        )
        side_2 = closest_node_linestrings_filtered.apply(
            lambda line: shapely.affinity.rotate(line, 180, origin=line.coords[1])
        )
        valid_sides = ~side_1.intersects(unpassable_area_polygon) | ~side_2.intersects(unpassable_area_polygon)
        valid_crossings = all_crossings[all_crossings.index.isin(valid_sides[valid_sides].index)]

        if self.debug:
            write_results_to_geopackage(self.out, side_1, f"{prefix}side_1")
            write_results_to_geopackage(self.out, side_2, f"{prefix}side_2")
            write_results_to_geopackage(self.out, closest_node_linestrings, f"{prefix}closest_node_linestrings")
            write_results_to_geopackage(
                self.out, closest_node_linestrings_filtered, f"{prefix}closest_node_linestrings_filtered"
            )
            write_results_to_geopackage(self.out, valid_crossings, f"{prefix}valid_crossings")

        return closest_node_linestrings_filtered, valid_crossings

    def _get_unpassable_area(
        self, cost_surface_nodes: gpd.GeoDataFrame, prefix: str = ""
    ) -> tuple[gpd.GeoSeries, shapely.Polygon]:
        """Determine the unpassable area for crossing."""
        unpassable_area = cost_surface_nodes[
            cost_surface_nodes["suitability_value"] >= self.suitability_value_obstacles_threshold
        ].buffer(self.hexagon_size)
        unpassable_area_polygon = unpassable_area.union_all()
        if self.debug:
            write_results_to_geopackage(self.out, unpassable_area, f"{prefix}unpassable_area")
            write_results_to_geopackage(self.out, unpassable_area_polygon, f"{prefix}unpassable_area_polygon")
        return unpassable_area, unpassable_area_polygon

    def _create_crossing_selection_to_add(
        self,
        crossings_to_add,
        closest_node_linestrings_filtered,
        closest_node_pairs,
        segment_group,
        origin: PipeRammingOrigin,
        node_id=None,
        prefix: str = "",
    ) -> list:
        crossings = []
        for index in crossings_to_add.index:
            weight = rx.dijkstra_shortest_path_lengths(
                self.cost_surface_graph,
                closest_node_pairs[index].iloc[0],
                lambda x: x.weight,
                closest_node_pairs[index].iloc[1],
            )
            crossing_to_add = (
                int(closest_node_pairs[index].iloc[0]),
                int(closest_node_pairs[index].iloc[1]),
                PipeRammingEdgeInfo(
                    osm_id_junction=node_id,
                    segment_group=segment_group,
                    origin=origin,
                    # TODO-discuss: what is the cost of going through the cost surface?
                    weight=int(weight[closest_node_pairs[index].iloc[1]] / 5),
                    length=closest_node_linestrings_filtered[index].length,
                    geometry=closest_node_linestrings_filtered[index],
                ),
            )
            crossings.append(crossing_to_add)

        if self.debug:
            # We have to be a bit creative here because we cant access the geometry due to the edge-id not being set yet.
            write_results_to_geopackage(
                self.out,
                shapely.MultiLineString(
                    [
                        shapely.LineString(
                            [
                                self.cost_surface_graph.get_node_data(i[0]).geometry,
                                self.cost_surface_graph.get_node_data(i[1]).geometry,
                            ]
                        )
                        for i in crossings
                    ]
                ),
                f"{prefix}best_rammings_edge",
            )
        return crossings
