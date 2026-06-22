# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import rustworkx as rx
import rasterio.mask
import shapely
import structlog
import geopandas as gpd

from settings import Config
from utility_route_planner.models.mcda.exceptions import InvalidRasterValues
from utility_route_planner.util.write import write_results_to_geopackage

logger = structlog.get_logger(__name__)


def coordinates_to_array_index(
    x: float, y: float, upper_left_x: float, upper_left_y: float, x_size: float, y_size: float
) -> tuple:
    """
    Calculates the index of the rastercel which intersects with the input coordinates.

    :param x: x coordinate in RD_new EPSG(28992).
    :param y: y coordinate in RD_new EPSG(28992).
    :param upper_left_x: upper left corner x coordinate (RD_New) of the suitability raster.
    :param upper_left_y: upper left corner y coordinate (RD_New) of the suitability raster.
    :param x_size: x cell size in meters as we are using the RD_new coordinate reference system.
    :param y_size: y cell size in meters as we are using the RD_new coordinate reference system.
    :return: tuple containing the rastercel index.
    """
    x_index = int((x - upper_left_x) / x_size)  # We round down because array indices start at 0.
    y_index = int((y - upper_left_y) / y_size)  # Instead of using math.floor, we truncate using int() as y is negative.

    return y_index, x_index  # Note that the order must be y, x because of indexing on the suitability raster.


def array_indices_to_linestring(raster_metadata: tuple, array_indices: list) -> shapely.LineString:
    """
    Create a linestring from the raster by stringing the centroids together for each cost path cell.

    :param raster_metadata: GeoTransform() gdal properties of the clipped raster.
    :param array_indices: list of the cost path.

    :return the cost path as linestring in a geoseries.
    """
    # Get the raster metadata
    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = raster_metadata

    # String the centroids of each raster cell together in the right sequence.
    cost_path_coords = []
    for idx in array_indices:
        # Calculate the coordinates of the rastercell centroids using map algebra
        x = upper_left_x + ((idx[1] * x_size) + (x_size / 2))
        y = upper_left_y - abs(((idx[0] * y_size) + (y_size / 2)))  # subtract from y as we start in upper_left corner.
        cost_path_coords.append((x, y))

    # Create the linestring.
    cost_path = shapely.geometry.LineString(cost_path_coords)

    return cost_path


def align_linestring(linestring: shapely.LineString, cell_size: float) -> shapely.LineString:
    """
    Write the cost path to a linestring. For now, this is a very primitive function.

    :param linestring: the resulting vectorized raster cost path.
    :param cell_size: resolution/cell_size of the raster.

    :return the cost path as simplified linestring.
    """
    aligned_linestring = linestring.simplify(cell_size, preserve_topology=True)

    return aligned_linestring


def get_first_last_point_from_linestring(linestring: shapely.LineString) -> tuple[shapely.Point, shapely.Point]:
    """
    Get the first and last point of a linestring.

    :param linestring: shapely linestring.
    :return: tuple containing the first and last point.
    """
    if isinstance(linestring, shapely.LineString):
        start_end = shapely.get_point(linestring, 0), shapely.get_point(linestring, -1)
    elif isinstance(linestring, shapely.MultiLineString):
        start_end = (
            shapely.get_point(shapely.get_geometry(linestring, 0), 0),
            shapely.get_point(shapely.get_geometry(linestring, -1), -1),
        )
    else:
        raise ValueError("Input is not a valid linestring or multilinestring.")
    return start_end


def get_empty_geodataframe() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        data=None,
        columns=["geometry"],
        geometry="geometry",
        crs=None,
    )


def load_suitability_raster_data(path_raster: Path | str, project_area: shapely.Polygon):
    """
    Read only the intersection of the project area with the large suitability raster from S3 (or local).
    """
    logger.info(f"Loading {path_raster} based on input project area.")

    with rasterio.Env():
        with rasterio.open(path_raster) as src:
            image, transform = rasterio.mask.mask(
                src,
                [project_area],
                all_touched=True,  # Include a pixel in the mask if it touches any of the shapes.
                crop=True,  # Crop result to input project area.
                filled=True,  # Values outside input project area will be set to nodata.
                indexes=1,
            )
            no_data = src.nodata

    if len(image) < 1:
        raise InvalidRasterValues("Unexpected values retrieved from suitability raster. Check project area.")
    # Replace with a negative value which is ignored in LCPA.
    image[image == no_data] = -1
    return image, transform.to_gdal()


def osm_graph_to_gdfs(graph: rx.PyGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if graph.num_nodes() > 0 and graph.num_edges() > 0:
        data = [(node.osm_id, node.node_id, node.geometry) for node in graph.nodes()]
        gdf_nodes = gpd.GeoDataFrame(data, crs=Config.CRS, columns=["osm_id", "node_id", "geometry"])
        gdf_nodes.set_index("node_id", inplace=True, drop=True)

        data = [(edge.edge_id, edge.osm_id, edge.length, edge.geometry) for edge in graph.edges()]  # type: ignore
        gdf_edges = gpd.GeoDataFrame(data, crs=Config.CRS, columns=["edge_id", "osm_id", "length", "geometry"])
        gdf_edges.set_index("edge_id", inplace=True, drop=True)
        return gdf_nodes, gdf_edges
    else:
        logger.warning("There are no edges or nodes. Cannot convert to GeoDataFrames.")
        return get_empty_geodataframe(), get_empty_geodataframe()


def get_angle_between_points(point_a: shapely.Point, point_b: shapely.Point, center_point: shapely.Point) -> float:
    """Calculate the angle between two points with respect to a center point."""
    if center_point.equals(point_a) or center_point.equals(point_b):
        raise ValueError("Center_point must not be equal to point_a or point_b.")

    vector_a = np.array([point_a.x - center_point.x, point_a.y - center_point.y])
    vector_b = np.array([point_b.x - center_point.x, point_b.y - center_point.y])
    cos_theta = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    angle_radians = np.arccos(np.clip(cos_theta, -1, 1))
    cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    angle_deg = np.degrees(angle_radians)
    if cross < 0:
        angle_deg = 360 - angle_deg
    return float(angle_deg)


def extrapolate_point_to_target(
    point_start: shapely.Point, direction_to_extend_to: shapely.Point, distance: float
) -> shapely.LineString:
    """Extend a point to a target point for a given distance."""
    if point_start.equals(direction_to_extend_to):
        raise ValueError("Point_start and point_to_extend must not be the same.")
    if distance <= 0:
        raise ValueError("Distance must be a positive value.")

    dx, dy = direction_to_extend_to.x - point_start.x, direction_to_extend_to.y - point_start.y
    length = np.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    new_x = point_start.x + dx * distance
    new_y = point_start.y + dy * distance
    return shapely.LineString([point_start, (new_x, new_y)])


def extend_linestring_both_ends(line: shapely.LineString, distance: float, debug: bool = False) -> shapely.LineString:
    """Extend a linestring at both ends by a given distance."""
    if distance <= 0:
        raise ValueError("Distance must be a positive value.")

    coords = np.array(line.coords)
    if len(coords) == 2:
        center_start, center_end = np.array(line.centroid.coords)[0], np.array(line.centroid.coords)[0]
    else:
        center_start = coords[1]
        center_end = coords[-2]

    # Start direction: from first to second point
    start_vec = center_start - coords[0]
    start_unit = start_vec / np.linalg.norm(start_vec)
    new_start = coords[0] - start_unit * distance

    # End direction: from last-1 to last point
    end_vec = center_end - coords[-1]
    end_unit = end_vec / np.linalg.norm(end_vec)
    new_end = coords[-1] - end_unit * distance

    if len(coords) == 2:
        result = shapely.LineString([new_start, new_end])
    else:
        result = shapely.LineString(np.vstack([new_start, coords[1:-1], new_end]))

    if debug:
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, line, "original_line", overwrite=True
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, result, "extended_line", overwrite=True
        )

    return result


def split_polygon_by_linestrings(
    polygon_to_split: shapely.Polygon, linestrings_for_splitting: list[shapely.LineString], debug: bool = False
) -> list[shapely.Polygon]:
    """Split a polygon by a list of linestrings into a list of polygons."""
    line_split_collection = [polygon_to_split.boundary, *linestrings_for_splitting]
    merged_lines = shapely.ops.linemerge(line_split_collection)
    border_lines = shapely.ops.unary_union(merged_lines)
    split_polygon = [i for i in shapely.ops.polygonize(border_lines)]
    if debug:
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, polygon_to_split, "polygon_to_split", overwrite=True
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
            shapely.MultiLineString(linestrings_for_splitting),
            "linestrings_for_splitting",
            overwrite=True,
        )

    return split_polygon


def get_endlines_of_multilinestring(multilinestring: shapely.MultiLineString) -> list[shapely.LineString]:
    """
    Get the endlines (last segments) of a multilinestring, oriented away from the shared center point.

    This function assumes a star-like topology (or set of connected lines) where there is a central
    "hub" point shared by the linestrings. It returns the outermost segment of each branch,
    oriented outgoing from the center. This might not work for a multilinestring with multiple "hubs" or loops.

    :param multilinestring: The input MultiLineString geometry.
    :return: A list of LineString geometries representing the end segments.
    """
    if not isinstance(multilinestring, shapely.MultiLineString) or len(multilinestring.geoms) <= 1:
        raise ValueError("Invalid input received, expected a MultiLineString with at least 2 LineStrings")

    # Flatten all start/end points to find the "hub" (most frequent endpoint)
    endpoints = []
    for line in multilinestring.geoms:
        endpoints.append(line.coords[0])
        endpoints.append(line.coords[-1])

    # The hub is the point that connects the lines (appears most frequently)
    counts = Counter(endpoints)
    hub_coords, _ = counts.most_common(1)[0]
    hub_point = shapely.Point(hub_coords)

    result_lines = []
    for line in multilinestring.geoms:
        coords = list(line.coords)

        # Determine orientation: we want to start from the hub
        start_dist = shapely.Point(coords[0]).distance(hub_point)
        end_dist = shapely.Point(coords[-1]).distance(hub_point)

        # Orient away from hub
        if start_dist <= end_dist:
            # Already starts at/near hub
            oriented_coords = coords
        else:
            # Ends at hub, so reverse it
            oriented_coords = coords[::-1]

        # Extract the last segment of the oriented line
        if len(oriented_coords) >= 2:
            segment = shapely.LineString([oriented_coords[-2], oriented_coords[-1]])
            result_lines.append(segment)

    return result_lines


def get_perpendicular_line(
    origin: shapely.Point, direction_point: shapely.Point, distance: float, debug: bool = False
) -> shapely.LineString:
    entrypoint_line_interpolated = extrapolate_point_to_target(origin, direction_point, 0.1)
    perpendicular_line = shapely.affinity.rotate(entrypoint_line_interpolated, 90, origin=origin)
    perpendicular_line_extended = extend_linestring_both_ends(perpendicular_line, distance)
    if debug:
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, perpendicular_line, "pytest_entry_point_line"
        )
    return perpendicular_line_extended
