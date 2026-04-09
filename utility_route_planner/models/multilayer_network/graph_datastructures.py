#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import enum
from dataclasses import dataclass, field
from typing import Optional

import shapely


@dataclass
class OSMNodeInfo:
    node_id: int = field(init=False)
    osm_id: int
    geometry: shapely.Point

    def set_node_id(self, node_id: int):
        self.node_id = node_id


@dataclass
class BaseEdgeInfo:
    edge_id: int = field(init=False)

    def set_edge_id(self, edge_id: int):
        self.edge_id = edge_id


@dataclass
class HexagonEdgeInfo(BaseEdgeInfo):
    weight: int


@dataclass
class BaseGeometryEdgeInfo(BaseEdgeInfo):
    length: float = field(init=False)
    geometry: shapely.LineString

    def __post_init__(self):
        self.length = round(self.geometry.length, 2)


@dataclass
class OSMEdgeInfo(BaseGeometryEdgeInfo):
    osm_id: int


@dataclass
class HexagonConnectionEdgeInfo(BaseGeometryEdgeInfo):
    weight: int
    connects_height_levels: (
        bool  # always True when this type of edge is used, but useful for debugging to make explicit
    )
    height_level: Optional[int] = None  # Only the non-main height level gets assigned explicitly.


class PipeRammingOrigin(enum.StrEnum):
    """Helps to identify the creator of the extra edge. Refers to the process that created the edge."""

    JUNCTION = enum.auto()
    STREET_SEGMENT = enum.auto()


@dataclass
class PipeRammingEdgeInfo(BaseGeometryEdgeInfo):
    weight: float
    osm_id_junction: int | None
    segment_group: int
    origin: PipeRammingOrigin
