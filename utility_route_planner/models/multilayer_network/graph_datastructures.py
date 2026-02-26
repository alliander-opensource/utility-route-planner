#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import enum
from dataclasses import dataclass, field
from typing import Optional

import shapely


@dataclass
class NodeInfo:
    node_id: int = field(init=False)

    def set_node_id(self, node_id: int):
        self.node_id = node_id


@dataclass
class OSMNodeInfo(NodeInfo):
    osm_id: int
    geometry: shapely.Point


@dataclass
class TempNode:
    node_id: int
    suitability_value: float


@dataclass
class EdgeInfo:
    edge_id: int = field(init=False)
    length: float = field(init=False)
    geometry: shapely.LineString

    def set_edge_id(self, edge_id: int):
        self.edge_id = edge_id

    def __post_init__(self):
        self.length = round(self.geometry.length, 2)


@dataclass
class OSMEdgeInfo(EdgeInfo):
    osm_id: int


@dataclass
class HexagonEdgeInfo(EdgeInfo):
    weight: float
    connects_height_levels: bool = False
    height_level: Optional[int] = None  # Only the non-main height level gets assigned explicitly.


class PipeRammingOrigin(enum.StrEnum):
    """Helps to identify the creator of the extra edge. Refers to the process that created the edge."""

    JUNCTION = enum.auto()
    STREET_SEGMENT = enum.auto()


@dataclass
class PipeRammingEdgeInfo(EdgeInfo):
    osm_id_junction: int | None
    segment_group: int
    weight: float
    origin: PipeRammingOrigin
