#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import enum
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

import shapely


# TODO: do we want to only set the node_id on the node objects and move all other parts to dataframe(s)?
@dataclass
class NodeInfo(ABC):
    node_id: int = field(init=False)

    def set_node_id(self, node_id: int):
        self.node_id = node_id


@dataclass
class HexagonNodeInfo(NodeInfo):
    weight: int


@dataclass
class OSMNodeInfo(NodeInfo):
    osm_id: int
    geometry: shapely.Point


@dataclass
class BaseEdgeInfo:
    edge_id: int = field(init=False)
    length: float = field(init=False)
    geometry: shapely.LineString

    def __post_init__(self):
        self.length = round(self.geometry.length, 2)

    def set_edge_id(self, edge_id: int):
        self.edge_id = edge_id


@dataclass
class BaseWeightedEdgeInfo(BaseEdgeInfo):
    weight: int


@dataclass
class OSMEdgeInfo(BaseEdgeInfo):
    osm_id: int


@dataclass
class HexagonConnectionEdgeInfo(BaseWeightedEdgeInfo):
    connects_height_levels: (
        bool  # always True when this type of edge is used, but useful for debugging to make explicit
    )
    height_level: Optional[int] = None  # Only the non-main height level gets assigned explicitly.


class PipeRammingOrigin(enum.StrEnum):
    """Helps to identify the creator of the extra edge. Refers to the process that created the edge."""

    JUNCTION = enum.auto()
    STREET_SEGMENT = enum.auto()


@dataclass
class PipeRammingEdgeInfo(BaseWeightedEdgeInfo):
    osm_id_junction: int | None
    segment_group: int
    origin: PipeRammingOrigin


hexagon_edge_info = namedtuple("hexagon_edge_info", "edge_id weight")
