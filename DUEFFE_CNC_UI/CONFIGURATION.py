from dataclasses import dataclass
from typing import Union
from enum import Enum, auto

accepted_file_drops = [".cnc", ".dxf"]


@dataclass(frozen=True)
class DualHeadCoordinates:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SingleHeadCoordinates:
    x: float
    y: float

Coordinates = Union[DualHeadCoordinates, SingleHeadCoordinates]

class SewingHeadType(Enum):
    DW11 = auto()
    DW12 = auto()
    DW13 = auto()
    DW14 = auto()
