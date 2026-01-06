from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Literal, List, Sequence
import math
from pathlib import Path

HeadMode = Literal["none", "single", "dual"]

@dataclass
class MachineState:
    head_mode: HeadMode = "none"


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


def fmt(n: float) -> str:
    # Stable controller-friendly formatting (no scientific notation)
    s = f"{n:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _unit_vec(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
    dx, dy = (x1 - x0), (y1 - y0)
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return (0.0, 0.0)
    return (dx / L, dy / L)

def _mi_xy(x: float, y: float) -> str:
    return f"MI X{fmt(x)}Y{fmt(y)}"

def _mr_xy(x: float, y: float) -> str:
    return f"MR X{fmt(x)}Y{fmt(y)}"

def _mr_xyz(x: float, y: float, z: float) -> str:
    return f"MR X{fmt(x)}Y{fmt(y)}Z{fmt(z)}"


def mr_move_head(c: Coordinates) -> str:
    if isinstance(c, DualHeadCoordinates):
        return f"MR X{fmt(c.x)}Y{fmt(c.y)}Z{fmt(c.z)}"
    return f"MR X{fmt(c.x)}Y{fmt(c.y)}"


def tool_down_single(state: MachineState) -> List[str]:
    out: List[str] = []
    if state.head_mode != "single":
        out.append("CALL ELY")
        state.head_mode = "single"
    out.append("CALL DW11")
    return out


def tool_down_dual(state: MachineState) -> List[str]:
    out: List[str] = []
    if state.head_mode != "dual":
        out.append("CALL ELYZ")
        state.head_mode = "dual"
    out.append("CALL DW13")
    return out


def starting_block(initial_coordinates: Coordinates, design_name: str = "DESIGN_NAME_DIMENSIONS_VERSION") -> str:
    block = f"""BLOCK VA1
VELL 166.67
ACCL 333.33
w195=1
ENDBL
BLOCK VA2
VELL 133.33
ACCL 166.67
w195=2
ENDBL
BLOCK VA3
VELL 100
ACCL 133.33
w195=3
ENDBL
;
PROGRAM LAV1
; DESIGN: {design_name}
ABS X=0
ABS Y=0
CORNER a=333.33
VEL X= 80
VEL Y= 80
ACC X= 30
ACC Y= 30
v990=1
v991=1
CALL INIZIO
LABEL 1
CALL INLAV1
CALL VA1
CALL FLZ"""

    if isinstance(initial_coordinates, DualHeadCoordinates):
        pass
        coords = ";"
        # coords = f"""CALL QLYZ {fmt(initial_coordinates.y)} {fmt(initial_coordinates.z)}"""
    else:
        coords = f"""CALL QLY {fmt(initial_coordinates.y)}"""

    return block + "\n" + coords


def end_block() -> str:
    return """CALL STOFF
MR X=v993 Y=v994
CALL FINLAV1
CALL FINECIC1
IF (w92=1) JUMP 1
CALL FINE
ENDPR
"""


def rectangle(state: MachineState, start: SingleHeadCoordinates, width: float, height: float, overlap_mm: float = 40) -> str:
    x = start.x
    y = start.y
    w = width
    h = height

    lines: List[str] = []
    lines.append(mr_move_head(start))
    lines.extend(tool_down_single(state))

    lines.extend([
        "FREEZE",
        f"MOVI X{fmt(x)}Y{fmt(y + h/2)}",
        f"MI X{fmt(x)}Y{fmt(y + h)}",

        "FREEZE",
        f"MOVI X{fmt(x + w/2)}Y{fmt(y + h)}",
        f"MI X{fmt(x + w)}Y{fmt(y + h)}",

        "FREEZE",
        f"MOVI X{fmt(x + w)}Y{fmt(y + h/2)}",
        f"MI X{fmt(x + w)}Y{fmt(y)}",

        "FREEZE",
        f"MOVI X{fmt(x + w/2)}Y{fmt(y)}",
        f"MI X{fmt(x)}Y{fmt(y)}",
        f"MI X{fmt(x)}Y{fmt(y + overlap_mm)}",
        "CALL UP1",
    ])

    return "\n".join(lines)


def rectangle_dual(state: MachineState,start: DualHeadCoordinates,width: float,height: float,overlap_mm: float = 40) -> str:
    x = start.x
    y = start.y
    z = start.z
    w = width
    h = height

    lines: List[str] = []
    lines.append(f"CALL QLYZ {fmt(y)} {fmt(z)}")
    lines.append(mr_move_head(start))
    lines.extend(tool_down_dual(state))

    lines.extend([
        "FREEZE",
        f"MOVI X{fmt(x)}Y{fmt(y + h / 2)}",
        f"MI X{fmt(x)}Y{fmt(y + h)}",

        "FREEZE",
        f"MOVI X{fmt(x + w / 2)}Y{fmt(y + h)}",
        f"MI X{fmt(x + w)}Y{fmt(y + h)}",

        "FREEZE",
        f"MOVI X{fmt(x + w)}Y{fmt(y + h / 2)}",
        f"MI X{fmt(x + w)}Y{fmt(y)}",

        "FREEZE",
        f"MOVI X{fmt(x + w / 2)}Y{fmt(y)}",
        f"MI X{fmt(x)}Y{fmt(y)}",
        f"MI X{fmt(x)}Y{fmt(y + overlap_mm)}",
        "CALL UP1",
    ])

    return "\n".join(lines)

def polyline_single(
    state: MachineState,
    points: Sequence[SingleHeadCoordinates],
    lead_in_mm: float = 0.0,
    lead_out_mm: float = 0.0,
    lift: bool = True,
    add_semicolon_line: bool = True,
) -> str:
    """
    MI-only polyline, KNAPP-style.

    - Rapid to an optional lead-in point (slightly "inside" the first vertex)
    - Tool down (DW11, with ELY only if switching mode)
    - MI through all vertices
    - Optional lead-out (slightly "inside" the last vertex)
    - Optional UP1

    This produces sequences like:
      MR X...Y...        (lead-in)
      ;
      CALL ELY
      CALL DW11
      MI Xp0Yp0
      MI Xp1Yp1
      ...
      MI X(last-inset)Y(last-inset)
      CALL UP1
    """
    if len(points) < 2:
        raise ValueError("polyline_single needs at least 2 points")

    p0, p1 = points[0], points[1]
    ux0, uy0 = _unit_vec(p0.x, p0.y, p1.x, p1.y)

    # lead-in point is "inside" from p0 towards p1
    start_x = p0.x + ux0 * lead_in_mm
    start_y = p0.y + uy0 * lead_in_mm

    pn_1, pn = points[-2], points[-1]
    uxn, uyn = _unit_vec(pn_1.x, pn_1.y, pn.x, pn.y)

    # lead-out point is "inside" from pn back toward pn_1 (i.e., reverse of last segment)
    end_x = pn.x - uxn * lead_out_mm
    end_y = pn.y - uyn * lead_out_mm

    lines: List[str] = []
    lines.append(_mr_xy(start_x, start_y))
    if add_semicolon_line:
        lines.append(";")
    lines.extend(tool_down_single(state))

    # cut back to the true first vertex, then along the chain
    lines.append(_mi_xy(p0.x, p0.y))
    for p in points[1:]:
        lines.append(_mi_xy(p.x, p.y))

    # lead-out/backoff
    if lead_out_mm > 0:
        lines.append(_mi_xy(end_x, end_y))

    if lift:
        lines.append("CALL UP1")

    return "\n".join(lines)


def line_single(
    state: MachineState,
    a: SingleHeadCoordinates,
    b: SingleHeadCoordinates,
    lead_mm: float = 10.0,
    lift: bool = True,
) -> str:
    """
    Single straight line, KNAPP-style:
      start inside A by lead_mm, cut back to A, to B, back inside by lead_mm, UP1.
    """
    return polyline_single(state, [a, b], lead_in_mm=lead_mm, lead_out_mm=lead_mm, lift=lift)


def polyline_dual(
    state: MachineState,
    points: Sequence[DualHeadCoordinates],
    lead_in_mm: float = 0.0,
    lead_out_mm: float = 0.0,
    lift: bool = True,
    add_semicolon_line: bool = True,
) -> str:
    """
    Dual-head MI-only polyline.

    Conventions (matching your dual circle style):
    - CALL QLYZ y z before the MR
    - MR includes Z (sets the offset reference)
    - MI moves are XY only (same as your ARC usage)
    """
    if len(points) < 2:
        raise ValueError("polyline_dual needs at least 2 points")

    # (Optional but recommended) ensure the dual offset is consistent across the polyline
    offsets = [p.z - p.y for p in points]
    if max(offsets) - min(offsets) > 1e-6:
        raise ValueError("polyline_dual: all points must have same (z - y) offset")

    p0, p1 = points[0], points[1]
    ux0, uy0 = _unit_vec(p0.x, p0.y, p1.x, p1.y)

    start_x = p0.x + ux0 * lead_in_mm
    start_y = p0.y + uy0 * lead_in_mm
    start_z = p0.z  # keep same Z reference

    pn_1, pn = points[-2], points[-1]
    uxn, uyn = _unit_vec(pn_1.x, pn_1.y, pn.x, pn.y)

    end_x = pn.x - uxn * lead_out_mm
    end_y = pn.y - uyn * lead_out_mm

    lines: List[str] = []
    lines.append(f"CALL QLYZ {fmt(p0.y)} {fmt(p0.z)}")
    lines.append(_mr_xyz(start_x, start_y, start_z))
    if add_semicolon_line:
        lines.append(";")
    lines.extend(tool_down_dual(state))

    lines.append(_mi_xy(p0.x, p0.y))
    for p in points[1:]:
        lines.append(_mi_xy(p.x, p.y))

    if lead_out_mm > 0:
        lines.append(_mi_xy(end_x, end_y))

    if lift:
        lines.append("CALL UP1")

    return "\n".join(lines)


def line_dual(
    state: MachineState,
    a: DualHeadCoordinates,
    b: DualHeadCoordinates,
    lead_mm: float = 10.0,
    lift: bool = True,
) -> str:
    """
    Dual straight line, KNAPP-style using MI only.
    """
    return polyline_dual(state, [a, b], lead_in_mm=lead_mm, lead_out_mm=lead_mm, lift=lift)


def circle_dual(state: MachineState, center: DualHeadCoordinates, radius: float) -> str:
    start = DualHeadCoordinates(center.x - radius, center.y, center.z)

    lines: List[str] = []
    lines.append(f"CALL QLYZ {fmt(center.y)} {fmt(center.z)}")
    lines.append(mr_move_head(start))
    lines.append(";")
    lines.extend(tool_down_dual(state))

    lines.extend([
        "FREEZE",
        f"ARC X{fmt(center.x + radius)}Y{fmt(center.y)} a=-180",
        f"ARC X{fmt(center.x)}Y{fmt(center.y + radius)} a=-270",
        "SYNC",
        "CALL UP1",
    ])

    return "\n".join(lines)


def circle_single(state: MachineState, center: SingleHeadCoordinates, radius: float) -> str:
    start = SingleHeadCoordinates(center.x - radius, center.y)

    lines: List[str] = []
    lines.append(mr_move_head(start))
    lines.append(";")
    lines.extend(tool_down_single(state))

    lines.extend([
        "FREEZE",
        f"ARC X{fmt(center.x + radius)}Y{fmt(center.y)} a=-180",
        f"ARC X{fmt(center.x)}Y{fmt(center.y + radius)} a=-270",
        "SYNC",
        "CALL UP1",
    ])

    return "\n".join(lines)

def emit_program(lines: List[str], crlf: bool = True) -> str:
    sep = "\r\n" if crlf else "\n"
    return sep.join(lines) + sep

def save_program(text: str, path: str, crlf: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if crlf:
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")

    with p.open("w", newline="") as f:
        f.write(text)
