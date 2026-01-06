# main_demo.py
import inspect
import ezdxf
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from typing import List
from dueffe_cnc_builder import (
    MachineState,
    SingleHeadCoordinates,
    DualHeadCoordinates,
    starting_block,
    end_block,
    rectangle,
    rectangle_dual,
    circle_single,
    circle_dual,
    line_single,
    line_dual,
    polyline_single,
    polyline_dual,
    emit_program,
    save_program,
)
import dueffe_cnc_visualizer
from dueffe_cnc_visualizer import show_interactive


@dataclass
class PathSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    entity_type: str  # "LINE" or "ARC"
    # For Arcs
    center: Tuple[float, float] = (0, 0)
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0


def dxf_to_cnc_single(dxf_path: str, design_name: str) -> List[str]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments: List[PathSegment] = []

    for entity in msp:
        if entity.dxftype() == 'LINE':
            segments.append(PathSegment(
                start=(entity.dxf.start.x, entity.dxf.start.y),
                end=(entity.dxf.end.x, entity.dxf.end.y),
                entity_type="LINE"
            ))
        elif entity.dxftype() == 'CIRCLE':
            # Convert circle to your circle_single format (center + radius)
            # These are usually treated as independent islands
            segments.append(PathSegment(
                start=(entity.dxf.center.x, entity.dxf.center.y),
                end=(entity.dxf.center.x, entity.dxf.center.y),
                entity_type="CIRCLE",
                center=(entity.dxf.center.x, entity.dxf.center.y),
                radius=entity.dxf.radius
            ))
        elif entity.dxftype() == 'ARC':
            segments.append(PathSegment(
                start=(entity.start_point.x, entity.start_point.y),
                end=(entity.end_point.x, entity.end_point.y),
                entity_type="ARC",
                center=(entity.dxf.center.x, entity.dxf.center.y),
                radius=entity.dxf.radius,
                start_angle=entity.dxf.start_angle,
                end_angle=entity.dxf.end_angle
            ))
        elif entity.dxftype() == 'LWPOLYLINE':
            # Explode polylines into lines and arcs
            pts = entity.get_points()
            for i in range(len(pts) - 1):
                segments.append(PathSegment(
                    start=(pts[i][0], pts[i][1]),
                    end=(pts[i + 1][0], pts[i + 1][1]),
                    entity_type="LINE"
                ))

    # --- Optimization: Greedy Nearest Neighbor ---
    optimized_segments = []
    if segments:
        current_pos = (0, 0)
        while segments:
            # Find closest segment start OR end (to allow reversing lines)
            best_idx = -1
            best_dist = float('inf')
            reverse_needed = False

            for i, seg in enumerate(segments):
                # Check distance to start
                d_start = math.hypot(seg.start[0] - current_pos[0], seg.start[1] - current_pos[1])
                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = i
                    reverse_needed = False

                # Check distance to end (only for lines)
                if seg.entity_type == "LINE":
                    d_end = math.hypot(seg.end[0] - current_pos[0], seg.end[1] - current_pos[1])
                    if d_end < best_dist:
                        best_dist = d_end
                        best_idx = i
                        reverse_needed = True

            seg = segments.pop(best_idx)
            if reverse_needed:
                seg.start, seg.end = seg.end, seg.start

            optimized_segments.append(seg)
            current_pos = seg.end

    # --- CNC Generation ---
    state = MachineState()
    program_lines = []
    program_lines.append(starting_block(SingleHeadCoordinates(0, 0), design_name))

    for seg in optimized_segments:
        if seg.entity_type == "LINE":
            program_lines.append(line_single(
                state,
                SingleHeadCoordinates(*seg.start),
                SingleHeadCoordinates(*seg.end),
                lead_mm=0  # Often 0 for internal DXF paths unless specified
            ))
        elif seg.entity_type == "CIRCLE":
            program_lines.append(circle_single(
                state,
                SingleHeadCoordinates(*seg.center),
                seg.radius
            ))
        elif seg.entity_type == "ARC":
            # Note: Your builder currently lacks a generic Arc helper.
            # You would need to add one that calculates the 'a' parameter
            # (sweep angle) from start/end angles.
            pass

    program_lines.append(end_block())
    return program_lines

@dataclass
class RectSpec:
    start: Tuple[float, float]
    width: float
    height: float
    overlap_mm: float = 40
    dual: bool = False

@dataclass
class GridSpec:
    start: Tuple[float, float]          # first circle center (x0, y0)
    nx: int
    ny: int
    dx: float
    dy: float
    radius: float
    snake: bool = True                  # serpentine order

@dataclass
class BedSpec:
    name: str
    rectangles: List[RectSpec]
    grid: GridSpec

    # Dual-head settings (optional)
    second_head_y_offset: Optional[float] = None
    # If you want to constrain dual to inner area instead of full panel:
    dual_max_y: Optional[float] = None  # dual allowed if y + offset <= dual_max_y


def iter_grid_points(grid: GridSpec):
    """Yield (x,y) in either snake or normal scan order."""
    x0, y0 = grid.start
    for row in range(grid.ny):
        y = y0 + row * grid.dy
        xs = [x0 + col * grid.dx for col in range(grid.nx)]
        if grid.snake and (row % 2 == 1):
            xs.reverse()
        for x in xs:
            yield x, y


def build_bed_program(spec: BedSpec) -> List[str]:
    state = MachineState()
    program: List[str] = []

    off = spec.second_head_y_offset

    if off is not None:
        program.append(starting_block(DualHeadCoordinates(0, 0, off), spec.name))
    else:
        program.append(starting_block(SingleHeadCoordinates(0, 0), spec.name))

    def D(x: float, y: float) -> DualHeadCoordinates:
        return DualHeadCoordinates(x, y, y + off)

    # Rectangles
    for r in spec.rectangles:
        if r.dual and off is not None:
            program.append(
                rectangle_dual(
                    state,
                    start=D(r.start[0], r.start[1]),
                    width=r.width,
                    height=r.height,
                    overlap_mm=r.overlap_mm,
                )
            )
        else:
            program.append(
                rectangle(
                    state,
                    start=SingleHeadCoordinates(*r.start),
                    width=r.width,
                    height=r.height,
                    overlap_mm=r.overlap_mm,
                )
            )

    # Circles (auto dual if configured + valid)
    grid = spec.grid
    off = spec.second_head_y_offset
    dual_max_y = spec.dual_max_y

    for x, y in iter_grid_points(grid):
        if off is not None:
            y2 = y + off
            dual_ok = True
            if dual_max_y is not None:
                dual_ok = (y2 <= dual_max_y)

            if dual_ok:
                program.append(
                    circle_dual(state, center=DualHeadCoordinates(x, y, y2), radius=grid.radius)
                )
                continue  # done for this point

        # fallback single
        program.append(circle_single(state, center=SingleHeadCoordinates(x, y), radius=grid.radius))

    program.append(end_block())
    return program

def save_vrp(cnc_path: str, name: str, max_x: float, max_y: float) -> None:
    vrp_path = cnc_path.replace(".CNC", ".VRP")
    vrp = (
        f"V201={max_x}\n"
        f"V202={max_x}\n"
        f"V203={max_y}\n"
        f"V204={max_y}\n"
        "V205=50\n"
        "V206=450\n"
        "V207=0\n"
        f"V208={max_x}\n"
        "V209=10\n"
    )
    save_program(vrp, vrp_path, crlf=True)

def ILVA_80X190_R_V52_SINGLE():
    state = MachineState()

    program_lines: List[str] = []
    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name="ILVA_80X190_R_V52_SINGLE"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=840, height=1940, overlap_mm=40))
    program_lines.append(
        rectangle(state, start=SingleHeadCoordinates(20, 20), width=800, height=1900, overlap_mm=40))

    # 3 bottom single
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(185, 170), radius=22.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(185 + 235, 170), radius=22.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(185 + 235 + 235, 170), radius=22.5))

    # 2 double
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(303 + 235, 370, 370 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(303, 370, 370 + 800), radius=22.5))

    # 3 double
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185, 570, 570 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185 + 235, 570, 570 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185 + 235 + 235, 570, 570 + 800), radius=22.5))

    # 2 double
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(303 + 235, 770, 770 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(303, 770, 770 + 800), radius=22.5))

    # 3 double
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185, 970, 970 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185 + 235, 970, 970 + 800), radius=22.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(185 + 235 + 235, 970, 970 + 800), radius=22.5))

    program_lines.append(end_block())
    return program_lines

def SLEEPWELL_STRIPES_140(version = "V2"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 0, 0.00), b=SingleHeadCoordinates(0 + 105 * 0, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 1, 1120), b=SingleHeadCoordinates(0 + 105 * 1, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 2, 0.00), b=SingleHeadCoordinates(0 + 105 * 2, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 3, 1120), b=SingleHeadCoordinates(0 + 105 * 3, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 4, 0.00), b=SingleHeadCoordinates(0 + 105 * 4, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 5, 1120), b=SingleHeadCoordinates(0 + 105 * 5, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 6, 0.00), b=SingleHeadCoordinates(0 + 105 * 6, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 7, 1120), b=SingleHeadCoordinates(0 + 105 * 7, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 8, 0.00), b=SingleHeadCoordinates(0 + 105 * 8, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 9, 1120), b=SingleHeadCoordinates(0 + 105 * 9, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 10, 0.00), b=SingleHeadCoordinates(0 + 105 * 10, 1120)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 11, 1120), b=SingleHeadCoordinates(0 + 105 * 11, 0.00)))
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 12, 0.00), b=SingleHeadCoordinates(0 + 105 * 12, 1120)))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_160X190(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=1630, height=1930, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=1600, height=1900, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100), width=1400, height=1700, overlap_mm=40))


    FIRST_CIRCLE = (ZERO[0] + 100 + 117.5, ZERO[1] + 100 + 106.25, ZERO[1] + 100 + 106.25 + 850)

    # 1 ROW
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 0, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 1, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 2, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 3, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 4, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 5, FIRST_CIRCLE[1] + 212.5 * 0, FIRST_CIRCLE[2] + 212.5 * 0), radius=12.5))

    # 2 ROW
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 5, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 4, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 3, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 2, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 1, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 0, FIRST_CIRCLE[1] + 212.5 * 1, FIRST_CIRCLE[2] + 212.5 * 1), radius=12.5))

    # 3 ROW
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 0, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 1, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 2, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 3, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 4, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 5, FIRST_CIRCLE[1] + 212.5 * 2, FIRST_CIRCLE[2] + 212.5 * 2), radius=12.5))

    # 4 ROW
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 5, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 4, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 3, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 2, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 1, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))
    program_lines.append(
        circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 233 * 0, FIRST_CIRCLE[1] + 212.5 * 3, FIRST_CIRCLE[2] + 212.5 * 3), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_80X190(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    SECOND_HEAD_OFFSET = 1000
    program_lines.append(starting_block(initial_coordinates=DualHeadCoordinates(0, 0, SECOND_HEAD_OFFSET), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(0, 0, SECOND_HEAD_OFFSET), width=1930, height=830, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(ZERO[0], ZERO[1], ZERO[1] + SECOND_HEAD_OFFSET), width=1900, height=800, overlap_mm=40))
    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100, ZERO[1] + 100 + SECOND_HEAD_OFFSET), width=1700, height=600, overlap_mm=40))


    FIRST_CIRCLE = (ZERO[0] + 100 + 106.25, ZERO[1] + 100 + 100, ZERO[1] + 100 + 100 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 0, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 1, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 2, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 3, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 4, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 5, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 6, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 7, FIRST_CIRCLE[1] + 200 * 0, FIRST_CIRCLE[2] + 200 * 0), radius=12.5))

    # 2 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 7, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 6, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 5, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 4, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 3, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 2, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 1, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 0, FIRST_CIRCLE[1] + 200 * 1, FIRST_CIRCLE[2] + 200 * 1), radius=12.5))

    # 3 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 0, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 1, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 2, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 3, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 4, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 5, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 6, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 212.5 * 7, FIRST_CIRCLE[1] + 200 * 2, FIRST_CIRCLE[2] + 200 * 2), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_80X190_SINGLE(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    CIRCLE_X_SEPARATION = 200
    CIRCLE_Y_SEPARATION = 212.5

    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=830, height=1930, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=800, height=1900, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100), width=600, height=1700, overlap_mm=40))

    SECOND_HEAD_OFFSET = 850
    FIRST_CIRCLE = (ZERO[0] + 100 + 100, ZERO[1] + 100 + 106.25, ZERO[1] + 100 + 106.25 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_160X200(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    SECOND_HEAD_OFFSET = 699
    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=2030, height=1630, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=2000, height=1600, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100), width=1800, height=1400, overlap_mm=40))


    FIRST_CIRCLE = (ZERO[0] + 100 + 112.5, ZERO[1] + 100 + 117.5, ZERO[1] + 100 + 117.5 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 233 * 0, FIRST_CIRCLE[2] + 233 * 0), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 233 * 1, FIRST_CIRCLE[2] + 233 * 1), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 233 * 2, FIRST_CIRCLE[2] + 233 * 2), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_90X200(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    SECOND_HEAD_OFFSET = 1000
    CIRCLE_X_SEPARATION = 225
    CIRCLE_Y_SEPARATION = 230

    program_lines.append(starting_block(initial_coordinates=DualHeadCoordinates(0, 0, SECOND_HEAD_OFFSET), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(0, 0, SECOND_HEAD_OFFSET), width=2030, height=930, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(ZERO[0], ZERO[1], ZERO[1] + SECOND_HEAD_OFFSET), width=2000, height=900, overlap_mm=40))
    program_lines.append(rectangle_dual(state, start=DualHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100, ZERO[1] + 100 + SECOND_HEAD_OFFSET), width=1800, height=700, overlap_mm=40))

    FIRST_CIRCLE = (ZERO[0] + 100 + 112.5, ZERO[1] + 100 + 120, ZERO[1] + 100 + 120 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 3, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 4, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 5, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 6, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 7, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 7, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 6, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 5, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 4, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 3, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 3, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 4, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 5, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 6, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 7, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_90X200_SINGLE(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    CIRCLE_X_SEPARATION = 230
    CIRCLE_Y_SEPARATION = 225

    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=930, height=2030, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=900, height=2000, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100), width=700, height=1800, overlap_mm=40))

    SECOND_HEAD_OFFSET = 900
    FIRST_CIRCLE = (ZERO[0] + 100 + 120, ZERO[1] + 100 + 112.5, ZERO[1] + 100 + 112.5 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 0, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 0), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 1, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 1), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 2, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 2), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 2, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 1, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + CIRCLE_X_SEPARATION * 0, FIRST_CIRCLE[1] + CIRCLE_Y_SEPARATION * 3, FIRST_CIRCLE[2] + CIRCLE_Y_SEPARATION * 3), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_140X200(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    program_lines: List[str] = []
    SECOND_HEAD_OFFSET = 480
    program_lines.append(starting_block(initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=2030, height=1430, overlap_mm=40))

    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=2000, height=1400, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0] + 100, ZERO[1] + 100), width=1800, height=1200, overlap_mm=40))


    FIRST_CIRCLE = (ZERO[0] + 100 + 112.5, ZERO[1] + 100 + 120, ZERO[1] + 100 + 120 + SECOND_HEAD_OFFSET)

    # 1 ROW
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 240 * 0), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 240 * 1, FIRST_CIRCLE[2] + 240 * 1), radius=12.5))

    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 0, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 1, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 2, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 3, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 4, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 5, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 6, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))
    program_lines.append(circle_dual(state, center=DualHeadCoordinates(FIRST_CIRCLE[0] + 225 * 7, FIRST_CIRCLE[1] + 240 * 2, FIRST_CIRCLE[2] + 240 * 2), radius=12.5))

    program_lines.append(end_block())
    return program_lines, name

def KINSO_80X190(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1930, 830, dual=True),
            RectSpec(ZERO, 1900, 800, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1700, 600, dual=True),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 85, ZERO[1] + 100 + 75),
            nx=10, ny=4,
            dx=170, dy=150,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=1000,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_80X190_SINGLE(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 830, 1930),
            RectSpec(ZERO, 800, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 600, 1700),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 75, ZERO[1] + 100 + 85),
            nx=4, ny=5,
            dx=150, dy=170,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_90X200(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 2030, 930, dual=True),
            RectSpec(ZERO, 2000, 900, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1800, 700, dual=True),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 90, ZERO[1] + 100 + 87),
            nx=10, ny=4,
            dx=180, dy=175,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=1000,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_90X200_SINGLE(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 930, 2030),
            RectSpec(ZERO, 900, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 700, 1800),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=4, ny=5,
            dx=175, dy=180,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_120X190(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1230, 1930),
            RectSpec(ZERO, 1200, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1000, 1700),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 82.5, ZERO[1] + 100 + 85),
            nx=6, ny=5,
            dx=167, dy=170,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_160X190(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 1930),
            RectSpec(ZERO, 1600, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1700),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87.5, ZERO[1] + 100 + 85),
            nx=8, ny=5,
            dx=175, dy=170,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_140X200(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1430, 2030),
            RectSpec(ZERO, 1400, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1200, 1800),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=7, ny=5,
            dx=175, dy=180,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_160X200(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 2030),
            RectSpec(ZERO, 1600, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1800),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=8, ny=5,
            dx=175, dy=180,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_180X200(version="V1"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1830, 2030),
            RectSpec(ZERO, 1800, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1600, 1800),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 88, ZERO[1] + 100 + 90),
            nx=9, ny=5,
            dx=178, dy=180,
            radius=10,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

if __name__ == "__main__":
    program, name = SLEEPWELL_STRIPES_140()
    model_name = name.rsplit("_", 1)[0]
    cnc = emit_program(program, crlf=False)
    save_program(cnc, f"outputs/{model_name}/{name}.CNC", crlf=True)
    x_max, y_max = show_interactive(f"outputs/{model_name}/{name}.CNC")
    save_vrp(f"outputs/{model_name}/{name}.CNC", name, int(x_max), int(y_max))

