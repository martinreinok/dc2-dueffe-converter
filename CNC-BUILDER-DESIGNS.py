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
# from dxf_to_cnc import dxf_to_cnc_single
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
    program_lines.append(starting_block(state, SingleHeadCoordinates(0, 0), design_name))

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
class RowSpec:
    xs: Optional[List[float]] = None
    x0: Optional[float] = None
    nx: Optional[int] = None
    dx: Optional[float] = None
    force_single: bool = False
    y: Optional[float] = None
    y_offset: float = 0.0
    y_step: float = 0.0

@dataclass
class RectSpec:
    start: Tuple[float, float]
    width: float
    height: float
    start_offset_y: float = 0
    overlap_mm: float = 40
    dual: bool = False

@dataclass
class GridSpec:
    start: Tuple[float, float]
    radius: float
    snake: bool = True
    nx: int = 0
    ny: int = 0
    dx: float = 0.0
    dy: float = 0.0
    rows: Optional[List[RowSpec]] = None
    y_offset: float = 0.0


@dataclass
class BedSpec:
    name: str
    rectangles: List[RectSpec]
    grid: GridSpec

    custom_circle_sweep: Optional[float] = -270
    custom_second_arc_offset: Optional[tuple] = (0, 0)
    # Dual-head settings (optional)
    second_head_y_offset: Optional[float] = None
    # If you want to constrain dual to inner area instead of full panel:
    dual_max_y: Optional[float] = None  # dual allowed if y + offset <= dual_max_y


def iter_grid_points(grid: GridSpec):
    x0, y0 = grid.start
    base_y0 = y0 + getattr(grid, "y_offset", 0.0)

    if not grid.rows:
        for row in range(grid.ny):
            y = base_y0 + row * grid.dy
            xs = [x0 + col * grid.dx for col in range(grid.nx)]
            reversed_row = bool(grid.snake and (row % 2 == 1))
            if reversed_row:
                xs.reverse()
            for x in xs:
                yield x, y, reversed_row, None
        return
    
    for row_idx, row_spec in enumerate(grid.rows):
        if getattr(row_spec, "y", None) is not None:
            row_base_y = base_y0 + row_spec.y
        else:
            row_base_y = base_y0 + row_idx * grid.dy
        row_base_y += getattr(row_spec, "y_offset", 0.0)
        if row_spec.xs is not None:
            xs = list(row_spec.xs)
        else:
            assert row_spec.x0 is not None and row_spec.nx is not None and row_spec.dx is not None
            xs = [row_spec.x0 + col * row_spec.dx for col in range(row_spec.nx)]
        reversed_row = bool(grid.snake and (row_idx % 2 == 1))
        if reversed_row:
            xs.reverse()
        y_step = getattr(row_spec, "y_step", 0.0)
        for col_idx, x in enumerate(xs):
            y = row_base_y + col_idx * y_step
            yield x, y, reversed_row, row_spec

def will_first_op_be_dual(spec: BedSpec) -> bool:
    off = spec.second_head_y_offset
    if off is None:
        return False

    # 1) rectangles come first
    for r in spec.rectangles:
        return bool(r.dual)  # first rectangle decides

    # 2) otherwise first grid point decides
    x0, y0 = spec.grid.start
    y2 = y0 + off
    if spec.dual_max_y is not None and y2 > spec.dual_max_y:
        return False
    return True

def build_bed_program(spec: BedSpec) -> List[str]:
    state = MachineState()
    program: List[str] = []

    off = spec.second_head_y_offset
    start_dual = will_first_op_be_dual(spec)

    if start_dual and off is not None:
        program.append(starting_block(state, DualHeadCoordinates(0, 0, off), spec.name))
    else:
        program.append(starting_block(state, SingleHeadCoordinates(0, 0), spec.name))

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
                    start_offset_y=r.start_offset_y,
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
                    start_offset_y=r.start_offset_y,
                    overlap_mm=r.overlap_mm,
                )
            )

    # Circles
    grid = spec.grid
    dual_max_y = spec.dual_max_y

    for x, y, reversed_row, row_spec in iter_grid_points(grid):
        # Match your manual behavior:
        # normal row: sweep = -190, reverse=False
        # reversed row: sweep = +190, reverse=True
        sweep = spec.custom_circle_sweep
        reverse = False
        if reversed_row:
            sweep = -sweep if sweep is not None else sweep
            reverse = True

        if row_spec is not None and row_spec.force_single:
            program.append(
                circle_single(
                    state,
                    center=SingleHeadCoordinates(x, y),
                    radius=grid.radius,
                    second_sweep_deg=sweep,
                    second_arc_offset=spec.custom_second_arc_offset,
                    reverse=reverse,
                )
            )
            continue

        if off is not None:
            y2 = y + off
            dual_ok = True
            if dual_max_y is not None:
                dual_ok = (y2 <= dual_max_y)

            if dual_ok:
                program.append(
                    circle_dual(
                        state,
                        center=DualHeadCoordinates(x, y, y2),
                        radius=grid.radius,
                        second_sweep_deg=sweep,
                        second_arc_offset=spec.custom_second_arc_offset,
                        reverse=reverse,
                    )
                )
                continue

        # fallback single
        program.append(
            circle_single(
                state,
                center=SingleHeadCoordinates(x, y),
                radius=grid.radius,
                second_sweep_deg=sweep,
                second_arc_offset=spec.custom_second_arc_offset,
                reverse=reverse,
            )
        )

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

def OLD_PROGRAM_TYPE():
    state = MachineState()

    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name="ILVA_80X190_R_V52_SINGLE"))

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

def CIRCLE_25MM_KINSO(version = "V2"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(circle_single(state, center=SingleHeadCoordinates(0, 0), radius=12.5, second_sweep_deg=-190))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(0, 100), radius=12.5, second_sweep_deg=-190))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(0, 200), radius=12.5, second_sweep_deg=-190))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(0, 300), radius=12.5, second_sweep_deg=-190))

    program_lines.append(end_block())
    return program_lines, name

def CIRCLE_45MM(version = "V1"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(circle_single(state, center=SingleHeadCoordinates(0, 0), radius=22.5))

    program_lines.append(end_block())
    return program_lines, name

def ILVA_80X190_SINGLE(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name="ILVA_80X190_R_V52_SINGLE"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=840, height=1940, overlap_mm=40))
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(20, 20), width=800, height=1900, overlap_mm=40))

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
    return program_lines, name

def ILVA_160X200_V(version = "V3"):
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    ZERO = [15, 15]
    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 2030, overlap_mm=40, start_offset_y=300),
            RectSpec((ZERO[0], ZERO[1]), 1600, 2000, overlap_mm=40, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(0, ZERO[1] + 150 + 50),
            dx=0, dy=200,
            radius=23,
            snake=True,
            rows=[
                RowSpec(x0=ZERO[0] + 110, nx=6, dx=276, force_single=True),
                RowSpec(x0=ZERO[0] + 110 + 138, nx=5, dx=276),
                RowSpec(x0=ZERO[0] + 110, nx=6, dx=276),
                RowSpec(x0=ZERO[0] + 110 + 138, nx=5, dx=276),
                RowSpec(x0=ZERO[0] + 110, nx=6, dx=276),
            ],
            nx=0, ny=0,
        ),
        second_head_y_offset=800,
        dual_max_y=None,
        custom_circle_sweep=-200,
        custom_second_arc_offset=(0, 0),
    )
    return build_bed_program(spec), name

def ILVA_160X200_H(version = "V3"):
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    ZERO = [15, 15]
    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 2030, 1630, overlap_mm=40, start_offset_y=300),
            RectSpec((ZERO[0], ZERO[1]), 2000, 1600, overlap_mm=40, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 150 + 50, ZERO[1] + 110),
            dx=0, dy=276,
            radius=23,
            snake=True,
            rows=[
                RowSpec(x0=ZERO[0] + 150 + 50, nx=5, dx=400),
                RowSpec(x0=ZERO[0] + 150 + 50, nx=5, dx=400),
                RowSpec(x0=ZERO[0] + 150 + 50, nx=5, dx=400),
                RowSpec(x0=ZERO[0] + 150 + 50 + 200, nx=4, dx=400, y_offset=(-1 * 276 * 3) +  138),
                RowSpec(x0=ZERO[0] + 150 + 50 + 200, nx=4, dx=400, y_offset=(-1 * 276 * 3) +  138),
                RowSpec(x0=ZERO[0] + 150 + 50 + 200, nx=4, dx=400, y_offset=(-1 * 276 * 3) + 138, force_single=True),

            ],
            nx=0, ny=0,
        ),
        second_head_y_offset=276*3,
        dual_max_y=None,
        custom_circle_sweep=-200,
        custom_second_arc_offset=(0, 0),
    )
    return build_bed_program(spec), name

def SLEEPWELL_STRIPES_140(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 10), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0 + 105 * 0, 0.00), b=SingleHeadCoordinates(0 + 105 * 0, 1120)))

    START = (200, 0)
    STEP = 105
    HEIGHT = 1120

    for i in range(13):
        x = START[0] + STEP * i

        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(line_single(state, a=SingleHeadCoordinates(START[0] + 200 + 105 * 12, 0.00), b=SingleHeadCoordinates(START[0] + 200 + 105 * 12, 1120)))

    program_lines.append(end_block())
    return program_lines, name

def ESPA_160X190(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 1930),
            RectSpec(ZERO, 1600, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1700, start_offset_y=400),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 117.5, ZERO[1] + 100 + 106.25),
            nx=6, ny=4,
            dx=233, dy=212.5,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_80X190(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1930, 830, dual=True),
            RectSpec(ZERO, 1900, 800, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1700, 600, dual=True, start_offset_y=200)
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 106.25, ZERO[1] + 100 + 100),
            nx=8, ny=3,
            dx=212.5, dy=200,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_80X190_SINGLE(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 830, 1930),
            RectSpec(ZERO, 800, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 600, 1700, start_offset_y=400)
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 100, ZERO[1] + 100 + 106.25),
            nx=3, ny=4,
            dx=200, dy=212.5,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_160X200(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 2030),
            RectSpec(ZERO, 1600, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1800, start_offset_y=400),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 117.5, ZERO[1] + 100 + 112.5),
            nx=6, ny=4,
            dx=233, dy=225,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_90X200(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 2030, 930, dual=True),
            RectSpec(ZERO, 2000, 900, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1800, 700, dual=True, start_offset_y=200)
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 112.5, ZERO[1] + 100 + 120),
            nx=8, ny=3,
            dx=225, dy=230,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=1000,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_90X200_SINGLE(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 930, 2030),
            RectSpec(ZERO, 900, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 700, 1800, start_offset_y=400)
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 120, ZERO[1] + 100 + 112.5),
            nx=3, ny=4,
            dx=230, dy=225,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def ESPA_140X200(version = "V3"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1430, 2030),
            RectSpec(ZERO, 1400, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1200, 1800, start_offset_y=400),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 120, ZERO[1] + 100 + 112.5),
            nx=5, ny=4,
            dx=240, dy=225,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
    )

    return build_bed_program(spec), name

def KINSO_80X190(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1930, 830, dual=True),
            RectSpec(ZERO, 1900, 800, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1700, 600, dual=True, start_offset_y=200),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 85, ZERO[1] + 100 + 75),
            nx=10, ny=4,
            dx=170, dy=150,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_80X190_SINGLE(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 830, 1930),
            RectSpec(ZERO, 800, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 600, 1700, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 75, ZERO[1] + 100 + 85),
            nx=4, ny=5,
            dx=150, dy=170,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_90X200(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 2030, 930, dual=True),
            RectSpec(ZERO, 2000, 900, dual=True),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1800, 700, dual=True, start_offset_y=200),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 90, ZERO[1] + 100 + 87),
            nx=10, ny=4,
            dx=180, dy=175,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=1000,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_90X200_SINGLE(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 930, 2030),
            RectSpec(ZERO, 900, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 700, 1800, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=4, ny=5,
            dx=175, dy=180,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_120X190(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1230, 1930),
            RectSpec(ZERO, 1200, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1000, 1700, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 82.5, ZERO[1] + 100 + 85),
            nx=6, ny=5,
            dx=167, dy=170,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_160X190(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 1930),
            RectSpec(ZERO, 1600, 1900),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1700, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87.5, ZERO[1] + 100 + 85),
            nx=8, ny=5,
            dx=175, dy=170,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=850,
        dual_max_y=None,
        custom_circle_sweep=-190
    )

    return build_bed_program(spec), name

def KINSO_140X200(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1430, 2030),
            RectSpec(ZERO, 1400, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1200, 1800, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=7, ny=5,
            dx=175, dy=180,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_160X200(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1630, 2030),
            RectSpec(ZERO, 1600, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1400, 1800, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 87, ZERO[1] + 100 + 90),
            nx=8, ny=5,
            dx=175, dy=180,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
        custom_circle_sweep=-200
    )

    return build_bed_program(spec), name

def KINSO_180X200(version="V7"):
    ZERO = (15, 15)
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    spec = BedSpec(
        name=name,
        rectangles=[
            RectSpec((0, 0), 1830, 2030),
            RectSpec(ZERO, 1800, 2000),
            RectSpec((ZERO[0] + 100, ZERO[1] + 100), 1600, 1800, start_offset_y=300),
        ],
        grid=GridSpec(
            start=(ZERO[0] + 100 + 88, ZERO[1] + 100 + 90),
            nx=9, ny=5,
            dx=178, dy=180,
            radius=12.5,
            snake=True,
        ),
        second_head_y_offset=900,
        dual_max_y=None,
        custom_circle_sweep=-200

    )

    return build_bed_program(spec), name

def RUTA_120_L(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #

    program_lines.append(line_single(state, a=SingleHeadCoordinates(0, 0), b=SingleHeadCoordinates(0, 1240)))

    START = (351, 0)
    STEP = 256
    HEIGHT = 1240
    WIDTH = 1470

    for i in range(4):
        x = START[0] + STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(line_single(state, a=SingleHeadCoordinates(WIDTH, 1240), b=SingleHeadCoordinates(WIDTH, 0)))

    START = (WIDTH, 964)
    STEP = 256

    for i in range(3):
        y = START[1] - STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(WIDTH, y)
            b = SingleHeadCoordinates(0.0, y)
        else:
            a = SingleHeadCoordinates(0.0, y)
            b = SingleHeadCoordinates(WIDTH, y)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(end_block())
    return program_lines, name

def RUTA_140_L(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #

    program_lines.append(line_single(state, a=SingleHeadCoordinates(0, 0), b=SingleHeadCoordinates(0, 1240)))

    START = (451, 0)
    STEP = 256
    HEIGHT = 1240
    WIDTH = 1670

    for i in range(4):
        x = START[0] + STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(line_single(state, a=SingleHeadCoordinates(WIDTH, 1240), b=SingleHeadCoordinates(WIDTH, 0)))

    START = (WIDTH, 964)
    STEP = 256

    for i in range(3):
        y = START[1] - STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(WIDTH, y)
            b = SingleHeadCoordinates(0.0, y)
        else:
            a = SingleHeadCoordinates(0.0, y)
            b = SingleHeadCoordinates(WIDTH, y)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(end_block())
    return program_lines, name

def RUTA_160_L(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #

    program_lines.append(line_single(state, a=SingleHeadCoordinates(0, 0), b=SingleHeadCoordinates(0, 1240)))

    START = (423, 0)
    STEP = 256
    HEIGHT = 1240
    WIDTH = 1870

    for i in range(5):
        x = START[0] + STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(line_single(state, a=SingleHeadCoordinates(WIDTH, 0), b=SingleHeadCoordinates(WIDTH, 1240)))

    START = (WIDTH, 964)
    STEP = 256
    for i in range(3):
        y = START[1] - STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(WIDTH, y)
            b = SingleHeadCoordinates(0.0, y)
        else:
            a = SingleHeadCoordinates(0.0, y)
            b = SingleHeadCoordinates(WIDTH, y)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(end_block())
    return program_lines, name

def RUTA_180_L(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #

    START = (395, 0)
    STEP = 256
    HEIGHT = 1240
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0, 0), b=SingleHeadCoordinates(0, HEIGHT)))



    for i in range(6):
        x = START[0] + STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    START = (1870, 964)
    STEP = 256
    WIDTH = 2070

    program_lines.append(line_single(state, a=SingleHeadCoordinates(WIDTH, HEIGHT), b=SingleHeadCoordinates(WIDTH, 0)))

    for i in range(3):
        y = START[1] - STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(WIDTH, y)
            b = SingleHeadCoordinates(0.0, y)
        else:
            a = SingleHeadCoordinates(0.0, y)
            b = SingleHeadCoordinates(WIDTH, y)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(end_block())
    return program_lines, name

def RUTA_210_L(version = "V3"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"
    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))
    #

    START = (417, 0)
    STEP = 256
    HEIGHT = 1240
    program_lines.append(line_single(state, a=SingleHeadCoordinates(0, 0), b=SingleHeadCoordinates(0, HEIGHT)))



    for i in range(7):
        x = START[0] + STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(x, HEIGHT)
            b = SingleHeadCoordinates(x, 0.00)
        else:
            a = SingleHeadCoordinates(x, 0.00)
            b = SingleHeadCoordinates(x, HEIGHT)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    START = (1870, 964)
    STEP = 256
    WIDTH = 2370

    program_lines.append(line_single(state, a=SingleHeadCoordinates(WIDTH, 0), b=SingleHeadCoordinates(WIDTH, HEIGHT)))

    for i in range(3):
        y = START[1] - STEP * i
        if i % 2 == 0:
            a = SingleHeadCoordinates(WIDTH, y)
            b = SingleHeadCoordinates(0.0, y)
        else:
            a = SingleHeadCoordinates(0.0, y)
            b = SingleHeadCoordinates(WIDTH, y)

        program_lines.append(
            line_single(state, a=a, b=b)
        )

    program_lines.append(end_block())
    return program_lines, name

def TEST_SCROLLBACK(version = "V2"):
    state = MachineState()
    name = f"{inspect.currentframe().f_code.co_name}_{version}"

    program_lines: List[str] = []
    program_lines.append(starting_block(state, initial_coordinates=SingleHeadCoordinates(0, 0), design_name=f"{inspect.currentframe().f_code.co_name}_{version}"))

    program_lines.append(rectangle(state, start=SingleHeadCoordinates(0, 0), width=80, height=230, overlap_mm=40))
    ZERO = (15, 15)
    program_lines.append(rectangle(state, start=SingleHeadCoordinates(ZERO[0], ZERO[1]), width=50, height=200, overlap_mm=40))

    # 3 bottom single5
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(ZERO[0] + 25, ZERO[1] + 25 * 1), radius=25))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(ZERO[0] + 25, ZERO[1] + 25 * 3), radius=25))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(ZERO[0] + 25, ZERO[1] + 25 * 5), radius=25))
    program_lines.append(circle_single(state, center=SingleHeadCoordinates(ZERO[0] + 25, ZERO[1] + 25 * 7), radius=25))

    program_lines.append(end_block())
    return program_lines, name

if __name__ == "__main__":
    program, name = TEST_SCROLLBACK()
    model_name = name.rsplit("_", 1)[0]
    cnc = emit_program(program, crlf=False)
    save_program(cnc, f"outputs/{model_name}/{name}.CNC", crlf=True)
    x_max, y_max = show_interactive(f"outputs/{model_name}/{name}.CNC", margin=250, show_graph=True)
    save_vrp(f"outputs/{model_name}/{name}.CNC", name, int(x_max), int(y_max))
