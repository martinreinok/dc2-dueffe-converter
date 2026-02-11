from __future__ import annotations

import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
Signal = QtCore.Signal  # type: ignore


# --- Matplotlib embedding (Qt6) --------------------------------------------
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import (  # type: ignore
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import dxf_to_cnc
import numpy as np


COORD_RE = re.compile(r"([XYZ])\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
ANGLE_RE = re.compile(r"\ba\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")

GRAPH_MARGIN = 400.0

def emit_program(lines: List[str], crlf: bool = True) -> str:
    sep = "\r\n" if crlf else "\n"
    return sep.join(lines) + sep


@dataclass(slots=True)
class Segment:
    head1: List[Tuple[float, float]]
    head2: List[Tuple[float, float]]
    style: str
    source_line: int


@dataclass(slots=True)
class Frame:
    source_line: int
    text: str
    x: float
    y: float  # active vertical axis (Y or Z)
    z_offset: float
    needle_down: bool
    dual_head: bool
    segment_count: int
    axis: str = "Y"  # "Y" or "Z" for y-field


def arc_points(x1: float, y1: float, x2: float, y2: float, sweep_deg: float) -> List[Tuple[float, float]]:
    if abs(sweep_deg) < 1e-9:
        return [(x1, y1), (x2, y2)]

    dx, dy = x2 - x1, y2 - y1
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        return [(x1, y1)]

    sweep_rad = math.radians(sweep_deg)
    mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if abs(abs(sweep_deg) - 180.0) < 1e-6:
        cx, cy = mid_x, mid_y
    else:
        nx, ny = -dy / chord, dx / chord
        k = (chord / 2.0) / math.tan(sweep_rad / 2.0)
        cx, cy = mid_x + k * nx, mid_y + k * ny

    r = math.hypot(x1 - cx, y1 - cy)
    start = math.atan2(y1 - cy, x1 - cx)

    steps = max(24, int(abs(sweep_deg) * 0.5))
    angles = start + np.linspace(0.0, sweep_rad, steps)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)

    pts = list(zip(xs.tolist(), ys.tolist()))
    pts[0] = (x1, y1)
    pts[-1] = (x2, y2)
    return pts


def simulate(source_lines: List[str]) -> Tuple[List[Frame], List[Segment]]:
    """
    Simulates the CNC program into frames and drawable segments.

    Important fix:
    - Programs may switch the "vertical" axis between Y (normal XY plane)
      and Z (XZ plane) using CALL QLY/QLYZ vs CALL QLZ, and ELY/ELYZ vs ELZ.
    - In Y-mode, a Z coordinate on motion commands is interpreted as the second-head
      Y coordinate (dual-head), so z_offset = Z - Y.
    - In Z-mode, Z is the active vertical axis and must NOT be treated as a head2 offset.
    """
    x = 0.0
    y = 0.0  # active vertical coordinate (Y or Z depending on axis)
    axis = "Y"

    z_offset = 0.0  # dual-head Y offset (only meaningful in axis=="Y")
    needle_down = False
    dual_head = False

    segments: List[Segment] = []
    frames: List[Frame] = [Frame(0, "", x, y, z_offset, needle_down, dual_head, 0, axis)]

    for line_no, raw in enumerate(source_lines, start=1):
        text = raw.rstrip("\n")
        upper = text.strip().upper()

        # --- axis/mode switches (observed in reference CNCs) ----------------
        if "CALL QLZ" in upper or "CALL ELZ" in upper:
            axis = "Z"
        elif "CALL QLYZ" in upper or "CALL QLY" in upper or "CALL ELYZ" in upper or "CALL ELY" in upper:
            axis = "Y"

        # --- tool state -----------------------------------------------------
        if "CALL DW11" in upper:
            needle_down, dual_head = True, False
        elif "CALL DW12" in upper:
            # Z-plane single head (reference files use DW12 after ELZ)
            needle_down, dual_head = True, False
        elif "CALL DW13" in upper:
            needle_down, dual_head = True, True
        elif "CALL UP1" in upper:
            needle_down = False

        # dual-head offset from QLYZ (Y-mode only)
        if axis == "Y" and "CALL QLYZ" in upper:
            nums = [float(s) for s in FLOAT_RE.findall(upper)]
            if len(nums) >= 2:
                z_offset = nums[1] - nums[0]

        token = upper.split()[0] if upper else ""
        coords = {axis_name.upper(): float(val) for axis_name, val in COORD_RE.findall(upper)}

        sweep = 0.0
        m = ANGLE_RE.search(upper)
        if m:
            sweep = float(m.group(1))

        if token in {"MR", "MI", "MOVI", "ARC"}:
            start_x, start_y = x, y
            target_x = coords.get("X", x)

            # vertical axis depends on mode
            if axis == "Y":
                target_y = coords.get("Y", y)

                # In Y-mode, Z means second-head Y (dual) or sets the offset reference (MR ... Z ...)
                if "Z" in coords:
                    z_offset = coords["Z"] - target_y
            else:
                # Z-mode: Z is the vertical axis
                target_y = coords.get("Z", y)

            if token in {"MI", "MOVI", "MR"}:
                head1 = [(start_x, start_y), (target_x, target_y)]
                style = "cut" if needle_down else "jump"
            else:
                head1 = arc_points(start_x, start_y, target_x, target_y, sweep)
                style = "cut" if needle_down else "jump"

            head2: List[Tuple[float, float]] = []
            if axis == "Y" and dual_head and needle_down and head1:
                head2 = [(px, py + z_offset) for px, py in head1]

            if head1:
                segments.append(Segment(head1, head2, style, line_no))

            x, y = target_x, target_y

        frames.append(Frame(line_no, text, x, y, z_offset, needle_down, dual_head, len(segments), axis))

    return frames, segments


def compute_bounds(segments: List[Segment]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for seg in segments:
        for pts in (seg.head1, seg.head2):
            for px, py in pts:
                xs.append(px)
                ys.append(py)

    if not xs:
        return 0.0, 1000.0, 0.0, 1000.0

    return (
        min(xs) - GRAPH_MARGIN,
        max(xs) + GRAPH_MARGIN,
        min(ys) - GRAPH_MARGIN,
        max(ys) + GRAPH_MARGIN,
    )


# ---------------------------------------------------------------------------
# CNC block parsing (kept; editor panel removed from UI)
# ---------------------------------------------------------------------------

def fmt_float(value: float) -> str:
    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


@dataclass(slots=True)
class CncCommand:
    raw: str
    token: str
    coords: dict[str, float]
    sweep_deg: Optional[float]

    @classmethod
    def parse(cls, line: str) -> "CncCommand":
        text = line.rstrip("\n")
        upper = text.strip().upper()
        token = upper.split()[0] if upper else ""
        coords = {axis_name.upper(): float(val) for axis_name, val in COORD_RE.findall(upper)}
        m = ANGLE_RE.search(upper)
        sweep = float(m.group(1)) if m else None
        return cls(raw=text, token=token, coords=coords, sweep_deg=sweep)

    def render(self) -> str:
        if self.token not in {"MR", "MI", "MOVI", "ARC"}:
            return self.raw

        coords = "".join(
            f"{axis}{fmt_float(self.coords[axis])}" for axis in ("X", "Y", "Z") if axis in self.coords
        )
        if self.token == "ARC" and self.sweep_deg is not None:
            return f"{self.token} {coords} a={fmt_float(self.sweep_deg)}".rstrip()
        return f"{self.token} {coords}".rstrip()


@dataclass(slots=True)
class CncBlock:
    prelude: List[str]
    lines: List[str]  # includes MR..CALL UP1
    line_start: int = 0
    line_end: int = 0

    def has_dual_head(self) -> bool:
        return any("CALL DW13" in ln.strip().upper() for ln in self.lines)

    def summary(self) -> str:
        mr = next((ln for ln in self.lines if ln.strip().upper().startswith("MR")), "")
        cmd = CncCommand.parse(mr) if mr else None
        x = cmd.coords.get("X") if cmd else None
        y = cmd.coords.get("Y") if cmd else None
        ops = sum(
            1
            for ln in self.lines
            if ln.strip().upper().split()[:1]
            and ln.strip().upper().split()[0] in {"MI", "MOVI", "ARC"}
        )
        arc_count = sum(1 for ln in self.lines if ln.strip().upper().startswith("ARC"))
        kind = "Dual" if self.has_dual_head() else "Single"
        pos = f"X={x:.2f}, Y={y:.2f}" if (x is not None and y is not None) else "(no MR)"
        extra = f", {arc_count} arc" + ("s" if arc_count != 1 else "") if arc_count else ""
        return f"{kind} • {ops} ops{extra} • {pos}"

    def render_lines(self) -> List[str]:
        return [ln.rstrip("\n") for ln in (self.prelude + self.lines)]


@dataclass(slots=True)
class CncProgram:
    header: List[str]
    blocks: List[CncBlock]
    footer: List[str]
    source_path: Optional[Path] = None

    def render_lines(self) -> List[str]:
        lines: List[str] = []
        lines.extend([ln.rstrip("\n") for ln in self.header])
        for block in self.blocks:
            start = len(lines) + 1
            block_lines = block.render_lines()
            lines.extend(block_lines)
            end = len(lines)
            block.line_start = start
            block.line_end = end
        lines.extend([ln.rstrip("\n") for ln in self.footer])
        return lines

    def block_for_source_line(self, line_no: int) -> Optional[int]:
        for i, blk in enumerate(self.blocks):
            if blk.line_start <= line_no <= blk.line_end:
                return i
        return None


class CncBlockParser:
    """Splits a CNC program into blocks.

    A block starts at the first MR and ends at CALL UP1 (inclusive).
    Lines before MR that look like prelude travel with the block.
    """

    def parse(self, lines: Sequence[str], *, source_path: Optional[Path] = None) -> CncProgram:
        header: List[str] = []
        footer: List[str] = []
        blocks: List[CncBlock] = []

        pending: List[str] = []
        in_block = False
        cur_prelude: List[str] = []
        cur_lines: List[str] = []

        def split_header_and_prelude(pending_lines: List[str]) -> Tuple[List[str], List[str]]:
            def is_prelude_line(ln: str) -> bool:
                u = ln.strip().upper()
                if not u:
                    return True
                if u.startswith(";"):
                    return True
                if u.startswith("CALL QLYZ") or u.startswith("CALL QLY") or u.startswith("CALL QLZ"):
                    return True
                return False

            split_idx = len(pending_lines)
            for i in range(len(pending_lines) - 1, -1, -1):
                if is_prelude_line(pending_lines[i]):
                    split_idx = i
                else:
                    break

            return pending_lines[:split_idx], pending_lines[split_idx:]

        for raw in lines:
            ln = raw.rstrip("\n")
            upper = ln.strip().upper()
            token = upper.split()[0] if upper else ""

            if not in_block:
                if token == "MR":
                    if not blocks:
                        head, prel = split_header_and_prelude(pending)
                        header.extend(head)
                        cur_prelude = prel
                        pending = []
                    else:
                        cur_prelude = pending
                        pending = []
                    cur_lines = [ln]
                    in_block = True
                else:
                    pending.append(ln)
                continue

            # in_block
            cur_lines.append(ln)
            if "CALL UP1" in upper:
                blocks.append(CncBlock(prelude=cur_prelude, lines=cur_lines))
                cur_prelude = []
                cur_lines = []
                in_block = False

        if in_block and cur_lines:
            blocks.append(CncBlock(prelude=cur_prelude, lines=cur_lines))
            pending = []

        footer = pending if blocks else []
        prog = CncProgram(header=header, blocks=blocks, footer=footer, source_path=source_path)
        prog.render_lines()  # compute spans
        return prog


# ---------------------------------------------------------------------------
# Matplotlib view widget
# ---------------------------------------------------------------------------

class CncPlotWidget(QtWidgets.QWidget):
    segmentClicked = Signal(int)  # segment index

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self._figure = Figure(figsize=(10, 7), tight_layout=False)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._ax = self._figure.add_subplot(111)
        self._figure.subplots_adjust(bottom=0.08)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._source_lines: List[str] = []
        self._frames: List[Frame] = []
        self._segments: List[Segment] = []
        self._bounds: Tuple[float, float, float, float] = (0.0, 1000.0, 0.0, 1000.0)
        self._selected_segments: set[int] = set()
        self._step: int = 0

        self._legend_lines = [
            Line2D([0], [0], color="black", lw=2),
            Line2D([0], [0], color="limegreen", lw=2),
            Line2D([0], [0], color="blue", linestyle=":", lw=1),
            Line2D([0], [0], color="#ff8c00", lw=3),
        ]

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("key_press_event", self._on_key)

    @property
    def segments(self) -> List[Segment]:
        return self._segments

    def set_program_lines(self, lines: Sequence[str]) -> None:
        self._source_lines = [ln.rstrip("\n") for ln in lines]
        self._frames, self._segments = simulate(list(self._source_lines))
        self._bounds = compute_bounds(self._segments)
        self._step = len(self._source_lines)
        self._selected_segments.clear()
        self._draw(self._step, reset_view=True)

    def set_selected_segments(self, indices: Iterable[int]) -> None:
        self._selected_segments = set(indices)
        self._draw(self._step, reset_view=False)

    def set_step(self, step: int) -> None:
        self._step = max(0, min(int(step), len(self._source_lines)))
        self._draw(self._step, reset_view=False)

    def _draw(self, step: int, *, reset_view: bool) -> None:
        if not self._frames:
            self._ax.cla()
            self._ax.set_title("No program loaded")
            self._canvas.draw_idle()
            return

        cur_xmin, cur_xmax = self._ax.get_xlim()
        cur_ymin, cur_ymax = self._ax.get_ylim()
        if reset_view or ((cur_xmin, cur_xmax) == (0.0, 1.0) and (cur_ymin, cur_ymax) == (0.0, 1.0)):
            cur_xmin, cur_xmax, cur_ymin, cur_ymax = self._bounds

        self._ax.cla()
        self._ax.set_xlim(cur_xmin, cur_xmax)
        self._ax.set_ylim(cur_ymin, cur_ymax)
        self._ax.set_aspect("equal")
        self._ax.grid(True, alpha=0.25)
        self._ax.set_xlabel("X (mm)")

        frame = self._frames[int(step)]
        axis = frame.axis.upper()
        self._ax.set_ylabel(f"{axis} (mm)")
        self._ax.set_title("CNC Preview")

        # Draw segments
        for i, seg in enumerate(self._segments[: frame.segment_count]):
            xs, ys = zip(*seg.head1)
            if seg.style == "jump":
                self._ax.plot(xs, ys, ":", color="blue", alpha=0.35, linewidth=1)
            else:
                self._ax.plot(xs, ys, "-", color="black", alpha=0.75, linewidth=1.5)

            if seg.head2:
                xs2, ys2 = zip(*seg.head2)
                self._ax.plot(xs2, ys2, "-", color="limegreen", alpha=0.55, linewidth=1.5)

            if i in self._selected_segments:
                self._ax.plot(xs, ys, "-", color="#ff8c00", alpha=0.95, linewidth=3.0)
                if seg.head2:
                    self._ax.plot(xs2, ys2, "-", color="#ff8c00", alpha=0.75, linewidth=3.0)

        # Current head position
        self._ax.plot(frame.x, frame.y, "o", color="black", markersize=7, zorder=10)
        if axis == "Y" and frame.dual_head and frame.needle_down:
            h2x, h2y = frame.x, frame.y + frame.z_offset
            self._ax.plot(h2x, h2y, "o", color="limegreen", markersize=7, zorder=10)

        vlabel = "Y" if axis == "Y" else "Z"
        status_lines = [f"H1: X={frame.x:.2f}, {vlabel}={frame.y:.2f}"]
        if axis == "Y" and frame.dual_head and frame.needle_down:
            status_lines.append(f"H2: X={frame.x:.2f}, Y={(frame.y + frame.z_offset):.2f}")
        else:
            status_lines.append("H2: Inactive")
        status_lines.append("Needle: DOWN" if frame.needle_down else "Needle: UP")
        status_lines.append(f"Plane: X{vlabel}")

        self._ax.text(
            0.02,
            0.98,
            "\n".join(status_lines),
            transform=self._ax.transAxes,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        line_text = "Line 0: (start)" if frame.source_line == 0 else f"Line {frame.source_line}: {frame.text}"
        self._ax.text(
            0.02,
            0.02,
            line_text,
            transform=self._ax.transAxes,
            verticalalignment="bottom",
            family="monospace",
            fontsize=9,
            color="darkred",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="#fff3b0", alpha=0.85),
        )

        self._ax.legend(
            self._legend_lines,
            ["Head 1 (Cut)", "Head 2 (Cut)", "Rapid Move", "Selected Block"],
            loc="upper right",
        )
        self._canvas.draw_idle()

    def _on_key(self, event) -> None:
        if not self._source_lines:
            return
        if event.key in {"left", "a"}:
            self.set_step(self._step - 1)
        elif event.key in {"right", "d"}:
            self.set_step(self._step + 1)
        elif event.key == "home":
            self.set_step(0)
        elif event.key == "end":
            self.set_step(len(self._source_lines))

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        idx = self._nearest_segment_index(float(event.xdata), float(event.ydata))
        if idx is not None:
            self.segmentClicked.emit(int(idx))

    def _nearest_segment_index(self, x: float, y: float) -> Optional[int]:
        if not self._segments:
            return None

        best_idx: Optional[int] = None
        best_d2 = float("inf")

        # Distance to polyline segments, approximate. Consider both heads if present.
        for i, seg in enumerate(self._segments):
            for pts in (seg.head1, seg.head2):
                if not pts or len(pts) < 2:
                    continue
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    d2 = self._point_to_segment_d2(x, y, x0, y0, x1, y1)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = i
        return best_idx

    @staticmethod
    def _point_to_segment_d2(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
        vx = x1 - x0
        vy = y1 - y0
        wx = px - x0
        wy = py - y0
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            dx = px - x0
            dy = py - y0
            return dx * dx + dy * dy
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        cx = x0 + t * vx
        cy = y0 + t * vy
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy


# ---------------------------------------------------------------------------
# Qt model
# ---------------------------------------------------------------------------

class BlockListModel(QtCore.QAbstractListModel):
    def __init__(self, program: Optional[CncProgram] = None):
        super().__init__()
        self._program = program

    def set_program(self, program: Optional[CncProgram]) -> None:
        self.beginResetModel()
        self._program = program
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid() or self._program is None:
            return 0
        return len(self._program.blocks)

    def data(self, index: QtCore.QModelIndex, role: int = 0):  # type: ignore[override]
        if not index.isValid() or self._program is None:
            return None
        block = self._program.blocks[index.row()]

        if role == int(QtCore.Qt.ItemDataRole.DisplayRole):
            return f"{index.row()+1:02d}  {block.summary()}"

        if role == int(QtCore.Qt.ItemDataRole.ToolTipRole):
            return block.summary()

        return None

    def flags(self, index: QtCore.QModelIndex):  # type: ignore[override]
        default = super().flags(index)
        if not self._program:
            return default
        if index.isValid():
            return (
                default
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
        return default | QtCore.Qt.ItemFlag.ItemIsDropEnabled

    def supportedDropActions(self):  # type: ignore[override]
        return QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self) -> List[str]:  # type: ignore[override]
        return ["application/x-cnc-block-row"]

    def mimeData(self, indexes: List[QtCore.QModelIndex]) -> QtCore.QMimeData:  # type: ignore[override]
        mime = QtCore.QMimeData()
        rows = sorted({i.row() for i in indexes if i.isValid()})
        if rows:
            mime.setData("application/x-cnc-block-row", str(rows[0]).encode("utf-8"))
        return mime

    def dropMimeData(self, data: QtCore.QMimeData, action, row: int, column: int, parent: QtCore.QModelIndex):  # type: ignore[override]
        if action != QtCore.Qt.DropAction.MoveAction:
            return False
        if self._program is None:
            return False
        if not data.hasFormat("application/x-cnc-block-row"):
            return False

        src_row = int(bytes(data.data("application/x-cnc-block-row")).decode("utf-8"))
        if row == -1:
            row = parent.row() if parent.isValid() else self.rowCount()
        return self.moveRows(QtCore.QModelIndex(), src_row, 1, QtCore.QModelIndex(), row)

    def moveRows(
        self,
        sourceParent: QtCore.QModelIndex,
        sourceRow: int,
        count: int,
        destinationParent: QtCore.QModelIndex,
        destinationChild: int,
    ) -> bool:  # type: ignore[override]
        if self._program is None or count != 1:
            return False
        n = len(self._program.blocks)
        if not (0 <= sourceRow < n):
            return False

        dest = destinationChild
        if dest > sourceRow:
            dest -= 1
        dest = max(0, min(dest, n - 1))
        if dest == sourceRow:
            return False

        self.beginMoveRows(QtCore.QModelIndex(), sourceRow, sourceRow, QtCore.QModelIndex(), destinationChild)
        blk = self._program.blocks.pop(sourceRow)
        self._program.blocks.insert(dest, blk)
        self.endMoveRows()
        return True


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNC Block Editor")
        self.setAcceptDrops(True)
        self.resize(1400, 850)

        self._parser = CncBlockParser()
        self._program: Optional[CncProgram] = None

        self._plot = CncPlotWidget()
        self._plot.segmentClicked.connect(self._on_segment_clicked)

        self._block_model = BlockListModel(None)
        self._block_view = QtWidgets.QListView()
        self._block_view.setModel(self._block_model)
        self._block_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._block_view.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._block_view.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self._block_view.selectionModel().selectionChanged.connect(self._on_block_selected)
        self._block_model.rowsMoved.connect(lambda *_: self._refresh_after_program_edit())

        # Buttons
        self._open_btn = QtWidgets.QPushButton("Open / Drop DXF or CNC")
        self._open_btn.clicked.connect(self.open_dialog)
        self._export_btn = QtWidgets.QPushButton("Export CNC")
        self._export_btn.clicked.connect(self.export_dialog)
        self._delete_btn = QtWidgets.QPushButton("Delete Block")
        self._delete_btn.clicked.connect(self.delete_selected_block)

        self._up_btn = QtWidgets.QToolButton()
        self._up_btn.setText("▲")
        self._up_btn.setToolTip("Move block up")
        self._up_btn.clicked.connect(lambda: self._move_selected_block(-1))
        self._down_btn = QtWidgets.QToolButton()
        self._down_btn.setText("▼")
        self._down_btn.setToolTip("Move block down")
        self._down_btn.clicked.connect(lambda: self._move_selected_block(+1))

        self._step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._step_slider.setRange(0, 0)
        self._step_slider.valueChanged.connect(self._plot.set_step)
        self._step_label = QtWidgets.QLabel("Line: 0")
        self._step_slider.valueChanged.connect(lambda v: self._step_label.setText(f"Line: {v}"))

        # Left panel layout
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self._open_btn, 1)
        top_row.addWidget(self._export_btn)
        left_layout.addLayout(top_row)

        reorder_row = QtWidgets.QHBoxLayout()
        reorder_row.addWidget(QtWidgets.QLabel("Blocks"))
        reorder_row.addStretch(1)
        reorder_row.addWidget(self._up_btn)
        reorder_row.addWidget(self._down_btn)
        reorder_row.addWidget(self._delete_btn)
        left_layout.addLayout(reorder_row)

        left_layout.addWidget(self._block_view, 1)

        slider_box = QtWidgets.QGroupBox("Preview step")
        slider_layout = QtWidgets.QVBoxLayout(slider_box)
        slider_layout.addWidget(self._step_label)
        slider_layout.addWidget(self._step_slider)
        left_layout.addWidget(slider_box, 0)

        # Splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._plot)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 980])
        self.setCentralWidget(splitter)

        self._build_menu()
        self._apply_modern_style()
        self.statusBar().showMessage("Drop a .DXF or .CNC file to begin")

    # --- UI helpers --------------------------------------------------------

    def _build_menu(self) -> None:
        open_act = QtGui.QAction("Open…", self)
        open_act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.open_dialog)

        export_act = QtGui.QAction("Export…", self)
        export_act.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        export_act.triggered.connect(self.export_dialog)

        quit_act = QtGui.QAction("Quit", self)
        quit_act.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_act)
        file_menu.addAction(export_act)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

    def _apply_modern_style(self) -> None:
        QtWidgets.QApplication.setStyle("Fusion")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(28, 28, 28))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(235, 235, 235))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(22, 22, 22))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(235, 235, 235))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(235, 235, 235))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(255, 140, 0))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(15, 15, 15))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(0, 0, 0))
        self.setPalette(palette)

        # Use stylesheet mainly for spacing/borders; keep colors aligned with palette.
        self.setStyleSheet(
            """
            QMainWindow { background: #1c1c1c; }
            QGroupBox { font-weight: 600; border: 1px solid #3a3a3a; border-radius: 10px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QPushButton, QToolButton {
                padding: 8px 10px;
                border-radius: 10px;
                border: 1px solid #3a3a3a;
                background-color: #2d2d2d;
                color: #ebebeb;
            }
            QPushButton:hover, QToolButton:hover { border-color: #5a5a5a; background-color: #353535; }
            QPushButton:pressed, QToolButton:pressed { background-color: #262626; }
            QListView {
                border: 1px solid #3a3a3a;
                border-radius: 10px;
                background: #161616;
                color: #ebebeb;
                padding: 6px;
            }
            QHeaderView::section {
                background: #2d2d2d;
                color: #ebebeb;
                padding: 6px;
                border: none;
                border-right: 1px solid #3a3a3a;
            }
            QSlider::groove:horizontal { height: 8px; border-radius: 4px; background: #3a3a3a; }
            QSlider::handle:horizontal { width: 18px; margin: -6px 0; border-radius: 9px; background: #ff8c00; }
            QMenuBar { background: #1c1c1c; color: #ebebeb; }
            QMenuBar::item:selected { background: #2d2d2d; }
            QMenu { background: #1c1c1c; color: #ebebeb; border: 1px solid #3a3a3a; }
            QMenu::item:selected { background: #2d2d2d; }
            """
        )

    # --- Drag & drop -------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.open_path(Path(path))

    # --- Actions -----------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open DXF or CNC",
            str(Path.cwd()),
            "DXF/CNC Files (*.dxf *.DXF *.cnc *.CNC);;All Files (*.*)",
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "Open", f"File not found:\n{path}")
            return

        try:
            if path.suffix.lower() == ".dxf":
                lines = self._convert_dxf_to_cnc_lines(path)
                program = self._parser.parse(lines, source_path=path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                program = self._parser.parse(text.splitlines(), source_path=path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Could not load {path.name}:\n{exc}")
            return

        self._set_program(program)
        self.statusBar().showMessage(f"Loaded: {path}")

    def export_dialog(self) -> None:
        if self._program is None:
            QtWidgets.QMessageBox.information(self, "Export", "Nothing to export yet.")
            return
        suggested = "output.CNC"
        if self._program.source_path is not None:
            suggested = self._program.source_path.with_suffix(".CNC").name

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CNC",
            str(Path.cwd() / suggested),
            "CNC Files (*.CNC *.cnc);;All Files (*.*)",
        )
        if not path:
            return

        out = Path(path)
        lines = self._program.render_lines()
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="ignore")
        self.statusBar().showMessage(f"Exported: {out}")

    def delete_selected_block(self) -> None:
        if self._program is None:
            return
        idx = self._current_block_index()
        if idx is None:
            return

        self._block_model.beginResetModel()
        self._program.blocks.pop(idx)
        self._block_model.endResetModel()
        self._refresh_after_program_edit()

    # --- Internals ---------------------------------------------------------

    def _set_program(self, program: CncProgram) -> None:
        self._program = program
        self._block_model.set_program(program)
        lines = program.render_lines()
        self._plot.set_program_lines(lines)

        self._step_slider.blockSignals(True)
        self._step_slider.setRange(0, len(lines))
        self._step_slider.setValue(len(lines))
        self._step_slider.blockSignals(False)
        self._step_label.setText(f"Line: {len(lines)}")

        if program.blocks:
            self._block_view.setCurrentIndex(self._block_model.index(0))
        else:
            self._plot.set_selected_segments([])

    def _refresh_after_program_edit(self) -> None:
        if self._program is None:
            return
        lines = self._program.render_lines()
        self._plot.set_program_lines(lines)
        self._step_slider.blockSignals(True)
        self._step_slider.setRange(0, len(lines))
        self._step_slider.setValue(len(lines))
        self._step_slider.blockSignals(False)
        self._step_label.setText(f"Line: {len(lines)}")

        idx = self._current_block_index()
        if idx is not None:
            self._highlight_block(idx)

    def _current_block_index(self) -> Optional[int]:
        sel = self._block_view.selectionModel().selectedIndexes()
        if not sel:
            return None
        return sel[0].row()

    def _on_block_selected(self, *_args) -> None:
        idx = self._current_block_index()
        if self._program is None or idx is None:
            self._plot.set_selected_segments([])
            return
        self._highlight_block(idx)

    def _highlight_block(self, idx: int) -> None:
        if self._program is None:
            return
        block = self._program.blocks[idx]
        segs = self._plot.segments
        selected = [i for i, seg in enumerate(segs) if block.line_start <= seg.source_line <= block.line_end]
        self._plot.set_selected_segments(selected)

    def _on_segment_clicked(self, seg_index: int) -> None:
        if self._program is None:
            return
        segs = self._plot.segments
        if not (0 <= seg_index < len(segs)):
            return
        line_no = segs[seg_index].source_line
        blk_idx = self._program.block_for_source_line(line_no)
        if blk_idx is None:
            return
        self._block_view.setCurrentIndex(self._block_model.index(blk_idx))

    def _move_selected_block(self, delta: int) -> None:
        if self._program is None:
            return
        idx = self._current_block_index()
        if idx is None:
            return
        new_idx = idx + int(delta)
        if not (0 <= new_idx < len(self._program.blocks)):
            return

        self._block_model.beginResetModel()
        blk = self._program.blocks.pop(idx)
        self._program.blocks.insert(new_idx, blk)
        self._block_model.endResetModel()
        self._refresh_after_program_edit()
        self._block_view.setCurrentIndex(self._block_model.index(new_idx))

    # --- DXF conversion ----------------------------------------------------

    def _convert_dxf_to_cnc_lines(self, dxf_path: Path) -> List[str]:
        program_obj = dxf_to_cnc.dxf_to_cnc_single_polylines(
            str(dxf_path),
            design_name=dxf_path.stem,
            include_inserts=True,
            force_close_tol=0.05,
            bulge_eps=1e-9,
            add_semicolon_line=False,
            optimize_travel=True,
        )

        return emit_program(program_obj, crlf=False).split("\n")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if len(sys.argv) >= 2:
        p = Path(sys.argv[1])
        if p.exists():
            QtCore.QTimer.singleShot(0, lambda: win.open_path(p))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
