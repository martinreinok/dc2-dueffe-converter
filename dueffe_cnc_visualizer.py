import re
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from matplotlib_tools import Ruler

# try:
#     matplotlib.use("macosx")
# except Exception:
#     pass
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from collections import defaultdict


COORD_RE = re.compile(r"([XYZ])\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
ANGLE_RE = re.compile(r"\ba\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")

_GRAPH_MARGIN = 300.0

def get_graph_margin() -> float:
    return _GRAPH_MARGIN

def set_graph_margin(value: float) -> None:
    global _GRAPH_MARGIN
    _GRAPH_MARGIN = float(value)

@dataclass
class Segment:
    head1: List[Tuple[float, float]]
    head2: List[Tuple[float, float]]
    style: str
    source_line: int


@dataclass
class Frame:
    source_line: int
    text: str
    x: float
    y: float
    z_offset: float
    needle_down: bool
    dual_head: bool
    segment_count: int


class SmartDimensioner:
    def __init__(self, ax, segments, tolerance=2.0, detail_level=1):
        self.ax = ax
        self.segments = segments
        self.detail_level = detail_level
        # Higher detail level reduces tolerance, capturing finer grid steps
        self.tol = tolerance if detail_level > 1 else max(tolerance, 10.0)

        self.occupied_zones = []
        self.global_min_x = self.global_max_x = self.global_min_y = self.global_max_y = 0

        self._calculate_bounds()

        # Extract the structural grid of the design instead of closed shapes
        self.x_levels = self._extract_grid('x')
        self.y_levels = self._extract_grid('y')

    def _calculate_bounds(self):
        xs, ys = [], []
        for seg in self.segments:
            for pts in (seg.head1, seg.head2):
                for px, py in pts:
                    xs.append(px)
                    ys.append(py)

        if xs:
            self.global_min_x, self.global_max_x = min(xs), max(xs)
            self.global_min_y, self.global_max_y = min(ys), max(ys)
            # Reserve the physical object area so dimensions are pushed outward
            self._add_occupied_zone((self.global_min_x - 5, self.global_min_y - 5,
                                     self.global_max_x + 5, self.global_max_y + 5))

    def _extract_grid(self, axis):
        raw_vals = set()
        for seg in self.segments:
            # Only use cutting moves to define the internal grid (ignore jumps)
            if seg.style == 'cut':
                for pts in (seg.head1, seg.head2):
                    for p in pts:
                        raw_vals.add(round(p[0] if axis == 'x' else p[1], 1))

        if not raw_vals: return []

        # Always lock the absolute outer bounds into the grid
        if axis == 'x':
            raw_vals.update([round(self.global_min_x, 1), round(self.global_max_x, 1)])
        else:
            raw_vals.update([round(self.global_min_y, 1), round(self.global_max_y, 1)])

        sorted_vals = sorted(list(raw_vals))

        # Cluster coordinates that are extremely close to remove micro-stutters
        clustered = [sorted_vals[0]]
        for val in sorted_vals[1:]:
            if val - clustered[-1] > self.tol:
                clustered.append(val)
            else:
                clustered[-1] = (clustered[-1] + val) / 2.0

        return clustered

    def _add_occupied_zone(self, rect):
        r_x1, r_y1, r_x2, r_y2 = rect
        self.occupied_zones.append((min(r_x1, r_x2), min(r_y1, r_y2), max(r_x1, r_x2), max(r_y1, r_y2)))

    def _is_overlapping(self, rect):
        r_x1, r_y1, r_x2, r_y2 = rect
        r_xmin, r_xmax = min(r_x1, r_x2), max(r_x1, r_x2)
        r_ymin, r_ymax = min(r_y1, r_y2), max(r_y1, r_y2)

        for (o_x1, o_y1, o_x2, o_y2) in self.occupied_zones:
            if (r_xmin < o_x2 and r_xmax > o_x1 and r_ymin < o_y2 and r_ymax > o_y1):
                return True
        return False

    def draw_chain(self, levels, axis):
        if len(levels) < 2: return

        text_style = dict(ha='center', va='center', fontsize=7, color='black',
                          bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8))
        ext_style = dict(color='black', lw=0.4, linestyle=':', alpha=0.5)

        base_offset = 40
        step_offset = 80

        if axis == 'x':
            y_ref = self.global_min_y
            for i in range(20):
                y_pos = y_ref - base_offset - (i * step_offset)
                rect = (levels[0], y_pos - 10, levels[-1], y_pos + 10)

                if not self._is_overlapping(rect):
                    self._add_occupied_zone(rect)
                    # Main continuous dimension line
                    self.ax.plot([levels[0], levels[-1]], [y_pos, y_pos], color='black', lw=0.6)

                    for j in range(len(levels) - 1):
                        x1, x2 = levels[j], levels[j + 1]
                        val = round(x2 - x1, 2)

                        if val < 1.0: continue

                        # Architectural tick marks
                        self.ax.plot([x1, x1], [y_pos - 3, y_pos + 3], color='black', lw=0.8)
                        self.ax.plot([x2, x2], [y_pos - 3, y_pos + 3], color='black', lw=0.8)

                        # Extension lines pointing to the grid
                        self.ax.plot([x1, x1], [self.global_min_y, y_pos], **ext_style)
                        self.ax.plot([x2, x2], [self.global_min_y, y_pos], **ext_style)

                        self.ax.text((x1 + x2) / 2, y_pos, f"{val:.1f}", **text_style)
                    return

        elif axis == 'y':
            x_ref = self.global_min_x
            for i in range(20):
                x_pos = x_ref - base_offset - (i * step_offset)
                rect = (x_pos - 10, levels[0], x_pos + 10, levels[-1])

                if not self._is_overlapping(rect):
                    self._add_occupied_zone(rect)
                    self.ax.plot([x_pos, x_pos], [levels[0], levels[-1]], color='black', lw=0.6)

                    for j in range(len(levels) - 1):
                        y1, y2 = levels[j], levels[j + 1]
                        val = round(y2 - y1, 2)

                        if val < 1.0: continue

                        self.ax.plot([x_pos - 3, x_pos + 3], [y1, y1], color='black', lw=0.8)
                        self.ax.plot([x_pos - 3, x_pos + 3], [y2, y2], color='black', lw=0.8)

                        self.ax.plot([self.global_min_x, x_pos], [y1, y1], **ext_style)
                        self.ax.plot([self.global_min_x, x_pos], [y2, y2], **ext_style)

                        self.ax.text(x_pos, (y1 + y2) / 2, f"{val:.1f}", rotation=90, **text_style)
                    return

    def draw_overall(self, p1, p2, axis):
        val = round(abs(p2[0] - p1[0]) if axis == 'x' else abs(p2[1] - p1[1]), 2)
        if val < 1.0: return

        text_style = dict(ha='center', va='center', fontsize=8, color='black', weight='bold',
                          bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.9))
        ext_style = dict(color='black', lw=0.4, linestyle=':', alpha=0.5)
        arrow_style = dict(arrowstyle='<->', color='black', lw=0.8)

        base_offset = 40
        step_offset = 80

        if axis == 'x':
            y_ref = self.global_min_y
            for i in range(20):
                y_pos = y_ref - base_offset - (i * step_offset)
                rect = (p1[0] - 10, y_pos - 10, p2[0] + 10, y_pos + 10)
                if not self._is_overlapping(rect):
                    self._add_occupied_zone(rect)
                    self.ax.annotate('', xy=(p1[0], y_pos), xytext=(p2[0], y_pos), arrowprops=arrow_style)
                    self.ax.text((p1[0] + p2[0]) / 2, y_pos, f"{val:.1f}", **text_style)
                    self.ax.plot([p1[0], p1[0]], [self.global_min_y, y_pos], **ext_style)
                    self.ax.plot([p2[0], p2[0]], [self.global_min_y, y_pos], **ext_style)
                    return
        else:
            x_ref = self.global_min_x
            for i in range(20):
                x_pos = x_ref - base_offset - (i * step_offset)
                rect = (x_pos - 10, p1[1] - 10, x_pos + 10, p2[1] + 10)
                if not self._is_overlapping(rect):
                    self._add_occupied_zone(rect)
                    self.ax.annotate('', xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]), arrowprops=arrow_style)
                    self.ax.text(x_pos, (p1[1] + p2[1]) / 2, f"{val:.1f}", rotation=90, **text_style)
                    self.ax.plot([self.global_min_x, x_pos], [p1[1], p1[1]], **ext_style)
                    self.ax.plot([self.global_min_x, x_pos], [p2[1], p2[1]], **ext_style)
                    return

    def process(self):
        if self.global_max_x - self.global_min_x < 1: return

        # 1. Pitch Chains (Inner Grid Details)
        if self.detail_level >= 1:
            if len(self.x_levels) > 2:
                self.draw_chain(self.x_levels, 'x')
            if len(self.y_levels) > 2:
                self.draw_chain(self.y_levels, 'y')

        # 2. Overall bounds (Outer Wrap)
        self.draw_overall((self.global_min_x, self.global_min_y), (self.global_max_x, self.global_min_y), 'x')
        self.draw_overall((self.global_min_x, self.global_min_y), (self.global_min_x, self.global_max_y), 'y')


def add_smart_dimensions(ax, segments, detail_level=3):
    dim = SmartDimensioner(ax, segments, detail_level=detail_level)
    dim.process()

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
    x = y = 0.0
    z_offset = 0.0
    needle_down = False
    dual_head = False

    segments: List[Segment] = []
    frames: List[Frame] = [Frame(0, "", x, y, z_offset, needle_down, dual_head, 0)]

    for line_no, raw in enumerate(source_lines, start=1):
        text = raw.rstrip("\n")
        upper = text.strip().upper()

        if "CALL DW11" in upper:
            needle_down, dual_head = True, False
        elif "CALL DW13" in upper:
            needle_down, dual_head = True, True
        elif "CALL UP1" in upper:
            needle_down = False
        elif "CALL QLYZ" in upper:
            nums = [float(s) for s in FLOAT_RE.findall(upper)]
            if len(nums) >= 2:
                z_offset = nums[1] - nums[0]

        token = upper.split()[0] if upper else ""
        coords = {axis.upper(): float(val) for axis, val in COORD_RE.findall(upper)}

        sweep = 0.0
        m = ANGLE_RE.search(upper)
        if m:
            sweep = float(m.group(1))

        if token in {"MR", "MI", "MOVI", "ARC"}:
            start_x, start_y = x, y
            target_x = coords.get("X", x)
            target_y = coords.get("Y", y)

            if "Z" in coords:
                z_offset = coords["Z"] - target_y

            if token in {"MI", "MOVI", "MR"}:
                head1 = [(start_x, start_y), (target_x, target_y)]
                style = "cut" if needle_down else "jump"
            else:
                head1 = arc_points(start_x, start_y, target_x, target_y, sweep)
                style = "cut" if needle_down else "jump"

            head2: List[Tuple[float, float]] = []
            if dual_head and needle_down and head1:
                head2 = [(px, py + z_offset) for px, py in head1]

            if head1:
                segments.append(Segment(head1, head2, style, line_no))

            x, y = target_x, target_y

        frames.append(Frame(line_no, text, x, y, z_offset, needle_down, dual_head, len(segments)))

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

    return min(xs) - get_graph_margin(), max(xs) + get_graph_margin(), min(ys) - get_graph_margin(), max(ys) + get_graph_margin()


class CNCVisualizer:
    def __init__(self, filename: str):
        self.filename = filename
        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                self.source_lines = f.readlines()
        except FileNotFoundError:
            raise SystemExit(f"File not found: {filename}")

        self.frames, self.segments = simulate(self.source_lines)
        self.xmin, self.xmax, self.ymin, self.ymax = compute_bounds(self.segments)

    def show(self):
        fig, ax = plt.subplots(figsize=(14, 9))

        # Make the main plot area occupy more of the window
        fig.subplots_adjust(
            left=0.04,
            right=0.98,
            top=0.95,
            bottom=0.10
        )

        # Move the slider lower so it uses less plot space
        slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
        slider = Slider(
            slider_ax,
            "Line",
            0,
            len(self.source_lines),
            valinit=len(self.source_lines),
            valstep=1,
        )

        legend_lines = [
            Line2D([0], [0], color="black", lw=2),
            Line2D([0], [0], color="limegreen", lw=2),
            Line2D([0], [0], color="blue", linestyle=":", lw=1),
        ]

        ruler = None

        def draw(step: int):
            nonlocal ruler
            cur_xmin, cur_xmax = ax.get_xlim()
            cur_ymin, cur_ymax = ax.get_ylim()

            if (cur_xmin, cur_xmax) == (0.0, 1.0) and (cur_ymin, cur_ymax) == (0.0, 1.0):
                cur_xmin, cur_xmax = self.xmin, self.xmax
                cur_ymin, cur_ymax = self.ymin, self.ymax

            ax.cla()
            ax.set_xlim(cur_xmin, cur_xmax)
            ax.set_ylim(cur_ymin, cur_ymax)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_title(f"CNC Visualizer - {self.filename}")

            frame = self.frames[int(step)]

            if step == len(self.source_lines):
                add_smart_dimensions(ax, self.segments[: frame.segment_count])

            for seg in self.segments[: frame.segment_count]:
                xs, ys = zip(*seg.head1)
                if seg.style == "jump":
                    ax.plot(xs, ys, ":", color="blue", alpha=0.4, linewidth=1)
                else:
                    ax.plot(xs, ys, "-", color="black", alpha=0.8, linewidth=1.5)

                if seg.head2:
                    xs2, ys2 = zip(*seg.head2)
                    ax.plot(xs2, ys2, "-", color="limegreen", alpha=0.6, linewidth=1.5)

            ax.plot(frame.x, frame.y, "o", color="black", markersize=8, zorder=10)

            status_lines = [f"H1: X={frame.x:.2f}, Y={frame.y:.2f}"]
            if frame.dual_head and frame.needle_down:
                h2x, h2y = frame.x, frame.y + frame.z_offset
                ax.plot(h2x, h2y, "o", color="limegreen", markersize=8, zorder=10)
                status_lines.append(f"H2: X={h2x:.2f}, Y={h2y:.2f}")
            else:
                status_lines.append("H2: Inactive")
            status_lines.append("Needle: DOWN" if frame.needle_down else "Needle: UP")

            ax.text(
                0.02,
                0.98,
                "\n".join(status_lines),
                transform=ax.transAxes,
                verticalalignment="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            line_text = "Line 0: (start)" if frame.source_line == 0 else f"Line {frame.source_line}: {frame.text}"
            ax.text(
                0.02,
                0.02,
                line_text,
                transform=ax.transAxes,
                verticalalignment="bottom",
                family="monospace",
                fontsize=10,
                color="darkred",
                weight="bold",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
            )

            ax.legend(legend_lines, ["Head 1 (Cut)", "Head 2 (Cut)", "Rapid Move"], loc="upper right")
            fig.canvas.draw_idle()
            ruler = Ruler(ax=ax, useblit=True)

        def on_key(event):
            step = int(slider.val)
            if event.key in {"left", "a"}:
                slider.set_val(max(0, step - 1))
            elif event.key in {"right", "d"}:
                slider.set_val(min(len(self.source_lines), step + 1))
            elif event.key == "home":
                slider.set_val(0)
            elif event.key == "end":
                slider.set_val(len(self.source_lines))

        fig.canvas.mpl_connect("key_press_event", on_key)
        slider.on_changed(lambda v: draw(int(v)))
        draw(int(slider.val))
        plt.show()

    def save_final_png(self, out_path: str | None = None, dpi: int = 200):
        from pathlib import Path
        import matplotlib.pyplot as plt

        step = len(self.source_lines)
        if out_path is None:
            out_path = str(Path(self.filename).with_suffix(".png"))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(Path(self.filename).name)

        frame = self.frames[step]
        for seg in self.segments[: frame.segment_count]:
            xs, ys = zip(*seg.head1)
            if seg.style == "jump":
                ax.plot(xs, ys, ":", color="blue", alpha=0.4, linewidth=1)
            else:
                ax.plot(xs, ys, "-", color="black", alpha=0.8, linewidth=1.5)
            if seg.head2:
                xs2, ys2 = zip(*seg.head2)
                ax.plot(xs2, ys2, "-", color="limegreen", alpha=0.6, linewidth=1.5)
        add_smart_dimensions(ax, self.segments[: frame.segment_count])
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "drawing.CNC"

    margin = 600
    if margin is not None:
        set_graph_margin(margin)

    viz = CNCVisualizer(filename)

    if "--save" in sys.argv:
        png = viz.save_final_png()
        print(f"Saved: {png}")
        return

    viz.show()

def show_interactive(filename, save_drawing: bool = True, margin: float = None, show_graph = True):
    if margin is not None:
        set_graph_margin(margin)
    viz = CNCVisualizer(filename)
    if show_graph:
        viz.show()
    if save_drawing:
        png = viz.save_final_png()
        print(f"Saved: {png}")
    return viz.xmax - get_graph_margin(), viz.ymax - get_graph_margin()


if __name__ == "__main__":
    main()
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_080_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_090_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_105_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_120_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_140_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_160_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_180_V_FLIP.CNC", show_graph=False)
    # show_interactive(R"D:\Clients\Hilding Anders Baltic\P-WALL\FLIP/PWALL_210_V_FLIP.CNC", show_graph=False)
