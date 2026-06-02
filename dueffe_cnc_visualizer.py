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
    def __init__(self, ax, segments, tolerance=2.0):
        self.ax = ax
        self.segments = segments
        self.tol = tolerance
        
        # Uniqueness filters applied globally per axis
        self.drawn_values_x = set()
        self.drawn_values_y = set()
        self.drawn_diameters = set()
        
        # Collision System
        self.occupied_zones = []
        self.shapes = self._extract_shapes()

        # Mark physical shapes as occupied to push text outwards
        for s in self.shapes:
            pad = 10
            self._add_occupied_zone((
                s['min_x'] - pad, s['min_y'] - pad,
                s['max_x'] + pad, s['max_y'] + pad
            ))

        # Global physical bounds
        if self.shapes:
            self.global_min_x = min(s['min_x'] for s in self.shapes)
            self.global_max_x = max(s['max_x'] for s in self.shapes)
            self.global_min_y = min(s['min_y'] for s in self.shapes)
            self.global_max_y = max(s['max_y'] for s in self.shapes)
        else:
            self.global_min_x = self.global_max_x = self.global_min_y = self.global_max_y = 0

    def _extract_shapes(self):
        shapes = []
        curr_h1 = []
        curr_h2 = []
        
        # Explicitly separate Head 1 and Head 2 so dual cuts aren't merged
        for seg in self.segments:
            if seg.style == 'jump':
                if curr_h1:
                    shapes.append(self._analyze_shape(curr_h1))
                    curr_h1 = []
                if curr_h2:
                    shapes.append(self._analyze_shape(curr_h2))
                    curr_h2 = []
            else:
                curr_h1.extend(seg.head1)
                if seg.head2:
                    curr_h2.extend(seg.head2)
        
        if curr_h1: shapes.append(self._analyze_shape(curr_h1))
        if curr_h2: shapes.append(self._analyze_shape(curr_h2))
        return shapes

    def _analyze_shape(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w, h = max_x - min_x, max_y - min_y

        shape_type = 'unknown'
        if w > h * 2 and w > 10:
            shape_type = 'horizontal'
        elif h > w * 2 and h > 10:
            shape_type = 'vertical'
        elif abs(w - h) <= 5.0 and w < 100 and h < 100:
            shape_type = 'circle'
        elif w > 100 and h > 100:
            shape_type = 'boundary'

        return {
            'cx': (min_x + max_x) / 2, 'cy': (min_y + max_y) / 2,
            'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y,
            'width': w, 'height': h, 'type': shape_type,
            'diameter': (w + h) / 2 if shape_type == 'circle' else None
        }

    def _is_overlapping(self, rect):
        r_x1, r_y1, r_x2, r_y2 = rect
        r_xmin, r_xmax = min(r_x1, r_x2), max(r_x1, r_x2)
        r_ymin, r_ymax = min(r_y1, r_y2), max(r_y1, r_y2)

        for (o_x1, o_y1, o_x2, o_y2) in self.occupied_zones:
            if (r_xmin < o_x2 and r_xmax > o_x1 and r_ymin < o_y2 and r_ymax > o_y1):
                return True
        return False

    def _add_occupied_zone(self, rect):
        r_x1, r_y1, r_x2, r_y2 = rect
        self.occupied_zones.append((min(r_x1, r_x2), min(r_y1, r_y2), max(r_x1, r_x2), max(r_y1, r_y2)))

    def draw_diameter(self, shape):
        if shape['diameter'] is None: return

        val = round(shape['diameter'], 2)
        if val in self.drawn_diameters: return

        text = f"Ø{val:.1f}"
        cx, cy = shape['cx'], shape['cy']
        radius = shape['diameter'] / 2

        self.ax.annotate('', xy=(cx - radius, cy), xytext=(cx + radius, cy), 
                         arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
        self.ax.text(cx, cy + radius + 10, text, ha='center', va='bottom', color='black', fontsize=8)
        self.drawn_diameters.add(val)

    def draw_dynamic_dim(self, p1, p2, label_val, axis='x'):
        val_rounded = round(abs(label_val), 2)
        if val_rounded < 1.0: return
        
        # Global uniqueness check per axis
        drawn_set = self.drawn_values_x if axis == 'x' else self.drawn_values_y
        if val_rounded in drawn_set:
            return
            
        val_str = f"{val_rounded:.2f}"

        base_offset = 30
        step_offset = 35
        max_attempts = 50

        # Switch to architectural ticks for tight gaps
        dist = abs(p2[0] - p1[0]) if axis == 'x' else abs(p2[1] - p1[1])
        arrow_style = dict(arrowstyle='|-|' if dist < 25 else '<->', color='black', lw=0.6, shrinkA=0, shrinkB=0)
        text_style = dict(ha='center', va='center', fontsize=8, color='black',
                          bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8))
        ext_style = dict(color='black', lw=0.4, linestyle=':', alpha=0.5)

        gap = 4 # Visual gap from the object

        if axis == 'x':
            x_start, x_end = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_ref = min(p1[1], p2[1])

            for i in range(max_attempts):
                y_pos = y_ref - (base_offset + (i * step_offset))
                proposed_rect = (x_start - 10, y_pos - 10, x_end + 10, y_pos + 10)

                if not self._is_overlapping(proposed_rect):
                    self.ax.annotate('', xy=(p1[0], y_pos), xytext=(p2[0], y_pos), arrowprops=arrow_style)
                    self.ax.text((x_start + x_end) / 2, y_pos, val_str, **text_style)
                    
                    ext_y1 = p1[1] - gap if y_pos < p1[1] else p1[1] + gap
                    self.ax.plot([p1[0], p1[0]], [ext_y1, y_pos], **ext_style)
                    self.ax.plot([p2[0], p2[0]], [ext_y1, y_pos], **ext_style)
                    
                    self._add_occupied_zone(proposed_rect)
                    drawn_set.add(val_rounded)
                    return

        elif axis == 'y':
            y_start, y_end = min(p1[1], p2[1]), max(p1[1], p2[1])
            x_ref = min(p1[0], p2[0])

            for i in range(max_attempts):
                x_pos = x_ref - (base_offset + (i * step_offset))
                proposed_rect = (x_pos - 10, y_start - 10, x_pos + 10, y_end + 10)

                if not self._is_overlapping(proposed_rect):
                    self.ax.annotate('', xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]), arrowprops=arrow_style)
                    self.ax.text(x_pos, (y_start + y_end) / 2, val_str, rotation=90, **text_style)
                    
                    ext_x1 = p1[0] - gap if x_pos < p1[0] else p1[0] + gap
                    self.ax.plot([ext_x1, x_pos], [p1[1], p1[1]], **ext_style)
                    self.ax.plot([ext_x1, x_pos], [p2[1], p2[1]], **ext_style)
                    
                    self._add_occupied_zone(proposed_rect)
                    drawn_set.add(val_rounded)
                    return

    def get_unique_levels(self, lines, key):
        if not lines: return []
        levels = []
        sorted_lines = sorted(lines, key=lambda s: s[key])
        
        current_level = sorted_lines[0][key]
        levels.append(current_level)
        
        for s in sorted_lines[1:]:
            if abs(s[key] - current_level) > self.tol:
                current_level = s[key]
                levels.append(current_level)
        return levels

    def process(self):
        if not self.shapes: return

        h_lines = [s for s in self.shapes if s['type'] == 'horizontal']
        v_lines = [s for s in self.shapes if s['type'] == 'vertical']
        circles = [s for s in self.shapes if s['type'] == 'circle']
        boundaries = [s for s in self.shapes if s['type'] == 'boundary']
        
        main_boundary = max(boundaries, key=lambda s: s['width'] * s['height']) if boundaries else None

        for c in circles:
            self.draw_diameter(c)

        y_levels = self.get_unique_levels(h_lines, 'cy')
        x_levels = self.get_unique_levels(v_lines, 'cx')

        # 1. PITCHES (Inner-most dimensions)
        for i in range(len(y_levels) - 1):
            pitch = y_levels[i+1] - y_levels[i]
            self.draw_dynamic_dim((self.global_min_x, y_levels[i]), (self.global_min_x, y_levels[i+1]), pitch, axis='y')

        for i in range(len(x_levels) - 1):
            pitch = x_levels[i+1] - x_levels[i]
            self.draw_dynamic_dim((x_levels[i], self.global_min_y), (x_levels[i+1], self.global_min_y), pitch, axis='x')

        # 2. MARGINS (Distance from outer boundary to grid)
        if main_boundary:
            b = main_boundary
            if y_levels:
                bot_margin = y_levels[0] - b['min_y']
                self.draw_dynamic_dim((b['min_x'], b['min_y']), (b['min_x'], y_levels[0]), bot_margin, axis='y')
                
                top_margin = b['max_y'] - y_levels[-1]
                self.draw_dynamic_dim((b['min_x'], y_levels[-1]), (b['min_x'], b['max_y']), top_margin, axis='y')

            if x_levels:
                left_margin = x_levels[0] - b['min_x']
                self.draw_dynamic_dim((b['min_x'], b['min_y']), (x_levels[0], b['min_y']), left_margin, axis='x')
                
                right_margin = b['max_x'] - x_levels[-1]
                self.draw_dynamic_dim((x_levels[-1], b['min_y']), (b['max_x'], b['min_y']), right_margin, axis='x')

        # 3. SHAPE LENGTHS (Physical size of the grid)
        if h_lines:
            longest_h = max(h_lines, key=lambda s: s['width'])
            self.draw_dynamic_dim((longest_h['min_x'], self.global_min_y), (longest_h['max_x'], self.global_min_y), longest_h['width'], axis='x')
        
        if v_lines:
            longest_v = max(v_lines, key=lambda s: s['height'])
            self.draw_dynamic_dim((self.global_min_x, longest_v['min_y']), (self.global_min_x, longest_v['max_y']), longest_v['height'], axis='y')

        # 4. OVERALL BOUNDS (Furthest out, wrapping everything else)
        w = self.global_max_x - self.global_min_x
        h = self.global_max_y - self.global_min_y
        self.draw_dynamic_dim((self.global_min_x, self.global_min_y), (self.global_max_x, self.global_min_y), w, axis='x')
        self.draw_dynamic_dim((self.global_min_x, self.global_min_y), (self.global_min_x, self.global_max_y), h, axis='y')


def add_smart_dimensions(ax, segments):
    dim = SmartDimensioner(ax, segments)
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
        plt.subplots_adjust(bottom=0.15)

        slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
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
