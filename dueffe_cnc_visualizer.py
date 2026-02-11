import re
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# try:
#     matplotlib.use("macosx")
# except Exception:
#     pass
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider


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

        def draw(step: int):
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

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "drawing.CNC"

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
