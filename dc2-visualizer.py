import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
# matplotlib.use('macosx')


def norm(a):
    return (a + 2 * np.pi) % (2 * np.pi)


class DC2Visualizer:
    """Visualizer for DC2 CAD files with interactive drawing order display."""

    def __init__(self, filename):
        self.ymax = None
        self.ymin = None
        self.xmax = None
        self.xmin = None
        self.filename = filename
        self.objects = []
        self.scale_factor = 1.0
        self.parse_file()

    def compute_bounds(self):
        """Compute global bounding box for all objects."""
        xs = []
        ys = []

        for obj in self.objects:
            if obj['type'] == 'line':
                for x, y in obj['coords']:
                    xs.append(x)
                    ys.append(y)
            elif obj['type'] == 'arc':
                cx, cy = obj['center']
                # approximate radius from start
                r = np.sqrt((obj['start'][0] - cx) ** 2 + (obj['start'][1] - cy) ** 2)
                # include bounding box of full circle (safe because arcs are partial)
                xs.extend([cx - r, cx + r])
                ys.extend([cy - r, cy + r])

        if xs and ys:
            self.xmin, self.xmax = min(xs) - 200, max(xs) + 200
            self.ymin, self.ymax = min(ys) - 200, max(ys) + 200
        else:
            self.xmin = self.ymin = 0
            self.xmax = self.ymax = 1

    def parse_file(self):
        """Parse DC2 file and extract drawing objects."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # Parse header
        if len(lines) > 2:
            scale_line = lines[2].strip()
            try:
                self.scale_factor = float(scale_line.split(',')[0])
            except (ValueError, IndexError):
                self.scale_factor = 1.0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Parse line objects
            if line.startswith('1 '):
                # Returns a list of segments and the new index
                new_objs, next_i = self._parse_line_object(lines, i)
                if new_objs:
                    self.objects.extend(new_objs)
                i = next_i
                continue

            # Parse arc objects
            elif line.startswith('16 4'):
                obj = self._parse_arc_object(lines, i)
                if obj:
                    self.objects.append(obj)
                    i = obj['next_index']
                    continue

            i += 1
        self.compute_bounds()

    def _get_visible_color(self, color_tuple):
        """
        Ensure color is visible on white background.
        If color is White (255, 255, 255) or None, return Black.
        """
        # Check for pure white or near white
        if color_tuple == (255, 255, 255):
            return (0, 0, 0)
        # You could add a brightness check here if needed,
        # but strict white replacement handles the common case.
        return color_tuple

    def _parse_line_object(self, lines, start_idx):
        """Parse a line object from DC2 file."""
        parts = lines[start_idx].strip().split()

        # Default to Black
        color = (0, 0, 0)

        # Extract color if available
        if len(parts) > 11:
            try:
                raw_color = (int(parts[9]), int(parts[10]), int(parts[11]))
                # Apply smart color fix (White -> Black)
                color = self._get_visible_color(raw_color)
            except (ValueError, IndexError):
                pass

        # Read coordinate pairs
        coords = []
        i = start_idx + 1
        while i < len(lines):
            coord_line = lines[i].strip()
            # Stop if we hit a new entity type or empty line
            if not coord_line or coord_line.startswith(('1 ', '16 ', '21 ')):
                break

            coord_parts = coord_line.split()
            if len(coord_parts) == 2:
                try:
                    coords.append((float(coord_parts[0]), float(coord_parts[1])))
                except ValueError:
                    break
            i += 1

        # Use Standard Polyline Parsing (Sliding Window)
        # This connects A->B, B->C, C->D.
        # This is required for standard files to have closed shapes.
        generated_objects = []

        if len(coords) >= 2:
            for k in range(len(coords) - 1):
                segment = coords[k: k + 2]
                generated_objects.append({
                    'type': 'line',
                    'coords': segment,
                    'color': color
                })

        return generated_objects, i

    def _parse_arc_object(self, lines, start_idx):
        """Parse an arc object from DC2 file."""
        parts = lines[start_idx].strip().split()

        # Default to Black
        color = (0, 0, 0)

        # Extract color
        if len(parts) > 11:
            try:
                raw_color = (int(parts[9]), int(parts[10]), int(parts[11]))
                color = self._get_visible_color(raw_color)
            except (ValueError, IndexError):
                pass

        # Read arc definition points
        if start_idx + 4 < len(lines):
            try:
                center = list(map(float, lines[start_idx + 1].strip().split()))
                start = list(map(float, lines[start_idx + 2].strip().split()))
                end = list(map(float, lines[start_idx + 3].strip().split()))
                control = list(map(float, lines[start_idx + 4].strip().split()))

                return {
                    'type': 'arc',
                    'center': center,
                    'start': start,
                    'end': end,
                    'control': control,
                    'color': color,
                    'next_index': start_idx + 5
                }
            except (ValueError, IndexError):
                pass

        return None

    def _draw_line_segment(self, ax, coords, color, alpha=1.0, show_points=False):
        """Draw a line segment."""
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        # Normalize color (0-255 -> 0.0-1.0)
        norm_color = tuple(c / 255 for c in color)

        ax.plot(x_coords, y_coords, '-', color=norm_color,
                linewidth=2.0, alpha=alpha)

        if show_points:
            # Mark start point (Green) and end point (Red)
            ax.plot(x_coords[0], y_coords[0], 'o', color='green',
                    markersize=6, zorder=5)
            ax.plot(x_coords[-1], y_coords[-1], 'o', color='red',
                    markersize=6, zorder=5)

    def _draw_arc_segment(self, ax, arc_data, color, alpha=1.0, show_points=False):
        """Draw an arc segment."""
        center = arc_data['center']
        start = arc_data['start']
        end = arc_data['end']
        control = arc_data['control']

        # Calculate radius
        radius = np.sqrt((start[0] - center[0]) ** 2 + (start[1] - center[1]) ** 2)

        # Calculate angles
        start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
        end_angle = np.arctan2(end[1] - center[1], end[0] - center[0])
        control_angle = np.arctan2(control[1] - center[1], control[0] - center[0])

        # Determine arc direction
        a = norm(start_angle)
        b = norm(end_angle)
        c = norm(control_angle)

        # Compute arc lengths
        ab = (b - a) % (2 * np.pi)
        ac = (c - a) % (2 * np.pi)

        if ac < ab:
            # CCW
            theta = np.linspace(a, a + ab, 100)
        else:
            # CW
            theta = np.linspace(a, a - (2 * np.pi - ab), 100)

        arc_x = center[0] + radius * np.cos(theta)
        arc_y = center[1] + radius * np.sin(theta)

        norm_color = tuple(c / 255 for c in color)

        ax.plot(arc_x, arc_y, '-', color=norm_color, linewidth=2.0, alpha=alpha)

        if show_points:
            ax.plot(center[0], center[1], '+', color='orange',
                    markersize=8, alpha=0.5, zorder=4)
            ax.plot(start[0], start[1], 'o', color='green',
                    markersize=6, zorder=5)
            ax.plot(end[0], end[1], 'o', color='red',
                    markersize=6, zorder=5)

    def visualize_interactive(self):
        """Create an interactive visualization showing drawing order."""
        if not self.objects:
            print("No objects to visualize!")
            return

        # Standard white background setup
        fig, ax = plt.subplots(figsize=(16, 11))

        plt.subplots_adjust(bottom=0.15)

        # Create slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax_slider, 'Drawing Step',
            0, len(self.objects),
            valinit=len(self.objects),
            valstep=1
        )

        def update(val):
            """Update visualization based on slider value."""
            # Save current zoom
            curr_xlim = ax.get_xlim()
            curr_ylim = ax.get_ylim()

            ax.cla()

            # Restore settings
            ax.set_xlim(curr_xlim)
            ax.set_ylim(min(curr_ylim), max(curr_ylim))

            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)  # Standard light grid

            step = int(slider.val)

            # Draw objects
            for i, obj in enumerate(self.objects[:step]):
                is_current = (i == step - 1)
                alpha = 1.0 if is_current else 0.5
                show_points = is_current

                if obj['type'] == 'line':
                    self._draw_line_segment(ax, obj['coords'], obj['color'],
                                            alpha=alpha, show_points=show_points)
                elif obj['type'] == 'arc':
                    self._draw_arc_segment(ax, obj, obj['color'],
                                           alpha=alpha, show_points=show_points)

            # Title + labels
            ax.set_title(
                f'DC2 Drawing Order - Step {step}/{len(self.objects)}\n'
                f'File: {self.filename}',
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Initial draw
        update(len(self.objects))
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        # Reset Toolbar home logic
        try:
            tb = fig.canvas.toolbar
            tb._views.clear()
            tb._positions.clear()
            ax._push_view()
        except:
            pass

        plt.show()

        print(f"Interactive visualization ready!")
        print(f"Total objects: {len(self.objects)}")


# Main execution
if __name__ == "__main__":
    visualizer = DC2Visualizer(R"drawing.dc2")

    print("\nGenerating interactive visualization...")
    visualizer.visualize_interactive()