import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import re
import sys


class CNCVisualizer:
    """
    Visualizer for Dueffe .CNC files.
    Simulates machine state (Needle Up/Down, Head 1/2) and geometry.
    Supports Lines and Arcs (ARC X... Y... a=...)
    """

    def __init__(self, filename):
        self.filename = filename
        self.commands = []  # Raw parsed commands
        self.segments = []  # Processed drawing segments (lines/arcs)

        # Machine State
        self.current_x = 0.0
        self.current_y = 0.0
        self.z_offset = 0.0  # Offset for Head 2
        self.head_down = False  # Is needle down?
        self.dual_head = False  # Is Head 2 active? (DW13)

        # Plotting Bounds
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 1000

        self.parse_file()
        self.process_commands()

    def parse_file(self):
        """Reads the CNC file and extracts relevant commands."""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File {self.filename} not found.")
            sys.exit(1)

        # Regex patterns for parsing
        # Matches X123.45 Y-50.0 Z100 etc.
        coord_pattern = re.compile(r'([XYZ])([-\d\.]+)')
        # Matches a=180 or a=-90 (case insensitive)
        angle_pattern = re.compile(r'a=([-\d\.]+)', re.IGNORECASE)

        for line in lines:
            line = line.strip().upper()
            if not line or line.startswith(';'): continue

            cmd_data = {'raw': line, 'type': 'other'}

            # Detect command type
            if line.startswith('MR '):
                cmd_data['type'] = 'MR'
            elif line.startswith('MI '):
                cmd_data['type'] = 'MI'
            elif line.startswith('ARC '):
                cmd_data['type'] = 'ARC'
            elif 'CALL DW11' in line:
                cmd_data['type'] = 'DW11'
            elif 'CALL DW13' in line:
                cmd_data['type'] = 'DW13'
            elif 'CALL UP1' in line:
                cmd_data['type'] = 'UP'
            elif 'CALL QLYZ' in line:
                cmd_data['type'] = 'QLYZ'

            # Extract Coordinates (X, Y, Z)
            coords = coord_pattern.findall(line)
            for axis, val in coords:
                cmd_data[axis] = float(val)

            # Extract Angle (a) for Arcs
            angle_match = angle_pattern.search(line)
            if angle_match:
                cmd_data['a'] = float(angle_match.group(1))

            # Extract QLYZ arguments if present (format: CALL QLYZ val1 val2)
            if cmd_data['type'] == 'QLYZ':
                parts = line.split()
                try:
                    # Find numbers in the line
                    nums = [float(x) for x in parts if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(nums) >= 2:
                        cmd_data['z_calc'] = nums[1] - nums[0]
                except:
                    pass

            self.commands.append(cmd_data)

    def process_commands(self):
        """Simulates the machine run to generate drawing segments."""
        for cmd in self.commands:
            segment = {
                'head1_coords': [],
                'head2_coords': [],
                'style': 'jump',  # jump, cut, arc
                'color': 'gray',
                'description': cmd['raw']
            }

            start_x, start_y = self.current_x, self.current_y

            # --- State Changes ---
            if cmd['type'] == 'DW11':
                self.head_down = True
                self.dual_head = False
                continue
            elif cmd['type'] == 'DW13':
                self.head_down = True
                self.dual_head = True
                continue
            elif cmd['type'] == 'UP':
                self.head_down = False
                continue
            elif cmd['type'] == 'QLYZ':
                if 'z_calc' in cmd:
                    self.z_offset = cmd['z_calc']
                continue

            # --- Movement ---
            target_x = cmd.get('X', self.current_x)
            target_y = cmd.get('Y', self.current_y)

            # Update Z offset if explicitly in move command
            if 'Z' in cmd and 'Y' in cmd:
                self.z_offset = cmd['Z'] - cmd['Y']

            if cmd['type'] == 'MR':
                # Move Rapid (Jump)
                segment['style'] = 'jump'
                segment['color'] = 'blue'  # Dashed line
                segment['head1_coords'] = [(start_x, start_y), (target_x, target_y)]

            elif cmd['type'] == 'MI':
                # Move Interpolated (Linear Cut)
                segment['style'] = 'cut'
                segment['color'] = 'black'
                segment['head1_coords'] = [(start_x, start_y), (target_x, target_y)]

            elif cmd['type'] == 'ARC':
                # Circular Interpolation
                segment['style'] = 'cut'
                segment['color'] = 'black'
                angle = cmd.get('a', 0.0)

                # Math to calculate Arc points
                pts = self.calculate_arc_points(start_x, start_y, target_x, target_y, angle)
                segment['head1_coords'] = pts

            # --- Dual Head Logic ---
            if self.dual_head and segment['style'] == 'cut':
                h2_pts = []
                for x, y in segment['head1_coords']:
                    h2_pts.append((x, y + self.z_offset))
                segment['head2_coords'] = h2_pts

            # Only add segment if valid
            if segment['head1_coords']:
                self.segments.append(segment)
                self.current_x = target_x
                self.current_y = target_y

        self.compute_bounds()

    def calculate_arc_points(self, x1, y1, x2, y2, angle_deg):
        """
        Calculates points for an arc given start, end, and included angle.
        Uses robust Tangent half-angle formula to determine center.
        """
        if abs(angle_deg) < 0.01:
            return [(x1, y1), (x2, y2)]

        # Chord vector
        dx = x2 - x1
        dy = y2 - y1
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)

        if d < 0.001: return []

        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0

        # Calculate Center
        # Normalized Left Normal vector: (-dy/d, dx/d)
        nx = -dy / d
        ny = dx / d

        # k is signed distance from chord midpoint to center along Normal
        # k = (d/2) / tan(theta/2)
        angle_rad = np.radians(angle_deg)

        # Handle 180 degrees (Semicircle) specifically to avoid tan(90) infinity
        if abs(abs(angle_deg) - 180.0) < 0.01:
            cx = mid_x
            cy = mid_y
        else:
            try:
                k = (d / 2.0) / np.tan(angle_rad / 2.0)
                cx = mid_x + k * nx
                cy = mid_y + k * ny
            except ZeroDivisionError:
                # Should not happen given < 0.01 check, but fallback to mid
                cx = mid_x
                cy = mid_y

        # Calculate Radius
        radius = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)

        # Generate angles for Start and End
        start_angle = np.arctan2(y1 - cy, x1 - cx)
        end_angle = np.arctan2(y2 - cy, x2 - cx)

        # Handle Wrap-around based on direction
        # If angle_deg > 0, we want CCW path from start to end
        # If angle_deg < 0, we want CW path

        diff = end_angle - start_angle

        # Normalize diff to -2pi ... 2pi
        while diff <= -np.pi: diff += 2 * np.pi
        while diff > np.pi: diff -= 2 * np.pi

        if angle_deg > 0:
            # CCW: We want diff > 0
            if diff <= 0: diff += 2 * np.pi
        else:
            # CW: We want diff < 0
            if diff >= 0: diff -= 2 * np.pi

        actual_end_angle = start_angle + diff

        # Generate Points (High resolution for smoothness)
        # Using 100 points or roughly 1 point per mm/degree
        steps = max(20, int(abs(angle_deg) / 2))
        t = np.linspace(start_angle, actual_end_angle, steps)

        arc_xs = cx + radius * np.cos(t)
        arc_ys = cy + radius * np.sin(t)

        return list(zip(arc_xs, arc_ys))

    def compute_bounds(self):
        xs, ys = [], []
        for seg in self.segments:
            # Check both heads
            for pts in [seg['head1_coords'], seg['head2_coords']]:
                if not pts: continue
                x_pts, y_pts = zip(*pts)
                xs.extend(x_pts)
                ys.extend(y_pts)

        if xs:
            margin = 100
            self.xmin, self.xmax = min(xs) - margin, max(xs) + margin
            self.ymin, self.ymax = min(ys) - margin, max(ys) + margin
        else:
            self.xmax, self.ymax = 1000, 1000

    def visualize(self):
        fig, ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(bottom=0.15)

        ax.set_title(f"CNC Visualizer - {self.filename}", fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Progress', 0, len(self.segments), valinit=len(self.segments), valstep=1)

        def update(val):
            step = int(slider.val)
            ax.cla()  # Clear axis

            # Reset view properties
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_title(f"CNC Visualizer - {self.filename} (Step {step}/{len(self.segments)})")

            # Batch drawing for performance
            # Separate lists for Jump, Head1 Cut, Head2 Cut
            jump_xs, jump_ys = [], []
            h1_xs, h1_ys = [], []
            h2_xs, h2_ys = [], []

            # We draw simulation up to 'step'
            # To preserve colors per segment if needed, we iterate.
            # But simple visualization usually needs: Blue Dotted (Jump), Black (Head 1), Green (Head 2)

            for i in range(step):
                seg = self.segments[i]

                # Head 1
                if seg['head1_coords']:
                    pts = seg['head1_coords']
                    xs, ys = zip(*pts)
                    if seg['style'] == 'jump':
                        ax.plot(xs, ys, ':', color='blue', alpha=0.4, linewidth=1)
                    else:
                        ax.plot(xs, ys, '-', color='black', alpha=0.8, linewidth=1.5)

                # Head 2
                if seg['head2_coords']:
                    pts2 = seg['head2_coords']
                    xs2, ys2 = zip(*pts2)
                    ax.plot(xs2, ys2, '-', color='limegreen', alpha=0.6, linewidth=1.5)

            # Draw "Machine Head" position marker at the end of the last segment
            if step > 0:
                last = self.segments[step - 1]
                if last['head1_coords']:
                    lx, ly = last['head1_coords'][-1]
                    ax.plot(lx, ly, 'o', color='black', markersize=8, label='Head 1', zorder=10)

                    if last['head2_coords']:
                        lx2, ly2 = last['head2_coords'][-1]
                        ax.plot(lx2, ly2, 'o', color='limegreen', markersize=8, label='Head 2', zorder=10)

            # Add Legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='black', lw=2),
                Line2D([0], [0], color='limegreen', lw=2),
                Line2D([0], [0], color='blue', linestyle=':', lw=1)
            ]
            ax.legend(custom_lines, ['Head 1 (Cut)', 'Head 2 (Cut)', 'Rapid Move'], loc='upper right')

            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Trigger initial update
        update(len(self.segments))
        plt.show()


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "drawing.CNC"
    sim = CNCVisualizer(target_file)
    sim.visualize()