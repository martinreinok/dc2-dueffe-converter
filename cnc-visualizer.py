import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import re
import sys


class CNCVisualizer:
    """
    Visualizer for Dueffe .CNC files.
    Simulates machine state (Needle Up/Down, Head 1/2) and geometry.
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
        # Matches a=180 or a=-90
        angle_pattern = re.compile(r'a=([-\d\.]+)')

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
                # Assuming format CALL QLYZ <Head1Y> <Head2Y>
                # We want the difference for the Z offset
                try:
                    # Find numbers in the line
                    nums = [float(x) for x in parts if x.replace('.', '', 1).isdigit()]
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

            # Update Z offset if explicitly in move command (e.g. MR ... Z1510)
            if 'Z' in cmd and 'Y' in cmd:
                # If Z is present in a move, usually Z = Head 2 Y position
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
                segment['style'] = 'cut'  # Arcs are always cuts in this context
                segment['color'] = 'black'
                angle = cmd.get('a', 0)

                # Math to calculate Arc points
                pts = self.calculate_arc_points(start_x, start_y, target_x, target_y, angle)
                segment['head1_coords'] = pts

            # --- Dual Head Logic ---
            if self.dual_head and segment['style'] == 'cut':
                # Generate identical geometry offset by Z for Head 2
                h2_pts = []
                for x, y in segment['head1_coords']:
                    h2_pts.append((x, y + self.z_offset))
                segment['head2_coords'] = h2_pts

            # Only add segment if actual movement occurred
            if segment['head1_coords']:
                self.segments.append(segment)
                self.current_x = target_x
                self.current_y = target_y

        self.compute_bounds()

    def calculate_arc_points(self, x1, y1, x2, y2, angle_deg):
        """
        Calculates points for an arc given start, end, and included angle.
        Note: The sign of 'a' in Dueffe determines curvature direction.
        """
        if angle_deg == 0: return [(x1, y1), (x2, y2)]

        # Chord calculations
        dx = x2 - x1
        dy = y2 - y1
        d2 = dx * dx + dy * dy
        d = np.sqrt(d2)

        if d == 0: return []

        # Radius formula: R = d / (2 * sin(theta/2))
        theta_rad = np.radians(abs(angle_deg))
        R = d / (2 * np.sin(theta_rad / 2.0))

        # Distance from chord midpoint to center
        # h = sqrt(R^2 - (d/2)^2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        try:
            h = np.sqrt(max(0, R * R - d2 / 4))
        except ValueError:
            h = 0

        # Offset to center (perpendicular to chord)
        # Perpendicular vector to (dx, dy) is (-dy, dx)
        # Normalize and scale by h
        offset_x = -dy * h / d
        offset_y = dx * h / d

        # Determine direction based on sign of angle
        # Heuristic based on file analysis:
        # Positive angle usually curves left (CCW), Negative curves right (CW) relative to chord
        if angle_deg > 0:
            cx = mid_x - offset_x
            cy = mid_y - offset_y
        else:
            cx = mid_x + offset_x
            cy = mid_y + offset_y

        # Generate angles for Start and End
        angle_start = np.arctan2(y1 - cy, x1 - cx)
        angle_end = np.arctan2(y2 - cy, x2 - cx)

        # Handle wrap-around logic
        if angle_deg > 0:  # CCW
            if angle_end < angle_start: angle_end += 2 * np.pi
        else:  # CW
            if angle_end > angle_start: angle_end -= 2 * np.pi

        # Generate points
        t = np.linspace(angle_start, angle_end, 50)
        arc_xs = cx + R * np.cos(t)
        arc_ys = cy + R * np.sin(t)

        return list(zip(arc_xs, arc_ys))

    def compute_bounds(self):
        xs, ys = [], []
        for seg in self.segments:
            for pts in [seg['head1_coords'], seg['head2_coords']]:
                if not pts: continue
                x_pts, y_pts = zip(*pts)
                xs.extend(x_pts)
                ys.extend(y_pts)

        if xs:
            margin = 50
            self.xmin, self.xmax = min(xs) - margin, max(xs) + margin
            self.ymin, self.ymax = min(ys) - margin, max(ys) + margin

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"CNC Simulation: {self.filename}", fontsize=14)

        # Create Slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Step', 0, len(self.segments), valinit=len(self.segments), valstep=1)

        def update(val):
            step = int(slider.val)
            ax.cla()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)

            # Draw Segment Loop
            for i, seg in enumerate(self.segments[:step]):
                # Determine opacity (make past steps slightly faded, current step bold)
                alpha = 1.0 if i == step - 1 else 0.6
                lw = 2.0 if i == step - 1 else 1.0

                # Head 1
                if seg['head1_coords']:
                    xs, ys = zip(*seg['head1_coords'])
                    style = '--' if seg['style'] == 'jump' else '-'
                    ax.plot(xs, ys, style, color=seg['color'], alpha=alpha, linewidth=lw)

                # Head 2 (Red Shadow)
                if seg['head2_coords']:
                    xs2, ys2 = zip(*seg['head2_coords'])
                    ax.plot(xs2, ys2, '-', color='red', alpha=alpha * 0.5, linewidth=lw,
                            label='Head 2' if i == 0 else "")

            # Show current machine position marker
            if step > 0:
                last_seg = self.segments[step - 1]
                if last_seg['head1_coords']:
                    lx, ly = last_seg['head1_coords'][-1]
                    ax.plot(lx, ly, 'o', color='green', markersize=8, label='Head 1')
                    if last_seg['head2_coords']:
                        lx2, ly2 = last_seg['head2_coords'][-1]
                        ax.plot(lx2, ly2, 'o', color='red', markersize=8, label='Head 2')

            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.legend(loc='upper right')
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(len(self.segments))  # Initial draw
        plt.show()


if __name__ == "__main__":
    # You can change the filename here or pass it as an argument
    target_file = sys.argv[1] if len(sys.argv) > 1 else "machine-code.CNC"
    sim = CNCVisualizer(target_file)
    sim.visualize()