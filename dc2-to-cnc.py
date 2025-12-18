import sys
import os
import math
import argparse

# --- Constants ---
DEFAULT_OFFSET = 680.0  # Default distance between Head 1 and Head 2 in mm


class Vector:
    """Helper for vector math."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def angle(self):
        return math.atan2(self.y, self.x)


class DC2Entity:
    """Base class for DC2 entities."""

    def __init__(self, color=None):
        self.color = color


class DC2Line(DC2Entity):
    def __init__(self, points, color=None):
        super().__init__(color)
        self.points = points  # List of (x, y) tuples


class DC2Arc(DC2Entity):
    def __init__(self, center, start, end, control, color=None):
        super().__init__(color)
        self.center = center
        self.start = start
        self.end = end
        self.control = control


class DC2Parser:
    """Parses .dc2 files into geometric entities with Color detection."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.entities = []
        self.parse()

    def parse(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with open(self.filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            return

        # --- Header Parsing ---
        try:
            parts0 = lines[0].split()
            self.offset_x = float(parts0[0])
            self.offset_y = float(parts0[1])
        except (IndexError, ValueError):
            self.offset_x, self.offset_y = 0.0, 0.0

        if len(lines) > 3:
            try:
                val = lines[3].split(',')[0].strip()
                self.scale_factor = float(val)
                if self.scale_factor == 0: self.scale_factor = 1.0
            except ValueError:
                self.scale_factor = 1.0

        print(f"Parsed DC2 - Scale: {self.scale_factor}, Offset: {self.offset_x},{self.offset_y}")

        # --- Entity Parsing ---
        i = 0
        while i < len(lines):
            line = lines[i]
            parts = line.split()

            # Identify Primitive Type (1=Line, 16=Arc, 21/23/34=Groups/Markers)
            prim_type = parts[0]

            if prim_type == '1':  # Line
                # Format: 1 {style} {layer?} ... {R} {G} {B}
                # Color is usually at the end.
                color = None
                if len(parts) >= 12:
                    try:
                        color = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
                    except:
                        pass

                points = []
                i += 1
                while i < len(lines):
                    pt_line = lines[i]
                    if pt_line.startswith(('1 ', '16 ', '21 ', '23 ', '34 ', '*')):
                        i -= 1
                        break
                    try:
                        coords = list(map(float, pt_line.split()))
                        if len(coords) == 2:
                            points.append(tuple(coords))
                    except ValueError:
                        pass
                    i += 1

                if len(points) >= 2:
                    self.entities.append(DC2Line(points, color))

            elif prim_type == '16':  # Arc
                # Format: 16 4 {layer?} ... {R} {G} {B}
                color = None
                if len(parts) >= 12:
                    try:
                        color = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
                    except:
                        pass

                if i + 4 < len(lines):
                    try:
                        c = list(map(float, lines[i + 1].split()))
                        s = list(map(float, lines[i + 2].split()))
                        e = list(map(float, lines[i + 3].split()))
                        ctrl = list(map(float, lines[i + 4].split()))

                        self.entities.append(DC2Arc(c, s, e, ctrl, color))
                        i += 4
                    except (ValueError, IndexError):
                        pass

            i += 1


class CNCGenerator:
    """Generates .CNC code with Dual Head support based on Layer Colors."""

    def __init__(self, parser, manual_offset=None):
        self.parser = parser
        self.commands = []

        # State tracking
        self.current_x = None
        self.current_y = None
        self.head_down = False
        self.active_tool_mode = None  # 1 = Single, 3 = Dual

        # Calculate Offset from 'SEPARAR' (Yellow) layer or use default
        self.head_offset = self.calculate_offset_from_separar(manual_offset)

        base = os.path.basename(parser.filepath)
        self.prog_name = os.path.splitext(base)[0].upper()

    def calculate_offset_from_separar(self, manual_val):
        """Finds length of the line in SEPARAR layer (Yellow 255,255,0)."""
        calculated = None

        for ent in self.parser.entities:
            if isinstance(ent, DC2Line) and ent.color == (255, 255, 0):
                # Found Yellow line
                p1 = ent.points[0]
                p2 = ent.points[-1]
                # Calculate Y projection length in mm
                dy = abs(p2[1] - p1[1]) / self.parser.scale_factor

                if dy > 10:  # Filter out tiny markers
                    calculated = dy
                    break

        if calculated:
            print(f"Auto-detected Head Offset from 'SEPARAR': {calculated:.2f} mm")
            return calculated

        val = manual_val if manual_val else DEFAULT_OFFSET
        print(f"Using Default/Manual Head Offset: {val:.2f} mm")
        return val

    def get_role(self, ent):
        """Determines entity role based on Color."""
        c = ent.color
        if c == (255, 0, 255): return "DUAL"  # Magenta -> COSER_A
        if c == (0, 255, 0):   return "SKIP"  # Green   -> COSER_B (Visual only)
        if c == (255, 255, 0): return "SKIP"  # Yellow  -> SEPARAR (Setup only)
        if c == (255, 0, 0):   return "SKIP"  # Red     -> Frame/Layer 0 (Skip)
        return "SINGLE"  # Others  -> CONTORNO

    def transform(self, x, y):
        """Scale DC2 units to mm."""
        tx = (x - self.parser.offset_x) / self.parser.scale_factor
        ty = (y - self.parser.offset_y) / self.parser.scale_factor
        return round(tx, 2), round(ty, 2)

    def add_cmd(self, line):
        self.commands.append(line)

    def set_tool_mode(self, mode, start_y):
        """Switches between Single (DW11) and Dual (DW13) modes."""
        if mode == 3:  # Dual Head (COSER_A)
            # Calculate Z for Head 2
            h1_y = start_y
            h2_z = round(start_y + self.head_offset, 2)

            # Setup Sequence
            if self.active_tool_mode != 3:
                self.add_cmd(f"MR Y{h1_y}")
                self.add_cmd(f"CALL QLYZ {h1_y} {h2_z}")
                self.add_cmd(f"MR X{self.current_x if self.current_x else 0}Y{h1_y}Z{h2_z}")
                self.add_cmd(";")
                self.add_cmd("CALL ELYZ")
                self.add_cmd("CALL DW13")

            self.active_tool_mode = 3

        else:  # Single Head (CONTORNO)
            if self.active_tool_mode != 1:
                self.add_cmd(";")
                self.add_cmd("CALL ELY")
                self.add_cmd("CALL DW11")

            self.active_tool_mode = 1

    def format_coord_string(self, x, y):
        """Generates coordinate string handling Z if in dual mode."""
        base = f"X{x}Y{y}"
        if self.active_tool_mode == 3:
            z = round(y + self.head_offset, 2)
            base += f"Z{z}"
        return base

    def move_rapid(self, x, y):
        if self.head_down:
            self.add_cmd("CALL UP1")
            self.head_down = False

        coords = self.format_coord_string(x, y)
        self.add_cmd(f"MR {coords}")
        self.current_x = x
        self.current_y = y

    def move_cut(self, x, y):
        if not self.head_down:
            if self.active_tool_mode == 3:
                self.add_cmd("CALL DW13")
            else:
                self.add_cmd("CALL DW11")
            self.head_down = True

        coords = self.format_coord_string(x, y)
        self.add_cmd(f"MI {coords}")
        self.current_x = x
        self.current_y = y

    def move_arc(self, end_x, end_y, angle):
        if not self.head_down:
            if self.active_tool_mode == 3:
                self.add_cmd("CALL DW13")
            else:
                self.add_cmd("CALL DW11")
            self.head_down = True

        a_str = f"{angle:.2f}".rstrip('0').rstrip('.')

        self.add_cmd("FREEZE")
        self.add_cmd(f"ARC X{end_x}Y{end_y} a={a_str}")
        self.add_cmd("SYNC")

        self.current_x = end_x
        self.current_y = end_y

    def calculate_arc_angle(self, center, start, end, control):
        vs = Vector(start[0] - center[0], start[1] - center[1])
        ve = Vector(end[0] - center[0], end[1] - center[1])
        vc = Vector(control[0] - center[0], control[1] - center[1])

        def norm(a): return (a + 2 * math.pi) % (2 * math.pi)

        ns, ne, nc = norm(vs.angle()), norm(ve.angle()), norm(vc.angle())

        span_se = (ne - ns) % (2 * math.pi)
        span_sc = (nc - ns) % (2 * math.pi)

        is_ccw = span_sc < span_se
        return math.degrees(span_se) if is_ccw else -math.degrees((2 * math.pi) - span_se)

    def generate(self):
        # Header
        self.add_cmd("BLOCK VA1\nVELL 166.67\nACCL 333.33\nw195=1\nENDBL")
        self.add_cmd("BLOCK VA2\nVELL 133.33\nACCL 166.67\nw195=2\nENDBL")
        self.add_cmd("BLOCK VA3\nVELL 100\nACCL 133.33\nw195=3\nENDBL")
        self.add_cmd(";")
        self.add_cmd(f"PROGRAM LAV1\n; DISEGNO: {self.prog_name}")
        self.add_cmd("ABS X=0\nABS Y=0\nCORNER a=333.33")
        self.add_cmd("VEL X= 80\nVEL Y= 80\nACC X= 30\nACC Y= 30")
        self.add_cmd("v990=1\nv991=1\nCALL INIZIO")
        self.add_cmd("LABEL 1\nCALL INLAV1\nCALL VA1")

        # Start logic (Safety QLY)
        self.add_cmd("CALL FLZ")
        self.add_cmd("CALL QLY 150")  # Safety
        self.add_cmd("MR X0Y150")
        self.add_cmd(";")

        first_move = True

        for ent in self.parser.entities:
            # Check Role
            role = self.get_role(ent)
            if role == "SKIP":
                continue

            # Determine Mode based on Role
            target_mode = 3 if role == "DUAL" else 1

            # Determine geometry type
            is_arc = isinstance(ent, DC2Arc)

            # Determine start point for moves
            if is_arc:
                raw_start = ent.start
            else:
                raw_start = ent.points[0]

            sx, sy = self.transform(raw_start[0], raw_start[1])

            # --- Tool Change / Setup Logic ---
            if target_mode != self.active_tool_mode:
                # Force pen up before switching modes
                if self.head_down:
                    self.add_cmd("CALL UP1")
                    self.head_down = False

                self.set_tool_mode(target_mode, sy)
                first_move = False  # set_tool_mode handles the initial move

            # --- Movement Logic ---
            if first_move:
                self.move_rapid(sx, sy)
                first_move = False
            else:
                # Only move if significant distance
                dist = math.sqrt((sx - (self.current_x or 0)) ** 2 + (sy - (self.current_y or 0)) ** 2)
                if dist > 1.0:
                    self.move_rapid(sx, sy)

            # --- Cutting Logic ---
            if is_arc:
                ex, ey = self.transform(ent.end[0], ent.end[1])
                angle = self.calculate_arc_angle(ent.center, ent.start, ent.end, ent.control)
                self.move_arc(ex, ey, angle)
            else:
                # Line
                for p in ent.points[1:]:
                    px, py = self.transform(p[0], p[1])
                    self.move_cut(px, py)

        # Footer
        if self.head_down:
            self.add_cmd("CALL UP1")

        self.add_cmd("CALL STOFF")
        self.add_cmd("MR X=v993 Y=v994")
        self.add_cmd("CALL FINLAV1")
        self.add_cmd("CALL FINECIC1")
        self.add_cmd("IF (w92=1) JUMP 1")
        self.add_cmd("CALL FINE")
        self.add_cmd("ENDPR")

    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.commands))
        print(f"Generated CNC: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DC2 to Dual-Head CNC")
    parser.add_argument("input", help="Input .dc2 file")
    parser.add_argument("output", nargs='?', help="Output .CNC file")
    parser.add_argument("--offset", type=float, default=None,
                        help="Manually override Head 2 offset (mm)")

    args = parser.parse_args()

    out_path = args.output if args.output else os.path.splitext(args.input)[0] + ".CNC"

    try:
        p = DC2Parser(args.input)
        g = CNCGenerator(p, args.offset)
        g.generate()
        g.save(out_path)
    except Exception as e:
        print(f"Error: {e}")