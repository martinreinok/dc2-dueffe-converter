import sys
import re


def parse_dc2_line(line):
    """Parses a line of text into a list of floats if possible."""
    try:
        parts = line.strip().split()
        return [float(x.replace(',', '')) for x in parts]
    except ValueError:
        return []


def get_entity_type(code_line):
    """Determines if the entity is a Line (1) or Arc/Curve (16/34)."""
    parts = code_line.split()
    if not parts: return None
    code = int(parts[0])
    if code == 1: return "LINE"
    if code in [16, 21, 34]: return "ARC"  # 16 and 34 appear to be curves in your files
    return "UNKNOWN"


def convert_dc2_to_cnc(input_filename, output_filename, scale_factor=0.125):
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # --- Header Generation ---
    cnc_output = []
    cnc_output.append("BLOCK VA1\nVELL 166.67\nACCL 333.33\nw195=1\nENDBL")
    cnc_output.append("BLOCK VA2\nVELL 133.33\nACCL 166.67\nw195=2\nENDBL")
    cnc_output.append("BLOCK VA3\nVELL 100\nACCL 133.33\nw195=3\nENDBL")
    cnc_output.append(";\nPROGRAM LAV1\n;")

    # Extract Design Name from filename
    design_name = input_filename.split('.')[0].upper()
    cnc_output.append(f"DISEGNO: {design_name}")

    # Setup Commands
    cnc_output.append("ABS X=0\nABS Y=0\nCORNER a=333.33")
    cnc_output.append("VEL X= 80\nVEL Y= 80\nACC X= 30\nACC Y= 30")
    cnc_output.append("v990=1\nv991=1\nCALL INIZIO\nLABEL 1\nCALL INLAV1\nCALL VA1")

    # --- Entity Parsing Loop ---
    # We need to skip the DC2 header (first 6 lines usually)
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "*":  # The * usually marks end of header in DC2
            start_index = i + 1
            break

    current_entity_type = None
    current_coords = []

    # Simple logic: Single Head (DW11) for now.
    # Dual head logic requires finding duplicate rows, which is complex for step 1.

    head_mode = "DW11"

    i = start_index
    while i < len(lines):
        line = lines[i].strip()

        # Check for Entity Definition (e.g., "1 8 16...")
        if len(line.split()) > 8 and line.split()[0].isdigit():
            # Process the PREVIOUS entity before starting new one
            if current_coords:
                process_entity(current_coords, current_entity_type, cnc_output, head_mode)
                current_coords = []

            current_entity_type = get_entity_type(line)
            i += 1
            continue

        # Parse Coordinates
        coords = parse_dc2_line(line)
        if len(coords) >= 2:
            # Apply Scale Factor immediately
            x = coords[0] * scale_factor
            y = coords[1] * scale_factor
            current_coords.append((x, y))

        i += 1

    # Process final entity
    if current_coords:
        process_entity(current_coords, current_entity_type, cnc_output, head_mode)

    # --- Footer ---
    cnc_output.append("CALL UP1")
    cnc_output.append("CALL STOFF")
    cnc_output.append("MR X=v993 Y=v994")
    cnc_output.append("CALL FINLAV1\nCALL FINECIC1")
    cnc_output.append("IF (w92=1) JUMP 1\nCALL FINE\nENDPR")

    # Write to file
    with open(output_filename, 'w') as f:
        f.write('\n'.join(cnc_output))
    print(f"Successfully created {output_filename}")


def process_entity(coords, type, output_list, head_mode):
    if not coords: return

    # Move to Start Point (Jump)
    start_x, start_y = coords[0]

    # Optional: Safety lift if distance is large?
    output_list.append("CALL UP1")  # Ensure needle up before moving
    output_list.append(f"MR X{start_x:.4g}Y{start_y:.4g}")

    # Needle Down
    output_list.append(f"CALL {head_mode}")

    if type == "LINE":
        # For lines, we output MI (Move Interpolated) for each point
        for x, y in coords[1:]:
            output_list.append(f"MI X{x:.4g}Y{y:.4g}")

    elif type == "ARC":
        # For arcs, DC2 gives points on the arc. Dueffe uses ARC commands.
        # This is a simplification; precise Arc fitting from points requires math.
        # For now, we treat DC2 arc points as segments of a curve.
        # In the sample, "a=180" implies semi-circles.

        for p_idx in range(1, len(coords)):
            x, y = coords[p_idx]
            # Heuristic: If it's a small recurring pattern, it uses ARC
            # We output ARC with a placeholder angle for now, or straight MI if unsure.
            # To get exact "a=180", we'd need to calculate the angle between vectors.

            # Use 180 as default for "loops" based on the majestic/ilva files
            output_list.append(f"ARC X{x:.4g}Y{y:.4g} a=180")

    output_list.append("SYNC")  # Ensure moves complete


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dueffe.py <input_file.dc2>")
    else:
        input_file = sys.argv[1]
        output_file = input_file.replace(".dc2", "_NEW.CNC")
        convert_dc2_to_cnc(input_file, output_file)