import re
import sys
import os


def process_cnc_file(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Read the input file
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. THE MAIN TRANSFORMATION (MOVI + 5 ARCS -> MOVI + 2 ARCS)
    # -----------------------------------------------------------
    # Pattern explanation:
    # (MOVI...) -> Group 1
    # ARC X(val)Y(val)... -> Group 2 & 3 (New MOVI coords)
    # (ARC X(val)Y(val) a=(val)) -> Group 4, 5, 6 (Second ARC to keep)
    # Next 3 ARCs are matched but not grouped (to be deleted)
    main_pattern = r"MOVI\s+X[\d\.]+Y[\d\.]+\nARC\s+X([\d\.]+)Y([\d\.]+)\s+a=[\d\.-]+\n(ARC\s+X([\d\.]+)Y([\d\.]+)\s+a=([\d\.-]+))\nARC\s+X[\d\.]+Y[\d\.]+\s+a=[\d\.-]+\nARC\s+X[\d\.]+Y[\d\.]+\s+a=[\d\.-]+\nARC\s+X[\d\.]+Y[\d\.]+\s+a=[\d\.-]+"

    def replacement_func(match):
        new_movi_x = match.group(1)
        new_movi_y = match.group(2)
        second_arc_line = match.group(3)
        second_arc_x = float(match.group(4))
        second_arc_y = float(match.group(5))

        # Calculate coordinates for the new third ARC
        calc_x = round(second_arc_x - 43.84, 2)
        calc_y = round(second_arc_y + 7.76, 2)
        new_third_arc = f"ARC X{calc_x}Y{calc_y} a=-200"

        return f"MOVI X{new_movi_x}Y{new_movi_y}\n{second_arc_line}\n{new_third_arc}"

    # Apply the logic for the 5-arc blocks
    content = re.sub(main_pattern, replacement_func, content)

    # 2. CLEANUP STRAY -180 LINES
    # -----------------------------------------------------------
    # This removes lines that contain only "-180" (potentially with whitespace)
    content = re.sub(r"^\s*-180\s*$", "", content, flags=re.MULTILINE)

    # Remove any double newlines created by the deletion
    content = re.sub(r"\n\s*\n", "\n", content)

    # 3. SAVE THE OUTPUT
    # -----------------------------------------------------------
    file_dir, file_name = os.path.split(file_path)
    name_part, extension = os.path.splitext(file_name)
    output_filename = f"{name_part}_fixed{extension}"
    output_path = os.path.join(file_dir, output_filename)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Success! Processed file saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path/to/your/file.txt")
    else:
        process_cnc_file(sys.argv[1])