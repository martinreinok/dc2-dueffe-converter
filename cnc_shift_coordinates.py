import re
import sys


def offset_cnc_data(data, offset=5.0):
    targets = ["CALL QLY", "CALL QLYZ", "MR", "MI", "MOVI", "ARC"]
    cmd_regex = re.compile(rf'^(\s*)({"|".join(targets)})\b(.*)', re.I)
    axis_regex = re.compile(r'([XYZ])\s*=?\s*(-?\d+\.?\d*)', re.I)

    def update_line(line):
        match = cmd_regex.match(line)
        if not match:
            return line

        indent, command, params = match.groups()

        def add_offset(m):
            prefix = m.group(1) if len(m.groups()) > 1 else ""
            val = float(m.group(1 if prefix == "" else 2))
            return f"{prefix}{val + offset:g}"

        if not any(a in params.upper() for a in "XYZ"):
            new_params = re.sub(r'(-?\d+\.?\d*)', add_offset, params)
        else:
            new_params = axis_regex.sub(add_offset, params)

        return f"{indent}{command}{new_params}"

    return "\n".join(update_line(line) for line in data.splitlines())


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "drawing.CNC"
    OFFSET = 15

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(offset_cnc_data(content, offset=OFFSET))
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")