#!/usr/bin/env python3
"""
Flip (mirror) a Dueffe-style CNC drawing vertically (Y axis in the XY plane).

Default behavior:
  - Finds all numeric Y (and relevant Z used as "second-head Y") values
  - Computes Ymin/Ymax
  - Rewrites:
      Y  -> (Ymin+Ymax - Y)
      Z  -> (Ymin+Ymax - Z)   (only when Z appears as a coordinate in move commands / QLYZ)
      ARC a= -> negated sweep angle (only on ARC lines)

Usage:
  python cnc_flip_vertical.py input.CNC
  python cnc_flip_vertical.py input.CNC -o output.CNC
  python cnc_flip_vertical.py input.CNC --y-min 0 --y-max 2000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

# Commands we rewrite
MOVE_CMDS = {"MR", "MI", "MOVI", "ARC"}
CALL_CMDS = {"CALL QLY", "CALL QLYZ"}

# Coordinate tokens like X0Y150, X=0, Y=150, Z1170 etc.
COORD_RE = re.compile(r"(?<![A-Z])([XYZ])\s*=?\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

# Floats in QLY / QLYZ when no axis letters are used
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")

# ARC sweep angle "a=180"
ARC_ANGLE_RE = re.compile(r"(\ba\s*=\s*)([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

# Line command token (first word)
FIRST_TOKEN_RE = re.compile(r"^\s*([A-Z]+)\b", re.IGNORECASE)


def _format_like(original: str, value: float) -> str:
    """
    Format 'value' similarly to 'original':
      - If original had decimals, keep same number of decimals.
      - Otherwise emit as integer-ish if close, else general.
    """
    if "." in original:
        decimals = len(original.split(".")[1])
        fmt = f"{{:.{decimals}f}}"
        return fmt.format(value)
    # no decimals originally
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _extract_cmd(line: str) -> Optional[str]:
    m = FIRST_TOKEN_RE.match(line)
    if not m:
        return None
    return m.group(1).upper()


def _is_call_qly(line_upper: str) -> bool:
    return line_upper.lstrip().startswith("CALL QLY " ) or line_upper.strip() == "CALL QLY"


def _is_call_qlyz(line_upper: str) -> bool:
    return line_upper.lstrip().startswith("CALL QLYZ")  # may have params


def _collect_y_values(lines: List[str]) -> List[float]:
    ys: List[float] = []

    for raw in lines:
        line = raw.rstrip("\n")
        upper = line.upper().strip()

        cmd = _extract_cmd(line) or ""
        is_move = cmd in MOVE_CMDS

        if is_move:
            # Collect numeric Y and numeric Z (Z treated as second-head Y in these files)
            for axis, val_s in COORD_RE.findall(line):
                axis_u = axis.upper()
                if axis_u in {"Y", "Z"}:
                    ys.append(float(val_s))
            continue

        # CALL QLY / QLYZ
        if _is_call_qly(upper):
            # If it uses axis letters, COORD_RE will capture it.
            coords = {a.upper(): float(v) for a, v in COORD_RE.findall(line)}
            if "Y" in coords:
                ys.append(coords["Y"])
            else:
                # Otherwise: first float is Y
                nums = FLOAT_RE.findall(line)
                if nums:
                    ys.append(float(nums[0]))
            continue

        if _is_call_qlyz(upper):
            coords = {a.upper(): float(v) for a, v in COORD_RE.findall(line)}
            if "Y" in coords:
                ys.append(coords["Y"])
            else:
                nums = FLOAT_RE.findall(line)
                # In your visualizer, QLYZ has at least 2 numbers: (Y, Z_like_secondY)
                if len(nums) >= 1:
                    ys.append(float(nums[0]))
                if len(nums) >= 2:
                    ys.append(float(nums[1]))
            continue

    return ys


def _flip_value(v: float, y_min: float, y_max: float) -> float:
    return (y_min + y_max) - v


def _rewrite_line(line: str, y_min: float, y_max: float) -> str:
    original = line.rstrip("\n")
    upper = original.upper().strip()
    cmd = _extract_cmd(original) or ""

    # 1) Rewrite move lines: MR/MI/MOVI/ARC
    if cmd in MOVE_CMDS:
        def repl_coord(m: re.Match) -> str:
            axis = m.group(1).upper()
            val_s = m.group(2)
            val = float(val_s)

            # Flip Y and Z (Z acts like second-head Y in these CNC files)
            if axis in {"Y", "Z"}:
                new_val = _flip_value(val, y_min, y_max)
                return f"{m.group(1)}{_format_like(val_s, new_val)}"
            return m.group(0)

        out = COORD_RE.sub(repl_coord, original)

        # Mirroring reverses arc direction -> negate sweep angle a=... (only on ARC command lines)
        if cmd == "ARC":
            def repl_a(ma: re.Match) -> str:
                prefix = ma.group(1)
                a_s = ma.group(2)
                a = float(a_s)
                new_a = -a
                return f"{prefix}{_format_like(a_s, new_a)}"
            out = ARC_ANGLE_RE.sub(repl_a, out)

        return out + ("\n" if line.endswith("\n") else "")

    # 2) Rewrite CALL QLY
    if _is_call_qly(upper):
        # If axis letters exist, flip Y tokens; else flip first float in params.
        if COORD_RE.search(original):
            def repl_coord(m: re.Match) -> str:
                axis = m.group(1).upper()
                val_s = m.group(2)
                val = float(val_s)
                if axis == "Y":
                    new_val = _flip_value(val, y_min, y_max)
                    return f"{m.group(1)}{_format_like(val_s, new_val)}"
                return m.group(0)
            out = COORD_RE.sub(repl_coord, original)
            return out + ("\n" if line.endswith("\n") else "")

        nums = list(FLOAT_RE.finditer(original))
        if not nums:
            return line
        # replace first number only
        first = nums[0]
        val_s = first.group(0)
        new_val = _flip_value(float(val_s), y_min, y_max)
        out = original[:first.start()] + _format_like(val_s, new_val) + original[first.end():]
        return out + ("\n" if line.endswith("\n") else "")

    # 3) Rewrite CALL QLYZ
    if _is_call_qlyz(upper):
        if COORD_RE.search(original):
            def repl_coord(m: re.Match) -> str:
                axis = m.group(1).upper()
                val_s = m.group(2)
                val = float(val_s)
                if axis in {"Y", "Z"}:
                    new_val = _flip_value(val, y_min, y_max)
                    return f"{m.group(1)}{_format_like(val_s, new_val)}"
                return m.group(0)
            out = COORD_RE.sub(repl_coord, original)
            return out + ("\n" if line.endswith("\n") else "")

        nums = list(FLOAT_RE.finditer(original))
        if len(nums) < 1:
            return line

        # QLYZ usually has two numbers: flip BOTH (Y and second-head-Y)
        # If there is only one number, flip that one.
        out = original
        # Replace from the back to keep indices valid
        for idx in reversed(range(min(2, len(nums)))):
            m = nums[idx]
            val_s = m.group(0)
            new_val = _flip_value(float(val_s), y_min, y_max)
            out = out[:m.start()] + _format_like(val_s, new_val) + out[m.end():]
        return out + ("\n" if line.endswith("\n") else "")

    # Default: unchanged
    return line


def flip_file(in_path: Path, out_path: Path, y_min_override: Optional[float], y_max_override: Optional[float]) -> Tuple[float, float]:
    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    if y_min_override is not None and y_max_override is not None:
        y_min, y_max = float(y_min_override), float(y_max_override)
    else:
        ys = _collect_y_values(lines)
        if not ys:
            raise SystemExit("No numeric Y/Z coordinates found to flip.")
        y_min, y_max = min(ys), max(ys)

        # Allow overriding only one side if desired
        if y_min_override is not None:
            y_min = float(y_min_override)
        if y_max_override is not None:
            y_max = float(y_max_override)

    flipped = [_rewrite_line(l, y_min, y_max) for l in lines]
    out_path.write_text("".join(flipped), encoding="utf-8", errors="ignore")
    return y_min, y_max


def main() -> None:
    ap = argparse.ArgumentParser(description="Flip a Dueffe-style CNC design vertically (mirror Y).")
    ap.add_argument("input", type=Path, help="Input CNC file")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output CNC file (default: *_flipY.CNC)")
    ap.add_argument("--y-min", type=float, default=None, help="Override Y-min for flip axis")
    ap.add_argument("--y-max", type=float, default=None, help="Override Y-max for flip axis")

    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    out_path = args.output
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_FLIP_V2" + in_path.suffix)

    y_min, y_max = flip_file(in_path, out_path, args.y_min, args.y_max)
    print(f"Flipped vertically using Ymin={y_min:g}, Ymax={y_max:g}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
