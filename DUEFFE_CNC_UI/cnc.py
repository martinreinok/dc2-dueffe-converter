"""
TODO: Too many QLYZ..
"""
from __future__ import annotations

from CONFIGURATION import Coordinates, SewingHeadType
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Section:
    sewing_head: SewingHeadType = None
    lines: List[str] = None

    def __init__(self, program_block: List[str]):
        for head in SewingHeadType:
            if any(head.name in line for line in program_block):
                self.sewing_head = head
                break
        self.lines = program_block

def parse_command_coordinates(s, axes="XYZ"):
    cmd, rest = s.strip().split(None, 1)
    for a in axes:
        rest = rest.replace(a, f" {a}")
    vals = {p[0]: float(p[1:]) for p in rest.split()}
    return cmd, vals

def fmt(n: float) -> str:
    s = f"{n:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

@dataclass
class Program:
    header_lines: List[str] = None
    sections: List[Section] = None
    footer_lines: List[str] = None

    def to_cnc(self) -> List[str]:
        output: List[str] = []
        output += self.header_lines

        last_motor: Optional[str] = None  # ELY, ELZ, ELYZ
        last_lock: Optional[str] = None  # FLZ, FLY, None
        last_ql: Optional[tuple] = None  # (QLY, y), (QLZ, z), (QLYZ, y, z)
        last_dual_dw: Optional[SewingHeadType] = None  # DW13 or DW14

        for section in self.sections:
            if not section.lines:
                continue

            mr_line = section.lines[0]
            move = parse_command_coordinates(mr_line.strip())[1]

            if section.sewing_head == SewingHeadType.DW11:
                desired_lock = "FLZ"
                desired_ql = ("QLY", fmt(move["Y"]))
                desired_motor = "ELY"

            elif section.sewing_head == SewingHeadType.DW12:
                desired_lock = "FLY"
                desired_ql = ("QLZ", fmt(move["Z"]))
                desired_motor = "ELZ"

            else:  # DW13 or DW14
                desired_lock = None
                desired_ql = ("QLYZ", fmt(move["Y"]), fmt(move["Z"]))
                desired_motor = "ELYZ"

            motor_changed = (desired_motor != last_motor)

            if motor_changed and desired_lock is not None:
                output.append(f"CALL {desired_lock}\n")

            emit_ql = motor_changed or (desired_motor == "ELYZ" and desired_ql != last_ql)
            if emit_ql:
                if desired_ql[0] == "QLY":
                    output.append(f"CALL QLY {desired_ql[1]}\n")
                elif desired_ql[0] == "QLZ":
                    output.append(f"CALL QLZ {desired_ql[1]}\n")
                else:
                    output.append(f"CALL QLYZ {desired_ql[1]} {desired_ql[2]}\n")

            output.append(mr_line)

            emit_motor = motor_changed or (
                    desired_motor == "ELYZ"
                    and section.sewing_head in (SewingHeadType.DW13, SewingHeadType.DW14)
                    and last_dual_dw is not None
                    and section.sewing_head != last_dual_dw
            )
            if emit_motor:
                output.append(f"CALL {desired_motor}\n")

            output += section.lines[1:]

            # update state
            last_motor = desired_motor
            last_lock = desired_lock
            last_ql = desired_ql

            if desired_motor == "ELYZ" and section.sewing_head in (SewingHeadType.DW13, SewingHeadType.DW14):
                last_dual_dw = section.sewing_head
            else:
                last_dual_dw = None

        output += self.footer_lines
        return output

def split_into_sections(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    block_start: int | None = None
    index = 0

    while index < len(lines):
        while index < len(lines) and not lines[index].strip().startswith("MR"):
            index += 1
        if index >= len(lines):
            break

        block_start = index
        index += 1

        while index < len(lines):
            token = lines[index].strip()
            if token.startswith("MR"):
                block_start = index
            if token == "CALL UP1":
                blocks.append(lines[block_start:index + 1])
                block_start = None
                index += 1
                break
            index += 1

        if block_start is not None and index >= len(lines):
            blocks.append(lines[block_start:])

    return blocks

def import_cnc(file: Path) -> Program:
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    header_idx = next((i for i, line in enumerate(lines) if line.strip() == "CALL VA1"), None) + 1
    header = lines[:header_idx]

    footer_idx = next((i for i, line in enumerate(lines) if line.strip() == "CALL STOFF"), None)
    footer = lines[footer_idx:]

    program = lines[header_idx:footer_idx]

    # The macros are added in afterward again. They are fully deterministic.
    MACRO_LINES_TO_REMOVE = {
        "FLY", "FLZ",
        "QLY", "QLZ", "QLYZ",
        "ELY", "ELZ", "ELYZ",
    }

    filtered_program = []
    for line in program:
        if not any(m in line for m in MACRO_LINES_TO_REMOVE):
            filtered_program.append(line)

    program = filtered_program

    sections = split_into_sections(program)
    sections_list = []
    for section in sections:
        sections_list.append(Section(section))

    program = Program(
        header_lines=header,
        footer_lines=footer,
        sections=sections_list,
    )

    return program

def export_cnc(program: Program, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        f.writelines(program.to_cnc())

if __name__ == "__main__":
    program = import_cnc(Path(r"C:\Users\server\Downloads\TEST_DRUGE_GLAVE_ORIGINAL.CNC"))
    export_cnc(program, Path(r"C:\Users\server\Downloads\TEST_DRUGE_GLAVE_GENERATED.CNC"))
