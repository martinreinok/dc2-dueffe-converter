from __future__ import annotations

from dataclasses import dataclass
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor
from typing import Dict, Any
from cnc import parse_command_coordinates, fmt


@dataclass(frozen=True)
class SectionSummary:
    index: int
    head_text: str          # e.g. "DW11 (Y)" / "DW14 (Mirror)"
    mr_text: str            # e.g. "MR X587.34 Y613 Z1237"
    stats_text: str         # e.g. "Lines: 11 • Arcs: 3"
    # later: thumbnail pixmap path or QPixmap


def _placeholder_pixmap(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor(240, 240, 240))
    p.setPen(QColor(210, 210, 210))
    p.drawRoundedRect(0, 0, size - 1, size - 1, 8, 8)
    p.setPen(QColor(160, 160, 160))
    p.drawText(pm.rect(), Qt.AlignCenter, "IMG")
    p.end()
    return pm


class SectionListRow(QtWidgets.QFrame):
    def __init__(self, summary: SectionSummary, parent=None):
        super().__init__(parent)
        self.summary = summary

        # sizing knobs (change these at runtime)
        self._row_height = 72
        self._thumb_size = 48
        self._dense = False

        self.setObjectName("SectionListRow")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # --- thumbnail ---
        self.thumb = QtWidgets.QLabel()
        self.thumb.setPixmap(_placeholder_pixmap(self._thumb_size))
        self.thumb.setFixedSize(self._thumb_size, self._thumb_size)

        # --- title line: "Section 03" + head badge ---
        self.title = QtWidgets.QLabel(f"Section {summary.index:02d}")
        title_font = QFont()
        title_font.setBold(True)
        self.title.setFont(title_font)

        self.badge = QtWidgets.QLabel(summary.head_text)
        self.badge.setObjectName("HeadBadge")
        self.badge.setAlignment(Qt.AlignCenter)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_row.addWidget(self.title, 0)
        title_row.addWidget(self.badge, 0, Qt.AlignLeft)
        title_row.addStretch(1)

        # --- second line: MR coords ---
        self.mr = QtWidgets.QLabel(summary.mr_text)
        self.mr.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mr.setObjectName("MrLine")

        # --- third line: small stats ---
        self.stats = QtWidgets.QLabel(summary.stats_text)
        self.stats.setObjectName("StatsLine")

        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        text_col.addLayout(title_row)
        text_col.addWidget(self.mr)
        text_col.addWidget(self.stats)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(10)
        root.addWidget(self.thumb, 0, Qt.AlignVCenter)
        root.addLayout(text_col, 1)

        # base styles (selection highlighting is handled by QListWidget;
        # but with setItemWidget, we style our own background)
        self.setStyleSheet("""
            QFrame#SectionListRow {
                border-radius: 10px;
                background: rgba(255,255,255,0.95);
            }
            QLabel#MrLine { color: rgba(0,0,0,0.75); }
            QLabel#StatsLine { color: rgba(0,0,0,0.55); }
            QLabel#HeadBadge {
                padding: 2px 8px;
                border-radius: 10px;
                background: rgba(0,0,0,0.06);
                color: rgba(0,0,0,0.75);
                font-size: 11px;
            }
        """)

        self._apply_density()

    def set_density(self, dense: bool):
        self._dense = dense
        self._apply_density()

    def set_row_height(self, h: int):
        self._row_height = max(48, int(h))
        self._apply_density()

    def set_thumbnail_size(self, s: int):
        self._thumb_size = max(24, int(s))
        self._apply_density()

    def _apply_density(self):
        if self._dense:
            self._row_height = min(self._row_height, 58)
            self.setContentsMargins(0, 0, 0, 0)

        self.thumb.setFixedSize(self._thumb_size, self._thumb_size)
        self.thumb.setPixmap(_placeholder_pixmap(self._thumb_size))

        # make the widget report the desired height
        self.setMinimumHeight(self._row_height)
        self.setMaximumHeight(self._row_height)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        return QSize(320, self._row_height)


def head_label(sewing_head) -> str:
    # SewingHeadType is your enum (DW11, DW12, DW13, DW14)
    name = sewing_head.name if sewing_head else "UNKNOWN"
    if name == "DW11":
        return "DW11 (Y)"
    if name == "DW12":
        return "DW12 (Z)"
    if name == "DW13":
        return "DW13 (Dual • Parallel)"
    if name == "DW14":
        return "DW14 (Dual • Mirror)"
    return name


def mr_to_text(mr_line: str) -> str:
    # reuse your parser
    _, vals = parse_command_coordinates(mr_line.strip(), axes="XYZ")
    parts = []
    for ax in ("X", "Y", "Z"):
        if ax in vals:
            parts.append(f"{ax}{fmt(vals[ax])}")
    return "MR " + " ".join(parts)


def section_stats(lines: list[str]) -> str:
    # lines include MR + CALL DWxx + geometry + UP1
    arc = sum(1 for l in lines if l.strip().startswith("ARC"))
    mi = sum(1 for l in lines if l.strip().startswith("MI"))
    movi = sum(1 for l in lines if l.strip().startswith("MOVI"))
    return f"Lines: {len(lines)} • MI: {mi} • MOVI: {movi} • ARC: {arc}"