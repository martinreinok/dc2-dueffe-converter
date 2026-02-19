import sys
import CONFIGURATION
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QFile, QTextStream, Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QListWidgetItem, QStyle, QFileDialog
)
from typing import List, Optional
import ui_main
from library import CncPlotWidget
from cnc import import_cnc, export_cnc, Program
from interface import *

def load_stylesheet(file_path: str) -> str:
    f = QFile(file_path)
    if f.open(QFile.OpenModeFlag.ReadOnly):
        stream = QTextStream(f)
        return stream.readAll()
    return ""



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ui_main.Ui_MainWindow()
        self.ui.setupUi(self)
        self.plot = CncPlotWidget(self)
        self.setCentralWidget(self.plot)
        self.setAcceptDrops(True)
        self.cnc_program: Optional[Program] = None

        # Signals
        self.ui.pushButton_2.clicked.connect(
            lambda: (p := QFileDialog.getSaveFileName(
                self,
                "Export CNC",
                "",
                "CNC Files (*.CNC);;All Files (*)"
            )[0]) and export_cnc(self.cnc_program, Path(p))
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            extension = Path(event.mimeData().urls()[0].toLocalFile()).suffix.lower()
            print(extension)
            if extension in CONFIGURATION.accepted_file_drops:
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.cnc_program = import_cnc(Path(file_path))
            self.populate_section_list()

    def populate_section_list(self):
        lw = self.ui.listWidget
        lw.clear()
        if not self.cnc_program or not self.cnc_program.sections:
            return

        lw.setUniformItemSizes(False)
        lw.setSpacing(6)

        for i, section in enumerate(self.cnc_program.sections):
            if not section.lines:
                continue

            summary = SectionSummary(
                index=i,
                head_text=head_label(section.sewing_head),
                mr_text=mr_to_text(section.lines[0]),
                stats_text=section_stats(section.lines),
            )

            widget = SectionListRow(summary)
            item = QListWidgetItem(lw)
            item.setSizeHint(widget.sizeHint())
            lw.addItem(item)
            lw.setItemWidget(item, widget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet("stylesheets/MacOS.qss"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
