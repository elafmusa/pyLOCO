"""Auditable list of run inputs and generated artifacts."""

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class FilesView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        self.table = QTableWidget(0, 4); self.table.setHorizontalHeaderLabels(["Role", "File", "Size", "Location"])
        self.table.horizontalHeader().setStretchLastSection(True)
        copy = QPushButton("Copy selected path"); copy.clicked.connect(self._copy)
        folder = QPushButton("Open containing folder"); folder.clicked.connect(self._open_folder)
        controls = QHBoxLayout(); controls.addWidget(copy); controls.addWidget(folder); controls.addStretch(1)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("Inputs and generated artifacts recorded for this run.")); layout.addLayout(controls); layout.addWidget(self.table)

    def set_loader(self, loader):
        self.loader = loader; records = []
        records.extend((role, path) for role, path in loader.input_files)
        records.extend(("Generated", path) for path in loader.generated_files)
        self.table.setRowCount(len(records))
        for row, (role, path) in enumerate(records):
            location = str(path) if path else "Referenced file not found"
            size = f"{path.stat().st_size:,} B" if path and path.exists() else "—"
            for col, text in enumerate((role, path.name if path else "—", size, location)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _selected_path(self):
        row = self.table.currentRow()
        return self.table.item(row, 3).text() if row >= 0 and self.table.item(row, 3) else None

    def _copy(self):
        path = self._selected_path()
        if path: QApplication.clipboard().setText(path)

    def _open_folder(self):
        from pathlib import Path
        value = self._selected_path()
        if value and Path(value).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(value).parent)))
