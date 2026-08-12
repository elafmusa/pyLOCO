"""Educational inspection of the exact persisted LOCO fit vector."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .plot_canvas import PlotCanvas


class ParametersView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loader = None
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._render)
        self.summary = QLabel("No parameter results loaded.")
        self.available = QLabel("No fitted parameter blocks loaded."); self.available.setWordWrap(True)
        self.summary.setWordWrap(True)
        top = QHBoxLayout(); top.addWidget(QLabel("Parameter block")); top.addWidget(self.selector, 1)
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=300)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Index", "Initialization", "Fitted value", "Change from initialization", "Unit"])
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout = QVBoxLayout(self); layout.addWidget(self.available); layout.addLayout(top); layout.addWidget(self.summary); layout.addWidget(self.plot); layout.addWidget(self.table)

    def set_loader(self, loader):
        self.loader = loader
        self.selector.blockSignals(True); self.selector.clear()
        for block in loader.parameter_blocks:
            self.selector.addItem(f"{block.label} ({block.values.size})", block.key)
        names = ", ".join(block.label for block in loader.parameter_blocks)
        self.available.setText(f"Total fitted DOFs: {loader.fitted_parameter_count or 0} · Available blocks: {names or 'none'}")
        self.selector.blockSignals(False)
        self._render()

    def _render(self):
        self.plot.clear(); self.table.setRowCount(0)
        if self.loader is None or self.selector.currentIndex() < 0:
            self.summary.setText("No persisted fitted parameter vector is available for this run."); return
        key = self.selector.currentData()
        block = next((item for item in self.loader.parameter_blocks if item.key == key), None)
        if block is None: return
        values = np.asarray(block.values, dtype=float)
        changes = block.changes
        self.summary.setText(
            f"{block.label}: {values.size} fitted DOFs · mean {np.mean(values):.6g} · "
            f"RMS {np.sqrt(np.mean(values**2)):.6g} · median {np.median(values):.6g} · range {np.min(values):.6g} to {np.max(values):.6g}. "
            "Values are read from the final persisted fit vector; no machine correction is applied here."
        )
        axis = self.plot.figure.add_subplot(111)
        axis.plot(np.arange(values.size), values, color="#19a974", linewidth=1.1)
        axis.set(xlabel="Fitted degree of freedom", ylabel=f"Fitted value [{block.unit}]", title=block.label)
        axis.grid(True, alpha=.25); self.plot.apply_theme(); self.plot.canvas.draw_idle()
        self.table.setSortingEnabled(False); self.table.setRowCount(values.size)
        for row, value in enumerate(values):
            change = "Not available" if changes is None else f"{changes[row]:.8g}"
            initial = "Not available" if block.baseline is None else f"{block.baseline[row]:.8g}"
            for col, text in enumerate((str(row), initial, f"{value:.8g}", change, block.unit)):
                item = QTableWidgetItem(text)
                if col in (0, 2): item.setData(Qt.UserRole, row if col == 0 else float(value))
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True); self.table.resizeColumnsToContents()
