"""Educational inspection of the exact persisted LOCO fit vector."""

from __future__ import annotations

import csv
import json
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .plot_canvas import PlotCanvas


class ParametersView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loader = None
        title = QLabel("Fitted Parameters"); title.setObjectName("resultsTitle")
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._render)
        self.summary = QLabel("No parameter results loaded.")
        self.summary.setWordWrap(True)
        top = QHBoxLayout(); top.addWidget(QLabel("Parameter block")); top.addWidget(self.selector, 1)
        self.export_csv = QPushButton("Save fitted data CSV…"); self.export_json = QPushButton("Save fitted data JSON…")
        self.export_csv.clicked.connect(lambda: self._export("csv")); self.export_json.clicked.connect(lambda: self._export("json"))
        top.addWidget(self.export_csv); top.addWidget(self.export_json)
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=210)
        self.plot.toolbar.setMaximumHeight(34)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Index", "Initialization", "Fitted value", "Change from initialization", "Unit"])
        self.table.setSortingEnabled(True)
        self.table.setMinimumHeight(150)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.content_splitter = QSplitter(Qt.Vertical)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.plot)
        self.content_splitter.addWidget(self.table)
        self.content_splitter.setStretchFactor(0, 5)
        self.content_splitter.setStretchFactor(1, 3)
        self.content_splitter.setSizes([300, 170])
        summary_group = QGroupBox("Summary"); summary_layout = QVBoxLayout(summary_group); summary_layout.addWidget(self.summary)
        summary_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        content = QWidget(); content.setMinimumHeight(500); content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored); content_layout = QVBoxLayout(content)
        content_layout.addWidget(title); content_layout.addLayout(top); content_layout.addWidget(summary_group, 0); content_layout.addWidget(self.content_splitter, 1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame); scroll.setWidget(content)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(scroll)

    def set_loader(self, loader):
        self.loader = loader
        self.selector.blockSignals(True); self.selector.clear()
        for block in loader.parameter_blocks:
            self.selector.addItem(f"{block.label} ({block.values.size})", block.key)
        if loader.quadrupole_corrections:
            self.selector.addItem("Quadrupole application correction ΔK", "__quad_delta__")
            self.selector.addItem("Quadrupole relative correction ΔK/K [%]", "__quad_relative__")
        self.selector.blockSignals(False)
        self._render()

    def _render(self):
        self.plot.clear(); self.table.setRowCount(0)
        if self.loader is None or self.selector.currentIndex() < 0:
            self.summary.setText("No persisted fitted parameter vector is available for this run."); return
        key = self.selector.currentData()
        if key in {"__quad_delta__", "__quad_relative__"}:
            data = self.loader.quadrupole_corrections
            values = np.asarray(data["delta_k_apply"] if key == "__quad_delta__" else data["relative_percent"], dtype=float)
            unit = "m⁻²" if key == "__quad_delta__" else "%"
            self.summary.setText(f"{data['sign_convention']} · RMS {np.sqrt(np.nanmean(values**2)):.6g} {unit}")
            axis = self.plot.figure.add_subplot(111); axis.plot(values, marker=".", linewidth=1.0)
            axis.set(xlabel="Selected quadrupole position", ylabel=f"Correction [{unit}]", title="Quadrupole correction")
            axis.grid(True, alpha=.25); self.plot.apply_theme(); self.plot.canvas.draw_idle()
            return
        block = next((item for item in self.loader.parameter_blocks if item.key == key), None)
        if block is None: return
        values = np.asarray(block.values, dtype=float)
        changes = block.changes
        plotted = np.asarray(changes if changes is not None else values, dtype=float)
        quantity = "Change from initialization" if changes is not None else "Fitted value"
        self.summary.setText(
            f"{block.label}: {values.size} fitted DOFs · {quantity.lower()} mean {np.mean(plotted):.6g} · "
            f"RMS {np.sqrt(np.mean(plotted**2)):.6g} · median {np.median(plotted):.6g} · range {np.min(plotted):.6g} to {np.max(plotted):.6g}. "
            "The chart shows the fitted change when the initial values are available."
        )
        axis = self.plot.figure.add_subplot(111)
        axis.plot(np.arange(values.size), plotted, color="#19a974", linewidth=1.1)
        axis.set(xlabel="Fitted degree of freedom", ylabel=f"{quantity} [{block.unit}]", title=block.label)
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

    def _rows(self):
        for block in self.loader.parameter_blocks if self.loader else []:
            for index, value in enumerate(block.values):
                initial = None if block.baseline is None else float(block.baseline[index])
                identity = self.loader.parameter_identity(block.key, index)
                yield {"block": block.key, "label": block.label, "index": index, "initial": initial,
                       "fitted": float(value), "correction": None if initial is None else float(value)-initial,
                       "unit": block.unit, **identity,
                       "sign_convention": "Backend fitted value; machine-application corrections use the explicitly labelled pyLOCO convention."}
        corrections = self.loader.quadrupole_corrections if self.loader else None
        if corrections:
            for index, value in enumerate(corrections["delta_k_apply"]):
                yield {"block": "quadrupole_application", "label": corrections["names"][index], "index": index,
                       "lattice_ordinal": int(corrections["ordinals"][index]), "element_name": corrections["names"][index],
                       "initial": float(corrections["initial"][index]), "fitted": float(corrections["fitted"][index]),
                       "correction": float(value), "unit": "m⁻²", "relative_percent": float(corrections["relative_percent"][index]),
                       "sign_convention": corrections["sign_convention"]}

    def _export(self, format_name):
        filename = QFileDialog.getSaveFileName(self, "Export fitted parameters", f"fitted_parameters.{format_name}", f"{format_name.upper()} (*.{format_name})")[0]
        if not filename: return
        rows = list(self._rows())
        if format_name == "json":
            with open(filename, "w", encoding="utf-8") as stream: json.dump(rows, stream, indent=2)
        else:
            with open(filename, "w", newline="", encoding="utf-8") as stream:
                fields = ["block", "label", "index", "selected_list_position", "lattice_ordinal", "element_name", "initial", "fitted", "correction", "unit", "relative_percent", "sign_convention"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
