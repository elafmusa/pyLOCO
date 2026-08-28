"""Educational inspection of the exact persisted LOCO fit vector."""

from __future__ import annotations

import csv
import json
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .plot_canvas import PlotCanvas


class ParametersView(QWidget):
    GENERIC_HEADERS = ["Index", "Initialization", "Fitted value", "Change from initialization", "Unit"]
    QUAD_HEADERS = ["Index", "Element / family", "Lattice ordinal", "Initial K", "Fitted K", "ΔK", "ΔK/K [%]", "Unit"]

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
        self.table.setHorizontalHeaderLabels(self.GENERIC_HEADERS)
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
        if loader.quadrupole_parameter_rows:
            self.selector.addItem("Quadrupole fitted correction ΔK", "__quad_delta__")
            self.selector.addItem("Quadrupole relative correction ΔK/K [%]", "__quad_relative__")
        self.selector.blockSignals(False)
        self._render()

    def _render(self):
        self.plot.clear(); self.table.setRowCount(0)
        if self.loader is None or self.selector.currentIndex() < 0:
            self.summary.setText("No persisted fitted parameter vector is available for this run."); return
        key = self.selector.currentData()
        if key in {"__quad_delta__", "__quad_relative__"}:
            rows = self.loader.quadrupole_parameter_rows
            values = np.asarray([
                np.nan if row["delta_k" if key == "__quad_delta__" else "relative_percent"] is None
                else row["delta_k" if key == "__quad_delta__" else "relative_percent"]
                for row in rows
            ], dtype=float)
            unit = "m⁻²" if key == "__quad_delta__" else "%"
            finite = values[np.isfinite(values)]
            mode = rows[0]["mode"] if rows else "unknown"
            if finite.size:
                statistics = (
                    f"min {np.min(finite):.6g} · max {np.max(finite):.6g} · mean {np.mean(finite):.6g} · "
                    f"RMS {np.sqrt(np.mean(finite**2)):.6g} · max |correction| {np.max(np.abs(finite)):.6g} {unit}"
                )
            else:
                statistics = "Correction statistics unavailable because the initial K values were not persisted."
            self.summary.setText(f"{len(rows)} {mode} fitted quadrupole parameters · ΔK = K_fitted − K_initial · {statistics}")
            axis = self.plot.figure.add_subplot(111); axis.plot(values, marker=".", linewidth=1.0)
            title = "Quadrupole ΔK" if key == "__quad_delta__" else "Quadrupole ΔK/K"
            axis.set(xlabel="Fitted-parameter position", ylabel=f"Correction [{unit}]", title=title)
            names = [row["name"] or f"parameter {row['index']}" for row in rows]
            axis.format_coord = lambda x, y: f"{names[min(max(int(round(x)), 0), len(names)-1)] if names else 'parameter'}: {y:.6g} {unit}"
            axis.grid(True, alpha=.25); self.plot.apply_theme(); self.plot.canvas.draw_idle()
            self.table.setSortingEnabled(False); self.table.setColumnCount(len(self.QUAD_HEADERS)); self.table.setHorizontalHeaderLabels(self.QUAD_HEADERS); self.table.setRowCount(len(rows))
            for table_row, row in enumerate(rows):
                ordinal = "—" if row["lattice_ordinal"] is None else str(row["lattice_ordinal"])
                texts = (
                    str(row["index"]), row["name"] or "Unavailable", ordinal,
                    "Unavailable" if row["initial"] is None else f"{row['initial']:.8g}",
                    f"{row['fitted']:.8g}", "Unavailable" if row["delta_k"] is None else f"{row['delta_k']:.8g}",
                    "Unavailable" if row["relative_percent"] is None else f"{row['relative_percent']:.8g}", row["unit"],
                )
                for column, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    if column == 1 and row["member_ordinals"]:
                        item.setToolTip("Member lattice ordinals: " + ", ".join(map(str, row["member_ordinals"])))
                    self.table.setItem(table_row, column, item)
            # Preserve the fitted-vector order: table row i must remain plot point i.
            self.table.resizeColumnsToContents()
            return
        block = next((item for item in self.loader.parameter_blocks if item.key == key), None)
        if block is None: return
        self.table.setColumnCount(len(self.GENERIC_HEADERS)); self.table.setHorizontalHeaderLabels(self.GENERIC_HEADERS)
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
            if block.key == "quads":
                continue
            for index, value in enumerate(block.values):
                initial = None if block.baseline is None else float(block.baseline[index])
                identity = self.loader.parameter_identity(block.key, index)
                yield {"block": block.key, "label": block.label, "index": index, "initial": initial,
                       "fitted": float(value), "correction": None if initial is None else float(value)-initial,
                       "unit": block.unit, **identity,
                       "sign_convention": "Backend fitted value; machine-application corrections use the explicitly labelled pyLOCO convention."}
        for row in self.loader.quadrupole_parameter_rows if self.loader else []:
            yield {"block": "quads", "label": row["name"], "index": row["index"],
                   "lattice_ordinal": row["lattice_ordinal"], "element_name": row["name"],
                   "member_ordinals": row["member_ordinals"], "initial": row["initial"],
                   "fitted": row["fitted"], "correction": row["delta_k"], "unit": row["unit"],
                   "relative_percent": row["relative_percent"], "fit_mode": row["mode"],
                   "sign_convention": row["sign_convention"]}

    def _export(self, format_name):
        filename = QFileDialog.getSaveFileName(self, "Export fitted parameters", f"fitted_parameters.{format_name}", f"{format_name.upper()} (*.{format_name})")[0]
        if not filename: return
        rows = list(self._rows())
        if format_name == "json":
            with open(filename, "w", encoding="utf-8") as stream: json.dump(rows, stream, indent=2)
        else:
            with open(filename, "w", newline="", encoding="utf-8") as stream:
                fields = ["block", "label", "index", "selected_list_position", "lattice_ordinal", "element_name",
                          "member_ordinals", "fit_mode", "initial", "fitted", "correction", "unit",
                          "relative_percent", "sign_convention"]
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
