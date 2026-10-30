"""Presentation for the read-only pre-FIT reference-model comparison."""

from __future__ import annotations

import json
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..reference_comparison import ReferenceComparisonResult
from ..results.plot_canvas import PlotCanvas


class ReferenceComparisonWindow(QDialog):
    def __init__(self, result: ReferenceComparisonResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Compare imported measurements with initial model")
        self.resize(1280, 820); self.setMinimumSize(680, 480)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Read-only FIT diagnostic comparing imported measurements with the active reference lattice. "
            "No acquisition, Jacobian, optimizer, FIT parameter update or control-system write was performed. "
            "Residual = Measured − Model."
        )
        intro.setWordWrap(True); intro.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(intro)
        mask = "excluded from the FIT objective" if result.cross_plane_excluded else "included in the FIT objective"
        self.mask_label = QLabel(f"Cross-plane blocks H→Y and V→X are displayed; they are <b>{mask}</b>.")
        self.mask_label.setWordWrap(True); layout.addWidget(self.mask_label)
        self.tabs = QTabWidget(); layout.addWidget(self.tabs, 1)
        self._add_orm_tab(); self._add_rf_tab(); self._add_provenance_tab()
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _plot_triplet(self, measured, model, *, title_prefix, unit):
        page = QWidget(); row = QHBoxLayout(page); residual = measured - model
        finite = np.concatenate((measured.ravel(), model.ravel()))
        finite = finite[np.isfinite(finite)]
        lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (-1.0, 1.0))
        for title, values, residual_plot in (
            (f"Measured {title_prefix}", measured, False),
            (f"Model {title_prefix}", model, False),
            ("Residual (Measured − Model)", residual, True),
        ):
            canvas = PlotCanvas(show_toolbar=True, minimum_height=360)
            ax = canvas.figure.add_subplot(111)
            if values.ndim == 2:
                kwargs = {"aspect": "auto", "origin": "lower", "cmap": "coolwarm" if residual_plot else "viridis"}
                if residual_plot:
                    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
                    kwargs.update(vmin=-limit, vmax=limit)
                else:
                    kwargs.update(vmin=float(lo), vmax=float(hi))
                image = ax.imshow(values, **kwargs); canvas.figure.colorbar(image, ax=ax, label=unit)
                ax.set_xlabel("Corrector index"); ax.set_ylabel("BPM row index")
            else:
                ax.plot(np.arange(values.size), values, linewidth=1.1)
                ax.axhline(0, color="#8d95a8", linewidth=.8); ax.set_xlabel("BPM row index"); ax.set_ylabel(unit)
            ax.set_title(title)
            if values.ndim == 1:
                ax.grid(True, alpha=.25)
            canvas.apply_theme(); row.addWidget(canvas, 1)
        return page

    def _add_orm_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(self._plot_triplet(self.result.measured_orm, self.result.model_orm,
                                            title_prefix="ORM", unit="Orbit response [m]"), 1)
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["Block", "Objective", "Measured RMS [m]", "Model RMS [m]", "Residual RMS [m]"])
        for name, values in self.result.statistics.items():
            row = table.rowCount(); table.insertRow(row)
            cross = name in {"H→Y", "V→X"}
            objective = "Excluded" if cross and self.result.cross_plane_excluded else "Included"
            for col, value in enumerate((name, objective, values["measured_rms"], values["model_rms"], values["residual_rms"])):
                table.setItem(row, col, QTableWidgetItem(str(value) if col < 2 else f"{value:.8g}"))
        table.horizontalHeader().setStretchLastSection(True); table.resizeColumnsToContents(); table.setMaximumHeight(210)
        layout.addWidget(table); self.tabs.addTab(page, "ORM comparison")

    def _add_rf_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        if self.result.measured_rf is None or self.result.model_rf is None:
            label = QLabel("No RF-dispersion measurement is loaded; RF comparison is unavailable.")
            label.setWordWrap(True); layout.addWidget(label); layout.addStretch(1)
            self.tabs.addTab(page, "RF dispersion"); return
        controls = QHBoxLayout(); controls.addWidget(QLabel("Plane:"))
        plane_selector = QComboBox()
        plane_selector.addItem("Horizontal", "horizontal")
        plane_selector.addItem("Vertical", "vertical")
        controls.addWidget(plane_selector)
        controls.addSpacing(18); controls.addWidget(QLabel("Displayed quantity:"))
        selector = QComboBox(); selector.addItem("RF orbit response [m]", "rf")
        if self.result.physical_measured is not None:
            selector.addItem("Physical dispersion D [mm]", "physical")
        controls.addWidget(selector); controls.addStretch(1); layout.addLayout(controls)
        message = QLabel(); message.setWordWrap(True); layout.addWidget(message)
        stats = QTableWidget(2, 3)
        stats.setHorizontalHeaderLabels(["Measured RMS", "Initial-model RMS", "Residual RMS (Measured − Model)"])
        stats.setVerticalHeaderLabels(["Horizontal (X)", "Vertical (Y)"])
        stats.setEditTriggers(QAbstractItemView.NoEditTriggers)
        stats.setSelectionMode(QAbstractItemView.NoSelection)
        stats.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Two rows plus headers remain fully visible with native macOS font
        # metrics; the table never becomes a nested scrolling area.
        stats.setFixedHeight(112)
        layout.addWidget(stats)
        holder = QVBoxLayout(); layout.addLayout(holder, 1)
        current = [None]
        def redraw():
            if current[0] is not None:
                current[0].setParent(None); current[0].deleteLater()
            physical = selector.currentData() == "physical"
            measured = self.result.physical_measured if physical else self.result.measured_rf
            model = self.result.physical_model if physical else self.result.model_rf
            # Scientific data remain in SI metres. Scale only the physical-
            # dispersion presentation so the values are easier to inspect.
            display_scale = 1000.0 if physical else 1.0
            measured = np.asarray(measured, dtype=float) * display_scale
            model = np.asarray(model, dtype=float) * display_scale
            all_measured = measured
            all_model = model
            plane = plane_selector.currentData()
            section = slice(0, self.result.n_bpms) if plane == "horizontal" else slice(self.result.n_bpms, None)
            measured = all_measured[section]
            model = all_model[section]
            if physical:
                message.setText(
                    "Physical dispersion is derived only for display: D = ΔorbitRF × ηAT × fRF / signed Δf. "
                    "Displayed values are converted from metres to millimetres. "
                    "The signed RF orbit response remains the FIT objective quantity."
                )
                prefix = "physical dispersion"
            else:
                message.setText("Signed RF-induced orbit difference used by FIT; it is not absolute physical dispersion.")
                prefix = "RF orbit response"
            unit = "mm" if physical else "m"
            for row, plane_section in enumerate((
                slice(0, self.result.n_bpms), slice(self.result.n_bpms, None),
            )):
                plane_measured = all_measured[plane_section]
                plane_model = all_model[plane_section]
                for column, value in enumerate((
                    np.sqrt(np.mean(plane_measured ** 2)),
                    np.sqrt(np.mean(plane_model ** 2)),
                    np.sqrt(np.mean((plane_measured - plane_model) ** 2)),
                )):
                    stats.setItem(row, column, QTableWidgetItem(f"{float(value):.8g} {unit}"))
            current[0] = self._plot_triplet(measured, model, title_prefix=prefix, unit=unit)
            holder.addWidget(current[0])
        selector.currentIndexChanged.connect(redraw)
        plane_selector.currentIndexChanged.connect(redraw)
        self.rf_plane_selector = plane_selector
        self.rf_quantity_selector = selector
        self.rf_statistics_table = stats
        redraw(); self.tabs.addTab(page, "RF dispersion")

    def _add_provenance_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        provenance = self.result.provenance
        summary = QLabel(
            f"<b>Reference lattice:</b> {provenance.get('lattice', '—')}<br>"
            f"<b>MAT variable:</b> {provenance.get('mat_variable', '—')} · "
            f"<b>Elements:</b> {provenance.get('lattice_elements', '—')}<br>"
            f"<b>Devices:</b> {provenance.get('bpm_count', '—')} BPMs, "
            f"{provenance.get('horizontal_corrector_count', '—')} H correctors, "
            f"{provenance.get('vertical_corrector_count', '—')} V correctors<br>"
            f"<b>Calculator:</b> {provenance.get('calculator', '—')} · "
            f"<b>RF step:</b> {provenance.get('rf_step_hz', '—')} Hz<br>"
            f"<b>Residual:</b> {provenance.get('residual_definition', 'Measured − reference model')}"
        )
        summary.setWordWrap(True); summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(summary)
        details = QGroupBox("Technical details")
        details.setCheckable(True); details.setChecked(False)
        details_layout = QVBoxLayout(details)
        editor = QPlainTextEdit(); editor.setReadOnly(True)
        editor.setPlainText(json.dumps(self.result.provenance, indent=2, ensure_ascii=False))
        editor.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        details_layout.addWidget(editor)
        editor.setVisible(False)
        details.toggled.connect(editor.setVisible)
        layout.addWidget(details, 1)
        self.technical_details_group = details
        self.tabs.addTab(page, "Inputs and provenance")
