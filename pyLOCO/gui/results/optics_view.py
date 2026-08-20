"""Scientific beta-beating and dispersion results for a completed run."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QGroupBox, QLabel, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from .plot_canvas import PlotCanvas


def _stats(values) -> str:
    finite = np.asarray(values, dtype=float); finite = finite[np.isfinite(finite)] * 100.0
    if not finite.size: return "Unavailable"
    return f"RMS {np.sqrt(np.mean(finite**2)):.3g}%   ·   mean {np.mean(finite):.3g}%   ·   max |Δβ/β| {np.max(np.abs(finite)):.3g}%"


class OpticsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget(); content_layout = QVBoxLayout(content)
        self.reference = QLabel("No optics results loaded."); self.reference.setWordWrap(True)
        self.beta_stats = QLabel(); self.beta_stats.setWordWrap(True); self.beta_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.beta_plot = PlotCanvas(show_toolbar=True, minimum_height=230)
        beta_group = QGroupBox("Beta beating"); beta_layout = QVBoxLayout(beta_group)
        beta_layout.addWidget(self.reference); beta_layout.addWidget(self.beta_stats); beta_layout.addWidget(self.beta_plot)
        self.dispersion_message = QLabel(); self.dispersion_message.setWordWrap(True)
        self.dispersion_stats = QLabel(); self.dispersion_stats.setWordWrap(True); self.dispersion_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dispersion_table = QTableWidget(0, 3)
        self.dispersion_table.setHorizontalHeaderLabels(["Raw residual diagnostic", "Horizontal", "Vertical"])
        self.dispersion_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dispersion_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.dispersion_table.verticalHeader().hide(); self.dispersion_table.horizontalHeader().setStretchLastSection(True)
        self.dispersion_plot = PlotCanvas(show_toolbar=True, minimum_height=260)
        dispersion_group = QGroupBox("Dispersion"); dispersion_layout = QVBoxLayout(dispersion_group)
        dispersion_layout.addWidget(self.dispersion_message); dispersion_layout.addWidget(self.dispersion_stats); dispersion_layout.addWidget(self.dispersion_table); dispersion_layout.addWidget(self.dispersion_plot)
        content_layout.addWidget(beta_group); content_layout.addWidget(dispersion_group); content_layout.addStretch(1)
        scroll.setWidget(content); layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(scroll)
        self.plot = self.beta_plot  # compatibility with the workspace theme hook

    def set_loader(self, loader):
        self.loader = loader; self._render_beta(loader); self._render_dispersion(loader)

    def _render_beta(self, loader):
        self.beta_plot.clear(); data = loader.beta_beating_data
        if not data:
            self.reference.setText("Beta beating is not available for this run. The required reference and fitted Twiss data were not persisted.")
            self.beta_stats.clear(); self.beta_plot.hide(); return
        self.beta_plot.show(); source = data["reference_kind"]
        self.reference.setText("Reference: fitted lattice loaded from the resumed run, before this run's corrections." if source == "resumed_fitted_lattice" else "Reference: input lattice loaded at the start of this run, before LOCO corrections.")
        self.beta_stats.setText(f"Horizontal: {_stats(data['beta_beating_x'])}\nVertical: {_stats(data['beta_beating_y'])}")
        ax = self.beta_plot.figure.add_subplot(111)
        ax.plot(data["s"], 100 * data["beta_beating_x"], label="Δβx/βx", linewidth=1.1)
        ax.plot(data["s"], 100 * data["beta_beating_y"], label="Δβy/βy", linewidth=1.1)
        ax.axhline(0, color="#8d95a8", linewidth=.8); ax.set(xlabel="Longitudinal position s [m]", ylabel="Beta beating [%]")
        ax.grid(True, alpha=.25); ax.legend(ncols=2); self.beta_plot.apply_theme(); self.beta_plot.canvas.draw_idle()

    def _render_dispersion(self, loader):
        self.dispersion_plot.clear(); self.dispersion_stats.clear(); self.dispersion_table.setRowCount(0)
        data = loader.dispersion_data
        if not data:
            objective = "included in" if loader.dispersion_included else "not included in"
            self.dispersion_message.setText(f"Dispersion was {objective} the LOCO objective. {loader.dispersion_unavailable_reason}")
            self.dispersion_table.hide(); self.dispersion_plot.hide(); return
        self.dispersion_plot.show(); self.dispersion_table.show(); planes = [(key, data[key]) for key in ("x", "y") if key in data]
        self.dispersion_message.setText("Dispersion was included in the LOCO objective." if loader.dispersion_included else "Dispersion was not included in the LOCO objective. The comparison below is an independent post-fit diagnostic.")
        self.dispersion_stats.setText("Measured RF orbit differences are converted to physical dispersion using pyLOCO's −αc·fRF/Δf convention. Residuals are measured − model in physical units, not χ².")
        axes = self.dispersion_plot.figure.subplots(len(planes), 1, squeeze=False).ravel(); stat_lines = []
        for ax, (plane, values) in zip(axes, planes):
            for kind, label, style in (("measured", "Measured", "-"), ("initial", "Initial model", ":"), ("fitted", "Fitted model", "--")):
                value = values.get(kind)
                if value is not None: ax.plot(data.get("axis", np.arange(np.size(value))), np.asarray(value) * 1000.0, style, label=label, linewidth=1.1)
            ax.set(ylabel=f"η{plane} [mm]", xlabel=data.get("axis_label", "BPM index in saved ordering")); ax.grid(True, alpha=.25); ax.legend(ncols=3)
        stats = loader.dispersion_statistics
        rows = (("RMS before", "rms_before", "mm"), ("RMS after", "rms_after", "mm"),
                ("Improvement", "improvement", "%"), ("Mean before", "mean_before", "mm"),
                ("Mean after", "mean_after", "mm"), ("Min / max before", "minmax_before", "mm"),
                ("Min / max after", "minmax_after", "mm"), ("Max |residual| before", "max_abs_before", "mm"),
                ("Max |residual| after", "max_abs_after", "mm"))
        self.dispersion_table.setRowCount(len(rows))
        for row, (label, key, unit) in enumerate(rows):
            self.dispersion_table.setItem(row, 0, QTableWidgetItem(label))
            for column, plane in enumerate(("x", "y"), 1):
                values = stats.get(plane, {})
                if key.startswith("minmax_"):
                    suffix = key.split("_", 1)[1]; low, high = values.get(f"min_{suffix}"), values.get(f"max_{suffix}")
                    text = "—" if low is None or high is None else f"{1000*low:.4g} / {1000*high:.4g} {unit}"
                else:
                    value = values.get(key)
                    scale = 1.0 if unit == "%" else 1000.0
                    text = "—" if value is None else f"{scale*value:.4g} {unit}"
                self.dispersion_table.setItem(row, column, QTableWidgetItem(text))
        self.dispersion_table.resizeColumnsToContents(); self.dispersion_table.setFixedHeight(min(300, self.dispersion_table.verticalHeader().length() + self.dispersion_table.horizontalHeader().height() + 4))
        self.dispersion_plot.apply_theme(); self.dispersion_plot.canvas.draw_idle()
