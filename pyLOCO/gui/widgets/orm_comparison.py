"""Interactive measured/model ORM comparison widgets for the Qt GUI."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class RenderedMatrix:
    """Decimated matrix plus index axes used for responsive rendering."""

    values: np.ndarray
    correctors: np.ndarray
    bpms: np.ndarray


class OrmComparisonWindow(QDialog):
    """Interactive side-by-side measured/model/difference ORM viewer.

    The viewer accepts already-computed NumPy arrays and never recomputes an ORM.
    Large matrices are decimated only for rendering; RMS and color limits are
    computed from the original full-resolution data.
    """

    def __init__(self, measured_orm, model_orm, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.measured_orm = self._as_matrix(measured_orm, "Measured ORM")
        self.model_orm = self._as_matrix(model_orm, "Model ORM")
        if self.measured_orm.shape != self.model_orm.shape:
            raise ValueError(
                "Measured and model ORMs must have the same shape; "
                f"got {self.measured_orm.shape} and {self.model_orm.shape}."
            )
        if importlib.util.find_spec("matplotlib") is None:
            raise RuntimeError("Matplotlib is required for the ORM Comparison viewer.")

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.FigureCanvasQTAgg = FigureCanvasQTAgg
        self.NavigationToolbar2QT = NavigationToolbar2QT
        self.Figure = Figure
        self.difference_orm = self.measured_orm - self.model_orm
        self._syncing_view = False
        self._last_limits: tuple[float, float] | None = None

        self.setWindowTitle("ORM Comparison")
        self.resize(1500, 900)
        layout = QVBoxLayout(self)
        self.rms_label = QLabel(self._rms_text())
        self.rms_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.rms_label)
        layout.addLayout(self._controls())
        self.figure = self.Figure(figsize=(12, 6), constrained_layout=True, facecolor="#1E1E2E")
        self.canvas = self.FigureCanvasQTAgg(self.figure)
        self.toolbar = self.NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.canvas.mpl_connect("button_release_event", self._sync_3d_views)
        self.canvas.mpl_connect("scroll_event", self._sync_3d_views)
        self._redraw()

    def _controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["3D surface", "2D heatmap"])
        self.plot_mode.currentTextChanged.connect(self._redraw)
        self.shared_scale = QCheckBox("Shared color scale")
        self.shared_scale.setChecked(True)
        self.shared_scale.toggled.connect(self._redraw)
        self.fixed_scale = QCheckBox("Fixed color limits")
        self.fixed_scale.toggled.connect(self._redraw)
        self.vmin_spin = QDoubleSpinBox()
        self.vmax_spin = QDoubleSpinBox()
        for spin in (self.vmin_spin, self.vmax_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(9)
            spin.setSingleStep(1e-6)
            spin.valueChanged.connect(self._redraw)
        vmin, vmax = self._auto_limits()
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        export = QPushButton("Export figure…")
        export.clicked.connect(self.export_figure)
        for widget in (
            QLabel("View"), self.plot_mode, self.shared_scale, self.fixed_scale,
            QLabel("Min"), self.vmin_spin, QLabel("Max"), self.vmax_spin, export,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        return row

    @staticmethod
    def _as_matrix(value, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{label} must be a 2D matrix; got shape {array.shape}.")
        return array

    def _rms_text(self) -> str:
        finite = np.isfinite(self.difference_orm)
        rms = float(np.sqrt(np.mean(np.square(self.difference_orm[finite])))) if np.any(finite) else float("nan")
        return f"RMS(Measured − Model): {rms:.6g} m/rad    Matrix shape: {self.measured_orm.shape[0]} BPM rows × {self.measured_orm.shape[1]} correctors"

    def _auto_limits(self) -> tuple[float, float]:
        values = np.concatenate([self.measured_orm.ravel(), self.model_orm.ravel()])
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0, 1.0
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        return float(vmin), float(vmax)

    def _limits(self, matrix: np.ndarray | None = None) -> tuple[float, float]:
        if self.fixed_scale.isChecked():
            return self.vmin_spin.value(), self.vmax_spin.value()
        if self.shared_scale.isChecked() or matrix is None:
            return self._auto_limits()
        finite = matrix[np.isfinite(matrix)]
        return (float(np.nanmin(finite)), float(np.nanmax(finite))) if finite.size else (-1.0, 1.0)

    def _decimate(self, matrix: np.ndarray, max_points: int = 120_000) -> RenderedMatrix:
        rows, cols = matrix.shape
        step = max(1, int(np.ceil(np.sqrt((rows * cols) / max_points))))
        bpms = np.arange(rows)[::step]
        correctors = np.arange(cols)[::step]
        return RenderedMatrix(matrix[::step, ::step], correctors, bpms)

    def _redraw(self) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor("#1E1E2E")
        matrices = [("Measured ORM", self.measured_orm), ("Model ORM", self.model_orm), ("Difference (Measured − Model)", self.difference_orm)]
        is_surface = self.plot_mode.currentText().startswith("3D")
        axes = []
        for idx, (title, matrix) in enumerate(matrices, start=1):
            ax = self.figure.add_subplot(1, 3, idx, projection="3d" if is_surface else None)
            axes.append(ax)
            vmin, vmax = self._limits(matrix)
            rendered = self._decimate(matrix)
            if is_surface:
                x, y = np.meshgrid(rendered.correctors, rendered.bpms)
                artist = ax.plot_surface(x, y, rendered.values, cmap="viridis", vmin=vmin, vmax=vmax, linewidth=0, antialiased=False, rcount=rendered.values.shape[0], ccount=rendered.values.shape[1])
                ax.set_zlabel("ORM value (m/rad)", color="#DDE3F0", labelpad=8)
                ax.zaxis.label.set_fontsize(10)
            else:
                artist = ax.imshow(rendered.values, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=[rendered.correctors[0], rendered.correctors[-1], rendered.bpms[0], rendered.bpms[-1]])
            self._style_axes(ax, is_surface)
            ax.set_title(title, color="#FFFFFF", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Correctors", color="#DDE3F0", labelpad=8)
            ax.set_ylabel("BPMs", color="#DDE3F0", labelpad=8)
            colorbar = self.figure.colorbar(artist, ax=ax, shrink=0.72, label="ORM (m/rad)")
            colorbar.ax.yaxis.label.set_color("#DDE3F0")
            colorbar.ax.tick_params(colors="#DDE3F0", labelsize=9)
            colorbar.outline.set_edgecolor("#4A4F68")
        self._axes = axes
        self.canvas.draw_idle()


    @staticmethod
    def _style_axes(ax, is_surface: bool) -> None:
        ax.set_facecolor("#25283A")
        ax.tick_params(colors="#DDE3F0", labelsize=9)
        for spine in getattr(ax, "spines", {}).values():
            spine.set_color("#4A4F68")
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)
        if is_surface:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.157, 0.173, 0.243, 1.0))
                axis._axinfo["grid"]["color"] = (0.36, 0.39, 0.52, 0.45)
            ax.zaxis.set_tick_params(colors="#DDE3F0", labelsize=9)
        else:
            ax.grid(color="#4A4F68", alpha=0.28, linewidth=0.6)

    def _sync_3d_views(self, event) -> None:
        if self._syncing_view or not hasattr(self, "_axes") or not self.plot_mode.currentText().startswith("3D"):
            return
        if event.inaxes not in self._axes:
            return
        source = event.inaxes
        self._syncing_view = True
        try:
            for ax in self._axes:
                if ax is not source:
                    ax.view_init(elev=source.elev, azim=source.azim, roll=getattr(source, "roll", 0))
                    ax.set_xlim(source.get_xlim())
                    ax.set_ylim(source.get_ylim())
                    ax.set_zlim(source.get_zlim())
            self.canvas.draw_idle()
        finally:
            self._syncing_view = False

    def export_figure(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self,
            "Export ORM comparison figure",
            "orm-comparison.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )[0]
        if not filename:
            return
        try:
            self.figure.savefig(filename, dpi=180)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
