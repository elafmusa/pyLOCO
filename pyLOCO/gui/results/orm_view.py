"""Interactive single-matrix ORM analysis view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .plot_canvas import PlotCanvas


class OrmView(QWidget):
    ITEMS = (
        ("Measured", "measured_orm"), ("Initial model", "initial_orm"),
        ("Fitted", "fitted_orm"), ("Residual before", "residual_before"),
        ("Residual after", "residual_after"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.loader = None
        self.selector = QComboBox()
        for label, key in self.ITEMS: self.selector.addItem(label, key)
        self.selector.currentIndexChanged.connect(self.redraw)
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Heatmap", "3D Surface"])
        self.view_mode.currentIndexChanged.connect(self.redraw)
        self.metrics = QLabel("No ORM result loaded.")
        self.metrics.setToolTip(
            "Residual before = measured − initial model; residual after = measured − fitted model. "
            "RMS = sqrt(mean(residual²)) using the full matrix, never the decimated display."
        )
        row = QGridLayout()
        row.addWidget(QLabel("Matrix"), 0, 0); row.addWidget(self.selector, 0, 1)
        row.addWidget(QLabel("View"), 0, 2); row.addWidget(self.view_mode, 0, 3)
        row.addWidget(self.metrics, 1, 0, 1, 4)
        row.setColumnStretch(1, 1); row.setColumnStretch(3, 1)
        self.metrics.setWordWrap(True)
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=140)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.unavailable = QLabel()
        self.unavailable.setAlignment(Qt.AlignCenter)
        self.unavailable.setWordWrap(True)
        self.unavailable.setObjectName("placeholderDescription")
        self.unavailable.setMinimumHeight(100)
        self.unavailable.hide()
        layout = QVBoxLayout(self); layout.addLayout(row); layout.addWidget(self.plot, 1); layout.addWidget(self.unavailable, 1)

    def set_loader(self, loader) -> None:
        self.loader = loader
        self.redraw()

    def redraw(self) -> None:
        canvas = self.plot; canvas.clear(); theme = canvas.theme()
        is_3d = self.view_mode.currentText().startswith("3D")
        key = self.selector.currentData()
        matrix = getattr(self.loader, key, None) if self.loader else None
        if matrix is None:
            self.metrics.setText("Unavailable")
            self.plot.hide()
            reason = self.loader.unavailable_reason(key) if self.loader else None
            self.unavailable.setText(
                f"{self.selector.currentText()} ORM is not available for this run.\n"
                + (f"{reason}" if reason else "Choose another matrix or confirm that the corresponding result artifact exists.")
            )
            self.unavailable.show()
            return
        self.unavailable.hide()
        self.plot.show()
        ax = canvas.figure.add_subplot(111, projection="3d" if is_3d else None); ax.set_facecolor(theme["axes"])
        import numpy as np
        residual = str(key).startswith("residual")
        if residual:
            limits = self.loader.orm_residual_limits
            kwargs = {"cmap": theme["colormap"], "vmin": limits[0], "vmax": limits[1]} if limits else {"cmap": theme["colormap"]}
        else:
            limits = self.loader.orm_raw_limits
            kwargs = {"cmap": theme["colormap"], "vmin": limits[0], "vmax": limits[1]} if limits else {"cmap": theme["colormap"]}
        rows, cols = matrix.shape
        max_render_points = 100_000 if is_3d else 250_000
        step = max(1, int(np.ceil(np.sqrt(matrix.size / max_render_points))))
        shown = matrix[::step, ::step]
        if is_3d:
            correctors = np.arange(cols)[::step]
            bpms = np.arange(rows)[::step]
            x, y = np.meshgrid(correctors, bpms)
            artist = ax.plot_surface(
                x, y, shown, linewidth=0, antialiased=False,
                rcount=shown.shape[0], ccount=shown.shape[1], **kwargs,
            )
            ax.set_zlabel("ORM response [m]", labelpad=8)
            ax.view_init(elev=28, azim=-62)
        else:
            artist = ax.imshow(shown, origin="lower", aspect="auto", extent=(0, cols, 0, rows), **kwargs)
        canvas.figure.colorbar(artist, ax=ax, label="ORM response [m]")
        ax.set_xlabel("Corrector column"); ax.set_ylabel("BPM response row"); ax.set_title(self.selector.currentText())
        parts = self.loader.partitions
        if parts and not is_3d:
            ax.axhline(parts.n_hbpm, color=theme["text"], linewidth=0.8, alpha=0.8)
            ax.axvline(parts.n_hcor, color=theme["text"], linewidth=0.8, alpha=0.8)
            label_style = {"ha": "center", "va": "center", "color": theme["text"], "fontsize": 8,
                           "bbox": {"facecolor": theme["axes"], "edgecolor": "none", "alpha": .72, "pad": 1.5}}
            ax.text(parts.n_hcor / 2, parts.n_hbpm / 2, "H BPM / H cor", **label_style)
            ax.text(parts.n_hcor + parts.n_vcor / 2, parts.n_hbpm / 2, "H BPM / V cor", **label_style)
            ax.text(parts.n_hcor / 2, parts.n_hbpm + parts.n_vbpm / 2, "V BPM / H cor", **label_style)
            ax.text(parts.n_hcor + parts.n_vcor / 2, parts.n_hbpm + parts.n_vbpm / 2, "V BPM / V cor", **label_style)
        finite = matrix[np.isfinite(matrix)]
        rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")
        maximum = float(np.max(np.abs(finite))) if finite.size else float("nan")
        maximum_label = "max |residual|" if residual else "max |value|"
        self.metrics.setText(f"RMS {rms:.3e}   {maximum_label} {maximum:.3e}   {rows} × {cols}" + (f"   display step {step}" if step > 1 else ""))
        canvas.apply_theme()
