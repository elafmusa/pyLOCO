"""Reusable theme-aware Matplotlib canvas for scientific result views."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class PlotCanvas(QWidget):
    def __init__(self, parent=None, *, show_toolbar: bool = True, minimum_height: int = 0) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(constrained_layout=True)
        self._requested_minimum_height = minimum_height
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setObjectName("matplotlibToolbar")
        self.toolbar.setVisible(show_toolbar)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(0, minimum_height)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        if show_toolbar:
            actions.addWidget(self.toolbar, 1)
        else:
            actions.addStretch(1)
        self.include_title = QCheckBox("Include plot title")
        self.include_title.setChecked(True)
        self.include_title.setToolTip("Uncheck to export the plot without its axes titles. The on-screen plot is unchanged.")
        actions.addWidget(self.include_title)
        self.save_button = QPushButton("Save plot…")
        self.save_button.setToolTip("Export this plot as PNG, PDF, or SVG")
        self.save_button.clicked.connect(self.save_plot); actions.addWidget(self.save_button)
        layout.addLayout(actions)
        layout.addWidget(self.canvas, 1)
        self.apply_theme()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        """Do not propagate FigureCanvasQTAgg's large preferred height."""
        return QSize(640, max(self._requested_minimum_height, 260))

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(0, self._requested_minimum_height)

    def theme(self) -> dict[str, str]:
        app = QApplication.instance()
        value = app.property("pyLOCOThemePlot") if app else None
        return value if isinstance(value, dict) else {
            "face": "#ffffff", "axes": "#ffffff", "text": "#222436",
            "grid": "#c9ceda", "spine": "#8d95a8", "colormap": "viridis",
        }

    def apply_theme(self) -> None:
        theme = self.theme()
        self.figure.set_facecolor(theme["face"])
        for axis in self.figure.axes:
            axis.set_facecolor(theme["axes"])
            axis.tick_params(colors=theme["text"])
            axis.xaxis.label.set_color(theme["text"])
            axis.yaxis.label.set_color(theme["text"])
            axis.title.set_color(theme["text"])
            for spine in axis.spines.values():
                spine.set_color(theme["spine"])
            for artist in axis.texts:
                if artist.get_gid()=="bpm-identification-tooltip":
                    artist.set_color(theme["text"])
                    patch=artist.get_bbox_patch()
                    if patch is not None:
                        patch.set_facecolor(theme["face"])
                        patch.set_edgecolor(theme["spine"])
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.figure.clear()
        self.figure.set_facecolor(self.theme()["face"])

    def save_plot(self) -> str | None:
        filename = QFileDialog.getSaveFileName(
            self, "Save result plot", "pyloco-result.png",
            "PNG image (*.png);;PDF document (*.pdf);;SVG vector image (*.svg)",
        )[0]
        if not filename: return None
        titles = [(axis, axis.get_title()) for axis in self.figure.axes]
        try:
            if not self.include_title.isChecked():
                for axis, _title in titles:
                    axis.set_title("")
            self.figure.savefig(filename, dpi=200, bbox_inches="tight")
        finally:
            for axis, title in titles:
                axis.set_title(title)
            self.canvas.draw_idle()
        return filename
