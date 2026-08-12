"""Dispersion result view; beta functions are shown only when persisted."""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from .plot_canvas import PlotCanvas


class OpticsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        self.message = QLabel("No optics results loaded."); self.message.setWordWrap(True)
        self.twiss = QLabel("Beta/Twiss results are not persisted for this run."); self.twiss.setWordWrap(True)
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=420)
        layout = QVBoxLayout(self); layout.addWidget(self.message); layout.addWidget(self.plot); layout.addWidget(QLabel("Beta beating / Twiss")); layout.addWidget(self.twiss)

    def set_loader(self, loader):
        self.loader = loader; self.plot.clear()
        data = loader.dispersion_data
        if not loader.dispersion_included:
            self.message.setText("Dispersion was not included in this LOCO fit. Beta functions were not persisted by this run, so no optics curve is invented.")
            self.plot.hide(); return
        if not data or not any(value is not None for value in data.values()):
            self.message.setText("Dispersion was included, but the vectors needed for a reliable comparison were not persisted or could not be resolved.")
            self.plot.hide(); return
        self.plot.show(); self.message.setText("Dispersion comparison. Residual is measured − fitted; plotted values use the saved full-resolution vectors.")
        axis = self.plot.figure.add_subplot(111)
        styles = (("measured", "Measured", "-"), ("initial", "Initial model", ":"), ("fitted", "Fitted model", "--"))
        for key, label, style in styles:
            value = data.get(key)
            if value is not None: axis.plot(value, style, label=label, linewidth=1.2)
        axis.set(xlabel="BPM row in saved ordering", ylabel="Dispersion response [m]", title="Dispersion")
        axis.grid(True, alpha=.25); axis.legend(); self.plot.apply_theme(); self.plot.canvas.draw_idle()
