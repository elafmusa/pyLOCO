"""Transparent SVD/Jacobian metadata without recomputation."""

import json
import shutil
from PySide6.QtWidgets import QFileDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from .plot_canvas import PlotCanvas


class SvdView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        self.form = QFormLayout(); self.values = {}
        for key, label in (("method", "Saved SVD method"), ("threshold", "Threshold"), ("rank", "Configured rank"), ("measurement_values", "Measurement-vector values"), ("fitted_dofs", "Fitted DOFs")):
            widget = QLabel("—"); self.values[key] = widget; self.form.addRow(label, widget)
        self.note = QLabel(); self.note.setWordWrap(True)
        self.save_jacobian = QPushButton("Save Jacobian…"); self.save_jacobian.setEnabled(False)
        self.save_jacobian.clicked.connect(self._save_jacobian)
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=140); self.plot.hide()
        layout = QVBoxLayout(self); layout.addLayout(self.form); layout.addWidget(self.note); layout.addWidget(self.save_jacobian); layout.addWidget(self.plot)

    def set_loader(self, loader):
        self.loader = loader; metadata = loader.svd_metadata
        for key, widget in self.values.items(): widget.setText("Not persisted" if metadata.get(key) is None else str(metadata[key]))
        self.save_jacobian.setEnabled(loader.jacobian_available)
        spectrum = metadata.get("spectrum")
        self.plot.clear()
        if spectrum is None:
            self.note.setText(f"Exact calculated Jacobian persisted with shape {tuple(loader.jacobian_metadata.get('shape', ()))}." if loader.jacobian_available else "The singular-value spectrum and Jacobian matrix were not persisted in this result directory. This view does not recompute them or substitute another quantity.")
            self.plot.hide(); return
        self.note.setText("Singular values loaded from the persisted run artifact."); self.plot.show()
        axis = self.plot.figure.add_subplot(111); axis.semilogy(spectrum, color="#19a974")
        axis.set(xlabel="Singular-value index", ylabel="Singular value", title="Persisted singular-value spectrum")
        axis.grid(True, alpha=.25); self.plot.apply_theme(); self.plot.canvas.draw_idle()

    def _save_jacobian(self):
        if self.loader is None or not self.loader.jacobian_available: return
        source = self.loader.jacobian_artifact
        filename = QFileDialog.getSaveFileName(self, "Save Jacobian", source.name, "HDF5 (*.h5);;NumPy archive (*.npz)")[0]
        if not filename: return
        target = filename if filename.endswith(source.suffix) else filename + source.suffix
        shutil.copy2(source, target)
        with open(target[:-len(source.suffix)] + "_metadata.json", "w", encoding="utf-8") as stream:
            json.dump(self.loader.jacobian_metadata, stream, indent=2)
