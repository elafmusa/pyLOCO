"""Transparent SVD/Jacobian metadata without recomputation."""

import json
import shutil
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from pyLOCO.gui.file_dialogs import FileDialog as QFileDialog
from .plot_canvas import PlotCanvas


class SvdView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        self.form = QFormLayout(); self.values = {}
        for key, label in (("method", "Saved SVD method"), ("threshold", "Threshold"), ("rank", "Configured rank"), ("measurement_values", "Measurement-vector values"), ("fitted_dofs", "Fitted DOFs")):
            widget = QLabel("—"); self.values[key] = widget; self.form.addRow(label, widget)
        for key, label in (("included", "Included singular values"),
                           ("rejected", "Cut singular values"),
                           ("absolute_cutoff", "Absolute cutoff")):
            widget = QLabel("—"); self.values[key] = widget; self.form.addRow(label, widget)
        self.note = QLabel(); self.note.setWordWrap(True)
        self.save_jacobian = QPushButton("Save Jacobian…"); self.save_jacobian.setEnabled(False)
        self.save_jacobian.clicked.connect(self._save_jacobian)
        # Leave enough vertical space for logarithmic tick labels.  At the old
        # 140 px minimum the toolbar consumed most of the canvas and macOS/Qt
        # rendered several powers-of-ten labels on top of one another.
        self.plot = PlotCanvas(show_toolbar=True, minimum_height=300); self.plot.hide()
        layout = QVBoxLayout(self); layout.addLayout(self.form); layout.addWidget(self.note); layout.addWidget(self.save_jacobian); layout.addWidget(self.plot)

    def set_loader(self, loader):
        self.loader = loader; metadata = loader.svd_metadata
        for key in ("method", "threshold", "rank", "measurement_values", "fitted_dofs"):
            value = metadata.get(key)
            self.values[key].setText("Not persisted" if value is None else str(value))
        for key in ("included", "rejected", "absolute_cutoff"):
            self.values[key].setText("—")
        self.save_jacobian.setEnabled(loader.jacobian_available)
        spectrum = metadata.get("spectrum")
        self.plot.clear()
        if spectrum is None:
            self.note.setText(f"Exact calculated Jacobian persisted with shape {tuple(loader.jacobian_metadata.get('shape', ()))}." if loader.jacobian_available else "The singular-value spectrum and Jacobian matrix were not persisted in this result directory. This view does not recompute them or substitute another quantity.")
            self.plot.hide(); return
        source = metadata.get("spectrum_source")
        source_note = (
            "final-run artifact"
            if source and Path(source).parent == loader.result_dir
            else "iteration artifact"
        )
        values = np.ravel(np.asarray(spectrum, dtype=float))
        positive = np.isfinite(values) & (values > 0.0)
        omitted = int(values.size - np.count_nonzero(positive))
        suffix = (
            f" {omitted} zero or non-finite value(s) are not shown on the logarithmic axis."
            if omitted else ""
        )
        self.note.setText(f"Singular values loaded from the persisted {source_note}.{suffix}"); self.plot.show()
        axis = self.plot.figure.add_subplot(111)
        indices = np.arange(values.size)
        if np.any(positive):
            axis.set_yscale("log")
            low = float(np.min(values[positive])); high = float(np.max(values[positive]))
            method=str(metadata.get("method") or "")
            threshold=metadata.get("threshold")
            rank=metadata.get("rank")
            cutoff=None
            if method=="threshold" and threshold is not None and np.isfinite(float(threshold)):
                cutoff=float(threshold)*high
                self.values["absolute_cutoff"].setText(f"{cutoff:.6g}")
                self.note.setText(self.note.text()+f" Absolute cutoff: {cutoff:.6g} ({float(threshold):.6g} × maximum singular value {high:.6g}).")
            retained = int(np.count_nonzero(positive))
            if rank is not None:
                retained=max(0,min(int(rank),values.size))
                if 0<retained<values.size:
                    axis.axvline(retained-.5,color="#8B5CF6",linestyle=":",linewidth=1.4,
                                 label=f"Retained rank = {retained}")
                if retained:
                    smallest=float(values[retained-1])
                    if np.isfinite(smallest) and smallest>0:
                        self.note.setText(self.note.text()+f" Smallest retained singular value at rank {retained}: {smallest:.6g}.")
            retained_mask = positive & (indices < retained)
            rejected_mask = positive & ~retained_mask
            self.values["included"].setText(str(int(np.count_nonzero(retained_mask))))
            self.values["rejected"].setText(str(int(np.count_nonzero(rejected_mask))))
            axis.plot(indices, np.where(retained_mask, values, np.nan), color="#19a974", linewidth=1.8,
                      marker="o", markersize=2.5, label=f"Included ({np.count_nonzero(retained_mask)})")
            if np.any(rejected_mask):
                axis.plot(indices, np.where(rejected_mask, values, np.nan), color="#8B93A7", linewidth=1.2,
                          alpha=.8, label=f"Cut ({np.count_nonzero(rejected_mask)})")
                axis.axvspan(retained-.5, values.size-.5, color="#E88B22", alpha=.08)
            # A constant or extremely narrow spectrum otherwise gives
            # effectively coincident log tick labels.  Add symmetric padding
            # in log space without changing any persisted values.
            if np.isclose(np.log10(low), np.log10(high), rtol=0.0, atol=1e-9):
                axis.set_ylim(low / 10.0, high * 10.0)
            if cutoff is not None and cutoff>0:
                bottom,top=axis.get_ylim()
                if bottom <= cutoff <= top:
                    axis.axhline(cutoff,color="#E88B22",linestyle="--",linewidth=1.4,
                                 label=f"Threshold cutoff = {cutoff:.4g}")
                else:
                    location = "below" if cutoff < bottom else "above"
                    axis.text(.99, .03 if location == "below" else .97,
                              f"Threshold cutoff {cutoff:.4g} ({location} visible spectrum)",
                              color="#E88B22", ha="right", va="bottom" if location == "below" else "top",
                              transform=axis.transAxes)
            from matplotlib.ticker import LogLocator
            axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
        else:
            axis.text(
                0.5, 0.5, "No positive finite singular values to plot",
                ha="center", va="center", transform=axis.transAxes,
            )
        axis.set(xlabel="Singular-value index", ylabel="Singular value", title="Singular-value spectrum")
        if len(axis.lines)>1:axis.legend()
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
