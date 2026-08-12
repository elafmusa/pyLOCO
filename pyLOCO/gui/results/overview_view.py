"""Concise scientific overview for a completed LOCO run."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .plot_canvas import PlotCanvas


def scientific(value, unavailable="Not persisted") -> str:
    return unavailable if value is None else f"{value:.3e}"


class Metric(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = QLabel(title); self.title.setObjectName("resultMetricTitle")
        self.value = QLabel("—"); self.value.setObjectName("resultMetricValue")
        self.value.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self); layout.setContentsMargins(10, 6, 10, 6); layout.addWidget(self.title); layout.addWidget(self.value)


class OverviewView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.loader = None
        heading = QHBoxLayout()
        title = QLabel("LOCO Results"); title.setObjectName("resultsTitle")
        self.status = QLabel("No completed run"); self.status.setObjectName("resultsStatus")
        heading.addWidget(title); heading.addStretch(1); heading.addWidget(self.status)
        self.metrics = {name: Metric(label) for name, label in (
            ("initial", "Initial χ²"), ("final", "Final χ²"), ("reduction", "Reduction"),
            ("iterations", "Iterations"), ("runtime", "Runtime"),
        )}
        metrics_row = QGridLayout()
        for index, metric in enumerate(self.metrics.values()):
            metrics_row.addWidget(metric, index // 3, index % 3)
        self.metadata = QLabel("No result metadata available."); self.metadata.setWordWrap(True)
        self.metadata.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.chi_plot = PlotCanvas(show_toolbar=False, minimum_height=240)
        chi_group = QGroupBox("χ² convergence"); chi_layout = QVBoxLayout(chi_group); chi_layout.addWidget(self.chi_plot)
        self.orm_summary = QLabel("ORM residual metrics are unavailable."); self.orm_summary.setObjectName("ormSummary")
        self.orm_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        orm_group = QGroupBox("Raw ORM residual RMS (not weighted χ²)"); orm_layout = QVBoxLayout(orm_group); orm_layout.addWidget(self.orm_summary)
        self.diagnostics = QLabel("No diagnostics available."); self.diagnostics.setWordWrap(True)
        diagnostics_group = QGroupBox("Diagnostics"); diagnostics_layout = QVBoxLayout(diagnostics_group); diagnostics_layout.addWidget(self.diagnostics)
        layout = QVBoxLayout(self); layout.addLayout(heading); layout.addLayout(metrics_row); layout.addWidget(self.metadata)
        layout.addWidget(chi_group, 2); layout.addWidget(orm_group); layout.addWidget(diagnostics_group)

    def set_loader(self, loader) -> None:
        self.loader = loader
        self.status.setText("✓ Fit completed")
        self.metrics["initial"].value.setText(scientific(loader.initial_chi2, "Not persisted for this run"))
        self.metrics["final"].value.setText(scientific(loader.final_chi2, "Unavailable"))
        reduction = loader.chi2_reduction_percent
        self.metrics["reduction"].value.setText("—" if reduction is None else f"{reduction:.1f}% {'↓' if reduction >= 0 else '↑'}")
        requested = loader.requested_iterations
        self.metrics["iterations"].value.setText(f"{loader.completed_iterations}" + (f" / {requested}" if requested is not None else ""))
        self.metrics["runtime"].value.setText("Unavailable" if loader.runtime is None else f"{loader.runtime:.1f} s")
        parts = loader.partitions
        meta = []
        if parts: meta += [f"{parts.n_hbpm} BPMs", f"{parts.n_hcor} H correctors", f"{parts.n_vcor} V correctors"]
        if loader.fitted_parameter_count is not None: meta.append(f"{loader.fitted_parameter_count} fitted DOFs")
        if loader.fit_method: meta.append(loader.fit_method)
        if loader.regularization is not None: meta.append(f"λ = {loader.regularization:g}")
        self.metadata.setText("   •   ".join(meta) if meta else "No run metadata available.")
        before, after, improvement = loader.orm_rms_before, loader.orm_rms_after, loader.orm_improvement_percent
        if before is None or after is None:
            self.orm_summary.setText("Measured, initial-model, or fitted ORM is unavailable for this run.")
        else:
            text = f"Before fit: {before:.3e} m/rad    After fit: {after:.3e} m/rad"
            if improvement is not None: text += f"    Improvement: {improvement:.1f}% {'↓' if improvement >= 0 else '↑'}"
            self.orm_summary.setText(text)
        icons = {"success": "✓", "warning": "⚠", "info": "ℹ"}
        self.diagnostics.setText("\n".join(f"{icons.get(level, '•')} {message}" for level, message in loader.diagnostics))
        self._draw_chi2()

    def _draw_chi2(self) -> None:
        canvas = self.chi_plot; canvas.clear(); theme = canvas.theme()
        values = self.loader.chi2_history if self.loader else []
        initial = self.loader.initial_chi2 if self.loader else None
        ax = canvas.figure.add_subplot(111); ax.set_facecolor(theme["axes"])
        if values or initial is not None:
            import numpy as np
            plotted = ([initial] if initial is not None else []) + list(values)
            x = np.arange(0 if initial is not None else 1, len(values) + 1)
            ax.plot(x, plotted, marker="o", color="#7E57C2", linewidth=1.8)
            if len(plotted) == 1: ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
            if initial is not None:
                ax.set_xticks(x, ["Initial"] + [str(i) for i in range(1, len(values) + 1)])
            elif values:
                ax.ticklabel_format(axis="x", style="plain")
            positive = [v for v in plotted if v is not None and v > 0]
            if positive and max(positive) / min(positive) >= 100: ax.set_yscale("log")
            ax.set_xlabel("LOCO correction iteration"); ax.set_ylabel("Weighted LOCO χ²")
        else:
            ax.text(0.5, 0.5, "No χ² history is available.", ha="center", va="center", transform=ax.transAxes, color=theme["text"])
            ax.set_xticks([]); ax.set_yticks([])
        canvas.apply_theme()
        canvas.canvas.draw()
