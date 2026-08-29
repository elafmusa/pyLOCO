"""Results workspace coordinating run state and scientific result views."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from .log_view import LogView
from .orm_view import OrmView
from .overview_view import OverviewView
from .results_loader import ResultsLoader
from .parameters_view import ParametersView
from .optics_view import OpticsView
from .svd_view import SvdView
from .files_view import FilesView
from .run_summary_view import RunSummaryView


class ResultsWorkspace(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.loader = None
        self.base_loader = None
        title = QLabel("Results Workspace"); title.setObjectName("pageTitle")
        self.run_status_label = QLabel("No LOCO run has been started.")
        self.run_status_label.setWordWrap(True)
        self.run_iteration_label = QLabel("Not running")
        self.run_elapsed_label = QLabel("0.0 s")
        self.run_progress = QProgressBar(); self.run_progress.setRange(0, 100); self.run_progress.setValue(0)
        self.run_progress.setFormat("%p%")
        self.run_progress.setMinimumHeight(24)
        self._last_progress_value = 0
        self.run_output_dir = QLineEdit("—")
        self.run_output_dir.setReadOnly(True)
        self.run_output_dir.setObjectName("runOutputDirectory")
        self.run_output_dir.setToolTip("The folder where this run's results are saved. The full path can be selected and copied.")
        self.run_output_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cancel_button = QPushButton("Cancel Run"); self.cancel_button.setEnabled(False)
        self.waiting_games_button = QPushButton("🎮 Take a LOCO break")
        self.waiting_games_button.setToolTip("Open optional lightweight games while the fit runs")
        self.waiting_games_button.hide()
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.addRow("Status", self.run_status_label)
        form.addRow("Iteration", self.run_iteration_label)
        form.addRow("Elapsed", self.run_elapsed_label)
        form.addRow("Progress", self.run_progress); form.addRow("Results directory", self.run_output_dir)
        self._progress_label = form.labelForField(self.run_progress)
        self._output_label = form.labelForField(self.run_output_dir)
        self.monitor = QGroupBox("LOCO Fit"); self.monitor.setLayout(form)
        monitor_actions = QVBoxLayout(); monitor_actions.addWidget(self.cancel_button); monitor_actions.addWidget(self.waiting_games_button); monitor_actions.addStretch(1)
        monitor_row = QHBoxLayout(); monitor_row.addWidget(self.monitor, 1); monitor_row.addLayout(monitor_actions)
        self.compact_monitor = QWidget()
        compact_layout = QHBoxLayout(self.compact_monitor); compact_layout.setContentsMargins(8, 2, 8, 2)
        self.compact_status = QLabel("✓ Completed")
        self.details_button = QPushButton("Details ▾")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_monitor_details)
        compact_layout.addWidget(self.compact_status); compact_layout.addStretch(1); compact_layout.addWidget(self.details_button)
        self.compact_monitor.hide()
        self.tabs = QTabWidget()
        self.overview = OverviewView(); self.orm = OrmView(); self.optics = OpticsView()
        self.parameters = ParametersView(); self.summary = RunSummaryView(); self.svd = SvdView(); self.files = FilesView(); self.log = LogView()
        # Result views own their space directly. Wrapping every view in another
        # scroll area made Matplotlib retain a large size hint instead of
        # shrinking with the main window.
        self.tabs.addTab(self.overview, "Overview"); self.tabs.addTab(self.orm, "ORM")
        self.tabs.addTab(self.optics, "Optics")
        self.tabs.addTab(self.parameters, "Parameters")
        self.tabs.addTab(self.summary, "Run Summary")
        self.tabs.addTab(self.svd, "Jacobian/SVD")
        self.tabs.addTab(self.files, "Files")
        self.tabs.addTab(self.log, "Log")
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.iteration_selector = QComboBox()
        self.iteration_selector.setObjectName("resultsIterationSelector")
        self.iteration_selector.setMinimumWidth(220)
        self.iteration_selector.currentIndexChanged.connect(self._select_iteration)
        self.iteration_notice = QLabel("No iteration history loaded.")
        self.iteration_notice.setWordWrap(True)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("View fitted state:"))
        selector_row.addWidget(self.iteration_selector)
        selector_row.addWidget(self.iteration_notice, 1)
        layout = QVBoxLayout(self); layout.setContentsMargins(16, 14, 16, 16)
        layout.addWidget(title); layout.addLayout(monitor_row); layout.addWidget(self.compact_monitor)
        layout.addLayout(selector_row); layout.addWidget(self.tabs, 1)

    def _toggle_monitor_details(self, checked: bool) -> None:
        self.monitor.setVisible(checked)
        self.details_button.setText("Details ▴" if checked else "Details ▾")

    def begin_run(self) -> None:
        self.loader = None
        self.log.clear_for_run()
        self.run_status_label.setText("Initializing LOCO fit…")
        self.run_iteration_label.setText("Preparing workflow")
        self.run_elapsed_label.setText("Elapsed: 0s")
        self._last_progress_value = 0
        self.run_progress.setRange(0, 100); self.run_progress.setValue(0)
        self.run_output_dir.setText("Preparing results directory…")
        self.run_output_dir.setCursorPosition(0)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.waiting_games_button.setVisible(True)
        self.compact_monitor.hide()
        self.monitor.show()
        self.details_button.setChecked(False)
        for widget in (self.run_progress, self.run_output_dir, self._progress_label, self._output_label):
            widget.setVisible(True)
        self.monitor.setMaximumHeight(16777215)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    @staticmethod
    def format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            value = f"{hours}h {minutes:02d}m {secs:02d}s"
        elif minutes:
            value = f"{minutes}m {secs:02d}s"
        else:
            value = f"{secs}s"
        return f"Elapsed: {value}"

    def update_progress(self, event: dict) -> None:
        fraction = min(1.0, max(0.0, float(event.get("workflow_fraction", 0.0))))
        value = max(self._last_progress_value, int(round(100.0 * fraction)))
        self._last_progress_value = value
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(value)
        iteration = int(event.get("iteration", 0) or 0)
        total = int(event.get("total_iterations", 0) or 0)
        self.run_iteration_label.setText(
            f"Iteration {iteration} of {total}" if iteration > 0 else f"Preparing {total} iteration(s)"
        )
        self.run_status_label.setText(str(event.get("message") or event.get("phase") or "Running LOCO"))

    def append_log(self, message: str) -> None:
        self.log.append(message)
        if message.startswith("Results directory:"):
            output_dir = message.split(":", 1)[1].strip()
            self.run_output_dir.setText(output_dir)
            self.run_output_dir.setToolTip(output_dir)
            self.run_output_dir.setCursorPosition(0)

    def complete_run(self, result) -> None:
        self._last_progress_value = 100
        self.run_progress.setRange(0, 100); self.run_progress.setValue(100)
        self.run_status_label.setText("LOCO completed")
        self.run_elapsed_label.setText(self.format_elapsed(result.elapsed_seconds))
        self.run_output_dir.setText(result.results_dir)
        self.run_output_dir.setToolTip(result.results_dir)
        self.run_output_dir.setCursorPosition(0); self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.waiting_games_button.hide()
        self.load_results(result.results_dir, runtime=result.elapsed_seconds)
        self.run_progress.setVisible(True)
        self._progress_label.setVisible(True)
        self.run_status_label.setText("LOCO completed")
        self.run_elapsed_label.setText(self.format_elapsed(result.elapsed_seconds))
        log_path = Path(result.results_dir) / "backend.log"
        if log_path.exists():
            try: self.log.set_log(log_path.read_text(encoding="utf-8"))
            except OSError: pass
        self.tabs.setCurrentIndex(0)
        self.compact_status.setText(f"✓ LOCO completed — 100% — Results saved")
        self.compact_monitor.hide()
        self.monitor.show()
        self.details_button.setChecked(False)
        self.monitor.setMaximumHeight(16777215)

    def load_results(self, results_dir, *, runtime=None) -> None:
        path = Path(results_dir).expanduser()
        if not path.exists():
            self.loader = None; self.run_status_label.setText(f"Saved results directory is unavailable: {path}"); return
        self.base_loader = ResultsLoader(path, runtime=runtime)
        entries = self.base_loader.iteration_entries
        self.iteration_selector.blockSignals(True)
        self.iteration_selector.clear()
        for entry in entries:
            self.iteration_selector.addItem(entry["label"], entry.get("iteration"))
        self.iteration_selector.setCurrentIndex(len(entries) - 1)
        self.iteration_selector.blockSignals(False)
        legacy = bool(entries[0].get("legacy"))
        self.iteration_notice.setText(
            "This older run saved only its final state; intermediate iterations are unavailable."
            if legacy else "The selected state updates all applicable result views; χ² retains the complete convergence history."
        )
        self._select_iteration(self.iteration_selector.currentIndex())
        self.loader = self.loader or self.base_loader
        log_path = path / "backend.log"
        if log_path.exists():
            try: self.log.set_log(log_path.read_text(encoding="utf-8"))
            except OSError: pass
        self.run_status_label.setText("Completed run restored from project")
        self.run_output_dir.setText(str(path)); self.run_elapsed_label.setText("Unavailable" if self.loader.runtime is None else f"{self.loader.runtime:.1f} s")
        self.cancel_button.setVisible(False); self.waiting_games_button.hide()
        for widget in (self.run_progress, self._progress_label): widget.setVisible(False)
        self.compact_status.setText("✓ Completed run restored — Results available")
        self.compact_monitor.show(); self.monitor.hide(); self.details_button.setChecked(False)

    def _select_iteration(self, index: int) -> None:
        if self.base_loader is None or index < 0:
            return
        iteration = self.iteration_selector.itemData(index)
        self.loader = self.base_loader.for_iteration(iteration)
        for view in (self.overview, self.orm, self.optics, self.parameters, self.summary, self.svd, self.files):
            view.set_loader(self.loader)

    def fail_run(self, *, cancelled: bool = False) -> Path | None:
        self.run_progress.setRange(0, 100); self.run_progress.setValue(self._last_progress_value)
        self.run_status_label.setText("Cancelled" if cancelled else "Failed"); self.cancel_button.setEnabled(False)
        self.waiting_games_button.hide()
        candidate = Path(self.run_output_dir.text()).expanduser()
        manifest = candidate / "iterations" / "manifest.json"
        if candidate.is_dir() and manifest.is_file():
            self.load_results(candidate)
            self.run_progress.setVisible(True)
            self._progress_label.setVisible(True)
            self.monitor.show()
            self.compact_monitor.hide()
            self.run_status_label.setText(("Run cancelled" if cancelled else "Run failed") + "; completed iteration states remain available")
            self.compact_status.setText(("■ Run cancelled" if cancelled else "⚠ Run incomplete") + " — completed iterations preserved")
            self.tabs.setCurrentIndex(0)
            return candidate
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        return None

    def apply_theme(self) -> None:
        self.overview.chi_plot.apply_theme()
        self.orm.plot.apply_theme()
        self.optics.apply_theme()
        self.parameters.plot.apply_theme()
        self.svd.plot.apply_theme()

    def set_mode(self, mode: str) -> None:
        """Keep the scientific essentials prominent in Basic mode."""
        advanced = mode == "Advanced"
        for label in ("Jacobian/SVD", "Files", "Log"):
            index = next((i for i in range(self.tabs.count()) if self.tabs.tabText(i) == label), -1)
            if index >= 0:
                self.tabs.setTabVisible(index, advanced)
