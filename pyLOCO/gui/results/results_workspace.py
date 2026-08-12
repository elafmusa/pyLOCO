"""Results workspace coordinating run state and scientific result views."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
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


class ResultsWorkspace(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.loader = None
        title = QLabel("Results Workspace"); title.setObjectName("pageTitle")
        self.run_status_label = QLabel("No LOCO run has been started.")
        self.run_elapsed_label = QLabel("Elapsed: 0.0 s")
        self.run_progress = QProgressBar(); self.run_progress.setRange(0, 1); self.run_progress.setValue(0)
        self.run_output_dir = QLabel("—"); self.run_output_dir.setWordWrap(True)
        self.cancel_button = QPushButton("Cancel Run"); self.cancel_button.setEnabled(False)
        form = QFormLayout(); form.addRow("Status", self.run_status_label); form.addRow("Elapsed", self.run_elapsed_label)
        form.addRow("Progress", self.run_progress); form.addRow("Results directory", self.run_output_dir)
        self._progress_label = form.labelForField(self.run_progress)
        self._output_label = form.labelForField(self.run_output_dir)
        self.monitor = QGroupBox("Backend Run Monitor"); self.monitor.setLayout(form)
        monitor_row = QHBoxLayout(); monitor_row.addWidget(self.monitor, 1); monitor_row.addWidget(self.cancel_button)
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
        self.parameters = ParametersView(); self.svd = SvdView(); self.files = FilesView(); self.log = LogView()
        self.tabs.addTab(self._scrollable(self.overview), "Overview"); self.tabs.addTab(self._scrollable(self.orm), "ORM")
        self.tabs.addTab(self._scrollable(self.optics), "Optics")
        self.tabs.addTab(self._scrollable(self.parameters), "Parameters")
        self.tabs.addTab(self._scrollable(self.svd), "Jacobian/SVD")
        self.tabs.addTab(self._scrollable(self.files), "Files")
        self.tabs.addTab(self._scrollable(self.log), "Log")
        layout = QVBoxLayout(self); layout.setContentsMargins(30, 24, 30, 30)
        layout.addWidget(title); layout.addLayout(monitor_row); layout.addWidget(self.compact_monitor); layout.addWidget(self.tabs, 1)

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(widget)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setMinimumSize(0, 0)
        return scroll

    def _toggle_monitor_details(self, checked: bool) -> None:
        self.monitor.setVisible(checked)
        self.details_button.setText("Details ▴" if checked else "Details ▾")

    def begin_run(self) -> None:
        self.loader = None
        self.log.clear_for_run()
        self.run_status_label.setText("Running pyLOCO backend…")
        self.run_progress.setRange(0, 0)
        self.run_output_dir.setText("Preparing results directory…")
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.compact_monitor.hide()
        self.monitor.show()
        self.details_button.setChecked(False)
        for widget in (self.run_progress, self.run_output_dir, self._progress_label, self._output_label):
            widget.setVisible(True)
        self.monitor.setMaximumHeight(16777215)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def append_log(self, message: str) -> None:
        self.log.append(message)
        if message.startswith("Results directory:"):
            self.run_output_dir.setText(message.split(":", 1)[1].strip())

    def complete_run(self, result) -> None:
        self.run_progress.setRange(0, 1); self.run_progress.setValue(1)
        self.run_status_label.setText(f"Completed in {result.elapsed_seconds:.1f} s")
        self.run_elapsed_label.setText(f"Elapsed: {result.elapsed_seconds:.1f} s")
        self.run_output_dir.setText(result.results_dir); self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.loader = ResultsLoader(result.results_dir, runtime=result.elapsed_seconds)
        for view in (self.overview, self.orm, self.optics, self.parameters, self.svd, self.files):
            view.set_loader(self.loader)
        log_path = Path(result.results_dir) / "backend.log"
        if log_path.exists():
            try: self.log.set_log(log_path.read_text(encoding="utf-8"))
            except OSError: pass
        self.tabs.setCurrentIndex(0)
        for widget in (self.run_progress, self.run_output_dir, self._progress_label, self._output_label):
            widget.setVisible(False)
        self.compact_status.setText(f"✓ Completed in {result.elapsed_seconds:.1f} s — Results saved")
        self.compact_monitor.show()
        self.monitor.hide()
        self.details_button.setChecked(False)
        self.monitor.setMaximumHeight(16777215)

    def fail_run(self) -> None:
        self.run_progress.setRange(0, 1); self.run_progress.setValue(0)
        self.run_status_label.setText("Failed"); self.cancel_button.setEnabled(False)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def apply_theme(self) -> None:
        self.overview.chi_plot.apply_theme()
        self.orm.plot.apply_theme()
        self.optics.plot.apply_theme()
        self.parameters.plot.apply_theme()
        self.svd.plot.apply_theme()

    def set_mode(self, mode: str) -> None:
        """Keep the scientific essentials prominent in Basic mode."""
        advanced = mode == "Advanced"
        for label in ("Jacobian/SVD", "Files", "Log"):
            index = next((i for i in range(self.tabs.count()) if self.tabs.tabText(i) == label), -1)
            if index >= 0:
                self.tabs.setTabVisible(index, advanced)
