"""Main window for the pyLOCO GUI.

The GUI manages project state, lattice metadata, imported measurement files,
backend-compatible LOCO configuration, and responsive execution monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup, QDoubleValidator, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .backend import LocoRunError, LocoRunRequest, run_loco_request
from .models.project import ImportedDataset, LatticeSelection, LocoConfiguration, ProjectMetadata
from .widgets.project_explorer import ProjectExplorer

APP_STYLESHEET = """
QMainWindow { background: #f4f7fb; }
QMenuBar, QToolBar#mainToolbar, QStatusBar { background: #ffffff; }
QToolBar#mainToolbar { border: 0; border-bottom: 1px solid #d9e2ef; spacing: 8px; padding: 8px 12px; }
QToolButton, QPushButton { border: 1px solid #c8d8ec; border-radius: 6px; color: #1f3b57; font-weight: 600; padding: 6px 10px; background: #ffffff; }
QToolButton:hover, QPushButton:hover { background: #eef5ff; }
QToolButton:checked { background: #2463a6; color: #ffffff; }
QTabWidget::pane { background: #f4f7fb; border: 0; padding-top: 10px; }
QTabBar::tab { background: #e8eef7; border: 1px solid #d3deec; border-bottom: 0; border-top-left-radius: 8px; border-top-right-radius: 8px; color: #2f4b66; margin-right: 4px; padding: 10px 18px; }
QTabBar::tab:selected { background: #ffffff; color: #12365f; font-weight: 700; }
QDockWidget::title { background: #183b66; color: #ffffff; font-weight: 700; padding: 8px 10px; }
QTreeWidget#projectExplorerTree, QGroupBox { background: #ffffff; border: 1px solid #d8e1ee; border-radius: 8px; margin: 8px; padding: 10px; }
QGroupBox::title { color: #1b426d; font-weight: 700; subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QLabel#statusPill { background: #e8f0fe; border: 1px solid #cbdcf4; border-radius: 9px; color: #174a7c; font-weight: 700; padding: 3px 10px; }
QLabel#pageTitle { color: #14395f; font-size: 24px; font-weight: 700; }
QLabel#validationOk { color: #1f7a3f; font-weight: 700; }
QLabel#validationMissing { color: #9a4b00; font-weight: 700; }
QWidget#placeholderPageCard { background: #ffffff; border: 1px solid #d9e2ef; border-radius: 14px; }
QLabel#placeholderTitle { color: #14395f; font-size: 26px; font-weight: 700; }
QLabel#placeholderDescription, QLabel#dashboardCardText { color: #52677d; font-size: 14px; }
QWidget#dashboardCard { background: #f8fbff; border: 1px solid #d9e6f5; border-radius: 10px; }
QLabel#dashboardCardTitle { color: #1b426d; font-size: 15px; font-weight: 700; }
"""


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that accepts and displays scientific notation."""

    def __init__(self) -> None:
        super().__init__()
        self._validator = QDoubleValidator(self)
        self._validator.setNotation(QDoubleValidator.ScientificNotation)

    def validate(self, text: str, pos: int):  # type: ignore[override]
        suffix = self.suffix()
        candidate = text.strip()
        if suffix and candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
        if candidate in {"", "+", "-", ".", "+.", "-."}:
            return QDoubleValidator.Intermediate, text, pos
        state, _, _ = self._validator.validate(candidate, pos)
        try:
            value = float(candidate)
        except ValueError:
            return state, text, pos
        if self.minimum() <= value <= self.maximum():
            return state, text, pos
        return QDoubleValidator.Invalid, text, pos

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        suffix = self.suffix()
        candidate = text.strip()
        if suffix and candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
        return float(candidate)

    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        return f"{value:.{self.decimals()}g}"


class LocoRunWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, request: LocoRunRequest) -> None:
        super().__init__()
        self.request = request
        self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            result = run_loco_request(
                self.request,
                log_callback=self.log.emit,
                cancel_callback=lambda: self.cancel_requested,
            )
        except Exception as exc:
            import traceback

            self.failed.emit(LocoRunError(str(exc), traceback.format_exc()))
        else:
            self.finished.emit(result)


class MainWindow(QMainWindow):
    """Top-level pyLOCO GUI window for project management and data import."""

    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectMetadata()
        self.setObjectName("pyLocoMainWindow")
        self.setWindowTitle("pyLOCO GUI")
        self.resize(1320, 860)
        self.setStyleSheet(APP_STYLESHEET)

        self._mode_label = QLabel("Basic mode")
        self._mode_label.setObjectName("statusPill")
        self._project_label = QLabel()
        self._workflow_label = QLabel("Workflow: Project")
        self._backend_label = QLabel("Backend: unchanged")
        self._validation_label = QLabel()
        self._project_explorer = ProjectExplorer()
        self._run_thread: QThread | None = None
        self._run_worker: LocoRunWorker | None = None
        self._run_started_at = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._workspace = self._create_workspace()

        self.setCentralWidget(self._workspace)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._project_explorer)
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._workspace.currentChanged.connect(self._on_tab_changed)
        self._refresh_ui("Ready — create or open a project")

    def _create_workspace(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        self.dashboard_name = QLineEdit(self.project.name)
        self.dashboard_name.editingFinished.connect(self._rename_project)
        self.dashboard_summary = QLabel()
        self.recent_list = QListWidget()
        tabs.addTab(self._project_page(), "Project")
        tabs.addTab(self._machine_page(), "Machine")
        tabs.addTab(self._measurements_page(), "Measurements")
        tabs.addTab(self._fit_page(), "Fit")
        self.results_page = self._results_page()
        tabs.addTab(self.results_page, "Results")
        return tabs


    def _results_page(self) -> QWidget:
        page = self._page("Results Workspace")
        self.run_status_label = QLabel("No LOCO run has been started.")
        self.run_elapsed_label = QLabel("Elapsed: 0.0 s")
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(0)
        self.run_output_dir = QLabel("—")
        self.run_log = QTextEdit()
        self.run_log.setReadOnly(True)
        self.cancel_loco_button = QPushButton("Cancel Run")
        self.cancel_loco_button.setEnabled(False)
        self.cancel_loco_button.clicked.connect(self.cancel_loco_run)
        form = QFormLayout()
        form.addRow("Status", self.run_status_label)
        form.addRow("Elapsed", self.run_elapsed_label)
        form.addRow("Progress", self.run_progress)
        form.addRow("Results directory", self.run_output_dir)
        group = QGroupBox("Backend Run Monitor")
        group.setLayout(form)
        page.layout().addWidget(group)
        page.layout().addWidget(self.cancel_loco_button)
        page.layout().addWidget(self.run_log, 1)
        return page

    def _project_page(self) -> QWidget:
        page = self._page("Project Dashboard")
        form = QFormLayout()
        form.addRow("Project name", self.dashboard_name)
        for text, slot in (
            ("New Project", self.new_project),
            ("Open Project…", self.open_project),
            ("Save Project…", self.save_project_as),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            form.addRow(button)
        group = QGroupBox("Project state")
        group.setLayout(form)
        page.layout().addWidget(group)
        page.layout().addWidget(self.dashboard_summary)
        page.layout().addWidget(QLabel("Recent projects"))
        page.layout().addWidget(self.recent_list, 1)
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.text()))
        )
        return page

    def _machine_page(self) -> QWidget:
        page = self._page("Machine Lattice")
        self.lattice_path = QLabel("No lattice selected")
        self.lattice_type = QLabel("—")
        self.lattice_elements = QLabel("Unknown")
        choose = QPushButton("Select lattice/model file…")
        choose.clicked.connect(self.select_lattice)
        form = QFormLayout()
        form.addRow(choose)
        form.addRow("Path", self.lattice_path)
        form.addRow("Type", self.lattice_type)
        form.addRow("Elements", self.lattice_elements)
        group = QGroupBox("Lattice selection and metadata")
        group.setLayout(form)
        page.layout().addWidget(group)
        page.layout().addStretch(1)
        return page

    def _measurements_page(self) -> QWidget:
        page = self._page("Measurement Import")
        self.measurement_role = QComboBox()
        self.measurement_role.addItems(
            ["orm", "dispersion", "bpm_noise", "mask", "other"]
        )
        import_button = QPushButton("Import HDF5, MAT, NumPy…")
        import_button.clicked.connect(self.import_measurement)
        self.measurement_list = QListWidget()
        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset role"))
        row.addWidget(self.measurement_role)
        row.addWidget(import_button)
        group = QGroupBox("File import")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        layout.addWidget(self.measurement_list)
        page.layout().addWidget(group)
        return page


    def _fit_page(self) -> QWidget:
        page = self._page("LOCO Configuration")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        self.rm_calculator = QComboBox()
        self.rm_calculator.addItem("Linear ORM (analytic)", "Linear")
        self.rm_calculator.addItem("Numerical ORM (tracking)", "Tracking")
        self.rm_calculator.setToolTip("Choose the backend ORM implementation: linear analytic calculation or numerical tracking.")
        self.rm_dispersion = QCheckBox("Include dispersion/RF response column")
        self.rm_dispersion.setToolTip("Append the response to an RF frequency shift to the ORM.")
        self.rm_coupling = QCheckBox("Include coupling ORM terms")
        self.rm_coupling.setToolTip("Include cross-plane response blocks in the ORM.")
        self.rm_bidirectional = QCheckBox("Bidirectional (+/- delta kick)")
        self.rm_bidirectional.setToolTip("Compute the ORM using positive and negative perturbations (central difference) instead of a single perturbation. This generally improves numerical accuracy.")
        self.rm_vectorized = QCheckBox("Use vectorized response calculation")
        self.rm_vectorized.setToolTip("Use the backend vectorized ORM path where available.")
        self.rm_dkick_h = self._double_spin(0.0, 1e-1, 1e-5, 9, " rad")
        self.rm_dkick_v = self._double_spin(0.0, 1e-1, 1e-5, 9, " rad")
        self.rm_rf_step = self._double_spin(-1e9, 1e9, -3000.0, 9, " Hz")
        self.rm_delta_coupling = self._double_spin(-1.0, 1.0, 1e-6, 9)
        self.rm_dkick_h.setToolTip("Horizontal corrector kick step in radians; scientific notation such as 1e-6 is accepted.")
        self.rm_dkick_v.setToolTip("Vertical corrector kick step in radians; scientific notation such as 5e-5 is accepted.")
        self.rm_rf_step.setToolTip("RF frequency step in Hz. Positive and negative shifts are supported and the sign is preserved.")
        self.rm_delta_coupling.setToolTip("Small dimensionless delta used to evaluate corrector coupling terms; scientific notation is accepted.")
        rm_form = QFormLayout()
        for label, widget in (
            ("ORM calculation method", self.rm_calculator),
            ("Horizontal kick step", self.rm_dkick_h),
            ("Vertical kick step", self.rm_dkick_v),
            ("RF frequency step", self.rm_rf_step),
            ("Coupling delta (dimensionless)", self.rm_delta_coupling),
        ):
            rm_form.addRow(label, widget)
        for widget in (self.rm_dispersion, self.rm_coupling, self.rm_bidirectional, self.rm_vectorized):
            rm_form.addRow(widget)
        rm_group = QGroupBox("Response Matrix")
        rm_group.setLayout(rm_form)
        layout.addWidget(rm_group)

        self.solver_algorithm = QComboBox()
        self.solver_algorithm.addItems(["lm", "gn"])
        self.solver_n_iter = self._spin(1, 100, 1)
        self.solver_lm_iter = self._spin(0, 100, 10)
        self.solver_lambda = self._double_spin(0.0, 1e9, 1e-3, 9)
        self.solver_max_lambda = self._double_spin(0.0, 1e9, 15.0, 3)
        self.solver_scaled = QCheckBox("Solve with scaled variables")
        solver_form = QFormLayout()
        for label, widget in (
            ("Algorithm", self.solver_algorithm),
            ("Outer iterations", self.solver_n_iter),
            ("LM inner iterations", self.solver_lm_iter),
            ("Starting lambda", self.solver_lambda),
            ("Maximum lambda", self.solver_max_lambda),
        ):
            solver_form.addRow(label, widget)
        solver_form.addRow(self.solver_scaled)
        solver_group = QGroupBox("Solver")
        solver_group.setLayout(solver_form)
        layout.addWidget(solver_group)

        self.svd_method = QComboBox()
        self.svd_method.addItems(["threshold", "rank", "auto"])
        self.svd_threshold = self._double_spin(0.0, 1.0, 1e-7, 10)
        self.svd_rank = self._spin(0, 100000, 500)
        self.svd_plot = QCheckBox("Show SVD plot")
        svd_form = QFormLayout()
        svd_form.addRow("Selection method", self.svd_method)
        svd_form.addRow("Threshold", self.svd_threshold)
        svd_form.addRow("Rank/cut", self.svd_rank)
        svd_form.addRow(self.svd_plot)
        svd_group = QGroupBox("SVD")
        svd_group.setLayout(svd_form)
        layout.addWidget(svd_group)

        self.outlier_enabled = QCheckBox("Reject outliers")
        self.outlier_sigma = self._double_spin(0.0, 1e6, 10.0, 3)
        self.norm_enabled = QCheckBox("Apply normalization")
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["component", "global", "none"])
        self.auto_delta = QCheckBox("Auto-correct delta")
        rej_form = QFormLayout()
        rej_form.addRow(self.outlier_enabled)
        rej_form.addRow("Sigma cut", self.outlier_sigma)
        rej_form.addRow(self.norm_enabled)
        rej_form.addRow("Normalization mode", self.norm_mode)
        rej_form.addRow(self.auto_delta)
        rej_group = QGroupBox("Iterations and Outlier Rejection")
        rej_group.setLayout(rej_form)
        layout.addWidget(rej_group)

        self.constraint_enabled = QCheckBox("Enable constraints")
        self.constraint_quad_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_skew_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_quad_weights = QLineEdit()
        self.constraint_skew_weights = QLineEdit()
        constraint_form = QFormLayout()
        constraint_form.addRow(self.constraint_enabled)
        constraint_form.addRow("Quadrupole sigma", self.constraint_quad_sigma)
        constraint_form.addRow("Skew sigma", self.constraint_skew_sigma)
        constraint_form.addRow("Quadrupole weights", self.constraint_quad_weights)
        constraint_form.addRow("Skew weights", self.constraint_skew_weights)
        constraint_group = QGroupBox("Constraints")
        constraint_group.setLayout(constraint_form)
        layout.addWidget(constraint_group)

        self.parameter_checks = {}
        param_group = QGroupBox("Parameter Selection")
        param_layout = QVBoxLayout(param_group)
        for key, label in (
            ("quads", "Quadrupoles"), ("skew_quads", "Skew quadrupoles"), ("quads_tilt", "Quadrupole tilts"),
            ("hbpm_gain", "Horizontal BPM gains"), ("vbpm_gain", "Vertical BPM gains"),
            ("hbpm_coupling", "Horizontal BPM coupling"), ("vbpm_coupling", "Vertical BPM coupling"),
            ("hcor_cal", "Horizontal corrector calibration"), ("vcor_cal", "Vertical corrector calibration"),
            ("hcor_coupling", "Horizontal corrector coupling"), ("vcor_coupling", "Vertical corrector coupling"),
            ("HCMEnergyShift", "Horizontal corrector energy shifts"), ("VCMEnergyShift", "Vertical corrector energy shifts"),
            ("delta_rf", "RF frequency shift"),
        ):
            check = QCheckBox(label)
            self.parameter_checks[key] = check
            param_layout.addWidget(check)
        self.params_individuals = QCheckBox("Fit individual elements instead of family groups")
        param_layout.addWidget(self.params_individuals)
        layout.addWidget(param_group)

        button_row = QHBoxLayout()
        import_button = QPushButton("Import configuration…")
        import_button.clicked.connect(self.import_loco_configuration)
        export_button = QPushButton("Export configuration…")
        export_button.clicked.connect(self.export_loco_configuration)
        button_row.addWidget(import_button)
        button_row.addWidget(export_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.fit_summary = QTextEdit()
        self.fit_summary.setReadOnly(True)
        summary_group = QGroupBox("Live Backend-Compatible Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.fit_summary)
        layout.addWidget(summary_group)
        layout.addStretch(1)
        scroll.setWidget(container)
        page.layout().addWidget(scroll, 1)
        self._load_config_to_widgets()
        self._connect_fit_controls()
        return page

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spin(
        self, minimum: float, maximum: float, value: float, decimals: int, suffix: str = ""
    ) -> QDoubleSpinBox:
        spin = ScientificDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setValue(value)
        spin.setSingleStep(abs(value) or 1.0)
        return spin

    def _page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 32, 36, 36)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        return page

    def _create_actions(self) -> None:
        self.new_project_action = QAction("New", self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action = QAction("Open…", self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action = QAction("Save", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action = QAction("Save As…", self)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.validate_project_action = QAction("Validate", self)
        self.validate_project_action.triggered.connect(self.validate_project)
        self.run_loco_action = QAction("Run LOCO", self)
        self.run_loco_action.triggered.connect(self.run_loco)
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        self.basic_mode_action = QAction("Basic", self, checkable=True, checked=True)
        self.advanced_mode_action = QAction("Advanced", self, checkable=True)
        self.mode_action_group = QActionGroup(self, exclusive=True)
        self.mode_action_group.addAction(self.basic_mode_action)
        self.mode_action_group.addAction(self.advanced_mode_action)
        self.mode_action_group.triggered.connect(self._on_mode_changed)
        self.about_action = QAction("About pyLOCO GUI", self)
        self.about_action.triggered.connect(self._show_about_dialog)

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.save_project_as_action,
        ):
            file_menu.addAction(action)
        self.recent_menu = file_menu.addMenu("Recent Projects")
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.validate_project_action)
        project_menu.addAction(self.run_loco_action)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._project_explorer.toggleViewAction())
        mode_menu = view_menu.addMenu("Workflow Mode")
        mode_menu.addAction(self.basic_mode_action)
        mode_menu.addAction(self.advanced_mode_action)
        self.menuBar().addMenu("&Help").addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.validate_project_action,
            self.run_loco_action,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.basic_mode_action)
        toolbar.addAction(self.advanced_mode_action)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _create_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._project_label, 1)
        status_bar.addWidget(self._workflow_label, 1)
        status_bar.addWidget(self._validation_label, 1)
        status_bar.addPermanentWidget(self._backend_label)
        status_bar.addPermanentWidget(self._mode_label)
        self.setStatusBar(status_bar)


    def _connect_fit_controls(self) -> None:
        widgets = [
            self.rm_calculator, self.rm_dispersion, self.rm_coupling, self.rm_bidirectional,
            self.rm_vectorized, self.rm_dkick_h, self.rm_dkick_v, self.rm_rf_step,
            self.rm_delta_coupling, self.solver_algorithm, self.solver_n_iter,
            self.solver_lm_iter, self.solver_lambda, self.solver_max_lambda,
            self.solver_scaled, self.svd_method, self.svd_threshold, self.svd_rank,
            self.svd_plot, self.outlier_enabled, self.outlier_sigma, self.norm_enabled,
            self.norm_mode, self.auto_delta, self.constraint_enabled,
            self.constraint_quad_sigma, self.constraint_skew_sigma,
            self.constraint_quad_weights, self.constraint_skew_weights,
            self.params_individuals,
        ] + list(self.parameter_checks.values())
        for widget in widgets:
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_fit_config_changed)

    def _set_calculator_value(self, calculator: str) -> None:
        backend_value = "Linear" if calculator == "Linear" else "Tracking"
        index = self.rm_calculator.findData(backend_value)
        if index >= 0:
            self.rm_calculator.setCurrentIndex(index)

    def _load_config_to_widgets(self) -> None:
        cfg = self.project.loco_config
        self._set_calculator_value(cfg.response_matrix.calculator)
        self.rm_dispersion.setChecked(cfg.response_matrix.includeDispersion)
        self.rm_coupling.setChecked(cfg.response_matrix.coupling_orm)
        self.rm_bidirectional.setChecked(cfg.response_matrix.bidirectional)
        self.rm_vectorized.setChecked(cfg.response_matrix.NewVectorizedMethod)
        self.rm_dkick_h.setValue(cfg.response_matrix.dkick_h)
        self.rm_dkick_v.setValue(cfg.response_matrix.dkick_v)
        self.rm_rf_step.setValue(cfg.response_matrix.rfStep)
        self.rm_delta_coupling.setValue(cfg.response_matrix.delta_coupling)
        self.solver_algorithm.setCurrentText(cfg.solver.algorithm)
        self.solver_n_iter.setValue(cfg.solver.nIter)
        self.solver_lm_iter.setValue(cfg.solver.nLMIter)
        self.solver_lambda.setValue(cfg.solver.Starting_Lambda)
        self.solver_max_lambda.setValue(cfg.solver.max_lm_lambda)
        self.solver_scaled.setChecked(cfg.solver.scaled)
        self.svd_method.setCurrentText(cfg.svd.svd_selection_method)
        self.svd_threshold.setValue(cfg.svd.svd_threshold)
        self.svd_rank.setValue(cfg.svd.cut_)
        self.svd_plot.setChecked(cfg.svd.show_svd_plot)
        self.outlier_enabled.setChecked(cfg.rejection.outlier_rejection)
        self.outlier_sigma.setValue(cfg.rejection.sigma_outlier)
        self.norm_enabled.setChecked(cfg.rejection.apply_normalization)
        self.norm_mode.setCurrentText(cfg.rejection.normalization_mode)
        self.auto_delta.setChecked(cfg.rejection.auto_correct_delta)
        self.constraint_enabled.setChecked(cfg.constraints.enable)
        self.constraint_quad_sigma.setValue(cfg.constraints.quad_sigma)
        self.constraint_skew_sigma.setValue(cfg.constraints.skew_sigma)
        self.constraint_quad_weights.setText(cfg.constraints.quad_weights)
        self.constraint_skew_weights.setText(cfg.constraints.skew_weights)
        for name, check in self.parameter_checks.items():
            check.setChecked(bool(getattr(cfg.parameters, name)))
        self.params_individuals.setChecked(cfg.parameters.individuals)
        self._update_fit_summary()

    def _collect_loco_configuration(self) -> LocoConfiguration:
        cfg = LocoConfiguration()
        cfg.response_matrix.calculator = self.rm_calculator.currentData() or self.rm_calculator.currentText()
        cfg.response_matrix.includeDispersion = self.rm_dispersion.isChecked()
        cfg.response_matrix.coupling_orm = self.rm_coupling.isChecked()
        cfg.response_matrix.bidirectional = self.rm_bidirectional.isChecked()
        cfg.response_matrix.NewVectorizedMethod = self.rm_vectorized.isChecked()
        cfg.response_matrix.dkick_h = self.rm_dkick_h.value()
        cfg.response_matrix.dkick_v = self.rm_dkick_v.value()
        cfg.response_matrix.rfStep = self.rm_rf_step.value()
        cfg.response_matrix.delta_coupling = self.rm_delta_coupling.value()
        cfg.solver.algorithm = self.solver_algorithm.currentText()
        cfg.solver.nIter = self.solver_n_iter.value()
        cfg.solver.nLMIter = self.solver_lm_iter.value()
        cfg.solver.Starting_Lambda = self.solver_lambda.value()
        cfg.solver.max_lm_lambda = self.solver_max_lambda.value()
        cfg.solver.scaled = self.solver_scaled.isChecked()
        cfg.svd.svd_selection_method = self.svd_method.currentText()
        cfg.svd.svd_threshold = self.svd_threshold.value()
        cfg.svd.cut_ = self.svd_rank.value()
        cfg.svd.show_svd_plot = self.svd_plot.isChecked()
        cfg.rejection.outlier_rejection = self.outlier_enabled.isChecked()
        cfg.rejection.sigma_outlier = self.outlier_sigma.value()
        cfg.rejection.apply_normalization = self.norm_enabled.isChecked()
        cfg.rejection.normalization_mode = self.norm_mode.currentText()
        cfg.rejection.auto_correct_delta = self.auto_delta.isChecked()
        cfg.constraints.enable = self.constraint_enabled.isChecked()
        cfg.constraints.quad_sigma = self.constraint_quad_sigma.value()
        cfg.constraints.skew_sigma = self.constraint_skew_sigma.value()
        cfg.constraints.quad_weights = self.constraint_quad_weights.text()
        cfg.constraints.skew_weights = self.constraint_skew_weights.text()
        for name, check in self.parameter_checks.items():
            setattr(cfg.parameters, name, check.isChecked())
        cfg.parameters.individuals = self.params_individuals.isChecked()
        return cfg

    @Slot()
    def _on_fit_config_changed(self) -> None:
        self.project.loco_config = self._collect_loco_configuration()
        self.project.modified = True
        self._update_fit_summary()
        self._refresh_ui("LOCO configuration updated")

    def _update_fit_summary(self) -> None:
        if not hasattr(self, "fit_summary"):
            return
        cfg = self.project.loco_config
        backend = json.dumps(cfg.to_backend_mapping(), indent=2)
        self.fit_summary.setPlainText("\n".join(cfg.summary_lines()) + "\n\nBackend constructor mapping:\n" + backend)

    @Slot()
    def import_loco_configuration(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Import LOCO configuration",
            "",
            "Configuration (*.json *.yaml *.yml);;JSON (*.json);;YAML (*.yaml *.yml)",
        )[0]
        if not filename:
            return
        try:
            self.project.loco_config = LocoConfiguration.load(filename)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self.project.modified = True
        self._load_config_to_widgets()
        self._refresh_ui(f"Imported LOCO configuration {Path(filename).name}")

    @Slot()
    def export_loco_configuration(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self,
            "Export LOCO configuration",
            f"{self.project.name}-loco-config.json",
            "JSON (*.json);;YAML (*.yaml *.yml)",
        )[0]
        if not filename:
            return
        self.project.loco_config = self._collect_loco_configuration()
        try:
            target = self.project.loco_config.save(filename)
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._refresh_ui(f"Exported LOCO configuration {target.name}")

    @Slot()
    def new_project(self) -> None:
        recent = self.project.recent_projects
        self.project = ProjectMetadata(recent_projects=recent)
        self.dashboard_name.setText(self.project.name)
        self._load_config_to_widgets()
        self._refresh_ui("New project created")

    @Slot()
    def open_project(self, path: Path | None = None) -> None:
        filename = (
            str(path)
            if path
            else QFileDialog.getOpenFileName(
                self,
                "Open pyLOCO project",
                "",
                "pyLOCO Project (*.pyloco.json);;JSON (*.json)",
            )[0]
        )
        if not filename:
            return
        self.project = ProjectMetadata.load(filename)
        self.dashboard_name.setText(self.project.name)
        self._load_config_to_widgets()
        self._refresh_ui(f"Opened {filename}")

    @Slot()
    def save_project(self) -> None:
        if not self.project.path:
            self.save_project_as()
            return
        self.project.save()
        self._refresh_ui(f"Saved {self.project.path}")

    @Slot()
    def save_project_as(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self,
            "Save pyLOCO project",
            self.project.path or f"{self.project.name}.pyloco.json",
            "pyLOCO Project (*.pyloco.json)",
        )[0]
        if filename:
            self.project.save(filename)
            self._refresh_ui(f"Saved {filename}")

    @Slot()
    def select_lattice(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Select lattice/model file",
            "",
            "Model files (*.mat *.h5 *.hdf5 *.npy *.npz *.json *.yaml *.yml);;All files (*)",
        )[0]
        if filename:
            path = Path(filename)
            self.project.lattice = LatticeSelection(
                path=str(path), file_type=path.suffix.lower().lstrip(".")
            )
            self.project.modified = True
            self._refresh_ui(f"Selected lattice {path.name}")

    @Slot()
    def import_measurement(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Import measurement file",
            "",
            "Measurement files (*.h5 *.hdf5 *.mat *.npy *.npz);;HDF5 (*.h5 *.hdf5);;MAT (*.mat);;NumPy (*.npy *.npz)",
        )[0]
        if filename:
            path = Path(filename)
            role = self.measurement_role.currentText()
            self.project.measurements[role] = ImportedDataset(
                role=role,
                path=str(path),
                file_type=path.suffix.lower().lstrip("."),
                size_bytes=path.stat().st_size,
            )
            self.project.modified = True
            self._refresh_ui(f"Imported {role}: {path.name}")

    def validate_project(self) -> None:
        messages = self.project.validation_messages()
        QMessageBox.information(
            self,
            "Project Validation",
            (
                "Project is complete; Run LOCO is enabled."
                if not messages
                else "Missing required inputs:\n\n" + "\n".join(messages)
            ),
        )
        self._refresh_ui("Validation complete")

    def _rename_project(self) -> None:
        self.project.name = (
            self.dashboard_name.text().strip() or "Untitled LOCO Project"
        )
        self.project.modified = True
        self._refresh_ui("Project renamed")

    @Slot(QAction)
    def _on_mode_changed(self, action: QAction) -> None:
        self.project.mode = action.text()
        self.project.modified = True
        self._mode_label.setText(f"{self.project.mode} mode")
        self._refresh_ui(f"{self.project.mode} mode selected")

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        self._workflow_label.setText(f"Workflow: {self._workspace.tabText(index)}")

    def _refresh_ui(self, message: str) -> None:
        suffix = " *" if self.project.modified else ""
        self._project_label.setText(f"Project: {self.project.name}{suffix}")
        self.setWindowTitle(f"pyLOCO GUI — {self.project.name}{suffix}")
        self._validation_label.setText(
            "Validation: complete"
            if self.project.is_complete
            else "Validation: incomplete"
        )
        self._validation_label.setObjectName(
            "validationOk" if self.project.is_complete else "validationMissing"
        )
        self.run_loco_action.setEnabled(self.project.is_complete)
        self.lattice_path.setText(self.project.lattice.path or "No lattice selected")
        self.lattice_type.setText(self.project.lattice.file_type or "—")
        self.lattice_elements.setText(
            str(self.project.lattice.element_count)
            if self.project.lattice.element_count
            else "Unknown"
        )
        self.measurement_list.clear()
        for role, dataset in sorted(self.project.measurements.items()):
            self.measurement_list.addItem(
                f"{role}: {dataset.name} ({dataset.file_type}, {dataset.size_bytes} bytes)"
            )
        self.recent_list.clear()
        self.recent_list.addItems(self.project.recent_projects)
        self.recent_menu.clear()
        for recent in self.project.recent_projects:
            action = self.recent_menu.addAction(recent)
            action.triggered.connect(
                lambda checked=False, value=recent: self.open_project(Path(value))
            )
        missing = self.project.validation_messages()
        self.dashboard_summary.setText(
            "Project complete. Run LOCO is enabled."
            if not missing
            else "Missing inputs:\n" + "\n".join(f"• {m}" for m in missing)
        )
        self._project_explorer.update_project(self.project)
        self.statusBar().showMessage(message, 4000)



    @Slot()
    def run_loco(self) -> None:
        messages = self.project.validation_messages()
        if messages:
            QMessageBox.warning(self, "Cannot run LOCO", "Missing required inputs:\n\n" + "\n".join(messages))
            return
        if self._run_thread is not None:
            QMessageBox.information(self, "LOCO already running", "A LOCO run is already in progress.")
            return
        self.project.loco_config = self._collect_loco_configuration()
        request = LocoRunRequest.from_project(self.project)
        self._run_started_at = __import__("time").monotonic()
        self.run_log.clear()
        self.run_status_label.setText("Running pyLOCO backend...")
        self.run_progress.setRange(0, 0)
        self.run_output_dir.setText("Preparing results directory...")
        self.cancel_loco_button.setEnabled(True)
        self.run_loco_action.setEnabled(False)
        self._workspace.setCurrentIndex(self._workspace.indexOf(self.results_page))
        self._run_thread = QThread(self)
        self._run_worker = LocoRunWorker(request)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.log.connect(self._append_run_log)
        self._run_worker.finished.connect(self._on_loco_finished)
        self._run_worker.failed.connect(self._on_loco_failed)
        self._run_worker.finished.connect(self._cleanup_run_thread)
        self._run_worker.failed.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_worker.deleteLater)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()
        self._elapsed_timer.start(500)
        self._refresh_ui("LOCO run started")

    @Slot()
    def cancel_loco_run(self) -> None:
        if self._run_worker is not None:
            self._run_worker.cancel_requested = True
            self.cancel_loco_button.setEnabled(False)
            self._append_run_log("Cancellation requested. The current backend step will finish before stopping if cancellation is feasible.")

    @Slot(str)
    def _append_run_log(self, message: str) -> None:
        self.run_log.append(message)
        if message.startswith("Results directory:"):
            self.run_output_dir.setText(message.split(":", 1)[1].strip())

    @Slot(object)
    def _on_loco_finished(self, result) -> None:
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(1)
        self.run_status_label.setText(f"Completed in {result.elapsed_seconds:.1f} s")
        self.run_output_dir.setText(result.results_dir)
        self._append_run_log("Saved outputs:\n" + "\n".join(result.output_files))
        QMessageBox.information(self, "LOCO complete", f"LOCO completed successfully.\n\nResults: {result.results_dir}")

    @Slot(object)
    def _on_loco_failed(self, error: LocoRunError) -> None:
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(0)
        self.run_status_label.setText("Failed")
        self._append_run_log(error.traceback)
        QMessageBox.critical(self, "LOCO failed", f"The backend reported an error:\n\n{error.message}")

    @Slot()
    def _cleanup_run_thread(self) -> None:
        self._elapsed_timer.stop()
        self.cancel_loco_button.setEnabled(False)
        if self._run_thread is not None:
            self._run_thread.quit()
            self._run_thread.wait()
        self._run_thread = None
        self._run_worker = None
        self.run_loco_action.setEnabled(self.project.is_complete)
        self._refresh_ui("LOCO run finished")

    @Slot()
    def _update_elapsed_time(self) -> None:
        if self._run_started_at:
            elapsed = __import__("time").monotonic() - self._run_started_at
            self.run_elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About pyLOCO GUI",
            "pyLOCO GUI\n\nMilestone 2 project management and data import shell. Numerical backend code is unchanged.",
        )
