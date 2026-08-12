"""Main window for the pyLOCO GUI.

The GUI manages project state, lattice metadata, imported measurement files,
backend-compatible LOCO configuration, and responsive execution monitoring.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup, QDoubleValidator, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QRadioButton,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from .backend import LocoRunError, LocoRunRequest, run_loco_request, _load_bad_bpm_positions
from .models.project import (
    ImportedDataset, LatticeSelection, LocoConfiguration, ProjectMetadata,
    load_example_project_data, resolve_element_name_file, resolve_example_machine_elements,
)
from .widgets.project_explorer import ProjectExplorer
from .widgets.orm_comparison import OrmComparisonWindow
from .themes import THEMES, apply_application_theme, configure_item_view, theme_for_key
from .results.results_workspace import ResultsWorkspace

APP_STYLESHEET = """
* { font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; }
QMainWindow, QDialog { background: #1E1E2E; color: #DDE3F0; }
QMenuBar, QMenu, QToolBar#mainToolbar, QStatusBar { background: #25283A; color: #E7EAF3; border: 0; }
QMenuBar::item:selected, QMenu::item:selected { background: #3B315A; color: #FFFFFF; }
QMenu { border: 1px solid #3C4058; padding: 6px; }
QToolBar#mainToolbar { border-bottom: 1px solid #3C4058; spacing: 10px; padding: 8px 12px; }
QToolButton, QPushButton { background: #2F3347; border: 1px solid #4A4F68; border-radius: 8px; color: #F4F6FB; font-weight: 600; padding: 7px 12px; }
QToolButton:hover, QPushButton:hover { background: #3B315A; border-color: #8A63D2; }
QToolButton:pressed, QPushButton:pressed, QToolButton:checked { background: #8A63D2; border-color: #A78BFA; color: #FFFFFF; }
QPushButton:disabled, QToolButton:disabled { background: #25283A; color: #737993; border-color: #34384D; }
QTabWidget::pane { background: #1E1E2E; border: 1px solid #34384D; border-radius: 10px; padding-top: 8px; }
QTabBar::tab { background: #25283A; border: 1px solid #34384D; border-bottom: 0; border-top-left-radius: 8px; border-top-right-radius: 8px; color: #BFC7D8; margin-right: 4px; padding: 10px 18px; }
QTabBar::tab:selected { background: #2A2D3E; color: #FFFFFF; border-color: #8A63D2; font-weight: 700; }
QDockWidget::title { background: #2A2D3E; color: #FFFFFF; font-weight: 700; padding: 8px 10px; border-bottom: 1px solid #8A63D2; }
QTreeWidget#projectExplorerTree, QTreeView, QTableView, QListWidget, QTextEdit { background: #202334; alternate-background-color: #25283A; border: 1px solid #3C4058; border-radius: 8px; color: #DDE3F0; selection-background-color: #5E45A0; selection-color: #FFFFFF; }
QHeaderView::section { background: #2A2D3E; color: #E7EAF3; border: 0; border-right: 1px solid #3C4058; padding: 6px; font-weight: 700; }
QGroupBox { background: #2A2D3E; border: 1px solid #3C4058; border-radius: 10px; color: #E7EAF3; margin: 10px; padding: 14px; }
QGroupBox::title { color: #C4B5FD; font-weight: 700; subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QLabel { color: #DDE3F0; }
QLabel#statusPill { background: #312A4A; border: 1px solid #8A63D2; border-radius: 10px; color: #EDE9FE; font-weight: 700; padding: 4px 11px; }
QLabel#pageTitle { color: #FFFFFF; font-size: 24px; font-weight: 750; }
QLabel#validationOk { color: #6EE7B7; font-weight: 700; }
QLabel#validationMissing { color: #FBBF24; font-weight: 700; }
QWidget#placeholderPageCard, QWidget#dashboardCard { background: #2A2D3E; border: 1px solid #3C4058; border-radius: 14px; }
QLabel#placeholderTitle { color: #FFFFFF; font-size: 26px; font-weight: 750; }
QLabel#placeholderDescription, QLabel#dashboardCardText { color: #BFC7D8; font-size: 14px; }
QLabel#dashboardCardTitle { color: #C4B5FD; font-size: 15px; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #202334; color: #F4F6FB; border: 1px solid #4A4F68; border-radius: 7px; padding: 6px 8px; selection-background-color: #8A63D2; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #A78BFA; }
QComboBox::drop-down { border: 0; width: 24px; }
QComboBox QAbstractItemView { background: #25283A; color: #F4F6FB; border: 1px solid #8A63D2; selection-background-color: #5E45A0; }
QCheckBox, QRadioButton { color: #DDE3F0; spacing: 8px; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid #6B7280; background: #202334; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #8A63D2; border-color: #C4B5FD; }
QProgressBar { background: #202334; color: #FFFFFF; border: 1px solid #3C4058; border-radius: 7px; text-align: center; }
QProgressBar::chunk { background: #8A63D2; border-radius: 7px; }
QScrollArea { background: #1E1E2E; border: 0; }
QScrollBar:vertical, QScrollBar:horizontal { background: #1E1E2E; width: 12px; height: 12px; }
QScrollBar::handle { background: #4A4F68; border-radius: 6px; }
QScrollBar::handle:hover { background: #8A63D2; }
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


ELEMENT_ROLES = {
    "bpm_ords": ("BPMs", "bpm"),
    "horizontal_corrector_ords": ("Horizontal correctors", "hcor"),
    "vertical_corrector_ords": ("Vertical correctors", "vcor"),
    "normal_quadrupole_ords": ("Normal quadrupoles", "quad"),
    "skew_quadrupole_ords": ("Skew quadrupoles", "skew"),
    "cavity_ords": ("RF cavities", "cavity"),
}


class ElementSelectionDialog(QDialog):
    """Select and preview lattice ordinals for one machine-element role."""

    def __init__(self, parent, role_key: str, current: list[int]) -> None:
        super().__init__(parent)
        self.role_key = role_key
        self.role_label, self.role_kind = ELEMENT_ROLES[role_key]
        self.setWindowTitle(f"Select {self.role_label}")
        self.resize(760, 560)
        self._lattice = parent._load_current_lattice()
        self.selected_ords = list(current)

        layout = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        self.auto_radio = QRadioButton("Automatic detection")
        self.type_radio = QRadioButton("AT element type")
        self.pattern_radio = QRadioButton("Family/name pattern")
        self.name_file_radio = QRadioButton("Load name file")
        self.file_radio = QRadioButton("Load index file")
        self.manual_radio = QRadioButton("Manual indices")
        self.manual_radio.setChecked(True)
        for button in (self.auto_radio, self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
            mode_row.addWidget(button)
        layout.addLayout(mode_row)

        form = QFormLayout()
        self.type_edit = QLineEdit(self._default_type_name())
        self.pattern_edit = QLineEdit(self._default_pattern())
        self.file_edit = QLineEdit()
        file_button = QPushButton("Browse…")
        file_button.clicked.connect(self._browse_index_file)
        file_row = QHBoxLayout(); file_row.addWidget(self.file_edit); file_row.addWidget(file_button)
        self.name_file_edit = QLineEdit()
        self.name_attribute = QComboBox()
        for label, value in (("Auto-detect attribute", "auto"), ("CommonName", "CommonName"), ("FamName", "FamName"), ("Name", "Name"), ("name", "name")):
            self.name_attribute.addItem(label, value)
        name_file_button = QPushButton("Browse…")
        name_file_button.clicked.connect(self._browse_name_file)
        name_file_row = QHBoxLayout(); name_file_row.addWidget(self.name_file_edit); name_file_row.addWidget(name_file_button)
        self.manual_edit = QPlainTextEdit(", ".join(str(i) for i in current))
        self.manual_edit.setPlaceholderText("Enter integer lattice ordinals separated by commas, spaces, or new lines.")
        form.addRow("AT class/type contains", self.type_edit)
        form.addRow("Family/name regex", self.pattern_edit)
        form.addRow("Element-name file", name_file_row)
        form.addRow("Name attribute", self.name_attribute)
        form.addRow("Index file", file_row)
        form.addRow("Manual lattice ordinals", self.manual_edit)
        layout.addLayout(form)

        preview_button = QPushButton("Preview selection")
        preview_button.clicked.connect(self._preview)
        layout.addWidget(preview_button)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Selection position", "Lattice ordinal", "Element name(s)", "Element class"])
        configure_item_view(self.table)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._set_preview(current)

    def _default_type_name(self) -> str:
        return {"bpm": "Monitor", "hcor": "Corrector", "vcor": "Corrector", "quad": "Quadrupole", "skew": "Quadrupole", "cavity": "RFCavity"}[self.role_kind]

    def _default_pattern(self) -> str:
        return {"bpm": "BPM|MON", "hcor": "HCM|HCOR|CH", "vcor": "VCM|VCOR|CV", "quad": "Q", "skew": "SQ|SKQ|SKEW", "cavity": "RFCAV|CAV|RF"}[self.role_kind]

    def _iter_elements(self):
        return list(enumerate(self._lattice or []))

    def _element_name(self, elem) -> str:
        values = []
        for attribute in ("CommonName", "FamName", "Name", "name"):
            value = getattr(elem, attribute, None)
            if value is not None and str(value) not in values:
                values.append(str(value))
        return " / ".join(values)

    def _element_class(self, elem) -> str:
        return type(elem).__name__

    def _browse_index_file(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load lattice-index array", "", "Index arrays (*.npy *.npz *.h5 *.hdf5 *.txt);;All files (*)")[0]
        if filename:
            self.file_edit.setText(filename); self.file_radio.setChecked(True); self._preview()

    def _browse_name_file(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load element-name list", "", "Text files (*.txt *.list *.dat);;All files (*)")[0]
        if filename:
            self.name_file_edit.setText(filename); self.name_file_radio.setChecked(True); self._preview()

    def _candidate_indices(self) -> list[int]:
        if self.auto_radio.isChecked():
            needle = self._default_type_name().lower()
            return [i for i, e in self._iter_elements() if needle in self._element_class(e).lower()]
        if self.type_radio.isChecked():
            needle = self.type_edit.text().strip().lower()
            return [i for i, e in self._iter_elements() if needle and needle in self._element_class(e).lower()]
        if self.pattern_radio.isChecked():
            pattern = re.compile(self.pattern_edit.text().strip(), re.I)
            return [i for i, e in self._iter_elements() if pattern.search(self._element_name(e))]
        if self.name_file_radio.isChecked():
            if not self._lattice:
                raise ValueError("Load a lattice before selecting elements from a name file.")
            return resolve_element_name_file(
                self._lattice,
                Path(self.name_file_edit.text()).expanduser(),
                self.name_attribute.currentData() or "auto",
            )
        if self.file_radio.isChecked():
            return self._load_index_file(Path(self.file_edit.text()).expanduser())
        return [int(x) for x in re.findall(r"[-+]?\d+", self.manual_edit.toPlainText())]

    def _load_index_file(self, path: Path) -> list[int]:
        import numpy as np
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".npz":
            data = np.load(path); arr = data[next(iter(data.files))]
        elif suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as h5:
                first = next(iter(h5.keys())); arr = h5[first][()]
        else:
            arr = np.loadtxt(path)
        return self._validate_array(arr)

    def _validate_array(self, arr) -> list[int]:
        import numpy as np
        a = np.asarray(arr)
        if a.ndim != 1:
            raise ValueError("Selection indices must be a one-dimensional array.")
        if not np.issubdtype(a.dtype, np.integer):
            if np.issubdtype(a.dtype, np.floating) and np.all(a == np.floor(a)):
                a = a.astype(int)
            else:
                raise ValueError("Selection indices must be integers.")
        values = [int(v) for v in a.tolist()]
        if len(set(values)) != len(values):
            raise ValueError("Selection indices must be unique.")
        if self._lattice and any(v < 0 or v >= len(self._lattice) for v in values):
            raise ValueError(f"Selection indices must be within lattice range [0, {len(self._lattice)-1}].")
        self._validate_compatible(values)
        return values

    def _validate_compatible(self, values: list[int]) -> None:
        if not self._lattice:
            return
        required = self._default_type_name().lower()
        for v in values:
            cls = self._element_class(self._lattice[v]).lower()
            if required not in cls:
                raise ValueError(f"Ordinal {v} is a {cls}, not compatible with {self.role_label}.")

    def _preview(self) -> bool:
        try:
            values = self._validate_array(self._candidate_indices())
        except Exception as exc:
            QMessageBox.warning(self, "Invalid selection", str(exc))
            return False
        self.selected_ords = values
        self._set_preview(values)
        return True

    def _set_preview(self, values: list[int]) -> None:
        self.table.setRowCount(len(values))
        for row, ordinal in enumerate(values):
            elem = self._lattice[ordinal] if self._lattice and 0 <= ordinal < len(self._lattice) else None
            cells = [row, ordinal, self._element_name(elem) if elem else "", self._element_class(elem) if elem else ""]
            for col, value in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))

    def _accept_if_valid(self) -> None:
        if self._preview():
            self.accept()


class MainWindow(QMainWindow):
    """Top-level pyLOCO GUI window for project management and data import."""

    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectMetadata()
        self.setObjectName("pyLocoMainWindow")
        self.setWindowTitle("pyLOCO GUI")
        self.resize(1320, 860)
        self._settings = QSettings()
        self.current_theme = theme_for_key(self._settings.value("appearance/theme", "dark"))
        apply_application_theme(QApplication.instance(), self.current_theme)

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
        self._last_loco_result = None
        self._orm_comparison_windows = []
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
        self._project_explorer.navigate_requested.connect(self._navigate_from_explorer)
        self._refresh_ui("Ready — create or open a project")

    def _create_workspace(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        self.dashboard_name = QLineEdit(self.project.name)
        self.dashboard_name.editingFinished.connect(self._rename_project)
        self.dashboard_summary = QLabel()
        self.recent_list = QListWidget()
        configure_item_view(self.recent_list)
        tabs.addTab(self._project_page(), "Project")
        tabs.addTab(self._machine_page(), "Machine")
        tabs.addTab(self._measurements_page(), "Measurements")
        tabs.addTab(self._fit_page(), "Fit")
        self.results_page = self._results_page()
        tabs.addTab(self.results_page, "Results")
        return tabs


    def _results_page(self) -> QWidget:
        self.results_workspace = ResultsWorkspace()
        self.run_status_label = self.results_workspace.run_status_label
        self.run_elapsed_label = self.results_workspace.run_elapsed_label
        self.run_progress = self.results_workspace.run_progress
        self.run_output_dir = self.results_workspace.run_output_dir
        self.run_log = self.results_workspace.log.text
        self.cancel_loco_button = self.results_workspace.cancel_button
        self.cancel_loco_button.clicked.connect(self.cancel_loco_run)
        return self.results_workspace

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

        self.element_count_labels = {}
        self.element_preview_tables = {}
        elements_group = QGroupBox("Machine Elements")
        elements_layout = QVBoxLayout(elements_group)
        elements_layout.addWidget(QLabel("Define BPMs, correctors, quadrupoles, and RF cavities after loading the lattice. Bad BPM positions are applied later as positions within the selected BPM list, not as lattice ordinals."))
        for key, (label, _kind) in ELEMENT_ROLES.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            count = QLabel("0 selected")
            self.element_count_labels[key] = count
            row.addWidget(count, 1)
            button = QPushButton("Edit/Select…")
            button.clicked.connect(lambda checked=False, role=key: self.edit_element_selection(role))
            row.addWidget(button)
            elements_layout.addLayout(row)
            table = QTableWidget(0, 4)
            table.setHorizontalHeaderLabels(["Selection position", "Lattice ordinal", "Element name(s)", "Element class"])
            configure_item_view(table)
            table.setMaximumHeight(160)
            self.element_preview_tables[key] = table
            elements_layout.addWidget(table)
        page.layout().addWidget(elements_group)
        page.layout().addStretch(1)
        return page

    def _measurements_page(self) -> QWidget:
        page = self._page("Measurement Import")
        self.measurement_role = QComboBox()
        self.measurement_role.addItems(
            ["orm", "dispersion", "bpm_noise", "bad_bpms", "other"]
        )
        import_button = QPushButton("Import HDF5, MAT, NumPy…")
        import_button.clicked.connect(self.import_measurement)
        self.measurement_list = QListWidget()
        configure_item_view(self.measurement_list)
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
        self.rm_bpm_ords = QLineEdit()
        self.rm_cm_ords = QLineEdit()
        self.rm_cav_ords = QLineEdit()
        self.rm_hcm_coupling = QLineEdit()
        self.rm_vcm_coupling = QLineEdit()
        self.rm_frequency = QLineEdit()
        self.rm_harm_number = QLineEdit()
        self.rm_rf_attr = QLineEdit("Frequency")
        self.rm_fixedpath = QCheckBox("fixedpathlength")
        self.rm_log_info = QCheckBox("log_info")
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
        for widget in (self.rm_dispersion, self.rm_coupling, self.rm_bidirectional, self.rm_vectorized, self.rm_fixedpath, self.rm_log_info):
            rm_form.addRow(widget)
        rm_group = QGroupBox("Response Matrix")
        rm_group.setLayout(rm_form)
        layout.addWidget(rm_group)

        self.solver_algorithm = QComboBox()
        self.solver_algorithm.addItem("Levenberg–Marquardt", "lm")
        self.solver_algorithm.addItem("Gauss–Newton", "gn")
        self.solver_n_iter = self._spin(1, 100, 1)
        self.solver_lm_iter = self._spin(0, 100, 10)
        self.solver_lambda = self._double_spin(0.0, 1e9, 1e-3, 9)
        self.solver_max_lambda = self._double_spin(0.0, 1e9, 15.0, 3)
        self.solver_scaled = QCheckBox("Scaled Levenberg–Marquardt")
        self.solver_scaled.setToolTip(
            "Enable the scaled-variable formulation of the Levenberg–Marquardt "
            "algorithm to improve conditioning when fit parameters have very different magnitudes."
        )
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
        for label, method in (
            ("Threshold", "threshold"),
            ("Rank", "rank"),
            ("User-defined cut", "user_input"),
            ("Interactive", "interactive"),
        ):
            self.svd_method.addItem(label, method)
        self.svd_threshold = self._double_spin(0.0, 1.0, 1e-7, 10)
        self.svd_rank = self._spin(0, 100000, 500)
        self.svd_parameter_label = QLabel("Threshold")
        self.svd_parameter_input = QWidget()
        svd_parameter_layout = QHBoxLayout(self.svd_parameter_input)
        svd_parameter_layout.setContentsMargins(0, 0, 0, 0)
        svd_parameter_layout.addWidget(self.svd_threshold)
        svd_parameter_layout.addWidget(self.svd_rank)
        self.svd_plot = QCheckBox("Show SVD plot")
        svd_form = QFormLayout()
        svd_form.addRow("Selection method", self.svd_method)
        svd_form.addRow(self.svd_parameter_label, self.svd_parameter_input)
        svd_form.addRow(self.svd_plot)
        svd_group = QGroupBox("SVD")
        svd_group.setLayout(svd_form)
        layout.addWidget(svd_group)

        self.outlier_enabled = QCheckBox("Reject outliers")
        self.outlier_sigma = self._double_spin(0.0, 1e6, 10.0, 3)
        self.norm_enabled = QCheckBox("Apply normalization")
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["component", "global", "none"])
        self.loco_include_dispersion = QCheckBox("includeDispersion")
        self.loco_hor_dispersion_weight = self._double_spin(0.0, 1e9, 1.0, 6)
        self.loco_ver_dispersion_weight = self._double_spin(0.0, 1e9, 1.0, 6)
        self.auto_delta = QCheckBox("auto_correct_delta")
        self.loco_fixedpath = QCheckBox("fixedpathlength")
        self.loco_individuals = QCheckBox("individuals")
        self.loco_remove_coupling = QCheckBox("remove_coupling_")
        self.loco_plot_fit_parameters = QCheckBox("plot_fit_parameters")
        rej_form = QFormLayout()
        rej_form.addRow(self.outlier_enabled)
        rej_form.addRow("Sigma cut", self.outlier_sigma)
        rej_form.addRow(self.norm_enabled)
        rej_form.addRow("Normalization mode", self.norm_mode)
        rej_form.addRow("hor_dispersion_weight", self.loco_hor_dispersion_weight)
        rej_form.addRow("ver_dispersion_weight", self.loco_ver_dispersion_weight)
        for widget in (self.loco_include_dispersion, self.auto_delta, self.loco_fixedpath, self.loco_individuals, self.loco_remove_coupling, self.loco_plot_fit_parameters):
            rej_form.addRow(widget)
        rej_group = QGroupBox("Iterations and Outlier Rejection")
        rej_group.setLayout(rej_form)
        layout.addWidget(rej_group)

        self.constraint_enabled = QCheckBox("Enable constraints")
        self.constraint_quad_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_skew_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_quad_weights = QLineEdit()
        self.constraint_skew_weights = QLineEdit()
        self.constraint_quad_mask = QLineEdit()
        self.constraint_skew_mask = QLineEdit()
        constraint_form = QFormLayout()
        constraint_form.addRow(self.constraint_enabled)
        constraint_form.addRow("Quadrupole sigma", self.constraint_quad_sigma)
        constraint_form.addRow("Skew sigma", self.constraint_skew_sigma)
        constraint_form.addRow("Quadrupole weights", self.constraint_quad_weights)
        constraint_form.addRow("Skew weights", self.constraint_skew_weights)
        constraint_form.addRow("Quadrupole mask", self.constraint_quad_mask)
        constraint_form.addRow("Skew mask", self.constraint_skew_mask)
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
        self.params_individuals = QCheckBox("individuals")
        param_layout.addWidget(self.params_individuals)
        self.cmstep_mode = QComboBox()
        self.cmstep_mode.addItem("Uniform", "uniform")
        self.cmstep_mode.addItem("Load from file", "file")
        self.params_init_policy = QLineEdit()
        self.params_init_policy.setPlaceholderText("Uses DEFAULT_INIT_POLICY unless overridden")
        self.params_cmstep_h = self._double_spin(-1e-1, 1e-1, 1e-5, 9, " rad")
        self.params_cmstep_v = self._double_spin(-1e-1, 1e-1, 1e-5, 9, " rad")
        self.params_cmstep_file = QLineEdit()
        self.params_cmstep_browse = QPushButton("Browse…")
        self.params_cmstep_browse.clicked.connect(self._browse_cmstep_file)
        self.params_cmstep_file_row = QWidget()
        cmstep_file_layout = QHBoxLayout(self.params_cmstep_file_row)
        cmstep_file_layout.setContentsMargins(0, 0, 0, 0)
        cmstep_file_layout.addWidget(self.params_cmstep_file, 1)
        cmstep_file_layout.addWidget(self.params_cmstep_browse)
        self.params_rfstep = self._double_spin(-1e9, 1e9, -3000.0, 9)
        self.params_init = QLineEdit()
        self.params_quads_attr = QLineEdit("PolynomB")
        self.params_quads_attr_index = self._spin(0, 100, 1)
        self.params_skew_attr = QLineEdit("PolynomA")
        self.params_skew_attr_index = self._spin(0, 100, 1)
        self.params_tilt_attr_r1 = QLineEdit("R1")
        self.params_tilt_attr_r2 = QLineEdit("R2")
        self.params_tilt_method = QLineEdit("set")
        cm_group = QGroupBox("Corrector Steps")
        cm_form = QFormLayout(cm_group)
        cm_form.addRow("Corrector step mode", self.cmstep_mode)
        cm_form.addRow("Horizontal step [rad]", self.params_cmstep_h)
        cm_form.addRow("Vertical step [rad]", self.params_cmstep_v)
        cm_form.addRow("CM-step .npz file", self.params_cmstep_file_row)
        param_layout.addWidget(cm_group)

        init_group = QGroupBox("Fit Initialization")
        param_form = QFormLayout(init_group)
        for label, widget in (("Initialization policy overrides", self.params_init_policy), ("Explicit initial values", self.params_init), ("Normal quadrupole attribute", self.params_quads_attr), ("Normal quadrupole attribute index", self.params_quads_attr_index), ("Skew quadrupole attribute", self.params_skew_attr), ("Skew quadrupole attribute index", self.params_skew_attr_index), ("Tilt R1 attribute", self.params_tilt_attr_r1), ("Tilt R2 attribute", self.params_tilt_attr_r2), ("Tilt update method", self.params_tilt_method)):
            param_form.addRow(label, widget)
        param_layout.addWidget(init_group)
        layout.addWidget(param_group)

        self.fixed_group = QGroupBox("RF and Momentum Compaction")
        fixed_form = QFormLayout(self.fixed_group)
        self.fixed_frequency = QLineEdit("499664399.4230182")
        self.fixed_harm_number = self._spin(1, 1000000, 3840)
        self.fixed_rfstep = self._double_spin(-1e9, 1e9, -3000.0, 9)
        self.fixed_dk = QLineEdit()
        self.fixed_delta_skew = self._double_spin(-1.0, 1.0, 1e-3, 9)
        self.fixed_delta_q_tilt = self._double_spin(-1.0, 1.0, 1e-6, 9)
        jac_group = QGroupBox("Jacobian Perturbations")
        jac_form = QFormLayout(jac_group)
        for label, widget in (("Normal quadrupole Jacobian step", self.fixed_dk), ("Skew quadrupole Jacobian step", self.fixed_delta_skew), ("Quadrupole tilt Jacobian step", self.fixed_delta_q_tilt)):
            jac_form.addRow(label, widget)
        layout.addWidget(jac_group)
        self.mcf_source = QComboBox()
        self.mcf_source.addItem("Automatic from lattice", "automatic")
        self.mcf_source.addItem("User-defined value", "user")
        self.mcf_user_value = QLineEdit()
        for label, widget in (("RF frequency [Hz]", self.fixed_frequency), ("Harmonic number", self.fixed_harm_number), ("RF frequency attribute", self.rm_rf_attr), ("rfStep", self.fixed_rfstep), ("Momentum compaction source", self.mcf_source), ("Momentum compaction factor", self.mcf_user_value)):
            fixed_form.addRow(label, widget)
        layout.addWidget(self.fixed_group)

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
        self.run_loco_action = QAction("▶ Run LOCO", self)
        self.run_loco_action.triggered.connect(self.run_loco)
        self.compare_orms_action = QAction("Compare ORMs", self)
        self.compare_orms_action.triggered.connect(self.compare_orms)
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        self.basic_mode_action = QAction("Basic", self, checkable=True, checked=True)
        self.advanced_mode_action = QAction("Advanced", self, checkable=True)
        self.mode_action_group = QActionGroup(self, exclusive=True)
        self.mode_action_group.addAction(self.basic_mode_action)
        self.mode_action_group.addAction(self.advanced_mode_action)
        self.mode_action_group.triggered.connect(self._on_mode_changed)
        self.theme_action_group = QActionGroup(self, exclusive=True)
        self.theme_actions = {}
        for key, theme in THEMES.items():
            action = QAction(theme.display_name, self, checkable=True)
            action.setData(key)
            action.setChecked(key == self.current_theme.key)
            self.theme_action_group.addAction(action)
            self.theme_actions[key] = action
        self.theme_action_group.triggered.connect(self._on_theme_changed)
        self.toggle_theme_action = QAction(self)
        self.toggle_theme_action.triggered.connect(self._toggle_theme)
        self._update_toggle_theme_action()
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
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.compare_orms_action)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._project_explorer.toggleViewAction())
        mode_menu = view_menu.addMenu("Workflow Mode")
        mode_menu.addAction(self.basic_mode_action)
        mode_menu.addAction(self.advanced_mode_action)
        theme_menu = view_menu.addMenu("Theme")
        for action in self.theme_actions.values():
            theme_menu.addAction(action)
        settings_menu = self.menuBar().addMenu("&Settings")
        appearance_menu = settings_menu.addMenu("Appearance")
        for action in self.theme_actions.values():
            appearance_menu.addAction(action)
        self.menuBar().addMenu("&Help").addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for action in (self.new_project_action, self.open_project_action, self.save_project_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.validate_project_action)
        toolbar.addAction(self.run_loco_action)
        toolbar.addAction(self.compare_orms_action)
        toolbar.addSeparator()
        toolbar.addAction(self.basic_mode_action)
        toolbar.addAction(self.advanced_mode_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_theme_action)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        run_button = toolbar.widgetForAction(self.run_loco_action)
        if run_button is not None:
            run_button.setObjectName("primaryToolbarAction")

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
            self.rm_delta_coupling, self.rm_fixedpath, self.rm_log_info, self.solver_algorithm, self.solver_n_iter,
            self.solver_lm_iter, self.solver_lambda, self.solver_max_lambda,
            self.solver_scaled, self.svd_method, self.svd_threshold, self.svd_rank,
            self.svd_plot, self.outlier_enabled, self.outlier_sigma, self.norm_enabled,
            self.norm_mode, self.loco_include_dispersion, self.loco_hor_dispersion_weight,
            self.loco_ver_dispersion_weight, self.auto_delta, self.loco_fixedpath, self.loco_individuals,
            self.loco_remove_coupling, self.loco_plot_fit_parameters, self.constraint_enabled,
            self.constraint_quad_sigma, self.constraint_skew_sigma,
            self.constraint_quad_weights, self.constraint_skew_weights,
            self.params_individuals, self.cmstep_mode, self.params_init_policy, self.params_cmstep_h, self.params_cmstep_v,
            self.params_cmstep_file, self.params_cmstep_browse, self.params_rfstep, self.params_init, self.params_quads_attr, self.params_quads_attr_index,
            self.params_skew_attr, self.params_skew_attr_index, self.params_tilt_attr_r1,
            self.params_tilt_attr_r2, self.params_tilt_method, self.fixed_frequency, self.fixed_harm_number,
            self.fixed_rfstep, self.fixed_dk, self.fixed_delta_skew, self.fixed_delta_q_tilt, self.mcf_source, self.mcf_user_value,
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
        self.solver_algorithm.currentIndexChanged.connect(self._update_solver_scaled_availability)
        self.svd_method.currentIndexChanged.connect(self._update_svd_input_availability)
        self.cmstep_mode.currentIndexChanged.connect(self._update_cmstep_input_availability)

    def _set_calculator_value(self, calculator: str) -> None:
        backend_value = "Linear" if calculator == "Linear" else "Tracking"
        index = self.rm_calculator.findData(backend_value)
        if index >= 0:
            self.rm_calculator.setCurrentIndex(index)

    def _set_solver_algorithm_value(self, algorithm: str) -> None:
        index = self.solver_algorithm.findData(algorithm)
        if index >= 0:
            self.solver_algorithm.setCurrentIndex(index)
        self._update_solver_scaled_availability()

    def _selected_solver_algorithm(self) -> str:
        return self.solver_algorithm.currentData() or self.solver_algorithm.currentText()

    def _update_solver_scaled_availability(self) -> None:
        self.solver_scaled.setEnabled(self._selected_solver_algorithm() == "lm")

    def _selected_svd_method(self) -> str:
        return self.svd_method.currentData() or self.svd_method.currentText()

    def _set_svd_method_value(self, method: str) -> None:
        index = self.svd_method.findData(method)
        if index >= 0:
            self.svd_method.setCurrentIndex(index)
        else:
            self.svd_method.setCurrentText(method)
        self._update_svd_input_availability()

    def _update_svd_input_availability(self) -> None:
        method = self._selected_svd_method()
        parameter_labels = {
            "threshold": "Threshold",
            "rank": "Rank",
            "user_input": "Number of singular values to keep",
            "interactive": "Number of singular values to keep",
        }
        self.svd_parameter_label.setText(parameter_labels.get(method, "SVD parameter"))
        self.svd_threshold.setVisible(method == "threshold")
        self.svd_threshold.setEnabled(method == "threshold")
        self.svd_rank.setVisible(method in {"rank", "user_input", "interactive"})
        self.svd_rank.setEnabled(method in {"rank", "user_input"})
        self.svd_parameter_input.setEnabled(method != "interactive")
        self.svd_plot.setEnabled(method != "interactive")
        if method == "interactive":
            self.svd_plot.setChecked(True)

    def _update_cmstep_input_availability(self) -> None:
        is_uniform = (self.cmstep_mode.currentData() or "uniform") == "uniform"
        self.params_cmstep_h.setEnabled(is_uniform)
        self.params_cmstep_v.setEnabled(is_uniform)
        self.params_cmstep_file.setEnabled(not is_uniform)
        self.params_cmstep_browse.setEnabled(not is_uniform)


    @Slot()
    def _browse_cmstep_file(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Load corrector-step file",
            "",
            "NumPy archives (*.npz);;All files (*)",
        )[0]
        if filename:
            self.params_cmstep_file.setText(filename)

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
        self.rm_hcm_coupling.setText(cfg.response_matrix.HCMCoupling)
        self.rm_vcm_coupling.setText(cfg.response_matrix.VCMCoupling)
        self.rm_frequency.setText(cfg.response_matrix.Frequency)
        self.rm_harm_number.setText(cfg.response_matrix.HarmNumber)
        self.rm_rf_attr.setText(cfg.response_matrix.RFAttr)
        self._refresh_element_selection_ui()
        self.rm_fixedpath.setChecked(cfg.response_matrix.fixedpathlength)
        self.rm_log_info.setChecked(cfg.response_matrix.log_info)
        self._set_solver_algorithm_value(cfg.solver.algorithm)
        self.solver_n_iter.setValue(cfg.solver.nIter)
        self.solver_lm_iter.setValue(cfg.solver.nLMIter)
        self.solver_lambda.setValue(cfg.solver.Starting_Lambda)
        self.solver_max_lambda.setValue(cfg.solver.max_lm_lambda)
        self.solver_scaled.setChecked(cfg.solver.scaled)
        self._set_svd_method_value(cfg.svd.svd_selection_method)
        self.svd_threshold.setValue(cfg.svd.svd_threshold)
        self.svd_rank.setValue(cfg.svd.cut_)
        self.svd_plot.setChecked(cfg.svd.show_svd_plot)
        self.outlier_enabled.setChecked(cfg.rejection.outlier_rejection)
        self.outlier_sigma.setValue(cfg.rejection.sigma_outlier)
        self.norm_enabled.setChecked(cfg.rejection.apply_normalization)
        self.norm_mode.setCurrentText(cfg.rejection.normalization_mode)
        self.loco_include_dispersion.setChecked(cfg.rejection.includeDispersion)
        self.loco_hor_dispersion_weight.setValue(cfg.rejection.hor_dispersion_weight)
        self.loco_ver_dispersion_weight.setValue(cfg.rejection.ver_dispersion_weight)
        self.auto_delta.setChecked(cfg.rejection.auto_correct_delta)
        self.loco_fixedpath.setChecked(cfg.rejection.fixedpathlength)
        self.loco_individuals.setChecked(cfg.rejection.individuals)
        self.loco_remove_coupling.setChecked(cfg.rejection.remove_coupling_)
        self.loco_plot_fit_parameters.setChecked(cfg.rejection.plot_fit_parameters)
        self.constraint_enabled.setChecked(cfg.constraints.enable)
        self.constraint_quad_sigma.setValue(cfg.constraints.quad_sigma)
        self.constraint_skew_sigma.setValue(cfg.constraints.skew_sigma)
        self.constraint_quad_weights.setText(cfg.constraints.quad_weights)
        self.constraint_skew_weights.setText(cfg.constraints.skew_weights)
        self.constraint_quad_mask.setText(cfg.constraints.quad_mask)
        self.constraint_skew_mask.setText(cfg.constraints.skew_mask)
        for name, check in self.parameter_checks.items():
            check.setChecked(bool(getattr(cfg.parameters, name)))
        self.params_individuals.setChecked(cfg.parameters.individuals)
        self.cmstep_mode.setCurrentIndex(max(0, self.cmstep_mode.findData(cfg.parameters.CMstep_mode)))
        self.params_init_policy.setText(cfg.parameters.init_policy)
        self.params_cmstep_h.setValue(float(cfg.parameters.CMstep_h))
        self.params_cmstep_v.setValue(float(cfg.parameters.CMstep_v))
        self.params_cmstep_file.setText(cfg.parameters.CMstep_file)
        self.params_rfstep.setValue(cfg.parameters.rfStep)
        self.params_init.setText(cfg.parameters.init)
        self.params_quads_attr.setText(cfg.parameters.quads_attr)
        self.params_quads_attr_index.setValue(cfg.parameters.quads_attr_index)
        self.params_skew_attr.setText(cfg.parameters.skew_attr)
        self.params_skew_attr_index.setValue(cfg.parameters.skew_attr_index)
        self.params_tilt_attr_r1.setText(cfg.parameters.quads_tilt_attr_R1)
        self.params_tilt_attr_r2.setText(cfg.parameters.quads_tilt_attr_R2)
        self.params_tilt_method.setText(cfg.parameters.quads_tilt_method)
        self.fixed_frequency.setText(str(cfg.fixed_parameters.Frequency))
        self.fixed_harm_number.setValue(cfg.fixed_parameters.HarmNumber)
        self.fixed_rfstep.setValue(cfg.fixed_parameters.rfstep)
        self.fixed_dk.setText(cfg.fixed_parameters.dk)
        self.fixed_delta_skew.setValue(cfg.fixed_parameters.delta_skew)
        self.fixed_delta_q_tilt.setValue(cfg.fixed_parameters.delta_q_tilt)
        self.mcf_source.setCurrentIndex(max(0, self.mcf_source.findData(cfg.mcf_source)))
        self.mcf_user_value.setText(cfg.mcf_user_value)
        self._apply_mode_visibility()
        self._update_svd_input_availability()
        self._update_cmstep_input_availability()
        self._update_fit_summary()

    def _collect_loco_configuration(self) -> LocoConfiguration:
        cfg = LocoConfiguration()
        cfg.output_directory = self.project.loco_config.output_directory
        cfg.response_matrix.calculator = self.rm_calculator.currentData() or self.rm_calculator.currentText()
        cfg.response_matrix.includeDispersion = self.rm_dispersion.isChecked()
        cfg.response_matrix.coupling_orm = self.rm_coupling.isChecked()
        cfg.response_matrix.bidirectional = self.rm_bidirectional.isChecked()
        cfg.response_matrix.NewVectorizedMethod = self.rm_vectorized.isChecked()
        cfg.response_matrix.dkick_h = self.rm_dkick_h.value()
        cfg.response_matrix.dkick_v = self.rm_dkick_v.value()
        cfg.response_matrix.rfStep = self.rm_rf_step.value()
        cfg.response_matrix.delta_coupling = self.rm_delta_coupling.value()
        cfg.machine_elements = self.project.loco_config.machine_elements
        cfg._sync_response_matrix_elements()
        cfg.response_matrix.fixedpathlength = self.rm_fixedpath.isChecked()
        cfg.response_matrix.log_info = self.rm_log_info.isChecked()
        cfg.response_matrix.HCMCoupling = self.rm_hcm_coupling.text()
        cfg.response_matrix.VCMCoupling = self.rm_vcm_coupling.text()
        cfg.response_matrix.Frequency = self.rm_frequency.text()
        cfg.response_matrix.HarmNumber = self.rm_harm_number.text()
        cfg.response_matrix.RFAttr = self.rm_rf_attr.text()
        cfg.solver.algorithm = self._selected_solver_algorithm()
        cfg.solver.nIter = self.solver_n_iter.value()
        cfg.solver.nLMIter = self.solver_lm_iter.value()
        cfg.solver.Starting_Lambda = self.solver_lambda.value()
        cfg.solver.max_lm_lambda = self.solver_max_lambda.value()
        cfg.solver.scaled = self.solver_scaled.isChecked() and cfg.solver.algorithm == "lm"
        cfg.svd.svd_selection_method = self._selected_svd_method()
        cfg.svd.svd_threshold = self.svd_threshold.value()
        cfg.svd.cut_ = self.svd_rank.value()
        cfg.svd.show_svd_plot = self.svd_plot.isChecked()
        cfg.rejection.outlier_rejection = self.outlier_enabled.isChecked()
        cfg.rejection.sigma_outlier = self.outlier_sigma.value()
        cfg.rejection.apply_normalization = self.norm_enabled.isChecked()
        cfg.rejection.normalization_mode = self.norm_mode.currentText()
        cfg.rejection.includeDispersion = self.loco_include_dispersion.isChecked() or self.rm_dispersion.isChecked()
        cfg.response_matrix.includeDispersion = cfg.rejection.includeDispersion
        cfg.rejection.hor_dispersion_weight = self.loco_hor_dispersion_weight.value()
        cfg.rejection.ver_dispersion_weight = self.loco_ver_dispersion_weight.value()
        cfg.rejection.auto_correct_delta = self.auto_delta.isChecked()
        cfg.rejection.fixedpathlength = self.loco_fixedpath.isChecked() or self.rm_fixedpath.isChecked()
        cfg.response_matrix.fixedpathlength = cfg.rejection.fixedpathlength
        cfg.rejection.individuals = self.loco_individuals.isChecked()
        cfg.rejection.remove_coupling_ = self.loco_remove_coupling.isChecked()
        cfg.rejection.plot_fit_parameters = self.loco_plot_fit_parameters.isChecked()
        cfg.constraints.enable = self.constraint_enabled.isChecked()
        cfg.constraints.quad_sigma = self.constraint_quad_sigma.value()
        cfg.constraints.skew_sigma = self.constraint_skew_sigma.value()
        cfg.constraints.quad_weights = self.constraint_quad_weights.text()
        cfg.constraints.skew_weights = self.constraint_skew_weights.text()
        cfg.constraints.quad_mask = self.constraint_quad_mask.text()
        cfg.constraints.skew_mask = self.constraint_skew_mask.text()
        for name, check in self.parameter_checks.items():
            setattr(cfg.parameters, name, check.isChecked())
        cfg.parameters.individuals = self.params_individuals.isChecked()
        cfg.parameters.CMstep_mode = self.cmstep_mode.currentData() or "uniform"
        cfg.parameters.init_policy = self.params_init_policy.text()
        cfg.parameters.CMstep_h = self.params_cmstep_h.value()
        cfg.parameters.CMstep_v = self.params_cmstep_v.value()
        cfg.parameters.CMstep_file = self.params_cmstep_file.text()
        cfg.parameters.rfStep = self.params_rfstep.value()
        cfg.parameters.init = self.params_init.text()
        cfg.parameters.quads_attr = self.params_quads_attr.text()
        cfg.parameters.quads_attr_index = self.params_quads_attr_index.value()
        cfg.parameters.skew_attr = self.params_skew_attr.text()
        cfg.parameters.skew_attr_index = self.params_skew_attr_index.value()
        cfg.parameters.quads_tilt_attr_R1 = self.params_tilt_attr_r1.text()
        cfg.parameters.quads_tilt_attr_R2 = self.params_tilt_attr_r2.text()
        cfg.parameters.quads_tilt_method = self.params_tilt_method.text()
        cfg.fixed_parameters.Frequency = self.fixed_frequency.text()
        cfg.fixed_parameters.HarmNumber = self.fixed_harm_number.value()
        cfg.fixed_parameters.rfstep = self.fixed_rfstep.value()
        cfg.fixed_parameters.dk = self.fixed_dk.text()
        cfg.fixed_parameters.delta_skew = self.fixed_delta_skew.value()
        cfg.fixed_parameters.delta_q_tilt = self.fixed_delta_q_tilt.value()
        cfg.mcf_source = self.mcf_source.currentData() or "automatic"
        cfg.mcf_user_value = self.mcf_user_value.text()
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
            config, measurements, lattice = load_example_project_data(filename)
            self.project.loco_config = config
            if lattice:
                source = Path(lattice)
                if not source.exists():
                    raise ValueError(f"Configured lattice file does not exist: {source}")
                self.project.lattice = LatticeSelection(path=str(source), file_type=source.suffix.lower().lstrip("."))
                loaded_lattice = self._load_current_lattice()
                if loaded_lattice is None:
                    raise ValueError(f"Unable to load configured lattice: {source}")
                self.project.lattice.element_count = len(loaded_lattice)
                resolved = resolve_example_machine_elements(filename, loaded_lattice)
                if any(getattr(resolved, key) for key in ELEMENT_ROLES):
                    self.project.loco_config.machine_elements = resolved
                    self.project.loco_config._sync_response_matrix_elements()
            for role, value in measurements.items():
                source = Path(value)
                if not source.exists():
                    raise ValueError(f"Configured {role.replace('_', ' ')} file does not exist: {source}")
                self.project.measurements[role] = ImportedDataset(
                    role=role, path=str(source), file_type=source.suffix.lower().lstrip("."), size_bytes=source.stat().st_size
                )
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


    def _load_current_lattice(self):
        """Load the currently selected AT lattice for element detection/preview."""

        if not self.project.lattice.path:
            return None
        try:
            import at
            return at.load_lattice(self.project.lattice.path)
        except Exception:
            return None

    def _element_preview_rows(self, ords: list[int]) -> list[tuple[int, int, str, str]]:
        lattice = self._load_current_lattice()
        rows = []
        for position, ordinal in enumerate(ords):
            elem = lattice[ordinal] if lattice and 0 <= ordinal < len(lattice) else None
            name = ElementSelectionDialog._element_name(None, elem) if elem else ""
            cls = type(elem).__name__ if elem else ""
            rows.append((position, ordinal, name, cls))
        return rows

    def _refresh_element_selection_ui(self) -> None:
        if not hasattr(self, "element_count_labels"):
            return
        elements = self.project.loco_config.machine_elements
        advanced = self.project.mode == "Advanced"
        for key in ELEMENT_ROLES:
            values = list(getattr(elements, key))
            self.element_count_labels[key].setText(f"{len(values)} selected")
            table = self.element_preview_tables[key]
            table.setVisible(advanced)
            rows = self._element_preview_rows(values) if advanced else []
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(str(value)))

    @Slot()
    def edit_element_selection(self, role_key: str) -> None:
        elements = self.project.loco_config.machine_elements
        current = list(getattr(elements, role_key))
        dialog = ElementSelectionDialog(self, role_key, current)
        if dialog.exec() != QDialog.Accepted:
            return
        setattr(elements, role_key, dialog.selected_ords)
        self.project.loco_config._sync_response_matrix_elements()
        self.project.modified = True
        self._refresh_element_selection_ui()
        self._update_fit_summary()
        self._refresh_ui(f"Updated {ELEMENT_ROLES[role_key][0]} selection")

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
            lattice = self._load_current_lattice()
            if lattice is not None:
                self.project.lattice.element_count = len(lattice)
            self.project.modified = True
            self._refresh_element_selection_ui()
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
            source = Path(filename)
            role = self.measurement_role.currentText()
            path = self._store_imported_measurement(source, role)
            self.project.measurements[role] = ImportedDataset(
                role=role,
                path=str(path),
                file_type=path.suffix.lower().lstrip("."),
                size_bytes=path.stat().st_size,
            )
            self.project.modified = True
            self._refresh_ui(f"Imported {role}: {path.name}")


    def _store_imported_measurement(self, source: Path, role: str) -> Path:
        """Copy imported measurement data into the project folder when possible."""

        if not self.project.path:
            return source
        project_dir = Path(self.project.path).expanduser().resolve().parent
        data_dir = project_dir / "measurements"
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / f"{role}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

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

    def _apply_theme_selection(self, theme_key: str | None) -> None:
        self.current_theme = theme_for_key(theme_key)
        apply_application_theme(QApplication.instance(), self.current_theme)
        self._settings.setValue("appearance/theme", self.current_theme.key)
        self._settings.sync()
        for key, action in self.theme_actions.items():
            action.setChecked(key == self.current_theme.key)
        self._update_toggle_theme_action()
        for window in self._orm_comparison_windows:
            if hasattr(window, "apply_theme"):
                window.apply_theme(self.current_theme)
        if hasattr(self, "results_workspace"):
            self.results_workspace.apply_theme()
        self._refresh_ui(f"{self.current_theme.display_name} theme selected")

    def _update_toggle_theme_action(self) -> None:
        if self.current_theme.key == "dark":
            self.toggle_theme_action.setText("☀️ Light")
            self.toggle_theme_action.setToolTip("Switch to the Light theme")
        else:
            self.toggle_theme_action.setText("🌙 Dark")
            self.toggle_theme_action.setToolTip("Switch to the Dark theme")


    def _apply_mode_visibility(self) -> None:
        if not hasattr(self, "fixed_group"):
            return
        advanced = self.project.mode == "Advanced"
        widgets = (
            self.rm_fixedpath, self.rm_log_info, self.loco_include_dispersion,
            self.loco_hor_dispersion_weight, self.loco_ver_dispersion_weight, self.loco_fixedpath,
            self.loco_individuals, self.loco_remove_coupling, self.loco_plot_fit_parameters,
            self.params_init_policy, self.params_cmstep_h, self.params_cmstep_v, self.params_cmstep_file_row, self.params_rfstep,
            self.params_init, self.params_quads_attr, self.params_quads_attr_index, self.params_skew_attr,
            self.params_skew_attr_index, self.params_tilt_attr_r1, self.params_tilt_attr_r2,
            self.params_tilt_method, self.fixed_group,
        )
        for widget in widgets:
            widget.setVisible(advanced)
        if hasattr(self, "results_workspace"):
            self.results_workspace.set_mode(self.project.mode)

    @Slot()
    def _toggle_theme(self) -> None:
        next_theme = "light" if self.current_theme.key == "dark" else "dark"
        self._apply_theme_selection(next_theme)

    @Slot(QAction)
    def _on_theme_changed(self, action: QAction) -> None:
        self._apply_theme_selection(action.data())

    @Slot(QAction)
    def _on_mode_changed(self, action: QAction) -> None:
        self.project.mode = action.text()
        self.project.modified = True
        self._mode_label.setText(f"{self.project.mode} mode")
        self._apply_mode_visibility()
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
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self.lattice_path.setText(self.project.lattice.path or "No lattice selected")
        self.lattice_type.setText(self.project.lattice.file_type or "—")
        self.lattice_elements.setText(
            str(self.project.lattice.element_count)
            if self.project.lattice.element_count
            else "Unknown"
        )
        self._refresh_element_selection_ui()
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
        self.results_workspace.begin_run()
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
        self.results_workspace.append_log(message)

    @Slot(object)
    def _on_loco_finished(self, result) -> None:
        self._append_run_log("Saved outputs:\n" + "\n".join(result.output_files))
        self.results_workspace.complete_run(result)
        self._project_explorer.set_result(self.results_workspace.loader)
        self._project_explorer.update_project(self.project)
        self._last_loco_result = result
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        QMessageBox.information(self, "LOCO complete", f"LOCO completed successfully.\n\nResults: {result.results_dir}")

    @Slot(str)
    def _navigate_from_explorer(self, target: str) -> None:
        if target in {"Machine", "Measurements", "Fit"}:
            index = next((i for i in range(self._workspace.count()) if self._workspace.tabText(i) == target), -1)
            if index >= 0:
                self._workspace.setCurrentIndex(index)
            return
        if not target.startswith("Results:"):
            return
        result_page = next((i for i in range(self._workspace.count()) if self._workspace.tabText(i) == "Results"), -1)
        if result_page >= 0:
            self._workspace.setCurrentIndex(result_page)
        label = target.split(":", 1)[1]
        index = next((i for i in range(self.results_workspace.tabs.count()) if self.results_workspace.tabs.tabText(i) == label), -1)
        if index >= 0 and self.results_workspace.tabs.isTabVisible(index):
            self.results_workspace.tabs.setCurrentIndex(index)

    @Slot(object)
    def _on_loco_failed(self, error: LocoRunError) -> None:
        self.results_workspace.fail_run()
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
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self._refresh_ui("LOCO run finished")

    @Slot()
    def _update_elapsed_time(self) -> None:
        if self._run_started_at:
            elapsed = __import__("time").monotonic() - self._run_started_at
            self.run_elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")


    def _can_compare_orms(self) -> bool:
        return "orm" in self.project.measurements and self._latest_model_orm_path() is not None

    def _latest_model_orm_path(self) -> Path | None:
        names = ("loco_results.npz", "model_orm_initial.h5")
        if self._last_loco_result is not None:
            result_dir = Path(self._last_loco_result.results_dir)
            for name in names:
                candidate = result_dir / name
                if candidate.exists():
                    return candidate
        if not self.project.path:
            return None
        results_root = Path(self.project.path).expanduser().resolve().parent / "results"
        if not results_root.exists():
            return None
        candidates = [path for name in names for path in results_root.glob(f"*/{name}")]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _load_measured_orm_for_comparison(self):
        import h5py
        import numpy as np

        dataset = self.project.measurements["orm"]
        path = Path(dataset.path).expanduser()
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as handle:
                if "response_matrix" in handle:
                    measured_orm = np.asarray(handle["response_matrix"])
                else:
                    keys = list(handle.keys())
                    if not keys:
                        raise ValueError(f"ORM measurement file {path} contains no datasets.")
                    measured_orm = np.asarray(handle[keys[0]])
        elif suffix == ".npy":
            measured_orm = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                key = "orm" if "orm" in archive else archive.files[0]
                measured_orm = np.asarray(archive[key])
        else:
            raise ValueError(f"Unsupported ORM comparison file type: {suffix}")

        bad_bpm_positions = _load_bad_bpm_positions(
            {key: dataset.path for key, dataset in self.project.measurements.items()}
        )
        if bad_bpm_positions is None:
            return measured_orm

        from pyLOCO.pyloco import remove_bad_bpms

        total_bpms = measured_orm.shape[0] // 2
        if measured_orm.shape[0] != total_bpms * 2:
            raise ValueError(
                "Measured ORM must have an even number of rows before applying the Bad BPM list; "
                f"got shape {measured_orm.shape}."
            )
        cleaned_orm, _removed = remove_bad_bpms(
            measured_orm,
            bad_bpm_positions,
            total_bpms=total_bpms,
            axis=0,
            input_type="positions",
        )
        return cleaned_orm

    def _load_model_orm_for_comparison(self):
        import numpy as np

        path = self._latest_model_orm_path()
        if path is None:
            raise ValueError("No initial or final model ORM result was found.")
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                if "response_matrix" not in handle:
                    raise ValueError(f"{path} does not contain a response_matrix dataset.")
                return np.asarray(handle["response_matrix"])
        with np.load(path, allow_pickle=True) as archive:
            if "orm_model" not in archive:
                raise ValueError(f"{path} does not contain an orm_model array.")
            return np.asarray(archive["orm_model"])

    @Slot()
    def compare_orms(self) -> None:
        try:
            measured_orm = self._load_measured_orm_for_comparison()
            model_orm = self._load_model_orm_for_comparison()
            window = OrmComparisonWindow(measured_orm, model_orm, self)
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            QMessageBox.warning(self, "ORM Comparison unavailable", str(exc))
            return
        self._orm_comparison_windows.append(window)
        window.destroyed.connect(lambda _obj=None, w=window: self._orm_comparison_windows.remove(w) if w in self._orm_comparison_windows else None)
        window.show()

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About pyLOCO GUI",
            "pyLOCO GUI\n\nMilestone 2 project management and data import shell. Numerical backend code is unchanged.",
        )
