"""Main window for the Milestone 2 pyLOCO GUI.

The GUI manages project state, lattice metadata, imported measurement files,
and validation readiness only. It deliberately does not import or execute the
numerical pyLOCO backend.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
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
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .models.project import ImportedDataset, LatticeSelection, ProjectMetadata
from .widgets.placeholders import PlaceholderPage
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
        tabs.addTab(
            PlaceholderPage(
                "Fit Preparation",
                "Select fit settings in a later milestone; Run LOCO stays disabled until project validation passes.",
            ),
            "Fit",
        )
        tabs.addTab(
            PlaceholderPage(
                "Results Workspace",
                "Run summaries and exports will appear after execution milestones are implemented.",
            ),
            "Results",
        )
        return tabs

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
        self.run_loco_action.triggered.connect(self._run_loco_notice)
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

    @Slot()
    def new_project(self) -> None:
        recent = self.project.recent_projects
        self.project = ProjectMetadata(recent_projects=recent)
        self.dashboard_name.setText(self.project.name)
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

    def _run_loco_notice(self) -> None:
        QMessageBox.information(
            self,
            "Run LOCO",
            "Project inputs are complete. Numerical LOCO execution is reserved for a later milestone and the backend remains unchanged.",
        )

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About pyLOCO GUI",
            "pyLOCO GUI\n\nMilestone 2 project management and data import shell. Numerical backend code is unchanged.",
        )
