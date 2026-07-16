"""Main window for the Milestone 1 pyLOCO GUI shell.

The window is deliberately focused on navigation and user orientation:
menus, toolbar actions, project explorer, workflow tabs, status updates,
and a Basic/Advanced mode toggle. Placeholder pages mark future
milestones; no numerical backend code is imported or executed here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from .widgets.placeholders import PlaceholderPage
from .widgets.project_explorer import ProjectExplorer


class MainWindow(QMainWindow):
    """Top-level pyLOCO GUI window for Milestone 1.

    The class wires together the static application shell. Slots currently
    show placeholder messages because project loading, validation, and LOCO
    execution belong to later milestones.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyLOCO GUI")
        self.resize(1280, 820)

        self._mode_label = QLabel("Mode: Basic")
        self._workspace = self._create_workspace()
        self._project_explorer = ProjectExplorer()

        self.setCentralWidget(self._workspace)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._project_explorer)

        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()

    def _create_workspace(self) -> QTabWidget:
        """Create the central workflow tab area."""

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(
            PlaceholderPage(
                title="Project",
                description=(
                    "Create or open a LOCO project. Project persistence and "
                    "recent-run management arrive in a later milestone."
                ),
            ),
            "Project",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Machine",
                description=(
                    "Load a machine model, inspect lattice elements, and define "
                    "BPM/corrector/quadrupole selections."
                ),
            ),
            "Machine",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Measurements",
                description=(
                    "Import ORM, BPM noise, dispersion, and mask data. Dataset "
                    "mapping and validation are planned for Milestone 2."
                ),
            ),
            "Measurements",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Fit",
                description=(
                    "Select Basic presets or Advanced fit settings. LOCO execution "
                    "is intentionally not implemented in Milestone 1."
                ),
            ),
            "Fit",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Results",
                description=(
                    "Review run summaries, plots, correction tables, and exports "
                    "after future execution milestones are implemented."
                ),
            ),
            "Results",
        )
        return tabs

    def _create_actions(self) -> None:
        """Create reusable actions for menus and toolbars."""

        self.new_project_action = QAction("New Project", self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.setStatusTip("Create a new pyLOCO GUI project shell")
        self.new_project_action.triggered.connect(
            lambda: self._show_placeholder("New Project")
        )

        self.open_project_action = QAction("Open Project…", self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.setStatusTip("Open an existing pyLOCO GUI project")
        self.open_project_action.triggered.connect(
            lambda: self._show_placeholder("Open Project")
        )

        self.save_project_action = QAction("Save Project", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setStatusTip("Save the current project")
        self.save_project_action.triggered.connect(
            lambda: self._show_placeholder("Save Project")
        )

        self.validate_project_action = QAction("Validate Project", self)
        self.validate_project_action.setStatusTip("Validate project inputs and dimensions")
        self.validate_project_action.triggered.connect(
            lambda: self._show_placeholder("Validate Project")
        )

        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Close the pyLOCO GUI")
        self.exit_action.triggered.connect(self.close)

        self.basic_mode_action = QAction("Basic", self)
        self.basic_mode_action.setCheckable(True)
        self.basic_mode_action.setChecked(True)
        self.basic_mode_action.setStatusTip("Use guided workflow pages and presets")

        self.advanced_mode_action = QAction("Advanced", self)
        self.advanced_mode_action.setCheckable(True)
        self.advanced_mode_action.setStatusTip("Expose advanced settings in future milestones")

        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)
        self.mode_action_group.addAction(self.basic_mode_action)
        self.mode_action_group.addAction(self.advanced_mode_action)
        self.mode_action_group.triggered.connect(self._on_mode_changed)

        self.about_action = QAction("About pyLOCO GUI", self)
        self.about_action.triggered.connect(self._show_about_dialog)

    def _create_menu_bar(self) -> None:
        """Build the application menu bar."""

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.validate_project_action)
        project_menu.addAction("Project Settings…", lambda: self._show_placeholder("Project Settings"))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._project_explorer.toggleViewAction())
        mode_menu = view_menu.addMenu("Mode")
        mode_menu.addAction(self.basic_mode_action)
        mode_menu.addAction(self.advanced_mode_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction("Plugin Manager…", lambda: self._show_placeholder("Plugin Manager"))
        tools_menu.addAction("Preferences…", lambda: self._show_placeholder("Preferences"))

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        """Build the primary toolbar."""

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.validate_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.basic_mode_action)
        toolbar.addAction(self.advanced_mode_action)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def _create_status_bar(self) -> None:
        """Build the status bar with mode feedback."""

        status_bar = QStatusBar(self)
        status_bar.showMessage("Ready — Milestone 1 GUI shell loaded")
        status_bar.addPermanentWidget(self._mode_label)
        self.setStatusBar(status_bar)

    @Slot(QAction)
    def _on_mode_changed(self, action: QAction) -> None:
        """Update status text when Basic/Advanced mode changes."""

        mode = action.text()
        self._mode_label.setText(f"Mode: {mode}")
        self.statusBar().showMessage(f"{mode} mode selected", 3000)
        self._project_explorer.set_mode(mode)

    def _show_placeholder(self, feature_name: str) -> None:
        """Display a standard placeholder notification for future features."""

        QMessageBox.information(
            self,
            f"{feature_name} Placeholder",
            (
                f"{feature_name} is part of a future milestone. "
                "Milestone 1 provides the GUI shell only."
            ),
        )

    def _show_about_dialog(self) -> None:
        """Show a short about dialog for the GUI shell."""

        QMessageBox.about(
            self,
            "About pyLOCO GUI",
            (
                "pyLOCO GUI\n\n"
                "Milestone 1 application shell built with PySide6.\n"
                "No LOCO execution or numerical backend changes are included."
            ),
        )
