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


APP_STYLESHEET = """
QMainWindow {
    background: #f4f7fb;
}
QMenuBar {
    background: #ffffff;
    border-bottom: 1px solid #d9e2ef;
    padding: 3px 8px;
}
QMenuBar::item {
    border-radius: 4px;
    padding: 5px 10px;
}
QMenuBar::item:selected {
    background: #eaf1fb;
    color: #183b66;
}
QMenu {
    background: #ffffff;
    border: 1px solid #cad6e5;
    padding: 6px;
}
QMenu::item {
    border-radius: 4px;
    padding: 7px 28px 7px 22px;
}
QMenu::item:selected {
    background: #e8f0fe;
    color: #12365f;
}
QToolBar#mainToolbar {
    background: #ffffff;
    border: 0;
    border-bottom: 1px solid #d9e2ef;
    spacing: 8px;
    padding: 8px 12px;
}
QToolButton {
    border: 1px solid transparent;
    border-radius: 6px;
    color: #1f3b57;
    font-weight: 600;
    padding: 6px 10px;
}
QToolButton:hover {
    background: #eef5ff;
    border-color: #c8d8ec;
}
QToolButton:checked {
    background: #2463a6;
    color: #ffffff;
}
QTabWidget::pane {
    background: #f4f7fb;
    border: 0;
    padding-top: 10px;
}
QTabBar::tab {
    background: #e8eef7;
    border: 1px solid #d3deec;
    border-bottom: 0;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: #2f4b66;
    margin-right: 4px;
    padding: 10px 18px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #12365f;
    font-weight: 700;
}
QDockWidget#projectExplorerDock {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: #183b66;
    color: #ffffff;
    font-weight: 700;
    padding: 8px 10px;
    text-align: left;
}
QTreeWidget#projectExplorerTree {
    background: #ffffff;
    border: 1px solid #d8e1ee;
    border-radius: 8px;
    color: #263c53;
    margin: 8px;
    outline: 0;
    padding: 6px;
}
QTreeWidget#projectExplorerTree::item {
    border-radius: 5px;
    min-height: 24px;
    padding: 3px 4px;
}
QTreeWidget#projectExplorerTree::item:hover {
    background: #edf4ff;
}
QTreeWidget#projectExplorerTree::item:selected {
    background: #dceafe;
    color: #12365f;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d9e2ef;
    color: #38516b;
    padding: 2px 8px;
}
QLabel#statusPill {
    background: #e8f0fe;
    border: 1px solid #cbdcf4;
    border-radius: 9px;
    color: #174a7c;
    font-weight: 700;
    padding: 3px 10px;
}
QWidget#placeholderPageCard {
    background: #ffffff;
    border: 1px solid #d9e2ef;
    border-radius: 14px;
}
QLabel#placeholderTitle {
    color: #14395f;
    font-size: 26px;
    font-weight: 700;
}
QLabel#placeholderDescription {
    color: #52677d;
    font-size: 14px;
}
QWidget#dashboardCard {
    background: #f8fbff;
    border: 1px solid #d9e6f5;
    border-radius: 10px;
}
QLabel#dashboardCardTitle {
    color: #1b426d;
    font-size: 15px;
    font-weight: 700;
}
QLabel#dashboardCardText {
    color: #5b7085;
    font-size: 12px;
}
"""


class MainWindow(QMainWindow):
    """Top-level pyLOCO GUI window for Milestone 1.

    The class wires together the static application shell. Slots currently
    show placeholder messages because project loading, validation, and LOCO
    execution belong to later milestones.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pyLocoMainWindow")
        self.setWindowTitle("pyLOCO GUI")
        self.resize(1320, 860)
        self.setStyleSheet(APP_STYLESHEET)

        self._mode_label = QLabel("Basic mode")
        self._mode_label.setObjectName("statusPill")
        self._project_label = QLabel("Project: Untitled")
        self._workflow_label = QLabel("Workflow: Project")
        self._backend_label = QLabel("Backend: not connected")
        self._workspace = self._create_workspace()
        self._project_explorer = ProjectExplorer()

        self.setCentralWidget(self._workspace)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._project_explorer)

        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._workspace.currentChanged.connect(self._on_tab_changed)

    def _create_workspace(self) -> QTabWidget:
        """Create the central workflow tab area."""

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.setUsesScrollButtons(True)
        tabs.addTab(
            PlaceholderPage(
                title="Project Dashboard",
                description=(
                    "Create or open a LOCO project. Project persistence and "
                    "recent-run management arrive in a later milestone."
                ),
                cards=(
                    (
                        "Project setup",
                        "Start with a named project and machine profile.",
                    ),
                    (
                        "Readiness checklist",
                        "Track model, ORM, dispersion, noise, and masks.",
                    ),
                    (
                        "Recent activity",
                        "Future runs and imported datasets will appear here.",
                    ),
                ),
            ),
            "Project",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Machine Overview",
                description=(
                    "Load a machine model, inspect lattice elements, and define "
                    "BPM/corrector/quadrupole selections."
                ),
                cards=(
                    (
                        "Lattice summary",
                        "Placeholder for element counts and selected families.",
                    ),
                    (
                        "Device selections",
                        "Future controls will summarize BPMs, correctors, RF, and quads.",
                    ),
                ),
            ),
            "Machine",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Measurement Intake",
                description=(
                    "Import ORM, BPM noise, dispersion, and mask data. Dataset "
                    "mapping and validation are planned for Milestone 2."
                ),
                cards=(
                    (
                        "Required inputs",
                        "ORM, BPM noise, dispersion, and masks stay UI-only.",
                    ),
                    (
                        "Validation preview",
                        "Future checks will report dimensions, units, and missing devices.",
                    ),
                ),
            ),
            "Measurements",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Fit Preparation",
                description=(
                    "Select Basic presets or Advanced fit settings. LOCO execution "
                    "is intentionally not implemented in Milestone 1."
                ),
                cards=(
                    (
                        "Basic presets",
                        "Guided defaults will emphasize compact required choices.",
                    ),
                    (
                        "Advanced controls",
                        "Expert solver, SVD, Jacobian, and block options surface on demand.",
                    ),
                ),
            ),
            "Fit",
        )
        tabs.addTab(
            PlaceholderPage(
                title="Results Workspace",
                description=(
                    "Review run summaries, plots, correction tables, and exports "
                    "after future execution milestones are implemented."
                ),
                cards=(
                    (
                        "Run summary",
                        "No calculations run here; future results will list here.",
                    ),
                    (
                        "Export queue",
                        "Placeholder for correction tables, plots, logs, and reports.",
                    ),
                ),
            ),
            "Results",
        )
        return tabs

    def _create_actions(self) -> None:
        """Create reusable actions for menus and toolbars."""

        self.new_project_action = QAction("New", self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.setStatusTip("Create a new pyLOCO GUI project shell")
        self.new_project_action.triggered.connect(
            lambda: self._show_placeholder("New Project")
        )

        self.open_project_action = QAction("Open…", self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.setStatusTip("Open an existing pyLOCO GUI project")
        self.open_project_action.triggered.connect(
            lambda: self._show_placeholder("Open Project")
        )

        self.save_project_action = QAction("Save", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.setStatusTip("Save the current project")
        self.save_project_action.triggered.connect(
            lambda: self._show_placeholder("Save Project")
        )

        self.validate_project_action = QAction("Validate", self)
        self.validate_project_action.setStatusTip(
            "Validate project inputs and dimensions"
        )
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
        self.advanced_mode_action.setStatusTip(
            "Expose advanced settings in future milestones"
        )

        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)
        self.mode_action_group.addAction(self.basic_mode_action)
        self.mode_action_group.addAction(self.advanced_mode_action)
        self.mode_action_group.triggered.connect(self._on_mode_changed)

        self.project_settings_action = QAction("Project Settings…", self)
        self.project_settings_action.setStatusTip(
            "Review future project-level settings"
        )
        self.project_settings_action.triggered.connect(
            lambda: self._show_placeholder("Project Settings")
        )

        self.preferences_action = QAction("Preferences…", self)
        self.preferences_action.setStatusTip("Configure future GUI preferences")
        self.preferences_action.triggered.connect(
            lambda: self._show_placeholder("Preferences")
        )

        self.plugin_manager_action = QAction("Plugin Manager…", self)
        self.plugin_manager_action.setStatusTip("Manage future GUI plugins")
        self.plugin_manager_action.triggered.connect(
            lambda: self._show_placeholder("Plugin Manager")
        )

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
        project_menu.addAction(self.project_settings_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._project_explorer.toggleViewAction())
        mode_menu = view_menu.addMenu("Workflow Mode")
        mode_menu.addAction(self.basic_mode_action)
        mode_menu.addAction(self.advanced_mode_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self.plugin_manager_action)
        tools_menu.addAction(self.preferences_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        """Build the primary toolbar."""

        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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
        status_bar.showMessage(
            "Ready — Milestone 1 GUI shell loaded; numerical backend is not connected"
        )
        status_bar.addWidget(self._project_label, 1)
        status_bar.addWidget(self._workflow_label, 1)
        status_bar.addPermanentWidget(self._backend_label)
        status_bar.addPermanentWidget(self._mode_label)
        self.setStatusBar(status_bar)

    @Slot(QAction)
    def _on_mode_changed(self, action: QAction) -> None:
        """Update status text when Basic/Advanced mode changes."""

        mode = action.text()
        self._mode_label.setText(f"{mode} mode")
        mode_detail = (
            "guided workflow with compact choices"
            if mode == "Basic"
            else "expanded placeholders for expert configuration"
        )
        self.statusBar().showMessage(f"{mode} mode selected — {mode_detail}", 4000)
        self._project_explorer.set_mode(mode)

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Update status information when the visible workflow tab changes."""

        self._workflow_label.setText(f"Workflow: {self._workspace.tabText(index)}")

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
