"""Project Explorer dock for the Milestone 1 GUI shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QWidget


class ProjectExplorer(QDockWidget):
    """Dockable tree describing the planned LOCO project structure.

    The tree is static in Milestone 1. Later milestones should connect these
    nodes to project state, validation badges, runs, and plugin metadata.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Project Explorer", parent)
        self.setObjectName("projectExplorerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(280)

        self._tree = QTreeWidget()
        self._tree.setObjectName("projectExplorerTree")
        self._tree.setHeaderLabels(["Project item", "Status"])
        self._tree.setColumnWidth(0, 190)
        self._tree.setAlternatingRowColors(False)
        self._tree.setIndentation(18)
        self._tree.setRootIsDecorated(True)
        self.setWidget(self._tree)

        self._mode_item: QTreeWidgetItem | None = None
        self._populate_tree()
        self._tree.expandAll()

    def set_mode(self, mode: str) -> None:
        """Reflect the current Basic/Advanced mode in the explorer."""

        if self._mode_item is not None:
            self._mode_item.setText(1, mode)

    def _populate_tree(self) -> None:
        """Populate the static Milestone 1 project tree."""

        self._tree.clear()
        root = QTreeWidgetItem(["LOCO Project", "Draft"])
        self._style_item(root, bold=True, color="#12365f")
        self._tree.addTopLevelItem(root)

        machine = self._add_group(root, "Machine Model", "Pending")
        for label in (
            "Lattice",
            "BPMs",
            "Correctors",
            "RF Cavities",
            "Quadrupoles",
            "Skew/Tilt Elements",
        ):
            self._add_leaf(machine, label, "Not loaded")

        measurements = self._add_group(root, "Measurements", "Pending")
        for label in ("ORM", "BPM Noise", "Dispersion", "Masks / Bad BPMs"):
            self._add_leaf(measurements, label, "Not imported")

        fit_setup = self._add_group(root, "Fit Setup", "Draft")
        for label in ("Preset", "Fit Blocks", "Solver", "SVD", "Jacobians"):
            self._add_leaf(fit_setup, label, "Placeholder")

        runs = self._add_group(root, "Runs", "Idle")
        self._add_leaf(runs, "No runs yet", "Milestone 1")

        results = self._add_group(root, "Results", "Empty")
        for label in ("Residuals", "Parameters", "Optics", "Exports"):
            self._add_leaf(results, label, "Unavailable")

        plugins = self._add_group(root, "Plugins", "Future")
        for label in ("Machine Profile", "Importers", "Exporters"):
            self._add_leaf(plugins, label, "Not configured")

        self._mode_item = self._add_leaf(root, "Workflow Mode", "Basic")

    def _add_group(
        self, parent: QTreeWidgetItem, label: str, status: str
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, status])
        self._style_item(item, bold=True, color="#1b426d")
        parent.addChild(item)
        return item

    def _add_leaf(
        self, parent: QTreeWidgetItem, label: str, status: str
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, status])
        self._style_item(item, bold=False, color="#40576e")
        parent.addChild(item)
        return item

    @staticmethod
    def _style_item(item: QTreeWidgetItem, *, bold: bool, color: str) -> None:
        font = QFont()
        font.setBold(bold)
        for column in range(2):
            item.setFont(column, font)
            item.setForeground(column, QBrush(QColor(color)))
