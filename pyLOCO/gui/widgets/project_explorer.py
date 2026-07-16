"""Project Explorer dock for the Milestone 1 GUI shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
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

        self._tree = QTreeWidget()
        self._tree.setObjectName("projectExplorerTree")
        self._tree.setHeaderHidden(True)
        self.setWidget(self._tree)

        self._mode_item: QTreeWidgetItem | None = None
        self._populate_tree()
        self._tree.expandAll()

    def set_mode(self, mode: str) -> None:
        """Reflect the current Basic/Advanced mode in the explorer."""

        if self._mode_item is not None:
            self._mode_item.setText(0, f"Mode: {mode}")

    def _populate_tree(self) -> None:
        """Populate the static Milestone 1 project tree."""

        self._tree.clear()
        root = QTreeWidgetItem(["LOCO Project"])
        self._tree.addTopLevelItem(root)

        machine = self._add_group(root, "Machine Model")
        for label in ("Lattice", "BPMs", "Correctors", "RF Cavities", "Quadrupoles", "Skew/Tilt Elements"):
            self._add_leaf(machine, label)

        measurements = self._add_group(root, "Measurements")
        for label in ("ORM", "BPM Noise", "Dispersion", "Masks / Bad BPMs"):
            self._add_leaf(measurements, label)

        fit_setup = self._add_group(root, "Fit Setup")
        for label in ("Preset", "Fit Blocks", "Solver", "SVD", "Jacobians"):
            self._add_leaf(fit_setup, label)

        runs = self._add_group(root, "Runs")
        self._add_leaf(runs, "No runs yet")

        results = self._add_group(root, "Results")
        for label in ("Residuals", "Parameters", "Optics", "Exports"):
            self._add_leaf(results, label)

        plugins = self._add_group(root, "Plugins")
        for label in ("Machine Profile", "Importers", "Exporters"):
            self._add_leaf(plugins, label)

        self._mode_item = self._add_leaf(root, "Mode: Basic")

    @staticmethod
    def _add_group(parent: QTreeWidgetItem, label: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        parent.addChild(item)
        return item

    @staticmethod
    def _add_leaf(parent: QTreeWidgetItem, label: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        parent.addChild(item)
        return item
