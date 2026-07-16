"""Project Explorer dock for the Milestone 1 GUI shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QStyle, QWidget


class ProjectExplorer(QDockWidget):
    """Dockable tree describing the planned LOCO project structure.

    The tree is static in Milestone 1. Later milestones should connect these
    nodes to project state, validation badges, runs, and plugin metadata.
    Basic mode hides advanced-oriented groups while Advanced mode restores the
    complete project tree.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Project Explorer", parent)
        self.setObjectName("projectExplorerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._tree = QTreeWidget()
        self._tree.setObjectName("projectExplorerTree")
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(18)
        self._tree.setAnimated(True)
        self.setWidget(self._tree)

        self._mode_item: QTreeWidgetItem | None = None
        self._advanced_items: list[QTreeWidgetItem] = []
        self._populate_tree()
        self._tree.expandAll()

    def set_mode(self, mode: str) -> None:
        """Reflect the current Basic/Advanced mode in the explorer."""

        if self._mode_item is not None:
            self._mode_item.setText(0, f"Mode: {mode}")
        show_advanced = mode == "Advanced"
        for item in self._advanced_items:
            item.setHidden(not show_advanced)

    def _populate_tree(self) -> None:
        """Populate the static Milestone 1 project tree."""

        self._tree.clear()
        self._advanced_items.clear()
        root = self._make_item("LOCO Project", QStyle.SP_ComputerIcon)
        self._tree.addTopLevelItem(root)

        machine = self._add_group(root, "Machine")
        for label in ("Lattice", "BPMs", "Correctors", "RF Cavities", "Quadrupoles"):
            self._add_leaf(machine, label)
        self._advanced_items.append(self._add_leaf(machine, "Skew / Tilt Elements"))

        measurements = self._add_group(root, "Measurements")
        for label in ("Orbit Response Matrix", "BPM Noise", "Dispersion"):
            self._add_leaf(measurements, label)
        self._advanced_items.append(self._add_leaf(measurements, "Masks / Bad BPMs"))

        fit_setup = self._add_group(root, "Fit Setup")
        self._add_leaf(fit_setup, "Preset")
        self._add_leaf(fit_setup, "Fit Blocks")
        for label in ("Solver", "SVD", "Jacobians"):
            self._advanced_items.append(self._add_leaf(fit_setup, label))

        runs = self._add_group(root, "Runs")
        self._add_leaf(runs, "No runs yet")

        results = self._add_group(root, "Results")
        for label in ("Residuals", "Parameters", "Exports"):
            self._add_leaf(results, label)
        self._advanced_items.append(self._add_leaf(results, "Optics"))

        plugins = self._add_group(root, "Extensions")
        self._advanced_items.append(plugins)
        for label in ("Machine Profile", "Importers", "Exporters"):
            self._add_leaf(plugins, label)

        self._mode_item = self._add_leaf(root, "Mode: Basic")

    def _add_group(self, parent: QTreeWidgetItem, label: str) -> QTreeWidgetItem:
        item = self._make_item(label, QStyle.SP_DirIcon)
        parent.addChild(item)
        return item

    def _add_leaf(self, parent: QTreeWidgetItem, label: str) -> QTreeWidgetItem:
        item = self._make_item(label, QStyle.SP_FileIcon)
        parent.addChild(item)
        return item

    def _make_item(self, label: str, standard_icon: QStyle.StandardPixmap) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        item.setIcon(0, self.style().standardIcon(standard_icon))
        return item
