"""Project Explorer dock for GUI project state."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QWidget

from ..models.project import ProjectMetadata, REQUIRED_MEASUREMENTS
from ..themes import configure_item_view


class ProjectExplorer(QDockWidget):
    """Dockable tree describing the current LOCO GUI project structure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Project Explorer", parent)
        self.setObjectName("projectExplorerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(300)

        self._tree = QTreeWidget()
        self._tree.setObjectName("projectExplorerTree")
        self._tree.setHeaderLabels(["Project item", "Status"])
        self._tree.setColumnWidth(0, 205)
        self._tree.setIndentation(18)
        self._tree.setRootIsDecorated(True)
        configure_item_view(self._tree)
        self.setWidget(self._tree)

        self._mode_item: QTreeWidgetItem | None = None
        self._result_loader = None
        self._tree.itemClicked.connect(self._navigate)
        self.update_project(ProjectMetadata())

    def set_result(self, loader) -> None:
        self._result_loader = loader

    def _navigate(self, item, _column) -> None:
        target = item.data(0, Qt.UserRole)
        if target:
            self.navigate_requested.emit(str(target))

    def set_mode(self, mode: str) -> None:
        """Reflect the current Basic/Advanced mode in the explorer."""

        if self._mode_item is not None:
            self._mode_item.setText(1, mode)

    def update_project(self, project: ProjectMetadata) -> None:
        """Rebuild the tree from the current project state."""

        self._tree.clear()
        root_status = "Complete" if project.is_complete else "Incomplete"
        if project.modified:
            root_status += " *"
        root = QTreeWidgetItem([project.name, root_status])
        self._style_item(root, bold=True)
        self._tree.addTopLevelItem(root)

        machine = self._add_group(
            root, "Machine", "Ready" if project.lattice.path else "Pending"
        )
        machine.setData(0, Qt.UserRole, "Machine")
        self._add_leaf(machine, "Lattice", project.lattice.name)
        self._add_leaf(machine, "File type", project.lattice.file_type or "Not loaded")
        count = (
            str(project.lattice.element_count)
            if project.lattice.element_count
            else "Unknown"
        )
        self._add_leaf(machine, "Elements", count)

        measurements = self._add_group(
            root, "Measurements", self._measurement_status(project)
        )
        measurements.setData(0, Qt.UserRole, "Measurements")
        for role in REQUIRED_MEASUREMENTS:
            dataset = project.measurements.get(role)
            label = role.replace("_", " ").title()
            self._add_leaf(
                measurements, label, dataset.name if dataset else "Not imported"
            )
        optional = sorted(set(project.measurements) - set(REQUIRED_MEASUREMENTS))
        for role in optional:
            dataset = project.measurements[role]
            self._add_leaf(measurements, role.replace("_", " ").title(), dataset.name)

        selected = project.loco_config.parameters.fit_list()
        fit = self._add_group(root, "Fit", "Configured" if selected else "Pending")
        fit.setData(0, Qt.UserRole, "Fit")
        self._add_leaf(fit, "Selected blocks", ", ".join(selected) if selected else "None")
        self._add_leaf(fit, "Solver", project.loco_config.solver.algorithm.upper())
        self._add_leaf(fit, "Iterations", str(project.loco_config.solver.nIter))
        included = project.loco_config.rejection.includeDispersion or project.loco_config.response_matrix.includeDispersion
        self._add_leaf(fit, "Dispersion", "Included" if included else "Excluded")

        if self._result_loader is not None:
            loader = self._result_loader
            results = self._add_group(root, "Results", "Completed")
            before = "—" if loader.initial_chi2 is None else f"{loader.initial_chi2:.3e}"
            after = "—" if loader.final_chi2 is None else f"{loader.final_chi2:.3e}"
            chi2 = f"χ² {before} → {after}"
            for label in ("Overview", "ORM", "Optics", "Parameters", "Jacobian/SVD", "Files", "Log"):
                item = self._add_leaf(results, label, chi2 if label == "Overview" else "")
                item.setData(0, Qt.UserRole, f"Results:{label}")

        validation = self._add_group(
            root, "Validation", "Ready" if project.is_complete else "Needs input"
        )
        validation.setData(0, Qt.UserRole, "Fit")
        messages = project.validation_messages() or [
            "Project is complete; Run LOCO is enabled."
        ]
        for message in messages:
            self._add_leaf(validation, message, "")

        recent = self._add_group(
            root, "Recent Projects", str(len(project.recent_projects))
        )
        for path in project.recent_projects or ["No recent projects"]:
            self._add_leaf(recent, path, "")

        self._mode_item = self._add_leaf(root, "Workflow Mode", project.mode)
        self._tree.expandAll()

    def _measurement_status(self, project: ProjectMetadata) -> str:
        imported = sum(
            1 for role in REQUIRED_MEASUREMENTS if role in project.measurements
        )
        return f"{imported}/{len(REQUIRED_MEASUREMENTS)} required"

    def _add_group(
        self, parent: QTreeWidgetItem, label: str, status: str
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, status])
        self._style_item(item, bold=True)
        parent.addChild(item)
        return item

    def _add_leaf(
        self, parent: QTreeWidgetItem, label: str, status: str
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label, status])
        self._style_item(item, bold=False)
        parent.addChild(item)
        return item

    @staticmethod
    def _style_item(item: QTreeWidgetItem, *, bold: bool) -> None:
        font = QFont()
        font.setBold(bold)
        for column in range(2):
            item.setFont(column, font)
    navigate_requested = Signal(str)
