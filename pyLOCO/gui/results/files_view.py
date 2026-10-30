"""Browse and export persisted artifacts from every FIT stage and iteration."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

CATEGORIES = ("Input files", "Fitted lattices", "ORM", "Optics", "Parameters",
              "Diagnostics", "Logs")


@dataclass(frozen=True)
class ArtifactRecord:
    stage: int
    stage_name: str
    iteration: int | None
    iteration_label: str
    category: str
    artifact: str
    path: Path | None
    status: str = "Saved"

    @property
    def stage_label(self):
        return f"Stage {self.stage} — {self.stage_name}"


class FilesView(QWidget):
    """Read-only file index; unavailable optional data are never recomputed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loader = None
        self._stages, self._records = [], []
        self.stage_selector = QComboBox(); self.stage_selector.setObjectName("filesStageSelector")
        self.iteration_selector = QComboBox(); self.iteration_selector.setObjectName("filesIterationSelector")
        self.category_selector = QComboBox(); self.category_selector.setObjectName("filesCategorySelector")
        self.category_selector.addItem("All categories", None)
        for category in CATEGORIES:
            self.category_selector.addItem(category, category)
        for selector in (self.stage_selector, self.iteration_selector, self.category_selector):
            selector.currentIndexChanged.connect(self._apply_filters)
        filters = QHBoxLayout()
        for label, selector in (("Stage:", self.stage_selector), ("Iteration:", self.iteration_selector),
                                ("Category:", self.category_selector)):
            filters.addWidget(QLabel(label)); filters.addWidget(selector)
        filters.addStretch(1)

        self.table = QTableWidget(0, 8); self.table.setObjectName("resultsArtifactTable")
        self.table.setHorizontalHeaderLabels([
            "Stage", "Iteration", "Category", "Artifact", "File", "Size", "Status", "Full path"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((180, 135, 115, 220, 190, 90, 155, 380)):
            self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self._update_actions)

        self.folder_button = QPushButton("Open containing folder")
        self.copy_button = QPushButton("Copy full path")
        self.export_lattice_button = QPushButton("Export selected fitted lattice…")
        self.export_scope_button = QPushButton("Export selected stage / iteration artifacts…")
        self.folder_button.clicked.connect(self._open_folder)
        self.copy_button.clicked.connect(self._copy)
        self.export_lattice_button.clicked.connect(self._export_lattice)
        self.export_scope_button.clicked.connect(self._export_scope)
        actions = QHBoxLayout()
        for button in (self.folder_button, self.copy_button, self.export_lattice_button,
                       self.export_scope_button):
            actions.addWidget(button)
        actions.addStretch(1)
        note = QLabel(
            "Files come from saved stage checkpoints and iteration snapshots. Stage-final and "
            "intermediate lattices are labelled separately; optional artifacts that were not "
            "saved are shown as unavailable and are not reconstructed.")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(note); layout.addLayout(filters); layout.addLayout(actions); layout.addWidget(self.table, 1)
        self._update_actions()

    @property
    def records(self):
        return list(self._records)

    def set_workflow_stages(self, stages):
        self._stages = [{"index": int(item["index"]), "name": str(item["name"]),
                         "path": Path(item["path"]).expanduser().resolve()} for item in stages]
        self._rebuild_inventory(False)

    def set_loader(self, loader):
        self.loader = loader
        root = loader.result_dir.resolve()
        if not self._stages or all(item["path"] != root for item in self._stages):
            self._stages = [{"index": 1, "name": "FIT run", "path": root}]
            self._rebuild_inventory(False)

    @staticmethod
    def _npz_has(path, key):
        try:
            with np.load(path, allow_pickle=False) as values:
                return key in values.files
        except (OSError, ValueError, TypeError):
            return False

    def _stage_records(self, stage):
        from .results_loader import ResultsLoader
        index, name, root = stage["index"], stage["name"], stage["path"]
        loader, records = ResultsLoader(root), []

        def add(category, artifact, path=None, iteration=None, label="Stage-level / final", status=None):
            value = Path(path).resolve() if path else None
            state = status or ("Saved" if value and value.exists() else "Referenced file unavailable")
            records.append(ArtifactRecord(index, name, iteration, label, category, artifact, value, state))

        for role, path in loader.input_files:
            add("Input files", role, path)
        direct = {path.name: path for path in root.iterdir() if path.is_file()} if root.is_dir() else {}
        lattice = next((direct[key] for key in ("final_lattice.mat", "ring_pyloco.mat") if key in direct), None)
        add("Fitted lattices", "Stage final fitted lattice", lattice,
            status="Saved stage-final lattice" if lattice else "Not saved")
        result_npz = direct.get("loco_results.npz")
        if result_npz:
            add("ORM", "Stage final modeled ORM", result_npz)
            add("Parameters", "Stage final parameter vector", result_npz)
        if direct.get("optics_results.npz"):
            add("Optics", "Stage final optics diagnostics", direct["optics_results.npz"])
        for filename, title in (("fit_dict.pkl", "Saved fitting parameters"),
                                ("fit_results.npy", "Saved fitted parameter vector"),
                                ("blocks.pkl", "Saved parameter blocks")):
            if filename in direct: add("Parameters", title, direct[filename])
        handled = {"final_lattice.mat", "ring_pyloco.mat", "loco_results.npz",
                   "optics_results.npz", "fit_dict.pkl", "fit_results.npy", "blocks.pkl"}
        for filename, path in direct.items():
            if filename not in handled:
                add("Logs" if path.suffix.lower() == ".log" else "Diagnostics", filename, path)
        # Include backend-specific saved artifacts in nested folders while
        # leaving iteration snapshots to the explicit iteration index below.
        for path in loader.generated_files:
            relative = path.relative_to(root)
            if len(relative.parts) == 1 or relative.parts[0] == "iterations":
                continue
            lowered = str(relative).lower()
            if "jacobian" in lowered:
                category = "Diagnostics"
            elif path.suffix.lower() == ".log":
                category = "Logs"
            elif "optic" in lowered:
                category = "Optics"
            elif "orm" in lowered or "response" in lowered:
                category = "ORM"
            else:
                category = "Diagnostics"
            add(category, str(relative), path)
        jacobian = next((path for path in loader.generated_files
                         if "jacobian" in str(path.relative_to(root)).lower()
                         and path.suffix.lower() in {".h5", ".hdf5", ".npz", ".npy"}), None)
        if not jacobian: add("Diagnostics", "Full Jacobian (optional)", status="Not saved (optional)")
        if result_npz and self._npz_has(result_npz, "singular_values"):
            add("Diagnostics", "Final SVD spectrum", result_npz)
        else: add("Diagnostics", "Final SVD spectrum (optional)", status="Not saved (optional)")

        entries = loader.iteration_entries
        if entries and not entries[0].get("legacy"):
            for entry in entries:
                number, label = int(entry["iteration"]), str(entry["label"])
                folder = root / "iterations" / f"iteration_{number:03d}"
                files = {path.name: path for path in folder.iterdir() if path.is_file()} if folder.is_dir() else {}
                add("Fitted lattices", "Iteration fitted lattice", files.get("fitted_lattice.mat"), number, label,
                    None if files.get("fitted_lattice.mat") else "Not saved")
                snapshot = files.get("loco_results.npz")
                if snapshot:
                    add("ORM", "Iteration modeled ORM", snapshot, number, label)
                    add("Parameters", "Iteration parameter vector", snapshot, number, label)
                add("Optics", "Iteration optics diagnostics", files.get("optics_results.npz"), number, label,
                    None if files.get("optics_results.npz") else "Not saved")
                add("Diagnostics", "Iteration metadata", files.get("iteration.json"), number, label,
                    None if files.get("iteration.json") else "Not saved")
                if snapshot and self._npz_has(snapshot, "singular_values"):
                    add("Diagnostics", "Iteration SVD spectrum", snapshot, number, label)
                else:
                    add("Diagnostics", "Iteration SVD spectrum (optional)", None, number, label, "Not saved (optional)")
        return records

    def _rebuild_inventory(self, preserve):
        old_stage = self.stage_selector.currentData() if preserve else None
        old_iteration = self.iteration_selector.currentData() if preserve else None
        self._records = [record for stage in self._stages for record in self._stage_records(stage)]
        self.stage_selector.blockSignals(True)
        self.stage_selector.clear(); self.stage_selector.addItem("All stages", None)
        for stage in self._stages:
            self.stage_selector.addItem(f"Stage {stage['index']} — {stage['name']}", stage["index"])
        found = self.stage_selector.findData(old_stage)
        self.stage_selector.setCurrentIndex(max(0, found)); self.stage_selector.blockSignals(False)
        self._rebuild_iterations(old_iteration); self._apply_filters()

    def _rebuild_iterations(self, preferred=None):
        stage = self.stage_selector.currentData()
        values = {(record.iteration, record.iteration_label) for record in self._records
                  if stage is None or record.stage == stage}
        self.iteration_selector.blockSignals(True)
        self.iteration_selector.clear(); self.iteration_selector.addItem("All iterations", "all")
        self.iteration_selector.addItem("Stage-level / final", "stage")
        for number, label in sorted((value for value in values if value[0] is not None)):
            self.iteration_selector.addItem(label, number)
        found = self.iteration_selector.findData(preferred)
        self.iteration_selector.setCurrentIndex(max(0, found)); self.iteration_selector.blockSignals(False)

    def _matches(self, record):
        stage, iteration, category = (self.stage_selector.currentData(),
                                      self.iteration_selector.currentData(), self.category_selector.currentData())
        return ((stage is None or record.stage == stage)
                and (iteration == "all" or (iteration == "stage" and record.iteration is None)
                     or record.iteration == iteration)
                and (category is None or record.category == category))

    def _apply_filters(self):
        if self.sender() is self.stage_selector: self._rebuild_iterations()
        visible = [record for record in self._records if self._matches(record)]
        self.table.setSortingEnabled(False); self.table.setRowCount(len(visible))
        for row, record in enumerate(visible):
            path = record.path
            values = (record.stage_label, record.iteration_label, record.category, record.artifact,
                      path.name if path else "Not available",
                      f"{path.stat().st_size:,} B" if path and path.exists() else "—",
                      record.status, str(path) if path else "Not available")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setToolTip(value); item.setData(Qt.UserRole, record)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True); self._update_actions()

    def _selected_record(self):
        item = self.table.item(self.table.currentRow(), 0) if self.table.currentRow() >= 0 else None
        value = item.data(Qt.UserRole) if item else None
        return value if isinstance(value, ArtifactRecord) else None

    def _update_actions(self):
        record = self._selected_record(); available = bool(record and record.path and record.path.exists())
        self.copy_button.setEnabled(available); self.folder_button.setEnabled(available)
        self.export_lattice_button.setEnabled(available and record.category == "Fitted lattices")
        self.export_scope_button.setEnabled(bool(self._records))

    def _copy(self):
        record = self._selected_record()
        if record and record.path: QApplication.clipboard().setText(str(record.path))

    def _open_folder(self):
        record = self._selected_record()
        if record and record.path and record.path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(record.path.parent)))

    def _export_lattice(self):
        record = self._selected_record()
        if not record or not record.path or not record.path.exists(): return
        target, _ = QFileDialog.getSaveFileName(self, "Export fitted lattice", record.path.name,
                                                 "MAT lattice (*.mat);;All files (*)")
        if target: shutil.copy2(record.path, Path(target))

    def _filtered_records(self):
        return [record for record in self._records if self._matches(record)
                and record.path and record.path.exists()]

    def export_filtered_records(self, destination):
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
        exported, seen = [], set()
        for record in self._filtered_records():
            key = (record.stage, record.iteration, str(record.path))
            if key in seen: continue
            seen.add(key)
            iteration = "stage-final" if record.iteration is None else f"iteration-{record.iteration:03d}"
            folder = destination / f"stage-{record.stage:02d}" / iteration / record.category.lower().replace(" ", "-")
            folder.mkdir(parents=True, exist_ok=True); target = folder / record.path.name
            shutil.copy2(record.path, target)
            exported.append({**asdict(record), "path": str(record.path),
                             "exported_path": str(target.relative_to(destination))})
        (destination / "export_manifest.json").write_text(
            json.dumps({"created": datetime.now().isoformat(), "artifacts": exported}, indent=2), encoding="utf-8")
        return destination

    def _export_scope(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose export destination")
        if not parent: return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = Path(parent) / f"pyloco-fit-artifacts-{stamp}"; counter = 1
        while destination.exists():
            destination = Path(parent) / f"pyloco-fit-artifacts-{stamp}-{counter}"; counter += 1
        try: self.export_filtered_records(destination)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc)); return
        QMessageBox.information(self, "Artifacts exported", f"Saved to:\n{destination}")
