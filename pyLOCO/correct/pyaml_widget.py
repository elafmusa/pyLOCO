"""pyAML configuration, identity mapping and correction transaction workspace."""
from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .pyaml_connection import (PyAMLCorrectionConnection, load_mapping,
                               resolve_mapping)
from .pyaml_transaction import PyAMLCorrectionTransaction


class TransactionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, action):
        super().__init__()
        self.action = action

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.action())
        except Exception as exc:
            self.failed.emit(str(exc))


class PyAMLCorrectionWidget(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.connection = None
        self.transaction = None
        self.explicit = []
        self.mapping_path = None
        self.thread = None
        self._rendered = None
        self.cancel = Event()
        layout = QVBoxLayout(self)
        self.settings_box = QWidget()
        form = QFormLayout(self.settings_box)
        self.fields = {}
        settings = QSettings("pyLOCO", "pyLOCO Correct")
        for key, label, default in (("config_path", "Configuration file", ""),
                                    ("normal_array", "Normal quadrupole array", "Quads"),
                                    ("skew_array", "Skew quadrupole array", "SkewQuads")):
            editor = QLineEdit(str(settings.value(f"correct/pyaml/{key}", default)))
            self.fields[key] = editor
            if key == "config_path":
                row = QWidget(); horizontal = QHBoxLayout(row)
                horizontal.setContentsMargins(0, 0, 0, 0)
                horizontal.addWidget(editor)
                self.browse_button = QPushButton("Browse…")
                self.browse_button.clicked.connect(self.browse)
                horizontal.addWidget(self.browse_button)
                form.addRow(label, row)
            else:
                form.addRow(label, editor)
            editor.textChanged.connect(self.invalidate_connection)
        self.mode = QComboBox(); self.mode.addItems(["live", "design"])
        self.mode.setCurrentText(str(settings.value("correct/pyaml/mode", "live")))
        self.mode.currentTextChanged.connect(self.invalidate_connection)
        form.addRow("Mode", self.mode)
        self.writes = QCheckBox("Enable magnet writes"); self.writes.setChecked(True)
        self.writes.toggled.connect(self.invalidate_connection)
        form.addRow(self.writes)
        self.connect_button = QPushButton("Connect / discover quadrupoles")
        self.connect_button.clicked.connect(lambda: self.run(self.connect_machine))
        form.addRow(self.connect_button)
        layout.addWidget(self.settings_box)
        self.mapping_box = owner.mapping_group
        self.mapping_label = owner.mapping_file_status
        self.clear_mapping_button = QPushButton("Use automatic identity matching")
        self.clear_mapping_button.clicked.connect(lambda: self.run(self.clear_mapping))
        owner.mapping_group.layout().addWidget(self.clear_mapping_button, 8, 0, 1, 2)
        self.clear_mapping_button.hide()
        self.table = owner.table
        self.extra_actions = QWidget()
        self.extra_actions.hide()
        actions = QHBoxLayout(self.extra_actions)
        self.preview_button = owner.preview_apply_button
        self.apply_button = owner.apply_button
        self.restore_button = owner.restore_button
        self.recover_button = QPushButton("Recover journal…")
        self.cancel_button = QPushButton("Cancel / restore")
        for button in (self.recover_button, self.cancel_button):
            actions.addWidget(button)
        self.recover_button.clicked.connect(lambda: self.run(self.recover))
        self.cancel_button.clicked.connect(self.cancel.set)
        self.status = owner.apply_status
        self.timer = QTimer(self); self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh); self.timer.start()
        self.refresh()

    def pending(self):
        return bool(self.transaction and self.transaction.pending)

    def options(self):
        return {**{key: editor.text().strip() for key, editor in self.fields.items()}, "mode": self.mode.currentText()}

    def invalidate_connection(self, *_):
        if self.pending() or self.thread:
            return
        self.connection = self.transaction = None
        if hasattr(self, "status"):
            self.status.setText("Configuration changed — reconnect")
            self.owner._set_connection(False, "DISCONNECTED")
            self.refresh()

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pyAML configuration", self.fields["config_path"].text(),
            "pyAML configuration (*.yaml *.yml *.json);;All files (*)", options=QFileDialog.DontUseNativeDialog)
        if path:
            self.fields["config_path"].setText(path)

    def connect_machine(self):
        if self.pending() or self.thread:
            raise RuntimeError("Restore the current transaction before reconnecting")
        if self.owner.backend_combo.currentData() != "pyaml":
            raise ValueError("Select the pyAML backend")
        self.connection = self.transaction = None
        self.owner._set_connection(False, "DISCONNECTED")
        options = self.options()
        if options["mode"] == "live" and self.writes.isChecked():
            if QMessageBox.question(self, "Enable live magnet writes?",
                "Confirm the selected configuration and operating permission. Apply may change normal and skew quadrupoles. "
                "Machine limits are enforced by your control system, not this GUI. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        connection = PyAMLCorrectionConnection(**options, allow_writes=self.writes.isChecked())
        for row in connection.inventory:
            connection.read(row["name"])
        self.connection = connection
        self.transaction = PyAMLCorrectionTransaction(connection, Path.cwd() / "correction-transactions" / "pyaml")
        settings = QSettings("pyLOCO", "pyLOCO Correct")
        for key, value in options.items():
            settings.setValue(f"correct/pyaml/{key}", value)
        self.owner._set_connection(True, "CONNECTED")
        self.owner.badge.setText(f"pyAML • {options['mode'].upper()} • {'WRITE' if connection.allow_writes else 'READ ONLY'}")
        self.status.setText(f"Connected: {len(connection.inventory)} quadrupoles. No writes performed.")

    def import_mapping(self):
        self.owner.load_petra_mapping()

    def load_mapping_path(self, path):
        if self.pending():
            raise RuntimeError("Restore before changing mapping")
        if path:
            entries = load_mapping(path)
            if self.owner.review and self.connection:
                resolve_mapping(self.owner.review, self.connection.inventory, entries)
            self.explicit = entries
            self.mapping_path = Path(path)
            self.owner.mapping_path = str(self.mapping_path.resolve())
            self.mapping_label.setText(str(path))
            self._rendered = None
            self.refresh()

    def clear_mapping(self):
        if self.pending():
            raise RuntimeError("Restore before changing mapping")
        self.explicit = []; self.mapping_path = None
        self.owner.mapping_path = None
        self.mapping_label.setText("Automatic exact lattice-identity matching")
        self._rendered = None
        self.refresh()

    def export_mapping(self):
        self.owner.generate_mapping_template()

    def read_current(self):
        if not self.connection or not self.owner.review:
            raise ValueError("Connect and load a correction source first")
        self.refresh()
        for item in self.owner.review.items:
            if item.control_name:
                item.machine_value = self.connection.read(item.control_name)
        self.owner.refresh_all()
        self.owner.current_k_status.setText("Read individual integrated strengths; no writes performed.")

    def current_mapping(self):
        if self.mapping_path and load_mapping(self.mapping_path) != self.explicit:
            raise RuntimeError("Mapping file changed; reload it before preview/apply")
        return self.explicit

    def preview(self):
        if not self.transaction or not self.owner.review:
            raise ValueError("Connect and load a correction source first")
        if self.owner._source_load_thread is not None:
            raise RuntimeError("Wait for correction source loading to finish")
        self.transaction.mapping_path = self.mapping_path
        record = self.transaction.preview(self.owner.review, self.current_mapping())
        self.status.setText(f"Preview verified: {record['fraction'] * 100:.6g}% cumulative; {len(record['changes'])} magnets. No writes.")

    def apply(self):
        if not self.transaction or not self.owner.review or self.thread:
            return
        if QMessageBox.question(self, "Apply pyAML quadrupole correction?",
            "Apply the displayed integrated-strength targets to the connected machine? Restoration can fail; inspect all results.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            explicit = self.current_mapping()
        except Exception as exc:
            self.error(str(exc)); return
        self.cancel = Event()
        self.start(lambda: self.transaction.apply(self.owner.review, explicit, confirmed=True, cancel=self.cancel))

    def restore(self):
        if self.pending() and not self.thread:
            if QMessageBox.question(self, "Restore original strengths?", "Write and verify every original pre-correction strength?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self.start(self.transaction.restore)

    def recover(self):
        if not self.transaction:
            raise ValueError("Reconnect to the matching configuration before recovery")
        path, _ = QFileDialog.getOpenFileName(self, "Recover pyAML journal", "", "JSON (*.json)")
        if path:
            self.transaction.recover(path)
            self.status.setText("Recovery loaded without writes. Use Restore original strengths.")

    def run(self, action):
        try:
            action()
        except Exception as exc:
            self.error(str(exc))
        self.refresh()

    def error(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "pyAML correction", message)

    def start(self, action):
        self.thread = QThread(self)
        self.worker = TransactionWorker(action); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.completed)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.finished)
        self.refresh(); self.thread.start()

    def completed(self, record):
        self.status.setText(f"{record['status']} — {record['fraction'] * 100:.6g}% cumulative. Journal: {self.transaction.path}")

    def finished(self):
        self.thread.deleteLater(); self.thread = None; self.worker = None
        self.refresh()

    def refresh(self):
        active = self.owner.backend_combo.currentData() == "pyaml"
        if not active:
            return
        pending = self.pending()
        busy = self.thread is not None
        locked = pending or busy
        self.settings_box.setEnabled(not locked)
        self.mapping_box.setEnabled(not locked)
        if active:
            self.owner.backend_combo.setEnabled(not locked)
            if self.connection:
                mode = self.connection.options["mode"].upper()
                access = "WRITE" if self.connection.allow_writes else "READ ONLY"
                self.owner.badge.setText(f"pyAML • {mode} • {access}")
            for index in range(3):
                self.owner.tabs.setTabEnabled(index, not locked)
        self.preview_button.setEnabled(active and self.connection is not None and not busy)
        self.restore_button.setEnabled(pending and not busy)
        self.recover_button.setEnabled(active and self.connection is not None and self.connection.allow_writes and not locked)
        self.cancel_button.setEnabled(busy)
        record = self.transaction.record if self.transaction else None
        if busy:
            self.apply_button.setEnabled(False)
            return
        if not self.connection or (not self.owner.review and not record):
            self.apply_button.setEnabled(False)
            self._rendered = None
            return
        can_apply = False
        try:
            rows = (resolve_mapping(self.owner.review, self.connection.inventory, self.explicit)
                    if self.owner.review else record["identity"]["review"]["rows"])
            if not pending:
                self.current_mapping()
            if record and record["status"] == "preview" and self.owner.review:
                if (rows != record["identity"]["review"]["rows"] or self.explicit != record["identity"]["explicit"]
                    or str(Path(self.owner.review.source_result).resolve()) != record["identity"]["review"]["source"]
                    or self.owner.review.global_scale != record["identity"]["review"]["global_scale"]):
                    record["status"] = "stale"
                    self.status.setText("Correction or mapping changed — fresh preview required (restore first if pending).")
                else:
                    can_apply = self.connection.allow_writes and self.owner._source_load_thread is None
            changes = {row["name"]: row for row in record["changes"]} if record else {}
            rendered = repr((rows, changes))
            if rendered == self._rendered:
                return
            self._rendered = rendered
            if self.owner.review:
                for item, row in zip(self.owner.review.items, rows):
                    item.control_name = row["magnet_name"] or None
                    item.metadata["mapping_status"] = row["status"]
                    item.metadata["match_source"] = row["match_source"]
                    item.metadata["control_component"] = "B2L" if item.correction_type == "normal_quadrupole" else "A2L"
                    item.metadata["control_unit"] = "m^-1"
                self.owner._sync_representation()
                self.owner._refresh_mapping_status()
                self.owner.refresh_all()
            from .application import CorrectionChange
            self.owner.correction_changes = tuple(CorrectionChange(
                change["name"], change["name"], change["current"], change["target"],
                change.get("readback"), change["status"]) for change in changes.values())
            self.owner._populate_machine_changes(self.owner.correction_changes)
        except Exception as exc:
            self._rendered = None
            can_apply = False
            self.status.setText(str(exc))
        finally:
            self.apply_button.setEnabled(can_apply)
