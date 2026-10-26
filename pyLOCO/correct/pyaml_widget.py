"""pyAML configuration, identity mapping and correction transaction workspace."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)
from pyLOCO.gui.file_dialogs import FileDialog as QFileDialog

from .pyaml_connection import (PyAMLCorrectionConnection, load_mapping,
    resolve_mapping, approved_mapping_candidates, find_reference_lattice,
    file_hash, make_mapping_bundle, validate_mapping_bundle,
    validate_mapping_document)
from pyLOCO.machine_mapping import (is_document as is_shared_machine_mapping,
    load as load_shared_mapping, merge_corrections, correction_projection,
    mapping_statistics as machine_mapping_statistics)
from .petra_readonly import MagnetMapping, apply_explicit_mapping
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
        self.mapping_hash = None
        self.thread = None
        self._rendered = None
        self._embedded_mapping_error = None
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
        self.writes = QCheckBox("Authorize magnet writes"); self.writes.setChecked(False)
        self.writes.toggled.connect(self.invalidate_connection)
        form.addRow(self.writes)
        connection_help=QLabel("This backend establishes the configured machine connection and discovers its quadrupole inventory in one safe operation."); connection_help.setWordWrap(True); form.addRow(connection_help)
        self.connect_button = QPushButton("Test connection and discover devices")
        self.connect_button.clicked.connect(lambda: self.run(self.connect_machine))
        form.addRow(self.connect_button)
        self.connection_progress=QProgressBar(); self.connection_progress.setRange(0,0)
        self.connection_progress.setFormat("Testing connection and reading inventory…")
        self.connection_progress.setVisible(False); form.addRow(self.connection_progress)
        self.connection_result=QLabel("Disconnected"); self.inventory_result=QLabel("Inventory not discovered"); self.authorization_result=QLabel("READ ONLY — writes not authorized")
        form.addRow("Connection",self.connection_result); form.addRow("Device inventory",self.inventory_result); form.addRow("Authorization",self.authorization_result)
        layout.addWidget(self.settings_box)
        self.mapping_box = owner.mapping_group
        self.mapping_label = owner.mapping_file_status
        self.clear_mapping_button = QPushButton("Find machine mappings")
        self.clear_mapping_button.clicked.connect(lambda: self.run(self.clear_mapping))
        owner.automatic_mapping_actions.insertWidget(0, self.clear_mapping_button)
        self.clear_mapping_button.hide()
        self.table = owner.table
        self.extra_actions = QWidget()
        self.extra_actions.hide()
        actions = QHBoxLayout(self.extra_actions)
        self.preview_button = owner.preview_apply_button
        self.apply_button = owner.apply_button
        self.restore_button = owner.restore_button
        self.recover_button = QPushButton("Restore after interrupted application…")
        self.recover_button.setToolTip(
            "Select and validate a saved JSON recovery record. Loading performs no writes; "
            "restoration requires the matching connection, write authorization and confirmation."
        )
        self.cancel_button = QPushButton("Stop application and restore originals")
        self.cancel_button.setToolTip(
            "Request cancellation between magnet operations and attempt restoration of every "
            "device already affected. Restoration is not guaranteed if connection or hardware operations fail."
        )
        for button in (self.recover_button, self.cancel_button):
            actions.addWidget(button)
        self.recover_button.clicked.connect(lambda: self.run(self.recover))
        self.cancel_button.clicked.connect(self.request_cancel)
        self.status = owner.apply_status
        self.timer = QTimer(self); self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh); self.timer.start()
        self.refresh()

    @staticmethod
    def _saved_connection_options(review):
        """Return non-sensitive pyAML options retained by Measure/FIT.

        These values only prefill the disconnected connection form.  They do
        not create a connection, validate a mapping, or authorize writes.
        """
        mapping = getattr(review, "machine_mapping", {}) or {}
        profile = mapping.get("machine_profile", {}) if isinstance(mapping, dict) else {}
        options = {
            "config_path": str(profile.get("configuration_path") or ""),
            "normal_array": str(profile.get("normal_array") or ""),
            "skew_array": str(profile.get("skew_array") or ""),
            "mode": str(profile.get("mode") or ""),
        }
        session = {}
        if getattr(review, "items", None):
            session = review.items[0].metadata.get("measurement_session") or {}
        full = (session.get("session_metadata") or {}).get("full_measure_session") or {}
        configuration = full.get("pyaml_configuration") or {}
        if not options["config_path"]:
            options["config_path"] = str(
                configuration.get("path") or configuration.get("external_path") or ""
            )
        return options

    @classmethod
    def has_saved_pyaml_provenance(cls, review):
        mapping = getattr(review, "machine_mapping", {}) or {}
        profile = mapping.get("machine_profile", {}) if isinstance(mapping, dict) else {}
        if str(profile.get("backend") or "").lower() == "pyaml":
            return True
        if profile.get("configuration_sha256") or profile.get("configuration_path"):
            return True
        if any(row.get("pyaml_device") or row.get("control_name")
               for row in mapping.get("mappings", ()) if isinstance(row, dict)):
            return True
        return bool(cls._saved_connection_options(review).get("config_path"))

    def apply_saved_provenance(self, review):
        """Prepare a safe disconnected pyAML form from retained provenance."""
        options = self._saved_connection_options(review)
        config_path = options.get("config_path", "")
        valid_config = bool(config_path and Path(config_path).expanduser().is_file())
        values = {
            "config_path": str(Path(config_path).expanduser().resolve()) if valid_config else "",
            "normal_array": options.get("normal_array") or "Quads",
            "skew_array": options.get("skew_array") or "SkewQuads",
        }
        for key, value in values.items():
            editor = self.fields[key]
            editor.blockSignals(True); editor.setText(value); editor.blockSignals(False)
        mode = options.get("mode") if options.get("mode") in {"live", "design"} else "live"
        self.mode.blockSignals(True); self.mode.setCurrentText(mode); self.mode.blockSignals(False)
        self.writes.blockSignals(True); self.writes.setChecked(False); self.writes.blockSignals(False)
        self.connection = self.transaction = None
        self.connection_result.setText("Disconnected")
        self.inventory_result.setText("Inventory not discovered")
        self.authorization_result.setText("READ ONLY — writes not authorized")
        if valid_config:
            self.owner.connection_guidance.setText(
                "Saved pyAML settings restored. Test the connection to discover the current device inventory; "
                "the saved mapping remains unverified until that check succeeds."
            )
        else:
            missing = f" Saved path is unavailable: {config_path}" if config_path else ""
            self.owner.connection_guidance.setText(
                "Select the approved pyAML configuration, confirm the arrays and mode, then test the "
                f"connection. No connection or write authorization has been inherited.{missing}"
            )

    def pending(self):
        return bool(self.transaction and self.transaction.pending)

    def options(self):
        return {**{key: editor.text().strip() for key, editor in self.fields.items()}, "mode": self.mode.currentText()}

    def invalidate_connection(self, *_):
        if self.pending() or self.thread:
            return
        self.connection = self.transaction = None
        if hasattr(self, "status"):
            if self.owner.review:
                embedded = getattr(self.owner.review, "machine_mapping", {})
                if is_shared_machine_mapping(embedded):
                    entries = [MagnetMapping(
                        row.get("lattice_name", ""), row.get("control_name", ""),
                        row.get("lattice_ordinal"), row.get("component"), row.get("unit"))
                        for row in correction_projection(embedded) if row.get("control_name")]
                    apply_explicit_mapping(self.owner.review, entries)
                    for item in self.owner.review.items:
                        if item.metadata.get("mapping_status") == "mapped":
                            item.metadata["mapping_status"] = "saved"
                            item.metadata["match_source"] = "saved_machine_mapping"
                            item.metadata["mapping_validation_level"] = "saved_requires_connection"
                else:
                    for item in self.owner.review.items:
                        item.control_name = None
                        item.metadata["mapping_status"] = "unmapped"
                        item.metadata.pop("match_source", None)
            self.connection_result.setText("Disconnected — configuration changed")
            self.inventory_result.setText("Inventory not discovered")
            self.authorization_result.setText("WRITE AUTHORIZED" if self.writes.isChecked() else "READ ONLY — writes not authorized")
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
        saved_record=getattr(self.owner,"saved_progress_record",None)
        saved_progress_message=None
        if saved_record:
            saved_identity=(saved_record.get("identity") or {}).get("connection")
            if saved_identity == connection.identity():
                self.transaction.record=copy.deepcopy(saved_record)
                self.transaction.persist()
                self.owner.saved_progress_record=None
                saved_progress_message="Saved correction history restored after machine-identity verification. No writes performed."
            else:
                saved_progress_message=(
                    "Connected, but saved correction progress belongs to a different machine identity. "
                    "It remains unverified and cannot authorize the next step."
                )
        settings = QSettings("pyLOCO", "pyLOCO Correct")
        for key, value in options.items():
            settings.setValue(f"correct/pyaml/{key}", value)
        self.owner._set_connection(True, "CONNECTED")
        self.owner.badge.setText(f"pyAML • {options['mode'].upper()} • {'WRITE' if connection.allow_writes else 'READ ONLY'}")
        self.connection_result.setText(
            "Connected — writes authorized" if connection.allow_writes else "Connected — read only"
        )
        self.inventory_result.setText(
            f"Inventory verified — {len(connection.inventory)} quadrupole devices"
        )
        self.authorization_result.setText("WRITE AUTHORIZED" if connection.allow_writes else "READ ONLY — writes not authorized")
        self.status.setText(f"Connected: {len(connection.inventory)} quadrupoles. No writes performed.")
        embedded = getattr(self.owner.review, "machine_mapping", {}) if self.owner.review else {}
        if is_shared_machine_mapping(embedded):
            try:
                lattice = find_reference_lattice(self.owner.review, connection)
                entries, _rows, provenance, _identity = validate_mapping_document(
                    self.owner.review, connection, embedded, lattice)
                self.explicit = entries
                self._embedded_mapping_error = None
                self.mapping_hash = None
                self.owner.manual_mapping_feedback.setText(
                    f"Embedded Complete Machine Mapping revalidated for this connection — "
                    f"{len(entries)} fitted magnet(s), {provenance}.")
                self.mapping_label.setText("Embedded FIT Machine Mapping — revalidated")
                self.owner.connection_guidance.setText(
                    f"Inventory verified. Mapping verified for {len(entries)} fitted magnet(s). "
                    "Magnet writes remain disabled unless separately authorized."
                )
            except Exception as exc:
                self.explicit = []
                self._embedded_mapping_error = str(exc)
                self.mapping_label.setText("Embedded FIT Machine Mapping — revalidation failed")
                self.owner.manual_mapping_feedback.setText(
                    f"Saved mapping retained for review but is not valid for this connection: {exc}")
        elif self.mapping_path:
            try:
                lattice = find_reference_lattice(self.owner.review, connection)
                entries, _rows, provenance, _identity = validate_mapping_bundle(
                    self.owner.review, connection, self.mapping_path, lattice
                )
                self.explicit = entries
                self.mapping_hash = file_hash(self.mapping_path)
                self.owner.manual_mapping_feedback.setText(
                    f"Saved mapping revalidated for this connection — {len(entries)} magnet(s), {provenance}."
                )
            except Exception as exc:
                self.explicit = []; self.mapping_hash = None
                self.mapping_path = None; self.owner.mapping_path = None
                self.mapping_label.setText("Saved mapping invalidated")
                self.owner.manual_mapping_feedback.setText(
                    f"Previously selected mapping is not valid for this connection: {exc}"
                )
        if saved_progress_message:
            self.status.setText(saved_progress_message)
        self.owner._refresh_correction_progress()

    def import_mapping(self):
        self.owner.load_petra_mapping()

    def load_mapping_path(self, path):
        if self.pending():
            raise RuntimeError("Restore before changing mapping")
        if path:
            complete=None
            try:
                candidate=load_shared_mapping(path)
                if is_shared_machine_mapping(candidate):complete=candidate
            except Exception:pass
            entries = load_mapping(path)
            provenance = "explicit"
            if self.owner.review and self.connection:
                lattice = find_reference_lattice(self.owner.review, self.connection)
                entries, _rows, provenance, _identity = validate_mapping_bundle(
                    self.owner.review, self.connection, path, lattice,
                    operator_selected=True,
                )
            self.explicit = entries
            if complete is not None and self.owner.review is not None:
                self.owner.review.machine_mapping=complete
            self._embedded_mapping_error = None
            self.mapping_path = Path(path)
            self.mapping_hash = file_hash(self.mapping_path)
            self.owner.mapping_path = str(self.mapping_path.resolve())
            self.mapping_label.setText(str(path))
            self.owner.mapping_action_message = "loaded"
            complete_counts=machine_mapping_statistics(complete) if complete is not None else None
            complete_text=(f" Complete Machine Mapping retained: {complete_counts['total']:,} entries, "
                           f"{complete_counts['mapped']:,} mapped." if complete_counts else "")
            self.owner.manual_mapping_feedback.setText(
                (f"Mapping loaded and validated — {len(entries)} fitted magnet(s), {provenance}."+complete_text
                 if self.connection else
                 f"Mapping loaded — {len(entries)} fitted magnet proposal(s); connect to revalidate."+complete_text)
            )
            self.owner.automatic_mapping_feedback.setText(
                "Not run — a validated manual mapping is active." if self.connection else
                "Not run — a saved mapping proposal is loaded pending connection revalidation.")
            QSettings("pyLOCO", "pyLOCO Correct").setValue(
                "correct/pyaml/approved_mapping_path", str(self.mapping_path.resolve())
            )
            self._rendered = None
            self.refresh()

    def clear_mapping(self):
        if self.pending():
            raise RuntimeError("Restore before changing mapping")
        self.explicit = []; self.mapping_path = None; self.mapping_hash = None; self.owner.mapping_path = None
        self._embedded_mapping_error = None
        self.owner.mapping_action_message = "automatic"
        # Prefer the Complete Machine Mapping carried by FIT. Its explicit AT
        # ordinals identify the fitted sources; resolve_mapping still verifies
        # every target against the currently connected pyAML inventory.
        embedded = getattr(self.owner.review, "machine_mapping", {}) if self.owner.review else {}
        if is_shared_machine_mapping(embedded) and self.owner.review and self.connection:
            proposed = correction_projection(embedded)
            try:
                rows = resolve_mapping(self.owner.review, self.connection.inventory, proposed)
                unresolved = [row for row in rows
                              if row["included"] and row["status"] != "mapped"]
                if not unresolved:
                    wanted = {(item.correction_type, item.lattice_ordinal)
                              for item in self.owner.review.items}
                    self.explicit = [row for row in proposed
                                     if (row.get("correction_type"),
                                         row.get("lattice_ordinal")) in wanted]
                    self.mapping_label.setText(
                        "Embedded Complete Machine Mapping — verified automatically"
                    )
                    self.owner.mapping_auto_feedback = (
                        f"Mapping completed — {len(rows)} fitted magnet(s) matched by "
                        "AT ordinal and validated against the connected pyAML inventory."
                    )
                    self.owner.automatic_mapping_feedback.setText(
                        self.owner.mapping_auto_feedback
                    )
                    self._rendered = None
                    self.refresh()
                    return
            except Exception as exc:
                self._embedded_mapping_error = str(exc)
        exact = resolve_mapping(self.owner.review, self.connection.inventory) if self.owner.review and self.connection else []
        exact_unresolved = [row for row in exact if row["included"] and row["status"] != "mapped"]
        if not exact_unresolved:
            self.mapping_label.setText("Automatic authoritative backend identities")
            self.owner.mapping_auto_feedback = (
                f"Matching completed — {len(exact)} automatically validated from backend lattice identities."
            )
            self.owner.automatic_mapping_feedback.setText(self.owner.mapping_auto_feedback)
        else:
            settings = QSettings("pyLOCO", "pyLOCO Correct")
            remembered = [settings.value("correct/pyaml/approved_mapping_path", "")]
            candidates = approved_mapping_candidates(self.connection, remembered)
            errors = []
            for candidate in candidates:
                try:
                    lattice = find_reference_lattice(self.owner.review, self.connection)
                    entries, rows, provenance, _identity = validate_mapping_bundle(
                        self.owner.review, self.connection, candidate, lattice
                    )
                    self.explicit = entries; self.mapping_path = candidate
                    self.mapping_hash = file_hash(candidate)
                    self.owner.mapping_path = str(candidate)
                    self.mapping_label.setText(candidate.name)
                    self.owner.mapping_auto_feedback = (
                        f"Mapping completed — {sum(row['status'] == 'mapped' for row in rows)} "
                        f"validated, 0 unresolved; {provenance}."
                    )
                    self.owner.automatic_mapping_feedback.setText(self.owner.mapping_auto_feedback)
                    settings.setValue("correct/pyaml/approved_mapping_path", str(candidate))
                    break
                except Exception as exc:
                    errors.append(f"{candidate.name}: {exc}")
            else:
                self.mapping_label.setText("No compatible authoritative mapping found")
                detail = errors[0] if errors else "No approved mapping is associated with this machine profile."
                exact_mapped = sum(row["status"] == "mapped" for row in exact)
                exact_ambiguous = sum(row["status"] == "ambiguous" for row in exact)
                self.owner.mapping_auto_feedback = (
                    f"Matching completed — {exact_mapped} validated, {len(exact_unresolved)} unresolved, "
                    f"{exact_ambiguous} ambiguous. {detail} "
                    "Load an approved mapping or create a template for the exceptions."
                )
                self.owner.automatic_mapping_feedback.setText(self.owner.mapping_auto_feedback)
        self._rendered = None
        self.refresh()

    def save_validated_mapping(self):
        if not self.connection or not self.owner.review:
            raise ValueError("Connect and load a correction source first")
        entries = self.current_mapping()
        if not entries:
            rows = resolve_mapping(self.owner.review, self.connection.inventory)
            entries = [dict(
                correction_type=row["correction_type"], lattice_name=row["lattice_name"],
                lattice_ordinal=row["lattice_ordinal"], control_name=row["magnet_name"],
                component="B2L" if row["correction_type"] == "normal_quadrupole" else "A2L",
                unit="m^-1",
            ) for row in rows if row["status"] == "mapped"]
        lattice = find_reference_lattice(self.owner.review, self.connection)
        bundle = make_mapping_bundle(
            self.owner.review, self.connection, entries, lattice,
            provenance="validated in pyLOCO Correct",
        )
        embedded=getattr(self.owner.review,"machine_mapping",{}) or {}
        preserved=embedded if is_shared_machine_mapping(embedded) else None
        if self.mapping_path:
            try:
                candidate=load_shared_mapping(self.mapping_path)
                if is_shared_machine_mapping(candidate):preserved=candidate
            except Exception:pass
        bundle=merge_corrections(preserved,bundle)
        suggested = Path(self.connection.options["config_path"]).with_suffix(".mapping.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save validated Machine Mapping", str(suggested), "JSON mapping (*.json)"
        )
        if not path:
            return
        target = Path(path).expanduser()
        target.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        self.mapping_path = target; self.owner.mapping_path = str(target.resolve())
        self.mapping_hash = file_hash(target)
        self.mapping_label.setText(target.name)
        QSettings("pyLOCO", "pyLOCO Correct").setValue(
            "correct/pyaml/approved_mapping_path", str(target.resolve())
        )
        self.owner.manual_mapping_feedback.setText(
            f"Validated reusable mapping saved — Complete Machine Mapping: {len(bundle['mappings']):,} entries; "
            f"{len(entries):,} fitted magnet mapping(s) updated."
        )

    def export_mapping(self):
        self.owner.generate_mapping_template()

    def read_current(self):
        if not self.connection or not self.owner.review:
            raise ValueError("Connect and load a correction source first")
        self.refresh()
        unresolved=[item.name for item in self.owner.review.items if self.owner.review.applies_to(item) and (item.metadata.get("mapping_status")!="mapped" or not item.control_name)]
        if unresolved:
            raise ValueError(f"Valid mappings are required before reading strengths; {len(unresolved)} included magnet(s) remain unresolved")
        read=[]
        for item in self.owner.review.items:
            if item.control_name and item.metadata.get("mapping_status")=="mapped":
                item.machine_value = self.connection.read(item.control_name); read.append(item)
        self.owner.refresh_all()
        self.owner.current_k_status.setText(f"Read {len(read)} integrated KL value(s) [m⁻¹] using adapter magnet.strength.get(); no writes performed. This is adapter-value evidence, not independent hardware readback.")

    def current_mapping(self):
        if self.mapping_path and self.mapping_hash != file_hash(self.mapping_path):
            raise RuntimeError("Mapping file changed; reload it before preview/apply")
        return self.explicit

    def preview(self):
        if not self.transaction or not self.owner.review:
            raise ValueError("Connect and load a correction source first")
        if self.owner._source_load_thread is not None:
            raise RuntimeError("Wait for correction source loading to finish")
        self.transaction.mapping_path = self.mapping_path
        record = self.transaction.preview(self.owner.review, self.current_mapping())
        self.status.setText(f"Next step ready: {record['previous_fraction'] * 100:.6g}% → {record['fraction'] * 100:.6g}% cumulative; {len(record['changes'])} magnets. No writes.")
        self.owner._refresh_correction_progress()

    def apply(self):
        if not self.transaction or not self.owner.review or self.thread:
            return
        record=self.transaction.record or {}
        if QMessageBox.question(self, "Apply confirmed pyAML correction step?",
            f"Apply the previewed correction from {float(record.get('previous_fraction',0))*100:g}% to {float(record.get('fraction',0))*100:g}% "
            f"for {len(record.get('changes',()))} magnets?\n\nThe adapter get() value is verified after every write. Restoration can fail; inspect all results.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            explicit = self.current_mapping()
        except Exception as exc:
            self.error(str(exc)); return
        self.cancel = Event()
        self.start(lambda: self.transaction.apply(self.owner.review, explicit, confirmed=True, cancel=self.cancel),
                   "Applying correction — waiting for verified readbacks…")

    def cancel_preview(self):
        if not self.transaction:
            raise ValueError("Connect before cancelling a preview")
        self.transaction.cancel_preview()
        self.status.setText("Preview cancelled — no writes were made. Change the correction percentage and preview again.")
        self.owner._refresh_correction_progress()

    def restore(self):
        if self.pending() and not self.thread:
            if QMessageBox.question(self, "Restore original strengths?", "Write and verify every original pre-correction strength?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                self.start(self.transaction.restore,"Restoring original strengths — waiting for verified readbacks…")

    def recover(self):
        if not self.transaction:
            raise ValueError("Reconnect to the matching configuration before recovery")
        path, _ = QFileDialog.getOpenFileName(self, "Select interrupted-application recovery record", "", "JSON recovery record (*.json)")
        if path:
            self.transaction.recover(path)
            self.status.setText("Recovery record validated without writes. Review Application safety & recovery, then use Restore original… to confirm restoration.")
            self.owner.transaction_recovery_group.setChecked(True)
            self._update_recovery_display(True,self.pending(),False,self.transaction.record)

    def request_cancel(self):
        self.cancel.set()
        self.owner.transaction_recovery_group.setChecked(True)
        self.owner.recovery_activity_status.setText("Cancellation requested. The current operation will stop between magnet operations and attempt restoration of affected magnets.")

    def run(self, action):
        connection_test=action==self.connect_machine
        if connection_test:
            self.connect_button.setEnabled(False); self.connection_progress.setVisible(True)
            self.connection_result.setText("Testing connection…"); QApplication.processEvents()
        try:
            action()
        except Exception as exc:
            self.error(str(exc))
        finally:
            if connection_test:
                self.connection_progress.setVisible(False); self.connect_button.setEnabled(True)
        self.refresh()
        self.owner._refresh_correction_progress()

    def error(self, message):
        self.status.setText(message)
        if hasattr(self.owner, "mapping_diagnostics"):
            self.owner.mapping_diagnostics.setText(message)
        QMessageBox.warning(self, "pyAML correction", message)
        self.refresh()
        self.owner._refresh_correction_progress()

    def start(self, action, activity="Machine operation in progress…"):
        self.thread = QThread(self)
        self.owner.apply_activity_progress.setFormat(activity)
        self.owner.apply_activity_progress.setVisible(True)
        self.status.setText(activity)
        self.worker = TransactionWorker(action); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.completed)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.finished)
        self.refresh(); self.thread.start()

    def completed(self, record):
        if record.get("status")=="applied":
            self.status.setText(f"✓ CORRECTION APPLIED AND READBACK VERIFIED — {record['fraction'] * 100:.6g}% cumulative. Journal: {self.transaction.path}")
            self.status.setStyleSheet("font-weight: 700; color: #22c55e;")
            self.owner.statusBar().showMessage(
                f"Correction applied and verified • {record['fraction'] * 100:g}% cumulative • "
                f"{len(record.get('changes',()))} magnets",
                15000,
            )
        else:
            self.status.setText(f"{record['status']} — {record['fraction'] * 100:.6g}% cumulative. Journal: {self.transaction.path}")
        self._update_recovery_display(True,self.pending(),False,record)
        self.owner._refresh_correction_progress()

    def finished(self):
        self.owner.apply_activity_progress.setVisible(False)
        self.thread.deleteLater(); self.thread = None; self.worker = None
        self.refresh()
        self.owner._refresh_correction_progress()

    def _update_recovery_display(self,active,pending,busy,record):
        owner=self.owner
        if busy:
            owner.transaction_recovery_group.setChecked(True)
            if self.cancel.is_set():
                owner.recovery_activity_status.setText("Cancellation requested. Waiting for the current magnet operation to finish before restoration is attempted.")
            else:
                owner.recovery_activity_status.setText("A correction application or restoration operation is currently running.")
        else:
            owner.recovery_activity_status.setText("No correction application is currently running.")
        if record and pending:
            attempted=set(record.get("attempted",()))
            owner.recovery_availability_status.setText(
                f"A validated recovery record is loaded: {len(attempted)} magnet(s) may have been modified and require restoration."
            )
            owner.transaction_recovery_group.setChecked(True)
        elif record:
            owner.recovery_availability_status.setText(
                f"A transaction record is loaded with status '{record.get('status','unknown')}', but it does not currently indicate pending restoration. You may select another saved recovery record."
            )
        else:
            owner.recovery_availability_status.setText(
                "No recovery record is currently loaded. Select a saved JSON recovery record to check whether restoration is available."
            )
        connection=record.get("identity",{}).get("connection",{}) if record else {}
        options=connection.get("options",{}) if isinstance(connection,dict) else {}
        profile=options.get("config_path") or "unavailable"
        mode=options.get("mode") or "unavailable"
        changes=record.get("changes",()) if record else ()
        attempted=set(record.get("attempted",())) if record else set()
        original_lines=[f"{row.get('name','?')}: {row.get('original','unavailable')} {row.get('unit','')}" for row in changes if row.get("name") in attempted]
        status=record.get("status") if record else None
        failed=[row.get("name","?") for row in changes if row.get("status")=="restore_failed"]
        not_verified=[row.get("name","?") for row in changes if row.get("name") in attempted and row.get("status") not in {"restored","restore_failed"}]
        if status=="restored" and not failed and not not_verified:
            outcome="Successfully restored and readback-verified."
        elif status=="restore_failed":
            outcome=f"Restoration failed or remains incomplete. Manual intervention required: {', '.join(failed or not_verified) or 'inspect recovery record'}."
        elif status in {"restoring","write_pending"}:
            outcome="Restoration/application is in progress; completion has not yet been readback-verified."
        elif pending:
            outcome="Restoration is required and has not yet been verified."
        else:
            outcome="No loaded record currently requires restoration."
        authorization=("write authorized" if self.connection and self.connection.allow_writes else "read only / writes not authorized")
        connection_state="connected" if self.connection else "disconnected"
        owner.recovery_feedback.setText(
            f"Recovery record: machine/profile {profile} ({mode}); {len(attempted)} potentially modified magnet(s); "
            f"writes required for restoration; current state: {connection_state}, {authorization}. {outcome}" if record else ""
        )
        details=(f"Record status: {status or 'none'}\nMachine/profile: {profile}\nConnection: {connection_state}; authorization: {authorization}\n"
                 f"Potentially modified magnets: {len(attempted)}\nOriginal strengths to restore:\n"+
                 ("\n".join(original_lines) if original_lines else "None recorded or no magnet write was attempted.")+
                 (f"\nRestoration errors: {'; '.join(record.get('restoration_errors',()))}" if record and record.get("restoration_errors") else ""))
        owner.recovery_record_details.setPlainText(details)

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
        if hasattr(self.owner, "save_validated_mapping_button"):
            mapped = bool(self.owner.review and self.owner.review.items and all(
                item.metadata.get("mapping_status") == "mapped" for item in self.owner.review.items if self.owner.review.applies_to(item)
            ))
            self.owner.save_validated_mapping_button.setEnabled(
                active and self.connection is not None and mapped and not busy and not locked
            )
        self.restore_button.setEnabled(pending and not busy)
        self.recover_button.setEnabled(active and self.connection is not None and self.connection.allow_writes and not locked)
        self.cancel_button.setEnabled(busy)
        record = self.transaction.record if self.transaction else None
        self._update_recovery_display(active,pending,busy,record)
        if busy:
            self.apply_button.setEnabled(False)
            return
        if not self.connection or (not self.owner.review and not record):
            self.apply_button.setEnabled(False)
            self._rendered = None
            return
        can_apply = False
        try:
            if self._embedded_mapping_error:
                raise RuntimeError(
                    "Embedded Machine Mapping failed current-connection revalidation: "
                    + self._embedded_mapping_error)
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
                    item.metadata["mapping_validation_level"] = "connected_inventory"
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
