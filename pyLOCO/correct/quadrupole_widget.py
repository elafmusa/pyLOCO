"""Explicit single-normal-quadrupole simulation workflow."""
import json
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QLabel,
                              QPushButton, QFileDialog, QMessageBox, QSpinBox,
                              QTableWidget, QTableWidgetItem, QHeaderView)
from pyLOCO.control_system.pysc_profiles import available_pysc_profiles
from .quadrupole_transaction import SimulationConnection, QuadrupoleTransaction
from .multi_quadrupole_transaction import MultiQuadrupoleTransaction


class QuadrupoleWidget(QWidget):
    def __init__(self, owner):
        super().__init__(); self.owner = owner; self.transaction = None; self.request = None
        layout = QVBoxLayout(self)
        self.identity = QLabel('Select pySC Server and an explicit machine/profile — no default machine')
        self.identity.setWordWrap(True); layout.addWidget(self.identity)
        form = QFormLayout(); layout.addLayout(form)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.profile = QComboBox(); self.profile.addItem('Choose machine / profile…', None)
        for p in available_pysc_profiles():
            self.profile.addItem(f"{p.label.split(' /')[0]} / {p.scenario}", p.key)
        form.addRow('Machine / profile', self.profile)
        self.port = QSpinBox(); self.port.setRange(1024, 65535); self.port.setValue(13132)
        form.addRow('Local diagnostic port', self.port)
        self.connect_button = QPushButton('Connect / discover B2 controls')
        self.inventory = QComboBox(); form.addRow('Verified B2 inventory', self.inventory)
        self.load_button = QPushButton('Load mapped correction…')
        self.preview_button = QPushButton('Preview — zero writes')
        actions = QHBoxLayout(); layout.addLayout(actions)
        for button in (self.connect_button,self.load_button,self.preview_button): actions.addWidget(button)
        self.table = QTableWidget(0, 2); self.table.setHorizontalHeaderLabels(['Quantity', 'Value'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers); layout.addWidget(self.table, 1)
        self.summary = QLabel('No transaction previewed'); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.apply_button = QPushButton('Confirm and Apply normal B2 transaction'); layout.addWidget(self.apply_button)
        self.restore_button = QPushButton('Restore original — control AND physical K'); layout.addWidget(self.restore_button)
        self.recover_button = QPushButton('Recover transaction journal…'); layout.addWidget(self.recover_button)
        self.status = QLabel('Simulation calibration only — NOT real PETRA hardware calibration. Single B2 or 3–5 synthetic B2 items only.')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.apply_button.setEnabled(False); self.restore_button.setEnabled(False)
        self.connect_button.clicked.connect(lambda: self.run(self.connect_profile))
        self.load_button.clicked.connect(self.load_request)
        self.preview_button.clicked.connect(lambda: self.run(self.preview))
        self.apply_button.clicked.connect(self.apply)
        self.restore_button.clicked.connect(lambda: self.run(self.restore))
        self.recover_button.clicked.connect(self.recover)
        self.profile.currentIndexChanged.connect(self.invalidate)
        self.port.valueChanged.connect(self.invalidate)

    def pending(self):
        return self.transaction and self.transaction.record and self.transaction.record['status'] in ('applied', 'write_pending', 'restoring', 'restore_failed')

    def invalidate(self):
        if not self.pending():
            self.transaction = None; self.apply_button.setEnabled(False)
            self.inventory.clear()
            self.identity.setText('Not connected — select the machine/profile and Connect')
            self.owner.profile_badge.setText('Machine/profile not verified')
            self.owner._set_connection(False, 'DISCONNECTED')

    def run(self, action):
        try:
            action()
        except Exception as exc:
            if self.transaction and self.transaction.record:
                self.show_record(self.transaction.record)
            self.status.setText(str(exc)); self.apply_button.setEnabled(False)
            QMessageBox.critical(self, 'Quadrupole transaction', str(exc))
        finally:
            pending = bool(self.pending())
            self.restore_button.setEnabled(pending)
            for widget in (self.profile, self.port, self.connect_button, self.load_button, self.preview_button, self.inventory, self.recover_button, self.owner.backend_combo):
                widget.setEnabled(not pending)

    def connect_profile(self):
        if self.owner.backend_combo.currentData() != 'pysc' or self.profile.currentData() is None:
            raise ValueError('Select pySC Server and a machine/profile explicitly')
        connection = SimulationConnection(self.port.value()); data = connection.snapshot()
        if data['identity']['profile'] != self.profile.currentData():
            raise ValueError('Selected profile does not match the running server')
        self.inventory.clear()
        for row in data['quadrupoles']:
            self.inventory.addItem(row['control'], row)
        identity = data['identity']
        text = f"{identity['machine'].split(' /')[0]} / {identity['scenario']} — Seed {identity['seed']} — {len(data['quadrupoles'])} B2 controls"
        self.identity.setText('DEMO • pySC SERVER\n' + text)
        self.owner.profile_badge.setText(text); self.owner._set_connection(True, 'CONNECTED')
        self.owner.registry.pysc_profile = self.profile.currentData()
        self.transaction = QuadrupoleTransaction(connection, Path.cwd() / 'correction-transactions')

    def load_request(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load explicit B2 request', '', 'JSON (*.json)')
        if path:
            def load():
                self.request = json.loads(Path(path).read_text()); self.apply_button.setEnabled(False)
                self.status.setText(f'Loaded {path}. No machine writes. Preview required.')
            self.run(load)

    def preview(self):
        if not self.transaction or not self.request:
            raise ValueError('Connect and load an explicit correction first')
        cls = MultiQuadrupoleTransaction if 'items' in self.request else QuadrupoleTransaction
        if type(self.transaction) is not cls:
            if self.pending(): raise ValueError('Restore the pending transaction first')
            self.transaction = cls(self.transaction.connection, self.transaction.directory)
        record = self.transaction.preview(self.request)
        first = record['items'][0] if 'items' in record else record
        self.inventory.setCurrentIndex(self.inventory.findText(first['original']['control']))
        self.show_record(record); self.apply_button.setEnabled(True)

    def show_record(self, r):
        if 'items' in r:
            self.table.setColumnCount(8)
            self.table.setHorizontalHeaderLabels(['Name (B2)', 'Current K [m⁻²]', 'Requested ΔK [m⁻²]', 'Calibration factor',
                                                  'Proposed control [m⁻²]', 'Control readback [m⁻²]', 'Achieved ΔK [m⁻²]', 'Restoration'])
            self.table.setRowCount(len(r['items']))
            for row, item in enumerate(r['items']):
                original = item['original']; result = item.get('applied_readback', {})
                values = [original['control'], original['physical'], item['requested_physical_delta'], original['factor'],
                          item['proposed'], result.get('control', '—'), result.get('physical_delta', '—'), item.get('restoration_status', 'not restored')]
                for col, value in enumerate(values):
                    cell = QTableWidgetItem(f'{value:.12g}' if isinstance(value, float) else str(value))
                    cell.setToolTip(str(value) + '\nOfficial: ' + original['common_name'] + f" / index {original['ordinal']}")
                    self.table.setItem(row, col, cell)
            self.table.setFixedHeight(self.table.horizontalHeader().height()+sum(self.table.rowHeight(i) for i in range(len(r['items'])))+12)
            s = r['summary']
            self.summary.setText(f"Requested: {s['requested']}   Applied: {s['applied']}   Verified: {s['verified']}   Restored: {s['restored']}\n"
                                 f"Untouched originals verified: {s['untouched_verified']}   All restored: {'YES' if s['all_restored'] else 'NO'}")
            self.status.setText(r['status'].upper() + '\n' + r['calibration_source'] + '\nJournal: ' + str(getattr(self.transaction, 'path', 'created before Apply')))
            return
        self.table.setColumnCount(2); self.table.setHorizontalHeaderLabels(['Quantity', 'Value'])
        self.summary.setText('Single B2 transaction')
        o = r['original']
        rows = [('Mapped control', o['control']), ('Component / unit', 'B2 / m^-2'),
                ('FIT/source identity', f"{r['mapping']['fit_identity']['name']} / index {r['mapping']['fit_identity']['ordinal']}"),
                ('Official lattice identity', f"{o['common_name']} / index {o['ordinal']}"),
                ('Current control setpoint', o['current']), ('Calibration factor — pySC simulation', o['factor']),
                ('Current physical K', o['physical']), ('Requested physical ΔK', r['requested_physical_delta']),
                ('Required control Δ = ΔK / factor', r['control_delta']), ('Proposed control setpoint', r['proposed']),
                ('Expected physical K after application', r['expected_physical'])]
        for key, label in (('applied_readback', 'Applied'), ('restore_readback', 'Restored')):
            for name, value in r.get(key, {}).items(): rows.append((f'{label}: {name.replace("_", " ")}', value))
        self.table.setRowCount(len(rows))
        for i, (key, value) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(key)); self.table.setItem(i, 1, QTableWidgetItem(str(value)))
        self.table.setFixedHeight(self.table.horizontalHeader().height()+sum(self.table.rowHeight(i) for i in range(len(rows)))+6)
        self.status.setText(r['status'].upper() + '\n' + r['calibration_source'] + '\nSource: ' + r['mapping']['fit_identity']['source'] + '\nJournal: ' + str(getattr(self.transaction, 'path', 'created before Apply')))

    def apply(self):
        count = len(self.transaction.record.get('items', [1])) if self.transaction and self.transaction.record else 0
        if QMessageBox.question(self, 'Confirm simulation quadrupoles',
                                f'Apply the {count} previewed physical ΔK values to PETRA III / realistic_errors?\nNo PETRA hardware writes.',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.run(lambda: self.show_record(self.transaction.apply(confirmed=True)))
        self.apply_button.setEnabled(False)

    def restore(self):
        self.show_record(self.transaction.restore())

    def recover(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Recover journal from this server instance', '', 'JSON (*.json)')
        if path:
            def load():
                if not self.transaction: raise ValueError('Connect first')
                if not self.pending():
                    data = json.loads(Path(path).read_text())
                    cls = MultiQuadrupoleTransaction if 'items' in data else QuadrupoleTransaction
                    self.transaction = cls(self.transaction.connection, self.transaction.directory)
                self.show_record(self.transaction.load_journal(path))
                self.apply_button.setEnabled(False)
            self.run(load)
