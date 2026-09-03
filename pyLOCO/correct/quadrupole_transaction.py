"""One explicitly mapped PETRA simulation B2 transaction; no bulk application."""
import json
import math
import os
from pathlib import Path
from urllib.request import urlopen


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


class SimulationConnection:
    def __init__(self, diagnostics_port=13132):
        self.url = f'http://127.0.0.1:{int(diagnostics_port)}/snapshot'

    def snapshot(self):
        with urlopen(self.url, timeout=5) as response:
            return json.load(response)

    def interface(self, identity):
        from pyLOCO.control_system.pysc_server import pySCServerOrbitInterface
        return pySCServerOrbitInterface(host='127.0.0.1', port=identity['control_port'])


class QuadrupoleTransaction:
    def __init__(self, connection, journal_directory):
        self.connection = connection
        self.directory = Path(journal_directory)
        self.record = None

    def preview(self, request):
        if self.record and self.record['status'] not in ('preview', 'restored', 'stale'):
            raise RuntimeError('Restore the pending transaction before another preview')
        snapshot = self.connection.snapshot()
        identity = snapshot['identity']
        if identity['profile'] != 'petra3_realistic':
            raise ValueError('Only PETRA III / realistic_errors is enabled')
        mapping = request['mapping']
        if request['profile'] != identity['profile'] or request['lattice_sha256'] != identity['lattice_sha256']:
            raise ValueError('Correction/profile lattice identity mismatch')
        if mapping['component'] != 'B2' or mapping['unit'] != 'm^-2':
            raise ValueError('Only normal B2 in m^-2 is supported')
        fit = mapping['fit_identity']
        if not fit.get('source') or not fit.get('name') or not isinstance(fit.get('ordinal'), int):
            raise ValueError('Explicit FIT/source lattice identity required')
        rows = [r for r in snapshot['quadrupoles'] if r['control'] == mapping['control']]
        if len(rows) != 1:
            raise ValueError('Unknown or ambiguous control; no name fallback')
        row = rows[0]
        if any(row[k] != mapping[k] for k in ('ordinal', 'common_name', 'component', 'unit')):
            raise ValueError('Official element mapping mismatch')
        if fit.get('lattice_sha256') != identity['lattice_sha256'] or fit['ordinal'] != row['ordinal'] or fit['name'] != row['common_name']:
            raise ValueError('This milestone requires a source identity on the same official lattice; cross-lattice FIT transfer is not enabled')
        delta = float(request['physical_delta'])
        if not all(math.isfinite(row[k]) for k in ('factor', 'offset', 'physical', 'current')) or not math.isfinite(delta):
            raise ValueError('Nonfinite value')
        if row['factor'] == 0 or not 0 < abs(delta) <= 1e-5:
            raise ValueError('Single-test physical change must be nonzero and <= 1e-5 m^-2')
        interface = self.connection.interface(identity)
        if not close(interface.get(row['control']), row['current']):
            raise ValueError('Control and diagnostic snapshots disagree')
        if not close(row['physical'], row['factor'] * row['current'] + row['offset']):
            raise ValueError('Unsupported physical/control transformation')
        self.record = dict(status='preview', identity=identity, mapping=mapping,
                           calibration_source='pySC simulation profile, NOT PETRA hardware calibration',
                           original=row, requested_physical_delta=delta, control_delta=delta/row['factor'],
                           proposed=row['current']+delta/row['factor'], expected_physical=row['physical']+delta)
        if hasattr(self, 'path'):
            del self.path
        return self.record

    def _verified_row(self):
        snapshot = self.connection.snapshot()
        if snapshot['identity'] != self.record['identity']:
            raise RuntimeError('Server identity changed; refusing write to a different machine instance')
        row = next(r for r in snapshot['quadrupoles'] if r['control'] == self.record['original']['control'])
        if any(row[k] != self.record['original'][k] for k in ('factor', 'offset', 'ordinal', 'common_name')):
            raise RuntimeError('Mapping/calibration changed')
        return row

    def _sample(self):
        row = self._verified_row()
        interface = self.connection.interface(self.record['identity'])
        value = interface.get(row['control'])
        if not close(value, row['current']):
            raise RuntimeError('Concurrent machine change detected')
        return value, row['physical']

    def _persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, 'path'):
            import uuid
            self.path = self.directory / (str(uuid.uuid4()) + '.json')
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(self.record, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def apply(self, *, confirmed=False):
        if not confirmed or not self.record or self.record['status'] != 'preview':
            raise PermissionError('A fresh preview and explicit confirmation are required')
        control, physical = self._sample()
        original = self.record['original']
        if not close(control, original['current']) or not close(physical, original['physical']):
            self.record['status'] = 'stale'
            raise RuntimeError('Stale preview: Preview again')
        self.record['status'] = 'write_pending'; self._persist()
        try:
            self.connection.interface(self.record['identity']).set(original['control'], self.record['proposed'])
            control, physical = self._sample()
            self.record['applied_readback'] = dict(control=control, physical=physical, physical_delta=physical-original['physical'])
            if not close(control, self.record['proposed']) or not close(physical, self.record['expected_physical']):
                raise RuntimeError('Control or independent lattice verification failed')
            self.record['status'] = 'applied'; self._persist()
        except Exception as exc:
            self.record['apply_error'] = str(exc)
            try:
                self.restore()
            except Exception as restore_error:
                raise RuntimeError(f'Apply failed: {exc}; RESTORATION FAILED: {restore_error}') from exc
            raise RuntimeError(f'Apply failed: {exc}; original restored and verified') from exc
        return self.record

    def restore(self):
        if not self.record or self.record['status'] not in ('applied', 'write_pending', 'restore_failed', 'restoring'):
            raise RuntimeError('No written transaction to restore')
        try:
            self._verified_row()  # identity check; failed command GET must not prevent rollback
            self.record['status'] = 'restoring'; self._persist()
            original = self.record['original']
            self.connection.interface(self.record['identity']).set(original['control'], original['current'])
            control, physical = self._sample()
            self.record['restore_readback'] = dict(control=control, physical=physical)
            if control != original['current'] or not close(physical, original['physical']):
                raise RuntimeError('Original control/physical K restoration verification failed')
            self.record['status'] = 'restored'; self._persist()
        except Exception as exc:
            self.record['status'] = 'restore_failed'; self.record['restore_error'] = str(exc); self._persist()
            raise
        return self.record

    def load_journal(self, path):
        if self.record and self.record['status'] in ('applied', 'write_pending', 'restoring', 'restore_failed'):
            raise RuntimeError('An unresolved transaction is already loaded')
        self.record = json.loads(Path(path).read_text())
        self.path = Path(path)
        self._sample()
        return self.record
