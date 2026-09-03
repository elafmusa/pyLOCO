"""Bounded synthetic normal-B2 transactions; reuses verified single-item mapping."""
import copy
import json
from pathlib import Path

from .quadrupole_transaction import QuadrupoleTransaction, close


class MultiQuadrupoleTransaction(QuadrupoleTransaction):
    """One durable journal for at most five items, including uncertain SETs."""

    def _item_service(self, item):
        service = QuadrupoleTransaction(self.connection, self.directory)
        service.record = item
        return service

    def preview(self, request):
        if self.record and self.record['status'] not in ('preview', 'restored', 'stale'):
            raise RuntimeError('Restore the pending transaction first')
        requests = request.get('items')
        if request.get('purpose') != 'synthetic_validation' or not isinstance(requests, list) or not 3 <= len(requests) <= 5:
            raise ValueError('Only 3–5 synthetic normal-B2 items are enabled')
        items = []
        for entry in requests:
            single = {**request, **entry}
            service = QuadrupoleTransaction(self.connection, self.directory)
            items.append(copy.deepcopy(service.preview(single)))
        identity = items[0]['identity']
        if any(item['identity'] != identity for item in items):
            raise ValueError('Server identity changed during Preview')
        for key in ('control', 'ordinal'):
            if len({item['original'][key] for item in items}) != len(items):
                raise ValueError('Duplicate control or lattice element')
        self.record = dict(schema='pyloco.small_b2_transaction.v1', status='preview',
                           identity=identity, purpose='synthetic_validation', items=items,
                           calibration_source=items[0]['calibration_source'])
        if hasattr(self, 'path'): del self.path
        self._summary()
        return self.record

    def _summary(self):
        items = self.record['items']
        self.record['summary'] = dict(
            requested=len(items), applied=sum(bool(i.get('set_completed')) for i in items),
            verified=sum(bool(i.get('verified')) for i in items),
            restored=sum(i.get('restoration_status') == 'restored' for i in items),
            untouched_verified=sum(i.get('restoration_status') == 'unchanged_verified' for i in items),
            all_restored=all(i.get('restoration_status') in ('restored', 'unchanged_verified') for i in items))

    def _persist(self):
        self._summary()
        super()._persist()

    def _check_original(self, item):
        control, physical = self._item_service(item)._sample()
        original = item['original']
        if not close(control, original['current']) or not close(physical, original['physical']):
            raise RuntimeError(f"Stale preview for {original['control']}: Preview again")

    def apply(self, *, confirmed=False):
        if not confirmed or not self.record or self.record['status'] != 'preview':
            raise PermissionError('Fresh Preview and explicit confirmation required')
        # Validate the WHOLE transaction before the first write.
        try:
            for item in self.record['items']: self._check_original(item)
        except Exception:
            self.record['status'] = 'stale'
            raise
        self.record['status'] = 'write_pending'; self._persist()
        try:
            for item in self.record['items']:
                self._check_original(item)  # also immediately before each SET
                item['attempted'] = True
                self._persist()  # current item must be recoverable even if SET loses its reply
                self.connection.interface(self.record['identity']).set(item['original']['control'], item['proposed'])
                item['set_completed'] = True
                control, physical = self._item_service(item)._sample()
                item['applied_readback'] = dict(control=control, physical=physical,
                                               physical_delta=physical-item['original']['physical'])
                if not close(control, item['proposed']) or not close(physical, item['expected_physical']):
                    raise RuntimeError(f"Independent verification failed: {item['original']['control']}")
                item['verified'] = True
                self._persist()
            # Check every final lattice value, not just the value immediately after its SET.
            for item in self.record['items']:
                control, physical = self._item_service(item)._sample()
                if not close(control, item['proposed']) or not close(physical, item['expected_physical']):
                    raise RuntimeError('Final whole-transaction verification failed')
            self.record['status'] = 'applied'; self._persist()
        except Exception as exc:
            self.record['apply_error'] = str(exc)
            try: self.restore()
            except Exception as restore_error:
                raise RuntimeError(f'Apply failed: {exc}; RESTORATION FAILED: {restore_error}') from exc
            raise RuntimeError(f'Apply failed: {exc}; all original values verified') from exc
        return self.record

    def restore(self):
        if not self.record or self.record['status'] not in ('applied', 'write_pending', 'restoring', 'restore_failed'):
            raise RuntimeError('No written transaction to restore')
        self.record['status'] = 'restoring'
        errors = []
        # Persistence failure must not stop best-effort physical restoration.
        try: self._persist()
        except Exception as exc: errors.append(f'Journal: {exc}')
        for item in reversed(self.record['items']):
            try:
                service = self._item_service(item)
                service._verified_row()  # endpoint/instance/calibration gate
                original = item['original']
                if item.get('attempted'):
                    self.connection.interface(self.record['identity']).set(original['control'], original['current'])
                control, physical = service._sample()
                item['restore_readback'] = dict(control=control, physical=physical)
                # Fixed simulated machine: demand exact original values in both domains.
                if control != original['current'] or physical != original['physical']:
                    raise RuntimeError('Original control/physical K not exactly restored')
                item['restoration_status'] = 'restored' if item.get('attempted') else 'unchanged_verified'
            except Exception as exc:
                item['restoration_status'] = 'failed'; item['restore_error'] = str(exc)
                errors.append(f"{item['original']['control']}: {exc}")
            try: self._persist()
            except Exception as exc: errors.append(f'Journal: {exc}')
        self.record['status'] = 'restore_failed' if errors else 'restored'
        self.record['restore_errors'] = errors
        self._persist()
        if errors: raise RuntimeError('; '.join(errors))
        return self.record

    def load_journal(self, path):
        if self.record and self.record['status'] not in ('preview', 'restored', 'stale'):
            raise RuntimeError('An unresolved transaction is already loaded')
        record = json.loads(Path(path).read_text())
        if record.get('schema') != 'pyloco.small_b2_transaction.v1' or not 3 <= len(record['items']) <= 5:
            raise ValueError('Not a small B2 transaction journal')
        for item in record['items']:
            if item['identity'] != record['identity']: raise ValueError('Journal identity mismatch')
            self._item_service(item)._verified_row()
        self.record = record; self.path = Path(path)
        return self.record
