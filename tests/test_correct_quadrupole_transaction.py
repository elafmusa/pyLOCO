import copy
import json
from pathlib import Path
import pytest
from pyLOCO.correct.quadrupole_transaction import QuadrupoleTransaction

REQUEST = json.loads((Path(__file__).parents[1] / 'Examples/Correct/petra_single_b2.json').read_text())


class Fake:
    def __init__(self):
        self.value = -.08; self.writes = []; self.bad_physics = False; self.fail_get = False
        self.identity = dict(profile='petra3_realistic', lattice_sha256=REQUEST['lattice_sha256'], instance='one')

    def snapshot(self):
        return dict(identity=self.identity.copy(), quadrupoles=[dict(control='Q0K2_7_1/B2', ordinal=1,
            common_name='Q0K2_SWR_0', component='B2', unit='m^-2', factor=.99, offset=0,
            current=self.value, physical=self.value*.99 + (1e-3 if self.bad_physics and self.value != -.08 else 0))])

    def interface(self, identity): return self
    def get(self, name):
        if self.fail_get:
            self.fail_get = False
            raise RuntimeError('Transient GET failure')
        return self.value
    def set(self, name, value):
        self.writes.append(value); self.value = value


def test_preview_apply_restore(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c, tmp_path)
    t.preview(copy.deepcopy(REQUEST)); assert c.writes == []
    with pytest.raises(PermissionError): t.apply()
    t.apply(confirmed=True)
    assert t.path.exists(); assert t.record['applied_readback']['physical_delta'] == pytest.approx(1e-6)
    t.restore(); assert c.value == -.08
    assert json.loads(t.path.read_text())['status'] == 'restored'


def test_stale_and_identity_rejected_without_writes(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c, tmp_path); t.preview(copy.deepcopy(REQUEST))
    c.value += 1e-4
    with pytest.raises(RuntimeError, match='Stale'): t.apply(confirmed=True)
    assert not c.writes
    t.preview(copy.deepcopy(REQUEST)); c.identity['instance'] = 'two'
    with pytest.raises(RuntimeError, match='identity'): t.apply(confirmed=True)
    assert not c.writes


def test_physical_failure_restores_current_item(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c, tmp_path); t.preview(copy.deepcopy(REQUEST)); c.bad_physics = True
    with pytest.raises(RuntimeError, match='original restored'): t.apply(confirmed=True)
    assert len(c.writes) == 2 and c.value == -.08


def test_write_succeeded_read_failed_restores(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c, tmp_path); t.preview(copy.deepcopy(REQUEST))
    original_set = c.set
    def fail_once(name, value):
        original_set(name, value)
        if len(c.writes) == 1: c.fail_get = True
    c.set = fail_once
    with pytest.raises(RuntimeError, match='original restored'): t.apply(confirmed=True)
    assert c.value == -.08


@pytest.mark.parametrize('field,value', [('control','unknown/B2'), ('component','A2'), ('unit','rad'), ('common_name','wrong')])
def test_bad_mapping(tmp_path, field, value):
    c = Fake(); request = copy.deepcopy(REQUEST); request['mapping'][field] = value
    with pytest.raises(ValueError): QuadrupoleTransaction(c,tmp_path).preview(request)
    assert not c.writes


def test_journal_precedes_write_and_recovery(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c,tmp_path); t.preview(copy.deepcopy(REQUEST))
    original_set = c.set
    def checked(name, value):
        assert json.loads(t.path.read_text())['status'] in ('write_pending','restoring')
        original_set(name,value)
    c.set=checked; t.apply(confirmed=True)
    recovered=QuadrupoleTransaction(c,tmp_path); recovered.load_journal(t.path); recovered.restore()
    assert c.value == -.08


def test_restore_failure_is_reported_and_journaled(tmp_path):
    c = Fake(); t = QuadrupoleTransaction(c,tmp_path); t.preview(copy.deepcopy(REQUEST))
    original_set = c.set
    def broken(name, value):
        if c.writes: raise RuntimeError('restore transport failed')
        original_set(name,value)
    c.set=broken; c.bad_physics=True
    with pytest.raises(RuntimeError, match='RESTORATION FAILED'): t.apply(confirmed=True)
    assert json.loads(t.path.read_text())['status'] == 'restore_failed'


def test_gui_explicit_profile_and_no_default():
    from pyLOCO.correct.app import build_application
    from pyLOCO.correct.main_window import CorrectMainWindow
    app=build_application(['single-b2-test']); window=CorrectMainWindow()
    widget=window.quadrupole_workspace
    assert widget.profile.currentData() is None
    assert not widget.apply_button.isEnabled()
    assert [widget.profile.itemData(i) for i in range(1,4)] == ['ebs','petra3','petra3_realistic']
    window.close()
