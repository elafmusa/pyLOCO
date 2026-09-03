import copy
import json
from pathlib import Path
import pytest
from pyLOCO.correct.multi_quadrupole_transaction import MultiQuadrupoleTransaction

REQUEST = json.loads((Path(__file__).parents[1]/'Examples/Correct/petra_four_b2.json').read_text())


class Fake:
    def __init__(self):
        self.rows = {}
        for i, item in enumerate(REQUEST['items']):
            m = item['mapping']; current = .1*(i+1); factor = 1+.001*i
            self.rows[m['control']] = dict(control=m['control'], ordinal=m['ordinal'], common_name=m['common_name'],
                component='B2', unit='m^-2', factor=factor, offset=0, current=current, physical=current*factor)
        self.original = copy.deepcopy(self.rows); self.writes=[]; self.fail_at=None; self.restore_fail=None
        self.identity = dict(profile='petra3_realistic', lattice_sha256=REQUEST['lattice_sha256'], instance='one')
    def snapshot(self): return dict(identity=self.identity.copy(), quadrupoles=copy.deepcopy(list(self.rows.values())))
    def interface(self, identity): return self
    def get(self, name): return self.rows[name]['current']
    def set(self, name, value):
        self.writes.append((name,value))
        if self.restore_fail == name and value == self.original[name]['current']: raise RuntimeError('restore failed')
        row=self.rows[name]; row['current']=value; row['physical']=row['factor']*value
        if len(self.writes)==self.fail_at: raise RuntimeError('injected after SET')


def test_success_and_exact_restore(tmp_path):
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    assert not c.writes
    with pytest.raises(PermissionError): t.apply()
    t.apply(confirmed=True); assert t.record['summary']['verified']==4
    for i in t.record['items']: assert i['applied_readback']['physical_delta']==pytest.approx(i['requested_physical_delta'])
    t.restore(); assert c.rows==c.original
    assert t.record['summary']['restored']==4 and t.record['summary']['all_restored']


def test_middle_failure_includes_current_and_previous(tmp_path):
    c=Fake(); c.fail_at=3; t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    with pytest.raises(RuntimeError,match='all original'): t.apply(confirmed=True)
    assert c.rows==c.original
    assert [name for name,_ in c.writes[3:]]==list(c.rows)[:3][::-1]
    s=t.record['summary']; assert (s['applied'],s['verified'],s['restored'],s['untouched_verified'])==(2,2,3,1)
    assert s['all_restored']


def test_rollback_continues_after_restore_failure(tmp_path):
    c=Fake(); c.fail_at=3; c.restore_fail=list(c.rows)[1]
    t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    with pytest.raises(RuntimeError,match='RESTORATION FAILED'): t.apply(confirmed=True)
    assert c.rows[list(c.rows)[0]]==c.original[list(c.rows)[0]]
    assert t.record['status']=='restore_failed'
    c.restore_fail=None; t.restore(); assert c.rows==c.original


def test_stale_last_item_blocks_all_writes(tmp_path):
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    c.rows[list(c.rows)[-1]]['current']+=.01
    with pytest.raises(RuntimeError): t.apply(confirmed=True)
    assert not c.writes


def test_complete_journal_before_first_write_and_recover(tmp_path):
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    write=c.set
    def checked(name,value):
        journal=json.loads(t.path.read_text()); assert len(journal['items'])==4
        assert journal['status'] in ('write_pending','restoring')
        write(name,value)
    c.set=checked; t.apply(confirmed=True)
    recovered=MultiQuadrupoleTransaction(c,tmp_path); recovered.load_journal(t.path); recovered.restore()
    assert c.rows==c.original


@pytest.mark.parametrize('change',['duplicate','skew','large','real_fit','too_many'])
def test_reject_invalid_requests(tmp_path,change):
    r=copy.deepcopy(REQUEST)
    if change=='duplicate':r['items'][1]=r['items'][0]
    if change=='skew':r['items'][1]['mapping']['component']='A2'
    if change=='large':r['items'][1]['physical_delta']=1
    if change=='real_fit':r['purpose']='FIT'
    if change=='too_many':r['items']*=2
    c=Fake()
    with pytest.raises(ValueError):MultiQuadrupoleTransaction(c,tmp_path).preview(r)
    assert not c.writes


def test_physical_readback_failure_rolls_back(tmp_path):
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST); write=c.set
    def incorrect(name,value):
        write(name,value)
        if len(c.writes)==2:c.rows[name]['physical']+=.001
    c.set=incorrect
    with pytest.raises(RuntimeError,match='all original'):t.apply(confirmed=True)
    assert c.rows==c.original and t.record['summary']['restored']==2


def test_no_writes_if_journal_cannot_be_saved(tmp_path):
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    def fail():raise OSError('disk full')
    t._persist=fail
    with pytest.raises(OSError):t.apply(confirmed=True)
    assert not c.writes


def test_compact_gui_table_and_summary(tmp_path):
    from pyLOCO.correct.app import build_application
    from pyLOCO.correct.main_window import CorrectMainWindow
    app=build_application(['multi-table']); w=CorrectMainWindow(); widget=w.quadrupole_workspace
    c=Fake(); t=MultiQuadrupoleTransaction(c,tmp_path); t.preview(REQUEST)
    widget.transaction=t; widget.show_record(t.record)
    assert widget.table.rowCount()==4 and widget.table.columnCount()==8
    t.apply(confirmed=True); t.restore(); widget.show_record(t.record)
    assert 'All restored: YES' in widget.summary.text()
    assert all(widget.table.item(i,7).text()=='restored' for i in range(4))
    w.close()
