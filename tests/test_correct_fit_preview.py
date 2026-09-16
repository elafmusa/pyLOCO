import copy
import json
from pathlib import Path

import pytest

from pyLOCO.correct.fit_preview import preview_fit, load_fit_bundle


class ReadOnlyConnection:
    def __init__(self):
        self.reads = 0
        self.row = dict(common_name='Q_test', family='unique_family', control='unique_family/B2',
                        ordinal=99, component='B2', unit='m^-2', factor=1.02, offset=.003,
                        current=.2, physical=.207)
        self.data = dict(identity=dict(profile='petra3_realistic', lattice_sha256='hash'),
                         quadrupoles=[self.row])
    def snapshot(self): return copy.deepcopy(self.data)
    def interface(self, identity): return self
    def get(self, control):
        self.reads += 1
        assert control == self.row['control']
        return self.row['current']
    def set(self, *args): raise AssertionError('NO WRITES ALLOWED')


def bundle():
    return dict(lattice_sha256='hash', parameters=[dict(source_ordinal=7,
                common_name='Q_test', family='unique_family', initial=.3, fitted=.31, full_delta=-.01)])


def test_sign_fraction_calibration_and_no_ordinal_transfer():
    c = ReadOnlyConnection(); r = preview_fit(bundle(), c); i = r['items'][0]
    assert c.reads == 1
    assert i['applied_delta'] == pytest.approx(-.001)
    assert i['control_delta'] == pytest.approx(-.001/1.02)
    assert i['expected_physical'] == pytest.approx(.206)
    assert i['original']['ordinal'] == 99  # source ordinal 7 is NOT transferred
    assert r['mapped'] == 1


def test_ebs_profile_preview_is_explicit_and_read_only():
    c = ReadOnlyConnection()
    c.data['identity']['profile'] = 'ebs'
    b = bundle()
    b['profile'] = 'ebs'
    result = preview_fit(b, c)
    assert result['mapped'] == 1
    assert c.reads == 1
    assert not hasattr(c, 'writes')


def test_ebs_bundle_rejects_petra_server():
    b = bundle()
    b['profile'] = 'ebs'
    with pytest.raises(ValueError, match='matching ebs'):
        preview_fit(b, ReadOnlyConnection())


@pytest.mark.parametrize('failure', ['family', 'common', 'duplicate', 'skew', 'unit', 'factor', 'transform', 'profile', 'hash'])
def test_reject_without_writes(failure):
    c = ReadOnlyConnection()
    if failure == 'family': c.row['family'] = 'other'
    if failure == 'common': c.row['common_name'] = 'other'
    if failure == 'duplicate': c.data['quadrupoles'].append(c.row.copy())
    if failure == 'skew': c.row['component'] = 'A2'
    if failure == 'unit': c.row['unit'] = 'm^-1'
    if failure == 'factor': c.row['factor'] = 0
    if failure == 'transform': c.row['physical'] += .1
    if failure == 'profile': c.data['identity']['profile'] = 'petra3'
    if failure == 'hash': c.data['identity']['lattice_sha256'] = 'other'
    with pytest.raises(ValueError): preview_fit(bundle(), c)


def test_source_hash_rejected(tmp_path):
    f = tmp_path / 'result'; f.write_text('changed')
    manifest = dict(schema='pyloco-fit-b2-preview-v1', parameter_mode='individual', unit='m^-2',
                    sources={'initial': dict(path=str(f), sha256='wrong')})
    p = tmp_path/'manifest.json'; p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash mismatch'): load_fit_bundle(p)


def test_family_mode_not_silently_expanded(tmp_path):
    p = tmp_path/'manifest.json'; p.write_text(json.dumps(dict(schema='pyloco-fit-b2-preview-v1', parameter_mode='family', unit='m^-2')))
    with pytest.raises(ValueError, match='no family expansion'): load_fit_bundle(p)


def test_gui_fit_preview_disables_apply():
    from pyLOCO.correct.app import build_application
    from pyLOCO.correct.main_window import CorrectMainWindow
    app = build_application(['fit-preview-test']); window = CorrectMainWindow()
    widget = window.quadrupole_workspace
    b = bundle(); b['sources'] = {'fitted': 'real-result.mat'}
    r = preview_fit(b, ReadOnlyConnection())
    widget.fit_bundle = b; widget.show_fit_preview(r)
    assert widget.table.columnCount() == 10
    assert '10%' in widget.summary.text()
    assert not widget.apply_button.isEnabled()
    widget.apply()  # even direct invocation cannot enter the write workflow
    assert 'preview-only' in widget.status.text()
    window.close()
