"""Opt-in local orbit correction with a deterministic pyAML-shaped backend."""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from pyLOCO.control_system.pyaml_adapter import PyAMLAdapter, PyAMLOrbitCorrectionError
from pyLOCO.measure.acquisition import BpmDevice, CorrectorDevice, ORMAcquirer, ORMAcquisitionError, ORMInterrupted


class Strength:
    def __init__(self, name, log):
        self.value = 0.0
        self.name, self.log = name, log
    def get(self):
        return self.value
    def set(self, value):
        self.log.append((self.name, value))
        self.value = value
    def unit(self):
        return "rad"


class Magnets(list):
    def names(self):
        return [mag.name for mag in self]


class Matrix:
    def __init__(self):
        self.matrix = np.array([[1., 0., 2.], [2., 0., 1.], [0., 1., 0.], [0., 2., 0.]])
        self.input_names = ['H1', 'V1', 'H2']
        self.input_planes = ['H', 'V', 'H']
        self.output_names = ['B1', 'B2', 'B1', 'B2']
        self.output_planes = ['H', 'H', 'V', 'V']
        self.bad_inputs, self.bad_outputs = [2], []
        self.virtual_weight = 3.
    def get_matrix_in_plane(self, plane):
        return self.matrix[np.array(self.output_planes) == plane][:, np.array(self.input_planes) == plane]
    def enable_all_inputs(self):
        self.bad_inputs = []
    def disable_all_inputs_but(self, names):
        self.bad_inputs = [i for i, name in enumerate(self.input_names) if name not in names]


def fixture(*, enabled=True, mutate=None, allow_writes=True):
    writes = []
    magnets = {name: SimpleNamespace(name=name, strength=Strength(name, writes),
               get_model_name=lambda model=model: model)
               for name, model in [('H1', 'combined'), ('V1', 'combined'), ('H2', 'standalone')]}
    bpms = SimpleNamespace(names=lambda: ['B1', 'B2'])
    bpms.positions = SimpleNamespace(get=lambda: np.array([
        [1e-6 + magnets['H1'].strength.value, 2e-6 + magnets['V1'].strength.value],
        [3e-6 + 2*magnets['H1'].strength.value, 4e-6 + 2*magnets['V1'].strength.value]]))
    matrix = Matrix()
    orbit = SimpleNamespace(bpms=bpms, response_matrix=object(), _pySC_response_matrix=matrix,
                            hcorrectors=Magnets([magnets['H1'], magnets['H2']]),
                            vcorrectors=Magnets([magnets['V1']]), calls=[])
    def correct(**kwargs):
        active = [name for i, name in enumerate(matrix.input_names) if i not in matrix.bad_inputs]
        assert len(active) == 1
        assert matrix.virtual_weight == 1e-16
        orbit.calls.append((active[0], kwargs))
        mag = magnets[active[0]]
        mag.strength.set(mag.strength.get() + 1e-7)
    orbit.correct = correct
    machine = SimpleNamespace(diagnostic=SimpleNamespace(bpms={'BPM': bpms}), magnet=magnets,
        magnets={'HCorr': Magnets([magnets['H1'], magnets['H2']]), 'VCorr': Magnets([magnets['V1']])},
        tool=SimpleNamespace(orbit=orbit))
    if mutate:
        mutate(machine, orbit, matrix)
    adapter = PyAMLAdapter('unused', accelerator=SimpleNamespace(design=machine), mode='design',
        rf_name='', allow_rf_writes=False, allow_writes=allow_writes, corrector_values_in_radians=True,
        orbit_correction_enabled=enabled)
    return adapter, orbit, matrix, writes


def devices(adapter):
    bpms = [BpmDevice(**item) for item in adapter.list_devices('bpm')]
    horizontal = [CorrectorDevice(**item) for item in adapter.list_devices('hcor')]
    vertical = [CorrectorDevice(**item) for item in adapter.list_devices('vcor')]
    return bpms, horizontal, vertical


def run(adapter, *, vertical=False, **kwargs):
    bpms, h, v = devices(adapter)
    return ORMAcquirer(adapter, bpms, h[:1], v if vertical else []).acquire(
        [1e-5], [1e-5] if vertical else [], readings=2, delay_seconds=0, **kwargs)


@pytest.mark.parametrize('mutate,match', [
    (lambda m,o,r: setattr(o, 'response_matrix', None), 'valid loaded'),
    (lambda m,o,r: setattr(o, 'bpms', SimpleNamespace(names=lambda: ['B1','B2'])), 'same BPM'),
    (lambda m,o,r: r.matrix.__setitem__((0,0), np.nan), 'dimensions, values'),
    (lambda m,o,r: setattr(r, 'matrix', np.ones((2,3))), 'dimensions, values'),
    (lambda m,o,r: setattr(r, 'output_names', ['B2','B1','B2','B1']), 'ordering'),
    (lambda m,o,r: setattr(r, 'input_names', ['unknown','V1','H2']), 'mapping'),
    (lambda m,o,r: setattr(m.magnet['H2'], 'get_model_name', lambda: 'combined'), 'Ambiguous'),
    (lambda m,o,r: r.matrix.__setitem__((slice(None),0), 0), 'no usable'),
])
def test_invalid_configuration(mutate, match):
    with pytest.raises(ValueError, match=match):
        fixture(mutate=mutate)
    fixture(enabled=False, mutate=mutate)  # No correction validation in the legacy path.


def test_requires_writes():
    with pytest.raises(ValueError, match='requires corrector magnet writes'):
        fixture(allow_writes=False)


@pytest.mark.parametrize('indices', [[0], [1,0]])
def test_selection_rejected_before_writes(indices):
    adapter, _, _, writes = fixture()
    bpms, h, _ = devices(adapter)
    with pytest.raises(ValueError, match='complete configured BPM array'):
        ORMAcquirer(adapter, [bpms[i] for i in indices], h[:1], []).acquire([1e-5], [], readings=2)
    assert writes == []


def test_pair_reference_baselines_and_matrix_settings():
    adapter, orbit, matrix, writes = fixture()
    result = run(adapter, vertical=True)
    assert [name for name, _ in orbit.calls] == ['H1', 'V1', 'V1', 'H1']
    for i, (_, args) in enumerate(orbit.calls):
        np.testing.assert_allclose(args['reference'], result.raw_reference_before_m[i//2].mean(axis=0))
        assert args['rf'] is False
        assert args['singular_values_h'] == 2
        assert args['singular_values_v'] == 1
    assert writes[:4] == [('H1', 5e-6), ('H1', 0.), ('H1', -5e-6), ('H1', 0.)]
    assert result.original_setpoints_rad[1] == pytest.approx(1e-7)
    assert adapter.machine.magnet['H1'].strength.get() == pytest.approx(2e-7)
    assert adapter.machine.magnet['V1'].strength.get() == pytest.approx(2e-7)
    assert adapter.machine.magnet['H2'].strength.get() == 0
    assert matrix.bad_inputs == [2] and matrix.virtual_weight == 3.
    assert len(result.orbit_correction_diagnostics) == 2
    assert result.restoration_status == ('restored', 'restored')


@pytest.mark.parametrize('name,expected', [('H1', ['H1','V1']), ('V1', ['V1','H1']), ('H2', ['H2'])])
def test_partner_selection(name, expected):
    adapter, orbit, _, _ = fixture()
    adapter.correct_orbit_after_column(name, np.zeros(4), cancel_event=Event(), settle=lambda: None, tolerance_rad=1e-12)
    assert [n for n, _ in orbit.calls] == expected


@pytest.mark.parametrize('cancelled', [False, True])
def test_partial_correction_rolls_back(cancelled):
    adapter, orbit, matrix, _ = fixture()
    cancel = Event()
    original = orbit.correct
    def fail(**kwargs):
        original(**kwargs)
        if cancelled:
            cancel.set()
        else:
            raise RuntimeError('injected failure')
    orbit.correct = fail
    with pytest.raises(ORMInterrupted if cancelled else ORMAcquisitionError) as caught:
        run(adapter, cancel_event=cancel)
    assert caught.value.restoration_status == 'restored'
    assert adapter.machine.magnet['H1'].strength.get() == 0
    assert adapter.machine.magnet['V1'].strength.get() == 0
    assert matrix.bad_inputs == [2] and matrix.virtual_weight == 3.
    assert caught.value.restoration_diagnostics['orbit_correction'][-1]['status'] == 'failed'


def test_later_failure_preserves_completed_correction():
    adapter, orbit, _, _ = fixture()
    original = orbit.correct
    def fail(**kwargs):
        original(**kwargs)
        if len(orbit.calls) == 3:
            raise RuntimeError('second column correction failed')
    orbit.correct = fail
    with pytest.raises(ORMAcquisitionError):
        run(adapter, vertical=True)
    for name in ['H1', 'V1']:
        assert adapter.machine.magnet[name].strength.get() == pytest.approx(1e-7)


def test_cancel_before_correction_never_corrects():
    adapter, orbit, _, _ = fixture()
    cancel = Event()
    with pytest.raises(ORMInterrupted):
        run(adapter, cancel_event=cancel, progress=lambda e: cancel.set() if e['event'] == 'final_restored' else None)
    assert not orbit.calls


def test_disabled_sequence_and_results():
    adapter, orbit, _, writes = fixture(enabled=False)
    result = run(adapter)
    assert not orbit.calls
    assert writes == [('H1', 5e-6), ('H1', 0.), ('H1', -5e-6), ('H1', 0.)]
    assert result.orbit_correction_diagnostics == ()
    assert result.restoration_status == ('restored',)
    np.testing.assert_allclose(result.response_matrix[:,0], [1e-5,2e-5,0,0], atol=1e-20)


def test_unrelated_change_is_rolled_back():
    adapter, orbit, _, _ = fixture()
    original = orbit.correct
    def faulty(**kwargs):
        original(**kwargs)
        adapter.machine.magnet['H2'].strength.set(5e-6)
    orbit.correct = faulty
    with pytest.raises(ORMAcquisitionError, match='unrelated'):
        run(adapter)
    assert all(m.strength.get() == 0 for m in adapter.machine.magnet.values())


def test_failed_rollback_is_reported_and_can_be_rechecked():
    adapter, orbit, _, _ = fixture()
    strength = adapter.machine.magnet['V1'].strength
    original_set = strength.set
    def failing_set(value):
        if value == 0:
            raise RuntimeError('rollback setter failed')
        original_set(value)
    strength.set = failing_set
    original = orbit.correct
    def faulty(**kwargs):
        original(**kwargs)
        if kwargs['plane'] == 'V':
            raise RuntimeError('partial failure')
    orbit.correct = faulty
    with pytest.raises(ORMAcquisitionError) as caught:
        run(adapter)
    assert caught.value.restoration_status == 'restore_failed'
    diagnostic = caught.value.restoration_diagnostics['orbit_correction'][-1]
    assert not adapter.verify_orbit_correction_restoration(diagnostic)
    original_set(0.)
    assert adapter.verify_orbit_correction_restoration(diagnostic)


def test_other_adapter_with_similar_attribute_is_unchanged():
    from test_measure_milestone4_orm import setup
    def acquire(flag):
        bpms, h, v, _, adapter = setup(2, 1, 0, 2)
        calls = []
        for method in ('read', 'read_many', 'write'):
            original = getattr(adapter, method)
            def recorded(*args, _method=method, _original=original, **kwargs):
                if _method == "read_many":
                    args = (tuple(args[0]), *args[1:])
                calls.append((_method, args))
                return _original(*args, **kwargs)
            setattr(adapter, method, recorded)
        if flag:
            adapter.orbit_correction_enabled = True
            adapter.correct_orbit_after_column = lambda *a, **k: pytest.fail('Non-pyAML correction called')
        result = ORMAcquirer(adapter, bpms, h, v).acquire([1e-5], [], readings=2, delay_seconds=0)
        return result, calls
    baseline, baseline_calls = acquire(False)
    result, calls = acquire(True)
    assert calls == baseline_calls
    np.testing.assert_array_equal(result.response_matrix, baseline.response_matrix)
    np.testing.assert_array_equal(result.final_setpoints_rad, baseline.final_setpoints_rad)
    assert result.orbit_correction_diagnostics == ()


@pytest.mark.parametrize('connection_method', ['_connect_pyaml', '_test_connection'])
def test_gui_checkbox_connection_and_persistence(monkeypatch, connection_method):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication, QMessageBox
    from pyLOCO.control_system import InterfaceRegistry
    from pyLOCO.measure.main_window import MeasureMainWindow
    from test_measure_pyaml import _wait_for_connection_test
    app = QApplication.instance() or QApplication([])
    window = MeasureMainWindow()
    options = []
    def create(registry, key):
        options.append(dict(registry.pyaml_options))
        adapter, _, _, _ = fixture(enabled=registry.pyaml_options['orbit_correction_enabled'])
        return SimpleNamespace(adapter=adapter)
    monkeypatch.setattr(InterfaceRegistry, 'create', create)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData('pyaml'))
        window.pyaml_fields['mode'].setText('design')
        window.pyaml_fields['config_path'].setText('unused.yaml')
        assert not window.pyaml_orbit_correction.isChecked()
        window.pyaml_orbit_correction.setChecked(True)
        getattr(window, connection_method)()
        if connection_method == '_test_connection':
            _wait_for_connection_test(app, window)
        assert options[-1]['orbit_correction_enabled'] is True
        assert window.connection_verified
        project = window._collect_project()
        assert project.metadata['pyaml_options']['orbit_correction_enabled'] is True
        window.pyaml_orbit_correction.setChecked(False)
        assert not window.connection_verified
        window.project = project
        window._load_project_widgets()
        assert window.pyaml_orbit_correction.isChecked()
        window.project.metadata['pyaml_options'].pop('orbit_correction_enabled')
        window._load_project_widgets()
        assert not window.pyaml_orbit_correction.isChecked()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize('enabled', [False, True])
def test_export_separates_correction_from_nominal_restore(tmp_path, enabled):
    import os
    import json
    import h5py
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from pyLOCO.measure.main_window import MeasureMainWindow
    app = QApplication.instance() or QApplication([])
    adapter, _, _, _ = fixture(enabled=enabled)
    result = run(adapter)
    bpms, h, v = devices(adapter)
    window = MeasureMainWindow(devices=bpms, adapter=adapter, horizontal_correctors=h, vertical_correctors=v)
    try:
        window.measurement_type.setCurrentIndex(2)
        window.output_directory.setText(str(tmp_path))
        window.measurement_name.setText('correction')
        window._save_result(result)
        with h5py.File(window.saved_measurement_path) as handle:
            metadata = json.loads(handle['metadata/json'][()])
            assert ('orbit_correction' in metadata) == enabled
            assert ('orbit_correction_enabled' in metadata) == enabled
            np.testing.assert_array_equal(handle['setpoints/final'][:], result.final_setpoints_rad)
            if enabled:
                assert metadata['orbit_correction'][0]['after_native'] == {'H1': 1e-7, 'V1': 1e-7}
                assert 'post-correction' in metadata['final_reference_role']
        text = window._orm_transaction_text(result, 0)
        assert ('retained strengths' in text) == enabled
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize('direction', ['positive', 'negative'])
def test_one_sided_correction_after_completed_column(direction):
    adapter, orbit, _, writes = fixture()
    result = run(adapter, direction=direction)
    sign = 1 if direction == 'positive' else -1
    assert writes[:3] == [('H1', sign*1e-5), ('H1', 0.), ('H1', 0.)]
    assert [name for name, _ in orbit.calls] == ['H1', 'V1']
    assert result.intermediate_restoration_status == ('not_applicable',)
    np.testing.assert_allclose(result.response_matrix[:,0], [sign*1e-5,sign*2e-5,0,0], atol=1e-20)


def test_cancelled_correction_before_first_call_makes_no_writes():
    adapter, orbit, _, writes = fixture()
    cancel = Event()
    cancel.set()
    with pytest.raises(PyAMLOrbitCorrectionError) as caught:
        adapter.correct_orbit_after_column('H1', np.zeros(4), cancel_event=cancel,
                                           settle=lambda: None, tolerance_rad=1e-12)
    assert caught.value.cancelled
    assert not orbit.calls and not writes
