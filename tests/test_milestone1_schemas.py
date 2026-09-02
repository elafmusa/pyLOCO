from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from pyLOCO.control_system import AdapterCapability, MockAdapter
from pyLOCO.data_schema import (
    CorrectionPlan,
    CorrectionRecord,
    MeasurementSession,
    SessionFile,
    load_correction_plan,
    load_session,
    save_correction_plan,
    save_session,
    validate_measurement_file,
    write_bpm_noise,
    write_dispersion,
    write_orm,
)
from pyLOCO.gui.backend import _load_measurements


def test_mock_adapter_is_deterministic_and_read_only_by_default():
    adapter = MockAdapter({"A": 3.0}, timestamp=10.0, timestamp_step=0.5)
    assert adapter.capabilities == frozenset({AdapterCapability.READ, AdapterCapability.BATCH_READ})
    assert adapter.read("A").timestamp == 10.0
    assert adapter.read("A").timestamp == 10.5
    with pytest.raises(PermissionError, match="write"):
        adapter.write("A", 4.0)


def test_mock_adapter_simulated_write_is_explicit_and_in_memory():
    adapter = MockAdapter({"A": 1}, allow_simulated_writes=True)
    assert AdapterCapability.WRITE in adapter.capabilities
    adapter.write("A", 2)
    assert adapter.read("A").value == 2
    assert adapter.history == [("write", "A", 2), ("read", "A", 2)]


def _measurement_files(tmp_path):
    orm = tmp_path / "orm.h5"
    write_orm(
        orm,
        response_matrix=np.arange(24, dtype=float).reshape(6, 4),
        bpm_names=["B1", "B2", "B3"],
        horizontal_corrector_names=["H1", "H2"],
        vertical_corrector_names=["V1", "V2"],
        requested_kick_h_rad=np.array([1e-4, 2e-4]),
        requested_kick_v_rad=np.array([3e-4, 4e-4]),
        actual_kick_h_rad=np.array([0.9e-4, 1.9e-4]),
        actual_kick_v_rad=np.array([2.9e-4, 3.9e-4]),
        orbit_plus_m=np.ones((4, 6)) * 2e-6,
        orbit_minus_m=np.ones((4, 6)) * -2e-6,
        metadata={"machine": "mock-ring"},
    )
    noise = tmp_path / "noise.h5"
    write_bpm_noise(
        noise,
        noise_x_m=[1e-7, 2e-7, 3e-7],
        noise_y_m=[4e-7, 5e-7, 6e-7],
        bpm_names=["B1", "B2", "B3"],
        raw_orbits_x_m=np.zeros((5, 3)),
        raw_orbits_y_m=np.ones((5, 3)) * 1e-7,
    )
    dispersion = tmp_path / "dispersion.h5"
    write_dispersion(
        dispersion,
        measured_eta_x=[0.1, 0.2, 0.3],
        measured_eta_y=[0.01, 0.02, 0.03],
        bpm_names=["B1", "B2", "B3"],
        rf_frequency_hz=[499_999_000.0, 500_001_000.0],
        rf_setpoint_hz=[499_999_000.0, 500_001_000.0],
        rf_readback_hz=[499_999_000.2, 500_000_999.8],
        raw_orbits_x_m=[[1e-6, 2e-6, 3e-6], [-1e-6, -2e-6, -3e-6]],
        raw_orbits_y_m=[[2e-7, 3e-7, 4e-7], [-2e-7, -3e-7, -4e-7]],
        rf_step_hz=2000.0,
        bidirectional=True,
    )
    return orm, noise, dispersion


def test_measurement_schema_preserves_raw_states_and_importer_compatibility(tmp_path):
    orm, noise, dispersion = _measurement_files(tmp_path)
    assert validate_measurement_file(orm)["kind"] == "orm"
    assert validate_measurement_file(noise)["kind"] == "bpm_noise"
    assert validate_measurement_file(dispersion)["kind"] == "dispersion"
    with h5py.File(orm, "r") as handle:
        np.testing.assert_allclose(handle["kicks/horizontal/requested"], [1e-4, 2e-4])
        np.testing.assert_allclose(handle["kicks/horizontal/actual"], [0.9e-4, 1.9e-4])
        assert handle["raw/orbit_plus_m"].shape == (4, 6)
    with h5py.File(dispersion, "r") as handle:
        assert handle["raw/orbits_x_m"].shape == (2, 3)
        np.testing.assert_allclose(handle["raw/rf_readback_hz"], [499_999_000.2, 500_000_999.8])
        np.testing.assert_allclose(handle["measured_eta_x"], [0.1, 0.2, 0.3])
    loaded = _load_measurements(
        {"orm": str(orm), "bpm_noise": str(noise), "dispersion": str(dispersion)}
    )
    assert loaded["orm"].shape == (6, 4)
    np.testing.assert_allclose(loaded["eta_y"], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(loaded["noise_x"], [1e-7, 2e-7, 3e-7])


def test_measurement_session_is_portable_and_populates_three_gui_roles(tmp_path):
    orm, noise, dispersion = _measurement_files(tmp_path)
    manifest = tmp_path / "session.pyloco-session.json"
    session = MeasurementSession(
        session_id="mock-2026-08-28",
        files=(
            SessionFile("orm", orm.name, {"dataset": "response_matrix"}),
            SessionFile("bpm_noise", noise.name),
            SessionFile("dispersion", dispersion.name),
        ),
        metadata={"optics": "mock"},
    )
    save_session(manifest, session)
    restored = load_session(manifest)
    assert restored == session
    mapping = restored.to_gui_measurements(manifest)
    assert set(mapping) == {"orm", "bpm_noise", "dispersion"}
    assert mapping["orm"]["path"] == str(orm.resolve())
    text = manifest.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    with pytest.raises(ValueError, match="relative"):
        SessionFile("orm", str(orm.resolve())).validate()


def test_correction_plan_keeps_each_stage_and_types_separate(tmp_path):
    records = (
        CorrectionRecord("normal_quadrupole", "Q1", 10, "m^-2", 0.5, -0.02, 0.02,
                         individual_scale=0.5, final_applied_delta=0.005),
        CorrectionRecord("skew_quadrupole", "SQ1", 12, "m^-2", 0.0, 0.001, -0.001),
        CorrectionRecord("quadrupole_tilt", "Q2", 14, "rad", 0.0, 0.002, -0.002),
    )
    comparison = tuple({
        "fraction": fraction,
        "max_abs_delta_k_over_k_percent": 4.0 * fraction,
        "max_abs_delta_i_ampere": 2.0 * fraction,
        "current_limit_violations": 0,
    } for fraction in (0.1, 0.25, 0.5, 1.0))
    plan = CorrectionPlan(
        "plan-1", "results/run-1", records, global_scale=0.5,
        fraction_comparison=comparison,
    )
    destination = tmp_path / "correction-plan.json"
    save_correction_plan(destination, plan)
    restored = load_correction_plan(destination)
    assert restored == plan
    assert restored.records_by_type("normal_quadrupole")[0].raw_fitted_delta == -0.02
    assert restored.records_by_type("normal_quadrupole")[0].recommended_machine_delta == 0.02
    assert restored.records_by_type("normal_quadrupole")[0].final_delta(0.5) == 0.005
    assert {record.correction_type for record in restored.records} == {
        "normal_quadrupole", "skew_quadrupole", "quadrupole_tilt"
    }
    assert json.loads(destination.read_text())["application_state"] == "dry_run"


def test_schema_validation_rejects_inconsistent_final_correction():
    record = CorrectionRecord(
        "normal_quadrupole", "Q", 1, "m^-2", 1.0, 0.1, -0.1,
        final_applied_delta=99.0,
    )
    with pytest.raises(ValueError, match="does not equal"):
        CorrectionPlan("x", "result", (record,), global_scale=0.5).validate()
