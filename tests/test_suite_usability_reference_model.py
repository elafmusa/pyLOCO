from pathlib import Path
import os
import numpy as np
import h5py
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.measurement_metadata import measurement_display_fields
from pyLOCO.gui.themes import apply_application_theme, theme_for_key
from pyLOCO.gui.appearance import select_suite_accent, suite_appearance_settings
from pyLOCO.measure.reference_model import comparison_metrics, model_dispersion, model_orm, reference_model_for_pysc, store_reference_model_arrays

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["suite-usability-test"])


def test_fit_accent_is_persisted_independently(qapp):
    settings=suite_appearance_settings(); old_fit=settings.value("fit/appearance/accent",None); old_suite=settings.value("appearance/accent",None)
    try:
        settings.setValue("appearance/accent","teal")
        settings.setValue("fit/appearance/accent","blue"); settings.sync()
        assert settings.value("fit/appearance/accent")=="blue"
        assert settings.value("appearance/accent")=="teal"
        apply_application_theme(qapp,theme_for_key("light"),"graphite")
        assert qapp.property("pyLOCOAccent")=="graphite"
        assert qapp.property("pyLOCOThemePlot")["colormap"]=="viridis"
    finally:
        settings.setValue("fit/appearance/accent",old_fit or "purple"); settings.setValue("appearance/accent",old_suite or "purple"); settings.sync()


def test_measurement_timestamp_metadata_precedes_file_time(tmp_path):
    path=tmp_path/"measurement.h5"
    with h5py.File(path,"w") as handle:
        handle.attrs["acquisition_timestamp_utc"]="2026-09-08T12:32:18+00:00"
        handle.attrs["machine_profile"]="PETRA III / realistic_errors"
    fields=measurement_display_fields(path)
    assert fields["timestamp_source"]=="HDF5 acquisition metadata"
    assert fields["machine_profile"]=="PETRA III / realistic_errors"
    fallback=tmp_path/"fallback.h5"
    with h5py.File(fallback,"w"):pass
    fields=measurement_display_fields(fallback)
    assert fields["timestamp_source"]=="file modification time (fallback)"
    assert fields["date"].endswith(" *")


def test_reference_profiles_and_selected_device_ordering():
    model=reference_model_for_pysc("ebs")
    from types import SimpleNamespace as D
    bpms=[D(name="second",identifier="2"),D(name="first",identifier="1")]
    dx,_=model_dispersion(model,bpms)
    _,_,data=model.ring.get_optics(refpts=[1,2])
    assert np.array_equal(dx,np.asarray(data.dispersion)[::-1,0])
    orm=model_orm(model,bpms,[D(name="h",identifier="6/B1L")],[D(name="v",identifier="6/A1L")],[1e-4],[2e-4],scaled=False)
    assert orm.shape==(4,2)
    scaled=model_orm(model,bpms,[D(name="h",identifier="6/B1L")],[D(name="v",identifier="6/A1L")],[1e-4],[2e-4],scaled=True)
    np.testing.assert_allclose(scaled,orm/np.array([1e-4,2e-4])[None,:])
    provenance=model.provenance()
    assert provenance["reference_model_sha256"] and provenance["model_slip_factor"]==pytest.approx(model.momentum_compaction-model.inverse_gamma_squared)
    assert model.energy_ev > 0 and model.circumference_m > 0
    assert model.harmonic_number is not None and model.nominal_rf_hz is not None
    assert model.nominal_rf_hz == pytest.approx(model.harmonic_number*model.revolution_frequency_hz, abs=1e-6)
    assert model.rf_harmonic_residual_hz == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.isfinite(model.chromaticity))


def test_comparison_does_not_mutate_measured_data():
    measured=np.arange(12,dtype=float).reshape(4,3); before=measured.copy()
    metrics=comparison_metrics(measured,measured*1.01)
    assert np.array_equal(measured,before)
    assert metrics["cosine_similarity"]==pytest.approx(1.0)


def test_model_arrays_are_separate_and_units_match(tmp_path):
    path=tmp_path/"m.h5"; acquired=np.arange(6.0).reshape(3,2)
    with h5py.File(path,"w") as handle:handle.create_dataset("response_matrix",data=acquired)
    model=acquired*.9
    store_reference_model_arrays(path,{"orm":model,"orm_difference":acquired-model},units={"orm":"m/rad","orm_difference":"m/rad"},provenance={"reference_model_source":"test"})
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["response_matrix"][:],acquired)
        np.testing.assert_array_equal(handle["reference_model/orm"][:],model)
        assert handle["reference_model/orm"].attrs["unit"]=="m/rad"


def test_legacy_measure_project_migration_keeps_new_reference_defaults(tmp_path):
    import json
    from pyLOCO.measure.project import load_measure_project
    path=tmp_path/"legacy.json"; path.write_text(json.dumps({"file_type":"pyloco.measure_project","schema_version":"1.0","measurement_type":"bpm_noise","measurement_name":"old","measurement_label":"old","adapter":"Mock","output_directory":"measurements"}))
    project=load_measure_project(path)
    assert project.reference_model_path=="" and project.reference_model_source=="automatic" and project.compare_with_reference_model


def test_measure_labels_and_pre_orm_correctors(qapp):
    from pyLOCO.measure.main_window import MeasureMainWindow,default_mock_devices
    window=MeasureMainWindow(devices=default_mock_devices(5))
    assert window.orm_corrector_group.isEnabled()
    assert window.rf_step_label.text()=="Total RF separation Δ"
    assert "f₀ +" in window.rf_states_explanation.text()
    assert "K₀ + Δ/2" in window.orm_state_explanation.text()
    visible=" ".join(widget.text() for widget in window.findChildren(type(window.heading)) if hasattr(widget,"text"))
    assert "Operator" not in visible and "operator" not in visible
    window.close()


def test_live_backend_never_inherits_profile_reference_model(qapp):
    from pyLOCO.measure.main_window import MeasureMainWindow,default_mock_devices
    window=MeasureMainWindow(devices=default_mock_devices(3)); window.reference_model=reference_model_for_pysc("ebs"); window._display_reference_model()
    window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra")); qapp.processEvents()
    assert window.reference_model is None and window.reference_model_path.text()==""
    assert "Not available" in window.reference_model_summary.text()
    window.close()
