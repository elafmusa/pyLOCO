from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from pyLOCO.control_system import AdapterCapability, MockAdapter
from pyLOCO.data_schema import load_session, validate_measurement_file
from pyLOCO.gui.backend import _load_measurements
from pyLOCO.measure.acquisition import AcquisitionCancelled, BpmDevice, BpmNoiseAcquirer, BpmNoiseResult
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow, build_mock_adapter, default_mock_devices
from pyLOCO.measure.project import MeasureProject, load_measure_project, save_measure_project


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or build_application(["pyloco-measure-test"])


def test_application_launches_offline_without_doocs(app):
    window = MeasureMainWindow(devices=default_mock_devices(3))
    assert window.windowTitle().startswith("pyLOCO Measure")
    assert window.status_badge.text() == "MOCK • READ ONLY"
    assert AdapterCapability.WRITE not in window.adapter.capabilities
    assert window.tabs.count() == 4
    assert "pydoocs" not in sys.modules and "doocs4py" not in sys.modules
    window.close()


def test_bpm_selection_preview_and_context_controls(app, tmp_path):
    window = MeasureMainWindow(devices=default_mock_devices(4))
    assert window.preview_table.rowCount() == 4
    window.selection_method.setCurrentIndex(window.selection_method.findData("manual"))
    window.manual_input.setText("BPM-004, 0")
    window.refresh_preview()
    assert window.preview_table.rowCount() == 2
    assert window.preview_table.item(0, 1).text() == "BPM-004"
    assert window.preview_table.item(1, 1).text() == "BPM-001"
    assert not window.manual_input.isHidden()
    assert window.names_row.isHidden()
    assert not window.manual_label.isHidden()
    assert window.names_label.isHidden()
    names = tmp_path / "bpms.txt"; names.write_text("BPM-002\nBPM-003\n")
    window.selection_method.setCurrentIndex(window.selection_method.findData("names_file"))
    window.names_file.setText(str(names)); window.refresh_preview()
    assert window.preview_table.rowCount() == 2
    window.close()


def test_deterministic_noise_progress_and_no_writes():
    devices = default_mock_devices(2)
    adapter = build_mock_adapter(devices, readings=4)
    assert [entry["name"] for entry in adapter.list_devices("bpm")] == ["BPM-001", "BPM-002"]
    progress = []
    result = BpmNoiseAcquirer(adapter, devices).acquire(
        4, 0, progress=lambda current, total, elapsed, x, y: progress.append((current, total)),
        sleeper=lambda _: None,
    )
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result.orbits_x_m.shape == (4, 2)
    assert np.all(result.noise_x_m > 0)
    assert all(operation == "read" for operation, _, _ in adapter.history)


def test_cancellation_does_not_return_partial_result():
    devices = default_mock_devices(1); adapter = build_mock_adapter(devices, readings=4); cancel = Event()
    def stop(_): cancel.set()
    with pytest.raises(AcquisitionCancelled):
        BpmNoiseAcquirer(adapter, devices).acquire(4, 0.1, cancel_event=cancel, sleeper=stop)


def test_saved_measurement_session_and_existing_importer(app, tmp_path):
    devices = default_mock_devices(2); adapter = build_mock_adapter(devices, readings=5)
    result = BpmNoiseAcquirer(adapter, devices).acquire(5, 0, sleeper=lambda _: None)
    window = MeasureMainWindow(devices=devices, adapter=adapter)
    window.project_path = tmp_path / "measure.pyloco-measure.json"
    window.output_directory.setText("session")
    window.measurement_name.setText("test-noise")
    window._save_result(result)
    assert validate_measurement_file(window.saved_measurement_path)["kind"] == "bpm_noise"
    session = load_session(window.saved_session_path)
    assert session.missing_roles == ("orm", "dispersion")
    with h5py.File(window.saved_measurement_path, "r") as handle:
        assert handle["raw/orbits_x_m"].shape == (5, 2)
        assert np.all(np.asarray(handle["Noise_BPMx"]) > 0)
    orm = tmp_path / "orm.h5"
    with h5py.File(orm, "w") as handle: handle["response_matrix"] = np.zeros((4, 2))
    loaded = _load_measurements({"orm": str(orm), "bpm_noise": str(window.saved_measurement_path)})
    np.testing.assert_allclose(loaded["noise_x"], result.noise_x_m)
    window.close()


def test_project_round_trip_is_portable(tmp_path):
    project = MeasureProject(measurement_name="noise", bpm_selection_method="manual",
                             bpm_manual="BPM-001, BPM-003", readings=12,
                             delay_seconds=.25, output_directory="output/session", theme="light")
    path = tmp_path / "mock.pyloco-measure.json"
    save_measure_project(path, project)
    assert load_measure_project(path) == project
    assert str(tmp_path) not in path.read_text()


def test_theme_switch_and_wheel_safety(app):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    window.apply_theme("light"); assert app.property("pyLOCOTheme") == "light"
    window.apply_theme("dark"); assert app.property("pyLOCOTheme") == "dark"
    original = window.readings.value()
    window.readings.clearFocus()
    event = QWheelEvent(QPointF(1,1), QPointF(1,1), QPoint(), QPoint(0,120),
                        Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    QApplication.sendEvent(window.readings, event)
    assert window.readings.value() == original
    window.close()


@pytest.mark.parametrize("size", [(1000, 700), (1200, 800), (1500, 900)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_responsive_measure_pages_do_not_overlap_or_elide(app, size, theme):
    window = MeasureMainWindow(devices=default_mock_devices(12))
    window.resize(*size); window.apply_theme(theme); window.show(); app.processEvents()
    tab_bar = window.tabs.tabBar()
    for index, expected in enumerate(("Machine", "BPMs", "Measurement", "Review & Save")):
        actual = window.tabs.tabText(index).replace("&&", "&").removeprefix("✓ ")
        assert actual == expected
        required = tab_bar.fontMetrics().horizontalAdvance(window.tabs.tabText(index)) + 32
        assert tab_bar.tabRect(index).width() >= required
    window.tabs.setCurrentIndex(0); app.processEvents()
    # PETRA read-only capability reporting adds four concise diagnostic rows;
    # the card should still retain its natural height rather than stretching.
    assert window.machine_group.height() < 430
    previous_bottom = -1
    for label, value in window.machine_rows:
        label_rect = label.geometry(); value_rect = value.geometry()
        assert label_rect.height() >= 36 and value_rect.height() >= 36
        assert label_rect.top() >= previous_bottom
        assert label_rect.right() < value_rect.left()
        previous_bottom = max(label_rect.bottom(), value_rect.bottom())
    window.tabs.setCurrentIndex(1); app.processEvents()
    assert window.preview_table.height() >= 260
    window.tabs.setCurrentIndex(2); app.processEvents()
    assert window.measurement_name.height() >= 30
    window.tabs.setCurrentIndex(3); app.processEvents()
    assert window.results_tabs.height() >= 410
    window.close()


def test_pysc_machine_profile_selector_uses_native_catalogs(app):
    window=MeasureMainWindow(devices=default_mock_devices(2))
    assert not window.pysc_profile_combo.isVisible()
    window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pysc"))
    window.show(); app.processEvents()
    assert window.pysc_profile_combo.isVisible()
    assert window.pysc_profile_combo.currentData()=="ebs"
    assert (len(window.devices),len(window.horizontal_correctors),len(window.vertical_correctors))==(320,288,288)
    window.pysc_profile_combo.setCurrentIndex(window.pysc_profile_combo.findData("petra3")); app.processEvents()
    assert (len(window.devices),len(window.horizontal_correctors),len(window.vertical_correctors))==(246,219,194)
    assert window.status_badge.text()=="DEMO • pySC SERVER"
    assert "PETRA III simulation" in window.machine_info["adapter"].text()
    window.pysc_profile_combo.setCurrentIndex(window.pysc_profile_combo.findData("petra3_realistic")); app.processEvents()
    assert (len(window.devices),len(window.horizontal_correctors),len(window.vertical_correctors))==(246,219,194)
    assert "realistic errors" in window.pysc_profile_combo.currentText()
    window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra")); app.processEvents()
    assert not window.pysc_profile_combo.isVisible()
    assert window.status_badge.text()=="LIVE • PETRA III DOOCS"
    window.close()


def test_measurement_plan_and_acquisition_status_are_structured(app):
    window = MeasureMainWindow(devices=default_mock_devices(3))
    assert {
        "measurement", "bpms", "readings", "delay", "est_duration", "adapter"
    }.issubset(window.plan_values)
    assert window.plan_values["measurement"].text() == "BPM noise"
    assert window.plan_values["bpms"].text() == "3"
    assert window.plan_output.toolTip() == str(window._resolved_output_directory())
    assert window.progress.minimumHeight() >= 34
    assert window.reading_status.text() == "Ready"
    assert not window.cancel_button.isEnabled()
    assert window.log_group.isCheckable() and not window.log_group.isChecked()
    assert window.log_body.isHidden()

    window._on_progress(7, 20, .35, np.zeros(3), np.ones(3) * 1e-6)
    assert window.reading_status.text() == "Reading 7 / 20"
    assert window.elapsed.text() == "Elapsed: 0.35 s"
    assert window.samples.text() == "Samples: 7 / 20"
    assert len(window.live_plot.figure.axes[0].lines) == 2
    window.close()


def test_measurement_tab_contains_visible_start_and_measurement_specific_plan(app):
    window=MeasureMainWindow(devices=default_mock_devices(3)); window.resize(1200,800); window.show(); window.tabs.setCurrentIndex(2); app.processEvents()
    assert window.run_group.isVisible() and window.start_button.isVisible()
    assert window.rect().contains(window.start_button.mapTo(window,window.start_button.rect().center()))
    assert window.start_button.isEnabled(); assert window.start_block_reason.text().startswith("Ready to acquire")
    assert all(widget.isHidden() for widget in window.dispersion_plan_widgets)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); window.nominal_rf.setText("500000000"); window.rf_step.setValue(1500); window.refresh_plan(); app.processEvents()
    assert all(not widget.isHidden() for widget in window.dispersion_plan_widgets)
    assert window.plan_values["rf_step_±δf"].text()=="750 Hz per side"
    assert window.plan_values["bipolar_separation"].text()=="1500 Hz total"
    assert window.plan_values["negative_rf"].text()=="499999250.000 Hz"
    assert window.plan_values["positive_rf"].text()=="500000750.000 Hz"
    window.close()


def test_acquisition_stays_on_measurement_tab_and_result_workspace_scrolls(app):
    window=MeasureMainWindow(devices=default_mock_devices(320)); window.resize(1200,800); window.show(); window.tabs.setCurrentIndex(2); app.processEvents()
    window._set_acquisition_running(True); app.processEvents()
    assert window.tabs.currentIndex()==2
    assert window.tabs.isTabEnabled(2)
    assert not window.tabs.isTabEnabled(0) and not window.tabs.isTabEnabled(1)
    scroll=window.acquisition_scroll
    assert scroll.verticalScrollBar().maximum()>0
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum()); app.processEvents()
    assert scroll.viewport().rect().intersects(window.paths.geometry().translated(window.paths.mapTo(scroll.viewport(),window.paths.rect().topLeft())-window.paths.rect().topLeft()))
    window.close()


def test_results_are_separate_and_workflow_completion_is_visible(app):
    devices = default_mock_devices(2)
    result = BpmNoiseAcquirer(build_mock_adapter(devices, readings=4), devices).acquire(
        4, 0, sleeper=lambda _: None
    )
    window = MeasureMainWindow(devices=devices)
    window._show_result(result)
    assert [window.results_tabs.tabText(i) for i in range(window.results_tabs.count())] == [
        "Horizontal BPM noise", "Vertical BPM noise",
        "Mean horizontal orbit", "Mean vertical orbit",
    ]
    assert window.summary_x.text().startswith("Horizontal:")
    assert window.summary_y.text().startswith("Vertical:")
    assert window.tabs.tabText(0).startswith("✓ ")
    assert window.tabs.tabText(1).startswith("✓ ")
    window.close()


def test_bpm_noise_uses_points_optional_backend_reference_and_simple_stats(app):
    devices=default_mock_devices(4)
    result=BpmNoiseAcquirer(build_mock_adapter(devices,readings=4),devices).acquire(4,0,sleeper=lambda _:None)
    window=MeasureMainWindow(devices=devices)
    window.adapter.backend_metadata={"configured_bpm_noise_sigma_x_m":1e-8,"configured_bpm_noise_sigma_y_m":1e-8}
    window._show_result(result)
    for canvas in (window.x_plot,window.y_plot):
        axis=canvas.figure.axes[0]
        assert axis.collections  # individual BPM scatter points
        assert any("σconfigured = 10 nm"==line.get_label() for line in axis.lines)
        assert axis.get_xlabel()=="BPM index / selection position"
    assert [part.split()[0] for part in window.summary_x.text().split(": ",1)[1].split(", ")]==["Mean","RMS","Min","Max"]
    window.close()


def test_micrometre_bpm_noise_uses_matching_dynamic_units(app):
    devices=default_mock_devices(4); adapter=build_mock_adapter(devices,readings=4)
    window=MeasureMainWindow(devices=devices,adapter=adapter)
    window.adapter.backend_metadata={"configured_bpm_noise_sigma_x_m":1.5e-6,"configured_bpm_noise_sigma_y_m":1.5e-6}
    base=np.arange(20,dtype=float)[:,None]*1.5e-6
    result=BpmNoiseResult(tuple(devices),np.repeat(base,4,axis=1),np.repeat(base,4,axis=1),0.0)
    window._show_result(result)
    for canvas in (window.x_plot,window.y_plot):
        axis=canvas.figure.axes[0]
        assert axis.get_title().endswith("[µm]")
        assert axis.get_ylabel()=="Noise [µm]"
        assert any(line.get_label()=="σconfigured = 1.5 µm" for line in axis.lines)
    assert window.summary_x.text().endswith("µm")
    window.close()
