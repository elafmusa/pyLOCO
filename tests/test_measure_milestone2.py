from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from pyLOCO.control_system import AdapterCapability, MockAdapter
from pyLOCO.data_schema import MeasurementSession, SessionFile, load_session, save_session, validate_measurement_file, write_bpm_noise
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
    assert window.plan_group.isCheckable() and not window.plan_group.isChecked()
    assert window.plan_body.isHidden()
    window.plan_group.setChecked(True)
    assert not window.plan_body.isHidden()
    assert "pydoocs" not in sys.modules and "doocs4py" not in sys.modules
    window.close()


def test_new_measurement_default_delay_is_one_second_but_saved_value_is_preserved(app, tmp_path):
    assert MeasureProject().delay_seconds == 1.0
    window = MeasureMainWindow(devices=default_mock_devices(2))
    assert window.delay.value() == 1.0
    path = tmp_path / "saved-delay.pyloco-measure.json"
    save_measure_project(path, MeasureProject(delay_seconds=0.25))
    window.project = load_measure_project(path)
    window.project_path = path
    window._load_project_widgets()
    assert window.delay.value() == 0.25
    window.close()


def test_delay_plan_format_is_theme_independent(app):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    window.delay.setValue(1.0)
    for theme in ("light", "dark"):
        window.apply_theme(theme)
        window.refresh_plan()
        assert window.plan_values["delay"].text() == "1 s"
    window.close()


def test_persistent_status_header_is_visible_and_paths_are_tooltipped(app, tmp_path):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    window.project_path = tmp_path / "a-very-long-configuration-name.pyloco-measure.json"
    window.output_directory.setText("nested/output")
    window._update_persistent_header()
    for widget in (window.status_badge, window.header_mode, window.connection_badge,
                   window.header_capabilities, window.header_access, window.machine_identity_badge,
                   window.header_configuration, window.header_output):
        assert not widget.isHidden()
    assert window.header_configuration.toolTip() == str(window.project_path)
    assert window.header_output.toolTip().endswith("nested/output")
    window.tabs.setCurrentIndex(3); app.processEvents()
    assert not window.header_configuration.isHidden()
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
    assert result.ddof == 0
    np.testing.assert_allclose(result.noise_x_m, np.std(result.orbits_x_m, axis=0, ddof=0))
    np.testing.assert_allclose(result.noise_y_m, np.std(result.orbits_y_m, axis=0, ddof=0))
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
    window.result = result
    window._save_result(result)
    assert validate_measurement_file(window.saved_measurement_path)["kind"] == "bpm_noise"
    session = load_session(window.saved_session_path)
    assert session.missing_roles == ("orm", "dispersion")
    with h5py.File(window.saved_measurement_path, "r") as handle:
        assert handle["raw/orbits_x_m"].shape == (5, 2)
        assert np.all(np.asarray(handle["Noise_BPMx"]) > 0)
        assert handle.attrs["noise_statistic"] == "population_standard_deviation"
        assert handle.attrs["noise_standard_deviation_ddof"] == 0
        assert handle["Noise_BPMx"].attrs["unit"] == "m"
        assert handle["raw"].attrs["dimensions"] == "sample,bpm"
        metadata = __import__("json").loads(handle["metadata/json"][()])
        assert metadata["selected_bpm_names"] == [device.name for device in devices]
        assert metadata["bpm_noise_ddof"] == 0
    orm = tmp_path / "orm.h5"
    with h5py.File(orm, "w") as handle: handle["response_matrix"] = np.zeros((4, 2))
    loaded = _load_measurements({"orm": str(orm), "bpm_noise": str(window.saved_measurement_path)})
    np.testing.assert_allclose(loaded["noise_x"], result.noise_x_m)
    window.tabs.setCurrentIndex(3); app.processEvents()
    assert window.review_status.text() == "✓ BPM noise saved — measurement session incomplete"
    assert "<b>BPM noise</b>" in window.review_details.text() and "Present" in window.review_details.text()
    assert "<b>Dispersion</b>" in window.review_details.text() and "<b>ORM</b>" in window.review_details.text()
    assert "Missing for FIT: orm, dispersion" in window.review_details.text()
    assert str(window.saved_measurement_path) in window.review_paths.text()
    assert "Automatically saved session manifest" in window.review_paths.text()
    assert window.project_save_note.text()=="Project = settings only"
    assert "does not create or replace measurement HDF5 files" in window.project_save_note.toolTip()
    assert window.review_validate_button.isEnabled() and window.review_open_button.isEnabled()
    assert window.open_output_button.isEnabled()
    window.close()


def test_review_page_reports_complete_automatic_session_without_extra_save_action(app,tmp_path):
    devices=default_mock_devices(2); adapter=build_mock_adapter(devices,readings=3)
    result=BpmNoiseAcquirer(adapter,devices).acquire(3,0,sleeper=lambda _:None)
    files=tuple(SessionFile(role,f"{role}.h5") for role in ("bpm_noise","dispersion","orm"))
    manifest=tmp_path/"measurement-session.pyloco-session.json"
    save_session(manifest,MeasurementSession("complete",files),validate_files=False)
    window=MeasureMainWindow(devices=devices,adapter=adapter)
    window.result=result; window.saved_measurement_path=tmp_path/"bpm_noise.h5"; window.saved_session_path=manifest
    window._refresh_review_page()
    assert window.review_status.text()=="✓ Measurement session complete for FIT"
    assert window.review_details.text().count("Present ✓")==3
    assert "Complete for FIT" in window.review_details.text()
    assert str(manifest) in window.review_paths.text()
    assert window.review_open_button.text()=="Open session in pyLOCO Fit"
    window.close()


def test_bpm_noise_rejects_insufficient_and_nonfinite_readings():
    devices = default_mock_devices(1)
    adapter = build_mock_adapter(devices, readings=2)
    with pytest.raises(ValueError, match="at least two"):
        BpmNoiseAcquirer(adapter, devices).acquire(1, 0, sleeper=lambda _: None)
    channel = devices[0].x_channel
    adapter._sequences[channel] = (0.0, np.nan)
    with pytest.raises(ValueError, match="Non-finite BPM data.*2"):
        BpmNoiseAcquirer(adapter, devices).acquire(2, 0, sleeper=lambda _: None)


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


def test_new_measure_window_respects_active_application_theme(app):
    from pyLOCO.gui.themes import apply_application_theme, theme_for_key
    apply_application_theme(app,theme_for_key("light"),"teal")
    window=MeasureMainWindow(devices=default_mock_devices(2))
    assert window.project.theme == "light"
    assert app.property("pyLOCOTheme") == "light"
    window.close()


@pytest.mark.parametrize("size", [(1000, 700), (1200, 800), (1600, 1000)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_responsive_measure_pages_do_not_overlap_or_elide(app, size, theme):
    window = MeasureMainWindow(devices=default_mock_devices(12))
    window.resize(*size); window.apply_theme(theme); window.show(); app.processEvents()
    tab_bar = window.tabs.tabBar()
    for index, expected in enumerate(("Machine", "Devices", "Measurement", "Review & Save")):
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
    # The result area is intentionally compact (and reached through the one
    # outer workspace scroll area) so controls remain usable at 1000x700.
    assert window.results_tabs.height() >= 320
    window.close()


def test_light_theme_contrast_tokens_and_measure_surfaces(app):
    from pyLOCO.gui.themes import LIGHT_THEME
    from pyLOCO.measure.main_window import measure_stylesheet

    def luminance(color):
        values=[int(color[index:index+2],16)/255 for index in (1,3,5)]
        linear=[value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4 for value in values]
        return .2126*linear[0]+.7152*linear[1]+.0722*linear[2]

    def contrast(foreground,background):
        light,dark=sorted((luminance(foreground),luminance(background)),reverse=True)
        return (light+.05)/(dark+.05)

    for foreground,background in (("#222436","#FFFFFF"),("#4D586D","#F7F8FB"),
                                  ("#5F687A","#EEF0F5"),("#687386","#FFFFFF"),
                                  ("#4D5568","#F3F4F8"),("#5E3EA1","#F7F8FB")):
        assert contrast(foreground,background) >= 4.5
    assert "placeholder-text-color: #687386" in LIGHT_THEME.stylesheet
    assert "QToolTip { background:#222436; color:#FFFFFF" in LIGHT_THEME.stylesheet
    measure_qss=measure_stylesheet("light")
    assert "rgba(" not in measure_qss
    assert "QLabel#summaryCard" in measure_qss and "QFrame#persistentStatus" in measure_qss


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
        assert axis.get_xlabel()=="BPM index (zero-based)"
    assert [part.split()[0] for part in window.summary_x.text().split(": ",1)[1].split(", ")]==["Mean","RMS","Min","Max"]
    assert window.statistics_table.rowCount() == 2
    assert window.statistics_table.item(0,0).text() == "Horizontal (X)"
    assert window.statistics_table.item(1,0).text() == "Vertical (Y)"
    assert [window.statistics_table.item(0,column).text() for column in range(1,5)] == [
        f"{value:.3f}" for value in (
            np.mean(result.noise_x_m)*1e9,
            np.sqrt(np.mean(result.noise_x_m**2))*1e9,
            np.min(result.noise_x_m)*1e9,
            np.max(result.noise_x_m)*1e9,
        )
    ]
    assert "[nm]" in window.statistics_table.horizontalHeaderItem(1).text()
    assert window.statistics_table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    expected_height = (
        window.statistics_table.horizontalHeader().sizeHint().height()
        + sum(window.statistics_table.verticalHeader().sectionSize(row) for row in range(2))
        + 2 * window.statistics_table.frameWidth()
    )
    assert window.statistics_table.height() >= expected_height
    assert "ddof=0" in window.orm_measurement_summary.text()
    assert "4 BPMs · 4 readings" in window.orm_measurement_summary.text()
    window.close()


def test_optional_sample_bpm_noise_convention_and_project_round_trip(app, tmp_path):
    devices=default_mock_devices(2); adapter=build_mock_adapter(devices,readings=4)
    result=BpmNoiseAcquirer(adapter,devices,ddof=1).acquire(4,0,sleeper=lambda _:None)
    assert result.ddof == 1
    np.testing.assert_allclose(result.noise_x_m,np.std(result.orbits_x_m,axis=0,ddof=1))
    window=MeasureMainWindow(devices=devices,adapter=adapter)
    window.bpm_noise_convention.setCurrentIndex(window.bpm_noise_convention.findData(1))
    assert window._collect_project().bpm_noise_ddof == 1
    path=tmp_path/"sample.pyloco-measure.json"; save_measure_project(path,window._collect_project())
    assert load_measure_project(path).bpm_noise_ddof == 1
    window.close()


def test_completed_measurement_uses_outer_scroll_without_section_overlap(app):
    devices=default_mock_devices(8)
    result=BpmNoiseAcquirer(build_mock_adapter(devices,readings=4),devices).acquire(4,0,sleeper=lambda _:None)
    window=MeasureMainWindow(devices=devices)
    window.resize(1200,800); window.tabs.setCurrentIndex(2)
    window.result=result; window._show_result(result); window._set_completed_result_view(True)
    window.show(); app.processEvents(); app.processEvents()
    content=window.acquisition_scroll.widget()
    sections=(window.results_tabs,window.stats_group,window.save_group)
    bounds=[]
    for widget in sections:
        top=widget.mapTo(content,widget.rect().topLeft()).y()
        bounds.append((top,top+widget.height()))
    assert bounds[1][0] >= bounds[0][1]
    assert bounds[2][0] >= bounds[1][1]
    assert window.acquisition_scroll.verticalScrollBar().maximum() > 0
    assert not window.statistics_table.verticalScrollBar().isVisible()
    assert window.statistics_table.rowCount() == 2
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


def _identified_bpm_result():
    devices=(BpmDevice("EBS-BPM-Z","z/x","z/y"),BpmDevice("EBS-BPM-A","a/x","a/y"),BpmDevice("EBS-BPM-M","m/x","m/y"))
    x=np.asarray([[0,0,0],[1e-9,2e-9,4e-9],[2e-9,4e-9,8e-9]])
    y=np.asarray([[0,0,0],[3e-9,5e-9,7e-9],[6e-9,10e-9,14e-9]])
    return BpmNoiseResult(devices,x,y,0.0)


def _plot_event(canvas,index,value,name):
    canvas.canvas.draw()
    x,y=canvas.figure.axes[0].transData.transform((index,value))
    return MouseEvent(name,canvas.canvas,x,y,button=1 if name=="button_press_event" else None)


def test_bpm_plot_hover_uses_original_point_to_device_mapping(app):
    result=_identified_bpm_result()
    window=MeasureMainWindow(devices=result.devices)
    assert hasattr(window,"_bpm_plot_state"),sys.modules[MeasureMainWindow.__module__].__file__
    window.result=result
    window._show_result(result)
    state=window._bpm_plot_state["y"]; event=_plot_event(window.y_plot,2,state["values"][2],"motion_notify_event")
    window._on_bpm_plot_hover("y",event)
    text=state["annotation"].get_text()
    assert state["annotation"].get_visible()
    assert text.count("\n")==2
    assert "EBS-BPM-M" in text and "BPM index 2 (zero-based)" in text
    assert f"X noise {result.noise_x_m[2]*1e9:.3f} nm" in text
    assert f"Y noise {result.noise_y_m[2]*1e9:.3f} nm" in text
    window.close()


def test_bpm_hover_tooltip_is_placed_inward_at_plot_edge(app):
    result=_identified_bpm_result(); window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result)
    state=window._bpm_plot_state["y"]; event=_plot_event(window.y_plot,2,state["values"][2],"motion_notify_event"); window._on_bpm_plot_hover("y",event)
    annotation=state["annotation"]; window.y_plot.canvas.draw(); renderer=window.y_plot.canvas.get_renderer()
    assert annotation.get_horizontalalignment()=="right"
    assert annotation.get_verticalalignment()=="top"
    assert state["axis"].bbox.contains(*annotation.get_window_extent(renderer).get_points()[0])
    assert state["axis"].bbox.contains(*annotation.get_window_extent(renderer).get_points()[1])
    window.close()


def test_bpm_plot_click_highlights_and_keeps_details_on_both_planes(app):
    result=_identified_bpm_result()
    window=MeasureMainWindow(devices=result.devices)
    assert hasattr(window,"_bpm_plot_state"),sys.modules[MeasureMainWindow.__module__].__file__
    window.result=result
    window._show_result(result)
    state=window._bpm_plot_state["x"]; window._on_bpm_plot_click("x",_plot_event(window.x_plot,1,state["values"][1],"button_press_event"))
    assert window._selected_bpm_index==1
    assert "EBS-BPM-A" in window.bpm_review_details.text()
    for plane in ("x","y"):
        offsets=window._bpm_plot_state[plane]["highlight"].get_offsets()
        assert int(offsets[0,0])==1
    # Sorting the linked table cannot change the immutable original index mapping.
    window.bpm_review_table.sortItems(1,Qt.AscendingOrder)
    selected=window.bpm_review_table.selectionModel().selectedRows()[0].row()
    assert window._bpm_review_index_for_row(selected)==1
    window.close()


def test_bpm_hover_annotation_tracks_light_and_dark_plot_theme(app):
    result=_identified_bpm_result(); window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result)
    annotation=window._bpm_plot_state["x"]["annotation"]
    for theme_key in ("light","dark"):
        window.apply_theme(theme_key)
        theme=window.x_plot.theme()
        assert annotation.get_color()==theme["text"]
        assert annotation.get_bbox_patch().get_facecolor()[:3]==pytest.approx(__import__("matplotlib").colors.to_rgb(theme["face"]))
    window.close()


def test_multiple_bpm_pins_persist_while_ordinary_selection_moves(app):
    result=_identified_bpm_result(); window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result)
    window._select_bpm(0,source="plot"); window._toggle_selected_bpm_pin()
    window._select_bpm(2,source="plot"); window._toggle_selected_bpm_pin()
    window._select_bpm(1,source="plot")
    assert window._selected_bpm_index==1
    assert window._pinned_bpm_indices=={0,2}
    assert window.bpm_review_table.rowCount()==3
    assert not hasattr(window,"bpm_highlighted_table")
    for row in range(3):
        index=window._bpm_review_index_for_row(row)
        assert (window.bpm_review_table.item(row,3).checkState()==Qt.Checked)==(index in {0,2})
    for state in window._bpm_plot_state.values():
        assert {int(offset[0]) for offset in state["pinned"].get_offsets()}=={0,2}
        assert len(state["pin_lines"])==2
        assert int(state["highlight"].get_offsets()[0,0])==1
    window.close()


def test_unified_review_table_sorting_notes_and_pin_removal_keep_original_identity(app):
    result=_identified_bpm_result(); window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result)
    for index in (0,2):window._select_bpm(index,source="plot"); window._toggle_selected_bpm_pin()
    window.bpm_review_table.sortItems(0,Qt.AscendingOrder)
    row=next(row for row in range(3) if window._bpm_review_index_for_row(row)==2)
    window.bpm_review_table.selectRow(row); window._bpm_review_table_selection_changed()
    window.bpm_review_note.setText("Review electronics")
    assert window._selected_bpm_index==2 and window._bpm_review_notes[2]=="Review electronics"
    window.bpm_review_table.item(row,3).setCheckState(Qt.Unchecked)
    assert window._pinned_bpm_indices=={0}
    window._clear_bpm_pins(); assert not window._pinned_bpm_indices
    window.close()


def test_bpm_review_panel_saves_and_reopens_status_highlight_note_and_thresholds(app,tmp_path):
    result=_identified_bpm_result(); path=tmp_path/"review-reopen.h5"
    write_bpm_noise(path,noise_x_m=result.noise_x_m,noise_y_m=result.noise_y_m,bpm_names=[device.name for device in result.devices],raw_orbits_x_m=result.orbits_x_m,raw_orbits_y_m=result.orbits_y_m)
    window=MeasureMainWindow(devices=result.devices); window.saved_measurement_path=path; window.result=result; window._show_result(result); window._select_bpm(1,source="plot")
    assert window.bpm_review_highlight.text()=="Keep highlighted on plots"
    assert not window.bpm_advanced_controls.isVisible() and window.bpm_threshold_x.value()==0 and window.bpm_threshold_y.value()==0
    window.bpm_review_status.setCurrentText("Suspect"); window.bpm_review_highlight.setChecked(True); window.bpm_review_note.setText("Check cable during next fill"); window.bpm_threshold_x.setValue(1.1)
    assert window._bpm_review_dirty and window.bpm_review_save_state.text()=="Unsaved review changes" and window.bpm_save_review_button.isEnabled()
    row=next(row for row in range(3) if window._bpm_review_index_for_row(row)==1)
    assert window.bpm_review_table.item(row,4).text()=="Suspect"
    assert window.bpm_review_table.horizontalHeader().sectionSize(4) >= window.bpm_review_table.fontMetrics().horizontalAdvance("Unreviewed")
    assert window.bpm_review_table.item(row,5).text()=="Check cable during next fill"
    assert window.bpm_review_table.item(row,5).toolTip()=="Check cable during next fill"
    assert window.bpm_review_table.item(row,0).text()=="EBS-BPM-A  [1]"
    assert window.bpm_review_table.item(row,0).toolTip()=="EBS-BPM-A"
    assert window._save_bpm_review() is True
    assert not window.bpm_save_review_button.isEnabled()
    reopened=MeasureMainWindow(devices=result.devices); reopened.saved_measurement_path=path; reopened.result=result; reopened._show_result(result); reopened._select_bpm(1,source="table")
    assert reopened.bpm_review_status.currentText()=="Suspect" and reopened.bpm_review_highlight.isChecked()
    assert reopened.bpm_review_note.text()=="Check cable during next fill"
    assert reopened.bpm_threshold_x.value()==pytest.approx(1.1)
    assert reopened.bpm_review_save_state.text()=="All review changes saved" and not reopened._bpm_review_dirty
    window.close(); reopened.close()


def test_bpm_review_save_failure_stays_unsaved(app,tmp_path,monkeypatch):
    result=_identified_bpm_result(); path=tmp_path/"review-failure.h5"
    write_bpm_noise(path,noise_x_m=result.noise_x_m,noise_y_m=result.noise_y_m,bpm_names=[device.name for device in result.devices],raw_orbits_x_m=result.orbits_x_m,raw_orbits_y_m=result.orbits_y_m)
    window=MeasureMainWindow(devices=result.devices); window.saved_measurement_path=path; window.result=result; window._show_result(result); window._select_bpm(0,source="table"); window.bpm_review_status.setCurrentText("Bad")
    errors=[]; monkeypatch.setattr("pyLOCO.measure.main_window.write_bpm_noise_review_annotations",lambda *args,**kwargs:(_ for _ in ()).throw(PermissionError("read-only file"))); monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.critical",lambda *args:errors.append(args))
    assert window._save_bpm_review() is False
    assert window._bpm_review_dirty and window.bpm_review_save_state.text()=="Unsaved review changes"
    assert errors and "read-only file" in errors[0][2]
    window.close()


def test_bpm_review_table_note_preview_short_name_and_desktop_theme_layout(app):
    base=_identified_bpm_result(); devices=(BpmDevice("srdiag/beam-position/c22-04","z/x","z/y"),)+base.devices[1:]; result=BpmNoiseResult(devices,base.orbits_x_m,base.orbits_y_m,0.0)
    window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result); window._select_bpm(0,source="plot"); long_note="Investigate pickup electronics during the next dedicated machine study"; window.bpm_review_note.setText(long_note)
    row=next(row for row in range(3) if window._bpm_review_index_for_row(row)==0)
    assert window.bpm_review_table.item(row,0).text()=="c22-04  [0]"
    assert window.bpm_review_table.item(row,0).toolTip()=="srdiag/beam-position/c22-04"
    assert window.bpm_review_table.item(row,5).text().endswith("…") and window.bpm_review_table.item(row,5).toolTip()==long_note
    assert window.bpm_review_notice.text()=="Review only — marking a BPM as Bad does not automatically exclude it from LOCO fitting. Original measurement data remains unchanged."
    assert window.bpm_review_notice.objectName()=="infoText"
    window.bpm_review_group.setChecked(True); window.tabs.setCurrentIndex(2); window.show()
    for theme,size in (("light",(1200,800)),("dark",(1600,1000))):
        window.apply_theme(theme); window.resize(*size); app.processEvents(); header=window.bpm_review_table.horizontalHeader()
        for column in (1,2,4):assert header.sectionSize(column)>=header.sectionSizeHint(column)
        assert window.bpm_review_table.columnCount()==6
    window.close()


def test_bpm_review_status_filter_combines_search_sorting_and_identity(app):
    result=_identified_bpm_result(); window=MeasureMainWindow(devices=result.devices); window.result=result; window._show_result(result)
    assert window.bpm_review_match_count.text()=="3 of 3 BPMs"
    window._select_bpm(0,source="table"); window.bpm_review_status.setCurrentText("Bad")
    window._select_bpm(1,source="table"); window.bpm_review_status.setCurrentText("Suspect"); window.bpm_review_highlight.setChecked(True); window.bpm_review_note.setText("Watch this BPM")
    window.bpm_review_status_filter.setCurrentIndex(window.bpm_review_status_filter.findData("bad_suspect"))
    assert window.bpm_review_match_count.text()=="2 of 3 BPMs"
    assert {window._bpm_review_index_for_row(row) for row in range(3) if not window.bpm_review_table.isRowHidden(row)}=={0,1}
    window.bpm_review_search.setText("EBS-BPM-A")
    assert window.bpm_review_match_count.text()=="1 of 3 BPMs"
    visible=next(row for row in range(3) if not window.bpm_review_table.isRowHidden(row)); assert window._bpm_review_index_for_row(visible)==1
    window.bpm_review_table.sortItems(1,Qt.DescendingOrder)
    visible=next(row for row in range(3) if not window.bpm_review_table.isRowHidden(row)); assert window._bpm_review_index_for_row(visible)==1
    # A live status edit re-applies the active filters without losing review state.
    window.bpm_review_status.setCurrentText("Good")
    assert window.bpm_review_match_count.text()=="0 of 3 BPMs" and not window.bpm_review_empty.isHidden()
    assert window._selected_bpm_index==1 and window._pinned_bpm_indices=={1} and window._bpm_review_notes[1]=="Watch this BPM"
    window.bpm_review_search.clear(); window.bpm_review_status_filter.setCurrentIndex(window.bpm_review_status_filter.findData("Good"))
    assert window.bpm_review_match_count.text()=="1 of 3 BPMs"
    visible=next(row for row in range(3) if not window.bpm_review_table.isRowHidden(row)); assert window._bpm_review_index_for_row(visible)==1
    window.bpm_review_status_filter.setCurrentIndex(window.bpm_review_status_filter.findData("Unreviewed"))
    assert window.bpm_review_match_count.text()=="1 of 3 BPMs"
    window.close()
