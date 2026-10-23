from __future__ import annotations

from PySide6.QtWidgets import QApplication

from pyLOCO.measure.acquisition import BpmNoiseAcquirer
from pyLOCO.measure.main_window import MeasureMainWindow, build_mock_adapter, default_mock_devices
from pyLOCO.measure.project import load_measure_project, save_measure_project


def app():
    return QApplication.instance() or QApplication([])


def labels(window):
    return [window.tabs.tabText(index).replace("&&", "&").removeprefix("✓ ")
            for index in range(window.tabs.count())]


def test_six_primary_workspaces_are_visible_without_measurement_dropdown():
    application=app(); window=MeasureMainWindow(); window.resize(900,650); window.show(); application.processEvents()
    try:
        assert labels(window)==["Machine","Devices","BPM Noise","RF Dispersion","ORM","Review & Save"]
        assert window.measurement_type.isHidden()
        for index in range(6):assert window.tabs.tabBar().tabRect(index).width()>0
    finally:window.close(); application.processEvents()


def test_measurement_tabs_preserve_independent_controls_and_selections():
    application=app(); window=MeasureMainWindow(devices=default_mock_devices(5)); window.show()
    try:
        window.tabs.setCurrentIndex(window.measurement_tab_indices["bpm_noise"]); application.processEvents()
        window.readings.setValue(7); window.measurement_name.setText("noise-seven")
        window.selection_method.setCurrentIndex(window.selection_method.findData("manual")); window.manual_input.setText("0 2")

        window.tabs.setCurrentIndex(window.measurement_tab_indices["dispersion"]); application.processEvents()
        window.readings.setValue(9); window.measurement_name.setText("dispersion-nine")
        window.selection_method.setCurrentIndex(window.selection_method.findData("manual")); window.manual_input.setText("1 3")

        window.tabs.setCurrentIndex(window.measurement_tab_indices["bpm_noise"]); application.processEvents()
        assert window.readings.value()==7 and window.measurement_name.text()=="noise-seven"
        assert window.manual_input.text()=="0 2"
        window.tabs.setCurrentIndex(window.measurement_tab_indices["dispersion"]); application.processEvents()
        assert window.readings.value()==9 and window.measurement_name.text()=="dispersion-nine"
        assert window.manual_input.text()=="1 3"
    finally:window.close(); application.processEvents()


def test_active_workspace_and_per_measurement_controls_survive_project_reopen(tmp_path):
    application=app(); first=MeasureMainWindow(); first.show()
    path=tmp_path/"navigation.pyloco-measure.json"
    try:
        first.tabs.setCurrentIndex(first.measurement_tab_indices["bpm_noise"]); application.processEvents()
        first.readings.setValue(6); first.measurement_name.setText("saved-noise")
        first.tabs.setCurrentIndex(first.measurement_tab_indices["orm"]); application.processEvents()
        first.readings.setValue(11); first.measurement_name.setText("saved-orm")
        save_measure_project(path,first._collect_project())
    finally:first.close(); application.processEvents()

    restored=MeasureMainWindow(); restored.project=load_measure_project(path); restored._load_project_widgets(); restored.show(); application.processEvents()
    try:
        assert restored.tabs.currentIndex()==restored.measurement_tab_indices["orm"]
        assert restored.readings.value()==11 and restored.measurement_name.text()=="saved-orm"
        restored.tabs.setCurrentIndex(restored.measurement_tab_indices["bpm_noise"]); application.processEvents()
        assert restored.readings.value()==6 and restored.measurement_name.text()=="saved-noise"
    finally:restored.close(); application.processEvents()


def test_other_measurement_tabs_are_disabled_during_acquisition_without_interrupting_active_tab():
    application=app(); window=MeasureMainWindow(); window.show()
    try:
        active=window.measurement_tab_indices["dispersion"]
        window.tabs.setCurrentIndex(active); application.processEvents()
        window._active_acquisition_kind="dispersion"; window._set_acquisition_running(True); application.processEvents()
        assert window.tabs.currentIndex()==active
        assert window.tabs.isTabEnabled(active)
        assert not window.tabs.isTabEnabled(window.measurement_tab_indices["bpm_noise"])
        assert not window.tabs.isTabEnabled(window.measurement_tab_indices["orm"])
        assert window.tabs.isTabEnabled(window.review_tab_index)
        window._set_acquisition_running(False); window._active_acquisition_kind=None; application.processEvents()
        assert window.tabs.isTabEnabled(window.measurement_tab_indices["bpm_noise"])
    finally:window.close(); application.processEvents()


def test_completed_result_is_retained_when_switching_measurement_tabs():
    application=app(); devices=default_mock_devices(2); adapter=build_mock_adapter(devices,readings=3)
    result=BpmNoiseAcquirer(adapter,devices).acquire(3,0,sleeper=lambda _delay:None)
    window=MeasureMainWindow(devices=devices,adapter=adapter); window.show()
    try:
        window.tabs.setCurrentIndex(window.measurement_tab_indices["bpm_noise"]); window.result=result; window._show_result(result)
        window.tabs.setCurrentIndex(window.measurement_tab_indices["dispersion"]); application.processEvents()
        assert window.result is None
        window.tabs.setCurrentIndex(window.measurement_tab_indices["bpm_noise"]); application.processEvents()
        assert window.result is result
    finally:window.close(); application.processEvents()
