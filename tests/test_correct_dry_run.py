from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtCore import QPoint,Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QGroupBox,QLabel,QMessageBox,QPushButton,QScrollArea,QToolBar

from pyLOCO.correct.app import build_application
from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.correct.model import (CorrectItem,CorrectionReview,apply_mock_diagnostics,load_review,save_review,save_review_csv)
from pyLOCO.data_schema import load_correction_plan


def item(index=0,**values):
    defaults=dict(index=index,correction_type="normal_quadrupole",name=f"Q{index}",lattice_ordinal=index+10,unit="m^-2",initial_value=1.0,fitted_value=.9,raw_fitted_delta=-.1,recommended_machine_delta=.1,sign_convention="recommended = initial - fitted")
    defaults.update(values); return CorrectItem(**defaults)


def test_raw_correction_is_immutable_and_scales_are_explicit():
    review=CorrectionReview([item()],"results/run",global_scale=.1); raw=review.items[0].raw_fitted_delta
    assert review.items[0].final_delta==pytest.approx(.01); review.items[0].individual_scale=.5; assert review.items[0].final_delta==pytest.approx(.005)
    review.set_global_scale(.25); assert review.items[0].final_delta==pytest.approx(.0125); assert review.items[0].raw_fitted_delta==raw
    review.items[0].included=False; assert review.items[0].final_delta==0


def test_application_scope_is_independent_of_selection_and_persists(tmp_path):
    normal=item(0,correction_type="normal_quadrupole")
    skew=item(1,correction_type="skew_quadrupole")
    review=CorrectionReview([normal,skew],str(tmp_path),application_scope="skew_quadrupole")
    assert normal.included and skew.included
    assert not review.applies_to(normal) and review.applies_to(skew)
    path=tmp_path/"scope.json"; save_review(path,review)
    restored=load_review(path)
    assert restored.application_scope=="skew_quadrupole"
    assert [entry.included for entry in restored.items]==[True,True]


def test_all_nonvalidated_mapping_states_are_warnings():
    for status in ("saved","proposed","unmapped","ambiguous","duplicate","invalid","unsupported"):
        entry=item(metadata={"mapping_status":status})
        assert f"mapping_{status}" in entry.warnings(CorrectionReview([entry],"fit").thresholds)


def test_relative_warnings_fraction_comparison_and_mock_limits():
    review=CorrectionReview([item(recommended_machine_delta=.2),item(1,initial_value=-.5,recommended_machine_delta=1.0)],"mock",global_scale=1)
    apply_mock_diagnostics(review); review.items[1].current_ampere=99; review.items[1].ampere_per_unit=10; review.items[1].min_current_ampere=-100; review.items[1].max_current_ampere=100
    assert review.items[0].relative_percent==pytest.approx(20); assert "serious_relative_correction" in review.items[0].warnings(review.thresholds)
    comparison=review.comparison(); assert [row["fraction"] for row in comparison]==[.1,.25,.5,1]; assert comparison[-1]["current_limit_violations"]==1


def test_legacy_json_and_plan_json_yaml_csv_roundtrip_are_portable(tmp_path):
    source=tmp_path/"clone"/"results"; source.mkdir(parents=True); legacy=tmp_path/"legacy.json"
    legacy.write_text(json.dumps({"global_scale":.25,"corrections":[{"name":"Q1","ordinal":4,"initial_k":1.0,"fitted_k":.9,"raw_fitted_delta":-.1,"delta_k_apply":.1,"unit":"m^-2","control_name":"MOCK-Q1","individual_scale":.5,"excluded":False}]}))
    review=load_review(legacy); review.source_result=str(source); apply_mock_diagnostics(review)
    json_path=tmp_path/"plans"/"plan.json"; json_path.parent.mkdir(); save_review(json_path,review); plan=load_correction_plan(json_path)
    assert not Path(plan.source_result).is_absolute(); assert str(tmp_path) not in json_path.read_text(); restored=load_review(json_path); assert restored.items[0].raw_fitted_delta==-.1; assert restored.items[0].ampere_per_unit is not None
    yaml_path=tmp_path/"plans"/"plan.yaml"; save_review(yaml_path,review); yaml_review=load_review(yaml_path); assert yaml_review.items[0].individual_scale==.5
    csv_path=tmp_path/"plan.csv"; save_review_csv(csv_path,review); assert "Raw fitted" not in csv_path.read_text() and "raw_fitted_delta" in csv_path.read_text()


def test_legacy_loader_refuses_to_guess_correction_sign(tmp_path):
    path=tmp_path/"ambiguous.json"; path.write_text(json.dumps({"corrections":[{"name":"Q","delta_k":.1}]}))
    with pytest.raises(ValueError,match="explicit initial and fitted"):load_review(path)


def test_current_results_loader_uses_persisted_order_and_explicit_sign(monkeypatch,tmp_path):
    class FakeLoader:
        def __init__(self,path,iteration=None):
            self.result_dir=Path(path)
            self.summary={"completed_utc":"2026-08-29T12:05:00Z"}
            self.request={"measurement_session":{"session_id":"suite-mock"}}
        quadrupole_corrections={"names":["Q2","Q1"],"ordinals":[20,10],"initial":np.array([2.,1.]),"fitted":np.array([1.8,1.1]),"delta_k_apply":np.array([.2,-.1]),"sign_convention":"initial - fitted"}
        parameter_blocks=[SimpleNamespace(key="skew_quads",values=np.array([.002]),baseline=np.array([0.]),unit="m^-2",label="Skew")]
        def parameter_identity(self,key,index):return {"element_name":"SQ1","lattice_ordinal":30}
    monkeypatch.setattr("pyLOCO.correct.model.ResultsLoader",FakeLoader); review=load_review(tmp_path)
    assert [(entry.name,entry.lattice_ordinal) for entry in review.items]==[("Q2",20),("Q1",10),("SQ1",30)]
    assert review.items[0].raw_fitted_delta==pytest.approx(-.2); assert review.items[0].recommended_machine_delta==pytest.approx(.2)


def test_current_results_loader_recovers_lengths_from_saved_lattice(monkeypatch, tmp_path):
    class FakeLoader:
        def __init__(self, path, iteration=None):
            self.result_dir = Path(path)
            self.summary = {}
            self.request = {}
        quadrupole_corrections = {
            "names": ["Q1"], "ordinals": [2], "initial": np.array([1.0]),
            "fitted": np.array([0.9]), "delta_k_apply": np.array([0.1]),
            "sign_convention": "initial - fitted",
        }
        parameter_blocks = []

    class Element:
        def __init__(self, length, fam_name, device="", common_name=""):
            self.Length, self.FamName = length, fam_name
            self.Device, self.CommonName = device, common_name

    (tmp_path / "final_lattice.mat").touch()
    monkeypatch.setattr("pyLOCO.correct.model.ResultsLoader", FakeLoader)
    monkeypatch.setitem(sys.modules, "at", SimpleNamespace(
        load_lattice=lambda _path: [Element(0.0, "D0"), Element(0.1, "D1"),
                                    Element(0.42, "Q1", "srmag/m-qf6d/c04-a", "Q1-COMMON")]))
    review = load_review(tmp_path)
    assert review.items[0].magnetic_length_m == pytest.approx(0.42)
    assert review.items[0].metadata["lattice_common_name"] == "Q1-COMMON"
    assert review.items[0].metadata["lattice_fam_name"] == "Q1"
    assert review.items[0].metadata["lattice_device"] == "srmag/m-qf6d/c04-a"
    assert review.items[0].metadata["lattice_device_defined"] is True
    assert review.items[0].integrated_recommended_delta == pytest.approx(0.042)


def test_current_results_loader_marks_genuinely_missing_at_device(monkeypatch, tmp_path):
    class FakeLoader:
        def __init__(self, path, iteration=None):
            self.result_dir, self.summary, self.request = Path(path), {}, {}
        quadrupole_corrections = {"names":["Q1"],"ordinals":[0],"initial":np.array([1.]),
            "fitted":np.array([.9]),"delta_k_apply":np.array([.1]),"sign_convention":"initial - fitted"}
        parameter_blocks = []
    element=SimpleNamespace(Length=.4,FamName="Q1",Device="",CommonName="")
    (tmp_path/"final_lattice.mat").touch()
    monkeypatch.setattr("pyLOCO.correct.model.ResultsLoader",FakeLoader)
    monkeypatch.setitem(sys.modules,"at",SimpleNamespace(load_lattice=lambda _path:[element]))
    item=load_review(tmp_path).items[0]
    assert item.metadata["lattice_device"] == ""
    assert item.metadata["lattice_device_defined"] is False


def test_selected_iteration_and_measurement_session_provenance_reach_plan(monkeypatch,tmp_path):
    class FakeLoader:
        def __init__(self,path,iteration=None):
            self.result_dir=Path(path).resolve(); self.iteration=iteration
            self.summary={"completed_utc":"2026-08-29T12:05:00Z"}
            self.request={"measurement_session":{"session_id":"suite-mock","manifest":"session.json"}}
            self.quadrupole_corrections={"names":["Q1"],"ordinals":[10],"initial":np.array([1.]),"fitted":np.array([.9]),"delta_k_apply":np.array([.1]),"sign_convention":"initial - fitted"}
            self.parameter_blocks=[]
    monkeypatch.setattr("pyLOCO.correct.model.ResultsLoader",FakeLoader)
    review=load_review(tmp_path,iteration=2); item=review.items[0]
    assert item.metadata["source_iteration"]==2
    assert item.metadata["source_state"]=="Iteration 2"
    assert item.metadata["measurement_session"]["session_id"]=="suite-mock"
    provenance=review.to_plan().metadata["source_provenance"]
    assert provenance["source_iteration"]==2
    assert provenance["fit_timestamp"]=="2026-08-29T12:05:00Z"


def test_correction_plan_reopen_preserves_complete_machine_mapping(tmp_path):
    mapping={"schema":"pyloco.machine_mapping","schema_version":3,"mappings":[{
        "role":"normal_quadrupole","lattice_ordinal":93,"lattice_fam_name":"QF6D",
        "lattice_device":"srmag/m-qf6d/c04-a","pyaml_device":"srmag/m-qf6d/c04-a-B2L"}]}
    item=CorrectItem(0,"normal_quadrupole","QF6D",93,"m⁻²",1.,.9,-.1,.1,
                     "initial-fitted",metadata={"machine_mapping":mapping})
    review=CorrectionReview([item],str(tmp_path/"results"))
    target=tmp_path/"plan.json"; save_review(target,review)
    reopened=load_review(target)
    assert reopened.machine_mapping==mapping
    assert reopened.items[0].lattice_ordinal==93


def test_gui_launch_theme_filters_table_and_mock_apply_is_disabled(tmp_path):
    app=QApplication.instance() or build_application(["correct-test"]); window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); app.processEvents()
    assert window.badge.text()=="MOCK • READ ONLY"; assert window.table.rowCount()==6; assert window.apply_button.text()=="Apply…"; assert not window.apply_button.isEnabled()
    raw=tuple(i.raw_fitted_delta for i in window.review.items); window._set_fraction(.5); assert tuple(i.raw_fitted_delta for i in window.review.items)==raw
    window.filter.setCurrentIndex(window.filter.findData("skew_quadrupole")); app.processEvents(); assert window.table.rowCount()==1
    window.apply_theme("dark"); assert window.theme_key=="dark"; window.apply_theme("light"); window.resize(1000,700); window.show(); app.processEvents(); assert window.size().width()>=1000; window.close()


def test_pysc_corrector_preview_labels_contextual_values_in_radians():
    app=QApplication.instance() or build_application(["correct-unit-test"]); window=CorrectMainWindow()
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pysc"))
    kick=SimpleNamespace(name="6/B1L",current=1e-6,proposed=1.2e-6,readback=1.2e-6,status="success")
    window._populate_machine_changes((kick,)); app.processEvents()
    headers=[window.machine_changes_table.horizontalHeaderItem(i).text() for i in range(6)]
    assert headers==["Control","Current value [rad]","Requested change [rad]","Proposed value [rad]","Readback [rad]","Status"]
    window._populate_machine_changes((SimpleNamespace(name="Q1/B2",current=1.,proposed=1.1,readback=None,status="pending"),))
    assert window.machine_changes_table.horizontalHeaderItem(1).text()=="Current value"
    window.close()


def test_source_summary_conventions_workflow_sort_and_operator_metrics():
    app=QApplication.instance() or build_application(["correct-polish"]); window=CorrectMainWindow(); window.show(); app.processEvents()
    assert window.source_path.text()=="No source loaded"; assert window.source_type.text()=="—"; assert window.source_parameters.text()=="—"; assert window.source_status.text()=="Waiting for correction data"; assert not window.conventions_body.isVisible()
    window._load(Path("Examples/Correct/mock_corrections.json").resolve()); app.processEvents(); assert window.source_type.text()=="Legacy correction JSON / YAML"; assert window.source_parameters.text().startswith("6 total")
    assert all(window.tabs.tabText(index).startswith("✓ ") for index in range(4))
    window.sort_by.setCurrentIndex(window.sort_by.findData("relative")); app.processEvents(); assert window.table.item(0,2).text()=="QD-MOCK-02"
    window.tabs.setCurrentIndex(3); app.processEvents(); assert window.review_metrics["Magnets loaded"].text()=="6"; assert window.review_metrics["Selected in plan"].text()=="5"; assert window.review_metrics["Applied by scope"].text()=="5"; assert window.review_metrics["Global correction fraction"].text()=="10%"; window.close()


@pytest.mark.parametrize("theme",["light","dark"])
@pytest.mark.parametrize("size",[(1000,700),(1200,800),(1500,900)])
def test_correct_pages_are_responsive_in_both_themes(theme,size):
    app=QApplication.instance() or build_application(["correct-responsive"]); window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); window.resize(*size); window.apply_theme(theme); window.show()
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index); app.processEvents(); assert window.tabs.currentWidget().width()>700; assert window.tabs.currentWidget().height()>240
        if index==2:assert window.table.height()>180
        if index==3:assert window.plot_tabs.height()>120
    assert not window.logo_button.pixmap().isNull(); window.close()


def test_correct_package_contains_no_backend_specific_control_system_calls():
    root=Path("pyLOCO/correct")
    text="\n".join(path.read_text() for path in root.glob("*.py"))
    assert "pydoocs" not in text and "doocs4py" not in text
    assert "self.adapter.write(" in (root/"application.py").read_text()


def test_machine_mapping_page_uses_separate_natural_height_sections_at_1000x700():
    app=QApplication.instance() or build_application(["correct-mapping-layout"]); window=CorrectMainWindow(); window.resize(1000,700); window.show(); window.tabs.setCurrentIndex(1); app.processEvents()
    mapping=window.findChild(QGroupBox,"machineMappingSection")
    assert mapping is not None and mapping.height()>=220
    assert window.findChild(QGroupBox,"mappingWarningsSection") is None
    assert window.findChild(QGroupBox,"offlineDiagnosticsSection") is None
    assert window.findChild(QGroupBox,"petraReadOnlySection") is None
    assert window.findChild(QGroupBox,"mappingHelpSection") is None
    assert not window.mapping_note.isVisible()
    assert isinstance(window.tabs.currentWidget(),QScrollArea)
    assert window.mapping_source_notice.isVisible() and not window.mapping_button.isEnabled()
    window._load(Path("Examples/Correct/mock_corrections.json").resolve()); window.tabs.setCurrentIndex(1); app.processEvents()
    assert not window.mapping_source_notice.isVisible() and window.mapping_button.isEnabled()
    window.tabs.setCurrentIndex(2); app.processEvents()
    assert window.findChild(QGroupBox,"correctionConventionSection") is not None
    connection=window.findChild(QGroupBox,"correctionPySCConnectionSection")
    assert connection is not None
    assert window.tabs.widget(1).isAncestorOf(connection)
    assert not window.tabs.widget(2).isAncestorOf(connection)
    assert "B2 simulation" not in [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert window.quadrupole_workspace.connect_button.text()=="Connect / verify"
    convention_text=" ".join(label.text() for label in window.plan_conventions_body.findChildren(QLabel))
    for term in ("Raw fitted ΔK","Recommended machine ΔK","Global fraction","Individual fraction","Final ΔK","ΔK/K [%]"):
        assert term in convention_text


def test_correction_progress_panel_advances_only_for_verified_readbacks():
    from types import SimpleNamespace
    app=QApplication.instance() or build_application(["correct-progress"])
    window=CorrectMainWindow(); window.resize(1000,700); window.show(); app.processEvents()
    panel=window.findChild(QGroupBox,"correctionProgressPanel")
    assert panel is not None and panel.isVisible() and panel.height() >= 142
    for control in (
        window.verified_percent,window.next_target_percent,
        window.progress_machine_status,window.progress_increment,
        window.progress_preview_button,window.progress_apply_button,
        window.progress_restore_button,window.progress_measured_button,
        window.correction_progress,*window.progress_segments,
    ):
        assert control.isVisible()
        assert control.mapTo(panel,control.rect().bottomLeft()).y() <= panel.rect().bottom()
    assert window.progress_increment.value()==10
    assert window.correction_progress.value()==0
    preview={"status":"preview_verified","previous_fraction":.1,"cumulative_fraction":.2,
             "fraction":.1,"history":[{"timestamp_utc":"2026-10-24T00:00:00Z",
             "cumulative_fraction":.1,"increment_fraction":.1,"execution_status":"verified"}],
             "items":[{"control":"Q1/B2","restore_physical_k":1.0,"full_physical_delta":.2,
                       "expected_physical_k":1.04}]}
    window.fullfit_transaction=SimpleNamespace(record=preview)
    window._refresh_correction_progress()
    assert window.correction_progress.value()==10
    assert window.verified_percent.text()=="Last verified: 10%"
    assert window.progress_magnets.item(0,4).text()=="Not read yet"
    applied={**preview,"status":"applied","history":preview["history"]+[
        {"timestamp_utc":"2026-10-24T00:01:00Z","cumulative_fraction":.2,
         "increment_fraction":.1,"execution_status":"verified"}]}
    applied["items"]=[{**preview["items"][0],"verified":True,
                        "applied_readback":{"physical":1.04}}]
    window.fullfit_transaction.record=applied; window._refresh_correction_progress()
    assert window.correction_progress.value()==20
    assert window.progress_magnets.item(0,6).text()=="Verified"
    assert window.progress_history.rowCount()==2
    window.close()
    window.close()


def test_correction_plan_reads_current_physical_k_from_verified_pysc_snapshot():
    app=QApplication.instance() or build_application(["correct-current-k"]); window=CorrectMainWindow()
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pysc"))
    window.quadrupole_workspace.profile.setCurrentIndex(window.quadrupole_workspace.profile.findData("petra3_realistic"))
    state=CorrectionReview([item(control_name="Q0K2_7_1/B2",metadata={"mapping_status":"mapped"})],"fit")
    window.review=state
    writes=[]
    snapshot={"identity":{"profile":"petra3_realistic","machine":"PETRA III","scenario":"realistic_errors"},"quadrupoles":[{"control":"Q0K2_7_1/B2","component":"B2","unit":"m^-2","physical":.241,"current":.24,"factor":1.01,"offset":-.0014}]}
    window.quadrupole_workspace.transaction=SimpleNamespace(connection=SimpleNamespace(snapshot=lambda:snapshot),writes=writes,record=None)
    assert window._read_current_pysc_k(strict=True)==1
    assert state.items[0].machine_value==pytest.approx(.241)
    assert state.items[0].target_value==pytest.approx(.251)
    assert state.items[0].metadata["current_control_setpoint"]==pytest.approx(.24)
    assert not writes
    window.close()


def test_selecting_machine_profile_selects_pysc_backend_without_header_roundtrip():
    app=QApplication.instance() or build_application(["correct-profile-selection"]); window=CorrectMainWindow()
    assert window.backend_combo.currentData()=="mock"
    profile=window.quadrupole_workspace.profile
    profile.setCurrentIndex(profile.findData("petra3_realistic")); app.processEvents()
    assert window.backend_combo.currentData()=="pysc"
    assert profile.currentData()=="petra3_realistic"
    window.close()


def test_review_is_compact():
    app=QApplication.instance() or build_application(["correct-repeat-safety"]); window=CorrectMainWindow()
    assert window.review_metrics_group.title()==""
    assert not hasattr(window,"comparison")
    window.close()


def test_correct_file_actions_live_in_their_workflow_sections_not_toolbar():
    app=QApplication.instance() or build_application(["correct-toolbar-labels"]); window=CorrectMainWindow(); window.resize(1200,800); window.show(); app.processEvents()
    assert [button.text() for button in window.toolbar_file_buttons] == ["Open Project…", "Save Project"]
    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert not any(
        isinstance(widget, QPushButton) and widget.text() in {
            "Load FIT results…", "Load correction plan…", "Save correction plan…"
        }
        for widget in toolbar.findChildren(QPushButton)
    )
    source_buttons = {
        button.text() for button in window.tabs.widget(0).findChildren(QPushButton)
    }
    assert "Open pyLOCO Results…" in source_buttons
    assert "Open correction plan / legacy JSON…" in source_buttons
    assert window.save_plan_button.text() == "Save correction plan…"
    assert window.save_plan_button.accessibleName() == "Save correction plan"
    window.close()


def test_correct_project_roundtrip_restores_workspace_but_not_machine_authority(tmp_path):
    app=QApplication.instance() or build_application(["correct-project-roundtrip"])
    window=CorrectMainWindow()
    normal=item(0,control_name="QF1",individual_scale=.75,
                metadata={"mapping_status":"saved","control_component":"B2L","control_unit":"m^-1"})
    skew=item(1,correction_type="skew_quadrupole",control_name="QS1",included=False,
              exclusion_reason="operator choice",metadata={"mapping_status":"saved","control_component":"A2L","control_unit":"m^-1"})
    review=CorrectionReview([normal,skew],str(tmp_path/"fit-results"),global_scale=.15,
                            comments="MDT shift note",application_scope="normal_quadrupole")
    review.machine_mapping={"schema":"pyloco.machine_mapping","schema_version":3,"mappings":[]}
    window.review=review
    window.comments.setPlainText("MDT shift note")
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pyaml"))
    window.pyaml_workspace.fields["config_path"].setText("/approved/esrf.yaml")
    window.pyaml_workspace.fields["normal_array"].setText("Quads")
    window.pyaml_workspace.fields["skew_array"].setText("SkewQuads")
    window.progress_increment.blockSignals(True); window.progress_increment.setValue(5); window.progress_increment.blockSignals(False)
    window.saved_progress_record={
        "status":"applied","fraction":.10,"increment_fraction":.10,
        "history":[{"timestamp_utc":"2026-10-25T18:00:00Z","cumulative_fraction":.10,
                    "increment_fraction":.10,"execution_status":"verified","measurements":{}}],
        "changes":[],"restore_required":True,
    }
    path=window.save_project_to(tmp_path/"shift.pyloco-correct.json")
    document=json.loads(path.read_text())
    assert document["safety"]["connection_saved"] is False
    assert document["safety"]["write_authorization_saved"] is False
    window.close()

    restored=CorrectMainWindow(); restored.load_project_path(path); app.processEvents()
    assert restored.backend_combo.currentData()=="pyaml"
    assert restored.pyaml_workspace.connection is None
    assert restored.pyaml_workspace.transaction is None
    assert not restored.pyaml_workspace.writes.isChecked()
    assert restored.pyaml_workspace.fields["config_path"].text()=="/approved/esrf.yaml"
    assert restored.review.application_scope=="normal_quadrupole"
    assert restored.review.global_scale==pytest.approx(.15)
    assert restored.review.comments=="MDT shift note"
    assert restored.review.items[0].individual_scale==pytest.approx(.75)
    assert restored.review.items[1].included is False
    assert restored.review.items[1].exclusion_reason=="operator choice"
    assert restored.review.machine_mapping["schema"]=="pyloco.machine_mapping"
    assert restored.saved_progress_record["fraction"]==pytest.approx(.10)
    assert restored.correction_progress.value()==10
    assert restored.progress_history.rowCount()==1
    assert restored.progress_increment.value()==5
    restored.close()


def test_correction_review_plots_use_list_indices_existing_strength_conversions_and_compact_counts():
    app=QApplication.instance() or build_application(["correct-review-plots"]); window=CorrectMainWindow()
    review=CorrectionReview([
        item(7,name="QD3J",lattice_ordinal=27,family="QD3",control_name="srmag/m-qd3/c04-a-B2L",recommended_machine_delta=.2,metadata={"magnetic_length_m":.5,"mapping_status":"mapped","control_component":"B2L"},machine_value=1.5),
        item(12,name="QS1",lattice_ordinal=31,family="QS",correction_type="skew_quadrupole",control_name="srmag/m-qs/c04-a-A2L",recommended_machine_delta=-.4,individual_scale=.5,metadata={"magnetic_length_m":.25,"mapping_status":"mapped","control_component":"A2L"},machine_value=-.2),
        item(20,name="QX",lattice_ordinal=44,recommended_machine_delta=.3,included=False),
    ],"smoke",global_scale=.5)
    window.review=review; window.refresh_review(); app.processEvents()
    window.refresh_table()
    assert window.correction_type_counts.text()=="Normal quadrupoles: 2 total / 1 included   •   Skew quadrupoles: 1 total / 1 included"
    review.application_scope="normal_quadrupole"; window.refresh_review()
    assert window.review_metrics["Selected in plan"].text()=="2"
    assert window.review_metrics["Applied by scope"].text()=="1"
    raw_axis=window._correction_plot_payload["raw"]["axis"]
    final_axis=window._correction_plot_payload["final"]["axis"]
    integrated_axis=window._correction_plot_payload["integrated"]["axis"]
    assert raw_axis.get_xlabel()=="Magnet index (correction list)"
    assert np.asarray(raw_axis.collections[0].get_offsets())[:,0].tolist()==[0.,1.,2.]
    assert window._correction_plot_payload["raw"]["values"].tolist()==pytest.approx([.2,-.4,.3])
    assert window._correction_plot_payload["final"]["values"].tolist()==pytest.approx([.1,-.1,0.])
    assert window._correction_plot_payload["integrated"]["values"][:2].tolist()==pytest.approx([.05,-.025])
    assert raw_axis.get_title()=="Raw recommended correction before global and individual fractions"
    assert final_axis.get_title()=="Final proposed correction after global and individual fractions"
    assert "validated magnetic length" in integrated_axis.get_title()
    strength_values=window._correction_plot_payload["strengths"]["values"]
    assert strength_values[:2].tolist()==pytest.approx([1.55,-.225])
    summary=window.correction_summary.text()
    assert "Included for review: 2 (66.7%)" in summary and "Excluded: 1 (33.3%)" in summary
    assert "Validated mappings: 2" in summary and "Unresolved mappings: 1" in summary
    assert "does not by itself mean safe" in summary
    assert [window.plot_tabs.tabText(i) for i in range(5)]==["Final proposed ΔK","Final Δ(KL)","Current / target KL","ΔK/K [%]","Raw recommended ΔK"]
    assert window.plot_tabs.currentIndex()==0
    window.close()


def test_correction_plot_hover_click_identity_and_callbacks_remain_stable_for_large_lists():
    app=QApplication.instance() or build_application(["correct-review-interaction"]); window=CorrectMainWindow()
    entries=[item(index,name=f"Q{index:04d}",lattice_ordinal=1000+index,family="QF",recommended_machine_delta=(index-401)*1e-5,metadata={"magnetic_length_m":.4}) for index in range(802)]
    window.review=CorrectionReview(entries,"large",global_scale=.1); window.refresh_review(); app.processEvents()
    callbacks=dict(window._correction_plot_callbacks)
    axis=window._correction_plot_payload["final"]["axis"]
    point_x,point_y=axis.transData.transform((123,window._correction_plot_payload["final"]["values"][123]))
    hover=SimpleNamespace(inaxes=axis,xdata=123.0,ydata=window._correction_plot_payload["final"]["values"][123],x=point_x,y=point_y,button=None)
    window._correction_plot_hover("final",hover)
    assert "Q0123" in window.correction_hover_info.text() and "correction-list index 123" in window.correction_hover_info.text()
    window._correction_plot_click("final",SimpleNamespace(inaxes=axis,xdata=123.0,button=1))
    assert window._correction_pinned_index==123 and "Q0123" in window.correction_pinned_info.text()
    assert "Correction-list index: 123" in window.correction_details_info.text()
    assert "Final proposed ΔK" in window.correction_pinned_info.text()
    window.plot_tabs.setCurrentWidget(window.plots["integrated"]); app.processEvents()
    assert "Final Δ(KL)" in window.correction_pinned_info.text()
    window._correction_plot_hover("final",SimpleNamespace(inaxes=axis,xdata=123.3,button=None))
    assert "Selected:" in window.correction_hover_info.text() and "Q0123" in window.correction_hover_info.text()
    for key,payload in window._correction_plot_payload.items():
        assert sum(artist.get_gid()=="correct-selected-magnet" for artist in tuple(payload["axis"].lines)+tuple(payload["axis"].collections))==2
    window._correction_plot_click("final",SimpleNamespace(inaxes=axis,xdata=123.0,button=1))
    assert window._correction_pinned_index is None and "Click a plotted magnet" in window.correction_pinned_info.text()
    window._correction_plot_click("final",SimpleNamespace(inaxes=axis,xdata=123.0,button=1))
    window._correction_plot_click("final",SimpleNamespace(inaxes=axis,xdata=123.5,button=1))
    assert window._correction_pinned_index is None
    window._correction_plot_click("final",SimpleNamespace(inaxes=axis,xdata=123.0,button=1))
    window.refresh_review(); app.processEvents()
    assert window._correction_plot_callbacks==callbacks
    assert len(window.plots["final"].figure.axes[0].collections)<=2
    connected_lines=len(window.plots["final"].figure.axes[0].lines)
    window.connect_correction_markers.setChecked(False); app.processEvents()
    assert len(window.plots["final"].figure.axes[0].lines)<connected_lines
    assert window._correction_plot_callbacks==callbacks
    window.close()


def test_current_and_target_strengths_have_one_validated_pair_per_magnet_and_info_is_below_plots():
    app=QApplication.instance() or build_application(["correct-strength-pairs"]); window=CorrectMainWindow()
    rows=[item(index,name=f"Q{index}",control_name=f"device-{index}",metadata={"magnetic_length_m":.5,"mapping_status":"mapped","control_component":"B2L"},machine_value=value) for index,value in enumerate((-.8,.56,-.63))]
    window.review=CorrectionReview(rows,"smoke"); window.refresh_review(); window.tabs.setCurrentIndex(3); window.show(); app.processEvents()
    payload=window._correction_plot_payload["strengths"]
    assert len(payload["hit_series"])==2
    assert [series[2].tolist() for series in payload["hit_series"]]==[[0,1,2],[0,1,2]]
    assert payload["values"].tolist()==pytest.approx([row.target_integrated_value for row in rows])
    assert window.correction_info_group.y()>window.plot_tabs.y()
    window.close()


def test_integrated_machine_strength_plot_requires_valid_mapping_and_readback():
    app=QApplication.instance() or build_application(["correct-integrated-unavailable"]); window=CorrectMainWindow()
    unmapped=item(control_name=None,metadata={"magnetic_length_m":.4,"mapping_status":"unmapped","control_component":"B2L"},machine_value=2.)
    window.review=CorrectionReview([unmapped],"fit"); window.refresh_review(); app.processEvents()
    assert "strengths" not in window._correction_plot_payload
    message=" ".join(text.get_text() for text in window.plots["strengths"].figure.axes[0].texts)
    assert "validated mappings" in message and "machine KL readbacks" in message
    window.close()


def test_correction_plan_background_clears_highlight_and_last_columns_are_reachable():
    app=QApplication.instance() or build_application(["correct-table-navigation"]); window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); window.tabs.setCurrentIndex(2); window.show(); app.processEvents()
    window.table.selectRow(0); assert window.table.selectionModel().hasSelection()
    QTest.mouseClick(window.plan_page,Qt.LeftButton,pos=QPoint(2,2)); app.processEvents()
    assert not window.table.selectionModel().hasSelection()
    bar=window.table.horizontalScrollBar(); assert bar.maximum()>bar.minimum()
    window.show_last_columns_button.click(); assert bar.value()==bar.maximum()
    assert window.table.columnWidth(len(window.COLUMNS)-1)>=200
    window.close()
