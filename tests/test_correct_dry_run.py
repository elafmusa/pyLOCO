from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtCore import QPoint,Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QGroupBox,QLabel,QMessageBox,QPushButton,QScrollArea

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
    window.sort_by.setCurrentIndex(window.sort_by.findData("relative")); app.processEvents(); assert window.table.item(0,3).text()=="QD-MOCK-02"
    window.tabs.setCurrentIndex(3); app.processEvents(); assert window.review_metrics["Magnets loaded"].text()=="6"; assert window.review_metrics["Magnets included"].text()=="5"; assert window.review_metrics["Global correction fraction"].text()=="10%"; window.close()


@pytest.mark.parametrize("theme",["light","dark"])
@pytest.mark.parametrize("size",[(1000,700),(1200,800),(1500,900)])
def test_correct_pages_are_responsive_in_both_themes(theme,size):
    app=QApplication.instance() or build_application(["correct-responsive"]); window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); window.resize(*size); window.apply_theme(theme); window.show()
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index); app.processEvents(); assert window.tabs.currentWidget().width()>700; assert window.tabs.currentWidget().height()>400
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
    assert window.mapping_note.parent() is mapping and window.mapping_note.isVisible()
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
    window.close()


def test_correction_plan_reads_current_physical_k_from_verified_pysc_snapshot():
    app=QApplication.instance() or build_application(["correct-current-k"]); window=CorrectMainWindow()
    state=CorrectionReview([item(control_name="Q0K2_7_1/B2",metadata={"mapping_status":"mapped"})],"fit")
    window.review=state
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pysc"))
    window.quadrupole_workspace.profile.setCurrentIndex(window.quadrupole_workspace.profile.findData("petra3_realistic"))
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


def test_correction_plan_background_clears_highlight_and_last_columns_are_reachable():
    app=QApplication.instance() or build_application(["correct-table-navigation"]); window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); window.tabs.setCurrentIndex(2); window.show(); app.processEvents()
    window.table.selectRow(0); assert window.table.selectionModel().hasSelection()
    QTest.mouseClick(window.plan_page,Qt.LeftButton,pos=QPoint(2,2)); app.processEvents()
    assert not window.table.selectionModel().hasSelection()
    bar=window.table.horizontalScrollBar(); assert bar.maximum()>bar.minimum()
    window.show_last_columns_button.click(); assert bar.value()==bar.maximum()
    assert window.table.columnWidth(len(window.COLUMNS)-1)>=200
    window.close()
