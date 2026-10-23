from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.correct.model import CorrectItem, CorrectionReview
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.mapping_table import CopyableMappingTable, identity_match_label
from pyLOCO.measure.main_window import MeasureMainWindow, default_mock_devices


def _headers(table):
    return [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]


def _mapping_row(**overrides):
    row = {
        "measured_name": "srmag/m-sd1/c04-a-B1L", "role": "horizontal_corrector",
        "lattice_name": "SD1", "lattice_device": "srmag/m-sd1/c04-a",
        "lattice_ordinal": 42, "identity_match": "legacy_component_device",
        "component": "B1L", "status": "mapped",
        "reason": "Identity: legacy component suffix interpreted, then exact AT Device; steering compatibility: Multipole B1",
    }
    row.update(overrides)
    return row


def test_measure_alignment_separates_at_famname_device_and_recorded_match(monkeypatch):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow(devices=default_mock_devices(2))
    window.measurement_device_alignment={"mappings":[_mapping_row()]}
    monkeypatch.setattr(QDialog,"exec",lambda _dialog:0)
    window._show_alignment_details(); table=window.alignment_details_table
    assert _headers(table)==["Measured device (pyAML)","Role","AT FamName","AT Device","AT ordinal (0-based)","Match method","Component / steering capability","Mapping status","Evidence / problem"]
    assert table.item(0,2).text()=="SD1" and table.item(0,3).text()=="srmag/m-sd1/c04-a"
    assert table.item(0,5).text()=="Legacy suffix → exact AT Device"
    assert table.horizontalHeaderItem(3).toolTip().startswith("Device attribute")
    window.close(); app.processEvents()


def test_fit_mapping_shows_identity_method_manual_evidence_and_missing_device():
    app=QApplication.instance() or QApplication([]); window=MainWindow()
    manual=_mapping_row(identity_match="exact_fam_name",lattice_device="",provenance="manual",reason="Identity: exact AT FamName; steering compatibility: KickAngle plane 0")
    window.project.measurement_device_mapping={"mappings":[manual]}
    window.measurement_mapping_filter.setCurrentIndex(window.measurement_mapping_filter.findData("all")); window._refresh_measurement_mapping_table()
    table=window.measurement_mapping_table
    assert table.item(0,2).text()=="SD1" and table.item(0,3).text()=="Not available"
    assert table.item(0,5).text()=="Exact AT FamName"
    assert "steering compatibility" in table.item(0,8).text()
    assert table.horizontalScrollBarPolicy()==Qt.ScrollBarAlwaysOn
    window.close(); app.processEvents()


def test_correct_distinguishes_fit_at_identity_from_pyaml_target_and_copies_rows():
    app=QApplication.instance() or QApplication([]); window=CorrectMainWindow()
    item=CorrectItem(0,"normal_quadrupole","QD3J",27,"m⁻²",1.0,1.1,.1,-.1,"test",control_name="srmag/m-qd3/c04-a-B2L",metadata={"lattice_device":"EBS/QD3J/04","element_class":"Quadrupole","mapping_status":"mapped","match_source":"identity","control_component":"B2L","control_unit":"m^-1"})
    window.review=CorrectionReview([item],"test"); window.refresh_all(); table=window.mapping_details_table
    assert _headers(table)[:7]==["Fitted magnet (AT FamName)","AT Device","AT ordinal (0-based)","Magnet class / type","pyAML target / control","Strength component","Units"]
    assert table.item(0,0).text()=="QD3J" and table.item(0,1).text()=="EBS/QD3J/04"
    assert table.item(0,4).text()=="srmag/m-qd3/c04-a-B2L" and table.item(0,5).text()=="B2L"
    table.selectRow(0); QTest.keyClick(table,Qt.Key_C,Qt.ControlModifier)
    copied=QApplication.clipboard().text(); assert "AT Device" in copied and "EBS/QD3J/04" in copied
    item.metadata.pop("lattice_device"); window.refresh_all(); assert table.item(0,1).text()=="Not available"
    window.close(); app.processEvents()


def test_match_labels_are_exactly_the_recorded_methods():
    assert identity_match_label("exact_device")=="Exact AT Device"
    assert identity_match_label("exact_fam_name")=="Exact AT FamName"
    assert identity_match_label("legacy_component_fam_name")=="Legacy suffix → exact AT FamName"
