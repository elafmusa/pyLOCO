from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QHeaderView

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


def _display_values(item):
    status=item["status"]
    return (
        item["measured_name"],item["role"],item["lattice_name"],item["lattice_device"],
        str(item["lattice_ordinal"]),identity_match_label(item["identity_match"]),
        (item.get("component") or "BPM identity")+(" — validated" if status=="mapped" else " — not validated"),
        status.capitalize(),item["reason"],
    )


def _assert_complete_rows(table,expected):
    assert table.rowCount()==len(expected)
    for visual_row in range(table.rowCount()):
        cells=[table.item(visual_row,column) for column in range(table.columnCount())]
        assert all(cell is not None and cell.text() for cell in cells)
        identities={cell.data(Qt.UserRole) for cell in cells}
        assert len(identities)==1
        identity=identities.pop()
        assert tuple(cell.text() for cell in cells)==expected[identity]


def test_measure_alignment_separates_at_famname_device_and_recorded_match(monkeypatch):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow(devices=default_mock_devices(2))
    window.measurement_device_alignment={"mappings":[_mapping_row()]}
    monkeypatch.setattr(QDialog,"exec",lambda _dialog:0)
    window._show_alignment_details(); table=window.alignment_details_table
    assert _headers(table)==["Measured device (pyAML)","Role","AT FamName","AT Device","AT index (0-based)","Match method","Component / steering capability","Mapping status","Evidence / problem"]
    assert table.item(0,2).text()=="SD1" and table.item(0,3).text()=="srmag/m-sd1/c04-a"
    assert table.item(0,5).text()=="Legacy suffix → exact AT Device"
    assert table.horizontalHeaderItem(3).toolTip().startswith("Device attribute")
    assert window.alignment_details_dialog.isSizeGripEnabled()
    assert table.horizontalHeader().sectionResizeMode(0)==QHeaderView.Interactive
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


def test_fit_complete_mapping_details_clear_stale_search_and_role_filters():
    app=QApplication.instance() or QApplication([]); window=MainWindow()
    rows=[
        {"role":"bpm","pyaml_device":"machine/bpm/1","lattice_fam_name":"BPM1",
         "lattice_device":"machine/bpm/1","lattice_ordinal":1,"identity_match":"exact_device",
         "component":"","physical_element_type":"Monitor","status":"proposed_unique_exact_match"},
        {"role":"normal_quadrupole","pyaml_device":"machine/q1-B2L","lattice_fam_name":"Q1",
         "lattice_device":"machine/q1","lattice_ordinal":2,"identity_match":"legacy_suffix_exact_device",
         "component":"B2L","physical_element_type":"Quadrupole","status":"proposed_unique_exact_match"},
        {"role":"sextupole","pyaml_device":"machine/s1-B3L","lattice_fam_name":"S1",
         "lattice_device":"machine/s1","lattice_ordinal":3,"identity_match":"legacy_suffix_exact_device",
         "component":"B3L","physical_element_type":"Sextupole","status":"proposed_unique_exact_match"},
    ]
    window.project.machine_mapping={"schema":"pyloco.machine_mapping","schema_version":3,"mappings":rows}
    window.mapping_table_scope.setCurrentIndex(window.mapping_table_scope.findData("machine"))
    window._refresh_measurement_mapping_table()
    window.mapping_table_role_filter.setCurrentIndex(window.mapping_table_role_filter.findData("bpm"))
    window.measurement_mapping_search.setText("no-match")
    assert window.measurement_mapping_table.rowCount()==0

    window.show_complete_machine_mapping_details()
    assert window.mapping_table_scope.currentData()=="machine"
    assert window.mapping_table_role_filter.currentData()==""
    assert window.measurement_mapping_search.text()==""
    assert window.measurement_mapping_table.rowCount()==3
    assert window.mapping_table_count.text()=="Showing 3 of 3 Complete Machine Mapping records"
    assert window.measurement_mapping_table.item(1,2).text()=="Q1"
    assert window.measurement_mapping_table.item(1,3).text()=="machine/q1"
    window.project.modified=False; window.close(); app.processEvents()


def test_correct_distinguishes_fit_at_identity_from_pyaml_target_and_copies_rows():
    app=QApplication.instance() or QApplication([]); window=CorrectMainWindow()
    item=CorrectItem(0,"normal_quadrupole","QD3J",27,"m⁻²",1.0,1.1,.1,-.1,"test",control_name="srmag/m-qd3/c04-a-B2L",metadata={"lattice_fam_name":"QD3J","lattice_device":"EBS/QD3J/04","lattice_common_name":"QFAMILY-03","lattice_device_defined":True,"element_class":"Quadrupole","mapping_status":"mapped","match_source":"identity","control_component":"B2L","control_unit":"m^-1"})
    window.review=CorrectionReview([item],"test"); window.refresh_all(); table=window.mapping_details_table
    assert _headers(table)[:9]==["Fitted magnet","AT FamName","AT Device","AT CommonName","AT index (0-based)","pyAML target / control","Magnet type","Strength component","Units"]
    assert table.item(0,0).text()=="QD3J" and table.item(0,1).text()=="QD3J"
    assert table.item(0,2).text()=="EBS/QD3J/04" and table.item(0,3).text()=="QFAMILY-03"
    assert table.item(0,5).text()=="srmag/m-qd3/c04-a-B2L" and table.item(0,7).text()=="B2L"
    table.selectRow(0); QTest.keyClick(table,Qt.Key_C,Qt.ControlModifier)
    copied=QApplication.clipboard().text(); assert "AT Device" in copied and "EBS/QD3J/04" in copied
    item.metadata["lattice_device"]=""; item.metadata["lattice_device_defined"]=False
    window.refresh_all(); assert table.item(0,2).text()=="Not defined in AT lattice"
    window.close(); app.processEvents()


def test_match_labels_are_exactly_the_recorded_methods():
    assert identity_match_label("exact_device")=="Exact AT Device"
    assert identity_match_label("exact_fam_name")=="Exact AT FamName"
    assert identity_match_label("legacy_component_fam_name")=="Legacy suffix → exact AT FamName"


def test_measure_mapping_rows_remain_atomic_through_every_sort_filter_and_reopen(monkeypatch):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow(devices=default_mock_devices(2))
    records=[
        _mapping_row(measured_name="srmag/hst-sh3/c03-e-B1L",role="horizontal_corrector",lattice_name="SH3E",lattice_device="srmag/hst-sh3/c03-e",lattice_ordinal=4429,component="B1L",reason="H record"),
        _mapping_row(measured_name="srdiag/beam-position/c01-01",role="bpm",lattice_name="BPM",lattice_device="srdiag/beam-position/c01-01",lattice_ordinal=17,identity_match="exact_device",component="",reason="BPM record"),
        _mapping_row(measured_name="srmag/vst-sh3/c03-e-A1L",role="vertical_corrector",lattice_name="SH3E",lattice_device="srmag/vst-sh3/c03-e",lattice_ordinal=4430,component="A1L",reason="V record"),
        _mapping_row(measured_name="srdiag/beam-position/c02-02-with-a-very-long-identity",role="bpm",lattice_name="BPM",lattice_device="srdiag/beam-position/c02-02",lattice_ordinal=101,identity_match="exact_device",component="",reason="Second BPM record"),
    ]
    window.measurement_device_alignment={"mappings":records}; monkeypatch.setattr(QDialog,"exec",lambda _dialog:0)
    expected={(index,item["role"],item["measured_name"],item["lattice_ordinal"]):_display_values(item) for index,item in enumerate(records)}
    for _reopen in range(2):
        window._show_alignment_details(); table=window.alignment_details_table; _assert_complete_rows(table,expected)
        for column in range(table.columnCount()):
            for order in (Qt.AscendingOrder,Qt.DescendingOrder):table.sortItems(column,order); _assert_complete_rows(table,expected)
        for row in range(table.rowCount()):table.setRowHidden(row,"corrector" in table.item(row,1).text())
        table.sortItems(0,Qt.AscendingOrder); _assert_complete_rows(table,expected)
        assert {table.item(row,1).text() for row in range(table.rowCount()) if not table.isRowHidden(row)}=={"bpm"}
        table.setColumnWidth(0,420); assert table.columnWidth(0)>=400
    window.close(); app.processEvents()


def test_measure_mapping_dialog_populates_all_896_mixed_roles_without_blanks(monkeypatch):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow(devices=default_mock_devices(2)); records=[]
    for position in range(320):records.append(_mapping_row(measured_name=f"BPM-{position:03d}",role="bpm",lattice_name=f"BPM{position}",lattice_device=f"device/bpm/{position}",lattice_ordinal=position,identity_match="exact_device",component="",reason="BPM mapped"))
    for position in range(288):
        records.append(_mapping_row(measured_name=f"H-{position:03d}-B1L",role="horizontal_corrector",lattice_name=f"S{position}",lattice_device=f"device/h/{position}",lattice_ordinal=320+position,component="B1L",reason="H mapped"))
        records.append(_mapping_row(measured_name=f"V-{position:03d}-A1L",role="vertical_corrector",lattice_name=f"S{position}",lattice_device=f"device/v/{position}",lattice_ordinal=608+position,component="A1L",reason="V mapped"))
    records=records[320::2]+records[:320]+records[321::2]
    window.measurement_device_alignment={"mappings":records}; monkeypatch.setattr(QDialog,"exec",lambda _dialog:0)
    window._show_alignment_details(); table=window.alignment_details_table; assert table.rowCount()==896; table.sortItems(0,Qt.AscendingOrder)
    for row in range(table.rowCount()):
        cells=[table.item(row,column) for column in range(table.columnCount())]
        assert all(cell is not None and cell.text() for cell in cells)
        role,measured=cells[1].text(),cells[0].text()
        assert (role=="bpm" and measured.startswith("BPM-")) or (role=="horizontal_corrector" and measured.startswith("H-")) or (role=="vertical_corrector" and measured.startswith("V-"))
        assert len({cell.data(Qt.UserRole) for cell in cells})==1
    window.close(); app.processEvents()
