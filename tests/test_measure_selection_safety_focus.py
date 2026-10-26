from threading import Event
from time import sleep, monotonic

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.control_system.backends import AbstractInterfaceAdapter
from pyLOCO.measure.acquisition import AcquisitionCancelled, BpmDevice, CorrectorDevice
from pyLOCO.measure.automatic import AutomaticDispersionAcquirer
from pyLOCO.measure.main_window import MeasureMainWindow, _device_filter_terms, _device_matches_filter
from pyLOCO.measure.project import load_measure_project, save_measure_project


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(["measure-selection-safety-test"])


def _window():
    bpms=tuple(BpmDevice(f"BPM-{index}",f"x{index}",f"y{index}") for index in range(6))
    h=tuple(CorrectorDevice(f"H-{index}",f"h{index}.SP",f"h{index}.RBV","H") for index in range(8))
    v=tuple(CorrectorDevice(f"V-{index}",f"v{index}.SP",f"v{index}.RBV","V") for index in range(7))
    return MeasureMainWindow(devices=bpms,horizontal_correctors=h,vertical_correctors=v)


def _visible_names(table):
    return [table.item(row,1).text() for row in range(table.rowCount()) if not table.isRowHidden(row)]


def _highlight_rows(table, rows):
    table.clearSelection()
    for row in rows:
        table.selectionModel().select(table.model().index(row,0),QItemSelectionModel.Select|QItemSelectionModel.Rows)


@pytest.mark.parametrize(("text","expected"),[
    ("c04",("c04",)),("c04,",("c04",)),(" c04, c05 ",("c04","c05")),(",,",()),
])
def test_comma_separated_filter_terms_ignore_spaces_and_empty_trailing_terms(text,expected):
    assert _device_filter_terms(text)==expected


@pytest.mark.parametrize(("query","matched"),[
    ("c04",True),("C04,",True),("c05, c04",True),("missing, C05",True),
    ("SF*, SJ*",True),("SJ*, qd*",False),("",True),(" , ",True),
])
def test_device_filter_is_case_insensitive_or_match_with_wildcards(query,matched):
    assert _device_matches_filter("SF-C04","srmag/SF-C05",query) is matched


def test_bpm_clear_highlight_three_and_add_has_one_authoritative_ordered_selection(app,monkeypatch):
    window=_window(); window._clear_bpms()
    assert window.selected_devices==() and window.manual_input.text()==""
    assert sum(window.preview_table.item(row,1).checkState()==Qt.Checked for row in range(window.preview_table.rowCount()))==0
    _highlight_rows(window.preview_table,(4,1,3))
    assert window.selected_devices==() and window.manual_input.text()==""
    assert "Highlighted: 3" in window.selection_message.text() and "0 / 6 BPMs selected" in window.selection_message.text()
    window._select_highlighted("bpm")
    expected=["BPM-1","BPM-3","BPM-4"]
    assert [item.name for item in window.selected_devices]==expected
    assert window.manual_input.text()==", ".join(expected)
    assert [window.preview_table.item(row,1).text() for row in range(window.preview_table.rowCount()) if window.preview_table.item(row,1).checkState()==Qt.Checked]==expected
    assert window._selection_snapshot()["bpm"]["resolved_names"]==expected
    monkeypatch.setattr("pyLOCO.measure.main_window.QThread.start",lambda _thread:None)
    window.start_acquisition()
    assert [item.name for item in window.worker.acquirer.devices]==expected
    window.thread.quit(); window.thread.wait(); window.close()


def test_full_session_restore_uses_saved_resolved_names_not_selection_recipe(app):
    window=_window()
    state=window._selection_states
    # An "all" recipe can resolve differently after reopening offline or on a
    # different adapter.  A saved session must restore its exact ordered
    # snapshot instead of rerunning that recipe against the new inventory.
    state["bpm_noise"]["bpm"].update(method="all",resolved_names=["BPM-4","BPM-1"])
    state["orm"]["hcor"].update(method="all",resolved_names=["H-6","H-2"])
    state["orm"]["vcor"].update(method="all",resolved_names=["V-5","V-0"])
    window._active_selection_kind="bpm_noise"
    window._restore_selection("bpm_noise",use_resolved_names=True)
    assert [device.name for device in window.selected_devices]==["BPM-4","BPM-1"]
    assert [device.name for device in window.selected_hcorrectors]==["H-6","H-2"]
    assert [device.name for device in window.selected_vcorrectors]==["V-5","V-0"]
    assert window.selection_method.currentData()=="manual"
    assert window.corrector_selection_widgets["hcor"]["method"].currentData()=="manual"
    assert window.corrector_selection_widgets["vcor"]["method"].currentData()=="manual"
    window.close()


def test_bpm_manual_method_search_visibility_and_connection_test_stay_synchronized(app,tmp_path):
    window=_window(); window._clear_bpms(); window.manual_input.setText("BPM-5, 2")
    assert [item.name for item in window.selected_devices]==["BPM-5","BPM-2"]
    window.bpm_search.setText("BPM-2"); window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("selected"))
    assert _visible_names(window.preview_table)==["BPM-2"] and [item.name for item in window.selected_devices]==["BPM-5","BPM-2"]
    window.bpm_search.clear(); window.selection_method.setCurrentIndex(window.selection_method.findData("all"))
    assert [item.name for item in window.selected_devices]==[f"BPM-{index}" for index in range(6)]
    names=tmp_path/"bpms.txt"; names.write_text("BPM-4\nBPM-0\n")
    window.names_file.setText(str(names)); window.selection_method.setCurrentIndex(window.selection_method.findData("names_file"))
    assert [item.name for item in window.selected_devices]==["BPM-4","BPM-0"]
    window._test_connection(); window.selection_method.setCurrentIndex(window.selection_method.findData("manual")); window.manual_input.setText("BPM-3, BPM-1")
    assert [item.name for item in window.selected_devices]==["BPM-3","BPM-1"]
    assert window._selection_snapshot()["bpm"]["resolved_names"]==["BPM-3","BPM-1"]
    window.close()


def test_bpm_add_highlighted_to_existing_selection_uses_inventory_order(app):
    window=_window(); window._select_device_subset("bpm","first",2)
    _highlight_rows(window.preview_table,(5,3)); window._select_highlighted("bpm")
    assert [item.name for item in window.selected_devices]==["BPM-0","BPM-1","BPM-3","BPM-5"]
    assert window.manual_input.text()=="BPM-0, BPM-1, BPM-3, BPM-5"
    window.close()


def test_corrector_manual_selection_accepts_comma_separated_wildcards_in_inventory_order(app):
    bpms=(BpmDevice("BPM-0","x0","y0"),)
    horizontal=tuple(CorrectorDevice(name,name+".SP",name+".RBV","H") for name in ("SF-2","QF-1","SJ-3","SF-1"))
    window=MeasureMainWindow(devices=bpms,horizontal_correctors=horizontal,vertical_correctors=())
    data=window.corrector_selection_widgets["hcor"]
    data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText("SF*, SJ*, SF*")
    window._refresh_corrector_preview("hcor",read_diagnostics=False)
    assert [device.name for device in window.selected_hcorrectors]==["SF-2","SJ-3","SF-1"]
    data["manual"].setText("SF*, MISSING*"); window._refresh_corrector_preview("hcor",read_diagnostics=False)
    assert "unmatched pattern 'MISSING*'" in data["table"].toolTip()
    window.close()


def test_bpm_select_only_highlighted_replaces_all_selected_and_buttons_follow_highlights(app):
    window=_window()
    assert len(window.selected_devices)==6 and not window.bpm_select_only_highlighted_button.isEnabled() and not window.bpm_add_highlighted_button.isEnabled()
    _highlight_rows(window.preview_table,(5,2,4))
    assert window.bpm_select_only_highlighted_button.isEnabled() and window.bpm_add_highlighted_button.isEnabled()
    assert "Highlighted: 3" in window.selection_message.text() and "6 / 6 BPMs selected" in window.selection_message.text()
    window._select_only_highlighted("bpm")
    assert [item.name for item in window.selected_devices]==["BPM-2","BPM-4","BPM-5"]
    assert window.manual_input.text()=="BPM-2, BPM-4, BPM-5"
    assert sum(window.preview_table.item(row,1).checkState()==Qt.Checked for row in range(6))==3
    window.close()


def test_bpm_highlight_actions_work_with_search_selected_view_and_roundtrip(app,tmp_path):
    window=_window(); window._select_device_subset("bpm","first",4); window.bpm_search.setText("BPM-"); window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("selected")); _highlight_rows(window.preview_table,(1,3))
    window._select_only_highlighted("bpm")
    assert [item.name for item in window.selected_devices]==["BPM-1","BPM-3"]
    window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("all")); _highlight_rows(window.preview_table,(5,)); window._select_highlighted("bpm")
    assert [item.name for item in window.selected_devices]==["BPM-1","BPM-3","BPM-5"]
    window._store_active_selection(); path=tmp_path/"highlighted-selection.pyloco-measure.json"; save_measure_project(path,window._collect_project()); loaded=load_measure_project(path)
    assert loaded.measurement_selections[window._active_selection_kind]["bpm"]["resolved_names"]==["BPM-1","BPM-3","BPM-5"]
    window.close()


@pytest.mark.parametrize("key",["hcor","vcor"])
def test_corrector_select_only_and_add_highlighted_are_explicit_and_ordered(app,key):
    window=_window(); data=window.corrector_selection_widgets[key]; devices=data["devices"]
    assert len(window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors)==len(devices)
    assert not data["select_only_highlighted"].isEnabled() and not data["add_highlighted"].isEnabled()
    _highlight_rows(data["table"],(4,1)); assert data["select_only_highlighted"].isEnabled() and data["add_highlighted"].isEnabled()
    window._select_only_highlighted(key); selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert [item.name for item in selected]==[devices[1].name,devices[4].name]
    _highlight_rows(data["table"],(6,1)); window._select_highlighted(key); selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert [item.name for item in selected]==[devices[1].name,devices[4].name,devices[6].name]
    assert len({item.name for item in selected})==3
    window.close()


@pytest.mark.parametrize("mode",["all","selected","unselected"])
def test_bpm_clear_preserves_highlights_and_visibility_policy(app,mode):
    window=_window(); window._select_device_subset("bpm","first",3)
    rows=(0,2) if mode!="unselected" else (4,5); _highlight_rows(window.preview_table,rows)
    window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData(mode)); search=window.bpm_search.text()
    window._clear_bpms()
    assert window.selected_devices==() and window.manual_input.text()=="" and window.bpm_search.text()==search
    assert window.bpm_visibility.currentData()==("all" if mode=="selected" else mode)
    assert {window.preview_table.item(index.row(),1).text() for index in window.preview_table.selectionModel().selectedRows()}=={f"BPM-{row}" for row in rows}
    assert f"Highlighted: {len(rows)}" in window.selection_message.text() and "0 / 6 BPMs selected" in window.selection_message.text()
    window._select_highlighted("bpm")
    assert [item.name for item in window.selected_devices]==[f"BPM-{row}" for row in rows]
    window.close()


def test_bpm_clear_keeps_search_and_highlight_action_available(app):
    window=_window(); window._select_device_subset("bpm","first",3); window.bpm_search.setText("BPM-1"); _highlight_rows(window.preview_table,(1,)); window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("selected"))
    window._clear_bpms()
    assert window.bpm_search.text()=="BPM-1" and window.bpm_visibility.currentData()=="all" and _visible_names(window.preview_table)==["BPM-1"]
    assert window.preview_table.item(1,1).isSelected()
    window._select_highlighted("bpm"); assert [item.name for item in window.selected_devices]==["BPM-1"]
    window.close()


@pytest.mark.parametrize("key",["hcor","vcor"])
@pytest.mark.parametrize("mode",["all","selected","unselected"])
def test_corrector_clear_is_local_and_preserves_highlights(app,key,mode):
    window=_window(); other="vcor" if key=="hcor" else "hcor"; window._select_device_subset(key,"first",3); before_other=[item.name for item in (window.selected_vcorrectors if other=="vcor" else window.selected_hcorrectors)]
    data=window.corrector_selection_widgets[key]; rows=(0,2) if mode!="unselected" else (4,5); _highlight_rows(data["table"],rows); data["visibility"].setCurrentIndex(data["visibility"].findData(mode)); data["search"].setText("")
    window._clear_correctors(key)
    selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert selected==() and data["manual"].text()=="" and data["visibility"].currentData()==("all" if mode=="selected" else mode)
    assert [item.name for item in (window.selected_vcorrectors if other=="vcor" else window.selected_hcorrectors)]==before_other
    highlighted={data["table"].item(index.row(),1).text() for index in data["table"].selectionModel().selectedRows()}
    expected={data["devices"][row].name for row in rows}; assert highlighted==expected
    assert f"{len(rows)} highlighted" in data["count"].text() and f"0 / {len(data['devices'])} selected" in data["count"].text()
    window._select_highlighted(key)
    selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert [item.name for item in selected]==[device.name for index,device in enumerate(data["devices"]) if index in rows]
    window.close()


def test_bpm_visibility_filter_combines_with_search_without_changing_selection_or_highlight(app):
    window=_window(); window._select_device_subset("bpm","first",3); selected=[item.name for item in window.selected_devices]
    window.preview_table.selectRow(1); highlighted=window.preview_table.item(1,1).text()
    window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("selected"))
    assert _visible_names(window.preview_table)==selected
    window.bpm_search.setText("BPM-1")
    assert _visible_names(window.preview_table)==["BPM-1"]
    assert [item.name for item in window.selected_devices]==selected
    window.bpm_search.clear(); window.bpm_visibility.setCurrentIndex(window.bpm_visibility.findData("all"))
    assert window.preview_table.item(1,1).text()==highlighted and window.preview_table.item(1,1).isSelected()
    assert "3 / 6 BPMs selected" in window.selection_message.text()
    window.close()


def test_corrector_visibility_filter_counts_empty_state_and_selection_persistence(app):
    window=_window(); window._select_device_subset("hcor","first",3); data=window.corrector_selection_widgets["hcor"]
    selected=[item.name for item in window.selected_hcorrectors]; data["table"].selectRow(1)
    data["visibility"].setCurrentIndex(data["visibility"].findData("selected"))
    assert _visible_names(data["table"])==selected
    data["search"].setText("H-1")
    assert _visible_names(data["table"])==["H-1"] and "3 / 8 selected" in data["count"].text()
    assert [item.name for item in window.selected_hcorrectors]==selected
    data["search"].clear(); data["visibility"].setCurrentIndex(data["visibility"].findData("all"))
    assert data["table"].item(1,1).isSelected()
    window._clear_correctors("hcor"); data["visibility"].setCurrentIndex(data["visibility"].findData("selected"))
    assert _visible_names(data["table"])==[] and not data["empty"].isHidden()
    assert "No correctors are selected" in data["empty"].text()
    window.close()


def test_uniform_subset_uses_filter_and_preserves_inventory_order(app):
    window=_window(); data=window.corrector_selection_widgets["hcor"]
    data["search"].setText("H-1"); window._select_device_subset("hcor","uniform",1)
    assert [item.name for item in window.selected_hcorrectors]==["H-1"]
    data["search"].clear(); window._select_device_subset("hcor","uniform",3)
    assert [item.name for item in window.selected_hcorrectors]==["H-0","H-3","H-7"]
    window.close()


def test_comma_filter_visibility_replace_and_helpers_share_one_ordered_match_set(app):
    window=_window(); data=window.corrector_selection_widgets["hcor"]
    original=[device.name for device in window.selected_hcorrectors]
    data["search"].setText("H-1,")
    assert _visible_names(data["table"])==["H-1"]
    assert [device.name for device in window.selected_hcorrectors]==original
    assert "8 / 8 selected for measurement" in data["count"].text() and "1 shown by filter" in data["count"].text()
    data["search"].setText("h-1, H-3, h-1")
    assert _visible_names(data["table"])==["H-1","H-3"]
    window._select_filtered_correctors("hcor")
    assert [device.name for device in window.selected_hcorrectors]==["H-1","H-3"]
    data["search"].setText("H-[024]"); window._select_device_subset("hcor","first",2)
    assert [device.name for device in window.selected_hcorrectors]==["H-0","H-2"]
    window._select_device_subset("hcor","uniform",2)
    assert [device.name for device in window.selected_hcorrectors]==["H-0","H-4"]
    window.close()


def test_bpm_and_vertical_filters_use_commas_wildcards_without_changing_selection(app):
    window=_window(); bpm_before=[device.name for device in window.selected_devices]
    window.bpm_search.setText("BPM-1, bpm-3,")
    assert _visible_names(window.preview_table)==["BPM-1","BPM-3"]
    assert [device.name for device in window.selected_devices]==bpm_before
    assert "6 / 6 BPMs selected for measurement" in window.selection_message.text() and "Shown by filter: 2" in window.selection_message.text()
    window._select_filtered_bpms(); assert [device.name for device in window.selected_devices]==["BPM-1","BPM-3"]
    vertical=window.corrector_selection_widgets["vcor"]; vertical["search"].setText("V-1, V-3, V-5")
    window._select_device_subset("vcor","uniform",2)
    assert [device.name for device in window.selected_vcorrectors]==["V-1","V-5"]
    window.close()


def test_bpm_checkbox_updates_authoritative_selection_with_active_comma_filter(app):
    bpms=tuple(BpmDevice(name,name+".X",name+".Y") for name in ("BPM-c04","BPM-c05","BPM-c06"))
    window=MeasureMainWindow(devices=bpms,horizontal_correctors=(),vertical_correctors=())
    window.measurement_device_alignment={"validated":True}; window.device_selection_reviewed.setChecked(True)
    window.bpm_search.setText("c04, c05")
    assert _visible_names(window.preview_table)==["BPM-c04","BPM-c05"]
    assert window.preview_table.item(0,1).flags()&Qt.ItemIsUserCheckable
    window.preview_table.selectRow(1); window.preview_table.item(0,1).setCheckState(Qt.Unchecked)
    assert [device.name for device in window.selected_devices]==["BPM-c05","BPM-c06"]
    assert window.manual_input.text()=="BPM-c05, BPM-c06"
    assert window._selection_states[window._active_selection_kind]["bpm"]["resolved_names"]==["BPM-c05","BPM-c06"]
    assert "2 / 3 BPMs selected for measurement" in window.selection_message.text()
    assert window.preview_table.item(1,1).isSelected()
    assert window.measurement_device_alignment is None and not window.device_selection_reviewed.isChecked()
    window.preview_table.item(0,1).setCheckState(Qt.Checked)
    assert [device.name for device in window.selected_devices]==["BPM-c04","BPM-c05","BPM-c06"]
    window.close()


@pytest.mark.parametrize(("key","plane"),[("hcor","H"),("vcor","V")])
def test_corrector_checkbox_updates_manual_state_without_reads_under_comma_filter(app,key,plane):
    devices=tuple(CorrectorDevice(f"{plane}C-c0{index}",f"{plane}{index}.SP",f"{plane}{index}.RBV",plane) for index in (4,5,6))
    kwargs={"horizontal_correctors":devices,"vertical_correctors":()} if key=="hcor" else {"horizontal_correctors":(),"vertical_correctors":devices}
    window=MeasureMainWindow(devices=(BpmDevice("BPM","X","Y"),),**kwargs)
    data=window.corrector_selection_widgets[key]; reads=[]
    window.adapter.read_corrector_diagnostics=lambda name:reads.append(name)
    window.measurement_device_alignment={"validated":True}; window.device_selection_reviewed.setChecked(True)
    data["search"].setText("c04, c05")
    assert _visible_names(data["table"])==[f"{plane}C-c04",f"{plane}C-c05"]
    assert data["table"].item(0,1).flags()&Qt.ItemIsUserCheckable
    data["table"].selectRow(1); data["table"].item(0,1).setCheckState(Qt.Unchecked)
    selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert [device.name for device in selected]==[f"{plane}C-c05",f"{plane}C-c06"]
    assert data["manual"].text()==f"{plane}C-c05, {plane}C-c06"
    assert window._selection_states["orm"][key]["resolved_names"]==[f"{plane}C-c05",f"{plane}C-c06"]
    assert "2 / 3 selected for measurement" in data["count"].text() and "2 shown by filter" in data["count"].text()
    assert data["table"].item(1,1).isSelected() and reads==[]
    assert window.measurement_device_alignment is None and not window.device_selection_reviewed.isChecked()
    data["table"].item(0,1).setCheckState(Qt.Checked)
    selected=window.selected_hcorrectors if key=="hcor" else window.selected_vcorrectors
    assert [device.name for device in selected]==[device.name for device in devices]
    window.close()


def test_highlight_is_distinct_and_adds_without_duplicates(app):
    window=_window(); window._clear_correctors("hcor")
    table=window.corrector_selection_widgets["hcor"]["table"]
    assert table.item(2,1).checkState()==Qt.Unchecked
    table.selectRow(2); window._select_highlighted("hcor")
    assert [item.name for item in window.selected_hcorrectors]==["H-2"]
    table.selectRow(2); window._select_highlighted("hcor")
    assert [item.name for item in window.selected_hcorrectors]==["H-2"]
    window.close()


def test_measure_project_roundtrip_keeps_search_and_resolved_identities(app,tmp_path):
    window=_window(); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    window.bpm_search.setText("BPM-"); window.corrector_selection_widgets["hcor"]["search"].setText("H-")
    window._select_device_subset("hcor","uniform",3); project=window._collect_project()
    path=tmp_path/"selection.pyloco-measure.json"; save_measure_project(path,project); loaded=load_measure_project(path)
    state=loaded.measurement_selections["orm"]
    assert state["bpm"]["search"]=="BPM-"
    assert state["hcor"]["search"]=="H-"
    assert state["hcor"]["resolved_names"]==["H-0","H-3","H-7"]
    window.close()


def test_device_selection_actions_are_visible_near_top_at_normal_window_size(app):
    window=_window(); window.resize(1200,800); window.show(); window.tabs.setCurrentIndex(1); app.processEvents()
    assert window.device_selection_configuration_group.isVisible()
    assert window.save_device_selection_button.isVisible() and window.save_device_selection_button.text()=="Save selection…"
    assert window.load_device_selection_button.isVisible() and window.load_device_selection_button.text()=="Load selection…"
    top=window.device_selection_configuration_group.mapTo(window,window.device_selection_configuration_group.rect().topLeft()).y()
    assert 0<=top<window.height()
    window.close()


def test_corrector_diagnostic_terms_are_expanded_without_asserting_units(app):
    window=_window()
    assert "SP = Setpoint" in window.corrector_diagnostic_legend.text()
    assert "RBV = Readback Value" in window.corrector_diagnostic_legend.text()
    assert "distinct quantities" in window.corrector_diagnostic_legend.text()
    table=window.corrector_selection_widgets["hcor"]["table"]
    assert [table.horizontalHeaderItem(column).text() for column in range(4,8)]==[
        "Kick setpoint","Kick readback","Current setpoint","Current readback",
    ]
    assert "KICK.SP" in table.horizontalHeaderItem(4).toolTip()
    assert "KICK.RBV" in table.horizontalHeaderItem(5).toolTip()
    assert "CURRENT.SP" in table.horizontalHeaderItem(6).toolTip()
    assert "CURRENT.RBV" in table.horizontalHeaderItem(7).toolTip()
    assert all("no unit is asserted" in table.horizontalHeaderItem(column).toolTip() for column in range(4,8))
    window.close()


def test_corrector_diagnostic_cells_distinguish_available_unread_unsupported_and_failed(app):
    window=_window(); table=window.corrector_selection_widgets["hcor"]["table"]
    # The disconnected table has not queried component state.
    window.connection_verified=False; window._refresh_corrector_preview("hcor")
    assert table.item(0,4).text()=="Not read yet"
    diagnostics={
        "KICK.SP":{"status":"available","value":1.25e-4,"unit":"rad","error":""},
        "KICK.RBV":{"status":"failed","value":None,"unit":None,"error":"readback timeout"},
        "CURRENT.SP":{"status":"unsupported","value":None,"unit":None,"error":"no hardware accessor"},
        "CURRENT.RBV":{"status":"available","value":2.5,"unit":"A","error":""},
    }
    window.adapter.read_corrector_diagnostics=lambda _name:diagnostics
    window.connection_verified=True; window._refresh_corrector_preview("hcor")
    assert [table.item(0,column).text() for column in range(4,8)]==["0.000125 rad","Read failed","Not supported","2.5 A"]
    assert "readback timeout" in table.item(0,5).toolTip()
    assert "no hardware accessor" in table.item(0,6).toolTip()
    window.close()


def test_loading_selection_replaces_all_three_roles_after_confirmation(app,tmp_path,monkeypatch):
    source=_window(); source.measurement_type.setCurrentIndex(source.measurement_type.findData("orm"))
    source.bpm_search.setText("BPM-"); source._select_device_subset("bpm","uniform",3)
    source.corrector_selection_widgets["hcor"]["search"].setText("H-"); source._select_device_subset("hcor","uniform",3)
    source.corrector_selection_widgets["vcor"]["search"].setText("V-"); source._select_device_subset("vcor","uniform",2)
    source._store_active_selection(); project=source._collect_project(); path=tmp_path/"selection.pyloco-measure.json"; save_measure_project(path,project)
    target=_window(); target.measurement_type.setCurrentIndex(target.measurement_type.findData("orm")); target._clear_bpms(); target._clear_correctors("hcor"); target._clear_correctors("vcor")
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args,**kwargs:(str(path),""))
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.question",lambda *args,**kwargs:QMessageBox.Yes)
    target._load_selection_configuration()
    assert [item.name for item in target.selected_devices]==[item.name for item in source.selected_devices]
    assert [item.name for item in target.selected_hcorrectors]==[item.name for item in source.selected_hcorrectors]
    assert [item.name for item in target.selected_vcorrectors]==[item.name for item in source.selected_vcorrectors]
    assert target.bpm_search.text()=="BPM-" and target.corrector_selection_widgets["hcor"]["search"].text()=="H-"
    source.close(); target.close()


def test_loading_selection_from_bpm_noise_restores_saved_orm_correctors_and_clear_message(app,tmp_path,monkeypatch):
    source=_window()
    assert source._active_selection_kind=="bpm_noise"
    source._select_device_subset("bpm","uniform",3)
    source._select_device_subset("hcor","first",2)
    source._select_device_subset("vcor","uniform",3)
    path=tmp_path/"all-device-selection.pyloco-measure.json"
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getSaveFileName",lambda *args,**kwargs:(str(path),""))
    source._save_selection_configuration()
    saved=load_measure_project(path)
    assert saved.measurement_selections["orm"]["hcor"]["resolved_names"]==[item.name for item in source.selected_hcorrectors]
    assert saved.measurement_selections["orm"]["vcor"]["resolved_names"]==[item.name for item in source.selected_vcorrectors]

    target=_window(); target._clear_bpms(); target._clear_correctors("hcor"); target._clear_correctors("vcor")
    prompt=[]
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args,**kwargs:(str(path),""))
    def accept(_parent,title,message,*_args):
        prompt.extend((title,message)); return QMessageBox.Yes
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.question",accept)
    target._load_selection_configuration()

    assert [item.name for item in target.selected_devices]==[item.name for item in source.selected_devices]
    assert [item.name for item in target.selected_hcorrectors]==[item.name for item in source.selected_hcorrectors]
    assert [item.name for item in target.selected_vcorrectors]==[item.name for item in source.selected_vcorrectors]
    assert "3 BPM, 2 H corrector, 3 V corrector" in prompt[1]
    assert "bpm noise" not in prompt[1].lower()
    source.close(); target.close()


def _counterpart_document(window, *, missing=(), ambiguous=()):
    rows=[]
    for ordinal,device in enumerate(window.selected_hcorrectors):
        rows.append({"measured_name":device.name,"role":"horizontal_corrector","component":"B1L","lattice_ordinal":ordinal,"status":"mapped","reason":"exact"})
    for ordinal,device in enumerate(window.vertical_correctors):
        if ordinal in missing:continue
        rows.append({"measured_name":device.name,"role":"vertical_corrector","component":"A1L","lattice_ordinal":ordinal,"status":"mapped","reason":"exact"})
        if ordinal in ambiguous:
            rows.append({"measured_name":device.name+"-duplicate","role":"vertical_corrector","component":"A1L","lattice_ordinal":ordinal,"status":"duplicate","reason":"duplicate"})
    return {"mappings":rows}


def test_h_to_v_counterpart_preview_full_partial_and_ambiguous(app):
    window=_window(); window._select_device_subset("hcor","first",4)
    full=window._h_to_v_proposal_from_document(_counterpart_document(window))
    assert [device.name for device in full["devices"]]==["V-0","V-1","V-2","V-3"]
    partial=window._h_to_v_proposal_from_document(_counterpart_document(window,missing={1},ambiguous={2}))
    assert [device.name for device in partial["devices"]]==["V-0","V-3"]
    assert [row["status"] for row in partial["rows"]]==["Matched","Missing","Ambiguous","Matched"]
    window.close()


def _esrf_pair_window(vertical_names):
    bpms=(BpmDevice("BPM","x","y"),)
    horizontal=tuple(CorrectorDevice(name,name+".SP",name+".RBV","H") for name in (
        "srmag/m-sd1/c04-a-B1L","srmag/m-sd1/c08-a-B1L","UNRECOGNIZED-H",
    ))
    vertical=tuple(CorrectorDevice(name,name+".SP",name+".RBV","V") for name in vertical_names)
    return MeasureMainWindow(devices=bpms,horizontal_correctors=horizontal,vertical_correctors=vertical)


def test_h_to_v_exact_esrf_pairs_work_without_reference_lattice_and_preserve_v_order(app,monkeypatch):
    window=_esrf_pair_window(("srmag/m-sd1/c08-a-A1L","spare-A1L","srmag/m-sd1/c04-a-A1L"))
    assert window.reference_model is None
    proposal=window._h_to_v_proposal()
    assert [row["status"] for row in proposal["rows"]]==["Matched","Matched","Unmatched"]
    assert [device.name for device in proposal["devices"]]==["srmag/m-sd1/c08-a-A1L","srmag/m-sd1/c04-a-A1L"]
    assert all("not validated AT mapping" in row["reason"] for row in proposal["rows"][:2])
    assert "Select V devices manually" in proposal["rows"][2]["reason"]
    window._clear_correctors("vcor"); monkeypatch.setattr(window,"_show_h_to_v_preview",lambda _proposal:"replace")
    window._copy_h_selection_to_v()
    assert [device.name for device in window.selected_vcorrectors]==["srmag/m-sd1/c08-a-A1L","srmag/m-sd1/c04-a-A1L"]
    window.close()


def test_h_to_v_missing_and_duplicate_exact_v_identities_fail_closed_without_lattice(app):
    duplicate="srmag/m-sd1/c04-a-A1L"
    window=_esrf_pair_window((duplicate,duplicate))
    proposal=window._h_to_v_proposal()
    assert proposal["devices"]==()
    assert proposal["rows"][0]["status"]=="Ambiguous" and "occurs 2 times" in proposal["rows"][0]["reason"]
    assert proposal["rows"][1]["status"]=="Unmatched" and "not present" in proposal["rows"][1]["reason"]
    window.close()


def test_h_to_v_uses_authoritative_backend_pairing_metadata_without_name_guessing(app):
    window=_esrf_pair_window(("VERTICAL-CONTROL",))
    window.adapter.corrector_pairings=lambda:{"UNRECOGNIZED-H":"VERTICAL-CONTROL"}
    window._select_device_subset("hcor","first",3)
    proposal=window._h_to_v_proposal()
    row=next(item for item in proposal["rows"] if item["horizontal"]=="UNRECOGNIZED-H")
    assert row["status"]=="Matched" and row["vertical"]=="VERTICAL-CONTROL"
    assert row["pairing"]=="backend_pairing" and "Authoritative backend" in row["reason"]
    window.close()


def test_h_to_v_reuses_completed_mapping_for_esrf_h_v_control_names(app):
    horizontal=(CorrectorDevice("SJ2A-C04-H","H.SP","H.RBV","H"),)
    vertical=(CorrectorDevice("SJ2A-C04-V","V.SP","V.RBV","V"),)
    window=MeasureMainWindow(
        devices=(BpmDevice("BPM","x","y"),),
        horizontal_correctors=horizontal,
        vertical_correctors=vertical,
    )
    # These real ESRF-style names are intentionally not the older terminal
    # -B1L/-A1L convention.  Their validated mappings establish the pair.
    window.measurement_device_alignment={"mappings":[
        {"measured_name":"SJ2A-C04-H","role":"horizontal_corrector","component":"B1L",
         "lattice_ordinal":23,"status":"mapped","reason":"validated"},
        {"measured_name":"SJ2A-C04-V","role":"vertical_corrector","component":"A1L",
         "lattice_ordinal":23,"status":"mapped","reason":"validated"},
    ]}
    proposal=window._h_to_v_proposal()
    assert proposal["rows"]==[{
        "horizontal":"SJ2A-C04-H","vertical":"SJ2A-C04-V","status":"Matched",
        "reason":"Complete Machine Mapping: shared AT index 23 (0-based; B1L → A1L)",
        "pairing":"complete_machine_mapping",
    }]
    assert [device.name for device in proposal["devices"]]==["SJ2A-C04-V"]
    window.close()


def test_h_to_v_uses_complete_mapping_after_vertical_selection_invalidates_alignment(app):
    horizontal=(CorrectorDevice("SJ2A-C04-H","H.SP","H.RBV","H"),)
    vertical=(CorrectorDevice("SJ2A-C04-V","V.SP","V.RBV","V"),)
    window=MeasureMainWindow(
        devices=(BpmDevice("BPM","x","y"),),
        horizontal_correctors=horizontal,vertical_correctors=vertical)
    window.measurement_device_alignment=None
    window.machine_mapping_document={
        "schema":"pyloco.machine_mapping","schema_version":3,"mappings":[
            {"pyaml_device":"SJ2A-C04-H","role":"horizontal_corrector","component":"B1L",
             "lattice_ordinal":23,"status":"mapped"},
            {"pyaml_device":"SJ2A-C04-V","role":"vertical_corrector","component":"A1L",
             "lattice_ordinal":23,"status":"mapped"},
        ]}
    window._clear_correctors("vcor")
    assert window.measurement_device_alignment is None
    proposal=window._h_to_v_proposal()
    assert proposal["rows"][0]["status"]=="Matched"
    assert proposal["rows"][0]["vertical"]=="SJ2A-C04-V"
    assert proposal["rows"][0]["pairing"]=="complete_machine_mapping"
    assert [device.name for device in proposal["devices"]]==["SJ2A-C04-V"]
    window.close()


def test_copy_h_to_v_apply_is_independent_and_persists(app,tmp_path,monkeypatch):
    window=_window(); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); window._select_device_subset("hcor","first",3); window._clear_correctors("vcor")
    proposal=window._h_to_v_proposal_from_document(_counterpart_document(window)); monkeypatch.setattr(window,"_h_to_v_proposal",lambda:proposal); monkeypatch.setattr(window,"_show_h_to_v_preview",lambda _proposal:"replace")
    window._copy_h_selection_to_v(); assert [device.name for device in window.selected_vcorrectors]==["V-0","V-1","V-2"]
    window._select_device_subset("hcor","first",1); assert [device.name for device in window.selected_vcorrectors]==["V-0","V-1","V-2"]
    window._store_active_selection(); path=tmp_path/"copied-selection.pyloco-measure.json"; save_measure_project(path,window._collect_project()); loaded=load_measure_project(path)
    assert loaded.measurement_selections["orm"]["vcor"]["resolved_names"]==["V-0","V-1","V-2"]
    assert window.copy_h_to_v_button.text()=="Copy H selection to V…"
    window.close()


def test_copy_h_to_v_add_preserves_v_inventory_order_without_duplicates(app,monkeypatch):
    window=_window(); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); window._select_device_subset("hcor","first",3)
    data=window.corrector_selection_widgets["vcor"]; data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText("V-4, V-1"); window._refresh_corrector_preview("vcor",read_diagnostics=False)
    proposal=window._h_to_v_proposal_from_document(_counterpart_document(window)); monkeypatch.setattr(window,"_h_to_v_proposal",lambda:proposal); monkeypatch.setattr(window,"_show_h_to_v_preview",lambda _proposal:"add")
    window._copy_h_selection_to_v()
    assert [device.name for device in window.selected_vcorrectors]==["V-0","V-1","V-2","V-4"]
    data["visibility"].setCurrentIndex(data["visibility"].findData("selected"))
    assert _visible_names(data["table"])==["V-0","V-1","V-2","V-4"]
    window.close()


def test_loading_selection_rejects_missing_inventory_identity(app,tmp_path,monkeypatch):
    window=_window(); project=window._collect_project()
    project.measurement_selections["bpm_noise"]["bpm"]["resolved_names"]=["BPM-MISSING"]
    path=tmp_path/"incompatible.pyloco-measure.json"; save_measure_project(path,project); warnings=[]
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args,**kwargs:(str(path),""))
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.warning",lambda *args:warnings.append(args[-1]))
    original=tuple(item.name for item in window.selected_devices); window._load_selection_configuration()
    assert tuple(item.name for item in window.selected_devices)==original
    assert warnings and "BPM-MISSING" in warnings[0]
    window.close()


class _RFInterface:
    def __init__(self, *, bad_restore=False, restore_offset_hz=None, fail_restore_readback=False):
        self.rf=500_000_000.0; self.original=self.rf; self.bad_restore=bad_restore; self.restore_offset_hz=restore_offset_hz; self.fail_restore_readback=fail_restore_readback; self.writes=[]
    def get_orbit(self):return np.zeros(2),np.zeros(2)
    def get(self,name):return 0.0
    def set(self,name,value):pass
    def get_rf_main_frequency(self):
        if self.fail_restore_readback and len(self.writes)>1 and self.writes[-1]==self.original:raise RuntimeError("RF readback unavailable")
        return self.rf
    def set_rf_main_frequency(self,value):
        self.writes.append(float(value))
        offset=self.restore_offset_hz if self.restore_offset_hz is not None else 1.0 if self.bad_restore else 0.0
        self.rf=float(value)+(offset if len(self.writes)>1 and float(value)==self.original else 0.0)


@pytest.mark.parametrize("cancel_state",[0,1,2])
def test_cancel_restores_and_reports_independent_readback(cancel_state):
    interface=_RFInterface(); adapter=AbstractInterfaceAdapter(interface,("B1","B2"),(),()); devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")); cancel=Event()
    def progress(state_index,*_):
        if state_index==cancel_state:cancel.set()
    with pytest.raises(AcquisitionCancelled) as captured:
        AutomaticDispersionAcquirer(adapter,devices).acquire(100,3,0,cancel_event=cancel,progress=progress,verify_restored_orbit=False,sleeper=lambda _:None)
    diagnostics=captured.value.restoration_diagnostics
    assert diagnostics["status"]=="restored" and diagnostics["final_readback_hz"]==diagnostics["original_rf_hz"]
    assert interface.rf==interface.original


def test_cancel_blocks_on_unverified_restore():
    interface=_RFInterface(bad_restore=True); adapter=AbstractInterfaceAdapter(interface,("B1","B2"),(),()); devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")); cancel=Event()
    def progress(*_):cancel.set()
    with pytest.raises(AcquisitionCancelled) as captured:
        AutomaticDispersionAcquirer(adapter,devices).acquire(100,3,0,cancel_event=cancel,progress=progress,verify_restored_orbit=False,sleeper=lambda _:None)
    assert captured.value.restoration_status=="verification_failed"
    assert captured.value.restoration_diagnostics["final_readback_hz"]!=interface.original


@pytest.mark.parametrize("offset,expected",[(0.0,"restored"),(.099,"restored"),(.101,"verification_failed")])
def test_rf_restoration_uses_configured_absolute_hz_tolerance(offset,expected):
    interface=_RFInterface(restore_offset_hz=offset); adapter=AbstractInterfaceAdapter(interface,("B1","B2"),(),()); devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")); cancel=Event()
    def progress(*_):cancel.set()
    with pytest.raises(AcquisitionCancelled) as captured:
        AutomaticDispersionAcquirer(adapter,devices).acquire(100,3,0,cancel_event=cancel,progress=progress,verify_restored_orbit=False,sleeper=lambda _:None,restoration_tolerance_hz=.1)
    diagnostics=captured.value.restoration_diagnostics
    assert diagnostics["status"]==expected
    assert diagnostics["absolute_difference_hz"]==pytest.approx(abs(offset))
    assert diagnostics["tolerance_hz"]==pytest.approx(.1)


def test_rf_restoration_waits_for_settling_before_readback():
    interface=_RFInterface(); adapter=AbstractInterfaceAdapter(interface,("B1","B2"),(),())
    devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")); sleeps=[]
    AutomaticDispersionAcquirer(adapter,devices).acquire(
        100,2,0,direction="bipolar",settling_delay_seconds=.25,
        verify_restored_orbit=False,sleeper=sleeps.append,
    )
    assert sleeps[-1]==pytest.approx(.25)  # restoration settles before its readback


def test_rf_restoration_missing_readback_remains_blocked():
    interface=_RFInterface(fail_restore_readback=True); adapter=AbstractInterfaceAdapter(interface,("B1","B2"),(),()); devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")); cancel=Event()
    def progress(*_):cancel.set()
    with pytest.raises(AcquisitionCancelled) as captured:
        AutomaticDispersionAcquirer(adapter,devices).acquire(100,3,0,cancel_event=cancel,progress=progress,verify_restored_orbit=False,sleeper=lambda _:None,restoration_tolerance_hz=.1)
    diagnostics=captured.value.restoration_diagnostics
    assert diagnostics["status"]=="restore_failed"
    assert diagnostics["final_readback_hz"] is None


def test_rf_restoration_tolerance_and_orm_settling_round_trip(app,tmp_path):
    window=_window(); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); window.orm_settling_delay.setValue(2.75); window.rf_restoration_tolerance.setValue(.125)
    path=tmp_path/"timing.pyloco-measure.json"; save_measure_project(path,window._collect_project()); project=load_measure_project(path)
    assert project.settling_delay_seconds==pytest.approx(2.75)
    assert project.rf_restoration_tolerance_hz==pytest.approx(.125)
    clone=_window(); clone.project=project; clone._load_project_widgets()
    assert clone.orm_settling_delay.value()==pytest.approx(2.75)
    assert clone.rf_restoration_tolerance.value()==pytest.approx(.125)
    window.close(); clone.close()


def _wait(app,predicate,timeout=.5):
    deadline=monotonic()+timeout
    while monotonic()<deadline and not predicate():app.processEvents(); sleep(.005)
    app.processEvents()


def test_connection_progress_success_and_duplicate_suppression(app):
    window=_window(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra"))
    calls=[]
    def test():calls.append(1); sleep(.03); return {"bpms":6,"corrector_readback":"ok","calibration":"ok","rf_readback":"unavailable"}
    window.adapter.test_connection=test; window._test_connection(); window._test_connection()
    assert not window.connection_progress.isHidden() and not window.test_connection_button.isEnabled()
    _wait(app,lambda:getattr(window,"connection_thread",None) is None)
    assert len(calls)==1 and window.test_connection_button.isEnabled() and "CONNECTED" in window.connection_badge.text()
    window.close()


def test_connection_timeout_rejects_late_success(app):
    window=_window(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra")); window._connection_timeout_ms=5
    window.adapter.test_connection=lambda:(sleep(.04) or {"bpms":6,"corrector_readback":"ok","calibration":"ok","rf_readback":"unavailable"})
    window._test_connection(); _wait(app,lambda:getattr(window,"connection_thread",None) is None)
    assert "TIMEOUT" in window.connection_badge.text() and "timed out" in window.machine_info["connection"].text().lower()
    window.close()


def test_connection_failure_restores_controls(app,monkeypatch):
    window=_window(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra"))
    window.adapter.test_connection=lambda:(_ for _ in ()).throw(RuntimeError("offline"))
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.warning",lambda *args:None)
    window._test_connection(); _wait(app,lambda:getattr(window,"connection_thread",None) is None)
    assert window.test_connection_button.isEnabled() and "DISCONNECTED" in window.connection_badge.text()
    assert window.machine_info["connection"].text()=="Unavailable"
    window.close()
