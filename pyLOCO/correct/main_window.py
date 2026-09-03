"""Correction review and explicitly gated single-B2 simulation validation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal, QUrl, QEvent
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView,QApplication,QCheckBox,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QFileDialog,QFormLayout,QGridLayout,QGroupBox,QHBoxLayout,QHeaderView,QInputDialog,QLabel,QLineEdit,QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QScrollArea,QSizePolicy,QTabWidget,QTableWidget,QTableWidgetItem,QToolBar,QVBoxLayout,QWidget)

from pyLOCO.gui import __version__ as PYLOCO_VERSION
from pyLOCO.gui.branding import DISPLAY_ASSET,set_asset,application_icon
from pyLOCO.gui.project_info import (PROJECT_ACKNOWLEDGEMENTS,PROJECT_CONTRIBUTORS,PROJECT_DOCUMENTATION,PROJECT_ISSUES,PROJECT_LICENSE,PROJECT_PAPER_TITLE,PROJECT_PAPER_URL,PROJECT_REPOSITORY,bibtex_text,citation_text)
from pyLOCO.gui.suite import present_single_about_dialog
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.themes import theme_for_key
from pyLOCO.gui.appearance import ensure_suite_appearance,select_suite_appearance
from pyLOCO.control_system import AdapterCapability,InterfaceRegistry
from pyLOCO.control_system.petra import OptionalDependencyUnavailable,PETRAReadOnlyAdapter
from .application import CorrectionApplicationService
from .model import CorrectionReview,apply_mock_diagnostics,load_review,save_review,save_review_csv
from .petra_readonly import PETRACorrectReadOnlyService,apply_explicit_mapping,load_mapping,load_name_set,save_snapshot

AMBER_QSS="""
QLabel#correctBrand { color:#E88B22; font-size:18pt; font-weight:850; padding:2px 4px; }
QLabel#safetyBadge { background:#4A3211; color:#FFD28A; border:1px solid #D99029; border-radius:10px; padding:5px 8px; font-size:9pt; font-weight:800; }
QLabel#correctConnection { border-radius:9px; padding:6px 10px; font-weight:850; background:#4A2424; color:#FFAAAA; border:1px solid #D96666; }
QLabel#correctConnection[connected="true"] { background:#123D2A; color:#7BE3A7; border:1px solid #38B875; }
QLabel#workflowBanner { font-size:12pt; font-weight:850; padding:10px; background:#382A18; color:#FFD28A; border:1px solid #D99029; border-radius:7px; }
QLabel#introTitle { font-size:12pt; font-weight:750; }
QLabel#safetyText { font-size:10.5pt; }
QLabel#metricValue { font-size:14pt; font-weight:800; color:#D67A13; }
QLabel#metricLabel { font-size:9.5pt; }
QPushButton#primary { background:#C87516; color:white; border-color:#F1A943; font-weight:750; }
QTabBar::tab:selected { border-color:#D88A25; }
QGroupBox::title { color:#D47D18; }
"""

class ClickableLogo(QLabel):
    clicked=Signal()
    def __init__(self): super().__init__(); self.setCursor(Qt.PointingHandCursor); self.setToolTip("About pyLOCO Correct")
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton and self.rect().contains(event.position().toPoint()): self.clicked.emit()
        super().mouseReleaseEvent(event)

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self,event):
        if not self.hasFocus(): event.ignore(); return
        super().wheelEvent(event)

class CorrectMainWindow(QMainWindow):
    COLUMNS=("Apply?","Index","Lattice ordinal","Element/family name","Control / power-supply name","Mapping status","Type","Initial K","Fitted K","Current machine K","Raw fitted ΔK","Recommended machine ΔK","ΔK/K [%]","Global scale","Individual scale","Final ΔK","Target K","Current [A]","Target current [A]","ΔI [A]","Min current [A]","Max current [A]","Limit margin [A]","Calibration status","Current-limit status","Exclusion reason")
    def __init__(self, *, registry=None):
        super().__init__(); self.resize(1500,900); self.setMinimumSize(1000,700); self.setWindowTitle("pyLOCO Correct — Review and Apply"); self.setWindowIcon(application_icon()); self.review:CorrectionReview|None=None; self.theme_key=ensure_suite_appearance(QApplication.instance()).key; self._updating=False; self.mapping_path=None; self.sign_difference_names=frozenset(); self.large_difference_names=frozenset(); self.machine_snapshot=None; self.registry=registry or InterfaceRegistry(); self.backend_session=None; self.correction_changes=(); self.setStyleSheet(AMBER_QSS); self._build(); self._sync_theme_chrome(); QApplication.instance().installEventFilter(self)
        screen=self.screen()
        if screen is not None:self.resize(min(1500,screen.availableGeometry().width()),min(900,screen.availableGeometry().height()))

    def _build(self):
        toolbar=QToolBar("Correct toolbar"); toolbar.setObjectName("mainToolbar"); toolbar.setMovable(False); self.addToolBar(toolbar)
        brand=QLabel("pyLOCO CORRECT"); brand.setObjectName("correctBrand"); brand.setFixedWidth(240); toolbar.addWidget(brand); toolbar.addSeparator()
        for text,tip,slot in (("Results…","Open a current pyLOCO Results directory",self.open_results),("Plan…","Open a correction plan or explicit legacy correction JSON",self.open_file),("Save…","Save correction plan as JSON or YAML",self.save_plan)):
            button=QPushButton(text); button.setFixedWidth(96); button.setToolTip(tip); button.clicked.connect(slot); toolbar.addWidget(button)
        spacer=QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,spacer.sizePolicy().verticalPolicy()); toolbar.addWidget(spacer)
        self.backend_combo=QComboBox()
        for descriptor in self.registry.descriptors():self.backend_combo.addItem(descriptor.label,descriptor.key)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed); toolbar.addWidget(self.backend_combo)
        self.badge=QLabel("MOCK • READ ONLY"); self.badge.setObjectName("safetyBadge"); self.badge.setFixedWidth(230); self.badge.setAlignment(Qt.AlignCenter); toolbar.addWidget(self.badge); self.connection_badge=QLabel("● OFFLINE"); self.connection_badge.setObjectName("correctConnection"); self.connection_badge.setProperty("connected",True); toolbar.addWidget(self.connection_badge); self.theme_button=QPushButton("☾ Dark"); self.theme_button.setFixedWidth(80); self.theme_button.clicked.connect(self.toggle_theme); toolbar.addWidget(self.theme_button)
        self.logo_button=ClickableLogo(); set_asset(self.logo_button,QSize(70,29),DISPLAY_ASSET,crop_transparency=False); self.logo_button.clicked.connect(self.about); toolbar.addWidget(self.logo_button)
        root=QWidget(); layout=QVBoxLayout(root); layout.setContentsMargins(18,14,18,18)
        heading_row=QHBoxLayout(); title=QLabel("Correction Review"); title.setStyleSheet("font-size:22pt;font-weight:800"); heading_row.addWidget(title); heading_row.addStretch(); safety=QLabel("DEMO = local simulation  •  LIVE PETRA writes unavailable"); safety.setObjectName("safetyText"); heading_row.addWidget(safety); layout.addLayout(heading_row)
        self.workflow_banner=QLabel("1  LOAD   →   2  PREVIEW   →   3  CONFIRM   →   4  APPLY   →   5  READBACK"); self.workflow_banner.setObjectName("workflowBanner"); self.workflow_banner.setAlignment(Qt.AlignCenter); self.workflow_banner.setMaximumHeight(38); layout.addWidget(self.workflow_banner)
        self.profile_badge=QLabel('Machine/profile not selected'); self.profile_badge.setWordWrap(True); self.profile_badge.setMaximumWidth(360); heading_row.addWidget(self.profile_badge)
        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True); layout.addWidget(self.tabs,1); self.setCentralWidget(root)
        self.workflow_names=("Correction Source","Machine / Mapping","Correction Plan","Review && Validate")
        self.tabs.addTab(self._source_page(),self.workflow_names[0]); self.tabs.addTab(self._mapping_page(),self.workflow_names[1]); self.tabs.addTab(self._plan_page(),self.workflow_names[2]); self.tabs.addTab(self._review_page(),self.workflow_names[3])
        self.tabs.tabBar().setExpanding(False); self.tabs.tabBar().setElideMode(Qt.ElideNone); self._update_workflow_tabs()
        from .quadrupole_widget import QuadrupoleWidget
        self.quadrupole_workspace = QuadrupoleWidget(self)
        self.tabs.addTab(self._scroll_page(self.quadrupole_workspace), 'B2 simulation')
        self.backend_combo.currentIndexChanged.connect(self.quadrupole_workspace.invalidate)
        self.statusBar().showMessage("Load a correction file; control-system writes remain confirmation-gated")

    def closeEvent(self, event):
        if self.quadrupole_workspace.pending():
            QMessageBox.warning(self, 'Restore required', 'Restore and verify the original quadrupole before closing. The journal is available for recovery.')
            event.ignore()
        else:
            super().closeEvent(event)

    @staticmethod
    def _scroll_page(content):
        content.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame); scroll.setAlignment(Qt.AlignTop|Qt.AlignLeft); scroll.setWidget(content); return scroll

    def _source_page(self):
        page=QWidget(); layout=QVBoxLayout(page); layout.setAlignment(Qt.AlignTop); layout.setContentsMargins(22,22,22,22)
        group=QGroupBox("Correction source"); group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); grid=QGridLayout(group); grid.setColumnMinimumWidth(0,180); grid.setColumnStretch(1,1); grid.setVerticalSpacing(10)
        self.source_path=QLabel("No source loaded"); self.source_path.setWordWrap(True); self.source_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.source_type=QLabel("—"); self.source_parameters=QLabel("—"); self.source_iteration=QLabel("—"); self.source_timestamp=QLabel("—"); self.source_session=QLabel("—"); self.source_status=QLabel("Waiting for correction data")
        for row,(name,value) in enumerate((("Correction source",self.source_path),("Source type",self.source_type),("Parameters",self.source_parameters),("Source iteration / state",self.source_iteration),("Fit timestamp",self.source_timestamp),("Measurement Session",self.source_session),("Status",self.source_status))): grid.addWidget(QLabel(name),row,0); grid.addWidget(value,row,1)
        actions=QHBoxLayout(); a=QPushButton("Open pyLOCO Results…"); a.setObjectName("primary"); a.clicked.connect(self.open_results); b=QPushButton("Open correction plan / legacy JSON…"); b.clicked.connect(self.open_file); actions.addWidget(a); actions.addWidget(b); actions.addStretch(); grid.addLayout(actions,7,0,1,2); layout.addWidget(group)
        explain=QGroupBox("ⓘ Correction conventions"); explain.setCheckable(True); explain.setChecked(False); explain.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); el=QVBoxLayout(explain); self.conventions_body=QWidget(); body=QGridLayout(self.conventions_body); body.setColumnMinimumWidth(0,210); body.setColumnStretch(1,1)
        definitions=(("Raw fitted ΔK","Original correction reported by pyLOCO. Never modified."),("Recommended machine ΔK","Correction after applying the explicit machine/sign convention."),("Global fraction","Fraction applied to every included magnet."),("Individual fraction","Additional scale for one magnet."),("Final ΔK","Recommended machine ΔK × global fraction × individual fraction."),("ΔK/K [%]","Final correction relative to the initial K value: 100 × Final ΔK / Initial K."))
        for row,(term,description) in enumerate(definitions): label=QLabel(term); label.setStyleSheet("font-weight:700"); text=QLabel(description); text.setWordWrap(True); body.addWidget(label,row,0,Qt.AlignTop); body.addWidget(text,row,1)
        el.addWidget(self.conventions_body); self.conventions_body.setVisible(False); explain.toggled.connect(self.conventions_body.setVisible); layout.addWidget(explain); return self._scroll_page(page)

    def _mapping_page(self):
        page=QWidget(); layout=QVBoxLayout(page); layout.setAlignment(Qt.AlignTop); layout.setContentsMargins(22,22,22,22); layout.setSpacing(14)

        def status_grid(group,rows):
            grid=QGridLayout(group); grid.setContentsMargins(20,18,20,18); grid.setHorizontalSpacing(24); grid.setVerticalSpacing(11); grid.setColumnMinimumWidth(0,190); grid.setColumnStretch(1,1)
            for row,(name,value) in enumerate(rows):
                label=QLabel(name); label.setMinimumHeight(24); value.setMinimumHeight(24); value.setWordWrap(True); value.setTextInteractionFlags(Qt.TextSelectableByMouse); grid.addWidget(label,row,0,Qt.AlignTop); grid.addWidget(value,row,1)
            return grid

        mapping_group=QGroupBox("Machine Mapping"); mapping_group.setObjectName("machineMappingSection"); mapping_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        self.mapping_summary=QLabel("No correction source loaded"); self.mapping_file_status=QLabel("—"); self.mapped_status=QLabel("—"); self.unmapped_status=QLabel("—"); self.ambiguous_status=QLabel("—")
        mapping_grid=status_grid(mapping_group,(("Mapping status",self.mapping_summary),("Mapping file",self.mapping_file_status),("Mapped magnets",self.mapped_status),("Unmapped magnets",self.unmapped_status),("Ambiguous mappings",self.ambiguous_status)))
        self.mapping_source_notice=QLabel("Load a correction source first to configure PETRA magnet mapping."); self.mapping_source_notice.setWordWrap(True); self.mapping_source_notice.setObjectName("mappingSourceNotice"); mapping_grid.addWidget(self.mapping_source_notice,5,0,1,2)
        self.mapping_button=QPushButton("Load PETRA mapping…"); self.mapping_button.setMinimumWidth(210); self.mapping_button.clicked.connect(self.load_petra_mapping); self.mapping_button.setEnabled(False); mapping_grid.addWidget(self.mapping_button,6,0,1,2,Qt.AlignLeft)
        layout.addWidget(mapping_group)

        warning_group=QGroupBox("Calibration / Warning Lists"); warning_group.setObjectName("mappingWarningsSection"); warning_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        self.sign_warning_status=QLabel("Not loaded"); self.calibration_warning_status=QLabel("Not loaded")
        warning_grid=status_grid(warning_group,(("Sign warnings",self.sign_warning_status),("Calibration warnings",self.calibration_warning_status)))
        sign=QPushButton("Load sign-warning list…"); sign.setMinimumWidth(220); sign.clicked.connect(lambda:self.load_calibration_list("sign")); large=QPushButton("Load calibration-warning list…"); large.setMinimumWidth(220); large.clicked.connect(lambda:self.load_calibration_list("large")); warning_grid.addWidget(sign,0,2,Qt.AlignTop); warning_grid.addWidget(large,1,2,Qt.AlignTop)
        layout.addWidget(warning_group)

        mock_group=QGroupBox("Offline Diagnostics"); mock_group.setObjectName("offlineDiagnosticsSection"); mock_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); mock_layout=QVBoxLayout(mock_group); mock_layout.setContentsMargins(20,16,20,16); mock_layout.setSpacing(8)
        self.mock=QCheckBox("Attach deterministic Mock current/calibration diagnostics"); self.mock.setMinimumHeight(28); self.mock.setEnabled(False); self.mock.setToolTip("Clearly labelled offline demonstration values only; no machine connection or write."); self.mock.toggled.connect(self._mock_changed); mock_layout.addWidget(self.mock)
        self.mock_status=QLabel("Optional offline demonstration values only; no machine connection and no write capability."); self.mock_status.setWordWrap(True); mock_layout.addWidget(self.mock_status)
        layout.addWidget(mock_group)

        petra_group=QGroupBox("PETRA Read-Only Machine State"); petra_group.setObjectName("petraReadOnlySection"); petra_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        self.petra_connection_status=QLabel("Not connected"); self.petra_access_status=QLabel("READ ONLY — no machine setpoints will be changed."); self.snapshot_status=QLabel("Not available")
        petra_grid=status_grid(petra_group,(("Connection",self.petra_connection_status),("Access",self.petra_access_status),("Snapshot",self.snapshot_status)))
        read_row=QHBoxLayout(); self.read_petra_button=QPushButton("Read PETRA State"); self.read_petra_button.setMinimumWidth(190); self.read_petra_button.setEnabled(False); self.read_petra_button.clicked.connect(self.read_petra_state); self.save_snapshot_button=QPushButton("Save machine snapshot…"); self.save_snapshot_button.setMinimumWidth(210); self.save_snapshot_button.setEnabled(False); self.save_snapshot_button.clicked.connect(self.save_machine_snapshot); read_row.addWidget(self.read_petra_button); read_row.addWidget(self.save_snapshot_button); read_row.addStretch(); petra_grid.addLayout(read_row,3,0,1,3)
        layout.addWidget(petra_group)

        help_group=QGroupBox("ⓘ About PETRA magnet mapping"); help_group.setObjectName("mappingHelpSection"); help_group.setCheckable(True); help_group.setChecked(False); help_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); help_layout=QVBoxLayout(help_group)
        self.mapping_note=QLabel("Explicit mapping is required. Lattice element names are never assumed to be PETRA control-system names unless they have been explicitly verified. Unmapped, ambiguous, and duplicate mappings remain blocked."); self.mapping_note.setWordWrap(True); self.mapping_note.setVisible(False); help_layout.addWidget(self.mapping_note); help_group.toggled.connect(self.mapping_note.setVisible); layout.addWidget(help_group)
        layout.addStretch(1); return self._scroll_page(page)

    def _plan_page(self):
        page=QWidget(); layout=QVBoxLayout(page)
        controls=QGroupBox("Correction scaling and filters"); controls.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); controls_layout=QVBoxLayout(controls); controls_layout.setSpacing(6); first=QHBoxLayout(); second=QHBoxLayout(); self.fraction=QComboBox()
        for text,value in (("1%",.01),("5%",.05),("10%",.1),("25%",.25),("50%",.5),("100%",1.0),("Custom",None)): self.fraction.addItem(text,value)
        self.fraction.setCurrentIndex(2); self.custom=NoWheelDoubleSpinBox(); self.custom.setRange(0,1000); self.custom.setDecimals(3); self.custom.setSuffix(" %"); self.custom.setValue(10); self.custom.hide(); self.fraction.currentIndexChanged.connect(self._fraction_changed); self.custom.valueChanged.connect(lambda value:self._set_fraction(value/100))
        self.filter=QComboBox();
        for text,key in (("All","all"),("Normal quadrupoles","normal_quadrupole"),("Skew quadrupoles","skew_quadrupole"),("Quadrupole tilt","quadrupole_tilt"),("Included","included"),("Excluded","excluded"),("Warnings","warnings")): self.filter.addItem(text,key)
        self.filter.currentIndexChanged.connect(self.refresh_table); self.sort_by=QComboBox()
        for text,key in (("Fitted order","index"),("Largest |ΔK/K|","relative"),("Largest |Final ΔK|","final"),("Largest |ΔI|","delta_i"),("Smallest current-limit margin","margin"),("Calibration warnings","calibration"),("Warnings first","warnings"),("Magnet name","name")):self.sort_by.addItem(text,key)
        self.sort_by.currentIndexChanged.connect(self.refresh_table); self.search=QLineEdit(); self.search.setPlaceholderText("Search magnet/control name"); self.search.textChanged.connect(self.refresh_table)
        first.addWidget(QLabel("Global fraction")); first.addWidget(self.fraction); first.addWidget(self.custom); first.addSpacing(12); first.addWidget(QLabel("Filter")); first.addWidget(self.filter); first.addWidget(QLabel("Sort")); first.addWidget(self.sort_by); first.addWidget(self.search,1)
        include=QPushButton("Include selected"); exclude=QPushButton("Exclude selected"); reason=QLineEdit(); reason.setPlaceholderText("Exclusion reason"); include.clicked.connect(lambda:self._set_selected(True,"")); exclude.clicked.connect(lambda:self._set_selected(False,reason.text())); load=QPushButton("Load exclusion list…"); load.clicked.connect(self.load_exclusions); warn=QPushButton("Exclude warning…"); warn.setToolTip("Exclude every correction matching a selected warning category"); warn.clicked.connect(self.exclude_warning_category)
        reason.setMinimumWidth(170); second.addWidget(include); second.addWidget(exclude); second.addWidget(reason,1); second.addWidget(load); second.addWidget(warn); controls_layout.addLayout(first); controls_layout.addLayout(second); layout.addWidget(controls)
        legend=QLabel('● Normal &nbsp;&nbsp; <span style="color:#D99029">● Amber: attention</span> &nbsp;&nbsp; <span style="color:#D32F2F">● Red: blocked / unsafe</span> &nbsp;&nbsp; <span style="color:#888888">● Gray: excluded</span>'); legend.setTextFormat(Qt.RichText); legend.setObjectName("warningLegend"); legend.setToolTip("Only diagnostic cells are highlighted; excluded rows are muted."); layout.addWidget(legend)
        self.table=QTableWidget(0,len(self.COLUMNS)); self.table.setHorizontalHeaderLabels(self.COLUMNS); self.table.setSortingEnabled(True); self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setSelectionMode(QAbstractItemView.ExtendedSelection); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.table.horizontalHeader().setStretchLastSection(True); self.table.itemChanged.connect(self._item_changed); layout.addWidget(self.table,1); return page

    def _review_page(self):
        page=QWidget(); layout=QVBoxLayout(page); actions=QHBoxLayout(); self.apply_status=QLabel("Select a backend and preview before Apply"); self.apply_status.setObjectName("safetyBadge"); self.apply_status.setAlignment(Qt.AlignCenter); self.apply_status.setMinimumWidth(500); self.apply_status.setMaximumWidth(650); actions.addWidget(self.apply_status); actions.addStretch(); self.preview_apply_button=QPushButton("Preview machine changes"); self.preview_apply_button.clicked.connect(self.preview_machine_changes); actions.addWidget(self.preview_apply_button); self.apply_button=QPushButton("Apply…"); self.apply_button.setEnabled(False); self.apply_button.clicked.connect(self.apply_machine_changes); actions.addWidget(self.apply_button); export=QPushButton("Export CSV…"); export.setToolTip("Export the human-readable correction table"); export.clicked.connect(self.export_csv); actions.addWidget(export); layout.addLayout(actions)
        metrics=QGroupBox("Operator review summary"); metrics.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); grid=QGridLayout(metrics); self.review_metrics={}
        labels=("Magnets loaded","Magnets included","Magnets excluded","Warnings","Current-limit violations","Global correction fraction","Max |ΔK/K|","Max |Final ΔK|")
        for index,name in enumerate(labels):
            column=(index%4)*2; row=index//4; value=QLabel("—"); value.setObjectName("metricValue"); caption=QLabel(name); caption.setObjectName("metricLabel"); grid.addWidget(caption,row,column); grid.addWidget(value,row,column+1); self.review_metrics[name]=value
        layout.addWidget(metrics); self.comparison=QTableWidget(0,6); self.comparison.setFixedHeight(105); self.comparison.setEditTriggers(QAbstractItemView.NoEditTriggers); self.comparison.setHorizontalHeaderLabels(["Fraction","Max |ΔK/K| [%]","Max |ΔI| [A]","Limit violations","Calibration warnings","Unmapped"]); self.comparison.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); layout.addWidget(self.comparison); self.comments=QPlainTextEdit(); self.comments.setPlaceholderText("Plan comments / operator review notes"); self.comments.setFixedHeight(34); layout.addWidget(self.comments)
        changes_group=QGroupBox("Machine change preview / readback"); changes_layout=QVBoxLayout(changes_group); self.machine_changes_table=QTableWidget(0,6); self.machine_changes_table.setHorizontalHeaderLabels(["Control","Current value","Requested change","Proposed value","Readback","Status"]); self.machine_changes_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.machine_changes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.machine_changes_table.setMinimumHeight(135); changes_layout.addWidget(self.machine_changes_table); layout.addWidget(changes_group)
        self.plot_tabs=QTabWidget(); self.plots={}
        self._plot_specs=(("raw","Raw ΔK"),("final","Final ΔK"),("relative","ΔK/K [%]"),("magnet","Correction by magnet"),("overview","Included / excluded"),("current","Current diagnostics"))
        for key,title in self._plot_specs:
            placeholder=QLabel("Load correction data to display this plot."); placeholder.setAlignment(Qt.AlignCenter); self.plot_tabs.addTab(placeholder,title)
        self.plot_tabs.setMinimumHeight(180); layout.addWidget(self.plot_tabs,1); return self._scroll_page(page)

    def _ensure_plots(self):
        if self.plots:return
        index=self.plot_tabs.currentIndex()
        for position,(key,title) in enumerate(self._plot_specs):
            placeholder=self.plot_tabs.widget(position); self.plot_tabs.removeTab(position); placeholder.deleteLater()
            canvas=PlotCanvas(minimum_height=270); self.plots[key]=canvas; self.plot_tabs.insertTab(position,canvas,title)
        self.plot_tabs.setCurrentIndex(index)

    def _load(self,path,iteration=None):
        try:self.review=load_review(path,iteration=iteration)
        except Exception as exc: QMessageBox.critical(self,"Cannot load correction source",str(exc)); return
        source=Path(path).resolve(); self.machine_snapshot=None; self.save_snapshot_button.setEnabled(False); self.badge.setText(self.registry.descriptor(self.backend_combo.currentData()).badge); self.snapshot_status.setText("Not available — opening Results never contacts the machine.")
        for item in self.review.items:item.metadata.setdefault("mapping_status","mapped" if item.control_name else "unmapped")
        self.correction_changes=(); self.apply_button.setEnabled(False); self.apply_status.setText("Correction loaded — preview machine changes")
        provenance=self.review.items[0].metadata if self.review.items else {}; session=provenance.get("measurement_session") or {}; self.source_iteration.setText(str(provenance.get("source_state") or "Final / loaded plan")); self.source_timestamp.setText(str(provenance.get("fit_timestamp") or "Not available")); self.source_session.setText(str(session.get("session_id") or "Not recorded")); self.source_path.setText(str(source)); self.source_path.setToolTip(str(source)); self.source_type.setText(self._source_kind(source)); self.source_parameters.setText(self._parameter_text()); self.source_status.setText("Correction data loaded and ready for mapping review"); self.mock.setChecked(False); self.mapping_button.setEnabled(True); self.mock.setEnabled(True); self.read_petra_button.setEnabled(True); self.mapping_source_notice.setVisible(False); self.mapping_file_status.setText("—"); self.petra_connection_status.setText("Not connected"); self.petra_access_status.setText("READ ONLY — no machine setpoints will be changed."); self._refresh_mapping_status(); self._sync_fraction(); self.refresh_all(); self.tabs.setCurrentIndex(2)

    @staticmethod
    def _source_kind(path):
        if path.is_dir():return "Current pyLOCO Results directory"
        try:data=json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower()==".json" else {}
        except Exception:data={}
        return "pyLOCO correction plan" if isinstance(data,dict) and data.get("file_type")=="pyloco.correction_plan" else "Legacy correction JSON / YAML"

    def _parameter_text(self):
        if not self.review:return "—"
        counts={key:sum(item.correction_type==key for item in self.review.items) for key in ("normal_quadrupole","skew_quadrupole","quadrupole_tilt")}
        return f"{len(self.review.items)} total — {counts['normal_quadrupole']} normal, {counts['skew_quadrupole']} skew, {counts['quadrupole_tilt']} tilt"

    def open_results(self):
        path=QFileDialog.getExistingDirectory(self,"Open current pyLOCO Results directory")
        if path:self._load(path)
    def open_file(self):
        path=QFileDialog.getOpenFileName(self,"Open correction plan or legacy JSON","","Correction files (*.json *.yaml *.yml)")[0]
        if path:self._load(path)

    def _backend_changed(self,*_):
        key=self.backend_combo.currentData(); descriptor=self.registry.descriptor(key); self.badge.setText(descriptor.badge); self.backend_session=None; self.correction_changes=(); self.apply_button.setEnabled(False)
        self._set_connection(False,"OFFLINE" if key=="mock" else "DISCONNECTED"); self._style_backend_badge(key); self._populate_machine_changes(())
        self.apply_status.setText("Mock is read-only" if key=="mock" else "Backend selected — preview reads current values")

    def _set_connection(self,connected,text):
        self.connection_badge.setText(("● " if connected else "○ ")+text); self.connection_badge.setProperty("connected",bool(connected)); self.connection_badge.style().unpolish(self.connection_badge); self.connection_badge.style().polish(self.connection_badge)

    def _style_backend_badge(self,key):
        if key=="pysc":self.badge.setStyleSheet("background:#123B42;color:#78F1ED;border:2px solid #20BFC4;border-radius:10px;padding:7px;font-size:11pt;font-weight:900")
        elif key=="petra":self.badge.setStyleSheet("background:#5A1717;color:#FFF0F0;border:2px solid #EF5350;border-radius:10px;padding:7px;font-size:11pt;font-weight:900")
        else:self.badge.setStyleSheet("")

    def _populate_machine_changes(self,changes):
        if not hasattr(self,"machine_changes_table"):return
        pysc_kicks=self.backend_combo.currentData()=="pysc" and bool(changes) and all(
            str(change.name).endswith(("/B1L","/A1L")) for change in changes
        )
        unit=" [rad]" if pysc_kicks else ""
        self.machine_changes_table.setHorizontalHeaderLabels([
            "Control",f"Current value{unit}",f"Requested change{unit}",
            f"Proposed value{unit}",f"Readback{unit}","Status",
        ])
        self.machine_changes_table.setRowCount(len(changes))
        for row,change in enumerate(changes):
            values=(change.name,change.current,change.proposed-change.current,change.proposed,"—" if change.readback is None else change.readback,change.status)
            for col,value in enumerate(values):self.machine_changes_table.setItem(row,col,QTableWidgetItem(self._fmt(value)))

    def preview_machine_changes(self):
        QMessageBox.information(self, 'B2 simulation milestone', 'Use B2 simulation for single or 3–5 synthetic normal quadrupoles. Full FIT and unverified-name application remain disabled.')

    def apply_machine_changes(self):
        QMessageBox.information(self, 'Application disabled', 'Only the bounded B2 simulation transaction is enabled.')
    def _mapping_text(self):
        if not self.review:return "No correction source loaded"
        counts={key:sum(i.correction_type==key for i in self.review.items) for key in ("normal_quadrupole","skew_quadrupole","quadrupole_tilt")}; statuses={key:sum(i.metadata.get("mapping_status") == key for i in self.review.items) for key in ("mapped","unmapped","ambiguous","duplicate")}; available=sum(i.control_name is not None for i in self.review.items)
        return f"Normal: {counts['normal_quadrupole']} · Skew: {counts['skew_quadrupole']} · Tilt: {counts['quadrupole_tilt']} · Mapped: {statuses['mapped'] or available} · Unmapped: {statuses['unmapped']} · Ambiguous: {statuses['ambiguous']} · Duplicate: {statuses['duplicate']}"

    def _refresh_mapping_status(self):
        if not self.review:
            self.mapping_summary.setText("No correction source loaded"); self.mapped_status.setText("—"); self.unmapped_status.setText("—"); self.ambiguous_status.setText("—"); return
        statuses={key:sum(item.metadata.get("mapping_status")==key for item in self.review.items) for key in ("mapped","unmapped","ambiguous","duplicate")}; available=sum(item.control_name is not None for item in self.review.items); mapped=statuses["mapped"] or available
        self.mapping_summary.setText("Mapping ready for review" if mapped else "Explicit PETRA mapping required")
        self.mapped_status.setText(str(mapped)); self.unmapped_status.setText(str(statuses["unmapped"])); self.ambiguous_status.setText(f"{statuses['ambiguous']} ambiguous; {statuses['duplicate']} duplicate")

    def load_petra_mapping(self):
        path=QFileDialog.getOpenFileName(self,"Load explicit PETRA magnet mapping","","Mapping files (*.json *.yaml *.yml)")[0]
        if not path or not self.review:return
        try:counts=apply_explicit_mapping(self.review,load_mapping(path))
        except Exception as exc:QMessageBox.critical(self,"Cannot load PETRA mapping",str(exc)); return
        self.mapping_path=str(Path(path).resolve()); self.mapping_file_status.setText(Path(path).name); self.mapping_file_status.setToolTip(self.mapping_path); self._refresh_mapping_status(); self.snapshot_status.setText(f"Mapping loaded: {counts['mapped']} mapped, {counts['unmapped']} unmapped, {counts['ambiguous']} ambiguous, {counts['duplicate']} duplicate. Machine state has not been read."); self.refresh_all()

    def load_calibration_list(self,kind):
        title="Load sign-difference magnet list" if kind=="sign" else "Load large calibration-difference magnet list"; path=QFileDialog.getOpenFileName(self,title,"","Lists (*.txt *.json *.yaml *.yml);;All files (*)")[0]
        if not path:return
        try:names=load_name_set(path)
        except Exception as exc:QMessageBox.critical(self,"Cannot load warning list",str(exc)); return
        if kind=="sign":self.sign_difference_names=names; self.sign_warning_status.setText(f"{len(names)} names loaded")
        else:self.large_difference_names=names; self.calibration_warning_status.setText(f"{len(names)} names loaded")
        self.snapshot_status.setText(f"Loaded {len(names)} {'sign-convention' if kind=='sign' else 'large-calibration-difference'} warning names. No correction signs or inclusion states were changed.")

    def read_petra_state(self):
        if not self.review:QMessageBox.information(self,"No correction source","Open pyLOCO Results before reading PETRA state."); return
        unresolved=sum(item.correction_type=="normal_quadrupole" and item.metadata.get("mapping_status")!="mapped" for item in self.review.items)
        if not any(item.metadata.get("mapping_status")=="mapped" for item in self.review.items):QMessageBox.warning(self,"Explicit mapping required","Load an explicit PETRA mapping before requesting machine state."); return
        try:
            service=PETRACorrectReadOnlyService(PETRAReadOnlyAdapter(),sign_difference_names=self.sign_difference_names,large_difference_names=self.large_difference_names); self.machine_snapshot=service.read_snapshot(self.review,mapping_file=self.mapping_path)
        except OptionalDependencyUnavailable as exc:QMessageBox.warning(self,"PETRA diagnostics unavailable",str(exc)); return
        except Exception as exc:QMessageBox.critical(self,"PETRA read-only snapshot failed",str(exc)); return
        self.badge.setText("LIVE • PETRA III DOOCS"); self.petra_connection_status.setText("Connected for read-only snapshot"); self.petra_access_status.setText("READ ONLY snapshot — no machine setpoints were changed."); self.save_snapshot_button.setEnabled(True); self.snapshot_status.setText(f"Snapshot {self.machine_snapshot.timestamp_utc}; {unresolved} normal quadrupole mapping(s) unresolved. Zero writes."); self.refresh_all()

    def save_machine_snapshot(self):
        if self.machine_snapshot is None:return
        path=QFileDialog.getSaveFileName(self,"Save PETRA read-only machine snapshot","petra-readonly-snapshot.json","JSON (*.json)")[0]
        if path:
            try:save_snapshot(path,self.machine_snapshot)
            except Exception as exc:QMessageBox.critical(self,"Cannot save snapshot",str(exc)); return
            self.statusBar().showMessage(f"Saved PETRA read-only snapshot: {path}")

    def _workflow_complete(self):
        source=self.review is not None
        mapping=bool(source and self.review.items and all(item.metadata.get("mapping_status")=="mapped" for item in self.review.items))
        plan=bool(mapping and self.review.items and any(item.included for item in self.review.items) and all(item.initial_value is not None for item in self.review.items))
        review=bool(plan and not any(item.current_limit_status=="VIOLATION" for item in self.review.items if item.included))
        return source,mapping,plan,review

    def _update_workflow_tabs(self):
        complete=self._workflow_complete()
        for index,name in enumerate(self.workflow_names):self.tabs.setTabText(index,("✓ " if complete[index] else "")+name)
    def _mock_changed(self,checked):
        if checked and self.review: apply_mock_diagnostics(self.review); self.mock_status.setText("Mock diagnostics active — deterministic offline values. No machine connection and no write capability exists.")
        elif not checked:self.mock_status.setText("Optional offline demonstration values only; no machine connection and no write capability.")
        self._refresh_mapping_status(); self.refresh_all()
    def _sync_fraction(self):
        if not self.review:return
        index=self.fraction.findData(self.review.global_scale); self.fraction.setCurrentIndex(index if index>=0 else self.fraction.count()-1); self.custom.setValue(self.review.global_scale*100)
    def _fraction_changed(self):
        custom=self.fraction.currentData() is None; self.custom.setVisible(custom)
        if not custom:self._set_fraction(float(self.fraction.currentData()))
    def _set_fraction(self,value):
        if self.review:self.review.set_global_scale(float(value)); self.refresh_all()

    @staticmethod
    def _fmt(value): return "Not available" if value is None or (isinstance(value,float) and not np.isfinite(value)) else f"{value:.8g}" if isinstance(value,(int,float,np.number)) else str(value)
    def _visible(self,item):
        key=self.filter.currentData(); search=self.search.text().strip().lower(); matches=not search or search in item.name.lower() or search in (item.control_name or "").lower()
        return matches and (key=="all" or key==item.correction_type or key=="included" and item.included or key=="excluded" and not item.included or key=="warnings" and bool(item.warnings(self.review.thresholds)))
    def _sorted_items(self):
        items=list(filter(self._visible,self.review.items)); key=self.sort_by.currentData()
        if key=="relative":items.sort(key=lambda item:abs(item.relative_percent or 0),reverse=True)
        elif key=="final":items.sort(key=lambda item:abs(item.final_delta),reverse=True)
        elif key=="delta_i":items.sort(key=lambda item:abs(item.delta_i_ampere or 0),reverse=True)
        elif key=="margin":items.sort(key=lambda item:float("inf") if item.current_limit_margin_ampere is None else item.current_limit_margin_ampere)
        elif key=="calibration":items.sort(key=lambda item:item.calibration_status=="Calibration OK")
        elif key=="warnings":items.sort(key=lambda item:(bool(item.warnings(self.review.thresholds)),len(item.warnings(self.review.thresholds))),reverse=True)
        elif key=="name":items.sort(key=lambda item:item.name.lower())
        else:items.sort(key=lambda item:item.index)
        return items
    def refresh_table(self):
        if not hasattr(self,"table"):return
        self._updating=True; self.table.setSortingEnabled(False); self.table.setRowCount(0)
        if self.review:
            for item in self._sorted_items():
                row=self.table.rowCount(); self.table.insertRow(row); values=(item.included,item.index,item.lattice_ordinal,item.name,item.control_name,item.metadata.get("mapping_status","unmapped"),item.correction_type,item.initial_value,item.fitted_value,item.machine_value,item.raw_fitted_delta,item.recommended_machine_delta,item.relative_percent,self.review.global_scale,item.individual_scale,item.final_delta,item.target_value,item.current_ampere,item.target_current_ampere,item.delta_i_ampere,item.min_current_ampere,item.max_current_ampere,item.current_limit_margin_ampere,item.calibration_status,item.current_limit_status,item.exclusion_reason)
                warnings=item.warnings(self.review.thresholds)
                for col,value in enumerate(values):
                    table_item=QTableWidgetItem("" if col==0 else self._fmt(value)); table_item.setData(Qt.UserRole,item.index)
                    if col==0:table_item.setFlags(table_item.flags()|Qt.ItemIsUserCheckable); table_item.setCheckState(Qt.Checked if item.included else Qt.Unchecked)
                    if col in (14,25):table_item.setFlags(table_item.flags()|Qt.ItemIsEditable)
                    elif col!=0:table_item.setFlags(table_item.flags()&~Qt.ItemIsEditable)
                    table_item.setToolTip("Warnings: "+", ".join(warnings) if warnings else "Normal")
                    if not item.included:
                        table_item.setForeground(QColor("#888888"))
                        if col==0:table_item.setBackground(QColor("#D0D0D0"))
                    elif any(code.startswith("mapping_") for code in warnings) and col in (0,5):table_item.setBackground(QColor("#D32F2F")); table_item.setForeground(QColor("white"))
                    elif "current_limit_violation" in warnings and col in (0,24):table_item.setBackground(QColor("#D32F2F")); table_item.setForeground(QColor("white"))
                    elif "serious_relative_correction" in warnings and col in (0,12):table_item.setBackground(QColor("#E88B22"))
                    elif warnings and col in (0,12,23):table_item.setBackground(QColor("#F4C56A"))
                    self.table.setItem(row,col,table_item)
                self.table.setRowHeight(row,30)
        self.table.setSortingEnabled(True); self._updating=False
    def _item_changed(self,cell):
        if self._updating or not self.review:return
        index=cell.data(Qt.UserRole); item=next((i for i in self.review.items if i.index==index),None)
        if item is None:return
        try:
            if cell.column()==0:item.included=cell.checkState()==Qt.Checked
            elif cell.column()==14:item.individual_scale=float(cell.text())
            elif cell.column()==25:item.exclusion_reason=cell.text()
        except ValueError:QMessageBox.warning(self,"Invalid scale","Individual scale must be numeric.")
        self.refresh_all()
    def _set_selected(self,included,reason):
        if not self.review:return
        indices={self.table.item(index.row(),0).data(Qt.UserRole) for index in self.table.selectionModel().selectedRows()}
        for item in self.review.items:
            if item.index in indices:item.included=included; item.exclusion_reason="" if included else (reason or "Excluded by reviewer")
        self.refresh_all()
    def load_exclusions(self):
        path=QFileDialog.getOpenFileName(self,"Load exclusion list","","Text files (*.txt *.csv);;All files (*)")[0]
        if not path or not self.review:return
        names={line.strip().split(",")[0] for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")}
        for item in self.review.items:
            if item.name in names or (item.control_name and item.control_name in names):item.included=False; item.exclusion_reason="Loaded exclusion list"
        self.refresh_all()
    def exclude_warning_category(self):
        if not self.review:return
        categories=sorted({warning for item in self.review.items for warning in item.warnings(self.review.thresholds)})
        if not categories:QMessageBox.information(self,"No warnings","No warning categories are currently present."); return
        category,accepted=QInputDialog.getItem(self,"Exclude warning category","Warning category",categories,0,False)
        if not accepted:return
        for item in self.review.items:
            if category in item.warnings(self.review.thresholds):item.included=False; item.exclusion_reason=f"Excluded warning category: {category}"
        self.refresh_all()

    def refresh_all(self): self.refresh_table(); self.refresh_review(); self._update_workflow_tabs()
    def refresh_review(self):
        if not self.review:return
        comparison=getattr(self.review,"real_fraction_comparison",self.review.comparison()); self.comparison.setRowCount(len(comparison))
        for row,entry in enumerate(comparison):
            for col,key in enumerate(("fraction","max_abs_delta_k_over_k_percent","max_abs_delta_i_ampere","current_limit_violations","calibration_warnings","unmapped_magnets")):self.comparison.setItem(row,col,QTableWidgetItem(self._fmt(entry[key]*100 if key=="fraction" else entry.get(key,0))))
        active=[item for item in self.review.items if item.included]; included=len(active); warnings=sum(bool(item.warnings(self.review.thresholds)) for item in self.review.items); violations=sum(item.current_limit_status=="VIOLATION" for item in active); relative=[abs(item.relative_percent) for item in active if item.relative_percent is not None]; final=[abs(item.final_delta) for item in active]
        values={"Magnets loaded":str(len(self.review.items)),"Magnets included":str(included),"Magnets excluded":str(len(self.review.items)-included),"Warnings":str(warnings),"Current-limit violations":str(violations),"Global correction fraction":f"{self.review.global_scale*100:g}%","Max |ΔK/K|":self._fmt(max(relative,default=None))+" %" if relative else "Not available","Max |Final ΔK|":self._fmt(max(final,default=None)) if final else "Not available"}
        for name,value in values.items():self.review_metrics[name].setText(value)
        self._draw_plots()
    def _draw_plots(self):
        if not self.review:return
        self._ensure_plots()
        items=self.review.items; x=np.arange(len(items)); raw=np.array([i.raw_fitted_delta for i in items]); final=np.array([i.final_delta for i in items]); relative=np.array([np.nan if i.relative_percent is None else i.relative_percent for i in items]); included=np.array([i.included for i in items])
        specs={"raw":(raw,"Raw fitted ΔK","Fitted-vector position"),"final":(final,"Final machine ΔK","Fitted-vector position"),"relative":(relative,"Final ΔK/K [%]","Fitted-vector position")}
        for key,(values,ylabel,xlabel) in specs.items():
            canvas=self.plots[key]; canvas.clear(); axis=canvas.figure.add_subplot(111); axis.plot(x,values,color="#D67A13",linewidth=1); axis.scatter(x,values,c=np.where(included,"#D67A13","#888888"),s=10); axis.set(xlabel=xlabel,ylabel=ylabel,title=ylabel); axis.grid(True,alpha=.25); canvas.apply_theme()
        canvas=self.plots["magnet"]; canvas.clear(); axis=canvas.figure.add_subplot(111); axis.bar(x,final,color=np.where(included,"#D67A13","#999999")); axis.set(xlabel="Magnet / fitted parameter position",ylabel="Final ΔK",title="Correction by magnet"); canvas.apply_theme()
        canvas=self.plots["overview"]; canvas.clear(); axis=canvas.figure.add_subplot(111); axis.bar(["Included","Excluded"],[included.sum(),(~included).sum()],color=["#D67A13","#888888"]); axis.set(ylabel="Corrections",title="Included / excluded overview"); canvas.apply_theme()
        have_current=any(i.delta_i_ampere is not None for i in items); self.plot_tabs.setTabVisible(self.plot_tabs.indexOf(self.plots["current"]),have_current)
        if have_current:
            canvas=self.plots["current"]; canvas.clear(); axis=canvas.figure.add_subplot(111); values=np.array([np.nan if i.delta_i_ampere is None else i.delta_i_ampere for i in items]); axis.bar(x,values,color="#D67A13"); axis.set(xlabel="Magnet",ylabel="ΔI [A]",title="Read-only current diagnostics"); canvas.apply_theme()
    def save_plan(self):
        if not self.review:QMessageBox.information(self,"No plan","Load a correction source first."); return
        path=QFileDialog.getSaveFileName(self,"Save correction plan","correction-plan.json","JSON (*.json);;YAML (*.yaml *.yml)")[0]
        if path:
            self.review.comments=self.comments.toPlainText()
            try:save_review(path,self.review)
            except Exception as exc:QMessageBox.critical(self,"Cannot save correction plan",str(exc)); return
            self.statusBar().showMessage(f"Saved dry-run correction plan: {path}")
    def export_csv(self):
        if not self.review:return
        path=QFileDialog.getSaveFileName(self,"Export human-readable correction table","correction-plan.csv","CSV (*.csv)")[0]
        if path:save_review_csv(path,self.review)
    def _sync_theme_chrome(self):
        self.theme_key=theme_for_key(QApplication.instance().property("pyLOCOTheme")).key
        self.theme_button.setText("☀ Light" if self.theme_key=="dark" else "☾ Dark")
        for canvas in getattr(self,"plots",{}).values():canvas.apply_theme()

    def eventFilter(self,watched,event):
        if watched is QApplication.instance() and event.type()==QEvent.DynamicPropertyChange and bytes(event.propertyName())==b"pyLOCOThemePlot":self._sync_theme_chrome()
        return super().eventFilter(watched,event)

    def apply_theme(self,key):
        theme=theme_for_key(getattr(key,"key",key))
        if QApplication.instance().property("pyLOCOTheme")!=theme.key:select_suite_appearance(QApplication.instance(),theme.key)
        self._sync_theme_chrome()
    def toggle_theme(self):self.apply_theme("light" if self.theme_key=="dark" else "dark")
    def _build_about_dialog(self):
        dialog=QDialog(self); dialog.setWindowTitle("About pyLOCO Correct"); dialog.resize(600,680); layout=QVBoxLayout(dialog); scroll=QScrollArea(); scroll.setWidgetResizable(True); content=QWidget(); body=QVBoxLayout(content); logo=QLabel(); set_asset(logo,QSize(360,240),DISPLAY_ASSET,crop_transparency=False); body.addWidget(logo,0,Qt.AlignHCenter)
        for text in ("pyLOCO Correct",f"Installed pyLOCO version {PYLOCO_VERSION}","pyLOCO Correct is the offline correction-review companion to pyLOCO. It translates fitted accelerator-model corrections into explicit, scaled, reviewable dry-run correction plans. This milestone cannot write machine setpoints.","pyLOCO — Storage Ring Optics Correction","pyLOCO fits measured accelerator response data to an accelerator model to diagnose and correct optics errors.","pyLOCO Suite workflow: Measure acquisition → pyLOCO Fit and optics analysis → Correct review and machine-application workflow.",f"Contributors: {PROJECT_CONTRIBUTORS}",f"With thanks to: {PROJECT_ACKNOWLEDGEMENTS}",f"License: {PROJECT_LICENSE}"):
            label=QLabel(text); label.setAlignment(Qt.AlignCenter); label.setWordWrap(True); body.addWidget(label)
        links=QLabel(f'<a href="{PROJECT_REPOSITORY}">Repository / Source code</a> · <a href="{PROJECT_DOCUMENTATION}">Documentation</a><br><a href="{PROJECT_PAPER_URL}">Scientific reference / methodology</a> · <a href="{PROJECT_ISSUES}">Report issue</a>'); links.setAlignment(Qt.AlignCenter); links.setOpenExternalLinks(True); body.addWidget(links); row=QHBoxLayout(); citation=QPushButton("Copy citation"); citation.clicked.connect(lambda:QApplication.clipboard().setText(citation_text())); bib=QPushButton("Copy BibTeX"); bib.clicked.connect(lambda:QApplication.clipboard().setText(bibtex_text())); row.addStretch(); row.addWidget(citation); row.addWidget(bib); row.addStretch(); body.addLayout(row); body.addStretch(); scroll.setWidget(content); layout.addWidget(scroll); buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons); return dialog
    def about(self): return present_single_about_dialog(self,self._build_about_dialog)
