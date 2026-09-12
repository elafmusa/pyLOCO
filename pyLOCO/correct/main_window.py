"""Correction review and explicitly gated single-B2 simulation validation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, QSize, Qt, Signal, Slot, QThread, QUrl, QEvent
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView,QApplication,QComboBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QFileDialog,QFormLayout,QGridLayout,QGroupBox,QHBoxLayout,QHeaderView,QInputDialog,QLabel,QLineEdit,QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QScrollArea,QSizePolicy,QTabWidget,QTableWidget,QTableWidgetItem,QToolBar,QVBoxLayout,QWidget)

from pyLOCO.gui import __version__ as PYLOCO_VERSION
from pyLOCO.gui.branding import DISPLAY_ASSET,set_asset,application_icon
from pyLOCO.gui.project_info import (PROJECT_ACKNOWLEDGEMENTS,PROJECT_CONTRIBUTORS,PROJECT_DOCUMENTATION,PROJECT_ISSUES,PROJECT_LICENSE,PROJECT_PAPER_TITLE,PROJECT_PAPER_URL,PROJECT_REPOSITORY,bibtex_text,citation_text)
from pyLOCO.gui.suite import present_single_about_dialog
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.themes import theme_for_key
from pyLOCO.gui.appearance import ensure_suite_appearance,select_suite_appearance
from pyLOCO.control_system import AdapterCapability,InterfaceRegistry
from .application import CorrectionApplicationService
from .model import CorrectionReview,load_review,save_review,save_review_csv
from .petra_readonly import apply_explicit_mapping,load_mapping

AMBER_QSS="""
QLabel#correctBrand { color:#E88B22; font-size:24pt; font-weight:850; padding:2px 8px; }
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

class CorrectionSourceLoader(QObject):
    """Load a correction source without blocking the Qt event loop."""
    loaded = Signal(object, str)
    failed = Signal(str)

    def __init__(self, path, iteration=None):
        super().__init__()
        self.path = str(path)
        self.iteration = iteration

    @Slot()
    def run(self):
        try:
            review = load_review(self.path, iteration=self.iteration)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.loaded.emit(review, self.path)

class CorrectionApplyWorker(QObject):
    """Run the verified simulation transaction without freezing the GUI."""
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str, str)

    def __init__(self, transaction):
        super().__init__()
        self.transaction = transaction

    @Slot()
    def run(self):
        try:
            record = self.transaction.apply(
                confirmed=True,
                progress=lambda current,total,control,state:
                    self.progress.emit(current,total,control,state),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(record)

class CorrectMainWindow(QMainWindow):
    COLUMNS=("Apply?","Index","Lattice ordinal","Element/family name","Control / power-supply name","Mapping status","Type","Initial K","Fitted K","Current machine K","Raw fitted ΔK","Recommended machine ΔK","ΔK/K [%]","Global scale","Individual scale","Final ΔK","Target K","Current [A]","Target current [A]","ΔI [A]","Min current [A]","Max current [A]","Limit margin [A]","Calibration status","Current-limit status","Exclusion reason")
    def __init__(self, *, registry=None):
        super().__init__(); self.resize(1500,900); self.setMinimumSize(1000,700); self.setWindowTitle("pyLOCO Correct — Review and Apply"); self.setWindowIcon(application_icon("correct")); self.review:CorrectionReview|None=None; self.theme_key=ensure_suite_appearance(QApplication.instance()).key; self._updating=False; self.mapping_path=None; self.sign_difference_names=frozenset(); self.large_difference_names=frozenset(); self.machine_snapshot=None; self.registry=registry or InterfaceRegistry(); self.backend_session=None; self.correction_changes=(); self._source_load_thread=None; self._source_load_worker=None; self._apply_thread=None; self._apply_worker=None; self.setStyleSheet(AMBER_QSS); self._build(); self._sync_theme_chrome(); QApplication.instance().installEventFilter(self)
        screen=self.screen()
        if screen is not None:self.resize(min(1500,screen.availableGeometry().width()),min(900,screen.availableGeometry().height()))

    def _build(self):
        toolbar=QToolBar("Correct toolbar"); toolbar.setObjectName("mainToolbar"); toolbar.setMovable(False); self.addToolBar(toolbar)
        brand=QLabel("pyLOCO CORRECT"); brand.setObjectName("correctBrand"); brand.setFixedWidth(285); toolbar.addWidget(brand); toolbar.addSeparator()
        for text,tip,slot in (("Results…","Open a current pyLOCO Results directory",self.open_results),("Plan…","Open a correction plan or explicit legacy correction JSON",self.open_file),("Save…","Save correction plan as JSON or YAML",self.save_plan)):
            button=QPushButton(text); button.setFixedWidth(96); button.setToolTip(tip); button.clicked.connect(slot); toolbar.addWidget(button)
        spacer=QWidget(); spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,spacer.sizePolicy().verticalPolicy()); toolbar.addWidget(spacer)
        self.backend_combo=QComboBox()
        for descriptor in self.registry.descriptors():self.backend_combo.addItem(descriptor.label,descriptor.key)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed); toolbar.addWidget(self.backend_combo)
        self.badge=QLabel("MOCK • READ ONLY"); self.badge.setObjectName("safetyBadge"); self.badge.setFixedWidth(230); self.badge.setAlignment(Qt.AlignCenter); toolbar.addWidget(self.badge); self.connection_badge=QLabel("● OFFLINE"); self.connection_badge.setObjectName("correctConnection"); self.connection_badge.setProperty("connected",True); toolbar.addWidget(self.connection_badge); self.theme_button=QPushButton("☾ Dark"); self.theme_button.setFixedWidth(80); self.theme_button.clicked.connect(self.toggle_theme); toolbar.addWidget(self.theme_button)
        self.logo_button=ClickableLogo(); set_asset(self.logo_button,QSize(70,29),DISPLAY_ASSET,crop_transparency=False); self.logo_button.clicked.connect(self.about); toolbar.addWidget(self.logo_button)
        root=QWidget(); layout=QVBoxLayout(root); layout.setContentsMargins(18,14,18,18)
        heading_row=QHBoxLayout(); title=QLabel("Correction Review"); title.setStyleSheet("font-size:22pt;font-weight:800"); heading_row.addWidget(title); heading_row.addStretch(); layout.addLayout(heading_row)
        self.workflow_banner=QLabel("1  LOAD   →   2  PREVIEW   →   3  CONFIRM   →   4  APPLY   →   5  READBACK"); self.workflow_banner.setObjectName("workflowBanner"); self.workflow_banner.setAlignment(Qt.AlignCenter); self.workflow_banner.setMaximumHeight(38); layout.addWidget(self.workflow_banner)
        self.profile_badge=QLabel('Machine/profile not selected'); self.profile_badge.setWordWrap(True); self.profile_badge.setMaximumWidth(360); self.profile_badge.setVisible(False)
        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True); layout.addWidget(self.tabs,1); self.setCentralWidget(root)
        self.workflow_names=("Correction Source","Machine / Mapping","Correction Plan","Review && Validate")
        # Keep the validated transaction controller, but expose its connection
        # controls directly in Correction Plan instead of a separate B2 tab.
        from .quadrupole_widget import QuadrupoleWidget
        self.quadrupole_workspace = QuadrupoleWidget(self)
        self.tabs.addTab(self._source_page(),self.workflow_names[0]); self.tabs.addTab(self._mapping_page(),self.workflow_names[1]); self.tabs.addTab(self._plan_page(),self.workflow_names[2]); self.tabs.addTab(self._review_page(),self.workflow_names[3])
        self.tabs.tabBar().setExpanding(False); self.tabs.tabBar().setElideMode(Qt.ElideNone); self._update_workflow_tabs()
        self.backend_combo.currentIndexChanged.connect(self.quadrupole_workspace.invalidate)
        self.quadrupole_workspace.profile.currentIndexChanged.connect(self._pysc_profile_selected)
        self.statusBar().showMessage("Load a correction file; control-system writes remain confirmation-gated")

    def closeEvent(self, event):
        fullfit = getattr(self, "fullfit_transaction", None)
        fullfit_pending = bool(
            fullfit is not None
            and fullfit.record is not None
            and fullfit.record.get("restore_required")
        )
        if self.quadrupole_workspace.pending() or fullfit_pending:
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

        connection_group=QGroupBox("pySC Server"); connection_group.setObjectName("correctionPySCConnectionSection"); connection_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); connection_form=QFormLayout(connection_group); connection_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        controller=self.quadrupole_workspace; controller.connect_button.setText("Connect / verify"); connection_form.addRow("Machine / profile",controller.profile); connection_form.addRow("Diagnostics port",controller.port); connection_form.addRow("",controller.connect_button); connection_form.addRow("Connection",controller.identity); layout.addWidget(connection_group)

        mapping_group=QGroupBox("Machine Mapping"); mapping_group.setObjectName("machineMappingSection"); mapping_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        self.mapping_summary=QLabel("No correction source loaded"); self.mapping_file_status=QLabel("—"); self.mapped_status=QLabel("—"); self.unmapped_status=QLabel("—"); self.ambiguous_status=QLabel("—")
        mapping_grid=status_grid(mapping_group,(("Mapping status",self.mapping_summary),("Mapping file",self.mapping_file_status),("Mapped magnets",self.mapped_status),("Unmapped magnets",self.unmapped_status),("Ambiguous mappings",self.ambiguous_status)))
        self.mapping_source_notice=QLabel("Load a correction source first to configure PETRA magnet mapping."); self.mapping_source_notice.setWordWrap(True); self.mapping_source_notice.setObjectName("mappingSourceNotice"); mapping_grid.addWidget(self.mapping_source_notice,5,0,1,2)
        self.mapping_button=QPushButton("Load mapping…"); self.mapping_button.setMinimumWidth(210); self.mapping_button.clicked.connect(self.load_petra_mapping); self.mapping_button.setEnabled(False); mapping_grid.addWidget(self.mapping_button,6,0,1,2,Qt.AlignLeft)
        self.mapping_note=QLabel("Explicit mapping is required. Each fitted lattice element must resolve uniquely to a verified control-system name. Unmapped, ambiguous, and duplicate mappings remain blocked."); self.mapping_note.setWordWrap(True); self.mapping_note.setObjectName("mappingHelpText"); mapping_grid.addWidget(self.mapping_note,7,0,1,2)
        layout.addWidget(mapping_group)
        current_row=QHBoxLayout(); self.read_current_k_button=QPushButton("Read current K from pySC Server"); self.read_current_k_button.clicked.connect(self.read_current_pysc_k); self.current_k_status=QLabel("Connect a verified pySC machine/profile and load its explicit mapping."); self.current_k_status.setWordWrap(True); current_row.addWidget(self.read_current_k_button); current_row.addWidget(self.current_k_status,1); layout.addLayout(current_row)
        layout.addStretch(1); return self._scroll_page(page)

    def _plan_page(self):
        page=QWidget(); layout=QVBoxLayout(page)
        convention=QGroupBox("Correction convention"); convention.setObjectName("correctionConventionSection"); convention.setCheckable(True); convention.setChecked(True); convention.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); convention_layout=QVBoxLayout(convention)
        self.plan_conventions_body=QWidget(); convention_grid=QGridLayout(self.plan_conventions_body); convention_grid.setColumnMinimumWidth(0,240); convention_grid.setColumnStretch(1,1)
        definitions=(("Raw fitted ΔK","Original correction reported by pyLOCO. Never modified."),("Recommended machine ΔK","Correction after applying the explicit machine/sign convention."),("Global fraction","Fraction applied to every included magnet."),("Individual fraction","Additional scale for one magnet."),("Final ΔK","Recommended machine ΔK × global fraction × individual fraction."),("ΔK/K [%]","Final correction relative to the initial K value: 100 × Final ΔK / Initial K."))
        for row,(term,description) in enumerate(definitions):
            label=QLabel(term); label.setStyleSheet("font-weight:700"); text=QLabel(description); text.setWordWrap(True); text.setTextInteractionFlags(Qt.TextSelectableByMouse); convention_grid.addWidget(label,row,0,Qt.AlignTop); convention_grid.addWidget(text,row,1)
        convention_layout.addWidget(self.plan_conventions_body)
        safety_note=QLabel("Target K = current machine K + final ΔK. Load and Preview perform no writes."); safety_note.setWordWrap(True); safety_note.setObjectName("safetyText"); convention_layout.addWidget(safety_note)
        convention.toggled.connect(self.plan_conventions_body.setVisible); layout.addWidget(convention)
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
        legend_row=QHBoxLayout(); legend=QLabel('● Normal &nbsp;&nbsp; <span style="color:#D99029">● Amber: attention</span> &nbsp;&nbsp; <span style="color:#D32F2F">● Red: blocked / unsafe</span> &nbsp;&nbsp; <span style="color:#888888">● Gray: excluded</span>'); legend.setTextFormat(Qt.RichText); legend.setObjectName("warningLegend"); legend.setToolTip("Only diagnostic cells are highlighted; excluded rows are muted."); legend_row.addWidget(legend); legend_row.addStretch(1)
        self.clear_table_selection_button=QPushButton("Clear highlight"); self.clear_table_selection_button.clicked.connect(lambda:self.table.clearSelection()); self.show_first_columns_button=QPushButton("← First columns"); self.show_first_columns_button.clicked.connect(lambda:self.table.horizontalScrollBar().setValue(self.table.horizontalScrollBar().minimum())); self.show_last_columns_button=QPushButton("Show last columns →"); self.show_last_columns_button.clicked.connect(lambda:self.table.horizontalScrollBar().setValue(self.table.horizontalScrollBar().maximum())); legend_row.addWidget(self.clear_table_selection_button); legend_row.addWidget(self.show_first_columns_button); legend_row.addWidget(self.show_last_columns_button); layout.addLayout(legend_row)
        self.table=QTableWidget(0,len(self.COLUMNS)); self.table.setMinimumHeight(260); self.table.setHorizontalHeaderLabels(self.COLUMNS); self.table.setSortingEnabled(True); self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setSelectionMode(QAbstractItemView.ExtendedSelection); self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn); self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.table.horizontalHeader().setStretchLastSection(False); self.table.horizontalHeader().setSectionResizeMode(len(self.COLUMNS)-1,QHeaderView.Interactive); self.table.setColumnWidth(len(self.COLUMNS)-1,220); self.table.itemChanged.connect(self._item_changed); layout.addWidget(self.table,1)
        self.plan_page=page; self._selection_clear_surfaces=(page,convention,controls,legend)
        for surface in self._selection_clear_surfaces:surface.installEventFilter(self)
        scroll=self._scroll_page(page); self.plan_scroll=scroll; scroll.viewport().installEventFilter(self); return scroll

    def _review_page(self):
        page=QWidget(); layout=QVBoxLayout(page); actions=QHBoxLayout(); self.apply_status=QLabel("Select a backend and preview before Apply"); self.apply_status.setObjectName("safetyBadge"); self.apply_status.setAlignment(Qt.AlignCenter); self.apply_status.setMinimumWidth(500); self.apply_status.setMaximumWidth(650); actions.addWidget(self.apply_status); actions.addStretch(); self.preview_apply_button=QPushButton("Preview machine changes"); self.preview_apply_button.clicked.connect(self.preview_machine_changes); actions.addWidget(self.preview_apply_button); self.apply_button=QPushButton("Apply…"); self.apply_button.setEnabled(False); self.apply_button.clicked.connect(self.apply_machine_changes); actions.addWidget(self.apply_button); self.undo_button=QPushButton("Undo last 10%…"); self.undo_button.setEnabled(False); self.undo_button.setToolTip("Return to the preceding verified cumulative correction level"); self.undo_button.clicked.connect(self.undo_last_machine_changes); actions.addWidget(self.undo_button); self.restore_button=QPushButton("Restore original…"); self.restore_button.setEnabled(False); self.restore_button.setToolTip("Restore exact pre-Apply B2 values in the pySC simulation"); self.restore_button.clicked.connect(self.restore_machine_changes); actions.addWidget(self.restore_button); export=QPushButton("Export CSV…"); export.setToolTip("Export the human-readable correction table"); export.clicked.connect(self.export_csv); actions.addWidget(export); layout.addLayout(actions)
        metrics=QGroupBox(); self.review_metrics_group=metrics; metrics.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); grid=QGridLayout(metrics); self.review_metrics={}
        labels=("Magnets loaded","Magnets included","Magnets excluded","Warnings","Current-limit violations","Global correction fraction","Max |ΔK/K|","Max |Final ΔK|")
        for index,name in enumerate(labels):
            column=(index%4)*2; row=index//4; value=QLabel("—"); value.setObjectName("metricValue"); caption=QLabel(name); caption.setObjectName("metricLabel"); grid.addWidget(caption,row,column); grid.addWidget(value,row,column+1); self.review_metrics[name]=value
        layout.addWidget(metrics); self.comments=QPlainTextEdit(); self.comments.setPlaceholderText("Plan comments / review notes"); self.comments.setFixedHeight(34); layout.addWidget(self.comments)
        changes_group=QGroupBox("Machine change preview / readback"); changes_layout=QVBoxLayout(changes_group); self.machine_changes_table=QTableWidget(0,6); self.machine_changes_table.setHorizontalHeaderLabels(["Control","Current value","Requested change","Proposed value","Readback","Status"]); self.machine_changes_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.machine_changes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.machine_changes_table.setMinimumHeight(135); changes_layout.addWidget(self.machine_changes_table); layout.addWidget(changes_group)
        self.expansion_summary=QLabel(
            "Family-mode LOCO: fitted quadrupole families → "
            "expanded physical quadrupoles"
        )
        self.expansion_summary.setObjectName("runState")
        self.expansion_summary.setWordWrap(True)
        layout.addWidget(self.expansion_summary)

        self.plot_tabs=QTabWidget(); self.plots={}
        self._plot_specs=(("raw","Raw ΔK"),("final","Final ΔK"),("relative","ΔK/K [%]"),("magnet","Correction by magnet"),("overview","Included / excluded"),("current","Current diagnostics"))
        for key,title in self._plot_specs:
            placeholder=QLabel("Load correction data to display this plot."); placeholder.setAlignment(Qt.AlignCenter); self.plot_tabs.addTab(placeholder,title)
        self.plot_tabs.setMinimumHeight(380)
        self.plot_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.plot_tabs)
        return self._scroll_page(page)

    def _ensure_plots(self):
        if self.plots:return
        index=self.plot_tabs.currentIndex()
        for position,(key,title) in enumerate(self._plot_specs):
            placeholder=self.plot_tabs.widget(position); self.plot_tabs.removeTab(position); placeholder.deleteLater()
            canvas=PlotCanvas(minimum_height=270); self.plots[key]=canvas; self.plot_tabs.insertTab(position,canvas,title)
        self.plot_tabs.setCurrentIndex(index)

    def _load(self,path,iteration=None):
        try:review=load_review(path,iteration=iteration)
        except Exception as exc: QMessageBox.critical(self,"Cannot load correction source",str(exc)); return
        self._accept_loaded_review(review,path)

    def _accept_loaded_review(self,review,path):
        self.review=review
        source=Path(path).resolve(); self.machine_snapshot=None; self.badge.setText(self.registry.descriptor(self.backend_combo.currentData()).badge)
        for item in self.review.items:item.metadata.setdefault("mapping_status","mapped" if item.control_name else "unmapped")
        self.correction_changes=(); self.apply_button.setEnabled(False); self.apply_status.setText("Correction loaded — preview machine changes")
        provenance=self.review.items[0].metadata if self.review.items else {}; session=provenance.get("measurement_session") or {}; self.source_iteration.setText(str(provenance.get("source_state") or "Final / loaded plan")); self.source_timestamp.setText(str(provenance.get("fit_timestamp") or "Not available")); self.source_session.setText(str(session.get("session_id") or "Not recorded")); self.source_path.setText(str(source)); self.source_path.setToolTip(str(source)); self.source_type.setText(self._source_kind(source)); self.source_parameters.setText(self._parameter_text()); self.source_status.setText("Correction data loaded and ready for mapping review"); self.mapping_button.setEnabled(True); self.mapping_source_notice.setVisible(False); self.mapping_file_status.setText("—"); self._refresh_mapping_status(); self._sync_fraction(); self._read_current_pysc_k(strict=False); self.refresh_all(); self.tabs.setCurrentIndex(2); self.statusBar().showMessage(f"Loaded {len(self.review.items)} correction item(s)",5000)

    def load_source_responsively(self,path,iteration=None):
        if self._source_load_thread is not None:
            self.statusBar().showMessage("A correction source is already loading.",3000)
            return
        self.statusBar().showMessage(f"Loading correction source… {path}")
        thread=QThread(self); worker=CorrectionSourceLoader(path,iteration); worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._source_loaded)
        worker.failed.connect(self._source_load_failed)
        worker.loaded.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater); thread.finished.connect(self._source_load_finished); thread.finished.connect(thread.deleteLater)
        self._source_load_thread=thread; self._source_load_worker=worker; thread.start()

    @Slot(object,str)
    def _source_loaded(self,review,path):
        self._accept_loaded_review(review,path)

    @Slot(str)
    def _source_load_failed(self,message):
        self.statusBar().showMessage("Correction source could not be loaded.",5000)
        QMessageBox.critical(self,"Cannot load correction source",message)

    @Slot()
    def _source_load_finished(self):
        self._source_load_thread=None; self._source_load_worker=None

    def _browse_start(self,key):
        settings=QSettings("pyLOCO","pyLOCO Correct")
        remembered=Path(str(settings.value(f"browse/{key}",""))).expanduser()
        if remembered.is_dir():return str(remembered)
        if self.review:
            source=Path(self.review.source_result).expanduser()
            candidate=source if source.is_dir() else source.parent
            if candidate.is_dir():return str(candidate)
        measurements=Path.cwd()/"measurements"
        return str(measurements if measurements.is_dir() else Path.cwd())

    @staticmethod
    def _remember_browse(key,path):
        selected=Path(path).expanduser(); folder=selected if selected.is_dir() else selected.parent
        QSettings("pyLOCO","pyLOCO Correct").setValue(f"browse/{key}",str(folder))

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
        options=QFileDialog.ShowDirsOnly|QFileDialog.DontUseNativeDialog
        path=QFileDialog.getExistingDirectory(self,"Open current pyLOCO Results directory",self._browse_start("results"),options)
        if path:self._remember_browse("results",path); self.load_source_responsively(path)
    def open_file(self):
        path=QFileDialog.getOpenFileName(self,"Open correction plan or legacy JSON",self._browse_start("plans"),"Correction files (*.json *.yaml *.yml)",options=QFileDialog.DontUseNativeDialog)[0]
        if path:self._remember_browse("plans",path); self.load_source_responsively(path)

    def _backend_changed(self,*_):
        key=self.backend_combo.currentData(); descriptor=self.registry.descriptor(key); self.badge.setText(descriptor.badge); self.backend_session=None; self.correction_changes=(); self.apply_button.setEnabled(False)
        self._set_connection(False,"OFFLINE" if key=="mock" else "DISCONNECTED"); self._style_backend_badge(key); self._populate_machine_changes(())
        self.apply_status.setText("Mock is read-only" if key=="mock" else "Backend selected — preview reads current values")

    def _pysc_profile_selected(self,*_):
        """Selecting a pySC machine/profile also selects its backend.

        This keeps the explicit safety choice while avoiding a two-location
        workflow where the user first chooses pySC in the header and then has
        to scroll elsewhere to choose the actual machine.
        """
        if self.quadrupole_workspace.profile.currentData() is None:return
        index=self.backend_combo.findData("pysc")
        if index>=0 and self.backend_combo.currentIndex()!=index:self.backend_combo.setCurrentIndex(index)

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

    def _read_current_pysc_k(self, *, strict: bool) -> int:
        """Populate physical B2 values from the verified pySC diagnostic snapshot."""
        if not self.review:
            if strict: raise RuntimeError("Load a correction source first.")
            return 0
        if self.backend_combo.currentData() != "pysc":
            if strict: raise RuntimeError("Select pySC Server first.")
            return 0
        workspace=getattr(self,"quadrupole_workspace",None); transaction=getattr(workspace,"transaction",None)
        profile=workspace.profile.currentData() if workspace is not None else None
        if transaction is None or profile is None:
            if strict: raise RuntimeError("Select the machine/profile and connect to the pySC Server above first.")
            return 0
        snapshot=transaction.connection.snapshot()
        from .quadrupole_transaction import require_supported_simulation_profile
        require_supported_simulation_profile(snapshot["identity"],profile)
        rows={}
        for row in snapshot.get("quadrupoles",[]):
            control=str(row.get("control",""))
            if control in rows:raise RuntimeError(f"Duplicate B2 control in pySC diagnostics: {control}")
            rows[control]=row
        updated=0; unresolved=[]
        for item in self.review.items:
            if item.correction_type!="normal_quadrupole" or item.metadata.get("mapping_status")!="mapped" or not item.control_name:continue
            control=str(item.control_name); row=rows.get(control)
            if row is None:
                unresolved.append(control); continue
            if row.get("component")!="B2" or row.get("unit")!="m^-2":raise RuntimeError(f"Unsupported pySC control metadata for {control}")
            item.machine_value=float(row["physical"])
            item.metadata["current_control_setpoint"]=float(row["current"])
            item.metadata["simulation_calibration_factor"]=float(row["factor"])
            item.metadata["simulation_calibration_offset"]=float(row["offset"])
            updated+=1
        if unresolved and strict:raise RuntimeError(f"{len(unresolved)} mapped B2 control(s) are absent from the running pySC profile; first: {unresolved[0]}")
        identity=snapshot["identity"]
        self.current_k_status.setText(f"Read {updated} physical B2 value(s) from DEMO • pySC SERVER — {identity.get('machine','')} / {identity.get('scenario','')}.")
        self.refresh_all()
        return updated

    def read_current_pysc_k(self):
        try:self._read_current_pysc_k(strict=True)
        except Exception as exc:
            self.current_k_status.setText(f"Current machine K unavailable: {exc}")
            QMessageBox.warning(self,"Cannot read current machine K",str(exc))

    def preview_machine_changes(self):
        # Preview is read-only. An applied pass may be continued only after
        # its complete live state has been independently verified.
        existing=getattr(self,"fullfit_transaction",None)
        previous_record = None
        if existing is not None and existing.record is not None:
            status = existing.record.get("status")
            if status == "applied":
                try:
                    previous_record = existing.verify_applied()
                except Exception as exc:
                    QMessageBox.warning(self, "Fresh readback failed", str(exc))
                    return
                if float(previous_record.get("cumulative_fraction", 0.10)) >= 1.0:
                    QMessageBox.warning(
                        self,
                        "100% reached",
                        "The full fitted correction is already applied. Restore the original values before starting another sequence.",
                    )
                    return
            elif status in {"write_pending", "restoring", "restore_failed"}:
                QMessageBox.warning(self,"Restore required","Finish and verify restoration before another correction preview.")
                return
            elif status == "preview_verified" and existing.record.get("restore_required"):
                QMessageBox.warning(self,"Preview already ready","Apply this cumulative preview or restore the original values first.")
                return
        if not self.review:
            QMessageBox.warning(self, "No correction", "Load a correction source first.")
            return

        included = [item for item in self.review.items if item.included]

        if not included:
            QMessageBox.warning(self, "Empty correction", "No corrections are included.")
            return

        # For this milestone only mapped normal B2 corrections are supported.
        bad = [
            item for item in included
            if item.correction_type != "normal_quadrupole"
            or item.metadata.get("mapping_status") != "mapped"
            or not item.control_name
            or not str(item.control_name).endswith("/B2")
        ]

        if bad:
            QMessageBox.warning(
                self,
                "Preview blocked",
                f"{len(bad)} included correction(s) are not uniquely mapped normal B2 controls."
            )
            return

        try:
            import math
            from .quadrupole_transaction import require_supported_simulation_profile
            from .application import CorrectionChange

            if self.backend_combo.currentData() != "pysc":
                raise RuntimeError("Select pySC Server before Preview.")
            selected_profile = self.quadrupole_workspace.profile.currentData()
            transaction = self.quadrupole_workspace.transaction
            if selected_profile not in ("petra3_realistic", "ebs") or transaction is None:
                raise RuntimeError(
                    "Connect explicitly to PETRA III / realistic_errors or "
                    "EBS / validated_demo before Preview."
                )
            connection = transaction.connection
            snapshot = connection.snapshot()
            identity = snapshot["identity"]
            require_supported_simulation_profile(identity, selected_profile)

            rows_by_control = {}
            for row in snapshot.get("quadrupoles", []):
                control = row.get("control")
                if control in rows_by_control:
                    raise RuntimeError(
                        f"Duplicate B2 control in diagnostics: {control}"
                    )
                rows_by_control[control] = row

            changes = []

            for item in included:
                control = str(item.control_name)

                if control not in rows_by_control:
                    raise RuntimeError(
                        f"Mapped control is absent from pySC diagnostics: {control}"
                    )

                row = rows_by_control[control]

                if row.get("component") != "B2" or row.get("unit") != "m^-2":
                    raise RuntimeError(
                        f"Unsupported control type for {control}"
                    )

                factor = float(row["factor"])
                offset = float(row["offset"])
                current = float(row["current"])
                physical = float(row["physical"])
                physical_delta = float(item.final_delta)

                values = (factor, offset, current, physical, physical_delta)
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError(
                        f"Non-finite B2 diagnostic for {control}"
                    )

                if factor == 0:
                    raise RuntimeError(
                        f"Zero calibration factor for {control}"
                    )

                expected_physical = factor * current + offset
                if not math.isclose(
                    physical,
                    expected_physical,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Physical/control calibration mismatch for {control}"
                    )

                # final_delta is PHYSICAL ΔK.
                # Convert it to the corresponding control-setpoint change.
                control_delta = physical_delta / factor
                proposed = current + control_delta

                changes.append(
                    CorrectionChange(
                        name=control,
                        channel=f"MAGNET:{control}",
                        current=current,
                        proposed=proposed,
                        readback=None,
                        status="READ-ONLY preview",
                    )
                )

            self.correction_changes = tuple(changes)
            self._populate_machine_changes(self.correction_changes)

            # Full-FIT application is permitted ONLY for an explicitly
            # validated local pySC simulation profile and
            # only at the validated 10% correction fraction.
            if not math.isclose(
                float(self.review.global_scale),
                0.10,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    "Full-FIT pySC application requires exactly "
                    "10% global correction fraction."
                )

            if not self.mapping_path:
                raise RuntimeError(
                    "A persisted explicit PETRA mapping file is required."
                )

            import hashlib
            import json
            from pathlib import Path
            from pyLOCO.gui.results.results_loader import ResultsLoader
            from .fullfit_transaction import (
                FullFitB2Transaction,
                SCHEMA as FULLFIT_SCHEMA,
            )

            source_result = Path(self.review.source_result).resolve()
            result_loader = ResultsLoader(source_result)

            source_lattice = result_loader._resolve_reference(
                result_loader.request.get("lattice_path", "")
            )
            fitted_lattice = result_loader.fitted_lattice_path
            mapping_file = Path(self.mapping_path).resolve()

            if source_lattice is None or not source_lattice.is_file():
                raise RuntimeError(
                    "Initial/source lattice required for full-FIT "
                    "application is unavailable."
                )

            if not fitted_lattice.is_file():
                raise RuntimeError(
                    "Fitted lattice required for full-FIT application "
                    "is unavailable."
                )

            if not mapping_file.is_file():
                raise RuntimeError(
                    "Explicit PETRA mapping file is unavailable."
                )

            from pyLOCO.control_system.pysc_profiles import load_pysc_profile
            profile_lattice = load_pysc_profile(selected_profile).resolve("lattice_file")
            if not profile_lattice.is_file():
                raise RuntimeError(f"Selected profile lattice is unavailable: {profile_lattice}")

            def sha256(path):
                return hashlib.sha256(
                    Path(path).read_bytes()
                ).hexdigest()

            profile_lattice_sha256 = sha256(profile_lattice)
            if identity.get("lattice_sha256") != profile_lattice_sha256:
                raise RuntimeError(
                    "Running pySC server lattice does not match the selected profile lattice."
                )
            if selected_profile == "ebs":
                import yaml
                mapping_data = yaml.safe_load(mapping_file.read_text(encoding="utf-8"))
                if mapping_data.get("lattice_sha256") != profile_lattice_sha256:
                    raise RuntimeError(
                        "EBS mapping does not match the EBS / validated_demo lattice."
                    )
                # The validated EBS profile lattice is the authoritative
                # transaction baseline; do not inherit a stale Results path.
                source_lattice = profile_lattice
            elif sha256(source_lattice) != profile_lattice_sha256:
                raise RuntimeError(
                    "Loaded FIT source lattice does not match the selected profile lattice."
                )

            records = []
            previous_items = {
                row["control"]: row
                for row in (previous_record or {}).get("items", [])
            }
            included_controls = {str(item.control_name) for item in included}
            if previous_record is not None and set(previous_items) != included_controls:
                raise RuntimeError(
                    "The included correction set changed since the preceding Apply. "
                    "Restore original values before changing the plan."
                )
            application_number = int(
                (previous_record or {}).get("application_number", 0)
            ) + 1

            for item in included:
                control = str(item.control_name)
                row = rows_by_control[control]

                factor = float(row["factor"])
                offset = float(row["offset"])
                current = float(row["current"])
                physical = float(row["physical"])
                physical_delta = float(item.final_delta)

                previous_item = previous_items.get(control)
                if previous_record is not None and previous_item is None:
                    raise RuntimeError(
                        f"Cumulative correction set changed; missing previous control: {control}"
                    )
                if previous_item is not None and not math.isclose(
                    physical_delta,
                    float(previous_item["physical_delta"]),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise RuntimeError(
                        "Correction values changed since the preceding Apply. "
                        "Restore original values before changing the plan."
                    )

                control_delta = physical_delta / factor
                proposed = current + control_delta
                expected_physical = physical + physical_delta

                records.append(
                    {
                        "control": control,
                        "lattice_ordinal": int(
                            item.lattice_ordinal
                        ),
                        "lattice_name": str(item.name),
                        "family": item.family,
                        "factor": factor,
                        "offset": offset,
                        "current_control": current,
                        "current_physical_k": physical,
                        "physical_delta": physical_delta,
                        "control_delta": control_delta,
                        "proposed_control": proposed,
                        "expected_physical_k": expected_physical,
                        "restore_control": float(
                            previous_item.get("restore_control", previous_item["current_control"])
                            if previous_item is not None else current
                        ),
                        "restore_physical_k": float(
                            previous_item.get("restore_physical_k", previous_item["current_physical_k"])
                            if previous_item is not None else physical
                        ),
                    }
                )

            manifest = {
                "schema": FULLFIT_SCHEMA,
                "mode": "READ_ONLY_PREVIEW",
                "count": len(records),
                "fraction": 0.10,
                "application_number": application_number,
                "cumulative_fraction": application_number * 0.10,
                "restore_required": previous_record is not None,
                "history": list((previous_record or {}).get("history", [])),
                "server_identity": identity,
                "source_lattice": {
                    "path": str(source_lattice),
                    "sha256": sha256(source_lattice),
                },
                "fitted_lattice": {
                    "path": str(fitted_lattice),
                    "sha256": sha256(fitted_lattice),
                },
                "mapping_file": {
                    "path": str(mapping_file),
                    "sha256": sha256(mapping_file),
                },
                "records": records,
            }

            manifest_path = (
                source_result
                / "correction"
                / f"fullfit_b2_preview_pass_{application_number:02d}.json"
            )
            manifest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )

            self.fullfit_transaction = FullFitB2Transaction(
                connection=connection,
                journal_directory=(
                    source_result
                    / "correction"
                    / "fullfit_transaction_journals"
                ),
            )

            # Re-open and independently verify the frozen preview against
            # the still-running pySC server before enabling Apply.
            self.fullfit_transaction.load_preview_manifest(
                manifest_path
            )

            self.fullfit_manifest_path = str(manifest_path)

            self.apply_button.setEnabled(True)
            self.restore_button.setEnabled(previous_record is not None)
            self.undo_button.setEnabled(previous_record is not None)
            self.preview_apply_button.setText("Preview ready")
            self.preview_apply_button.setEnabled(False)

            self.apply_status.setText(
                f"VERIFIED pySC preview: {len(changes)} mapped B2 "
                f"controls — next +10%, cumulative {application_number * 10}%"
            )

            self.profile_badge.setText(
                f"{identity.get('machine', selected_profile)} / {identity.get('scenario', '')}"
            )

            self._set_connection(True, "READ-ONLY PREVIEW")

        except Exception as exc:
            self.correction_changes = ()
            self._populate_machine_changes(())
            self.apply_button.setEnabled(False)
            self.preview_apply_button.setText("Preview machine changes")
            self.preview_apply_button.setEnabled(True)
            self._set_connection(False, "PREVIEW FAILED")

            QMessageBox.critical(
                self,
                "Cannot preview machine changes",
                str(exc),
            )

    def apply_machine_changes(self):
        import math

        transaction = getattr(
            self,
            "fullfit_transaction",
            None,
        )

        if transaction is None:
            QMessageBox.warning(
                self,
                "Fresh Preview required",
                "Generate and verify a fresh 10% pySC preview first.",
            )
            return

        if (
            not self.review
            or not math.isclose(
                float(self.review.global_scale),
                0.10,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            QMessageBox.warning(
                self,
                "Application blocked",
                "Full-FIT simulation application requires exactly "
                "10% correction fraction.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Confirm pySC full-FIT application",
            "Apply the next verified 10% correction in the same fitted direction "
            f"(cumulative {transaction.record.get('cumulative_fraction', .1) * 100:.0f}%) to the mapped normal "
            "B2 controls of the explicitly connected pySC simulation?\n\n"
            "This writes ONLY to the local pySC simulation. "
            "No PETRA/DOOCS hardware writes are performed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        self.apply_button.setEnabled(False)
        self.preview_apply_button.setEnabled(False)
        self.restore_button.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.apply_status.setText("Applying verified correction…")
        thread=QThread(self); worker=CorrectionApplyWorker(transaction); worker.moveToThread(thread)
        thread.started.connect(worker.run); worker.progress.connect(self._apply_progress)
        worker.completed.connect(self._apply_completed); worker.failed.connect(self._apply_failed)
        worker.completed.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater); thread.finished.connect(self._apply_finished); thread.finished.connect(thread.deleteLater)
        self._apply_thread=thread; self._apply_worker=worker; thread.start()

    @Slot(int,int,str,str)
    def _apply_progress(self,current,total,control,state):
        self.apply_status.setText(f"Applying {current} / {total} — {control} — {state}")
        self.statusBar().showMessage(f"Correction progress: {current} / {total}")

    @Slot(object)
    def _apply_completed(self,record):
        try:

            readbacks = {
                item["control"]: item.get(
                    "applied_readback",
                    {},
                )
                for item in record["items"]
            }

            updated = []

            from .application import CorrectionChange

            for change in self.correction_changes:
                rb = readbacks.get(change.name, {})
                value = rb.get("control")

                updated.append(
                    CorrectionChange(
                        name=change.name,
                        channel=change.channel,
                        current=change.current,
                        proposed=change.proposed,
                        readback=value,
                        status=(
                            "VERIFIED"
                            if value is not None
                            else "NO READBACK"
                        ),
                    )
                )

            self.correction_changes = tuple(updated)
            self._populate_machine_changes(
                self.correction_changes
            )

            self.apply_button.setEnabled(False)
            self.restore_button.setEnabled(True)
            self.undo_button.setEnabled(True)
            self.preview_apply_button.setText("Preview next +10%")
            self.preview_apply_button.setEnabled(True)

            self.apply_status.setText(
                f"APPLIED AND VERIFIED in pySC simulation: "
                f"{len(record['items'])} B2 controls — cumulative "
                f"{record.get('cumulative_fraction', .1) * 100:.0f}% — "
                "Preview again for the next +10%, or restore original"
            )

            self.statusBar().showMessage(f"Applied and verified • {len(record['items'])} B2 controls • DEMO only")

        except Exception as exc:
            self._apply_failed(str(exc))

    @Slot(str)
    def _apply_failed(self,message):
        self.apply_button.setEnabled(False)
        self.preview_apply_button.setText("Preview machine changes")
        self.preview_apply_button.setEnabled(True)
        QMessageBox.critical(self,"Full-FIT simulation application failed",message)

    @Slot()
    def _apply_finished(self):
        self._apply_thread=None; self._apply_worker=None

    def undo_last_machine_changes(self):
        transaction = getattr(self, "fullfit_transaction", None)
        if transaction is None or not transaction.record:
            QMessageBox.warning(self, "Nothing to undo", "No applied 10% increment is available.")
            return
        current = float(transaction.record.get("cumulative_fraction", 0.0))
        target = max(0.0, current - 0.10)
        answer = QMessageBox.question(
            self,
            "Undo last 10%",
            f"Return the verified pySC correction from {current * 100:.0f}% "
            f"to {target * 100:.0f}%?\n\nThe current machine state is checked before any write.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            record = transaction.undo_last()
            remaining = float(record.get("cumulative_fraction", 0.0))
            self.apply_button.setEnabled(False)
            self.undo_button.setEnabled(remaining > 0)
            self.restore_button.setEnabled(remaining > 0)
            self.preview_apply_button.setEnabled(True)
            self.preview_apply_button.setText(
                "Preview next +10%" if remaining > 0 else "Preview machine changes"
            )
            self.apply_status.setText(
                f"LAST 10% UNDONE AND VERIFIED — cumulative {remaining * 100:.0f}%"
            )
            self.statusBar().showMessage(
                f"Undo verified • cumulative correction {remaining * 100:.0f}% • DEMO only"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Undo failed", str(exc))

    def restore_machine_changes(self):
        transaction = getattr(
            self,
            "fullfit_transaction",
            None,
        )

        if transaction is None:
            QMessageBox.warning(
                self,
                "No transaction",
                "No applied pySC transaction is available to restore.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Restore original pySC values",
            "Restore all B2 controls from this transaction to their "
            "exact pre-Apply values?\n\n"
            "This writes ONLY to the explicitly connected local "
            "pySC simulation. No PETRA/DOOCS hardware writes are performed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer != QMessageBox.Yes:
            return

        try:
            record = transaction.restore()

            restored = sum(
                item.get("restoration_status") == "restored"
                for item in record["items"]
            )
            failed = sum(
                item.get("restoration_status") == "failed"
                for item in record["items"]
            )

            self.restore_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.apply_button.setEnabled(False)
            self.preview_apply_button.setText("Preview machine changes")
            self.preview_apply_button.setEnabled(True)

            self.apply_status.setText(
                f"RESTORED AND VERIFIED in pySC simulation: "
                f"{restored} B2 controls"
            )

            # Refresh the displayed readback from the restored machine.
            from .application import CorrectionChange

            snapshot = transaction.connection.snapshot()
            rows = {
                row["control"]: row
                for row in snapshot["quadrupoles"]
            }

            updated = []

            for change in self.correction_changes:
                row = rows.get(change.name)
                readback = (
                    float(row["current"])
                    if row is not None
                    else None
                )

                updated.append(
                    CorrectionChange(
                        name=change.name,
                        channel=change.channel,
                        current=change.current,
                        proposed=change.proposed,
                        readback=readback,
                        status=(
                            "RESTORED"
                            if row is not None
                            else "NO READBACK"
                        ),
                    )
                )

            self.correction_changes = tuple(updated)
            self._populate_machine_changes(
                self.correction_changes
            )

            if record["status"] != "restored" or failed:
                raise RuntimeError(
                    f"Restore incomplete: status={record['status']}, "
                    f"failed={failed}"
                )

            self.statusBar().showMessage(f"Restored and verified • {restored} B2 controls")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Restore failed",
                str(exc),
            )

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
        path=QFileDialog.getOpenFileName(self,"Load explicit magnet mapping",self._browse_start("mapping"),"Mapping files (*.json *.yaml *.yml)",options=QFileDialog.DontUseNativeDialog)[0]
        if not path or not self.review:return
        self._remember_browse("mapping",path)
        try:counts=apply_explicit_mapping(self.review,load_mapping(path))
        except Exception as exc:QMessageBox.critical(self,"Cannot load PETRA mapping",str(exc)); return
        self.mapping_path=str(Path(path).resolve()); self.mapping_file_status.setText(Path(path).name); self.mapping_file_status.setToolTip(self.mapping_path); self._refresh_mapping_status(); self._read_current_pysc_k(strict=False); self.statusBar().showMessage(f"Mapping loaded: {counts['mapped']} mapped, {counts['unmapped']} unmapped, {counts['ambiguous']} ambiguous, {counts['duplicate']} duplicate."); self.refresh_all()

    def _workflow_complete(self):
        source=self.review is not None
        mapping=bool(source and self.review.items and all(item.metadata.get("mapping_status")=="mapped" for item in self.review.items))
        plan=bool(mapping and self.review.items and any(item.included for item in self.review.items) and all(item.initial_value is not None for item in self.review.items))
        review=bool(plan and not any(item.current_limit_status=="VIOLATION" for item in self.review.items if item.included))
        return source,mapping,plan,review

    def _update_workflow_tabs(self):
        complete=self._workflow_complete()
        for index,name in enumerate(self.workflow_names):self.tabs.setTabText(index,("✓ " if complete[index] else "")+name)
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
        if not categories:self.statusBar().showMessage("No warnings to exclude"); return
        category,accepted=QInputDialog.getItem(self,"Exclude warning category","Warning category",categories,0,False)
        if not accepted:return
        for item in self.review.items:
            if category in item.warnings(self.review.thresholds):item.included=False; item.exclusion_reason=f"Excluded warning category: {category}"
        self.refresh_all()

    def refresh_all(self): self.refresh_table(); self.refresh_review(); self._update_workflow_tabs()
    def refresh_review(self):
        if not self.review:return
        active=[item for item in self.review.items if item.included]; included=len(active); warnings=sum(bool(item.warnings(self.review.thresholds)) for item in self.review.items); violations=sum(item.current_limit_status=="VIOLATION" for item in active); relative=[abs(item.relative_percent) for item in active if item.relative_percent is not None]; final=[abs(item.final_delta) for item in active]
        values={"Magnets loaded":str(len(self.review.items)),"Magnets included":str(included),"Magnets excluded":str(len(self.review.items)-included),"Warnings":str(warnings),"Current-limit violations":str(violations),"Global correction fraction":f"{self.review.global_scale*100:g}%","Max |ΔK/K|":self._fmt(max(relative,default=None))+" %" if relative else "Not available","Max |Final ΔK|":self._fmt(max(final,default=None)) if final else "Not available"}
        for name,value in values.items():self.review_metrics[name].setText(value)

        families={
            item.family
            for item in self.review.items
            if item.correction_type=="normal_quadrupole"
            and item.family is not None
        }
        physical=sum(
            item.correction_type=="normal_quadrupole"
            for item in self.review.items
        )

        if families:
            self.expansion_summary.setText(
                f"Family-mode LOCO: {len(families)} fitted quadrupole "
                f"families → {physical} physical quadrupoles"
            )
        else:
            self.expansion_summary.setText(
                f"Correction review: {physical} physical quadrupoles"
            )

        self._draw_plots()
    def _draw_plots(self):
        if not self.review:return
        self._ensure_plots()
        items=self.review.items; x=np.arange(len(items)); raw=np.array([i.raw_fitted_delta for i in items]); final=np.array([i.final_delta for i in items]); relative=np.array([np.nan if i.relative_percent is None else i.relative_percent for i in items]); included=np.array([i.included for i in items])
        # Correct operates on expanded physical magnets.  A family-mode
        # LOCO fit may therefore contain fewer fitted DOFs than the number
        # of correction rows shown here (for this demo: 193 families ->
        # 399 physical quadrupoles).
        specs={
            "raw":(
                raw,
                "ΔK [m⁻²]",
                "Expanded family correction → physical quadrupoles",
            ),
            "final":(
                final,
                "ΔK [m⁻²]",
                "Final machine correction",
            ),
            "relative":(
                relative,
                "ΔK/K [%]",
                "Final relative machine correction",
            ),
        }

        for key,(values,ylabel,title) in specs.items():
            canvas=self.plots[key]
            canvas.clear()
            axis=canvas.figure.add_subplot(111)
            axis.plot(x,values,color="#D67A13",linewidth=1)
            axis.scatter(
                x,values,
                c=np.where(included,"#D67A13","#888888"),
                s=10,
            )
            axis.set(
                xlabel="Physical quadrupole position",
                ylabel=ylabel,
                title=title,
            )
            axis.grid(True,alpha=.25)
            canvas.apply_theme()
        canvas=self.plots["magnet"]; canvas.clear(); axis=canvas.figure.add_subplot(111); axis.bar(x,final,color=np.where(included,"#D67A13","#999999")); axis.set(
            xlabel="Physical quadrupole position",
            ylabel="ΔK [m⁻²]",
            title="Correction by physical quadrupole",
        ); canvas.apply_theme()
        canvas=self.plots["overview"]; canvas.clear(); axis=canvas.figure.add_subplot(111); axis.bar(["Included","Excluded"],[included.sum(),(~included).sum()],color=["#D67A13","#888888"]); axis.set(ylabel="Corrections",title="Included / excluded overview"); canvas.apply_theme()
        have_current=any(i.delta_i_ampere is not None for i in items); self.plot_tabs.setTabVisible(self.plot_tabs.indexOf(self.plots["current"]),have_current)
        if have_current:
            canvas=self.plots["current"]; canvas.clear(); axis=canvas.figure.add_subplot(111); values=np.array([np.nan if i.delta_i_ampere is None else i.delta_i_ampere for i in items]); axis.bar(x,values,color="#D67A13"); axis.set(xlabel="Magnet",ylabel="ΔI [A]",title="Read-only current diagnostics"); canvas.apply_theme()
    def save_plan(self):
        if not self.review:self.statusBar().showMessage("Load a correction source first"); return
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
        if event.type()==QEvent.MouseButtonPress and hasattr(self,"table") and (watched in getattr(self,"_selection_clear_surfaces",()) or watched is getattr(getattr(self,"plan_scroll",None),"viewport",lambda:None)()):
            self.table.clearSelection()
        if watched is QApplication.instance() and event.type()==QEvent.DynamicPropertyChange and bytes(event.propertyName())==b"pyLOCOThemePlot":self._sync_theme_chrome()
        return super().eventFilter(watched,event)

    def apply_theme(self,key):
        theme=theme_for_key(getattr(key,"key",key))
        if QApplication.instance().property("pyLOCOTheme")!=theme.key:select_suite_appearance(QApplication.instance(),theme.key)
        self._sync_theme_chrome()
    def toggle_theme(self):self.apply_theme("light" if self.theme_key=="dark" else "dark")
    def _build_about_dialog(self):
        dialog=QDialog(self); dialog.setWindowTitle("About pyLOCO Correct"); dialog.resize(600,680); layout=QVBoxLayout(dialog); scroll=QScrollArea(); scroll.setWidgetResizable(True); content=QWidget(); body=QVBoxLayout(content); logo=QLabel(); set_asset(logo,QSize(360,240),DISPLAY_ASSET,crop_transparency=False); body.addWidget(logo,0,Qt.AlignHCenter)
        for text in ("pyLOCO Correct",f"Installed pyLOCO version {PYLOCO_VERSION}","Review, scale and safely apply mapped corrections. Hardware writes remain unavailable unless an explicitly supported backend is connected.","pyLOCO — Storage Ring Optics Correction","pyLOCO fits measured response data to an accelerator model to diagnose and correct optics errors.","pyLOCO Suite workflow: Measure → Fit and analyze → Review and correct.",f"Contributors: {PROJECT_CONTRIBUTORS}",f"With thanks to: {PROJECT_ACKNOWLEDGEMENTS}",f"License: {PROJECT_LICENSE}"):
            label=QLabel(text); label.setAlignment(Qt.AlignCenter); label.setWordWrap(True); body.addWidget(label)
        links=QLabel(f'<a href="{PROJECT_REPOSITORY}">Repository / Source code</a> · <a href="{PROJECT_DOCUMENTATION}">Documentation</a><br><a href="{PROJECT_PAPER_URL}">Scientific reference / methodology</a> · <a href="{PROJECT_ISSUES}">Report issue</a>'); links.setAlignment(Qt.AlignCenter); links.setOpenExternalLinks(True); body.addWidget(links); row=QHBoxLayout(); citation=QPushButton("Copy citation"); citation.clicked.connect(lambda:QApplication.clipboard().setText(citation_text())); bib=QPushButton("Copy BibTeX"); bib.clicked.connect(lambda:QApplication.clipboard().setText(bibtex_text())); row.addStretch(); row.addWidget(citation); row.addWidget(bib); row.addStretch(); body.addLayout(row); body.addStretch(); scroll.setWidget(content); layout.addWidget(scroll); buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons); return dialog
    def about(self): return present_single_about_dialog(self,self._build_about_dialog)
