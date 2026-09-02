"""pyLOCO Measure GUI for structured, read-only measurement acquisition."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Sequence

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot, QSize, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QDoubleSpinBox, QSizePolicy, QTabWidget, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

from pyLOCO.control_system import (AdapterCapability, InterfaceRegistry, MockAdapter,
                                   PETRAReadOnlyAdapter, OptionalDependencyUnavailable,
                                   available_pysc_profiles)
from pyLOCO.data_schema import (
    MeasurementSession, SessionFile, load_session, save_session,
    validate_measurement_file, write_bpm_noise, write_dispersion, write_orm,
)
from pyLOCO.gui.branding import DISPLAY_ASSET, set_asset
from pyLOCO.gui.project_info import (
    PROJECT_ACKNOWLEDGEMENTS, PROJECT_CONTRIBUTORS, PROJECT_DOCUMENTATION,
    PROJECT_ISSUES, PROJECT_LICENSE, PROJECT_PAPER_TITLE, PROJECT_PAPER_URL,
    PROJECT_REPOSITORY, bibtex_text, citation_text,
)
from pyLOCO.gui import __version__ as PYLOCO_VERSION
from pyLOCO.data_schema import SCHEMA_VERSION as MEASUREMENT_SCHEMA_VERSION
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.themes import apply_application_theme, theme_for_key
from pyLOCO.gui.suite import launch_suite_application,present_single_about_dialog
from .acquisition import (
    AcquisitionCancelled, BpmDevice, BpmNoiseAcquirer, BpmNoiseResult, CorrectorDevice,
    DispersionResult, DispersionStateAcquirer, ORMAcquirer, ORMResult,
)
from .project import MeasureProject, load_measure_project, save_measure_project
from .automatic import AutomaticDispersionAcquirer
from .dispersion import MOMENTUM_RELATION, physical_dispersion, relative_momentum_deviation


TEAL_QSS = """
QToolBar#mainToolbar { spacing: 5px; padding: 6px 8px; }
QToolBar#mainToolbar QPushButton { font-size: 10pt; min-height: 27px; padding: 4px 8px; }
QLabel#measureBrand { color: #12BFC4; font-size: 21pt; font-weight: 850; padding: 1px 6px 1px 2px; }
QLabel#connectionStatus { background: #123B42; color: #67E8E8; border: 1px solid #20BFC4;
 border-radius: 10px; padding: 7px 14px; font-size: 12pt; font-weight: 900; }
QLabel#machineIdentity { color:#CFFAFE; padding:3px 9px; font-size:9.5pt; font-weight:750; }
QLabel#connectionState { border-radius: 9px; padding: 6px 11px; font-size: 10pt; font-weight: 850; }
QLabel#connectionState[connected="true"] { background:#123D2A; color:#7BE3A7; border:1px solid #38B875; }
QLabel#connectionState[connected="false"] { background:#4A2424; color:#FFAAAA; border:1px solid #D96666; }
QLabel#savedPath { background:#102F35; color:#A8F4F1; border:1px solid #20BFC4; border-radius:7px; padding:10px; font-weight:750; }
QFrame#measureLogoContainer { background: #FFFFFF; border: 1px solid #D7DCE6; border-radius: 8px; }
QLabel#measureLogo { border: 0; background: transparent; }
QLabel#measurementHelpTitle { color: #12BFC4; font-size: 12pt; font-weight: 750; }
QLabel#measurementConvention { font-family: monospace; font-weight: 650; }
QLabel#planValue { font-weight: 650; }
QLabel#runState { font-size: 14pt; font-weight: 750; color: #12BFC4; }
QLabel#runMetric { font-size: 10.5pt; }
QPushButton#measurePrimary { background: #0F9FA6; border-color: #31D5D8; color: white; }
QPushButton#measurePrimary:hover { background: #0C858B; }
QTabBar::tab:selected { border-color: #18B8BE; }
QGroupBox::title { color: #13A9AF; }
QProgressBar { min-height: 34px; }
QProgressBar::chunk { background: #12BFC4; }
"""


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        if not self.hasFocus(): event.ignore(); return
        super().wheelEvent(event)


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # type: ignore[override]
        if not self.hasFocus(): event.ignore(); return
        super().wheelEvent(event)


class ElidedPathLabel(QLabel):
    """Middle-elided display that retains the complete path as a tooltip."""
    def __init__(self):
        super().__init__(); self._full_text=""; self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def setFullText(self, text: str) -> None:
        self._full_text=text; self.setToolTip(text); self._refresh()

    def resizeEvent(self,event):  # type: ignore[override]
        super().resizeEvent(event); self._refresh()

    def _refresh(self):
        if self._full_text: super().setText(self.fontMetrics().elidedText(self._full_text,Qt.ElideMiddle,max(40,self.width())))


class ClickableLogoLabel(QLabel):
    """The approved logo rendered unchanged, with button-like accessibility."""

    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("measureLogo")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("About pyLOCO Measure")
        self.setAccessibleName("About pyLOCO Measure")

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


def default_mock_devices(count: int = 12) -> tuple[BpmDevice, ...]:
    return tuple(BpmDevice(f"BPM-{number:03d}", f"MOCK/BPM/{number:03d}/X", f"MOCK/BPM/{number:03d}/Y")
                 for number in range(1, count + 1))


def default_mock_correctors(horizontal: int = 6, vertical: int = 6):
    h=tuple(CorrectorDevice(f"HCOR-{i:03d}",f"MOCK/HCOR/{i:03d}/KICK",f"MOCK/HCOR/{i:03d}/KICK","Horizontal") for i in range(1,horizontal+1))
    v=tuple(CorrectorDevice(f"VCOR-{i:03d}",f"MOCK/VCOR/{i:03d}/KICK",f"MOCK/VCOR/{i:03d}/KICK","Vertical") for i in range(1,vertical+1))
    return h,v


def build_mock_adapter(devices: Sequence[BpmDevice], readings: int = 100, *, horizontal_correctors=None, vertical_correctors=None, response_matrix=None, enable_orm_writes=False) -> MockAdapter:
    channels = {}
    sequences = {}
    rf_channels = {}
    sample = np.arange(readings, dtype=float)
    for index, device in enumerate(devices):
        base_x = (index + 1) * 2e-6; base_y = -(index + 1) * 1e-6
        noise_x = (index + 1) * 8e-9 * np.sin(0.71 * sample + index)
        noise_y = (index + 1) * 5e-9 * np.cos(0.53 * sample + index)
        channels[device.x_channel] = float(base_x); channels[device.y_channel] = float(base_y)
        sequences[device.x_channel] = noise_x.tolist(); sequences[device.y_channel] = noise_y.tolist()
        rf_channels[device.x_channel] = (base_x, (index + 1) * 2.0e-9)
        rf_channels[device.y_channel] = (base_y, -(index + 1) * 0.7e-9)
    hcors=tuple(horizontal_correctors or ()); vcors=tuple(vertical_correctors or ()); correctors=hcors+vcors
    dependencies={}
    if correctors:
        if response_matrix is None:
            rows=np.arange(2*len(devices))[:,None]; cols=np.arange(len(correctors))[None,:]; response_matrix=2.5e-3*np.sin(.37*(rows+1)+.61*(cols+1))
        matrix=np.asarray(response_matrix,float)
        if matrix.shape!=(2*len(devices),len(correctors)): raise ValueError("Mock ORM shape does not match device catalog")
        for row,device in enumerate(devices):
            dependencies[device.x_channel]=(channels[device.x_channel],{cor.setpoint_channel:matrix[row,col] for col,cor in enumerate(correctors)})
            dependencies[device.y_channel]=(channels[device.y_channel],{cor.setpoint_channel:matrix[len(devices)+row,col] for col,cor in enumerate(correctors)})
        for cor in correctors: channels[cor.setpoint_channel]=0.0
    catalog = {"bpm": [
        {"name": device.name, "x_channel": device.x_channel, "y_channel": device.y_channel}
        for device in devices
    ],"hcor":[{"name":c.name,"setpoint_channel":c.setpoint_channel,"readback_channel":c.readback_channel,"plane":c.plane} for c in hcors],"vcor":[{"name":c.name,"setpoint_channel":c.setpoint_channel,"readback_channel":c.readback_channel,"plane":c.plane} for c in vcors]}
    return MockAdapter(
        channels, timestamp_step=0.01, sequences=sequences, device_catalog=catalog,
        rf_dependent_channels=rf_channels, nominal_rf_hz=500_000_000.0,
        setpoint_dependent_channels=dependencies, allow_simulated_writes=bool(correctors) and bool(enable_orm_writes),
    )


class AcquisitionWorker(QObject):
    progress = Signal(int, int, float, object, object)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, acquirer, readings, delay, cancel_event):
        super().__init__(); self.acquirer=acquirer; self.readings=readings; self.delay=delay; self.cancel_event=cancel_event

    @Slot()
    def run(self):
        try:
            result = self.acquirer.acquire(self.readings, self.delay, cancel_event=self.cancel_event,
                                           progress=lambda *args: self.progress.emit(*args))
        except AcquisitionCancelled as exc: self.cancelled.emit(str(exc))
        except Exception as exc: self.failed.emit(str(exc))
        else: self.completed.emit(result)


class DispersionAcquisitionWorker(QObject):
    progress = Signal(int, int, float, object, object)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, acquirer, label, requested_rf_hz, readings, delay, settling, cancel_event):
        super().__init__(); self.acquirer=acquirer; self.label=label; self.requested_rf_hz=requested_rf_hz; self.readings=readings; self.delay=delay; self.settling=settling; self.cancel_event=cancel_event

    @Slot()
    def run(self):
        try:
            if self.settling and self.cancel_event.wait(self.settling):
                raise AcquisitionCancelled("Dispersion acquisition was cancelled during RF settling")
            result = self.acquirer.acquire(
                self.label, self.requested_rf_hz, self.readings, self.delay,
                cancel_event=self.cancel_event, progress=lambda *args: self.progress.emit(*args),
            )
        except AcquisitionCancelled as exc: self.cancelled.emit(str(exc))
        except Exception as exc: self.failed.emit(str(exc))
        else: self.completed.emit(result)


class AutomaticDispersionWorker(QObject):
    progress = Signal(int, int, int, float, object, object)
    status = Signal(str, object)
    completed = Signal(object); cancelled = Signal(str); failed = Signal(str)

    def __init__(self, acquirer, options, cancel_event):
        super().__init__(); self.acquirer=acquirer; self.options=options; self.cancel_event=cancel_event

    @Slot()
    def run(self):
        try:
            result=self.acquirer.acquire(cancel_event=self.cancel_event,
                                         progress=lambda *args:self.progress.emit(*args),
                                         status=lambda phase,details:self.status.emit(phase,details), **self.options)
        except AcquisitionCancelled as exc:self.cancelled.emit(str(exc))
        except Exception as exc:self.failed.emit(str(exc))
        else:self.completed.emit(result)


class ORMAcquisitionWorker(QObject):
    event=Signal(object); completed=Signal(object); cancelled=Signal(str); failed=Signal(str)
    def __init__(self,acquirer,kick_h,kick_v,options,cancel_event): super().__init__(); self.acquirer=acquirer; self.kick_h=kick_h; self.kick_v=kick_v; self.options=options; self.cancel_event=cancel_event
    @Slot()
    def run(self):
        try:self.completed.emit(self.acquirer.acquire(self.kick_h,self.kick_v,cancel_event=self.cancel_event,progress=self.event.emit,**self.options))
        except AcquisitionCancelled as exc:self.cancelled.emit(f"{exc} (restoration: {getattr(exc,'restoration_status','unknown')})")
        except Exception as exc:self.failed.emit(f"{exc} (restoration: {getattr(exc,'restoration_status','unknown')})")


class MeasureMainWindow(QMainWindow):
    def __init__(self, *, devices: Sequence[BpmDevice] | None = None, adapter=None,
                 horizontal_correctors=None, vertical_correctors=None) -> None:
        super().__init__()
        self.setWindowTitle("pyLOCO Measure")
        self.resize(1200, 800); self.setMinimumSize(900, 650)
        if devices is None and adapter is not None and hasattr(adapter, "list_devices"):
            devices = tuple(BpmDevice(**item) for item in adapter.list_devices("bpm"))
        self.devices = tuple(devices or default_mock_devices())
        default_h,default_v=default_mock_correctors()
        self.horizontal_correctors=tuple(horizontal_correctors or default_h); self.vertical_correctors=tuple(vertical_correctors or default_v)
        self.adapter = adapter or build_mock_adapter(self.devices,horizontal_correctors=self.horizontal_correctors,vertical_correctors=self.vertical_correctors)
        self.project = MeasureProject()
        self._selection_states = self.project.measurement_selections
        self._active_selection_kind = "bpm_noise"
        self._loading_selection_state = False
        self.project_path: Path | None = None
        self.selected_devices = self.devices
        self.selected_hcorrectors=self.horizontal_correctors; self.selected_vcorrectors=self.vertical_correctors
        self.result: BpmNoiseResult | None = None
        self.dispersion_states = []
        self.dispersion_step_index = 0
        self.dispersion_started_at: float | None = None
        self.saved_measurement_path: Path | None = None
        self.saved_session_path: Path | None = None
        self.cancel_event = Event(); self.thread: QThread | None = None; self.worker = None
        self.connection_verified = True
        self._build_ui(); self._measurement_type_changed(); self._update_machine_identity(); self.apply_theme(self.project.theme); self.refresh_preview(); self.refresh_plan()

    def _build_ui(self):
        toolbar = QToolBar("Measure toolbar"); toolbar.setObjectName("mainToolbar"); toolbar.setMovable(False); self.addToolBar(toolbar)
        self.brand_title = QLabel("pyLOCO  MEASURE")
        self.brand_title.setObjectName("measureBrand")
        self.brand_title.setAccessibleName("pyLOCO Measure")
        toolbar.addWidget(self.brand_title)
        toolbar.addSeparator()
        for text, handler in (("New", self.new_project), ("Open", self.open_project), ("Save Project", self.save_project), ("Save Project As…", self.save_project_as)):
            button=QPushButton(text); button.clicked.connect(handler); toolbar.addWidget(button)
        spacer=QWidget(); spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred); toolbar.addWidget(spacer)
        self.status_badge=QLabel("MOCK • READ ONLY"); self.status_badge.setObjectName("connectionStatus"); toolbar.addWidget(self.status_badge)
        self.machine_identity_badge=QLabel("Machine: Mock\nProfile: offline"); self.machine_identity_badge.setObjectName("machineIdentity"); toolbar.addWidget(self.machine_identity_badge)
        self.connection_badge=QLabel("● OFFLINE"); self.connection_badge.setObjectName("connectionState"); self.connection_badge.setProperty("connected",True); toolbar.addWidget(self.connection_badge)
        self.theme_button=QPushButton("☀ Light"); self.theme_button.clicked.connect(self.toggle_theme); toolbar.addWidget(self.theme_button)
        logo_container=QFrame(); logo_container.setObjectName("measureLogoContainer"); logo_layout=QHBoxLayout(logo_container); logo_layout.setContentsMargins(4,2,4,2)
        self.logo_button=ClickableLogoLabel(); set_asset(self.logo_button,QSize(107,44),DISPLAY_ASSET,crop_transparency=False); self.logo_button.clicked.connect(self._show_about_dialog); logo_layout.addWidget(self.logo_button); toolbar.addWidget(logo_container)

        root=QWidget(); layout=QVBoxLayout(root); layout.setContentsMargins(20,16,20,20)
        self.heading=QLabel("BPM Noise Measurement"); self.heading.setObjectName("pageTitle"); layout.addWidget(self.heading)
        self.subtitle=QLabel("Offline measurement planning and deterministic Mock acquisition — no machine writes are available."); layout.addWidget(self.subtitle)
        self.tabs=QTabWidget(); self.tabs.currentChanged.connect(lambda _: self.refresh_plan()); layout.addWidget(self.tabs,1)
        self.tabs.addTab(self._machine_page(),"Machine")
        self.tabs.addTab(self._bpms_page(),"Devices")
        configuration=self._measurement_page(); acquisition=self._review_page()
        measurement=QSplitter(Qt.Horizontal); self.measurement_splitter=measurement
        measurement.setChildrenCollapsible(False); measurement.setHandleWidth(12)
        measurement.setStyleSheet("QSplitter::handle { background:#18B8BE; margin:1px 3px; border-radius:3px; } QSplitter::handle:hover { background:#58D8DE; }")
        # Scroll areas otherwise advertise their content's wide size hint and make
        # one pane appear impossible to shrink.  The scroll contents still lay out
        # normally; only the splitter is allowed to negotiate compact pane widths.
        for pane in (configuration,acquisition):
            pane.setMinimumWidth(180); pane.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        measurement.addWidget(configuration); measurement.addWidget(acquisition)
        measurement.setStretchFactor(0,2); measurement.setStretchFactor(1,3); measurement.setSizes([440,760])
        self.tabs.addTab(measurement,"Measurement")
        saved=QWidget(); saved_layout=QVBoxLayout(saved); saved_layout.setContentsMargins(28,28,28,28); saved_title=QLabel("Saved measurement files"); saved_title.setObjectName("pageTitle"); self.review_machine_identity=QLabel(); self.review_machine_identity.setObjectName("planValue"); saved_note=QLabel("Acquisition, live plots, final plots and saving are now kept together on the Measurement tab. The saved filename is highlighted there after completion."); saved_note.setWordWrap(True); saved_layout.addWidget(saved_title); saved_layout.addWidget(self.review_machine_identity); saved_layout.addWidget(saved_note); saved_layout.addStretch()
        self.tabs.addTab(saved,"Review && Save")
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.setCentralWidget(root); self.statusBar().showMessage("Mock adapter ready — read-only acquisition"); self._update_workflow_tabs()

    def _scroll_page(self, content):
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft); scroll.setWidget(content); return scroll

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

    def _set_connection_state(self, connected: bool, text: str) -> None:
        if not hasattr(self,"connection_badge"):return
        self.connection_badge.setText(("● " if connected else "○ ")+text)
        self.connection_badge.setProperty("connected",bool(connected))
        self.connection_verified=bool(connected)
        self.connection_badge.style().unpolish(self.connection_badge); self.connection_badge.style().polish(self.connection_badge)
        if hasattr(self,"plan_values"):self.refresh_plan()

    def _set_acquisition_running(self, running: bool) -> None:
        # Keep the active Measurement workspace visible throughout acquisition.
        # Disabling its tab makes QTabWidget automatically navigate to Review & Save.
        for index in range(min(2,self.tabs.count())):self.tabs.setTabEnabled(index,not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.cancel_button.setText("Cancel acquisition" if running else "Cancel")
        self.repeat_button.setEnabled(not running)
        if running:self.statusBar().showMessage(f"ACQUIRING • {self.status_badge.text()}")

    def _machine_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.machine_group=QGroupBox("Control-system connection")
        self.machine_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        grid=QGridLayout(self.machine_group); grid.setContentsMargins(20,12,20,10); grid.setHorizontalSpacing(28); grid.setVerticalSpacing(2)
        grid.setColumnMinimumWidth(0,180); grid.setColumnStretch(0,0); grid.setColumnStretch(1,1)
        self.adapter_combo=QComboBox()
        for descriptor in InterfaceRegistry.DESCRIPTORS:
            self.adapter_combo.addItem(descriptor.label,descriptor.key)
        self.adapter_combo.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.pysc_profile_combo=QComboBox()
        for profile in available_pysc_profiles():
            self.pysc_profile_combo.addItem(profile.label,profile.key)
        self.pysc_profile_combo.setToolTip("Machine configuration served by the generic pySC Server backend")
        rows=(
            ("Control system",self.adapter_combo),
            ("Machine profile",self.pysc_profile_combo,None,"profile"),
            ("Adapter",QLabel("Mock — deterministic offline data source"),"adapter"),
            ("Connection",QLabel("Offline simulation"),"connection"),
            ("Access",QLabel("Read only — no real machine writes"),"access"),
            ("BPM orbit",QLabel(f"available — {len(self.devices)} mock BPM devices"),"bpm_orbit"),
            ("Correctors",QLabel("simulated inventory available"),"corrector_readback"),
            ("Calibration",QLabel("not applicable in Mock mode"),"calibration"),
            ("RF readback",QLabel("simulated internally"),"rf_readback"),
        )
        self.machine_value_labels=[]; self.machine_rows=[]; self.machine_info={}
        for row,item in enumerate(rows):
            name,value=item[:2]; key=item[2] if len(item)>2 else None
            label=QLabel(name); label.setMinimumWidth(180); label.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
            value.setMinimumHeight(36); value.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
            if isinstance(value,QLabel): value.setAlignment(Qt.AlignLeft|Qt.AlignVCenter); self.machine_value_labels.append(value)
            grid.addWidget(label,row,0); grid.addWidget(value,row,1)
            if not (len(item)>3 and item[3]=="profile"): grid.setRowMinimumHeight(row,36)
            if len(item)>3 and item[3]=="profile": self.pysc_profile_row=(label,value)
            else: self.machine_rows.append((label,value))
            if key:self.machine_info[key]=value
        self.test_connection_button=QPushButton("Test connection"); self.test_connection_button.setMinimumHeight(36); self.test_connection_button.clicked.connect(self._test_connection); grid.addWidget(self.test_connection_button,0,2)
        self.adapter_combo.currentIndexChanged.connect(self._adapter_changed)
        self.pysc_profile_combo.currentIndexChanged.connect(self._pysc_profile_changed)
        for widget in self.pysc_profile_row: widget.setVisible(False)
        layout.addWidget(self.machine_group); return self._scroll_page(content)

    def _bpms_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.devices_identity=QLabel("Machine inventory"); self.devices_identity.setObjectName("planValue"); self.devices_identity.setWordWrap(True); layout.addWidget(self.devices_identity)
        group=QGroupBox("BPMs — selection for the active measurement"); group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); form=QFormLayout(group); self._configure_form(form)
        self.selection_method=QComboBox(); self.selection_method.addItem("All available BPMs","all"); self.selection_method.addItem("Load names file","names_file"); self.selection_method.addItem("Manual names / positions","manual"); self.selection_method.currentIndexChanged.connect(self._selection_method_changed); form.addRow("Selection method",self.selection_method)
        self.names_row=QWidget(); nr=QHBoxLayout(self.names_row); nr.setContentsMargins(0,0,0,0); self.names_file=QLineEdit(); browse=QPushButton("Browse…"); browse.clicked.connect(self.browse_names); nr.addWidget(self.names_file,1); nr.addWidget(browse); form.addRow("BPM names file",self.names_row); self.names_label=form.labelForField(self.names_row)
        self.manual_input=QLineEdit(); self.manual_input.setPlaceholderText("BPM-001, BPM-004 or positions 0, 3"); form.addRow("Manual selection",self.manual_input); self.manual_label=form.labelForField(self.manual_input)
        self.bpm_exclusions=QLineEdit(); self.bpm_exclusions.setPlaceholderText("Selected-list positions to exclude, e.g. 1, 4"); form.addRow("Excluded BPM positions",self.bpm_exclusions); self.bpm_exclusions_label=form.labelForField(self.bpm_exclusions); self.bpm_exclusions.textChanged.connect(self.refresh_preview)
        self.bpm_search=QLineEdit(); self.bpm_search.setPlaceholderText("Filter preview by BPM name or identifier…"); self.bpm_search.textChanged.connect(self._filter_bpm_table); form.addRow("Search / filter",self.bpm_search)
        quick=QHBoxLayout(); select_all=QPushButton("Select All"); select_all.clicked.connect(self._select_all_bpms); clear=QPushButton("Clear"); clear.clicked.connect(self._clear_bpms); select_filtered=QPushButton("Select Filtered"); select_filtered.clicked.connect(self._select_filtered_bpms); highlighted=QPushButton("Use Highlighted"); highlighted.clicked.connect(lambda:self._select_highlighted("bpm")); preview=QPushButton("Refresh"); preview.clicked.connect(self.refresh_preview); quick.addWidget(select_all); quick.addWidget(clear); quick.addWidget(select_filtered); quick.addWidget(highlighted); quick.addStretch(); quick.addWidget(preview); form.addRow("",quick)
        subset=QHBoxLayout(); self.bpm_subset_n=NoWheelSpinBox(); self.bpm_subset_n.setRange(1,100000); self.bpm_subset_n.setValue(20); first_n=QPushButton("First N"); first_n.clicked.connect(lambda:self._select_device_subset("bpm","first",self.bpm_subset_n.value())); uniform_n=QPushButton("Uniform N"); uniform_n.clicked.connect(lambda:self._select_device_subset("bpm","uniform",self.bpm_subset_n.value())); subset.addWidget(self.bpm_subset_n); subset.addWidget(first_n); subset.addWidget(uniform_n); subset.addStretch(); form.addRow("Selection helpers",subset); layout.addWidget(group)
        table_group=QGroupBox("Selection preview"); table_group.setMinimumHeight(350); tl=QVBoxLayout(table_group); self.preview_table=QTableWidget(0,3); self.preview_table.setMinimumHeight(260); self.preview_table.setHorizontalHeaderLabels(["Selection position","BPM name","Adapter/device identifier"]); self.preview_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents); self.preview_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents); self.preview_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch); self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers); self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection); tl.addWidget(self.preview_table); self.selection_message=QLabel(); tl.addWidget(self.selection_message); layout.addWidget(table_group)
        self.orm_corrector_group=QGroupBox("Correctors — ORM selection"); cg=QVBoxLayout(self.orm_corrector_group); self.corrector_selection_widgets={}
        demo=QHBoxLayout(); one=QPushButton("Demo: 1 H + 1 V"); one.clicked.connect(self._select_demo_one_each); small=QPushButton("Demo: small uniform ORM"); small.clicked.connect(self._select_demo_small_orm); demo.addWidget(one); demo.addWidget(small); demo.addStretch(); cg.addLayout(demo)
        for key,title,devices in (("hcor","Horizontal correctors",self.horizontal_correctors),("vcor","Vertical correctors",self.vertical_correctors)):
            box=QGroupBox(title); form=QFormLayout(box); self._configure_form(form); method=QComboBox(); method.addItem("All available","all"); method.addItem("Load names file","names_file"); method.addItem("Manual names / positions","manual")
            manual=QLineEdit(); manual.setPlaceholderText("Names or selected-list positions, comma separated"); file_edit=QLineEdit(); browse=QPushButton("Browse…"); file_row=QWidget(); fr=QHBoxLayout(file_row); fr.setContentsMargins(0,0,0,0); fr.addWidget(file_edit,1); fr.addWidget(browse)
            exclusion=QLineEdit(); exclusion.setPlaceholderText("Selected-list positions to exclude, e.g. 1, 4")
            search=QLineEdit(); search.setPlaceholderText("Filter corrector names / identifiers…"); preview=QPushButton("Refresh"); table=QTableWidget(0,8); table.setMinimumHeight(150); table.setHorizontalHeaderLabels(["Selection position","Device name","Adapter/device identifier","Plane","KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV"]); table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)
            table.setSelectionBehavior(QAbstractItemView.SelectRows); table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            controls=QHBoxLayout(); all_button=QPushButton("Select All"); all_button.clicked.connect(lambda _=False,k=key:self._select_all_correctors(k)); clear_button=QPushButton("Clear"); clear_button.clicked.connect(lambda _=False,k=key:self._clear_correctors(k)); filtered=QPushButton("Select Filtered"); filtered.clicked.connect(lambda _=False,k=key:self._select_filtered_correctors(k)); highlighted=QPushButton("Use Highlighted"); highlighted.clicked.connect(lambda _=False,k=key:self._select_highlighted(k)); controls.addWidget(all_button); controls.addWidget(clear_button); controls.addWidget(filtered); controls.addWidget(highlighted); controls.addStretch(); controls.addWidget(preview)
            subset=QHBoxLayout(); nbox=NoWheelSpinBox(); nbox.setRange(1,100000); nbox.setValue(5); first_n=QPushButton("First N"); first_n.clicked.connect(lambda _=False,k=key,n=nbox:self._select_device_subset(k,"first",n.value())); uniform_n=QPushButton("Uniform N"); uniform_n.clicked.connect(lambda _=False,k=key,n=nbox:self._select_device_subset(k,"uniform",n.value())); subset.addWidget(nbox); subset.addWidget(first_n); subset.addWidget(uniform_n); subset.addStretch()
            count=QLabel(); count.setObjectName("planValue")
            form.addRow("Selection method",method); form.addRow("Search / filter",search); form.addRow("Names file",file_row); file_label=form.labelForField(file_row); form.addRow("Manual selection",manual); manual_label=form.labelForField(manual); form.addRow("Excluded positions",exclusion); form.addRow("",controls); form.addRow("Selection helpers",subset); form.addRow("Selected / available",count); form.addRow(table)
            data={"method":method,"manual":manual,"file":file_edit,"file_row":file_row,"file_label":file_label,"manual_label":manual_label,"exclusion":exclusion,"search":search,"table":table,"devices":devices,"count":count,"n":nbox}
            self.corrector_selection_widgets[key]=data
            method.currentIndexChanged.connect(lambda _=0,k=key:self._corrector_method_changed(k)); preview.clicked.connect(lambda _=False,k=key:self._refresh_corrector_preview(k)); search.textChanged.connect(lambda _="",k=key:self._filter_corrector_table(k)); browse.clicked.connect(lambda _=False,e=file_edit:self._browse_into(e)); exclusion.textChanged.connect(self.refresh_plan)
            cg.addWidget(box); self._corrector_method_changed(key); self._refresh_corrector_preview(key)
        layout.addWidget(self.orm_corrector_group)
        self._selection_method_changed(); return self._scroll_page(content)

    @staticmethod
    def _load_repository_names(filename):
        path=Path(__file__).resolve().parents[2]/"Examples"/"PETRAIII"/"data"/filename
        return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())

    def _adapter_changed(self,*_):
        key=self.adapter_combo.currentData(); petra=key=="petra"; pysc=key=="pysc"
        for widget in self.pysc_profile_row: widget.setVisible(pysc)
        if petra:
            bpm_names=self._load_repository_names("BPM_names.txt"); hnames=self._load_repository_names("HCM_names_control.txt"); vnames=self._load_repository_names("VCM_names_control.txt")
            self.adapter=PETRAReadOnlyAdapter(bpm_names,hnames,vnames)
            self.devices=tuple(BpmDevice(**item) for item in self.adapter.list_devices("bpm")); self.horizontal_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("hcor")); self.vertical_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("vcor"))
            self.status_badge.setText("LIVE • PETRA III DOOCS"); self._set_connection_state(False,"DISCONNECTED"); self.subtitle.setText("LIVE PETRA III selection — PETRA read-only safety access remains active until a verified write interface is connected."); self.machine_info["adapter"].setText("PETRA III DOOCS"); self.machine_info["connection"].setText("Not tested — use Test connection"); self.machine_info["access"].setText("READ ONLY safety adapter"); self.machine_info["bpm_orbit"].setText("not tested"); self.machine_info["corrector_readback"].setText("not tested"); self.machine_info["calibration"].setText("not tested"); self.machine_info["rf_readback"].setText("unavailable — no verified channel mapping")
        elif pysc:
            profile_key=self.pysc_profile_combo.currentData() or "ebs"
            profile_label=self.pysc_profile_combo.currentText()
            try:
                session=InterfaceRegistry(interface_loaders={},pysc_profile=profile_key).create("pysc")
            except Exception as exc:
                self.devices=(); self.horizontal_correctors=(); self.vertical_correctors=(); self.selected_devices=(); self.selected_hcorrectors=(); self.selected_vcorrectors=()
                self.status_badge.setText("DEMO • pySC SERVER"); self._set_connection_state(False,"DISCONNECTED"); self.subtitle.setText(f"pySC Server {profile_label} simulation selected — connect to start acquisition."); self.machine_info["adapter"].setText(f"pySC Server — {profile_label} simulation"); self.machine_info["connection"].setText(f"Unavailable: {exc}"); self.machine_info["access"].setText("DEMO — acquisition blocked while unavailable"); self.machine_info["bpm_orbit"].setText("unavailable"); self.machine_info["corrector_readback"].setText("unavailable"); self.machine_info["calibration"].setText("backend managed"); self.machine_info["rf_readback"].setText("unavailable");
                if hasattr(self,"start_button"):self.start_button.setEnabled(False)
                return
            self.adapter=session.adapter; self.devices=tuple(BpmDevice(**item) for item in self.adapter.list_devices("bpm")); self.horizontal_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("hcor")); self.vertical_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("vcor"))
            self.status_badge.setText(session.badge); self._set_connection_state(False,"DISCONNECTED"); self.subtitle.setText(f"DEMO MODE — pySC Server {profile_label} simulation. Temporary measurement writes are restoration-protected."); self.machine_info["adapter"].setText(f"pySC Server — {profile_label} simulation"); self.machine_info["connection"].setText("Configured — use Test connection"); self.machine_info["access"].setText("DEMO — temporary writes are restoration-protected"); self.machine_info["bpm_orbit"].setText("not tested"); self.machine_info["corrector_readback"].setText("not tested"); self.machine_info["calibration"].setText("backend managed"); self.machine_info["rf_readback"].setText("not tested")
        else:
            self.devices=default_mock_devices(); self.horizontal_correctors,self.vertical_correctors=default_mock_correctors(); self.adapter=build_mock_adapter(self.devices,horizontal_correctors=self.horizontal_correctors,vertical_correctors=self.vertical_correctors)
            self.status_badge.setText("MOCK • READ ONLY"); self._set_connection_state(True,"OFFLINE READY"); self.subtitle.setText("Offline measurement planning and deterministic Mock acquisition — no machine writes are available."); self.machine_info["adapter"].setText("Mock — deterministic offline data source"); self.machine_info["connection"].setText("Offline simulation"); self.machine_info["access"].setText("Read only — no real machine writes"); self.machine_info["bpm_orbit"].setText(f"available — {len(self.devices)} mock BPM devices"); self.machine_info["corrector_readback"].setText("simulated diagnostics available"); self.machine_info["calibration"].setText("not applicable in Mock mode"); self.machine_info["rf_readback"].setText("simulated internally")
        self.selected_devices=self.devices; self.selected_hcorrectors=self.horizontal_correctors; self.selected_vcorrectors=self.vertical_correctors
        for key,devices in (("hcor",self.horizontal_correctors),("vcor",self.vertical_correctors)):
            if hasattr(self,"corrector_selection_widgets"):self.corrector_selection_widgets[key]["devices"]=devices; self._refresh_corrector_preview(key,read_diagnostics=False)
        if hasattr(self,"measurement_type"):
            orm_item=self.measurement_type.model().item(self.measurement_type.findData("orm")); orm_item.setEnabled(not petra); self.orm_unavailable_label.setVisible(petra)
            if petra and self.measurement_type.currentData()=="orm":self.measurement_type.setCurrentIndex(self.measurement_type.findData("bpm_noise"))
            automatic=self.rf_control_mode.model().item(self.rf_control_mode.findData("automatic")); automatic.setEnabled(AdapterCapability.RF_WRITE in self.adapter.capabilities)
            if pysc: self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("automatic"))
            if pysc and (self.measurement_label.text().startswith("Mock") or " pySC " in self.measurement_label.text()):
                prefix=f"{profile_label} pySC"; labels={"bpm_noise":f"{prefix} BPM noise","dispersion":f"{prefix} dispersion","orm":f"{prefix} small ORM"}; self.measurement_label.setText(labels[self.measurement_type.currentData()])
            elif key=="mock" and " pySC " in self.measurement_label.text():
                labels={"bpm_noise":"Mock BPM noise","dispersion":"Mock manual-RF dispersion","orm":"Mock orbit response matrix"}; self.measurement_label.setText(labels[self.measurement_type.currentData()])
        self._reset_selection_states_for_inventory(); self._update_machine_identity()
        self.refresh_preview(); self.statusBar().showMessage("PETRA adapter selected — no connection attempted" if petra else "pySC Server configured — test connection before acquisition" if pysc else "Mock adapter ready — read-only acquisition"); self._sync_rf_presentation()

    def _machine_identity(self):
        key=self.adapter_combo.currentData() if hasattr(self,"adapter_combo") else "mock"
        if key=="pysc":
            profile=next((item for item in available_pysc_profiles() if item.key==self.pysc_profile_combo.currentData()),None)
            if profile:
                seed=profile.configuration.get("random_seed")
                text=f"{profile.label.split(' /')[0]} / {profile.scenario}"+(f"\nSeed {seed}" if seed is not None else "")
                return text,profile
        if key=="petra":return "PETRA III / LIVE read-only",None
        return "Mock / offline",None

    def _update_machine_identity(self):
        text,_=self._machine_identity(); self.machine_identity_badge.setText(text)
        if hasattr(self,"devices_identity"):
            self.devices_identity.setText(f"Machine/profile: {text.replace(chr(10),' — ')}\nAvailable inventory: {len(self.devices)} BPM / {len(self.horizontal_correctors)} H correctors / {len(self.vertical_correctors)} V correctors")
        if hasattr(self,"measurement_machine_identity"):self.measurement_machine_identity.setText(f"Measurement machine/profile: {text.replace(chr(10),' — ')}")
        if hasattr(self,"review_machine_identity"):self.review_machine_identity.setText(f"Saved data machine/profile: {text.replace(chr(10),' — ')}")

    def _pysc_profile_changed(self,*_):
        if self.adapter_combo.currentData()=="pysc": self._adapter_changed()

    def _sync_rf_presentation(self, *_):
        if not hasattr(self,"rf_control_mode"):return
        automatic=self.adapter_combo.currentData()=="pysc" and AdapterCapability.RF_WRITE in self.adapter.capabilities
        if automatic:
            self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("automatic")); self.rf_control_mode.setEnabled(False); self.nominal_rf.setReadOnly(True)
            self.verify_restored_orbit.setEnabled(True)
            self.nominal_rf_source.setText("Source: pySC Server RF readback — automatically controlled")
            self.rf_safety.setText("DEMO automatic RF control — the original frequency is restored and read back in cancellation-safe cleanup.")
        else:
            self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("manual")); self.rf_control_mode.setEnabled(True); self.nominal_rf.setReadOnly(False)
            self.verify_restored_orbit.setEnabled(False)
            self.nominal_rf_source.setText("Source: manual entry (no RF readback available)")
            self.rf_safety.setText("READ ONLY — pyLOCO Measure will guide RF changes but will never write RF.")
        bipolar=self.dispersion_direction.currentData()=="bipolar"
        self.rf_step_label.setText("Total bipolar RF separation" if bipolar else "RF offset Δf")
        self._configure_physical_dispersion_option()
        self.refresh_plan()

    def _test_connection(self):
        if self.adapter_combo.currentData()=="pysc":
            try:self._adapter_changed(); result=self.adapter.test_connection()
            except Exception as exc:self._set_connection_state(False,"DISCONNECTED"); QMessageBox.warning(self,"pySC Server unavailable",str(exc)); return
            profile=self.pysc_profile_combo.currentText(); self._set_connection_state(True,"CONNECTED"); self.machine_info["connection"].setText(f"Connected — {profile} demo-server reads succeeded"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(str(result["corrector_readback"])); self.machine_info["rf_readback"].setText(f"available — {result['rf_readback']:.6f} Hz"); self.nominal_rf.setText(f"{result['rf_readback']:.12f}"); self.statusBar().showMessage(f"CONNECTED • {self.status_badge.text()} • {profile} • {result['bpms']} BPMs"); QMessageBox.information(self,"pySC Server",f"Connected to {profile} DEMO profile; {result['bpms']} BPMs available."); return
        if not isinstance(self.adapter,PETRAReadOnlyAdapter):
            QMessageBox.information(self,"Mock connection","Deterministic Mock adapter is ready. No external system is contacted."); return
        self.test_connection_button.setEnabled(False); self.machine_info["connection"].setText("Testing safe reads…"); QApplication.processEvents()
        try:result=self.adapter.test_connection()
        except Exception as exc:
            self.machine_info["connection"].setText("Unavailable"); self.machine_info["bpm_orbit"].setText("unavailable"); self.machine_info["corrector_readback"].setText("unavailable"); QMessageBox.warning(self,"PETRA connection unavailable",str(exc))
        else:
            self.machine_info["connection"].setText("Connected — safe reads succeeded"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(result["corrector_readback"]); self.machine_info["calibration"].setText(result["calibration"]); self.machine_info["rf_readback"].setText(result["rf_readback"]); QMessageBox.information(self,"PETRA read-only connection",f"Read {result['bpms']} horizontal and vertical BPM values.\nCorrector readback: {result['corrector_readback']}\nCalibration: {result['calibration']}\nRF readback: {result['rf_readback']}\n\nNo writes were issued.")
        finally:self.test_connection_button.setEnabled(True)

    def _measurement_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.measurement_machine_identity=QLabel(); self.measurement_machine_identity.setObjectName("planValue"); self.measurement_machine_identity.setWordWrap(True); layout.addWidget(self.measurement_machine_identity)
        type_group=QGroupBox("Measurement type"); type_form=QFormLayout(type_group); self._configure_form(type_form); self.measurement_type=QComboBox(); self.measurement_type.addItem("BPM Noise","bpm_noise"); self.measurement_type.addItem("Dispersion","dispersion"); self.measurement_type.addItem("ORM","orm"); self.measurement_type.currentIndexChanged.connect(self._measurement_type_changed); type_form.addRow("Measurement",self.measurement_type); self.orm_unavailable_label=QLabel("ORM acquisition requires write-enabled control-system access. PETRA remains hard read-only in this milestone."); self.orm_unavailable_label.setWordWrap(True); type_form.addRow("",self.orm_unavailable_label); self.orm_unavailable_label.setVisible(False); layout.addWidget(type_group)
        self.measurement_help_group=QGroupBox("ⓘ About this measurement")
        self.measurement_help_group.setCheckable(True); self.measurement_help_group.setChecked(False)
        help_layout=QVBoxLayout(self.measurement_help_group); self.measurement_help_body=QWidget(); help_body_layout=QVBoxLayout(self.measurement_help_body); help_body_layout.setContentsMargins(2,4,2,4)
        self.measurement_help_title=QLabel(); self.measurement_help_title.setObjectName("measurementHelpTitle")
        self.measurement_help_text=QLabel(); self.measurement_help_text.setWordWrap(True); self.measurement_help_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.measurement_help_convention=QLabel(); self.measurement_help_convention.setObjectName("measurementConvention"); self.measurement_help_convention.setWordWrap(True); self.measurement_help_convention.setTextInteractionFlags(Qt.TextSelectableByMouse)
        help_body_layout.addWidget(self.measurement_help_title); help_body_layout.addWidget(self.measurement_help_text); help_body_layout.addWidget(self.measurement_help_convention)
        help_layout.addWidget(self.measurement_help_body); self.measurement_help_group.toggled.connect(self.measurement_help_body.setVisible); self.measurement_help_body.setVisible(False); layout.addWidget(self.measurement_help_group)
        group=QGroupBox("Measurement configuration"); group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); form=QFormLayout(group); self._configure_form(form)
        self.readings=NoWheelSpinBox(); self.readings.setRange(2,100000); self.readings.setValue(20)
        self.delay=NoWheelDoubleSpinBox(); self.delay.setRange(0,3600); self.delay.setDecimals(3); self.delay.setSingleStep(.1); self.delay.setSuffix(" s"); self.delay.setValue(.1)
        self.measurement_name=QLineEdit("bpm-noise"); self.measurement_label=QLineEdit("Mock BPM noise"); self.comments=QPlainTextEdit(); self.comments.setMaximumHeight(90)
        self.output_directory=QLineEdit("measurements"); out=QWidget(); ol=QHBoxLayout(out); ol.setContentsMargins(0,0,0,0); ol.addWidget(self.output_directory,1); choose=QPushButton("Choose…"); choose.clicked.connect(self.choose_output); ol.addWidget(choose)
        self.duration=QLabel();
        for label,widget in (("Number of orbit readings",self.readings),("Delay between readings",self.delay),("Estimated acquisition duration",self.duration),("Measurement name",self.measurement_name),("Measurement label",self.measurement_label),("Operator comments",self.comments),("Output/session directory",out)): form.addRow(label,widget)
        self.readings.valueChanged.connect(self.refresh_plan); self.delay.valueChanged.connect(self.refresh_plan)
        for widget in (self.measurement_name,self.measurement_label,self.output_directory): widget.textChanged.connect(self.refresh_plan)
        layout.addWidget(group)
        self.dispersion_config_group=QGroupBox("Manual RF dispersion configuration"); dispersion_form=QFormLayout(self.dispersion_config_group); self._configure_form(dispersion_form)
        self.rf_control_mode=QComboBox(); self.rf_control_mode.addItem("Manual RF change","manual"); self.rf_control_mode.addItem("Automatic (adapter RF-write capability required)","automatic"); self.rf_control_mode.model().item(1).setEnabled(AdapterCapability.RF_WRITE in self.adapter.capabilities)
        self.nominal_rf=QLineEdit(); self.nominal_rf.setPlaceholderText("Required — enter nominal RF frequency in Hz"); self.nominal_rf_source=QLabel("Source: manual entry (no RF readback available)")
        nominal_widget=QWidget(); nominal_layout=QVBoxLayout(nominal_widget); nominal_layout.setContentsMargins(0,0,0,0); nominal_layout.addWidget(self.nominal_rf); nominal_layout.addWidget(self.nominal_rf_source)
        self.rf_step=NoWheelDoubleSpinBox(); self.rf_step.setRange(1,1e9); self.rf_step.setDecimals(0); self.rf_step.setValue(200.0); self.rf_step.setSuffix(" Hz")
        self.dispersion_direction=QComboBox(); self.dispersion_direction.addItem("Bipolar ±Δf","bipolar"); self.dispersion_direction.addItem("One-sided +Δf","positive"); self.dispersion_direction.addItem("One-sided −Δf","negative")
        self.settling_delay=NoWheelDoubleSpinBox(); self.settling_delay.setRange(0,3600); self.settling_delay.setDecimals(3); self.settling_delay.setSuffix(" s")
        self.verify_restored_orbit=QCheckBox("Verify restored orbit"); self.verify_restored_orbit.setChecked(True)
        self.verify_restored_orbit.setToolTip("After exact RF restoration, acquire a read-only reference orbit for comparison. This orbit is never used in the dispersion calculation.")
        self.rf_safety=QLabel("READ ONLY — pyLOCO Measure will guide RF changes but will never write RF."); self.rf_safety.setWordWrap(True)
        for label,widget in (("RF control mode",self.rf_control_mode),("Nominal RF frequency",nominal_widget),("Total bipolar RF separation",self.rf_step),("Measurement direction",self.dispersion_direction),("Settling delay before acquisition",self.settling_delay),("Restoration diagnostic",self.verify_restored_orbit),("",self.rf_safety)): dispersion_form.addRow(label,widget)
        self.rf_step_label=dispersion_form.labelForField(self.rf_step)
        for widget in (self.nominal_rf,self.rf_step,self.dispersion_direction,self.settling_delay):
            signal=widget.textChanged if isinstance(widget,QLineEdit) else widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.valueChanged
            signal.connect(self.refresh_plan)
        self.dispersion_direction.currentIndexChanged.connect(self._update_measurement_help)
        self.dispersion_direction.currentIndexChanged.connect(self._sync_rf_presentation)
        self.verify_restored_orbit.toggled.connect(self.refresh_plan)
        layout.addWidget(self.dispersion_config_group)
        self.orm_config_group=QGroupBox("ORM corrector perturbation configuration"); orm_form=QFormLayout(self.orm_config_group); self._configure_form(orm_form)
        self.orm_direction=QComboBox(); self.orm_direction.addItem("Bipolar","bipolar"); self.orm_direction.addItem("Positive one-sided","positive"); self.orm_direction.addItem("Negative one-sided","negative")
        self.orm_kick_mode=QComboBox(); self.orm_kick_mode.addItem("Common kick","common"); self.orm_kick_mode.addItem("Per-corrector kicks from file","file")
        self.orm_hkick=NoWheelDoubleSpinBox(); self.orm_hkick.setRange(.001,1e6); self.orm_hkick.setDecimals(3); self.orm_hkick.setValue(100); self.orm_hkick.setSuffix(" µrad")
        self.orm_vkick=NoWheelDoubleSpinBox(); self.orm_vkick.setRange(.001,1e6); self.orm_vkick.setDecimals(3); self.orm_vkick.setValue(100); self.orm_vkick.setSuffix(" µrad")
        self.orm_kick_file=QLineEdit(); kick_browse=QPushButton("Browse…"); kick_browse.clicked.connect(lambda:self._browse_into(self.orm_kick_file)); self.orm_kick_file_row=QWidget(); kfl=QHBoxLayout(self.orm_kick_file_row); kfl.setContentsMargins(0,0,0,0); kfl.addWidget(self.orm_kick_file,1); kfl.addWidget(kick_browse)
        self.orm_scaled=QCheckBox("Store scaled ORM in metres/radian (otherwise store orbit differences in metres)")
        orm_form.addRow("Measurement direction",self.orm_direction); orm_form.addRow("Corrector kick mode",self.orm_kick_mode); orm_form.addRow("Horizontal kick",self.orm_hkick); self.orm_hkick_label=orm_form.labelForField(self.orm_hkick); orm_form.addRow("Vertical kick",self.orm_vkick); self.orm_vkick_label=orm_form.labelForField(self.orm_vkick); orm_form.addRow("Kick arrays file",self.orm_kick_file_row); self.orm_kick_file_label=orm_form.labelForField(self.orm_kick_file_row); orm_form.addRow("Scaling",self.orm_scaled)
        self.kick_preview=QTableWidget(0,5); self.kick_preview.setMinimumHeight(180); self.kick_preview.setHorizontalHeaderLabels(["Position","Plane","Corrector","Effective kick [µrad]","Source"]); self.kick_preview.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch); orm_form.addRow("Effective kicks",self.kick_preview)
        for widget in (self.orm_direction,self.orm_kick_mode,self.orm_hkick,self.orm_vkick,self.orm_kick_file,self.orm_scaled):
            signal=widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.valueChanged if isinstance(widget,QDoubleSpinBox) else widget.textChanged if isinstance(widget,QLineEdit) else widget.toggled
            signal.connect(self._orm_config_changed)
        layout.addWidget(self.orm_config_group)
        self._measurement_type_changed(); return self._scroll_page(content)

    def _review_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        plan_group=QGroupBox("Measurement plan"); plan_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); plan=QGridLayout(plan_group); plan.setHorizontalSpacing(18); plan.setVerticalSpacing(9); self.plan_values={}; self.dispersion_plan_widgets=[]; self.orm_plan_widgets=[]; self.settling_plan_widgets=[]
        plan_pairs=(("Measurement","BPMs"),("H correctors","V correctors"),("Total correctors","Kick source"),("H kick","V kick"),("Scaled ORM","ORM direction"),("RF mode","Nominal RF"),("RF step ±Δf","Bipolar separation"),("Negative RF","Positive RF"),("Direction","RF states"),("Readings","Delay"),("Settling delay","Est. duration"),("Adapter","Output"))
        for row,(left,right) in enumerate(plan_pairs):
            for pair,label in enumerate((left,right)):
                column=pair*2; key=label.lower().replace(".","").replace(" ","_"); name=QLabel(label); value=QLabel("—"); value.setObjectName("planValue"); self.plan_values[key]=value; plan.addWidget(name,row,column); plan.addWidget(value,row,column+1); plan.setColumnStretch(column+1,1)
                if label in {"RF mode","Nominal RF","RF step ±Δf","Bipolar separation","Negative RF","Positive RF","Direction","RF states"}: self.dispersion_plan_widgets.extend((name,value))
                if label in {"H correctors","V correctors","Total correctors","Kick source","H kick","V kick","Scaled ORM","ORM direction"}: self.orm_plan_widgets.extend((name,value))
                if label=="Settling delay":self.settling_plan_widgets.extend((name,value))
        self.plan_output=self.plan_values["output"]
        run_group=QGroupBox("Acquisition status"); self.run_group=run_group; rl=QVBoxLayout(run_group); self.step_instruction=QLabel(); self.step_instruction.setWordWrap(True); self.step_instruction.setObjectName("planValue"); rl.addWidget(self.step_instruction); self.reading_status=QLabel("Ready"); self.reading_status.setObjectName("runState"); self.progress=QProgressBar(); self.progress.setMinimumHeight(34); self.progress.setFormat("Ready — 0 / 0"); rl.addWidget(self.reading_status); rl.addWidget(self.progress)
        status_row=QHBoxLayout(); self.elapsed=QLabel("Elapsed: 0.00 s"); self.remaining=QLabel("Remaining: —"); self.samples=QLabel("Samples: 0 / 0")
        for metric in (self.elapsed,self.remaining,self.samples): metric.setObjectName("runMetric"); status_row.addWidget(metric)
        status_row.addStretch(1); rl.addLayout(status_row)
        self.live_plot=PlotCanvas(show_toolbar=False,minimum_height=230); self.live_plot.save_button.setVisible(False); rl.addWidget(self.live_plot)
        buttons=QHBoxLayout(); self.start_button=QPushButton("Start BPM-noise measurement"); self.start_button.setObjectName("measurePrimary"); self.start_button.clicked.connect(self.start_acquisition); self.repeat_button=QPushButton("Repeat measurement"); self.repeat_button.setObjectName("measurePrimary"); self.repeat_button.clicked.connect(self.repeat_measurement); self.repeat_button.setVisible(False); self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_acquisition); self.preview_toggle=QPushButton("Show acquisition preview"); self.preview_toggle.setCheckable(True); self.preview_toggle.setVisible(False); self.preview_toggle.toggled.connect(self._toggle_completed_preview); buttons.addWidget(self.start_button); buttons.addWidget(self.repeat_button); buttons.addWidget(self.cancel_button); buttons.addWidget(self.preview_toggle); buttons.addStretch(1); rl.addLayout(buttons); self.start_block_reason=QLabel(); self.start_block_reason.setWordWrap(True); self.start_block_reason.setStyleSheet("color:#F0A35E;font-weight:700"); rl.addWidget(self.start_block_reason); layout.addWidget(run_group); self.plan_group=plan_group; layout.addWidget(plan_group)
        self.log_group=QGroupBox("Acquisition Log"); self.log_group.setCheckable(True); self.log_group.setChecked(False); log_layout=QVBoxLayout(self.log_group); self.log_body=QWidget(); body_layout=QVBoxLayout(self.log_body); body_layout.setContentsMargins(0,0,0,0); self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(110); log_actions=QHBoxLayout(); clear_log=QPushButton("Clear"); clear_log.clicked.connect(self.log.clear); save_log=QPushButton("Save log…"); save_log.clicked.connect(self.save_log); log_actions.addWidget(clear_log); log_actions.addWidget(save_log); log_actions.addStretch(1); body_layout.addLayout(log_actions); body_layout.addWidget(self.log); log_layout.addWidget(self.log_body); self.log_group.toggled.connect(self.log_body.setVisible); self.log_body.setVisible(False); layout.addWidget(self.log_group)
        self.orm_column_row=QWidget(); ocr=QHBoxLayout(self.orm_column_row); ocr.setContentsMargins(0,0,0,0); ocr.addWidget(QLabel("Selected ORM column")); self.orm_column_selector=QComboBox(); self.orm_column_selector.currentIndexChanged.connect(self._update_selected_orm_column); ocr.addWidget(self.orm_column_selector,1); layout.addWidget(self.orm_column_row); self.orm_column_row.setVisible(False)
        self.dispersion_display_row=QWidget(); ddr=QHBoxLayout(self.dispersion_display_row); ddr.setContentsMargins(0,0,0,0); ddr.addWidget(QLabel("Dispersion display")); self.dispersion_display=QComboBox(); self.dispersion_display.addItem("RF orbit difference (pyLOCO-compatible)","raw"); self.dispersion_display.addItem("Physical dispersion","physical"); self.dispersion_display.currentIndexChanged.connect(self._dispersion_display_changed); ddr.addWidget(self.dispersion_display,1); self.dispersion_display_reason=QLabel(); self.dispersion_display_reason.setWordWrap(True); ddr.addWidget(self.dispersion_display_reason,2); layout.addWidget(self.dispersion_display_row); self.dispersion_display_row.setVisible(False)
        self.results_tabs=QTabWidget(); self.results_tabs.setMinimumHeight(430); self.results_tabs.tabBar().setExpanding(False); self.results_tabs.tabBar().setElideMode(Qt.ElideNone)
        self.x_plot=PlotCanvas(minimum_height=310); self.y_plot=PlotCanvas(minimum_height=310)
        self.raw_x_plot=PlotCanvas(minimum_height=310); self.raw_y_plot=PlotCanvas(minimum_height=310)
        self.mean_x_plot=PlotCanvas(minimum_height=310); self.mean_y_plot=PlotCanvas(minimum_height=310)
        self.rf_shift_x_plot=PlotCanvas(minimum_height=310); self.rf_shift_y_plot=PlotCanvas(minimum_height=310)
        self.orm_vv_plot=PlotCanvas(minimum_height=310); self.orm_column_plot=PlotCanvas(minimum_height=310); self.orm_kick_plot=PlotCanvas(minimum_height=310)
        self.results_tabs.addTab(self.x_plot,"Horizontal BPM noise"); self.results_tabs.addTab(self.y_plot,"Vertical BPM noise"); self.results_tabs.addTab(self.mean_x_plot,"Mean horizontal orbit"); self.results_tabs.addTab(self.mean_y_plot,"Mean vertical orbit"); layout.addWidget(self.results_tabs)
        self.stats_group=QGroupBox("Statistics"); stats_layout=QVBoxLayout(self.stats_group); self.orm_measurement_summary=QLabel(); self.orm_measurement_summary.setWordWrap(True); self.orm_restoration_status=QLabel(); self.orm_restoration_status.setObjectName("runState"); self.orm_restoration_status.setWordWrap(True); self.summary_x=QLabel("Horizontal: no completed measurement."); self.summary_y=QLabel("Vertical: no completed measurement."); self.rf_diagnostics=QLabel(); self.restoration_label=QLabel(); self.summary_x.setWordWrap(True); self.summary_y.setWordWrap(True); self.rf_diagnostics.setWordWrap(True); self.restoration_label.setWordWrap(True); stats_layout.addWidget(self.orm_measurement_summary); stats_layout.addWidget(self.orm_restoration_status); stats_layout.addWidget(self.summary_x); stats_layout.addWidget(self.summary_y); stats_layout.addWidget(self.rf_diagnostics); stats_layout.addWidget(self.restoration_label); layout.addWidget(self.stats_group)
        self.dispersion_summary=QWidget(); dispersion_summary_layout=QVBoxLayout(self.dispersion_summary); dispersion_summary_layout.setContentsMargins(0,0,0,0)
        rf_response_group=QGroupBox("1. RF orbit difference — pyLOCO-compatible"); rf_response_layout=QVBoxLayout(rf_response_group); self.rf_response_formula=QLabel("Δx_RF = mean[x(f−)] − mean[x(f+)]\nΔy_RF = mean[y(f−)] − mean[y(f+)]"); self.rf_response_formula.setWordWrap(True); self.rf_response_note=QLabel("Stored in metres; displayed in mm; this is the RF-response column supplied to pyLOCO."); self.rf_response_note.setWordWrap(True); self.rf_response_stats=QLabel(); self.rf_response_stats.setWordWrap(True); rf_response_layout.addWidget(self.rf_response_formula); rf_response_layout.addWidget(self.rf_response_note); rf_response_layout.addWidget(self.rf_response_stats); dispersion_summary_layout.addWidget(rf_response_group)
        physical_group=QGroupBox("2. Physical dispersion"); physical_layout=QVBoxLayout(physical_group); self.physical_formula=QLabel("η = αc − 1/γ²\nδ(f) ≈ −(f−f₀)/(η f₀)\nΔδ = δ− − δ+\nDₓ = Δx_RF / Δδ\nDᵧ = Δy_RF / Δδ"); self.physical_formula.setWordWrap(True); self.physical_stats=QLabel(); self.physical_stats.setWordWrap(True); physical_layout.addWidget(self.physical_formula); physical_layout.addWidget(self.physical_stats)
        self.calculation_details_group=QGroupBox("Show calculation details"); self.calculation_details_group.setCheckable(True); self.calculation_details_group.setChecked(False); calculation_layout=QVBoxLayout(self.calculation_details_group); self.calculation_details_body=QLabel(); self.calculation_details_body.setWordWrap(True); self.calculation_details_body.setTextInteractionFlags(Qt.TextSelectableByMouse); calculation_layout.addWidget(self.calculation_details_body); self.calculation_details_group.toggled.connect(self.calculation_details_body.setVisible); self.calculation_details_body.setVisible(False); physical_layout.addWidget(self.calculation_details_group); dispersion_summary_layout.addWidget(physical_group)
        restoration_group=QGroupBox("3. Restoration verification"); restoration_layout=QVBoxLayout(restoration_group); self.restoration_status=QLabel("RF restoration: —"); self.restoration_status.setObjectName("runState"); self.restoration_status.setWordWrap(True); self.restoration_values=QLabel(); self.restoration_values.setWordWrap(True); self.restoration_diagnostic=QLabel(); self.restoration_diagnostic.setWordWrap(True); restoration_layout.addWidget(self.restoration_status); restoration_layout.addWidget(self.restoration_values); restoration_layout.addWidget(self.restoration_diagnostic); dispersion_summary_layout.addWidget(restoration_group)
        self.dispersion_summary.setVisible(False); layout.insertWidget(layout.indexOf(self.stats_group),self.dispersion_summary)
        save_group=QGroupBox("Saved files"); sl=QVBoxLayout(save_group); self.paths=QLabel("Measurement file: —\nSession manifest: —"); self.paths.setTextInteractionFlags(Qt.TextSelectableByMouse); actions=QHBoxLayout(); self.validate_button=QPushButton("Validate for pyLOCO"); self.validate_button.setEnabled(False); self.validate_button.clicked.connect(self.validate_saved); self.open_button=QPushButton("Open session in pyLOCO"); self.open_button.setEnabled(False); self.open_button.clicked.connect(self.explain_open); actions.addWidget(self.validate_button); actions.addWidget(self.open_button); actions.addStretch(1); sl.addWidget(self.paths); sl.addLayout(actions); layout.addWidget(save_group)
        self._reset_live_plot()
        self._update_diagnostics_visibility()
        self.acquisition_scroll=self._scroll_page(content)
        self.acquisition_scroll.setObjectName("acquisitionWorkspaceScroll")
        return self.acquisition_scroll

    def _toggle_completed_preview(self, visible):
        self.live_plot.setVisible(bool(visible))
        self.preview_toggle.setText("Hide acquisition preview" if visible else "Show acquisition preview")

    def _set_completed_result_view(self, completed):
        """Prioritize scientific result plots once acquisition has completed."""
        completed=bool(completed)
        self.plan_group.setVisible(not completed)
        self.preview_toggle.blockSignals(True)
        self.preview_toggle.setChecked(False)
        self.preview_toggle.blockSignals(False)
        self.preview_toggle.setText("Show acquisition preview")
        self.preview_toggle.setVisible(completed)
        self.live_plot.setVisible(not completed)
        if completed:
            self.acquisition_scroll.verticalScrollBar().setValue(0)

    def _selection_method_changed(self):
        method=self.selection_method.currentData(); names_visible=method=="names_file"; manual_visible=method=="manual"
        self.names_row.setVisible(names_visible); self.names_label.setVisible(names_visible)
        self.manual_input.setVisible(manual_visible); self.manual_label.setVisible(manual_visible)

    def _filter_bpm_table(self, text=""):
        if not hasattr(self,"preview_table"):return
        query=str(text).strip().lower()
        for row in range(self.preview_table.rowCount()):
            hay=" ".join(self.preview_table.item(row,col).text() for col in (1,2) if self.preview_table.item(row,col))
            self.preview_table.setRowHidden(row,bool(query and query not in hay.lower()))

    def _select_all_bpms(self):
        self.selection_method.setCurrentIndex(self.selection_method.findData("all")); self.refresh_preview()

    def _clear_bpms(self):
        self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.clear(); self.refresh_preview()

    def _select_filtered_bpms(self):
        query=self.bpm_search.text().strip().lower()
        matches=[device.name for device in self.devices if not query or query in (device.name+" "+device.identifier).lower()]
        self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.setText(", ".join(matches)); self.refresh_preview()

    @staticmethod
    def _uniform_devices(devices, count):
        count=min(max(0,int(count)),len(devices))
        if count==0:return ()
        if count==len(devices):return tuple(devices)
        return tuple(devices[index] for index in np.linspace(0,len(devices)-1,count,dtype=int))

    def _select_device_subset(self,key,mode,count):
        if key=="bpm":
            devices=self.devices; method=self.selection_method; manual=self.manual_input
        else:
            data=self.corrector_selection_widgets[key]; devices=tuple(data["devices"]); method=data["method"]; manual=data["manual"]
        selected=tuple(devices[:min(count,len(devices))]) if mode=="first" else self._uniform_devices(devices,count)
        method.setCurrentIndex(method.findData("manual")); manual.setText(", ".join(device.name for device in selected))
        self.refresh_preview() if key=="bpm" else self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_all_correctors(self,key):
        data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(data["method"].findData("all")); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _clear_correctors(self,key):
        data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].clear(); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_filtered_correctors(self,key):
        data=self.corrector_selection_widgets[key]; query=data["search"].text().strip().lower(); matches=[device.name for device in data["devices"] if not query or query in (device.name+" "+device.identifier).lower()]
        data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(matches)); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_highlighted(self,key):
        table=self.preview_table if key=="bpm" else self.corrector_selection_widgets[key]["table"]
        names=[table.item(index.row(),1).text() for index in table.selectionModel().selectedRows() if table.item(index.row(),1)]
        if key=="bpm":
            self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.setText(", ".join(names)); self.refresh_preview()
        else:
            data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(names)); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_demo_one_each(self):
        self._select_device_subset("hcor","uniform",1); self._select_device_subset("vcor","uniform",1)

    def _select_demo_small_orm(self):
        self._select_device_subset("bpm","uniform",20); self._select_device_subset("hcor","uniform",5); self._select_device_subset("vcor","uniform",5)

    def _select_first_corrector(self,key):
        data=self.corrector_selection_widgets[key]
        data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText("0"); self._refresh_corrector_preview(key)

    def _filter_corrector_table(self,key):
        data=self.corrector_selection_widgets[key]; query=data["search"].text().strip().lower(); table=data["table"]
        for row in range(table.rowCount()):
            hay=" ".join(table.item(row,col).text() for col in (1,2) if table.item(row,col))
            table.setRowHidden(row,bool(query and query not in hay.lower()))

    def _browse_into(self,edit):
        path=QFileDialog.getOpenFileName(self,"Select names/kick file","","Data files (*.txt *.csv *.json *.npy *.npz);;All files (*)")[0]
        if path: edit.setText(path)

    @staticmethod
    def _positions(text):
        if not text.strip(): return set()
        try:return {int(token.strip()) for token in re.split(r"[,;\s]+",text) if token.strip()}
        except ValueError as exc: raise ValueError("Exclusions must be integer selected-list positions") from exc

    def _resolve_correctors(self,key):
        data=self.corrector_selection_widgets[key]; devices=tuple(data["devices"]); method=data["method"].currentData(); lookup={d.name:d for d in devices}
        if method=="all": selected=list(devices)
        else:
            tokens=[line.strip() for line in self._resolved_resource_path(data["file"].text()).read_text().splitlines() if line.strip()] if method=="names_file" else [t.strip() for t in re.split(r"[,;\n]+",data["manual"].text()) if t.strip()]
            selected=[]; invalid=[]
            for token in tokens:
                if token in lookup:selected.append(lookup[token]); continue
                try:index=int(token)
                except ValueError:invalid.append(token); continue
                if 0<=index<len(devices):selected.append(devices[index])
                else:invalid.append(token)
            if invalid: raise ValueError("Unknown corrector names/positions: "+", ".join(invalid))
        excluded=self._positions(data["exclusion"].text()); invalid=sorted(pos for pos in excluded if pos<0 or pos>=len(selected))
        if invalid: raise ValueError("Excluded positions outside selection: "+", ".join(map(str,invalid)))
        return tuple(device for pos,device in enumerate(selected) if pos not in excluded)

    def _corrector_method_changed(self,key):
        data=self.corrector_selection_widgets[key]; method=data["method"].currentData(); data["file_row"].setVisible(method=="names_file"); data["file_label"].setVisible(method=="names_file"); data["manual"].setVisible(method=="manual"); data["manual_label"].setVisible(method=="manual")

    def _refresh_corrector_preview(self,key,read_diagnostics=True):
        data=self.corrector_selection_widgets[key]
        try:selected=self._resolve_correctors(key)
        except Exception as exc:data["table"].setRowCount(0); data["table"].setToolTip(str(exc)); return
        if key=="hcor":self.selected_hcorrectors=selected
        else:self.selected_vcorrectors=selected
        table=data["table"]; table.setRowCount(len(selected)); table.setToolTip(f"{len(selected)} retained")
        for row,device in enumerate(selected):
            diagnostics={}
            if read_diagnostics and isinstance(self.adapter,PETRAReadOnlyAdapter):
                try:diagnostics=self.adapter.read_corrector_diagnostics(device.name)
                except Exception as exc:table.setToolTip(f"Source: adapter discovery/readback unavailable — {exc}")
            values=(row,device.name,device.identifier,device.plane,*(f"{diagnostics.get(field):.8g}" if field in diagnostics else "—" for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")))
            for col,value in enumerate(values):table.setItem(row,col,QTableWidgetItem(str(value)))
        self._filter_corrector_table(key)
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}[data["method"].currentData()]; table.setToolTip((table.toolTip()+"\n" if table.toolTip() else "")+f"Source: {source}; {len(selected)} retained")
        data["count"].setText(f"{len(selected)} / {len(data['devices'])}")
        if hasattr(self,"kick_preview"):self._update_kick_preview()
        self.refresh_plan()

    def browse_names(self):
        path=QFileDialog.getOpenFileName(self,"Select BPM names file","","Text files (*.txt);;All files (*)")[0]
        if path: self.names_file.setText(path)

    def choose_output(self):
        path=QFileDialog.getExistingDirectory(self,"Select output directory")
        if path: self.output_directory.setText(path)

    def _selection_snapshot(self):
        def values(method,manual,names_file,excluded):
            return {"method":method.currentData(),"manual":manual.text(),"names_file":names_file.text(),"excluded_positions":excluded.text()}
        result={"bpm":values(self.selection_method,self.manual_input,self.names_file,self.bpm_exclusions)}
        for key in ("hcor","vcor"):
            data=self.corrector_selection_widgets[key]
            result[key]=values(data["method"],data["manual"],data["file"],data["exclusion"])
        return result

    def _store_active_selection(self):
        if not self._loading_selection_state and hasattr(self,"selection_method"):
            self._selection_states[self._active_selection_kind]=self._selection_snapshot()

    def _restore_selection(self,kind):
        state=self._selection_states.get(kind) or MeasureProject().measurement_selections[kind]
        self._loading_selection_state=True
        try:
            bpm=state["bpm"]; self.selection_method.setCurrentIndex(max(0,self.selection_method.findData(bpm["method"]))); self.manual_input.setText(bpm.get("manual","")); self.names_file.setText(bpm.get("names_file","")); self.bpm_exclusions.setText(bpm.get("excluded_positions",""))
            for key in ("hcor","vcor"):
                settings=state[key]; data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(max(0,data["method"].findData(settings["method"]))); data["manual"].setText(settings.get("manual","")); data["file"].setText(settings.get("names_file","")); data["exclusion"].setText(settings.get("excluded_positions","")); self._corrector_method_changed(key); self._refresh_corrector_preview(key,read_diagnostics=False)
            self._selection_method_changed(); self.refresh_preview()
        finally:self._loading_selection_state=False

    def _reset_selection_states_for_inventory(self):
        defaults=MeasureProject().measurement_selections
        self._selection_states={kind:{device:dict(settings) for device,settings in selection.items()} for kind,selection in defaults.items()}
        self._active_selection_kind=self.measurement_type.currentData() if hasattr(self,"measurement_type") else "bpm_noise"
        if hasattr(self,"selection_method"):self._restore_selection(self._active_selection_kind)

    def _measurement_type_changed(self, *_):
        if not hasattr(self,"measurement_type"): return
        kind=self.measurement_type.currentData()
        if kind!=self._active_selection_kind:
            self._store_active_selection(); self._active_selection_kind=kind; self._restore_selection(kind)
        dispersion=kind=="dispersion"; orm=kind=="orm"
        if orm and isinstance(self.adapter,MockAdapter) and hasattr(self.adapter,"set_simulated_writes_enabled"):
            self.adapter.set_simulated_writes_enabled(True)
        self.dispersion_config_group.setVisible(dispersion)
        self.orm_config_group.setVisible(orm)
        if hasattr(self,"orm_corrector_group"): self.orm_corrector_group.setEnabled(orm)
        if hasattr(self,"bpm_exclusions"): self.bpm_exclusions.setVisible(orm); self.bpm_exclusions_label.setVisible(orm)
        self._update_measurement_help()
        title={"dispersion":"Dispersion Measurement","orm":"Orbit Response Matrix Measurement"}.get(kind,"BPM Noise Measurement")
        self.heading.setText(title); self.setWindowTitle("pyLOCO Measure — "+({"dispersion":"Dispersion","orm":"ORM"}.get(kind,"BPM Noise")))
        for widget in getattr(self,"dispersion_plan_widgets",[]): widget.setVisible(dispersion)
        if hasattr(self,"results_tabs"):
            self._configure_result_tabs(kind)
            self.dispersion_display_row.setVisible(dispersion)
            self._configure_physical_dispersion_option()
        self._update_diagnostics_visibility()
        self.dispersion_states=[]; self.dispersion_step_index=0; self.result=None
        self.saved_measurement_path=None; self.saved_session_path=None
        if hasattr(self,"repeat_button"):self.repeat_button.setVisible(False)
        if hasattr(self,"repeat_button"):self.repeat_button.setText("Repeat ORM measurement" if orm else "Repeat measurement")
        if hasattr(self,"start_button"):self.start_button.setVisible(True)
        if hasattr(self,"validate_button"):
            self.validate_button.setEnabled(False); self.open_button.setEnabled(False)
        if hasattr(self,"live_plot"): self._reset_live_plot()
        if hasattr(self,"plan_values"): self.refresh_plan()
        backend=self.adapter_combo.currentData() if hasattr(self,"adapter_combo") else "mock"; prefix=f"{self.pysc_profile_combo.currentText()} pySC" if backend=="pysc" else "Mock"
        if dispersion and self.measurement_name.text() in {"bpm-noise","mock-dispersion","pysc-dispersion"}:
            self.measurement_name.setText("pysc-dispersion" if backend=="pysc" else "mock-dispersion"); self.measurement_label.setText(f"{prefix} dispersion" if backend=="pysc" else "Mock manual-RF dispersion")
        elif not dispersion and not orm and self.measurement_name.text() in {"mock-dispersion","pysc-dispersion","mock-orm","pysc-small-orm"}:
            self.measurement_name.setText("bpm-noise"); self.measurement_label.setText(f"{prefix} BPM noise")
        if orm and self.measurement_name.text() in {"bpm-noise","mock-dispersion","pysc-dispersion"}:
            self.measurement_name.setText("pysc-small-orm" if backend=="pysc" else "mock-orm"); self.measurement_label.setText("EBS pySC small ORM" if backend=="pysc" else "Mock orbit response matrix")
        self._update_kick_preview()

    def _configure_physical_dispersion_option(self):
        if not hasattr(self,"dispersion_display"):return
        item=self.dispersion_display.model().item(self.dispersion_display.findData("physical"))
        lattice_metadata=self._dispersion_lattice_metadata()
        available=lattice_metadata is not None and self.dispersion_direction.currentData()=="bipolar"
        item.setEnabled(available)
        reason="the selected backend provides no verified lattice slip factor" if lattice_metadata is None else "positive and negative RF states are required"
        item.setToolTip("" if available else f"Unavailable: {reason}.")
        self.dispersion_display_reason.setText("Lattice η available — physical dispersion uses recorded f+/f− readbacks." if available else f"Physical dispersion unavailable: {reason}.")
        if not available and self.dispersion_display.currentData()=="physical":self.dispersion_display.setCurrentIndex(0)

    def _dispersion_lattice_metadata(self):
        """Return explicit AT/design-lattice slip-factor conventions, or None."""
        metadata=getattr(self.adapter,"backend_metadata",{})
        try:
            alpha=float(metadata["momentum_compaction_factor"])
            correction=float(metadata["relativistic_correction_inverse_gamma_squared"])
            at_slip=float(metadata["slip_factor"])
            eta=float(metadata["eta_alpha_minus_inverse_gamma_squared"])
        except (KeyError,TypeError,ValueError):return None
        if not np.isfinite([alpha,correction,at_slip,eta]).all() or eta==0:return None
        if not np.isclose(eta,alpha-correction,rtol=1e-10,atol=1e-15):return None
        if not np.isclose(at_slip,-eta,rtol=1e-10,atol=1e-15):return None
        return {"alpha":alpha,"inverse_gamma_squared":correction,"at_slip_factor":at_slip,"eta":eta}

    def _dispersion_display_changed(self,*_):
        if not hasattr(self,"results_tabs") or self.measurement_type.currentData()!="dispersion":return
        if isinstance(self.result,DispersionResult):self._show_dispersion_result(self.result)
        target=self.raw_x_plot if self.dispersion_display.currentData()=="raw" else self.x_plot
        index=self.results_tabs.indexOf(target)
        if index>=0:self.results_tabs.setCurrentIndex(index)

    def _update_diagnostics_visibility(self):
        if not hasattr(self,"stats_group"):return
        kind=self.measurement_type.currentData(); bpm_noise=kind=="bpm_noise"
        self.rf_diagnostics.setVisible(not bpm_noise)
        self.restoration_label.setVisible(not bpm_noise)
        self.orm_measurement_summary.setVisible(kind=="orm"); self.orm_restoration_status.setVisible(kind=="orm")
        self.stats_group.setTitle("Statistics" if bpm_noise else "Statistics and RF diagnostics" if kind=="dispersion" else "Statistics and ORM diagnostics")
        self.stats_group.setVisible(kind!="dispersion")
        if hasattr(self,"dispersion_summary"):self.dispersion_summary.setVisible(kind=="dispersion")

    def _configure_result_tabs(self,kind):
        self.orm_column_row.setVisible(kind=="orm")
        while self.results_tabs.count():self.results_tabs.removeTab(0)
        if kind=="orm":
            entries=((self.x_plot,"Full ORM"),(self.y_plot,"Horizontal BPM / H corrector"),(self.mean_x_plot,"Horizontal BPM / V corrector"),(self.mean_y_plot,"Vertical BPM / H corrector"),(self.orm_vv_plot,"Vertical BPM / V corrector"),(self.orm_column_plot,"Selected ORM column"),(self.orm_kick_plot,"Kick diagnostics"))
        elif kind=="dispersion":entries=((self.x_plot,"Physical Dx [mm]"),(self.y_plot,"Physical Dy [mm]"),(self.raw_x_plot,"RF orbit difference Δx_RF [mm]"),(self.raw_y_plot,"RF orbit difference Δy_RF [mm]"),(self.mean_x_plot,"RF-state horizontal orbits"),(self.mean_y_plot,"RF-state vertical orbits"),(self.rf_shift_x_plot,"RF-induced horizontal shifts"),(self.rf_shift_y_plot,"RF-induced vertical shifts"))
        else:entries=((self.x_plot,"Horizontal BPM noise"),(self.y_plot,"Vertical BPM noise"),(self.mean_x_plot,"Mean horizontal orbit"),(self.mean_y_plot,"Mean vertical orbit"))
        for widget,label in entries:self.results_tabs.addTab(widget,label)

    def _update_measurement_help(self, *_):
        if not hasattr(self,"measurement_help_text"): return
        if self.measurement_type.currentData()=="bpm_noise":
            self.measurement_help_title.setText("BPM Noise")
            self.measurement_help_text.setText(
                "Repeated orbit readings acquired under stable machine conditions are used to estimate "
                "the mean orbit and BPM-to-BPM measurement repeatability/noise in the horizontal and "
                "vertical planes.\n\nThe resulting uncertainties can be used as measurement weights in "
                "pyLOCO.\n\nThis measurement is read-only and does not require changing machine setpoints."
            )
            self.measurement_help_convention.clear(); self.measurement_help_convention.setVisible(False)
            return
        if self.measurement_type.currentData()=="orm":
            self.measurement_help_title.setText("Orbit Response Matrix (ORM)")
            self.measurement_help_text.setText("Correctors are perturbed one at a time while repeated horizontal and vertical BPM orbits are acquired. The canonical matrix rows are horizontal BPMs followed by vertical BPMs; columns are horizontal correctors followed by vertical correctors. Mock mode performs only deterministic in-memory simulated writes and always attempts restoration.")
            direction=self.orm_direction.currentData() if hasattr(self,"orm_direction") else "bipolar"
            convention={"bipolar":"K+ = K0 + ΔK/2; K− = K0 − ΔK/2\nORM column = orbit(+) − orbit(−)","positive":"K+ = K0 + ΔK\nORM column = orbit(+) − reference","negative":"K− = K0 − ΔK\nORM column = orbit(−) − reference"}.get(direction,"")
            self.measurement_help_convention.setText("Current convention:\n"+convention); self.measurement_help_convention.setVisible(True); return
        self.measurement_help_title.setText("Dispersion / RF response")
        self.measurement_help_text.setText(
            "Orbit measurements at different RF frequencies determine the beam-orbit response to an "
            "RF-frequency change.\n\npyLOCO Measure preserves the raw RF-state orbit measurements and "
            "produces the canonical horizontal and vertical RF-response columns used directly by pyLOCO."
        )
        convention={
            "bipolar":"RF orbit difference (pyLOCO-compatible) = mean(−Δf) − mean(+Δf)\ncanonical signed RF step = f− − f+ = −2Δf\nphysical dispersion = RF orbit difference / (δ− − δ+)",
            "positive":"measured_eta = mean(+Δf) − mean(reference)\ncanonical RF step = +Δf",
            "negative":"measured_eta = mean(−Δf) − mean(reference)\ncanonical RF step = −Δf",
        }.get(self.dispersion_direction.currentData(),"")
        self.measurement_help_convention.setText("Current convention:\n"+convention)
        self.measurement_help_convention.setVisible(True)

    def _load_kick_arrays(self):
        nh,nv=len(self.selected_hcorrectors),len(self.selected_vcorrectors)
        if self.orm_kick_mode.currentData()=="common": return np.full(nh,self.orm_hkick.value()*1e-6),np.full(nv,self.orm_vkick.value()*1e-6)
        path=self._resolved_resource_path(self.orm_kick_file.text())
        if not path.exists(): raise ValueError("Select an existing per-corrector kick file")
        if path.suffix.lower()==".npz":
            data=np.load(path); h=np.asarray(data["horizontal" if "horizontal" in data else "h"],float); v=np.asarray(data["vertical" if "vertical" in data else "v"],float)
        elif path.suffix.lower()==".npy":
            values=np.load(path,allow_pickle=False); h=np.asarray(values[:nh],float); v=np.asarray(values[nh:],float)
        elif path.suffix.lower()==".json":
            import json
            values=json.loads(path.read_text()); h=np.asarray(values["horizontal"],float); v=np.asarray(values["vertical"],float)
        else:
            values=np.loadtxt(path,delimiter="," if path.suffix.lower()==".csv" else None); values=np.asarray(values,float).ravel(); h=values[:nh]; v=values[nh:]
        h=h.ravel(); v=v.ravel()
        if h.size!=nh or v.size!=nv: raise ValueError(f"Kick file must provide {nh} horizontal and {nv} vertical values")
        if not np.isfinite(h).all() or not np.isfinite(v).all() or np.any(h<=0) or np.any(v<=0): raise ValueError("Kick arrays must be positive finite SI-radian values")
        return h,v

    def _orm_config_changed(self,*_):
        file_mode=self.orm_kick_mode.currentData()=="file"; self.orm_kick_file_row.setVisible(file_mode); self.orm_kick_file_label.setVisible(file_mode)
        for widget in (self.orm_hkick,self.orm_vkick,self.orm_hkick_label,self.orm_vkick_label):widget.setVisible(not file_mode)
        self._update_kick_preview(); self._update_measurement_help(); self.refresh_plan()

    def _update_kick_preview(self):
        if not hasattr(self,"kick_preview"):return
        try:h,v=self._load_kick_arrays(); error=""
        except Exception as exc:h=np.array([]); v=np.array([]); error=str(exc)
        rows=list(zip(("Horizontal",)*len(h),self.selected_hcorrectors,h))+list(zip(("Vertical",)*len(v),self.selected_vcorrectors,v)); self.kick_preview.setRowCount(len(rows)); self.kick_preview.setToolTip(error)
        for row,(plane,device,kick) in enumerate(rows):
            for col,value in enumerate((row,plane,device.name,f"{kick*1e6:.6g}","File" if self.orm_kick_mode.currentData()=="file" else "Common")):self.kick_preview.setItem(row,col,QTableWidgetItem(str(value)))

    @staticmethod
    def _software_citation() -> str:
        return citation_text()

    @staticmethod
    def _software_bibtex() -> str:
        return bibtex_text()

    def _build_about_dialog(self) -> QDialog:
        dialog=QDialog(self); dialog.setWindowTitle("About pyLOCO Measure"); dialog.setModal(True); dialog.resize(600,720); dialog.setMinimumSize(440,500)
        outer=QVBoxLayout(dialog); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(28,22,28,22); layout.setSpacing(8)
        logo=QLabel(); set_asset(logo,QSize(360,240),DISPLAY_ASSET,crop_transparency=False); layout.addWidget(logo,0,Qt.AlignHCenter)

        def centered(text: str, *, rich: bool=False, object_name: str="") -> QLabel:
            label=QLabel(text); label.setAlignment(Qt.AlignCenter); label.setWordWrap(True)
            if object_name: label.setObjectName(object_name)
            if rich: label.setTextFormat(Qt.RichText); label.setOpenExternalLinks(True)
            layout.addWidget(label); return label

        centered("pyLOCO Measure",object_name="aboutTitle")
        centered(f"Installed pyLOCO version {PYLOCO_VERSION}  •  Measurement schema {MEASUREMENT_SCHEMA_VERSION}")
        centered(f"Measure application version: bundled with pyLOCO {PYLOCO_VERSION}")
        layout.addSpacing(8)
        centered("pyLOCO Measure is the measurement-acquisition companion to pyLOCO. It provides structured acquisition of the machine measurements required for LOCO analysis, including BPM noise, dispersion/RF response and orbit-response matrices.")
        layout.addSpacing(8)
        centered("pyLOCO\nStorage Ring Optics Correction",object_name="aboutTitle")
        centered("pyLOCO fits measured accelerator response data to an accelerator model to diagnose and correct optics errors.\n\npyLOCO Measure produces measurement files and sessions that can be consumed directly by pyLOCO.")
        centered("pyLOCO Suite workflow: Measure acquisition → pyLOCO Fit and optics analysis → Correct review and machine-application workflow.")
        layout.addSpacing(8)
        centered(f"Contributors: {PROJECT_CONTRIBUTORS}")
        centered(f"With thanks to: {PROJECT_ACKNOWLEDGEMENTS}")
        centered(f"License: {PROJECT_LICENSE}")
        centered(f"<i>{PROJECT_PAPER_TITLE}</i><br>IPAC’26, paper WEP5011",rich=True)
        centered(f'<a href="{PROJECT_REPOSITORY}">Repository / Source code</a><br><a href="{PROJECT_DOCUMENTATION}">Documentation</a> · <a href="{PROJECT_PAPER_URL}">Scientific reference / methodology</a>',rich=True)
        for pair in ((("Documentation",PROJECT_DOCUMENTATION),("Methodology",PROJECT_PAPER_URL)),(("Source code",PROJECT_REPOSITORY),("Report issue",PROJECT_ISSUES))):
            row=QHBoxLayout(); row.addStretch(1)
            for text,url in pair:
                button=QPushButton(text); button.clicked.connect(lambda _checked=False,value=url: QDesktopServices.openUrl(QUrl(value))); row.addWidget(button)
            row.addStretch(1); layout.addLayout(row)
        copy_row=QHBoxLayout(); copy_row.addStretch(1); copy_citation=QPushButton("Copy citation"); copy_citation.clicked.connect(lambda: QApplication.clipboard().setText(self._software_citation())); copy_bibtex=QPushButton("Copy BibTeX"); copy_bibtex.clicked.connect(lambda: QApplication.clipboard().setText(self._software_bibtex())); copy_row.addWidget(copy_citation); copy_row.addWidget(copy_bibtex); copy_row.addStretch(1); layout.addLayout(copy_row); layout.addStretch(1)
        scroll.setWidget(content); outer.addWidget(scroll,1); buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject); outer.addWidget(buttons)
        return dialog

    def _show_about_dialog(self): return present_single_about_dialog(self,self._build_about_dialog)

    def _nominal_rf_value(self):
        text=self.nominal_rf.text().strip()
        if not text:return None
        try:value=float(text)
        except ValueError:return None
        return value if np.isfinite(value) and value>0 else None

    def _dispersion_state_specs(self):
        nominal=self._nominal_rf_value()
        if nominal is None:return []
        step=self._rf_offset_hz(); direction=self.dispersion_direction.currentData()
        specs=[("reference",nominal,"Reference")]
        if direction in ("bipolar","positive"): specs.append(("positive",nominal+step,"Positive RF offset"))
        if direction in ("bipolar","negative"): specs.append(("negative",nominal-step,"Negative RF offset"))
        return specs

    def _rf_offset_hz(self):
        """Convert GUI total bipolar separation to the service's per-side offset."""
        step=float(self.rf_step.value())
        return step/2.0 if self.dispersion_direction.currentData()=="bipolar" else step

    def _update_dispersion_guide(self):
        if self.measurement_type.currentData()!="dispersion":
            self.step_instruction.setVisible(False); self.start_button.setText("Start ORM measurement" if self.measurement_type.currentData()=="orm" else "Start BPM-noise measurement"); return
        self.step_instruction.setVisible(True); specs=self._dispersion_state_specs(); total=len(specs)+1
        if isinstance(self.result,DispersionResult):
            self.step_instruction.setText("Dispersion measurement complete. Original RF restoration was verified. Use Repeat measurement to rerun the current configuration.")
            self.start_button.setVisible(False); self.repeat_button.setVisible(True); return
        self.start_button.setVisible(True)
        if self.rf_control_mode.currentData()=="automatic":
            nominal=self._nominal_rf_value(); step=self._rf_offset_hz()
            frequencies="RF readback will establish f₀" if nominal is None else f"f− = {nominal-step:.3f} Hz, f₀ = {nominal:.3f} Hz, f+ = {nominal+step:.3f} Hz"
            separation=2*step if self.dispersion_direction.currentData()=="bipolar" else step
            verification="\nVERIFY RESTORED ORBIT — acquire reference_after at f₀ and compare it with reference_before; excluded from dispersion." if self.verify_restored_orbit.isChecked() else ""
            self.step_instruction.setText(f"AUTOMATIC RF SCAN — total bipolar separation = {separation:g} Hz; ±Δf = {step:g} Hz per side.\n1/3 Reference RF: offset 0 Hz, RF = f₀, acquire reference_before.\n2/3 Positive RF: offset +Δf, RF = f+, acquire positive.\n3/3 Negative RF: offset −Δf, RF = f−, acquire negative.\nRESTORE (not a measurement state): offset → 0 Hz, RF → original f₀, verify readback.{verification}\n{frequencies}\nRestoration remains protected by cancellation-safe cleanup.")
            self.start_button.setText("Start automatic dispersion measurement"); return
        if not specs:
            self.step_instruction.setText("Enter the nominal RF frequency to create the guided manual sequence."); self.start_button.setText("Nominal RF required"); return
        if self.dispersion_step_index < len(specs):
            label,frequency,title=specs[self.dispersion_step_index]
            offset=frequency-self._nominal_rf_value()
            action="Set/confirm the machine at nominal RF." if label=="reference" else f"Set RF externally to {frequency:.3f} Hz (requested offset {offset:+.3f} Hz)."
            self.step_instruction.setText(f"STEP {self.dispersion_step_index+1} OF {total} — {title}\n{action}\nActual RF: not available — operator confirmation only.")
            self.start_button.setText("Measure reference orbit" if label=="reference" else "I have set the RF — Measure orbit")
        else:
            self.step_instruction.setText(f"STEP {total} OF {total} — Restore RF\nRestore RF externally to nominal frequency {self._nominal_rf_value():.3f} Hz. Actual RF: not available.")
            self.start_button.setText("Confirm RF restored")

    def _resolve_selection(self):
        method=self.selection_method.currentData()
        if method=="all": return self.devices
        lookup={device.name:device for device in self.devices}
        if method=="names_file":
            path=self._resolved_resource_path(self.names_file.text()); tokens=[line.strip() for line in path.read_text().splitlines() if line.strip()]
        else: tokens=[token.strip() for token in re.split(r"[,\n;]+",self.manual_input.text()) if token.strip()]
        result=[]; invalid=[]
        for token in tokens:
            if token in lookup: result.append(lookup[token]); continue
            try: position=int(token)
            except ValueError: invalid.append(token); continue
            if 0 <= position < len(self.devices): result.append(self.devices[position])
            else: invalid.append(token)
        if invalid: raise ValueError("Unknown BPM names/positions: "+", ".join(invalid))
        if hasattr(self,"measurement_type") and self.measurement_type.currentData()=="orm":
            excluded=self._positions(self.bpm_exclusions.text()); bad=sorted(pos for pos in excluded if pos<0 or pos>=len(result))
            if bad: raise ValueError("Excluded BPM positions outside selection: "+", ".join(map(str,bad)))
            result=[device for pos,device in enumerate(result) if pos not in excluded]
        if not result: raise ValueError("The BPM selection is empty")
        if len({d.name for d in result}) != len(result): raise ValueError("The BPM selection contains duplicates")
        return tuple(result)

    def refresh_preview(self):
        try: selected=self._resolve_selection()
        except Exception as exc: self.selected_devices=(); self.preview_table.setRowCount(0); self.selection_message.setText(str(exc)); self.refresh_plan(); self._update_workflow_tabs(); return
        self.selected_devices=tuple(selected); self.preview_table.setRowCount(len(selected))
        for row,device in enumerate(selected):
            for column,value in enumerate((row,device.name,device.identifier)): self.preview_table.setItem(row,column,QTableWidgetItem(str(value)))
        self._filter_bpm_table(self.bpm_search.text())
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}.get(self.selection_method.currentData(),"selection")
        self.selection_message.setText(f"{len(selected)} BPMs selected and ready for measurement. Source: {source}."); self.refresh_plan(); self._update_workflow_tabs()

    def refresh_plan(self, *_):
        if not hasattr(self,"plan_values"): return
        kind=self.measurement_type.currentData(); dispersion=kind=="dispersion"; orm=kind=="orm"; state_count=len(self._dispersion_state_specs()) if dispersion else 1
        per_state=max(0,self.readings.value()-1)*self.delay.value()+(self.settling_delay.value() if (dispersion or orm) else 0)
        estimate=(len(self.selected_hcorrectors)+len(self.selected_vcorrectors))*(2*per_state+self.settling_delay.value()) if orm else state_count*per_state; self.duration.setText(f"approximately {estimate:.2f} s")
        output=self._resolved_output_directory()
        try:self._load_kick_arrays() if orm else None; kick_valid=True
        except Exception:kick_valid=False
        automatic_rf=dispersion and self.rf_control_mode.currentData()=="automatic"
        requires_connection=self.adapter_combo.currentData()=="pysc"; connection_ok=not requires_connection or self.connection_verified
        valid=bool(connection_ok and self.selected_devices and self.measurement_name.text().strip() and self.output_directory.text().strip() and (not dispersion or automatic_rf or self._nominal_rf_value() is not None) and (not automatic_rf or AdapterCapability.RF_WRITE in self.adapter.capabilities) and (not orm or (self.selected_hcorrectors or self.selected_vcorrectors) and kick_valid and AdapterCapability.WRITE in self.adapter.capabilities))
        direction={"bipolar":"Bipolar ±Δf","positive":"One-sided +Δf","negative":"One-sided −Δf"}.get(self.dispersion_direction.currentData(),"—")
        try:kh,kv=self._load_kick_arrays()
        except Exception:kh=kv=np.array([])
        descriptor=InterfaceRegistry().descriptor(self.adapter_combo.currentData()); adapter_text=descriptor.badge
        nominal=self._nominal_rf_value(); offset=self._rf_offset_hz(); negative="Waiting for RF readback" if nominal is None else f"{nominal-offset:.3f} Hz"; positive="Waiting for RF readback" if nominal is None else f"{nominal+offset:.3f} Hz"
        separation=2*offset if self.dispersion_direction.currentData()=="bipolar" else offset
        values={"measurement":"ORM" if orm else "Dispersion" if dispersion else "BPM noise","bpms":str(len(self.selected_devices)),"h_correctors":str(len(self.selected_hcorrectors)),"v_correctors":str(len(self.selected_vcorrectors)),"total_correctors":str(len(self.selected_hcorrectors)+len(self.selected_vcorrectors)),"kick_source":"Per-corrector file" if self.orm_kick_mode.currentData()=="file" else "Common","h_kick":f"{np.min(kh)*1e6:g}–{np.max(kh)*1e6:g} µrad" if kh.size else "Invalid","v_kick":f"{np.min(kv)*1e6:g}–{np.max(kv)*1e6:g} µrad" if kv.size else "Invalid","scaled_orm":"Yes — m/rad" if self.orm_scaled.isChecked() else "No — m","orm_direction":self.orm_direction.currentText(),"rf_mode":"Automatic backend RF with restoration protection" if automatic_rf else "Manual (no writes)","nominal_rf":"Read from backend" if automatic_rf and nominal is None else f"{nominal:.3f} Hz" if nominal else "Required","rf_step_±δf":f"{offset:g} Hz per side","bipolar_separation":f"{separation:g} Hz total","negative_rf":negative,"positive_rf":positive,"direction":direction,"readings":str(self.readings.value()),"delay":f"{self.delay.value():g} s","settling_delay":f"{self.settling_delay.value():g} s","rf_states":", ".join(spec[0] for spec in self._dispersion_state_specs()) or "backend-derived","est_duration":f"{estimate:.2f} s","adapter":adapter_text,"output":str(output)}
        for key,value in values.items(): self.plan_values[key].setText(value)
        self.plan_output.setToolTip(str(output))
        self.start_button.setEnabled(valid and self.thread is None)
        reasons=[]
        if not connection_ok:reasons.append("pySC Server is disconnected — return to Machine and click Test connection")
        if not self.selected_devices:reasons.append("no valid BPMs are selected")
        if not self.measurement_name.text().strip():reasons.append("measurement name is empty")
        if not self.output_directory.text().strip():reasons.append("output/session directory is empty")
        if dispersion and not automatic_rf and self._nominal_rf_value() is None:reasons.append("nominal RF frequency is required")
        if automatic_rf and AdapterCapability.RF_WRITE not in self.adapter.capabilities:reasons.append("selected backend does not support RF writes")
        if orm and not (self.selected_hcorrectors or self.selected_vcorrectors):reasons.append("select at least one corrector")
        if orm and not kick_valid:reasons.append("corrector kick settings are invalid")
        self.start_block_reason.setText("Cannot start: "+"; ".join(reasons)+"." if reasons else "Ready to acquire through "+descriptor.badge+".")
        for widget in getattr(self,"orm_plan_widgets",[]):widget.setVisible(orm)
        for widget in getattr(self,"dispersion_plan_widgets",[]):widget.setVisible(dispersion)
        for widget in getattr(self,"settling_plan_widgets",[]):widget.setVisible(dispersion or orm)
        self._update_dispersion_guide()
        self._update_workflow_tabs()

    def _resolved_output_directory(self):
        value=Path(self.output_directory.text()).expanduser()
        if value.is_absolute(): return value
        base=self.project_path.parent if self.project_path else Path.cwd()
        return (base/value).resolve()

    def _resolved_resource_path(self,value):
        path=Path(value).expanduser()
        if path.is_absolute(): return path
        return ((self.project_path.parent if self.project_path else Path.cwd())/path).resolve()

    def start_acquisition(self):
        self.start_button.setVisible(True); self.repeat_button.setVisible(False)
        self._set_completed_result_view(False)
        self.refresh_preview()
        if not self.selected_devices: QMessageBox.warning(self,"Cannot start","Select at least one valid BPM first."); return
        if self.measurement_type.currentData()=="orm": return self._start_orm()
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="automatic": return self._start_automatic_dispersion()
        if self.measurement_type.currentData()=="dispersion": return self._start_dispersion_step()
        self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat(f"Starting — 0 / {self.readings.value()}"); self.reading_status.setText("Starting…"); self.elapsed.setText("Elapsed: 0.00 s"); self.remaining.setText(f"Remaining: ~{max(0,self.readings.value()-1)*self.delay.value():.2f} s"); self.samples.setText(f"Samples: 0 / {self.readings.value()}"); self._reset_live_plot(); self.log.clear(); self.log.appendPlainText(f"Starting BPM-noise acquisition through {self.status_badge.text()}…")
        self.thread=QThread(self); self.worker=AcquisitionWorker(BpmNoiseAcquirer(self.adapter,self.selected_devices),self.readings.value(),self.delay.value(),self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed): signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    def _start_orm(self):
        if isinstance(self.adapter,PETRAReadOnlyAdapter):QMessageBox.warning(self,"ORM unavailable","ORM acquisition requires write-enabled control-system access. PETRA is hard read-only."); return
        self._refresh_corrector_preview("hcor"); self._refresh_corrector_preview("vcor")
        try:kick_h,kick_v=self._load_kick_arrays(); acquirer=ORMAcquirer(self.adapter,self.selected_devices,self.selected_hcorrectors,self.selected_vcorrectors)
        except Exception as exc:QMessageBox.warning(self,"Cannot start ORM",str(exc)); return
        self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat("Preparing ORM"); self.reading_status.setText("Preparing first corrector…"); self.log.clear(); self.log.appendPlainText(f"Starting ORM through {self.status_badge.text()}; each original setpoint will be restored.")
        options={"direction":self.orm_direction.currentData(),"scaled":self.orm_scaled.isChecked(),"readings":self.readings.value(),"delay_seconds":self.delay.value(),"settling_delay_seconds":self.settling_delay.value()}
        self.thread=QThread(self); self.worker=ORMAcquisitionWorker(acquirer,kick_h,kick_v,options,self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.event.connect(self._on_orm_event); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed):signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    def _start_automatic_dispersion(self):
        try:acquirer=AutomaticDispersionAcquirer(self.adapter,self.selected_devices)
        except Exception as exc:QMessageBox.warning(self,"Automatic RF unavailable",str(exc)); return
        self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat("Preparing RF scan"); self.reading_status.setText("Reading original RF frequency…"); self.log.clear(); self.log.appendPlainText(f"Starting automatic dispersion through {self.status_badge.text()}; original RF is restored in cleanup.")
        options={"rf_step_hz":self._rf_offset_hz(),"readings":self.readings.value(),"delay_seconds":self.delay.value(),"direction":self.dispersion_direction.currentData(),"settling_delay_seconds":self.settling_delay.value(),"verify_restored_orbit":self.verify_restored_orbit.isChecked()}
        self.thread=QThread(self); self.worker=AutomaticDispersionWorker(acquirer,options,self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_automatic_dispersion_progress); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        self.worker.status.connect(self._on_automatic_dispersion_status)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed):signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    @Slot(int,int,int,float,object,object)
    def _on_automatic_dispersion_progress(self,state_index,current,total,elapsed,x,y):
        measurement_states=len(self._dispersion_state_specs()); work_states=measurement_states+(1 if self.verify_restored_orbit.isChecked() else 0); overall=state_index*total+current; overall_total=max(1,work_states*total)
        labels=("1/3 — Reference RF (offset 0 Hz)",f"2/3 — Positive RF (offset +{self._rf_offset_hz():g} Hz)",f"3/3 — Negative RF (offset −{self._rf_offset_hz():g} Hz)") if self.dispersion_direction.currentData()=="bipolar" else tuple(spec[0] for spec in self._dispersion_state_specs())
        label="Verifying — restored reference orbit" if state_index==measurement_states else labels[state_index] if state_index<len(labels) else f"measurement {state_index+1}"
        offset=self._rf_offset_hz()
        preview={0:"Orbit preview — reference RF",1:f"Orbit preview — positive RF (+{offset:g} Hz)",2:f"Orbit preview — negative RF (−{offset:g} Hz)",3:"Orbit preview — restored reference RF"}.get(state_index,f"Orbit preview — {label}")
        self.progress.setValue(round(100*overall/overall_total)); self.progress.setFormat(f"{label} — orbit {current} / {total}"); self.reading_status.setText(f"Automatic RF: {label} — reading {current} / {total}"); self.samples.setText(f"Samples: {overall} / {overall_total}"); self.elapsed.setText(f"Elapsed: {elapsed:.2f} s"); self._update_live_orbit(x,y,title=preview)

    @Slot(str,object)
    def _on_automatic_dispersion_status(self,phase,details):
        if phase=="restoring":
            self.reading_status.setText("Restoring — RF offset 0 Hz")
            self.progress.setFormat("Restoring original RF f₀ and verifying readback")
            self.log.appendPlainText(f"RESTORE: RF offset → 0 Hz; RF → original f₀ = {details['rf_hz']:.6f} Hz; verify RF readback.")
        elif phase=="verifying_orbit":
            self.reading_status.setText("Verifying — restored reference orbit")
            self.log.appendPlainText("RF restoration verified. Acquiring reference_after orbit; excluded from dispersion calculation.")
        elif phase=="acquiring":
            names={"reference":"Reference RF","positive":"Positive RF","negative":"Negative RF"}; index=int(details["index"])+1; count=int(details["count"]); offset=float(details["offset_hz"])
            sign="0" if offset==0 else f"{offset:+g}"
            self.log.appendPlainText(f"{index}/{count} {names.get(details['label'],details['label'])}: offset = {sign} Hz; RF = {details['rf_hz']:.6f} Hz; acquire {details['label']}-state orbit.")

    @Slot(object)
    def _on_orm_event(self,event):
        total=max(1,int(event.get("correctors",1))); current=int(event.get("corrector",1)); state=event.get("state","")
        if event.get("event")=="orbit":
            reading=int(event["reading"]); readings=int(event["readings"]); fraction=((current-1)+(0.5 if state in {"−kick","reference"} else 0)+(reading/readings)*.5)/total
            self.reading_status.setText(f"Corrector {current} / {total} — {event.get('plane')} — {event.get('device')} — {state} — orbit {reading} / {readings}")
            self.samples.setText(f"Orbit reading: {reading} / {readings}"); self._update_live_orbit(event["x"],event["y"])
        else:fraction=(current-1)/total; self.reading_status.setText(f"Corrector {current} / {total} — {event.get('plane','')} — {event.get('device','')} — {state}")
        self.progress.setValue(round(100*fraction)); self.progress.setFormat(f"Corrector {current} / {total}"); elapsed=float(event.get("elapsed",0)); self.elapsed.setText(f"Elapsed: {elapsed:.2f} s")
        if event.get("event")=="column":
            column=np.asarray(event["column"]); self.log.appendPlainText(f"Completed ORM column {current}/{total}; restoration pending.")
            self._show_orm_progress(np.asarray(event["matrix"]),column)

    def _start_dispersion_step(self):
        specs=self._dispersion_state_specs()
        if not specs:
            QMessageBox.warning(self,"Nominal RF required","Enter an explicit nominal RF frequency before measuring dispersion."); return
        if self.dispersion_step_index >= len(specs):
            return self._complete_dispersion()
        if self.dispersion_step_index==0:
            self.dispersion_states=[]; self.dispersion_started_at=monotonic(); self.log.clear()
        label,frequency,title=specs[self.dispersion_step_index]
        if hasattr(self.adapter,"set_simulated_rf_state"): self.adapter.set_simulated_rf_state(frequency)
        self.cancel_event=Event(); self.result=None; total=len(specs)*self.readings.value(); completed=self.dispersion_step_index*self.readings.value(); self.progress.setValue(round(100*completed/total)); self.progress.setFormat(f"RF state {self.dispersion_step_index+1} / {len(specs)} — 0 / {self.readings.value()}"); self.reading_status.setText(f"{title} — settling/acquiring"); self.log.appendPlainText(f"Operator confirmed requested state {label}: {frequency:.3f} Hz. Actual RF readback unavailable.")
        self.thread=QThread(self); self.worker=DispersionAcquisitionWorker(DispersionStateAcquirer(self.adapter,self.selected_devices),label,frequency,self.readings.value(),self.delay.value(),self.settling_delay.value(),self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.completed.connect(self._on_dispersion_state_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed): signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    @Slot(int,int,float,object,object)
    def _on_progress(self,current,total,elapsed,x,y):
        if self.measurement_type.currentData()=="dispersion":
            states=max(1,len(self._dispersion_state_specs())); overall=self.dispersion_step_index*total+current; overall_total=states*total; remaining=(elapsed/current)*(overall_total-overall) if current else 0.0; self.progress.setValue(round(100*overall/overall_total)); self.progress.setFormat(f"RF state {self.dispersion_step_index+1} / {states} — orbit {current} / {total}"); self.reading_status.setText(f"RF state {self.dispersion_step_index+1} / {states} — Reading {current} / {total}"); self.samples.setText(f"Samples: {overall} / {overall_total}")
        else:
            remaining=(elapsed/current)*(total-current) if current else 0.0; self.progress.setValue(round(100*current/total)); self.progress.setFormat(f"Reading {current} / {total}"); self.reading_status.setText(f"Reading {current} / {total}"); self.samples.setText(f"Samples: {current} / {total}")
        self.elapsed.setText(f"Elapsed: {elapsed:.2f} s"); self.remaining.setText(f"Remaining: ~{remaining:.2f} s")
        if current in (1,total) or current % max(1,total//20)==0: self._update_live_orbit(x,y)

    @Slot(object)
    def _on_completed(self,result):
        if isinstance(result,ORMResult):
            self.result=result; total=len(result.correctors); self.reading_status.setText("Completed — all correctors restored"); self.progress.setValue(100); self.progress.setFormat(f"Completed — {total} / {total} correctors"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Correctors: {total} / {total}"); self.log.appendPlainText("ORM acquisition completed; saving canonical matrix and raw diagnostics…")
            try:self._save_result(result)
            except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
            self._show_result(result); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("ORM saved and validated for pyLOCO."); self._update_workflow_tabs(); return
        if isinstance(result,DispersionResult):
            self.result=result; self._restored_rf_readback=float(self.adapter.get_rf_frequency()); total=sum(state.orbits_x_m.shape[0] for state in result.states); self.reading_status.setText("Completed — original RF restored"); self.progress.setValue(100); self.progress.setFormat("Completed — RF restored"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {total} / {total}"); self.log.appendPlainText("Automatic RF acquisition completed and restoration verified. Saving…")
            try:self._save_result(result)
            except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
            self.dispersion_display.setCurrentIndex(self.dispersion_display.findData("raw")); self._show_result(result); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self._update_workflow_tabs(); return
        self.result=result; total=result.orbits_x_m.shape[0]; self.reading_status.setText("Completed"); self.progress.setValue(100); self.progress.setFormat(f"Completed — {total} / {total}"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {total} / {total}"); self.log.appendPlainText("Acquisition completed. Saving canonical measurement and session…")
        try: self._save_result(result)
        except Exception as exc: self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
        self._show_result(result); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("Saved and validated for pyLOCO."); self._update_workflow_tabs()

    def repeat_measurement(self):
        """Re-run current settings only after read-only connection/restoration checks."""
        previous=self.result
        try:
            if self.adapter_combo.currentData()!="mock" and hasattr(self.adapter,"test_connection"):self.adapter.test_connection()
            if isinstance(previous,DispersionResult):
                current=float(self.adapter.get_rf_frequency())
                if current != previous.nominal_rf_hz:raise RuntimeError(f"RF is not exactly restored: {current!r} != {previous.nominal_rf_hz!r}")
            if isinstance(previous,ORMResult):
                if any(status!="restored" for status in previous.restoration_status):raise RuntimeError("One or more corrector restorations were not confirmed")
                for device,original in zip(previous.correctors,previous.original_setpoints_rad):
                    current=float(self.adapter.read(device.readback_channel).value)
                    if current != float(original):raise RuntimeError(f"{device.name} is not exactly restored: {current!r} != {float(original)!r}")
        except Exception as exc:
            QMessageBox.warning(self,"Cannot repeat measurement",str(exc)); return
        self.result=None; self.saved_measurement_path=None; self.saved_session_path=None; self.repeat_button.setVisible(False); self.start_button.setVisible(True)
        self.log.appendPlainText("Repeat preflight passed: backend connected and prior temporary state restored.")
        self.start_acquisition()

    @Slot(object)
    def _on_dispersion_state_completed(self,state):
        self.dispersion_states.append(state); self.dispersion_step_index+=1
        self.log.appendPlainText(f"Measured {state.label}: {state.orbits_x_m.shape[0]} orbit samples retained.")
        self.reading_status.setText("State complete — continue when the next RF state is confirmed")
        self._update_dispersion_guide(); self._update_workflow_tabs()

    def _complete_dispersion(self):
        specs=self._dispersion_state_specs()
        if len(self.dispersion_states)!=len(specs):
            QMessageBox.warning(self,"Incomplete dispersion","Measure every requested RF state before confirming restoration."); return
        result=DispersionResult(
            tuple(self.selected_devices),tuple(self.dispersion_states),self.dispersion_direction.currentData(),
            self._nominal_rf_value(),self._rf_offset_hz(),"confirmed_by_operator",
            0.0 if self.dispersion_started_at is None else monotonic()-self.dispersion_started_at,
        )
        self.result=result; self.reading_status.setText("Completed — RF restoration confirmed by operator"); self.progress.setValue(100); self.progress.setFormat("Completed — RF restored"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {len(specs)*self.readings.value()} / {len(specs)*self.readings.value()}")
        try:self._save_result(result)
        except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
        self.dispersion_display.setCurrentIndex(self.dispersion_display.findData("raw")); self._show_result(result); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("RF restoration confirmed by operator. Dispersion saved and validated for pyLOCO."); self._update_workflow_tabs(); self.refresh_plan()

    def _finish_thread(self,*_):
        thread=self.thread
        if thread: thread.quit(); thread.wait(3000); thread.deleteLater()
        self.thread=None; self.worker=None; self._set_acquisition_running(False); self.refresh_plan()

    def cancel_acquisition(self): self.cancel_event.set(); self.log.appendPlainText("Cancellation requested…")
    def _on_cancelled(self,message): self.progress.setValue(0); self.progress.setFormat("Cancelled"); self.reading_status.setText("Cancelled — no measurement saved"); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message); self._update_workflow_tabs()
    def _on_failed(self,message): self.progress.setValue(0); self.progress.setFormat("Failed"); self.reading_status.setText("Failed — no measurement saved"); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message); QMessageBox.critical(self,"Measurement failed",message); self._update_workflow_tabs()

    def _reset_live_plot(self):
        self.live_plot.clear(); axis=self.live_plot.figure.add_subplot(111); axis.set_title("Orbit Preview"); axis.set_xlabel("BPM selection position"); axis.set_ylabel("Orbit [µm]"); axis.grid(True,alpha=.25); self.live_plot.apply_theme()

    def _update_live_orbit(self,x,y,*,title="Orbit preview — most recent reading"):
        self.live_plot.clear(); axis=self.live_plot.figure.add_subplot(111); axis.plot(np.asarray(x)*1e6,"o-",label="X",color="#12BFC4",markersize=3); axis.plot(np.asarray(y)*1e6,"s-",label="Y",color="#F59E42",markersize=3); axis.set_title(title); axis.set_xlabel("BPM selection position"); axis.set_ylabel("Orbit [µm]"); axis.grid(True,alpha=.25); axis.legend(ncol=2); self.live_plot.apply_theme()

    def save_log(self):
        filename=QFileDialog.getSaveFileName(self,"Save acquisition log","pyloco-measure.log","Log files (*.log *.txt)")[0]
        if filename: Path(filename).write_text(self.log.toPlainText()+"\n",encoding="utf-8")

    def _save_result(self,result):
        output=self._resolved_output_directory(); output.mkdir(parents=True,exist_ok=True); safe=re.sub(r"[^A-Za-z0-9_.-]+","_",self.measurement_name.text().strip()).strip("_") or "measurement"; stamp=datetime.now().strftime("%Y%m%d-%H%M%S"); measurement=output/f"{safe}-{stamp}.h5"; is_dispersion=isinstance(result,DispersionResult)
        descriptor=InterfaceRegistry().descriptor(self.adapter_combo.currentData())
        identity,profile=self._machine_identity(); profile_metadata={}
        if profile is not None:
            profile_metadata={"machine":profile.machine,"profile_key":profile.key,"profile":profile.scenario,"profile_label":profile.label,"profile_manifest":str(profile.manifest_path),"lattice_file":str(profile.resolve("lattice_file")),"lattice_provenance":profile.configuration.get("provenance",{}),"random_seed":profile.configuration.get("random_seed")}
        metadata={"measurement_name":self.measurement_name.text(),"measurement_label":self.measurement_label.text(),"operator_comments":self.comments.toPlainText(),"adapter":self.adapter_combo.currentText(),"backend_key":descriptor.key,"backend_badge":descriptor.badge,"backend_environment":descriptor.environment,"backend_real_machine":descriptor.real_machine,"machine_identity":identity,**profile_metadata,"pysc_machine_profile":self.pysc_profile_combo.currentData() if descriptor.key=="pysc" else None,"readings_per_state":self.readings.value(),"delay_seconds":self.delay.value(),"elapsed_seconds":result.elapsed_seconds}
        if isinstance(result,ORMResult):
            nh=len(result.horizontal_correctors); write_orm(measurement,response_matrix=result.response_matrix,bpm_names=[d.name for d in result.bpms],horizontal_corrector_names=[d.name for d in result.horizontal_correctors],vertical_corrector_names=[d.name for d in result.vertical_correctors],requested_kick_h_rad=result.requested_kicks_rad[:nh],requested_kick_v_rad=result.requested_kicks_rad[nh:],actual_kick_h_rad=result.effective_kicks_rad[:nh],actual_kick_v_rad=result.effective_kicks_rad[nh:],orbit_plus_m=result.raw_state_a_m,orbit_minus_m=result.raw_state_b_m,scaled=result.scaled,direction=result.direction,original_setpoints_rad=result.original_setpoints_rad,requested_state_a_rad=result.requested_state_a_rad,requested_state_b_rad=result.requested_state_b_rad,actual_state_a_rad=result.actual_state_a_rad,actual_state_b_rad=result.actual_state_b_rad,final_setpoints_rad=result.final_setpoints_rad,timestamps_plus_s=result.timestamps_state_a_s,timestamps_minus_s=result.timestamps_state_b_s,restoration_status=result.restoration_status,metadata={**metadata,"orm_validation_convention":"historical_petra_bipolar_v1","kick_convention":"total bipolar delta_K; K+=K0+delta_K/2; K-=K0-delta_K/2","subtraction_convention":"mean(positive)-mean(negative)","normalization_convention":"divide each column by actual K+ minus K- readback only when scaled","response_matrix_unit":"m/rad" if result.scaled else "m","row_order":"horizontal_bpms,vertical_bpms","column_order":"horizontal_correctors,vertical_correctors","selected_bpm_names":[d.name for d in result.bpms],"selected_horizontal_corrector_names":[d.name for d in result.horizontal_correctors],"selected_vertical_corrector_names":[d.name for d in result.vertical_correctors]})
        elif is_dispersion:
            automatic=self.rf_control_mode.currentData()=="automatic"; states=result.states; metadata.update({"rf_control_mode":"automatic" if automatic else "manual","nominal_rf_hz":result.nominal_rf_hz,"nominal_rf_source":"backend_readback" if automatic else "manual","requested_rf_offset_hz":result.requested_offset_hz,"direction":result.direction,"canonical_measured_eta_definition":"mean_orbit_negative - mean_orbit_positive; historical PETRA III pyLOCO RF-response convention; reference_before and reference_after are excluded","canonical_measured_eta_unit":"m","rf_difference_sign_convention":"negative_minus_positive","rf_normalized_response_unit":"m/Hz","actual_rf_available":automatic,"restoration_status":result.restoration_status,"settling_delay_seconds":self.settling_delay.value(),"dispersion_measurement_states":"reference_before,positive,negative","rf_restoration_is_measurement_state":False,"verify_restored_orbit":any(state.label=="reference_after" for state in states),"reference_after_role":"post-restoration diagnostic only; excluded from physical dispersion and RF orbit difference"})
            state_rf={state.label:float(state.actual_rf_hz) for state in states}; restored=float(getattr(self,"_restored_rf_readback",np.nan)); metadata.update({"rf_original_hz":result.nominal_rf_hz,"rf_positive_hz":state_rf.get("positive",np.nan),"rf_negative_hz":state_rf.get("negative",np.nan),"rf_restored_hz":restored,"rf_restoration_difference_hz":restored-result.nominal_rf_hz if np.isfinite(restored) else np.nan})
            diagnostics={"raw_orbit_difference_x_m":result.measured_eta_x,"raw_orbit_difference_y_m":result.measured_eta_y}
            if any(state.label=="reference_after" for state in states):
                before=result.state("reference"); after=result.state("reference_after")
                diagnostics.update({"restored_orbit_difference_x_m":after.mean_x_m-before.mean_x_m,"restored_orbit_difference_y_m":after.mean_y_m-before.mean_y_m})
            lattice_metadata=self._dispersion_lattice_metadata()
            recorded=np.asarray([state.actual_rf_hz for state in states],dtype=float)
            if result.direction=="bipolar" and lattice_metadata is not None and np.isfinite(recorded).all():
                plus=result.state("positive"); minus=result.state("negative"); eta=lattice_metadata["eta"]
                deltas=relative_momentum_deviation([minus.actual_rf_hz,plus.actual_rf_hz],result.nominal_rf_hz,eta); dx,span=physical_dispersion(minus.mean_x_m,plus.mean_x_m,*deltas); dy,_=physical_dispersion(minus.mean_y_m,plus.mean_y_m,*deltas)
                metadata.update({"momentum_compaction_factor":lattice_metadata["alpha"],"relativistic_correction_inverse_gamma_squared":lattice_metadata["inverse_gamma_squared"],"at_slip_factor":lattice_metadata["at_slip_factor"],"slip_factor_eta":eta,"slip_factor_convention":"AT ring.get_slip_factor() = 1/gamma^2 - alpha_c; eta = alpha_c - 1/gamma^2 = -AT slip factor","momentum_relation":MOMENTUM_RELATION,"delta_negative":float(deltas[0]),"delta_positive":float(deltas[1]),"delta_span":float(span),"delta_span_definition":"delta_negative - delta_positive","physical_dispersion_definition":"(mean_orbit_negative - mean_orbit_positive) / (delta_negative - delta_positive)","physical_dispersion_unit":"m"})
                diagnostics.update({"physical_dispersion_x_m":dx,"physical_dispersion_y_m":dy,"relative_momentum_deviation":deltas})
            saved_labels=["reference_before" if s.label=="reference" else s.label for s in states]
            write_dispersion(measurement,measured_eta_x=result.measured_eta_x,measured_eta_y=result.measured_eta_y,bpm_names=[d.name for d in result.devices],rf_frequency_hz=[s.requested_rf_hz for s in states],rf_setpoint_hz=[s.requested_rf_hz for s in states],rf_readback_hz=[s.actual_rf_hz for s in states],raw_orbits_x_m=np.stack([s.orbits_x_m for s in states]),raw_orbits_y_m=np.stack([s.orbits_y_m for s in states]),rf_step_hz=result.canonical_rf_step_hz,bidirectional=result.direction=="bipolar",metadata=metadata,diagnostics=diagnostics,state_labels=saved_labels,operator_confirmed=[s.operator_confirmed for s in states],sample_timestamps_s=np.stack([s.timestamps_s for s in states]),restoration_status=result.restoration_status)
        else:
            metadata["readings"]=self.readings.value(); write_bpm_noise(measurement,noise_x_m=result.noise_x_m,noise_y_m=result.noise_y_m,bpm_names=[d.name for d in result.devices],raw_orbits_x_m=result.orbits_x_m,raw_orbits_y_m=result.orbits_y_m,metadata=metadata)
        validate_measurement_file(measurement); manifest=output/"measurement-session.pyloco-session.json"
        entries=[]
        role="orm" if isinstance(result,ORMResult) else "dispersion" if is_dispersion else "bpm_noise"
        if manifest.exists(): entries=list(load_session(manifest,validate_files=False).files); entries=[e for e in entries if e.role!=role]
        options={"dataset":"response_matrix"} if isinstance(result,ORMResult) else {"horizontal_dataset":"measured_eta_x","vertical_dataset":"measured_eta_y"} if is_dispersion else {"horizontal_dataset":"Noise_BPMx","vertical_dataset":"Noise_BPMy"}
        entries.append(SessionFile(role,measurement.name,options))
        session=MeasurementSession(session_id=output.name or safe,files=tuple(entries),metadata={"label":self.measurement_label.text(),"updated":datetime.now().isoformat(timespec="seconds"),"machine_identity":identity,**profile_metadata}); save_session(manifest,session)
        self.saved_measurement_path=measurement; self.saved_session_path=manifest; self.paths.setText(f"SAVED ✓\nMeasurement file: {measurement}\nSession manifest: {manifest}"); self.paths.setToolTip(str(measurement)); self.paths.setObjectName("savedPath"); self.paths.style().unpolish(self.paths); self.paths.style().polish(self.paths); self.statusBar().showMessage(f"Saved: {measurement}")

    def _stats(self,values):
        return {"Mean":np.mean(values),"RMS":np.sqrt(np.mean(values**2)),"Min":np.min(values),"Max":np.max(values)}

    def _configured_bpm_noise(self, plane: str):
        metadata=getattr(self.adapter,"backend_metadata",{})
        value=metadata.get(f"configured_bpm_noise_sigma_{plane.lower()}_m")
        try:value=float(value)
        except (TypeError,ValueError):return None
        return value if np.isfinite(value) and value >= 0 else None

    @staticmethod
    def _noise_display_unit(values, configured=None):
        candidates=np.abs(np.asarray(values,dtype=float))
        magnitude=float(np.nanmax(candidates)) if candidates.size else 0.0
        if configured is not None:magnitude=max(magnitude,abs(float(configured)))
        if not np.isfinite(magnitude) or magnitude == 0:return 1e9,"nm"
        exponent=int(np.floor(np.log10(magnitude)/3)*3)
        exponent=max(-12,min(0,exponent))
        return 10.0**(-exponent),{0:"m",-3:"mm",-6:"µm",-9:"nm",-12:"pm"}[exponent]

    def _show_result(self,result):
        if isinstance(result,ORMResult): return self._show_orm_result(result)
        if isinstance(result,DispersionResult): return self._show_dispersion_result(result)
        display={}
        for plane,canvas,values,title,color in (("x",self.x_plot,result.noise_x_m,"Horizontal BPM noise σx","#11B7C1"),("y",self.y_plot,result.noise_y_m,"Vertical BPM noise σy","#F59E42")):
            configured=self._configured_bpm_noise(plane)
            scale,unit=self._noise_display_unit(values,configured); display[plane]=(scale,unit)
            canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(values)); shown=np.asarray(values)*scale
            ax.plot(positions,shown,color=color,linewidth=.65,alpha=.45,zorder=1)
            ax.scatter(positions,shown,color=color,s=13,alpha=.9,edgecolors="none",zorder=2,label=f"Measured σ{plane}")
            if configured is not None:
                configured_shown=configured*scale; ax.axhline(configured_shown,color="#D946EF",linestyle="--",linewidth=1.4,label=f"σconfigured = {configured_shown:g} {unit}",zorder=0); ax.legend(loc="best")
            ax.set_title(f"{title} [{unit}]"); ax.set_ylabel(f"Noise [{unit}]"); ax.set_xlabel("BPM index / selection position"); ax.grid(True,alpha=.25); canvas.apply_theme()
        for canvas,values,title,color in ((self.mean_x_plot,result.mean_x_m,"Mean horizontal orbit","#11B7C1"),(self.mean_y_plot,result.mean_y_m,"Mean vertical orbit","#F59E42")):
            canvas.clear(); ax=canvas.figure.add_subplot(111); ax.plot(np.asarray(values)*1e6,"o-",color=color); ax.set_title(title); ax.set_ylabel("Mean orbit [µm]"); ax.set_xlabel("BPM selection position"); ax.grid(True,alpha=.25); canvas.apply_theme()
        def text(label,values,plane):
            scale,unit=display[plane]; return label+": "+", ".join(f"{k} {v*scale:.3f} {unit}" for k,v in self._stats(values).items())
        self.summary_x.setText(text("Horizontal",result.noise_x_m,"x")); self.summary_y.setText(text("Vertical",result.noise_y_m,"y"))
        self.rf_diagnostics.clear()
        self.restoration_label.clear()

    @staticmethod
    def _heatmap(canvas,data,title,unit):
        canvas.clear(); ax=canvas.figure.add_subplot(111); im=ax.imshow(np.asarray(data),aspect="auto",origin="lower",cmap="RdBu_r"); ax.set_title(title); ax.set_xlabel("Corrector selection position"); ax.set_ylabel("BPM selection position"); canvas.figure.colorbar(im,ax=ax,label=f"Response [{unit}]"); canvas.apply_theme()
        return ax

    def _show_orm_progress(self,matrix,column):
        finite=np.where(np.isfinite(matrix),matrix,np.nan); self._heatmap(self.x_plot,finite,"Evolving ORM heatmap","m")
        self.orm_column_plot.clear(); ax=self.orm_column_plot.figure.add_subplot(111); nb=len(self.selected_devices); ax.plot(column[:nb]*1e6,label="Horizontal BPMs"); ax.plot(column[nb:]*1e6,label="Vertical BPMs"); ax.set_title("Current ORM column"); ax.set_ylabel("Orbit difference [µm]"); ax.legend(); ax.grid(True,alpha=.25); self.orm_column_plot.apply_theme()

    def _show_orm_result(self,result):
        matrix=result.response_matrix; nb=len(result.bpms); nh=len(result.horizontal_correctors); unit="m/rad" if result.scaled else "m"
        full_ax=self._heatmap(self.x_plot,matrix,"Full orbit response matrix",unit); self._heatmap(self.y_plot,matrix[:nb,:nh],"H corrector → X BPM",unit); self._heatmap(self.mean_x_plot,matrix[:nb,nh:],"V corrector → X BPM (cross-plane)",unit); self._heatmap(self.mean_y_plot,matrix[nb:,:nh],"H corrector → Y BPM (cross-plane)",unit); self._heatmap(self.orm_vv_plot,matrix[nb:,nh:],"V corrector → Y BPM",unit)
        if nh and len(result.vertical_correctors):full_ax.axvline(nh-.5,color="white",linewidth=2,linestyle="--")
        full_ax.axhline(nb-.5,color="white",linewidth=2,linestyle="--")
        full_ax.set_xlabel(f"Corrector columns: H [0…{max(0,nh-1)}] | V [{nh}…{len(result.correctors)-1}]")
        full_ax.set_ylabel(f"BPM rows: X [0…{nb-1}] | Y [{nb}…{2*nb-1}]"); self.x_plot.apply_theme()
        self._displayed_orm_result=result; self.orm_column_selector.blockSignals(True); self.orm_column_selector.clear()
        for index,device in enumerate(result.correctors):self.orm_column_selector.addItem(f"{index}: {device.name} ({device.plane})",index)
        self.orm_column_selector.setCurrentIndex(len(result.correctors)-1); self.orm_column_selector.blockSignals(False); self._update_selected_orm_column()
        self.orm_kick_plot.clear(); ax=self.orm_kick_plot.figure.add_subplot(111); pos=np.arange(len(result.correctors)); ax.plot(pos,result.requested_kicks_rad*1e6,"o-",label="Requested |ΔK|"); ax.plot(pos,np.abs(result.effective_kicks_rad)*1e6,"s--",label="Actual |ΔK|"); ax.set_title("Requested versus actual corrector kicks"); ax.set_ylabel("Kick [µrad]"); ax.legend(); ax.grid(True,alpha=.25); self.orm_kick_plot.apply_theme()
        values=np.asarray(matrix); self.summary_x.setText(f"ORM: min {np.min(values):.6g}, max {np.max(values):.6g}, mean {np.mean(values):.6g}, RMS {np.sqrt(np.mean(values**2)):.6g} {unit}"); errors=(result.effective_kicks_rad-np.where(result.direction=="negative",-result.requested_kicks_rad,result.requested_kicks_rad))*1e6; self.summary_y.setText(f"Kick error: min {np.min(errors):.6g}, max {np.max(errors):.6g}, RMS {np.sqrt(np.mean(errors**2)):.6g} µrad"); restored=all(status=="restored" for status in result.restoration_status) and np.allclose(result.final_setpoints_rad,result.original_setpoints_rad,rtol=0,atol=1e-12); self.orm_restoration_status.setText("All correctors restored: ✓ YES" if restored else "All correctors restored: ✗ NO"); kicks=result.requested_kicks_rad*1e6; kick_text=f"{kicks[0]:g} µrad" if np.allclose(kicks,kicks[0]) else f"{np.min(kicks):g}–{np.max(kicks):g} µrad"; identity,_=self._machine_identity(); readings=result.raw_state_a_m.shape[1] if result.raw_state_a_m.ndim==3 else 1; self.orm_measurement_summary.setText(f"Machine/profile: {identity.replace(chr(10),' — ')}\nBPMs used: {nb} | H correctors used: {nh} | V correctors used: {len(result.vertical_correctors)}\nTotal bipolar kick: {kick_text} | Readings/state: {readings}\nMatrix shape: {matrix.shape[0]} × {matrix.shape[1]} | Matrix unit: {unit}"); self.restoration_label.setText("Restoration: "+", ".join(f"{device.name}={status}" for device,status in zip(result.correctors,result.restoration_status)))

    def _orm_transaction_text(self,result,index):
        device=result.correctors[index]; scale=1e6; error=(result.final_setpoints_rad[index]-result.original_setpoints_rad[index])*scale
        return (f"Corrector transaction — {device.name} ({device.plane})\n"
                f"Backend identifier: {device.identifier}\n"
                f"Original: {result.original_setpoints_rad[index]*scale:+.9g} µrad; requested +/−: {result.requested_state_a_rad[index]*scale:+.9g} / {result.requested_state_b_rad[index]*scale:+.9g} µrad\n"
                f"Readback +/−: {result.actual_state_a_rad[index]*scale:+.9g} / {result.actual_state_b_rad[index]*scale:+.9g} µrad; actual separation: {result.effective_kicks_rad[index]*scale:+.9g} µrad\n"
                f"Restored readback: {result.final_setpoints_rad[index]*scale:+.9g} µrad; restoration error: {error:+.9g} µrad — {result.restoration_status[index]}\n"
                f"Matrix convention: orbit(+) − orbit(−); rows X BPM then Y BPM; columns H then V; {'normalized by actual separation' if result.scaled else 'raw orbit difference' }.")

    def _update_selected_orm_column(self,*_):
        result=getattr(self,"_displayed_orm_result",None); index=self.orm_column_selector.currentData()
        if result is None or index is None:return
        index=int(index); nb=len(result.bpms); column=result.response_matrix[:,index]; scale,unit=(1.0,"m/rad") if result.scaled else (1e6,"µm raw orbit difference"); self.orm_column_plot.clear(); ax=self.orm_column_plot.figure.add_subplot(111); ax.plot(column[:nb]*scale,"o-",markersize=3,label="X BPM response"); ax.plot(column[nb:]*scale,"s-",markersize=3,label="Y BPM response"); device=result.correctors[index]; ax.set_title(f"{device.name} — {device.plane} corrector"); ax.set_xlabel("BPM selection position"); ax.set_ylabel(f"Response [{unit}]"); ax.legend(); ax.grid(True,alpha=.25); self.orm_column_plot.apply_theme()
        self.rf_diagnostics.setText(self._orm_transaction_text(result,index))

    def _show_dispersion_result(self,result):
        lattice_metadata=self._dispersion_lattice_metadata(); eta=None if lattice_metadata is None else lattice_metadata["eta"]
        deltas=None; physical_x=physical_y=None
        plus=result.state("positive") if result.direction in {"bipolar","positive"} else None
        minus=result.state("negative") if result.direction in {"bipolar","negative"} else None
        reference=result.state("reference")
        if eta is not None and result.direction=="bipolar":
            recorded=np.asarray([minus.actual_rf_hz,plus.actual_rf_hz],dtype=float)
            if np.isfinite(recorded).all():
                deltas=relative_momentum_deviation(recorded,result.nominal_rf_hz,eta)
                physical_x,delta_span=physical_dispersion(minus.mean_x_m,plus.mean_x_m,*deltas)
                physical_y,_=physical_dispersion(minus.mean_y_m,plus.mean_y_m,*deltas)
        plot_sets=((self.raw_x_plot,result.measured_eta_x,"RF orbit difference Δx_RF = x(f−) − x(f+)",1e3,"mm","#11B7C1"),(self.raw_y_plot,result.measured_eta_y,"RF orbit difference Δy_RF = y(f−) − y(f+)",1e3,"mm","#F59E42"))
        if physical_x is not None:
            plot_sets+=((self.x_plot,physical_x,"Physical horizontal dispersion Dₓ",1e3,"mm","#11B7C1"),(self.y_plot,physical_y,"Physical vertical dispersion Dᵧ",1e3,"mm","#F59E42"))
        else:
            for canvas,title in ((self.x_plot,"Physical horizontal dispersion unavailable"),(self.y_plot,"Physical vertical dispersion unavailable")):
                canvas.clear(); ax=canvas.figure.add_subplot(111); ax.set_title(title); ax.text(.5,.5,"Backend momentum compaction and bipolar RF readbacks are required.",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off(); canvas.apply_theme()
        for canvas,values,title,scale,unit,color in plot_sets:
            canvas.clear(); ax=canvas.figure.add_subplot(111); ax.plot(np.asarray(values)*scale,"o-",color=color,markersize=4); ax.set_title(f"{title} [{unit}]"); ax.set_ylabel(f"{'Physical dispersion' if title.startswith('Physical') else 'RF orbit difference'} [{unit}]"); ax.set_xlabel("BPM selection position"); ax.grid(True,alpha=.25); canvas.apply_theme()
        for canvas,plane,title in ((self.mean_x_plot,"x","Raw and mean horizontal orbit by RF state"),(self.mean_y_plot,"y","Raw and mean vertical orbit by RF state")):
            canvas.clear(); ax=canvas.figure.add_subplot(111)
            colors=("#64748B","#F59E42","#11B7C1","#D946EF")
            offset=result.requested_offset_hz
            display_labels={"reference":"Reference RF (0 Hz)","reference_after":"Restored RF (0 Hz)","positive":f"Positive RF (+{offset:g} Hz)","negative":f"Negative RF (−{offset:g} Hz)"}
            for state,color in zip(result.states,colors):
                raw=getattr(state,f"orbits_{plane}_m")*1e6
                for sample in raw: ax.plot(sample,color=color,alpha=.10,linewidth=.7)
                ax.plot(np.arange(len(result.devices)),getattr(state,f"mean_{plane}_m")*1e6,"o-",label=display_labels.get(state.label,state.label),color=color,markersize=3)
            ax.set_title(title); ax.set_ylabel("Orbit [µm]"); ax.set_xlabel("BPM selection position"); ax.grid(True,alpha=.25); ax.legend(); canvas.apply_theme()
        symmetry=[]
        if plus is not None and minus is not None:
            for canvas,plane,title in ((self.rf_shift_x_plot,"x","Horizontal RF-induced orbit shift"),(self.rf_shift_y_plot,"y","Vertical RF-induced orbit shift")):
                shift_plus=getattr(plus,f"mean_{plane}_m")-getattr(reference,f"mean_{plane}_m")
                shift_minus=getattr(minus,f"mean_{plane}_m")-getattr(reference,f"mean_{plane}_m")
                canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(result.devices))
                ax.plot(positions,shift_plus*1e6,"o-",markersize=3,color="#F59E42",label=f"{plane}(f+) − {plane}(f₀)")
                ax.plot(positions,shift_minus*1e6,"o-",markersize=3,color="#11B7C1",label=f"{plane}(f−) − {plane}(f₀)")
                ax.axhline(0,color="#64748B",linewidth=.8); ax.set_title(title); ax.set_ylabel("Orbit shift [µm]"); ax.set_xlabel("BPM selection position"); ax.grid(True,alpha=.25); ax.legend(); canvas.apply_theme()
                mismatch=(shift_plus+shift_minus)*1e6
                correlation=float(np.corrcoef(shift_plus,-shift_minus)[0,1]) if np.std(shift_plus)>0 and np.std(shift_minus)>0 else float("nan")
                symmetry.append(f"{plane.upper()} RMS mismatch {np.sqrt(np.mean(mismatch**2)):.3f} µm, correlation {correlation:.6f}")
        else:
            for canvas in (self.rf_shift_x_plot,self.rf_shift_y_plot):
                canvas.clear(); ax=canvas.figure.add_subplot(111); ax.text(.5,.5,"Positive and negative RF states are required.",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off(); canvas.apply_theme()
        display_physical=self.dispersion_display.currentData()=="physical" and physical_x is not None
        values_x,values_y=(physical_x,physical_y) if display_physical else (result.measured_eta_x,result.measured_eta_y)
        scale,unit=(1e3,"mm")
        titles=("Physical Dx","Physical Dy") if display_physical else ("RF orbit difference Δx_RF","RF orbit difference Δy_RF")
        def text(label,values): return label+": "+", ".join(f"{k} {v*scale:.3f} {unit}" for k,v in self._stats(values).items())
        self.summary_x.setText(text(titles[0],values_x)); self.summary_y.setText(text(titles[1],values_y))
        self.rf_response_stats.setText(text("X",result.measured_eta_x)+"\n"+text("Y",result.measured_eta_y))
        self.physical_stats.setText((text("Dₓ",physical_x)+"\n"+text("Dᵧ",physical_y)) if physical_x is not None else "Unavailable: a verified lattice slip factor and bipolar RF readbacks are required.")
        readbacks={state.label:state.actual_rf_hz for state in result.states}; restored=getattr(self,"_restored_rf_readback",np.nan); difference=restored-result.nominal_rf_hz
        automatic=self.rf_control_mode.currentData()=="automatic"
        if automatic:self.restoration_label.setText(f"RF readbacks — original: {result.nominal_rf_hz:.6f} Hz; negative: {readbacks.get('negative',np.nan):.6f} Hz; positive: {readbacks.get('positive',np.nan):.6f} Hz; restored: {restored:.6f} Hz; restoration difference: {difference:+g} Hz ({result.restoration_status}).")
        else:self.restoration_label.setText(f"RF restoration: {result.restoration_status}. Readback verification: not available. Canonical step: {result.canonical_rf_step_hz:g} Hz.")
        if lattice_metadata is None:self.rf_diagnostics.setText("Physical dispersion unavailable: the backend provides no verified 4D/design-lattice slip factor. The pyLOCO-compatible RF orbit difference remains available.")
        elif result.direction!="bipolar":self.rf_diagnostics.setText("Physical dispersion requires recorded positive and negative RF states; the one-sided RF orbit response remains available.")
        else:
            if deltas is None:deltas=relative_momentum_deviation([readbacks["negative"],readbacks["positive"]],result.nominal_rf_hz,eta); delta_span=float(deltas[0]-deltas[1])
            diagnostic=f"Momentum diagnostics — αc: {lattice_metadata['alpha']:.12g}; 1/γ²: {lattice_metadata['inverse_gamma_squared']:.12g}; η=αc−1/γ²: {eta:.12g}; AT get_slip_factor(): {lattice_metadata['at_slip_factor']:.12g}; δ−: {deltas[0]:+.12g}; δ+: {deltas[1]:+.12g}; δ− − δ+: {delta_span:+.12g}. Convention: {MOMENTUM_RELATION}."
            if any(state.label=="reference_after" for state in result.states):
                before=result.state("reference"); after=result.state("reference_after"); dx=(after.mean_x_m-before.mean_x_m)*1e6; dy=(after.mean_y_m-before.mean_y_m)*1e6
                diagnostic+=f"\nRestored-orbit diagnostic (excluded from dispersion): ΔX reference-after − reference-before: RMS {np.sqrt(np.mean(dx**2)):.3f} µm / max {np.max(np.abs(dx)):.3f} µm; ΔY: RMS {np.sqrt(np.mean(dy**2)):.3f} µm / max {np.max(np.abs(dy)):.3f} µm."
            self.rf_diagnostics.setText(diagnostic)
        if deltas is None:
            delta_details="δ−: unavailable\nδ+: unavailable\nΔδ = δ− − δ+: unavailable"
        else:
            delta_details=f"δ−: {deltas[0]:+.12g}\nδ+: {deltas[1]:+.12g}\nΔδ = δ− − δ+: {float(deltas[0]-deltas[1]):+.12g}"
        self.calculation_details_body.setText(
            f"Momentum compaction αc: {lattice_metadata['alpha']:.12g}\n"
            f"Relativistic correction 1/γ²: {lattice_metadata['inverse_gamma_squared']:.12g}\n"
            f"AT ring.get_slip_factor() (1/γ²−αc): {lattice_metadata['at_slip_factor']:.12g}\n"
            f"Slip factor η = αc−1/γ²: {eta:.12g}\n"
            f"Nominal RF f₀: {result.nominal_rf_hz:.9f} Hz\n"
            f"f−: {readbacks.get('negative',np.nan):.9f} Hz\n"
            f"f+: {readbacks.get('positive',np.nan):.9f} Hz\n"
            f"Total RF separation f+ − f−: {readbacks.get('positive',np.nan)-readbacks.get('negative',np.nan):.9g} Hz\n{delta_details}"
            if lattice_metadata is not None else
            f"Momentum compaction αc: unavailable\nRelativistic correction 1/γ²: unavailable\nSlip factor η: unavailable\nNominal RF f₀: {result.nominal_rf_hz:.9f} Hz\nf−: {readbacks.get('negative',np.nan):.9f} Hz\nf+: {readbacks.get('positive',np.nan):.9f} Hz\nTotal RF separation f+ − f−: {readbacks.get('positive',np.nan)-readbacks.get('negative',np.nan):.9g} Hz\n{delta_details}"
        )
        restored_ok=result.restoration_status in {"restored","confirmed_by_operator"} and (not np.isfinite(restored) or np.isclose(difference,0,rtol=0,atol=1e-9))
        self.restoration_status.setText("RF restoration: ✓ RESTORED" if restored_ok else f"RF restoration: ✗ {result.restoration_status.upper()}")
        self.restoration_values.setText(f"Initial RF: {result.nominal_rf_hz:.9f} Hz\nRestored RF: {restored:.9f} Hz\nRestoration error [Hz]: {difference:+.9g}")
        if any(state.label=="reference_after" for state in result.states):
            after=result.state("reference_after"); return_x=(after.mean_x_m-reference.mean_x_m)*1e6; return_y=(after.mean_y_m-reference.mean_y_m)*1e6
            restored_orbit=f"Reference-orbit return — X RMS {np.sqrt(np.mean(return_x**2)):.3f} µm / max {np.max(np.abs(return_x)):.3f} µm; Y RMS {np.sqrt(np.mean(return_y**2)):.3f} µm / max {np.max(np.abs(return_y)):.3f} µm."
        else:restored_orbit="Reference-orbit return: not acquired."
        symmetry_text=("\nBipolar symmetry — "+"; ".join(symmetry)+".") if symmetry else ""
        self.restoration_diagnostic.setText(restored_orbit+"\nThe restored-reference orbit is diagnostic-only and excluded from dispersion."+symmetry_text)
        target=self.raw_x_plot if self.dispersion_display.currentData()=="raw" else self.x_plot
        index=self.results_tabs.indexOf(target)
        if index>=0:self.results_tabs.setCurrentIndex(index)

    def _update_workflow_tabs(self):
        if not hasattr(self,"tabs") or self.tabs.count()<4:return
        machine_done=self.connection_verified
        bpms_done=bool(self.selected_devices)
        measurement_done=self.result is not None
        review_done=self.saved_measurement_path is not None and self.saved_session_path is not None
        labels=((machine_done,"Machine"),(bpms_done,"Devices"),(measurement_done,"Measurement"),(review_done,"Review && Save"))
        for index,(done,label) in enumerate(labels): self.tabs.setTabText(index,("✓ " if done else "")+label)

    def validate_saved(self):
        if not self.saved_measurement_path:return
        info=validate_measurement_file(self.saved_measurement_path); session=load_session(self.saved_session_path)
        sign="\nRF response verified: mean orbit(f−) − mean orbit(f+) in metres." if info["kind"]=="dispersion" else ""
        QMessageBox.information(self,"Valid for pyLOCO",f"Measurement schema {info['schema_version']} is valid.{sign}\nSession is incomplete for fitting; missing: {', '.join(session.missing_roles)}")

    def explain_open(self):
        session=load_session(self.saved_session_path)
        try:ok,detail=launch_suite_application("fit","--measurement-session",str(self.saved_session_path))
        except Exception as exc:QMessageBox.warning(self,"Cannot open pyLOCO Fit",str(exc)); return
        if not ok:QMessageBox.warning(self,"Cannot open pyLOCO Fit",detail); return
        missing=", ".join(session.missing_roles) or "none"; self.statusBar().showMessage(f"Opening session in pyLOCO Fit ({detail}); missing roles: {missing}")

    def _collect_project(self):
        self._store_active_selection()
        h=self.corrector_selection_widgets["hcor"]; v=self.corrector_selection_widgets["vcor"]
        identity,profile=self._machine_identity(); profile_meta={} if profile is None else {"machine":profile.machine,"profile":profile.scenario,"profile_key":profile.key,"profile_manifest":str(profile.manifest_path),"lattice_file":str(profile.resolve("lattice_file")),"provenance":profile.configuration.get("provenance",{}),"random_seed":profile.configuration.get("random_seed")}
        return MeasureProject(measurement_type=self.measurement_type.currentData(),measurement_name=self.measurement_name.text(),measurement_label=self.measurement_label.text(),operator_comments=self.comments.toPlainText(),adapter=self.adapter_combo.currentText(),pysc_profile=self.pysc_profile_combo.currentData(),bpm_selection_method=self.selection_method.currentData(),bpm_manual=self.manual_input.text(),bpm_names_file=self.names_file.text(),excluded_bpm_positions=self.bpm_exclusions.text(),hcor_selection_method=h["method"].currentData(),hcor_manual=h["manual"].text(),hcor_names_file=h["file"].text(),vcor_selection_method=v["method"].currentData(),vcor_manual=v["manual"].text(),vcor_names_file=v["file"].text(),excluded_hcor_positions=h["exclusion"].text(),excluded_vcor_positions=v["exclusion"].text(),measurement_selections=self._selection_states,readings=self.readings.value(),delay_seconds=self.delay.value(),settling_delay_seconds=self.settling_delay.value(),verify_restored_orbit=self.verify_restored_orbit.isChecked(),rf_control_mode=self.rf_control_mode.currentData(),nominal_rf_hz=self._nominal_rf_value(),nominal_rf_source="manual",rf_step_hz=self.rf_step.value(),dispersion_direction=self.dispersion_direction.currentData(),orm_direction=self.orm_direction.currentData(),orm_kick_mode=self.orm_kick_mode.currentData(),orm_horizontal_kick_rad=self.orm_hkick.value()*1e-6,orm_vertical_kick_rad=self.orm_vkick.value()*1e-6,orm_kick_file=self.orm_kick_file.text(),orm_scaled=self.orm_scaled.isChecked(),output_directory=self.output_directory.text(),theme=self.project.theme,metadata={**self.project.metadata,"machine_identity":identity,"profile":profile_meta})

    def _load_project_widgets(self):
        self.pysc_profile_combo.setCurrentIndex(max(0,self.pysc_profile_combo.findData(self.project.pysc_profile)))
        self.adapter_combo.setCurrentIndex(max(0,self.adapter_combo.findText(self.project.adapter)))
        p=self.project; self._selection_states=p.measurement_selections; self._active_selection_kind=p.measurement_type; self.measurement_type.blockSignals(True); self.measurement_type.setCurrentIndex(max(0,self.measurement_type.findData(p.measurement_type))); self.measurement_type.blockSignals(False); self._restore_selection(p.measurement_type); self.measurement_name.setText(p.measurement_name); self.measurement_label.setText(p.measurement_label); self.comments.setPlainText(p.operator_comments); self.readings.setValue(p.readings); self.delay.setValue(p.delay_seconds); self.settling_delay.setValue(p.settling_delay_seconds); self.verify_restored_orbit.setChecked(p.verify_restored_orbit); self.nominal_rf.setText("" if p.nominal_rf_hz is None else f"{p.nominal_rf_hz:g}"); self.rf_step.setValue(p.rf_step_hz); self.dispersion_direction.setCurrentIndex(max(0,self.dispersion_direction.findData(p.dispersion_direction))); self.orm_direction.setCurrentIndex(max(0,self.orm_direction.findData(p.orm_direction))); self.orm_kick_mode.setCurrentIndex(max(0,self.orm_kick_mode.findData(p.orm_kick_mode))); self.orm_hkick.setValue(p.orm_horizontal_kick_rad*1e6); self.orm_vkick.setValue(p.orm_vertical_kick_rad*1e6); self.orm_kick_file.setText(p.orm_kick_file); self.orm_scaled.setChecked(p.orm_scaled)
        self.output_directory.setText(p.output_directory); self.apply_theme(p.theme); self.refresh_preview(); self._measurement_type_changed(); self._update_machine_identity()

    def new_project(self): self.project=MeasureProject(); self.project_path=None; self._load_project_widgets()
    def open_project(self):
        path=QFileDialog.getOpenFileName(self,"Open pyLOCO Measure project","","pyLOCO Measure (*.pyloco-measure.json *.json)")[0]
        if path: self.project=load_measure_project(path); self.project_path=Path(path).resolve(); self._load_project_widgets(); self.statusBar().showMessage(f"Opened {path}")
    def save_project(self):
        if self.project_path is None: return self.save_project_as()
        self.project=self._collect_project(); save_measure_project(self.project_path,self.project); self.statusBar().showMessage(f"Saved {self.project_path}")
    def save_project_as(self):
        path=QFileDialog.getSaveFileName(self,"Save pyLOCO Measure project",f"{self.measurement_type.currentData().replace('_','-')}.pyloco-measure.json","pyLOCO Measure (*.pyloco-measure.json)")[0]
        if not path:return
        self.project_path=Path(path).resolve(); self.save_project()

    def apply_theme(self,key):
        self.project.theme=key; app=QApplication.instance(); apply_application_theme(app,theme_for_key(key)); app.setStyleSheet(app.styleSheet()+TEAL_QSS); self.theme_button.setText("☀ Light" if key=="dark" else "🌙 Dark")
        for plot in (getattr(self,"live_plot",None),getattr(self,"x_plot",None),getattr(self,"y_plot",None),getattr(self,"mean_x_plot",None),getattr(self,"mean_y_plot",None),getattr(self,"rf_shift_x_plot",None),getattr(self,"rf_shift_y_plot",None),getattr(self,"orm_vv_plot",None),getattr(self,"orm_column_plot",None),getattr(self,"orm_kick_plot",None)):
            if plot is not None: plot.apply_theme()
    def toggle_theme(self): self.apply_theme("light" if self.project.theme=="dark" else "dark")
