"""pyLOCO Measure GUI for structured, read-only measurement acquisition."""
from __future__ import annotations

import re
import csv
from html import escape
from datetime import datetime
from pathlib import Path
from importlib.util import find_spec
from threading import Event
from time import monotonic, sleep
from typing import Sequence

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot, QSize, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
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
    read_bpm_noise_review_annotations, validate_measurement_file, write_bpm_noise, write_bpm_noise_review_annotations,
    write_dispersion, write_orm,
)
from pyLOCO.gui.branding import DISPLAY_ASSET, application_icon, set_asset
from pyLOCO.gui.project_info import (
    PROJECT_ACKNOWLEDGEMENTS, PROJECT_CONTRIBUTORS, PROJECT_DOCUMENTATION,
    PROJECT_ISSUES, PROJECT_LICENSE, PROJECT_PAPER_TITLE, PROJECT_PAPER_URL,
    PROJECT_REPOSITORY, bibtex_text, citation_text,
)
from pyLOCO.gui import __version__ as PYLOCO_VERSION
from pyLOCO.data_schema import SCHEMA_VERSION as MEASUREMENT_SCHEMA_VERSION
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.themes import ACCENTS, accent_for_key, apply_application_theme, theme_for_key
from pyLOCO.gui.suite import launch_suite_application,present_single_about_dialog
from .acquisition import (
    AcquisitionCancelled, BpmDevice, BpmNoiseAcquirer, BpmNoiseResult, CorrectorDevice,
    DispersionResult, DispersionStateAcquirer, ORMAcquirer, ORMResult,
)
from .project import MeasureProject, load_measure_project, save_measure_project
from .automatic import AutomaticDispersionAcquirer
from .dispersion import MOMENTUM_RELATION, physical_dispersion, relative_momentum_deviation
from pyLOCO.reference_model import comparison_metrics, load_reference_model, model_dispersion, model_orm, reference_model_for_pysc, store_reference_model_arrays


MEASURE_COMMON_QSS = """
QToolBar#mainToolbar { spacing: 5px; padding: 6px 8px; }
QToolBar#mainToolbar QPushButton { font-size: 10pt; min-height: 27px; padding: 4px 8px; }
QLabel#measureBrand { font-size: 25pt; font-weight: 850; padding: 1px 9px 1px 2px; }
QLabel#connectionStatus { border-radius: 8px; padding: 4px 9px; font-size: 9.5pt; font-weight: 850; }
QLabel#connectionState { border-radius: 8px; padding: 4px 9px; font-size: 9.5pt; font-weight: 850; }
QLabel#savedPath { border-radius:7px; padding:10px; font-weight:750; }
QFrame#measureLogoContainer { border-radius: 8px; }
QLabel#measureLogo { border: 0; background: transparent; }
QLabel#measurementHelpTitle { font-size: 12pt; font-weight: 750; }
QLabel#measurementConvention { font-family: monospace; font-weight: 650; }
QLabel#planValue { font-weight: 650; }
QLabel#runState { font-size: 14pt; font-weight: 750; }
QLabel#runMetric { font-size: 10.5pt; }
QLabel#summaryCard { border-radius:8px; padding:14px; font-size:11pt; }
QLabel#statLine { border-radius:5px; padding:8px 10px; font-size:10.5pt; }
QFrame#persistentStatus { border-radius:8px; }
QLabel#statusCaption { font-size:8.5pt; font-weight:700; }
QLabel#statusValue { font-size:9.5pt; font-weight:700; }
QTableWidget#statisticsTable { border-radius:6px; }
QTableWidget#statisticsTable::item { padding:5px 9px; }
QLabel#referenceModelCard { border-radius:8px; padding:14px; font-size:10.5pt; }
QLabel#warningText { font-weight:700; }
QLabel#infoText { border-radius:6px; padding:8px 10px; }
QProgressBar { min-height: 34px; }
"""


def measure_stylesheet(theme_key: str) -> str:
    """Measure-specific styling derived from the shared theme and accent tokens."""
    theme = theme_for_key(theme_key)
    accent = accent_for_key("teal")
    primary, hover, soft, pale, title, tint, deep, pressed = accent[1:]
    dark = theme.key == "dark"
    surface = "#34374E" if dark else "#F7F8FB"
    subtle = "#2F3247" if dark else "#EEF1F6"
    border = "#59607A" if dark else "#CBD1DF"
    secondary = "#B8C0D2" if dark else "#4D586D"
    success_bg, success_text, success_border = (("#123D2A", "#7BE3A7", "#38B875") if dark else ("#E8F7EF", "#11643B", "#55A878"))
    danger_bg, danger_text, danger_border = (("#4A2424", "#FFAAAA", "#D96666") if dark else ("#FDECEC", "#9F2929", "#D98686"))
    warning = "#F8C66B" if dark else "#8A4B08"
    text_accent = soft if dark else primary
    group_title = soft if dark else title
    return MEASURE_COMMON_QSS + f"""
QLabel#measureBrand, QLabel#measurementHelpTitle, QLabel#runState {{ color: {text_accent}; }}
QLabel#connectionStatus {{ background:{deep}; color:{pale}; border:1px solid {primary}; }}
QLabel#connectionState[connected="true"] {{ background:{success_bg}; color:{success_text}; border:1px solid {success_border}; }}
QLabel#connectionState[connected="false"] {{ background:{danger_bg}; color:{danger_text}; border:1px solid {danger_border}; }}
QLabel#savedPath {{ background:{tint}; color:{deep}; border:1px solid {primary}; }}
QFrame#measureLogoContainer {{ background:#FFFFFF; border:1px solid {border}; }}
QLabel#summaryCard, QLabel#statLine, QLabel#referenceModelCard {{ background:{subtle}; color:{theme.plot_text}; border:1px solid {border}; }}
QFrame#persistentStatus {{ background:{surface}; border:1px solid {border}; }}
QLabel#statusCaption {{ color:{secondary}; }}
QTableWidget#statisticsTable {{ border:1px solid {border}; gridline-color:{border}; }}
QPushButton#measurePrimary {{ background:{primary}; border-color:{hover}; color:#FFFFFF; }}
QPushButton#measurePrimary:hover {{ background:{hover}; }}
QPushButton#measurePrimary:pressed {{ background:{pressed}; }}
QTabBar::tab:selected {{ border-color:{primary}; }}
QGroupBox::title {{ color:{group_title}; }}
QProgressBar::chunk {{ background:{primary}; }}
QLabel#warningText {{ color:{warning}; }}
QLabel#infoText {{ background:{subtle}; color:{secondary}; border:1px solid {border}; }}
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


class MeasurementStatisticsTable(QTableWidget):
    """Compact, theme-neutral presentation for measurement statistics."""

    def __init__(self) -> None:
        super().__init__(0, 5)
        self.setObjectName("statisticsTable")
        self.setHorizontalHeaderLabels(["Plane / quantity", "Mean", "RMS", "Min", "Max"])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clear_statistics()

    def clear_statistics(self) -> None:
        self.setRowCount(0)
        self.setToolTip("Statistics appear after a completed measurement.")
        self._fit_height_to_contents()

    def _fit_height_to_contents(self) -> None:
        """Fit all rows without scrollbars, including high-DPI frame rounding."""
        self.resizeRowsToContents()
        header_height = self.horizontalHeader().sizeHint().height()
        rows_height = sum(self.verticalHeader().sectionSize(row) for row in range(self.rowCount()))
        self.setFixedHeight(header_height + rows_height + 2 * self.frameWidth() + 10)

    def set_statistics(self, rows, unit: str) -> None:
        self.setHorizontalHeaderLabels([
            "Plane / quantity", f"Mean [{unit}]", f"RMS [{unit}]", f"Min [{unit}]", f"Max [{unit}]",
        ])
        self.setRowCount(len(rows))
        colors = ("#11B7C1", "#F59E42")
        for row, (label, values) in enumerate(rows):
            label_item = QTableWidgetItem(str(label))
            label_item.setForeground(QColor(colors[row % len(colors)]))
            font = label_item.font(); font.setBold(True); label_item.setFont(font)
            self.setItem(row, 0, label_item)
            for column, key in enumerate(("Mean", "RMS", "Min", "Max"), start=1):
                item = QTableWidgetItem(f"{float(values[key]):.3f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.setItem(row, column, item)
        self._fit_height_to_contents()
        self.setToolTip(f"Measurement statistics in {unit}")


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

    def __init__(self, acquirer, label, requested_rf_hz, actual_rf_hz, actual_rf_source, rf_readback_verified, readings, delay, settling, cancel_event):
        super().__init__(); self.acquirer=acquirer; self.label=label; self.requested_rf_hz=requested_rf_hz; self.actual_rf_hz=actual_rf_hz; self.actual_rf_source=actual_rf_source; self.rf_readback_verified=rf_readback_verified; self.readings=readings; self.delay=delay; self.settling=settling; self.cancel_event=cancel_event

    @Slot()
    def run(self):
        try:
            if self.settling and self.cancel_event.wait(self.settling):
                raise AcquisitionCancelled("Dispersion acquisition was cancelled during RF settling")
            result = self.acquirer.acquire(
                self.label, self.requested_rf_hz, self.readings, self.delay,
                actual_rf_hz=self.actual_rf_hz, actual_rf_source=self.actual_rf_source,
                rf_readback_verified=self.rf_readback_verified,
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
        self.setWindowTitle("pyLOCO Measure"); self.setWindowIcon(application_icon("measure"))
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
        self._manual_restoration_pending = False
        self._manual_nominal_source = "manual_entry_unverified"
        self._manual_restoration_source = "not_confirmed"
        self._manual_restoration_verified = False
        self._live_dispersion_reference = None
        self._live_dispersion_plot_state = None
        self._dispersion_selected_bpm_index = None
        self._dispersion_pinned_bpm_indices = set()
        self.saved_measurement_path: Path | None = None
        self.saved_session_path: Path | None = None
        self.reference_model = None

        # Every newly opened Measure window starts a new logical
        # measurement session. Existing files on disk are preserved,
        # but their roles must not leak into this session.
        self._fresh_measurement_session = True
        self.cancel_event = Event(); self.thread: QThread | None = None; self.worker = None
        self.connection_verified = True
        self._build_ui(); self._measurement_type_changed(); self._update_machine_identity()
        active_theme=QApplication.instance().property("pyLOCOTheme") if QApplication.instance() is not None else None
        self.apply_theme(str(active_theme or self.project.theme)); self.refresh_preview(); self.refresh_plan()

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
        self.theme_button=QPushButton("☀ Light"); self.theme_button.clicked.connect(self.toggle_theme); toolbar.addWidget(self.theme_button)
        logo_container=QFrame(); logo_container.setObjectName("measureLogoContainer"); logo_layout=QHBoxLayout(logo_container); logo_layout.setContentsMargins(4,2,4,2)
        self.logo_button=ClickableLogoLabel(); set_asset(self.logo_button,QSize(107,44),DISPLAY_ASSET,crop_transparency=False); self.logo_button.clicked.connect(self._show_about_dialog); logo_layout.addWidget(self.logo_button); toolbar.addWidget(logo_container)

        root=QWidget(); layout=QVBoxLayout(root); layout.setContentsMargins(20,12,20,20); layout.setSpacing(9)
        status_frame=QFrame(); status_frame.setObjectName("persistentStatus")
        status_layout=QGridLayout(status_frame); status_layout.setContentsMargins(10,7,10,7); status_layout.setHorizontalSpacing(14); status_layout.setVerticalSpacing(3)
        self.status_badge=QLabel("MOCK • READ ONLY"); self.status_badge.setObjectName("connectionStatus")
        self.connection_badge=QLabel("● OFFLINE READY"); self.connection_badge.setObjectName("connectionState"); self.connection_badge.setProperty("connected",True)
        self.machine_identity_badge=QLabel("Mock / offline"); self.machine_identity_badge.setObjectName("statusValue")
        self.header_mode=QLabel("Mode · Offline"); self.header_mode.setObjectName("statusValue")
        self.header_capabilities=QLabel("Available · Read Only"); self.header_capabilities.setObjectName("statusValue")
        self.header_access=QLabel("Authorized · Read Only"); self.header_access.setObjectName("statusValue")
        for column,widget in enumerate((self.status_badge,self.header_mode,self.connection_badge,self.header_capabilities,self.header_access)):
            widget.setWordWrap(False)
            status_layout.addWidget(widget,0,column)
            status_layout.setColumnStretch(column,1 if column in (3,4) else 0)
        machine_caption=QLabel("MACHINE"); machine_caption.setObjectName("statusCaption")
        config_caption=QLabel("CONFIGURATION"); config_caption.setObjectName("statusCaption")
        output_caption=QLabel("OUTPUT"); output_caption.setObjectName("statusCaption")
        self.header_configuration=ElidedPathLabel(); self.header_configuration.setObjectName("statusValue")
        self.header_output=ElidedPathLabel(); self.header_output.setObjectName("statusValue")
        status_layout.addWidget(machine_caption,1,0); status_layout.addWidget(self.machine_identity_badge,1,1)
        status_layout.addWidget(config_caption,1,2); status_layout.addWidget(self.header_configuration,1,3)
        status_layout.addWidget(output_caption,1,4); status_layout.addWidget(self.header_output,1,5)
        status_layout.setColumnStretch(1,1); status_layout.setColumnStretch(3,2); status_layout.setColumnStretch(5,2)
        layout.addWidget(status_frame)
        self.heading=QLabel("Machine setup"); self.heading.setObjectName("pageTitle"); layout.addWidget(self.heading)
        self.subtitle=QLabel("Offline planning with deterministic Mock data — no machine writes."); self.subtitle.setVisible(False)
        self.tabs=QTabWidget(); self.tabs.currentChanged.connect(self._workspace_changed); layout.addWidget(self.tabs,1)
        self.tabs.addTab(self._machine_page(),"Machine")
        self.tabs.addTab(self._bpms_page(),"Devices")
        configuration=self._measurement_page(); acquisition=self._review_page()
        measurement=QSplitter(Qt.Horizontal); self.measurement_splitter=measurement
        measurement.setChildrenCollapsible(False); measurement.setHandleWidth(12)
        measurement.setStyleSheet("QSplitter::handle { background:#18B8BE; margin:1px 3px; border-radius:3px; } QSplitter::handle:hover { background:#58D8DE; }")
        # Scroll areas otherwise advertise their content's wide size hint and make
        # one pane appear impossible to shrink.  The scroll contents still lay out
        # normally; only the splitter is allowed to negotiate compact pane widths.
        # Keep both panes useful while dragging: the configuration side must
        # retain enough width for wrapped help/form text, and the result side
        # must retain enough width for plot controls and axes labels.
        configuration.setMinimumWidth(300)
        acquisition.setMinimumWidth(360)
        for pane in (configuration,acquisition):
            pane.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        measurement.addWidget(configuration); measurement.addWidget(acquisition)
        measurement.setStretchFactor(0,2); measurement.setStretchFactor(1,3); measurement.setSizes([440,760])
        self.tabs.addTab(measurement,"Measurement")
        saved=QWidget(); saved_layout=QVBoxLayout(saved); saved_layout.setContentsMargins(28,28,28,28)
        saved_title=QLabel("Completed measurement"); saved_title.setObjectName("pageTitle"); saved_layout.addWidget(saved_title)
        self.review_machine_identity=QLabel(); self.review_machine_identity.setObjectName("planValue"); self.review_machine_identity.setWordWrap(True); saved_layout.addWidget(self.review_machine_identity)
        review_group=QGroupBox("Review"); review_layout=QVBoxLayout(review_group)
        self.review_status=QLabel("No completed measurement yet."); self.review_status.setObjectName("runState"); self.review_status.setWordWrap(True)
        self.review_details=QLabel("Complete a measurement to review its saved data and validation status."); self.review_details.setObjectName("summaryCard"); self.review_details.setWordWrap(True); self.review_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.review_paths=QLabel("Measurement file: —\nSession manifest: —"); self.review_paths.setWordWrap(True); self.review_paths.setTextInteractionFlags(Qt.TextSelectableByMouse)
        review_layout.addWidget(self.review_status); review_layout.addWidget(self.review_details); review_layout.addWidget(self.review_paths); saved_layout.addWidget(review_group)
        review_actions=QHBoxLayout(); self.review_validate_button=QPushButton("Validate for pyLOCO"); self.review_validate_button.setEnabled(False); self.review_validate_button.clicked.connect(self.validate_saved); self.review_open_button=QPushButton("Open session in pyLOCO Fit"); self.review_open_button.setEnabled(False); self.review_open_button.clicked.connect(self.explain_open); back_button=QPushButton("Back to results"); back_button.clicked.connect(lambda:self.tabs.setCurrentIndex(2)); review_actions.addWidget(self.review_validate_button); review_actions.addWidget(self.review_open_button); review_actions.addWidget(back_button); review_actions.addStretch(1); saved_layout.addLayout(review_actions); saved_layout.addStretch()
        self.tabs.addTab(saved,"Review && Save")
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.setCentralWidget(root); self.statusBar().showMessage("Mock ready — read only"); self._update_workflow_tabs()

    def _scroll_page(self, content):
        # The scroll viewport, not the content's preferred height, should
        # determine the visible pane height. This prevents content-heavy
        # modes such as Dispersion from making the Measurement workspace
        # appear much taller than BPM Noise.
        content.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Ignored,
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(content)
        return scroll

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
        self._update_persistent_header()
        if hasattr(self,"plan_values"):self.refresh_plan()

    def _update_persistent_header(self) -> None:
        if not hasattr(self, "header_configuration") or not hasattr(self, "adapter_combo"):
            return
        key = self.adapter_combo.currentData()
        descriptor = next((item for item in InterfaceRegistry.DESCRIPTORS if item.key == key), None)
        mode = descriptor.environment.title() if descriptor is not None else "Live"
        if key == "pyaml" and hasattr(self, "pyaml_fields"):
            mode = self.pyaml_fields["mode"].text().strip().title() or "Live"
        capabilities = self.adapter.capabilities
        if key == "pyaml" and hasattr(self, "pyaml_fields"):
            magnet_available = bool(self.pyaml_fields["hcor_group"].text().strip() or self.pyaml_fields["vcor_group"].text().strip())
            rf_available = bool(self.pyaml_fields["rf_name"].text().strip())
        elif key == "pysc":
            magnet_available = rf_available = True
        else:
            magnet_available = rf_available = False

        def access_text(magnet: bool, rf: bool) -> str:
            if magnet and rf:return "Magnet + RF"
            if magnet:return "Magnet"
            if rf:return "RF"
            return "Read Only"

        magnet_authorized = self.connection_verified and AdapterCapability.WRITE in capabilities
        rf_authorized = self.connection_verified and AdapterCapability.RF_WRITE in capabilities
        available = access_text(magnet_available, rf_available)
        authorized = access_text(magnet_authorized, rf_authorized)
        self.header_mode.setText(f"Mode · {mode}")
        self.header_capabilities.setText(f"Available · {available}")
        self.header_access.setText(f"Authorized · {authorized}")
        self.header_capabilities.setToolTip("Backend functions exposed by the selected configuration; this does not grant permission to write.")
        self.header_access.setToolTip("Write access currently granted to this connected adapter after confirmation. Disconnected backends are read only.")
        identity, profile = self._machine_identity()
        pyaml_path = self.pyaml_fields["config_path"].text().strip() if key == "pyaml" and hasattr(self, "pyaml_fields") else ""
        machine_display = identity.replace("\n", " · ")
        if key == "pyaml":
            machine_display = f"pyAML · {mode}" + (f" · {Path(pyaml_path).name}" if pyaml_path else "")
        self.machine_identity_badge.setText(machine_display)
        self.machine_identity_badge.setToolTip(identity)
        if pyaml_path:
            configuration = pyaml_path
        elif self.project_path is not None:
            configuration = str(self.project_path)
        elif profile is not None:
            configuration = str(profile.manifest_path)
        else:
            configuration = "Unsaved Measure project"
        self.header_configuration.setFullText(configuration)
        self.header_output.setFullText(str(self._resolved_output_directory()) if hasattr(self, "output_directory") else "—")

    def _set_acquisition_running(self, running: bool) -> None:
        # Keep the active Measurement workspace visible throughout acquisition.
        # Disabling its tab makes QTabWidget automatically navigate to Review & Save.
        for index in range(min(2,self.tabs.count())):self.tabs.setTabEnabled(index,not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.cancel_button.setText("Cancel measurement" if running else "Cancel")
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
        layout.addWidget(self.machine_group)
        self.pyaml_group = QGroupBox("pyAML configuration")
        pyaml_form = QFormLayout(self.pyaml_group)
        self.pyaml_fields = {}
        for field, default in (("config_path", ""), ("mode", "live"), ("bpm_group", "BPM"), ("hcor_group", "HCorr"), ("vcor_group", "VCorr"), ("rf_name", "RF")):
            editor = QLineEdit(default)
            self.pyaml_fields[field] = editor
            if field == "config_path":
                config_row = QWidget()
                config_layout = QHBoxLayout(config_row)
                config_layout.setContentsMargins(0, 0, 0, 0)
                config_layout.addWidget(editor, 1)
                self.pyaml_browse_button = QPushButton("Browse…")
                self.pyaml_browse_button.clicked.connect(self._choose_pyaml_config)
                config_layout.addWidget(self.pyaml_browse_button)
                pyaml_form.addRow("Configuration file", config_row)
            else:
                pyaml_form.addRow(field.replace("_", " ").title(), editor)
            editor.textChanged.connect(self._invalidate_pyaml)
        self.pyaml_magnet_writes = QCheckBox("Enable corrector magnet writes (ORM)")
        self.pyaml_rf_writes = QCheckBox("Enable RF frequency writes (dispersion)")
        for checkbox in (self.pyaml_magnet_writes, self.pyaml_rf_writes):
            checkbox.setChecked(True)
            pyaml_form.addRow(checkbox)
            checkbox.toggled.connect(self._invalidate_pyaml)
        pyaml_form.addRow(QLabel("Mode: design or live. RF plant defaults to RF. Writes default to enabled; uncheck to disable. Live connections require confirmation."))
        self.pyaml_group.setVisible(False)
        layout.addWidget(self.pyaml_group)
        reference_group=QGroupBox("Optional reference model — requires Accelerator Toolbox (AT)"); reference_layout=QFormLayout(reference_group); self._configure_form(reference_layout)
        self.reference_model_path=QLineEdit(); self.reference_model_path.setPlaceholderText("Automatic from pySC profile, or select an authoritative nominal .mat for LIVE")
        reference_row=QWidget(); rr=QHBoxLayout(reference_row); rr.setContentsMargins(0,0,0,0); rr.addWidget(self.reference_model_path,1); choose=QPushButton("Choose .mat…"); choose.clicked.connect(self._choose_reference_model); rr.addWidget(choose)
        self.reference_model_summary=QLabel("No reference model loaded."); self.reference_model_summary.setObjectName("referenceModelCard"); self.reference_model_summary.setWordWrap(True); self.reference_model_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reference_model_requirement=QLabel("AT is installed — reference-model calculations are available." if find_spec("at") is not None else "AT is not installed — reference-model calculations and physical dispersion are unavailable."); self.reference_model_requirement.setWordWrap(True)
        self.compare_reference=QCheckBox("Compare measurements with reference model (diagnostic only)"); self.compare_reference.setChecked(True)
        reference_layout.addRow("Requirement",self.reference_model_requirement); reference_layout.addRow("Reference lattice",reference_row); reference_layout.addRow("MODEL — not live machine data",self.reference_model_summary); reference_layout.addRow("",self.compare_reference)
        layout.addWidget(reference_group); return self._scroll_page(content)

    def _choose_reference_model(self):
        path=QFileDialog.getOpenFileName(self,"Select authoritative reference lattice","","AT lattice (*.mat);;All files (*)")[0]
        if path:self.reference_model_path.setText(path); self._load_reference_model(path,"User-selected reference model")

    def _load_reference_model(self,path,source,mat_key=None):
        self._model_orm_cache=None
        try:self.reference_model=load_reference_model(path,source=source,mat_key=mat_key)
        except Exception as exc:self.reference_model=None; self.reference_model_summary.setText(f"Reference model unavailable: {exc}"); return
        self._display_reference_model()

    def _display_reference_model(self):
        model=self.reference_model; self.reference_model_path.setText(str(model.path))
        harmonic=str(model.harmonic_number) if model.harmonic_number is not None else "Not available"
        rf=f"{model.nominal_rf_hz/1e6:.9f} MHz" if model.nominal_rf_hz is not None else "Not available"
        residual=f"{model.rf_harmonic_residual_hz:.6g} Hz" if model.rf_harmonic_residual_hz is not None else "Not available"
        rows=(
            ("Fractional tunes Qx / Qy",f"{model.tune[0]:.6f} / {model.tune[1]:.6f}","AT linear optics"),
            ("Chromaticity ξx / ξy",f"{model.chromaticity[0]:.6f} / {model.chromaticity[1]:.6f}","AT linear optics"),
            ("Beam energy",f"{model.energy_ev/1e9:.6g} GeV","Reference lattice"),
            ("Circumference",f"{model.circumference_m:.6f} m","AT lattice geometry"),
            ("Revolution frequency f<sub>rev</sub>",f"{model.revolution_frequency_hz:.6f} Hz","c / circumference"),
            ("Harmonic number h",harmonic,model.harmonic_number_source),
            ("Model RF frequency",rf,model.rf_source),
            ("RF consistency: f<sub>RF</sub> − h·f<sub>rev</sub>",residual,"AT/model consistency check"),
            ("Momentum compaction α<sub>c</sub>",f"{model.momentum_compaction:.9g}","AT ring.get_mcf()"),
            ("Relativistic term 1/γ²",f"{model.inverse_gamma_squared:.9g}","AT lattice energy/slip factor"),
            ("Slip factor η = α<sub>c</sub> − 1/γ²",f"{model.slip_factor:.9g}","AT ring.get_slip_factor() convention"),
        )
        body="".join(f"<tr><td><b>{name}</b></td><td>{value}</td><td>{source}</td></tr>" for name,value,source in rows)
        selection=("Selected automatically by the pySC profile manifest" if str(model.source).startswith("pySC profile manifest") else escape(str(model.source)))
        self.reference_model_summary.setText(
            f"<h3 style='color:#12BFC4;margin:0 0 8px 0'>Reference-lattice quantities</h3>"
            f"<div><b>Calculation source:</b> Accelerator Toolbox, using the loaded 4D reference lattice.</div>"
            f"<div><b>Lattice:</b> {escape(model.path.name)} &nbsp; <b>SHA-256:</b> {escape(model.checksum_sha256[:12])}…</div>"
            f"<div><b>File selection:</b> {selection}</div><br>"
            f"<table cellspacing='0' cellpadding='4' width='100%'><tr><th align='left'>Quantity</th><th align='left'>Value</th><th align='left'>Source</th></tr>{body}</table>")

    def _load_profile_reference_model(self):
        self._model_orm_cache=None
        try:self.reference_model=reference_model_for_pysc(self.pysc_profile_combo.currentData())
        except Exception as exc:self.reference_model=None; self.reference_model_summary.setText(f"Automatic profile reference model unavailable: {exc}"); return
        self._display_reference_model()

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
        self.pyaml_group.setVisible(key == "pyaml")
        self.reference_model=None
        if hasattr(self,"reference_model_path"):self.reference_model_path.clear(); self.reference_model_summary.setText("Not available — select an authoritative reference lattice." if petra else "No reference model loaded.")
        self._model_orm_cache=None
        for widget in self.pysc_profile_row: widget.setVisible(pysc)
        if key == "pyaml":
            self._invalidate_pyaml()
            self.measurement_type.setCurrentIndex(self.measurement_type.findData("bpm_noise"))
            if self.measurement_label.text().startswith("pyLOCO BPM noise — Mock"):
                self.measurement_label.setText("pyLOCO BPM noise — pyAML")
            self.measurement_type.model().item(self.measurement_type.findData("orm")).setEnabled(False)
            self.orm_unavailable_label.setText("ORM needs write access. pyAML is read only.")
            self.orm_unavailable_label.setVisible(True)
            return
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
                self.status_badge.setText("DEMO • pySC SERVER"); self._set_connection_state(False,"DISCONNECTED"); self.subtitle.setText(f"pySC Server {profile_label} simulation selected — connect to measure."); self.machine_info["adapter"].setText(f"pySC Server — {profile_label} simulation"); self.machine_info["connection"].setText(f"Unavailable: {exc}"); self.machine_info["access"].setText("DEMO — connect before measuring"); self.machine_info["bpm_orbit"].setText("unavailable"); self.machine_info["corrector_readback"].setText("unavailable"); self.machine_info["calibration"].setText("backend managed"); self.machine_info["rf_readback"].setText("unavailable");
                if hasattr(self,"start_button"):self.start_button.setEnabled(False)
                return
            self.adapter=session.adapter; self.devices=tuple(BpmDevice(**item) for item in self.adapter.list_devices("bpm")); self.horizontal_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("hcor")); self.vertical_correctors=tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("vcor"))
            self.status_badge.setText(session.badge); self._set_connection_state(False,"DISCONNECTED"); self.subtitle.setText(f"DEMO MODE — pySC Server {profile_label} simulation. Temporary measurement writes are restoration-protected."); self.machine_info["adapter"].setText(f"pySC Server — {profile_label} simulation"); self.machine_info["connection"].setText("Configured — use Test connection"); self.machine_info["access"].setText("DEMO — temporary writes are restoration-protected"); self.machine_info["bpm_orbit"].setText("not tested"); self.machine_info["corrector_readback"].setText("not tested"); self.machine_info["calibration"].setText("backend managed"); self.machine_info["rf_readback"].setText("not tested")
        else:
            self.devices=default_mock_devices(); self.horizontal_correctors,self.vertical_correctors=default_mock_correctors(); self.adapter=build_mock_adapter(self.devices,horizontal_correctors=self.horizontal_correctors,vertical_correctors=self.vertical_correctors)
            self.status_badge.setText("MOCK • READ ONLY"); self._set_connection_state(True,"OFFLINE READY"); self.subtitle.setText("Offline planning with deterministic Mock data — no machine writes."); self.machine_info["adapter"].setText("Mock — deterministic offline data source"); self.machine_info["connection"].setText("Offline simulation"); self.machine_info["access"].setText("Read only — no real machine writes"); self.machine_info["bpm_orbit"].setText(f"available — {len(self.devices)} mock BPM devices"); self.machine_info["corrector_readback"].setText("simulated diagnostics available"); self.machine_info["calibration"].setText("not applicable in Mock mode"); self.machine_info["rf_readback"].setText("simulated internally")
        self.selected_devices=self.devices; self.selected_hcorrectors=self.horizontal_correctors; self.selected_vcorrectors=self.vertical_correctors
        for key,devices in (("hcor",self.horizontal_correctors),("vcor",self.vertical_correctors)):
            if hasattr(self,"corrector_selection_widgets"):self.corrector_selection_widgets[key]["devices"]=devices; self._refresh_corrector_preview(key,read_diagnostics=False)
        if hasattr(self,"measurement_type"):
            orm_item=self.measurement_type.model().item(self.measurement_type.findData("orm")); orm_item.setEnabled(not petra); self.orm_unavailable_label.setText("ORM needs write access. PETRA is read only."); self.orm_unavailable_label.setVisible(petra)
            if petra and self.measurement_type.currentData()=="orm":self.measurement_type.setCurrentIndex(self.measurement_type.findData("bpm_noise"))
            automatic=self.rf_control_mode.model().item(self.rf_control_mode.findData("automatic")); automatic.setEnabled(AdapterCapability.RF_WRITE in self.adapter.capabilities)
            if pysc: self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("automatic"))
            current_label=self.measurement_label.text()
            if current_label.startswith(("Mock", "pyLOCO")) or " pySC " in current_label:
                context=profile_label if pysc else "PETRA III LIVE" if petra else "pyAML" if self.adapter_combo.currentData()=="pyaml" else "Mock"
                labels={"bpm_noise":f"pyLOCO BPM noise — {context}","dispersion":f"pyLOCO dispersion — {context}","orm":f"pyLOCO ORM — {context}"}; self.measurement_label.setText(labels[self.measurement_type.currentData()])
        self._reset_selection_states_for_inventory(); self._update_machine_identity()
        self.refresh_preview(); self.statusBar().showMessage("PETRA adapter selected — not connected" if petra else "pySC Server ready — test connection" if pysc else "Mock ready — read only"); self._sync_rf_presentation()

    def _machine_identity(self):
        key=self.adapter_combo.currentData() if hasattr(self,"adapter_combo") else "mock"
        if key == "pyaml":
            return f"pyAML / {self.pyaml_fields['mode'].text()} / {self.pyaml_fields['config_path'].text()}", None
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
        self._update_persistent_header()

    def _pysc_profile_changed(self,*_):
        if self.adapter_combo.currentData()=="pysc": self._adapter_changed()

    def _sync_rf_presentation(self, *_):
        if not hasattr(self,"rf_control_mode"):return
        automatic_available=self.adapter_combo.currentData() in ("pysc", "pyaml") and AdapterCapability.RF_WRITE in self.adapter.capabilities
        self.rf_control_mode.model().item(self.rf_control_mode.findData("automatic")).setEnabled(automatic_available)
        if self.rf_control_mode.currentData()=="automatic" and not automatic_available:
            self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("manual"))
        automatic=self.rf_control_mode.currentData()=="automatic"
        self.rf_control_mode.setEnabled(automatic_available); self.nominal_rf.setReadOnly(automatic)
        self.manual_actual_rf.setVisible(not automatic)
        self.verify_restored_orbit.setEnabled(True)
        if automatic:
            self.nominal_rf_source.setText("Source: pySC Server RF readback — automatically controlled")
            self.rf_safety.setText("DEMO automatic RF control — the original frequency is restored and read back in cancellation-safe cleanup.")
            if self.adapter_combo.currentData() == "pyaml":
                self.nominal_rf_source.setText("Source: pyAML RF frequency readback — automatically controlled")
                self.rf_safety.setText("WRITES ENABLED — RF frequency is changed during measurement, then restoration is attempted and verified. Check the restoration result.")
        else:
            has_readback=AdapterCapability.RF_READ in self.adapter.capabilities
            self.nominal_rf_source.setText("Source: backend RF readback (manual control; no writes)" if has_readback else "Source: operator entry (not independently verified)")
            self.rf_safety.setText("MANUAL / READ ONLY — change RF externally. pyLOCO will read or record the actual frequency, acquire orbits, and never issue an RF write, including during cancellation or errors.")
        bipolar=self.dispersion_direction.currentData()=="bipolar"
        self.rf_step_label.setText("Total RF separation Δ" if bipolar else "RF offset Δf")
        if hasattr(self,"rf_states_explanation"):
            half=self.rf_step.value()/2
            self.rf_states_explanation.setText(f"Applied states: f₀ + {half:g} Hz / f₀ − {half:g} Hz\nTotal separation: Δ = {self.rf_step.value():g} Hz" if bipolar else "One-sided RF offset")
        self._configure_physical_dispersion_option()
        if hasattr(self,"plan_values"):self.refresh_plan()

    def _rf_mode_changed(self, *_):
        self._sync_rf_presentation(); self._update_dispersion_guide(); self.refresh_plan()

    def _choose_pyaml_config(self):
        path = QFileDialog.getOpenFileName(
            self, "Select pyAML configuration", self.pyaml_fields["config_path"].text(),
            "pyAML configuration (*.yaml *.yml *.json);;All files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if path:
            self.pyaml_fields["config_path"].setText(path)

    def _invalidate_pyaml(self, *_):
        if self.adapter_combo.currentData() != "pyaml":
            return
        self.adapter = MockAdapter()
        self.devices = self.horizontal_correctors = self.vertical_correctors = ()
        self.selected_devices = self.selected_hcorrectors = self.selected_vcorrectors = ()
        self.status_badge.setText("pyAML • READ ONLY")
        if hasattr(self, "measurement_type"):
            self.measurement_type.model().item(self.measurement_type.findData("orm")).setEnabled(False)
            self.orm_unavailable_label.setVisible(True)
        self.subtitle.setText("pyAML configuration selected — test connection before acquiring data.")
        self._set_connection_state(False, "DISCONNECTED")
        for key, label in self.machine_info.items():
            label.setText("Not connected")
        self.machine_info["adapter"].setText("pyAML")
        self.machine_info["access"].setText("Read only")
        if hasattr(self, "corrector_selection_widgets"):
            for kind in ("hcor", "vcor"):
                self.corrector_selection_widgets[kind]["devices"] = ()
                self._refresh_corrector_preview(kind, read_diagnostics=False)
        self._reset_selection_states_for_inventory()
        self._update_machine_identity()
        self.refresh_preview()
        self._sync_rf_presentation()

    def _connect_pyaml(self):
        self._invalidate_pyaml()
        options = {key: editor.text().strip() for key, editor in self.pyaml_fields.items()}
        options.update(allow_writes=self.pyaml_magnet_writes.isChecked(),
                       allow_rf_writes=self.pyaml_rf_writes.isChecked(),
                       corrector_values_in_radians=True)
        if options["mode"] == "live" and (options["allow_writes"] or options["allow_rf_writes"]):
            answer = QMessageBox.question(self, "Enable live pyAML writes?",
                "Measurements may change configured corrector magnets and/or RF frequency. "
                "Verify the configuration, units, machine limits and operating permission. "
                "Restoration is attempted after measurement but can fail. Enable selected writes?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        try:
            session = InterfaceRegistry(pyaml_options=options).create("pyaml")
            result = session.adapter.test_connection()
        except Exception as exc:
            self.machine_info["connection"].setText(str(exc))
            QMessageBox.warning(self, "pyAML connection unavailable", str(exc))
            return
        self.adapter = session.adapter
        self.devices = tuple(BpmDevice(**item) for item in self.adapter.list_devices("bpm"))
        self.horizontal_correctors = tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("hcor"))
        self.vertical_correctors = tuple(CorrectorDevice(**item) for item in self.adapter.list_devices("vcor"))
        self.selected_devices = self.devices
        self.selected_hcorrectors = self.horizontal_correctors
        self.selected_vcorrectors = self.vertical_correctors
        for kind, devices in (("hcor", self.horizontal_correctors), ("vcor", self.vertical_correctors)):
            self.corrector_selection_widgets[kind]["devices"] = devices
            self._refresh_corrector_preview(kind, read_diagnostics=False)
        self.project.metadata["pyaml_options"] = {key: value for key, value in options.items() if key not in ("allow_writes", "allow_rf_writes")}
        self._set_connection_state(True, "CONNECTED")
        self.machine_info["connection"].setText("Connected — read only")
        self.machine_info["bpm_orbit"].setText(f"{result['bpms']} BPMs")
        for key in ("corrector_readback", "calibration", "rf_readback"):
            self.machine_info[key].setText(str(result[key]))
        writable = AdapterCapability.WRITE in self.adapter.capabilities
        rf_writable = AdapterCapability.RF_WRITE in self.adapter.capabilities
        if rf_writable:
            self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("automatic"))
        access = f"Correctors: {'WRITE' if writable else 'read only'}; RF: {'WRITE' if rf_writable else 'read only'}"
        self.machine_info["access"].setText(access)
        self.status_badge.setText("pyAML • WRITES ENABLED" if writable or rf_writable else "pyAML • READ ONLY")
        self.subtitle.setText(f"pyAML {options['mode']} connected — {access}")
        self.measurement_type.model().item(self.measurement_type.findData("orm")).setEnabled(writable)
        self.orm_unavailable_label.setVisible(not writable)
        if AdapterCapability.RF_READ in self.adapter.capabilities:
            self.nominal_rf.setText(f"{result['rf_readback']:.12f}")
        self._sync_rf_presentation()
        self._reset_selection_states_for_inventory()
        self._update_machine_identity()
        self.refresh_preview()

    def _test_connection(self):
        if self.adapter_combo.currentData() == "pyaml":
            self._connect_pyaml()
            return
        if self.adapter_combo.currentData()=="pysc":
            try:self._adapter_changed(); result=self.adapter.test_connection()
            except Exception as exc:self._set_connection_state(False,"DISCONNECTED"); QMessageBox.warning(self,"pySC Server unavailable",str(exc)); return
            profile=self.pysc_profile_combo.currentText(); self._set_connection_state(True,"CONNECTED"); self.machine_info["connection"].setText(f"Connected — {profile}"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(str(result["corrector_readback"])); self.machine_info["rf_readback"].setText(f"available — {result['rf_readback']:.6f} Hz"); self.nominal_rf.setText(f"{result['rf_readback']:.12f}"); self._load_profile_reference_model(); self.statusBar().showMessage(f"CONNECTED • {profile} • {result['bpms']} BPMs"); return
        if not isinstance(self.adapter,PETRAReadOnlyAdapter):
            self.statusBar().showMessage("Mock ready — no external connection"); return
        self.test_connection_button.setEnabled(False); self.machine_info["connection"].setText("Testing safe reads…"); QApplication.processEvents()
        try:result=self.adapter.test_connection()
        except Exception as exc:
            self.machine_info["connection"].setText("Unavailable"); self.machine_info["bpm_orbit"].setText("unavailable"); self.machine_info["corrector_readback"].setText("unavailable"); QMessageBox.warning(self,"PETRA connection unavailable",str(exc))
        else:
            self.machine_info["connection"].setText("Connected — read only"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(result["corrector_readback"]); self.machine_info["calibration"].setText(result["calibration"]); self.machine_info["rf_readback"].setText(result["rf_readback"]); self.statusBar().showMessage(f"CONNECTED • PETRA read only • {result['bpms']} BPMs")
        finally:self.test_connection_button.setEnabled(True)

    def _measurement_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.measurement_machine_identity=QLabel(); self.measurement_machine_identity.setObjectName("planValue"); self.measurement_machine_identity.setWordWrap(True); layout.addWidget(self.measurement_machine_identity)
        type_group=QGroupBox("Measurement type"); type_form=QFormLayout(type_group); self._configure_form(type_form); self.measurement_type=QComboBox(); self.measurement_type.addItem("BPM Noise","bpm_noise"); self.measurement_type.addItem("Dispersion","dispersion"); self.measurement_type.addItem("ORM","orm"); self.measurement_type.currentIndexChanged.connect(self._measurement_type_changed); type_form.addRow("Measurement",self.measurement_type); self.orm_unavailable_label=QLabel("ORM needs write access. PETRA is read only."); self.orm_unavailable_label.setWordWrap(True); type_form.addRow("",self.orm_unavailable_label); self.orm_unavailable_label.setVisible(False); layout.addWidget(type_group)
        self.measurement_help_group=QGroupBox("ⓘ About this measurement")
        self.measurement_help_group.setCheckable(True); self.measurement_help_group.setChecked(False)
        help_layout=QVBoxLayout(self.measurement_help_group); self.measurement_help_body=QWidget(); help_body_layout=QVBoxLayout(self.measurement_help_body); help_body_layout.setContentsMargins(2,4,2,4)
        self.measurement_help_title=QLabel(); self.measurement_help_title.setObjectName("measurementHelpTitle")
        self.measurement_help_text=QLabel(); self.measurement_help_text.setWordWrap(True); self.measurement_help_text.setMinimumWidth(0); self.measurement_help_text.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred); self.measurement_help_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.measurement_help_convention=QLabel(); self.measurement_help_convention.setObjectName("measurementConvention"); self.measurement_help_convention.setWordWrap(True); self.measurement_help_convention.setTextInteractionFlags(Qt.TextSelectableByMouse)
        help_body_layout.addWidget(self.measurement_help_title); help_body_layout.addWidget(self.measurement_help_text); help_body_layout.addWidget(self.measurement_help_convention)
        help_layout.addWidget(self.measurement_help_body); self.measurement_help_group.toggled.connect(self.measurement_help_body.setVisible); self.measurement_help_body.setVisible(False); layout.addWidget(self.measurement_help_group)
        group=QGroupBox("Measurement configuration"); group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); form=QFormLayout(group); self._configure_form(form)
        self.readings=NoWheelSpinBox(); self.readings.setRange(2,100000); self.readings.setValue(20)
        self.delay=NoWheelDoubleSpinBox(); self.delay.setRange(0,3600); self.delay.setDecimals(3); self.delay.setSingleStep(.1); self.delay.setSuffix(" s"); self.delay.setValue(1.0)
        self.bpm_noise_convention=QComboBox(); self.bpm_noise_convention.addItem("Population standard deviation (ddof=0) — default",0); self.bpm_noise_convention.addItem("Sample standard deviation (ddof=1)",1)
        self.measurement_name=QLineEdit("pyloco-bpm-noise"); self.measurement_label=QLineEdit("pyLOCO BPM noise — Mock"); self.comments=QPlainTextEdit(); self.comments.setMaximumHeight(90)
        self.output_directory=QLineEdit("measurements"); out=QWidget(); ol=QHBoxLayout(out); ol.setContentsMargins(0,0,0,0); ol.addWidget(self.output_directory,1); choose=QPushButton("Choose…"); choose.clicked.connect(self.choose_output); ol.addWidget(choose)
        self.duration=QLabel();
        for label,widget in (("Number of orbit readings",self.readings),("Delay between readings",self.delay),("BPM-noise convention",self.bpm_noise_convention),("Estimated duration",self.duration),("Measurement name",self.measurement_name),("Measurement label",self.measurement_label),("Notes",self.comments),("Output/session directory",out)): form.addRow(label,widget)
        self.bpm_noise_convention_label=form.labelForField(self.bpm_noise_convention)
        self.readings.valueChanged.connect(self.refresh_plan); self.delay.valueChanged.connect(self.refresh_plan); self.bpm_noise_convention.currentIndexChanged.connect(self.refresh_plan); self.bpm_noise_convention.currentIndexChanged.connect(self._update_measurement_help)
        for widget in (self.measurement_name,self.measurement_label,self.output_directory): widget.textChanged.connect(self.refresh_plan)
        layout.addWidget(group)
        self.dispersion_config_group=QGroupBox("Manual RF dispersion configuration"); dispersion_form=QFormLayout(self.dispersion_config_group); self._configure_form(dispersion_form)
        self.rf_control_mode=QComboBox(); self.rf_control_mode.addItem("Manual RF change","manual"); self.rf_control_mode.addItem("Automatic (adapter RF-write capability required)","automatic"); self.rf_control_mode.model().item(1).setEnabled(AdapterCapability.RF_WRITE in self.adapter.capabilities)
        self.nominal_rf=QLineEdit(); self.nominal_rf.setPlaceholderText("Required — enter nominal RF frequency in Hz"); self.nominal_rf_source=QLabel("Source: manual entry (no RF readback available)")
        nominal_widget=QWidget(); nominal_layout=QVBoxLayout(nominal_widget); nominal_layout.setContentsMargins(0,0,0,0); nominal_layout.addWidget(self.nominal_rf); nominal_layout.addWidget(self.nominal_rf_source)
        self.manual_actual_rf=QLineEdit(); self.manual_actual_rf.setPlaceholderText("Enter the actual applied RF frequency in Hz")
        self.manual_actual_rf.setToolTip("Required only when the backend cannot provide an RF readback. The value is saved as operator-entered and not independently verified.")
        self.rf_step=NoWheelDoubleSpinBox(); self.rf_step.setRange(1,1e9); self.rf_step.setDecimals(0); self.rf_step.setValue(200.0); self.rf_step.setSuffix(" Hz")
        self.dispersion_direction=QComboBox(); self.dispersion_direction.addItem("Bipolar ±Δf","bipolar"); self.dispersion_direction.addItem("One-sided +Δf","positive"); self.dispersion_direction.addItem("One-sided −Δf","negative")
        self.settling_delay=NoWheelDoubleSpinBox(); self.settling_delay.setRange(0,3600); self.settling_delay.setDecimals(3); self.settling_delay.setSuffix(" s")
        self.verify_restored_orbit=QCheckBox("Verify restored orbit"); self.verify_restored_orbit.setChecked(True)
        self.verify_restored_orbit.setToolTip("After exact RF restoration, acquire a read-only reference orbit for comparison. This orbit is never used in the dispersion calculation.")
        self.rf_safety=QLabel("READ ONLY — pyLOCO Measure will guide RF changes but will never write RF."); self.rf_safety.setWordWrap(True)
        self.rf_states_explanation=QLabel("Applied states: f₀ + 100 Hz / f₀ − 100 Hz\nTotal separation: Δ = 200 Hz"); self.rf_states_explanation.setWordWrap(True)
        for label,widget in (("RF control mode",self.rf_control_mode),("Nominal RF frequency",nominal_widget),("Actual applied RF",self.manual_actual_rf),("Total RF separation Δ",self.rf_step),("Applied RF states",self.rf_states_explanation),("Measurement direction",self.dispersion_direction),("Settling delay",self.settling_delay),("Restoration check",self.verify_restored_orbit),("",self.rf_safety)): dispersion_form.addRow(label,widget)
        self.rf_step_label=dispersion_form.labelForField(self.rf_step)
        for widget in (self.nominal_rf,self.rf_step,self.dispersion_direction,self.settling_delay):
            signal=widget.textChanged if isinstance(widget,QLineEdit) else widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.valueChanged
            signal.connect(self.refresh_plan)
        self.dispersion_direction.currentIndexChanged.connect(self._update_measurement_help)
        self.dispersion_direction.currentIndexChanged.connect(self._sync_rf_presentation)
        self.rf_control_mode.currentIndexChanged.connect(self._rf_mode_changed)
        self.verify_restored_orbit.toggled.connect(self.refresh_plan)
        layout.addWidget(self.dispersion_config_group)
        self.orm_config_group=QGroupBox("ORM corrector perturbation configuration"); orm_form=QFormLayout(self.orm_config_group); self._configure_form(orm_form)
        self.orm_direction=QComboBox(); self.orm_direction.addItem("Bipolar","bipolar"); self.orm_direction.addItem("Positive one-sided","positive"); self.orm_direction.addItem("Negative one-sided","negative")
        self.orm_kick_mode=QComboBox(); self.orm_kick_mode.addItem("Common kick","common"); self.orm_kick_mode.addItem("Per-corrector kicks from file","file")
        self.orm_hkick=NoWheelDoubleSpinBox(); self.orm_hkick.setRange(.001,1e6); self.orm_hkick.setDecimals(3); self.orm_hkick.setValue(100); self.orm_hkick.setSuffix(" µrad")
        self.orm_vkick=NoWheelDoubleSpinBox(); self.orm_vkick.setRange(.001,1e6); self.orm_vkick.setDecimals(3); self.orm_vkick.setValue(100); self.orm_vkick.setSuffix(" µrad")
        self.orm_kick_file=QLineEdit(); kick_browse=QPushButton("Browse…"); kick_browse.clicked.connect(lambda:self._browse_into(self.orm_kick_file)); self.orm_kick_file_row=QWidget(); kfl=QHBoxLayout(self.orm_kick_file_row); kfl.setContentsMargins(0,0,0,0); kfl.addWidget(self.orm_kick_file,1); kfl.addWidget(kick_browse)
        self.orm_output_format=QLabel("Orbit difference [m] — required for pyLOCO FIT"); self.orm_output_format.setObjectName("infoText"); self.orm_output_format.setWordWrap(True)
        self.orm_normalized_analysis=QCheckBox("Also calculate normalized response [m/rad] for analysis")
        self.orm_normalized_analysis.setToolTip("Display-only analysis: divide each measured column by its actual kick readback separation. The saved FIT matrix remains the orbit difference in metres.")
        self.orm_state_explanation=QLabel("Applied states: K₀ + Δ/2 / K₀ − Δ/2\nTotal separation: Δ"); self.orm_state_explanation.setWordWrap(True)
        orm_form.addRow("ORM output format",self.orm_output_format); orm_form.addRow("Optional analysis",self.orm_normalized_analysis); orm_form.addRow("Measurement direction",self.orm_direction); orm_form.addRow("Corrector kick mode",self.orm_kick_mode); orm_form.addRow("Total horizontal bipolar kick Δ",self.orm_hkick); self.orm_hkick_label=orm_form.labelForField(self.orm_hkick); orm_form.addRow("Total vertical bipolar kick Δ",self.orm_vkick); self.orm_vkick_label=orm_form.labelForField(self.orm_vkick); orm_form.addRow("Applied corrector states",self.orm_state_explanation); orm_form.addRow("Kick arrays file",self.orm_kick_file_row); self.orm_kick_file_label=orm_form.labelForField(self.orm_kick_file_row)
        self.kick_preview=QTableWidget(0,5); self.kick_preview.setMinimumHeight(180); self.kick_preview.setHorizontalHeaderLabels(["Position","Plane","Corrector","Effective kick [µrad]","Source"]); self.kick_preview.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch); orm_form.addRow("Effective kicks",self.kick_preview)
        for widget in (self.orm_direction,self.orm_kick_mode,self.orm_hkick,self.orm_vkick,self.orm_kick_file,self.orm_normalized_analysis):
            signal=widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.valueChanged if isinstance(widget,QDoubleSpinBox) else widget.textChanged if isinstance(widget,QLineEdit) else widget.toggled
            signal.connect(self._orm_config_changed)
        layout.addWidget(self.orm_config_group)
        self._measurement_type_changed(); return self._scroll_page(content)

    def _review_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        plan_group=QGroupBox("Show measurement plan"); plan_group.setCheckable(True); plan_group.setChecked(False); plan_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        plan_outer=QVBoxLayout(plan_group); self.plan_body=QWidget(); plan=QGridLayout(self.plan_body); plan.setContentsMargins(2,4,2,4); plan.setHorizontalSpacing(18); plan.setVerticalSpacing(9); plan_outer.addWidget(self.plan_body); plan_group.toggled.connect(self.plan_body.setVisible); self.plan_body.setVisible(False)
        self.plan_values={}; self.dispersion_plan_widgets=[]; self.orm_plan_widgets=[]; self.settling_plan_widgets=[]
        plan_pairs=(("Measurement","Measurement I/O"),("BPMs","Noise definition"),("H correctors","V correctors"),("Total correctors","Kick source"),("H kick","V kick"),("Scaled ORM","ORM direction"),("RF mode","Nominal RF"),("RF step ±Δf","Bipolar separation"),("Negative RF","Positive RF"),("Direction","RF states"),("Readings","Delay"),("Settling delay","Est. duration"),("Adapter","Output"))
        for row,(left,right) in enumerate(plan_pairs):
            for pair,label in enumerate((left,right)):
                column=pair*2; key=label.lower().replace(".","").replace(" ","_"); name=QLabel(label); value=QLabel("—"); value.setObjectName("planValue"); self.plan_values[key]=value; plan.addWidget(name,row,column); plan.addWidget(value,row,column+1); plan.setColumnStretch(column+1,1)
                if label in {"RF mode","Nominal RF","RF step ±Δf","Bipolar separation","Negative RF","Positive RF","Direction","RF states"}: self.dispersion_plan_widgets.extend((name,value))
                if label in {"H correctors","V correctors","Total correctors","Kick source","H kick","V kick","Scaled ORM","ORM direction"}: self.orm_plan_widgets.extend((name,value))
                if label=="Settling delay":self.settling_plan_widgets.extend((name,value))
        self.plan_output=self.plan_values["output"]
        run_group=QGroupBox("Measurement status"); self.run_group=run_group; rl=QVBoxLayout(run_group); self.step_instruction=QLabel(); self.step_instruction.setWordWrap(True); self.step_instruction.setObjectName("planValue"); self.step_instruction.setMaximumHeight(52); self.step_instruction.setToolTip("Full measurement sequence is available in the Measurement plan."); rl.addWidget(self.step_instruction); self.reading_status=QLabel("Ready"); self.reading_status.setObjectName("runState"); self.progress=QProgressBar(); self.progress.setMinimumHeight(34); self.progress.setFormat("Ready — 0 / 0"); rl.addWidget(self.reading_status); rl.addWidget(self.progress)
        status_row=QHBoxLayout(); self.elapsed=QLabel("Elapsed: 0.00 s"); self.remaining=QLabel("Remaining: —"); self.samples=QLabel("Samples: 0 / 0")
        for metric in (self.elapsed,self.remaining,self.samples): metric.setObjectName("runMetric"); status_row.addWidget(metric)
        status_row.addStretch(1); rl.addLayout(status_row)
        self.live_plot=PlotCanvas(show_toolbar=False,minimum_height=230); self.live_plot.save_button.setVisible(False); self.live_plot.include_title.setVisible(False); rl.addWidget(self.live_plot)
        self.live_plot.canvas.mpl_connect("motion_notify_event",self._on_live_dispersion_hover)
        self.live_plot.canvas.mpl_connect("button_press_event",self._on_live_dispersion_click)
        self.live_dispersion_controls=QWidget(); ldc=QHBoxLayout(self.live_dispersion_controls); ldc.setContentsMargins(0,0,0,0); ldc.addWidget(QLabel("Live orbit view")); self.live_dispersion_view=QComboBox(); self.live_dispersion_view.addItem("Absolute orbit","absolute"); self.live_dispersion_view.addItem("Change from reference","relative"); self.live_dispersion_view.currentIndexChanged.connect(self._redisplay_live_orbit); ldc.addWidget(self.live_dispersion_view); self.live_dispersion_selection=QLabel("Click a BPM point to inspect it"); self.live_dispersion_selection.setObjectName("statusValue"); ldc.addWidget(self.live_dispersion_selection,1); self.live_dispersion_pin=QPushButton("Keep highlighted"); self.live_dispersion_pin.setEnabled(False); self.live_dispersion_pin.clicked.connect(self._toggle_live_dispersion_pin); ldc.addWidget(self.live_dispersion_pin); self.live_dispersion_clear=QPushButton("Clear selection and highlights"); self.live_dispersion_clear.clicked.connect(self._clear_dispersion_selection); ldc.addWidget(self.live_dispersion_clear); rl.addWidget(self.live_dispersion_controls); self.live_dispersion_controls.setVisible(False)
        buttons=QHBoxLayout(); self.start_button=QPushButton("Start BPM-noise measurement"); self.start_button.setObjectName("measurePrimary"); self.start_button.clicked.connect(self.start_acquisition); self.repeat_button=QPushButton("Repeat measurement"); self.repeat_button.setObjectName("measurePrimary"); self.repeat_button.clicked.connect(self.repeat_measurement); self.repeat_button.setVisible(False); self.cancel_button=QPushButton("Cancel"); self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_acquisition); self.preview_toggle=QPushButton("Show live preview"); self.preview_toggle.setCheckable(True); self.preview_toggle.setVisible(False); self.preview_toggle.toggled.connect(self._toggle_completed_preview); buttons.addWidget(self.start_button); buttons.addWidget(self.repeat_button); buttons.addWidget(self.cancel_button); buttons.addWidget(self.preview_toggle); buttons.addStretch(1); rl.addLayout(buttons); self.start_block_reason=QLabel(); self.start_block_reason.setWordWrap(True); self.start_block_reason.setObjectName("warningText"); rl.addWidget(self.start_block_reason); self.plan_group=plan_group; layout.addWidget(plan_group); layout.addWidget(run_group)
        # Compact actions shown after a measurement completes.
        self.completed_actions = QWidget()
        completed_actions_layout = QHBoxLayout(self.completed_actions)
        completed_actions_layout.setContentsMargins(0, 0, 0, 0)

        self.completed_repeat_button = QPushButton("Repeat measurement")
        self.completed_repeat_button.setObjectName("measurePrimary")
        self.completed_repeat_button.clicked.connect(self.repeat_measurement)
        completed_actions_layout.addWidget(self.completed_repeat_button)
        completed_actions_layout.addStretch(1)

        self.completed_actions.setVisible(False)
        layout.addWidget(self.completed_actions)

        self.log_group=QGroupBox("Measurement log"); self.log_group.setCheckable(True); self.log_group.setChecked(False); log_layout=QVBoxLayout(self.log_group); self.log_body=QWidget(); body_layout=QVBoxLayout(self.log_body); body_layout.setContentsMargins(0,0,0,0); self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(110); log_actions=QHBoxLayout(); clear_log=QPushButton("Clear"); clear_log.clicked.connect(self.log.clear); save_log=QPushButton("Save log…"); save_log.clicked.connect(self.save_log); log_actions.addWidget(clear_log); log_actions.addWidget(save_log); log_actions.addStretch(1); body_layout.addLayout(log_actions); body_layout.addWidget(self.log); log_layout.addWidget(self.log_body); self.log_group.toggled.connect(self.log_body.setVisible); self.log_body.setVisible(False); layout.addWidget(self.log_group)
        self.orm_column_row=QWidget(); ocr=QHBoxLayout(self.orm_column_row); ocr.setContentsMargins(0,0,0,0); ocr.addWidget(QLabel("Selected ORM column")); self.orm_column_selector=QComboBox(); self.orm_column_selector.currentIndexChanged.connect(self._update_selected_orm_column); ocr.addWidget(self.orm_column_selector,1); self.orm_column_corrector_info=QLabel("Corrector: —"); self.orm_column_corrector_info.setObjectName("statusValue"); self.orm_column_corrector_info.setWordWrap(True); ocr.addWidget(self.orm_column_corrector_info,2); layout.addWidget(self.orm_column_row); self.orm_column_row.setVisible(False)
        self.orm_view_row=QWidget(); ovr=QHBoxLayout(self.orm_view_row); ovr.setContentsMargins(0,0,0,0); ovr.addWidget(QLabel("ORM matrix plot")); self.orm_view_mode=QComboBox(); self.orm_view_mode.addItem("2D heatmap","2d"); self.orm_view_mode.addItem("3D surface","3d"); self.orm_view_mode.currentIndexChanged.connect(self._redraw_orm_result); ovr.addWidget(self.orm_view_mode); ovr.addWidget(QLabel("Palette")); self.orm_palette=QComboBox(); self.orm_palette.addItem("FIT viridis — default","viridis"); self.orm_palette.addItem("Signed RdBu_r — symmetric about zero","red_blue"); self.orm_palette.currentIndexChanged.connect(self._redraw_orm_result); ovr.addWidget(self.orm_palette); ovr.addStretch(1); layout.addWidget(self.orm_view_row); self.orm_view_row.setVisible(False)
        self.orm_element_info=QLabel("Hover over an ORM cell or selected-column BPM point to identify it."); self.orm_element_info.setObjectName("infoText"); self.orm_element_info.setWordWrap(True); self.orm_element_info.setVisible(False); layout.addWidget(self.orm_element_info)
        self.dispersion_display_row=QWidget(); ddr=QHBoxLayout(self.dispersion_display_row); ddr.setContentsMargins(0,0,0,0); self.dispersion_display_label=QLabel("Showing: RF orbit difference"); self.dispersion_display_label.setStyleSheet("font-weight:750"); ddr.addWidget(self.dispersion_display_label); self.dispersion_display=QComboBox(); self.dispersion_display.addItem("RF orbit difference (pyLOCO-compatible)","raw"); self.dispersion_display.addItem("Physical dispersion (optional — requires AT)","physical"); self.dispersion_display.currentIndexChanged.connect(self._dispersion_display_changed); ddr.addWidget(self.dispersion_display,1); self.dispersion_connect_points=QCheckBox("Connect points"); self.dispersion_connect_points.setChecked(True); self.dispersion_connect_points.setToolTip("Connected markers are shown by default; uncheck for a scatter-only view."); self.dispersion_connect_points.toggled.connect(self._redraw_dispersion_plots); ddr.addWidget(self.dispersion_connect_points); self.dispersion_display_reason=QLabel(); self.dispersion_display_reason.setWordWrap(True); ddr.addWidget(self.dispersion_display_reason,2); layout.addWidget(self.dispersion_display_row); self.dispersion_display_row.setVisible(False)
        self.dispersion_selection_row=QWidget(); dsr=QHBoxLayout(self.dispersion_selection_row); dsr.setContentsMargins(0,0,0,0); dsr.addWidget(QLabel("Selected BPM")); self.dispersion_selected_details=QLabel("Click a plotted BPM to inspect it"); self.dispersion_selected_details.setObjectName("statusValue"); self.dispersion_selected_details.setWordWrap(True); dsr.addWidget(self.dispersion_selected_details,1); self.dispersion_pin_button=QPushButton("Keep highlighted"); self.dispersion_pin_button.setEnabled(False); self.dispersion_pin_button.clicked.connect(self._toggle_live_dispersion_pin); dsr.addWidget(self.dispersion_pin_button); self.dispersion_clear_button=QPushButton("Clear selection and highlights"); self.dispersion_clear_button.clicked.connect(self._clear_dispersion_selection); dsr.addWidget(self.dispersion_clear_button); layout.addWidget(self.dispersion_selection_row); self.dispersion_selection_row.setVisible(False)
        self.results_tabs=QTabWidget()
        # Keep completed measurement plots within the laptop viewport.
        # PlotCanvas itself uses an Expanding vertical size policy, so a
        # minimum height alone does not constrain the result area.
        self.results_tabs.setFixedHeight(340)
        # The long ORM/Dispersion tab labels must not force the whole
        # scientific result wider than the acquisition viewport.
        self.results_tabs.setMinimumWidth(0)
        self.results_tabs.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )

        result_tab_bar = self.results_tabs.tabBar()
        result_tab_bar.setExpanding(False)
        result_tab_bar.setUsesScrollButtons(True)
        # Preserve complete scientific titles.  When all tabs do not fit, Qt's
        # tab-bar scroll buttons expose the remaining full-width titles.
        result_tab_bar.setElideMode(Qt.ElideNone)
        self.x_plot=PlotCanvas(minimum_height=260); self.y_plot=PlotCanvas(minimum_height=260)
        self.raw_x_plot=PlotCanvas(minimum_height=260); self.raw_y_plot=PlotCanvas(minimum_height=260)
        self.mean_x_plot=PlotCanvas(minimum_height=260); self.mean_y_plot=PlotCanvas(minimum_height=260)
        self.rf_shift_x_plot=PlotCanvas(minimum_height=260); self.rf_shift_y_plot=PlotCanvas(minimum_height=260)
        self.orm_vv_plot=PlotCanvas(minimum_height=260); self.orm_column_plot=PlotCanvas(minimum_height=260); self.orm_kick_plot=PlotCanvas(minimum_height=260)
        self.orm_model_plot=PlotCanvas(minimum_height=260); self.orm_difference_plot=PlotCanvas(minimum_height=260)
        self.results_tabs.addTab(self.x_plot,"Horizontal BPM noise"); self.results_tabs.addTab(self.y_plot,"Vertical BPM noise"); self.results_tabs.addTab(self.mean_x_plot,"Mean horizontal orbit"); self.results_tabs.addTab(self.mean_y_plot,"Mean vertical orbit"); layout.addWidget(self.results_tabs)
        self._bpm_plot_state={}; self._mean_plot_state={}; self._dispersion_plot_state={}; self._orm_plot_state={}; self._orm_column_state=None; self._orm_selected_info=""; self._selected_bpm_index=None; self._manual_bad_bpm_indices=set(); self._threshold_flagged_bpm_indices=set(); self._pinned_bpm_indices=set(); self._bpm_review_notes={}; self._saved_bpm_review_notes={}; self._bpm_review_statuses={}; self._bpm_review_dirty=False; self._loading_bpm_review_panel=False
        for plane,canvas in (("x",self.x_plot),("y",self.y_plot)):
            canvas.canvas.mpl_connect("motion_notify_event",lambda event,p=plane:self._on_bpm_plot_hover(p,event))
            canvas.canvas.mpl_connect("button_press_event",lambda event,p=plane:self._on_bpm_plot_click(p,event))
        for plane,canvas in (("x",self.mean_x_plot),("y",self.mean_y_plot)):
            canvas.canvas.mpl_connect("motion_notify_event",lambda event,p=plane:self._on_mean_plot_hover(p,event))
            canvas.canvas.mpl_connect("button_press_event",lambda event,p=plane:self._on_mean_plot_click(p,event))
        for key,canvas in (("raw_x",self.raw_x_plot),("raw_y",self.raw_y_plot),("physical_x",self.x_plot),("physical_y",self.y_plot)):
            canvas.canvas.mpl_connect("motion_notify_event",lambda event,k=key:self._on_dispersion_plot_hover(k,event))
            canvas.canvas.mpl_connect("button_press_event",lambda event,k=key:self._on_dispersion_plot_click(k,event))
        for key,canvas in (("states_x",self.mean_x_plot),("states_y",self.mean_y_plot),("shift_x",self.rf_shift_x_plot),("shift_y",self.rf_shift_y_plot)):
            canvas.canvas.mpl_connect("motion_notify_event",lambda event,k=key:self._on_dispersion_plot_hover(k,event))
            canvas.canvas.mpl_connect("button_press_event",lambda event,k=key:self._on_dispersion_plot_click(k,event))
        for canvas in (self.x_plot,self.y_plot,self.mean_x_plot,self.mean_y_plot,self.orm_vv_plot,self.orm_model_plot,self.orm_difference_plot):
            canvas.canvas.mpl_connect("motion_notify_event",lambda event,c=canvas:self._on_orm_plot_hover(c,event))
            canvas.canvas.mpl_connect("button_press_event",lambda event,c=canvas:self._on_orm_plot_click(c,event))
        self.orm_column_plot.canvas.mpl_connect("motion_notify_event",self._on_orm_column_hover)
        self.orm_column_plot.canvas.mpl_connect("button_press_event",self._on_orm_column_click)
        self.stats_group=QGroupBox("Statistics"); self.stats_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); stats_layout=QVBoxLayout(self.stats_group); stats_layout.setContentsMargins(10,10,10,10); stats_layout.setSpacing(7); self.orm_measurement_summary=QLabel(); self.orm_measurement_summary.setObjectName("summaryCard"); self.orm_measurement_summary.setWordWrap(True); self.orm_measurement_summary.setTextInteractionFlags(Qt.TextSelectableByMouse); self.orm_restoration_status=QLabel(); self.orm_restoration_status.setObjectName("runState"); self.orm_restoration_status.setWordWrap(True); self.statistics_table=MeasurementStatisticsTable(); self.summary_x=QLabel("Horizontal: no completed measurement."); self.summary_y=QLabel("Vertical: no completed measurement."); self.summary_x.hide(); self.summary_y.hide(); self.rf_diagnostics=QLabel(); self.restoration_label=QLabel(); self.rf_diagnostics.setObjectName("summaryCard"); self.restoration_label.setObjectName("statLine"); self.rf_diagnostics.setWordWrap(True); self.restoration_label.setWordWrap(True); stats_layout.addWidget(self.orm_measurement_summary); stats_layout.addWidget(self.orm_restoration_status); stats_layout.addWidget(self.statistics_table); stats_layout.addWidget(self.rf_diagnostics); stats_layout.addWidget(self.restoration_label); layout.addWidget(self.stats_group)
        self.bpm_review_group=QGroupBox("BPM identification and review — no automatic FIT exclusion"); self.bpm_review_group.setCheckable(True); self.bpm_review_group.setChecked(False); self.bpm_review_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        bpm_review_layout=QVBoxLayout(self.bpm_review_group); self.bpm_review_body=QWidget(); brl=QVBoxLayout(self.bpm_review_body); brl.setContentsMargins(0,2,0,0)
        self.bpm_review_details=QLabel("Hover or click a BPM point to identify it. Manual selections are review annotations only."); self.bpm_review_details.setObjectName("summaryCard"); self.bpm_review_details.setWordWrap(True); brl.addWidget(self.bpm_review_details)
        review_controls=QHBoxLayout(); self.bpm_review_search=QLineEdit(); self.bpm_review_search.setPlaceholderText("Search BPM name or original index…"); self.bpm_review_search.textChanged.connect(self._filter_bpm_review_table); review_controls.addWidget(self.bpm_review_search,1); review_controls.addWidget(QLabel("Review status")); self.bpm_review_status_filter=QComboBox(); self.bpm_review_status_filter.addItem("All BPMs","all"); self.bpm_review_status_filter.addItem("Bad + Suspect","bad_suspect"); self.bpm_review_status_filter.addItem("Bad only","Bad"); self.bpm_review_status_filter.addItem("Suspect only","Suspect"); self.bpm_review_status_filter.addItem("Good only","Good"); self.bpm_review_status_filter.addItem("Unreviewed","Unreviewed"); self.bpm_review_status_filter.currentIndexChanged.connect(self._apply_bpm_review_filter); review_controls.addWidget(self.bpm_review_status_filter); self.bpm_review_match_count=QLabel("0 of 0 BPMs"); self.bpm_review_match_count.setObjectName("statusValue"); review_controls.addWidget(self.bpm_review_match_count); self.bpm_clear_pins_button=QPushButton("Clear highlights"); self.bpm_clear_pins_button.clicked.connect(self._clear_bpm_pins); review_controls.addWidget(self.bpm_clear_pins_button)
        brl.addLayout(review_controls)
        self.bpm_advanced_group=QGroupBox("Advanced noise thresholds"); self.bpm_advanced_group.setCheckable(True); self.bpm_advanced_group.setChecked(False); advanced_group_layout=QVBoxLayout(self.bpm_advanced_group); self.bpm_advanced_controls=QWidget(); advanced_layout=QHBoxLayout(self.bpm_advanced_controls); advanced_layout.setContentsMargins(0,0,0,0)
        self.bpm_threshold_x=NoWheelDoubleSpinBox(); self.bpm_threshold_y=NoWheelDoubleSpinBox()
        for plane,spin in (("X",self.bpm_threshold_x),("Y",self.bpm_threshold_y)):
            spin.setRange(0,1e12); spin.setDecimals(3); spin.setSuffix(" nm"); spin.setSpecialValueText("Off"); spin.setToolTip(f"Flag {plane}-plane noise above this value; zero disables the threshold"); spin.valueChanged.connect(self._threshold_review_changed); advanced_layout.addWidget(QLabel(f"{plane} threshold")); advanced_layout.addWidget(spin)
        advanced_layout.addStretch(1); advanced_group_layout.addWidget(self.bpm_advanced_controls); self.bpm_advanced_controls.setVisible(False); self.bpm_advanced_group.toggled.connect(self.bpm_advanced_controls.setVisible); brl.addWidget(self.bpm_advanced_group)
        self.bpm_review_table=QTableWidget(0,6); self.bpm_review_table.setHorizontalHeaderLabels(["BPM (original index)","Horizontal [nm]","Vertical [nm]","Highlight","Review status","Notes"]); self.bpm_review_table.setAlternatingRowColors(True); self.bpm_review_table.setSortingEnabled(True); self.bpm_review_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.bpm_review_table.setSelectionMode(QAbstractItemView.SingleSelection); self.bpm_review_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.bpm_review_table.setMinimumHeight(190); self.bpm_review_table.setMaximumHeight(260); header=self.bpm_review_table.horizontalHeader(); header.setSectionResizeMode(0,QHeaderView.Stretch); header.setSectionResizeMode(1,QHeaderView.ResizeToContents); header.setSectionResizeMode(2,QHeaderView.ResizeToContents); header.setSectionResizeMode(3,QHeaderView.ResizeToContents); header.setSectionResizeMode(4,QHeaderView.ResizeToContents); header.setSectionResizeMode(5,QHeaderView.Stretch); self.bpm_review_table.itemSelectionChanged.connect(self._bpm_review_table_selection_changed); self.bpm_review_table.itemChanged.connect(self._bpm_review_item_changed); brl.addWidget(self.bpm_review_table)
        self.bpm_review_empty=QLabel("No BPMs match the current search and review-status filter."); self.bpm_review_empty.setObjectName("infoText"); self.bpm_review_empty.setAlignment(Qt.AlignCenter); self.bpm_review_empty.setVisible(False); brl.addWidget(self.bpm_review_empty)
        selected_group=QGroupBox("Review selected BPM"); selected_layout=QGridLayout(selected_group); selected_layout.addWidget(QLabel("Status"),0,0); self.bpm_review_status=QComboBox(); self.bpm_review_status.addItems(["Unreviewed","Good","Suspect","Bad"]); self.bpm_review_status.setEnabled(False); self.bpm_review_status.currentTextChanged.connect(self._selected_bpm_review_changed); selected_layout.addWidget(self.bpm_review_status,0,1); self.bpm_review_highlight=QCheckBox("Keep highlighted on plots"); self.bpm_review_highlight.setEnabled(False); self.bpm_review_highlight.toggled.connect(self._selected_bpm_review_changed); selected_layout.addWidget(self.bpm_review_highlight,0,2); selected_layout.addWidget(QLabel("Notes"),1,0); self.bpm_review_note=QLineEdit(); self.bpm_review_note.setPlaceholderText("Optional machine-study note…"); self.bpm_review_note.setEnabled(False); self.bpm_review_note.textChanged.connect(self._selected_bpm_review_changed); selected_layout.addWidget(self.bpm_review_note,1,1,1,2); self.bpm_save_review_button=QPushButton("Save Review"); self.bpm_save_review_button.setEnabled(False); self.bpm_save_review_button.clicked.connect(self._save_bpm_review); selected_layout.addWidget(self.bpm_save_review_button,2,0); self.bpm_review_save_state=QLabel("All review changes saved"); self.bpm_review_save_state.setObjectName("validationOk"); selected_layout.addWidget(self.bpm_review_save_state,2,1); self.bpm_export_review_button=QPushButton("Export Review CSV…"); self.bpm_export_review_button.clicked.connect(self._export_bpm_review_csv); selected_layout.addWidget(self.bpm_export_review_button,2,2); brl.addWidget(selected_group)
        annotation_note=QLabel("Review only — marking a BPM as Bad does not automatically exclude it from LOCO fitting. Original measurement data remains unchanged."); self.bpm_review_notice=annotation_note; annotation_note.setWordWrap(True); annotation_note.setObjectName("infoText"); brl.addWidget(annotation_note)
        bpm_review_layout.addWidget(self.bpm_review_body); self.bpm_review_group.toggled.connect(self.bpm_review_body.setVisible); self.bpm_review_body.setVisible(False); self.bpm_review_group.setVisible(False); layout.addWidget(self.bpm_review_group)
        self.dispersion_summary=QWidget(); dispersion_summary_layout=QVBoxLayout(self.dispersion_summary); dispersion_summary_layout.setContentsMargins(0,0,0,0)
        rf_response_group=QGroupBox("1. RF orbit difference — pyLOCO-compatible"); rf_response_layout=QVBoxLayout(rf_response_group); self.rf_response_formula=QLabel("Δx_RF = mean[x(f−)] − mean[x(f+)]\nΔy_RF = mean[y(f−)] − mean[y(f+)]"); self.rf_response_formula.setWordWrap(True); self.rf_response_note=QLabel("Stored in metres; displayed in mm; this is the RF-response column supplied to pyLOCO."); self.rf_response_note.setWordWrap(True); self.rf_response_stats=MeasurementStatisticsTable(); rf_response_layout.addWidget(self.rf_response_formula); rf_response_layout.addWidget(self.rf_response_note); rf_response_layout.addWidget(self.rf_response_stats); dispersion_summary_layout.addWidget(rf_response_group)
        physical_group=QGroupBox("2. Physical dispersion"); physical_layout=QVBoxLayout(physical_group); self.physical_formula=QLabel("η = αc − 1/γ²\nδ(f) ≈ −(f−f₀)/(η f₀)\nΔδ = δ− − δ+\nDₓ = Δx_RF / Δδ\nDᵧ = Δy_RF / Δδ"); self.physical_formula.setWordWrap(True); self.physical_stats=MeasurementStatisticsTable(); self.physical_unavailable=QLabel(); self.physical_unavailable.setObjectName("infoText"); self.physical_unavailable.setWordWrap(True); physical_layout.addWidget(self.physical_formula); physical_layout.addWidget(self.physical_stats); physical_layout.addWidget(self.physical_unavailable)
        self.calculation_details_group=QGroupBox("Show calculation details"); self.calculation_details_group.setCheckable(True); self.calculation_details_group.setChecked(False); calculation_layout=QVBoxLayout(self.calculation_details_group); self.calculation_details_body=QLabel(); self.calculation_details_body.setWordWrap(True); self.calculation_details_body.setTextInteractionFlags(Qt.TextSelectableByMouse); calculation_layout.addWidget(self.calculation_details_body); self.calculation_details_group.toggled.connect(self.calculation_details_body.setVisible); self.calculation_details_body.setVisible(False); physical_layout.addWidget(self.calculation_details_group); dispersion_summary_layout.addWidget(physical_group)
        restoration_group=QGroupBox("3. Restoration verification"); restoration_layout=QVBoxLayout(restoration_group); self.restoration_status=QLabel("RF restoration: —"); self.restoration_status.setObjectName("runState"); self.restoration_status.setWordWrap(True); self.restoration_values=QLabel(); self.restoration_values.setWordWrap(True); self.restoration_diagnostic=QLabel(); self.restoration_diagnostic.setWordWrap(True); restoration_layout.addWidget(self.restoration_status); restoration_layout.addWidget(self.restoration_values); restoration_layout.addWidget(self.restoration_diagnostic); dispersion_summary_layout.addWidget(restoration_group)
        comparison_group=QGroupBox("4. Measured vs reference model"); comparison_layout=QVBoxLayout(comparison_group); self.dispersion_model_comparison=QLabel("Comparison not requested."); self.dispersion_model_comparison.setObjectName("summaryCard"); self.dispersion_model_comparison.setWordWrap(True); self.dispersion_model_comparison.setTextInteractionFlags(Qt.TextSelectableByMouse); comparison_layout.addWidget(self.dispersion_model_comparison); dispersion_summary_layout.addWidget(comparison_group)
        self.dispersion_summary.setVisible(False); layout.insertWidget(layout.indexOf(self.stats_group),self.dispersion_summary)
        save_group=QGroupBox("Saved files"); self.save_group=save_group; sl=QVBoxLayout(save_group); self.paths=QLabel("Measurement file: —\nSession manifest: —"); self.paths.setTextInteractionFlags(Qt.TextSelectableByMouse); actions=QHBoxLayout(); self.validate_button=QPushButton("Validate for pyLOCO"); self.validate_button.setEnabled(False); self.validate_button.clicked.connect(self.validate_saved); self.open_button=QPushButton("Open session in pyLOCO"); self.open_button.setEnabled(False); self.open_button.clicked.connect(self.explain_open); actions.addWidget(self.validate_button); actions.addWidget(self.open_button); actions.addStretch(1); sl.addWidget(self.paths); sl.addLayout(actions); layout.addWidget(save_group)
        self.paths.setWordWrap(True)
        self.open_output_button=QPushButton("Open output folder"); self.open_output_button.setEnabled(False); self.open_output_button.clicked.connect(self.open_output_folder); actions.insertWidget(2,self.open_output_button)
        self._reset_live_plot()
        self._update_diagnostics_visibility()
        self.acquisition_scroll=self._scroll_page(content)
        self.acquisition_scroll.setObjectName("acquisitionWorkspaceScroll")
        return self.acquisition_scroll

    def _toggle_completed_preview(self, visible):
        self.live_plot.setVisible(bool(visible))
        self.preview_toggle.setText("Hide live preview" if visible else "Show live preview")

    def _set_completed_result_view(self, completed):
        """Prioritize scientific result plots once acquisition has completed."""
        completed = bool(completed)

        # Acquisition controls are useful before/during acquisition, but after
        # completion the scientific result should become the primary view.
        self.run_group.setVisible(not completed)
        self.plan_group.setVisible(not completed)

        # Keep a compact Repeat action available when the acquisition
        # panel itself is hidden.
        if hasattr(self, "completed_actions"):
            self.completed_actions.setVisible(completed)

        self.preview_toggle.blockSignals(True)
        self.preview_toggle.setChecked(False)
        self.preview_toggle.blockSignals(False)
        self.preview_toggle.setText("Show live preview")
        self.preview_toggle.setVisible(False)

        self.live_plot.setVisible(not completed)

        if completed:
            # A completed measurement must always expose its scientific
            # result area. Mode-specific diagnostics are then controlled
            # by _update_diagnostics_visibility().
            self.results_tabs.setVisible(True)

            kind = self.measurement_type.currentData()

            if kind == "bpm_noise":
                self.stats_group.setVisible(True)

            elif kind == "dispersion":
                self.dispersion_display_row.setVisible(True)
                self.dispersion_summary.setVisible(True)

            elif kind == "orm":
                self.orm_column_row.setVisible(True)
                self.orm_view_row.setVisible(True)
                self.stats_group.setVisible(True)

        # Before completion the compact dual-pane form may shrink vertically.
        # Completed scientific results must instead retain every section's size
        # and use the acquisition scroll area when the viewport is too short.
        content = self.acquisition_scroll.widget() if hasattr(self, "acquisition_scroll") else None
        if content is not None:
            if completed:
                content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                self._fit_completed_content()
                # Width-dependent labels acquire their final wrapped height only
                # after the splitter has completed its layout pass.
                QTimer.singleShot(0, self._fit_completed_content)
            else:
                content.setMinimumHeight(0)
                content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            content.updateGeometry()

        if completed:
            # Bring the scientific result directly into view.
            self.acquisition_scroll.ensureWidgetVisible(self.results_tabs, 0, 0)

    def _fit_completed_content(self) -> None:
        if not hasattr(self, "acquisition_scroll"):
            return
        content = self.acquisition_scroll.widget()
        if content is None or not self.completed_actions.isVisible():
            return
        content.layout().activate()
        # A small guard absorbs style-dependent group-box title overhang.
        content.setMinimumHeight(content.layout().minimumSize().height() + 12)
        content.updateGeometry()

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
        self._model_orm_cache=None
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
            elif read_diagnostics and self.adapter_combo.currentData() == "pyaml" and self.connection_verified:
                try:
                    diagnostics["KICK.RBV"] = self.adapter.read(device.readback_channel).value
                except Exception as exc:
                    table.setToolTip(f"pyAML strength read unavailable — {exc}")
            values=(row,device.name,device.identifier,device.plane,*(f"{diagnostics.get(field):.8g}" if field in diagnostics else "—" for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")))
            for col,value in enumerate(values):table.setItem(row,col,QTableWidgetItem(str(value)))
        self._filter_corrector_table(key)
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}[data["method"].currentData()]; table.setToolTip((table.toolTip()+"\n" if table.toolTip() else "")+f"Source: {source}; {len(selected)} retained")
        data["count"].setText(f"{len(selected)} / {len(data['devices'])}")
        # Correctors are ORM inputs even when prepared before ORM is selected.
        # Store them directly in the ORM-specific state so changing measurement
        # type cannot replace the user's prepared selection with old defaults.
        if not self._loading_selection_state:
            self._selection_states["orm"][key]={
                "method":data["method"].currentData(),
                "manual":data["manual"].text(),
                "names_file":data["file"].text(),
                "excluded_positions":data["exclusion"].text(),
            }
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

    def _reset_measurement_display(self):
        """Reset transient acquisition/result UI without deleting saved files."""
        self.result = None
        self.saved_measurement_path = None
        self.saved_session_path = None
        self.dispersion_states = []
        self.dispersion_step_index = 0

        if hasattr(self, "progress"):
            self.progress.setValue(0)
            self.progress.setFormat("Not started")

        if hasattr(self, "reading_status"):
            self.reading_status.setText("Ready")

        if hasattr(self, "elapsed"):
            self.elapsed.setText("Elapsed: 0.00 s")

        if hasattr(self, "remaining"):
            self.remaining.setText("Remaining: —")

        if hasattr(self, "samples"):
            self.samples.setText("Samples: 0 / 0")

        if hasattr(self, "log"):
            self.log.clear()

        if hasattr(self, "paths"):
            self.paths.clear()
            self.paths.setText(
                "No measurement saved for the current measurement type."
            )

        if hasattr(self, "repeat_button"):
            self.repeat_button.setVisible(False)

        if hasattr(self, "start_button"):
            self.start_button.setVisible(True)
            self.start_button.setEnabled(True)

        if hasattr(self, "restoration_values"):
            self.restoration_values.clear()

        if hasattr(self, "restoration_diagnostic"):
            self.restoration_diagnostic.clear()

        # Clear acquisition preview only after the review/acquisition
        # page has created self.live_plot. During initial UI construction
        # measurement_type_changed() is called before that widget exists.
        if hasattr(self, "live_plot"):
            self._reset_live_plot()

        # Clear every completed scientific-result canvas.
        for name in (
            "x_plot",
            "y_plot",
            "raw_x_plot",
            "raw_y_plot",
            "mean_x_plot",
            "mean_y_plot",
            "rf_shift_x_plot",
            "rf_shift_y_plot",
            "orm_vv_plot",
            "orm_column_plot",
            "orm_kick_plot",
            "orm_model_plot",
            "orm_difference_plot",
        ):
            canvas = getattr(self, name, None)
            if canvas is not None:
                canvas.clear()
                canvas.apply_theme()

        # Reset result/statistics text.
        if hasattr(self, "summary_x"):
            self.summary_x.setText(
                "Horizontal: no completed measurement."
            )

        if hasattr(self, "summary_y"):
            self.summary_y.setText(
                "Vertical: no completed measurement."
            )

        if hasattr(self, "statistics_table"):
            self.statistics_table.clear_statistics()

        if hasattr(self,"bpm_review_group"):
            self.bpm_review_group.setVisible(False); self.bpm_review_group.setChecked(False)
            self.bpm_review_table.setRowCount(0); self.bpm_review_search.clear(); self.bpm_review_status_filter.setCurrentIndex(0); self.bpm_review_match_count.setText("0 of 0 BPMs"); self.bpm_review_empty.setVisible(False)
            self._bpm_plot_state.clear(); self._mean_plot_state.clear(); self._dispersion_plot_state.clear(); self._selected_bpm_index=None
            self._manual_bad_bpm_indices.clear(); self._threshold_flagged_bpm_indices.clear(); self._pinned_bpm_indices.clear(); self._bpm_review_notes.clear(); self._saved_bpm_review_notes.clear(); self._bpm_review_statuses.clear(); self.bpm_review_note.clear(); self.bpm_review_note.setEnabled(False); self.bpm_review_status.setEnabled(False); self.bpm_review_highlight.setEnabled(False); self._set_bpm_review_dirty(False)

        if hasattr(self, "orm_measurement_summary"):
            self.orm_measurement_summary.clear()

        if hasattr(self, "orm_restoration_status"):
            self.orm_restoration_status.clear()

        if hasattr(self, "rf_diagnostics"):
            self.rf_diagnostics.clear()

        if hasattr(self, "restoration_label"):
            self.restoration_label.clear()

        if hasattr(self, "rf_response_stats"):
            self.rf_response_stats.clear_statistics()

        if hasattr(self, "physical_stats"):
            self.physical_stats.clear_statistics()
        if hasattr(self,"physical_unavailable"):self.physical_unavailable.clear()

        if hasattr(self, "calculation_details_body"):
            self.calculation_details_body.clear()

        if hasattr(self, "restoration_status"):
            self.restoration_status.setText("RF restoration: —")

        if hasattr(self, "restoration_values"):
            self.restoration_values.clear()

        if hasattr(self, "restoration_diagnostic"):
            self.restoration_diagnostic.clear()

        # Leave completed-result presentation mode only after
        # _review_page() has created the required widgets.
        if hasattr(self, "plan_group"):
            self._set_completed_result_view(False)

        if hasattr(self, "validate_button"):
            self.validate_button.setEnabled(False)

        if hasattr(self, "open_button"):
            self.open_button.setEnabled(False)

        self._refresh_review_page()
        self._update_workflow_tabs()

    def _workspace_changed(self,index):
        if not hasattr(self,"heading"):return
        index=int(index)
        if index==0:title="Machine setup"
        elif index==1:title="Device selection"
        elif index==3:title="Review & Save"
        else:
            kind=self.measurement_type.currentData() if hasattr(self,"measurement_type") else "bpm_noise"
            title={"dispersion":"Dispersion measurement","orm":"Orbit response matrix measurement"}.get(kind,"BPM noise measurement")
        self.heading.setText(title)
        self.setWindowTitle("pyLOCO Measure" if index<2 else f"pyLOCO Measure — {title.replace(' measurement','').title()}")
        if index==3:self._refresh_review_page()
        if hasattr(self,"plan_values"):self.refresh_plan()

    def _refresh_review_page(self):
        if not hasattr(self,"review_status"):return
        ready=self.result is not None and self.saved_measurement_path is not None and self.saved_session_path is not None
        self.review_validate_button.setEnabled(ready); self.review_open_button.setEnabled(ready)
        if not ready:
            self.review_status.setText("No completed measurement yet.")
            self.review_details.setText("Complete a measurement to review its saved data and validation status.")
            self.review_paths.setText("Measurement file: —\nSession manifest: —")
            return
        kind="ORM" if isinstance(self.result,ORMResult) else "Dispersion" if isinstance(self.result,DispersionResult) else "BPM noise"
        restoration="Not applicable"
        if isinstance(self.result,DispersionResult):restoration=self.result.restoration_status.replace("_"," ").title()
        elif isinstance(self.result,ORMResult):
            statuses=tuple(self.result.restoration_status)
            restoration="All correctors restored" if statuses and all(status in {"restored","confirmed_by_operator"} for status in statuses) else "Check restoration status"
        try:
            session=load_session(self.saved_session_path,validate_files=False)
            roles={entry.role for entry in session.files}; available=", ".join(label for role,label in (("bpm_noise","BPM noise"),("dispersion","Dispersion"),("orm","ORM")) if role in roles)
            completeness="Complete for FIT" if session.is_complete_for_loco else "Missing for FIT: "+", ".join(session.missing_roles)
        except Exception as exc:
            available="Unavailable"; completeness=f"Session could not be read: {exc}"
        self.review_status.setText(f"✓ {kind} saved and schema-validated")
        self.review_details.setText(f"<b>Measurement</b> &nbsp; {kind}<br><b>Restoration</b> &nbsp; {restoration}<br><b>Session contains</b> &nbsp; {available}<br><b>FIT readiness</b> &nbsp; {completeness}")
        self.review_paths.setText(f"Measurement file: {self.saved_measurement_path}\nSession manifest: {self.saved_session_path}")

    def _measurement_type_changed(self, *_):
        if not hasattr(self,"measurement_type"): return
        kind=self.measurement_type.currentData()
        if kind!=self._active_selection_kind:
            self._store_active_selection(); self._active_selection_kind=kind; self._restore_selection(kind)
        dispersion=kind=="dispersion"; orm=kind=="orm"
        if orm and not getattr(self,"_fresh_orm_defaults_applied",False):
            # Match the established PETRA III ORM procedure for a fresh GUI
            # workflow. Loaded projects subsequently restore their own values.
            self.readings.setValue(10); self.delay.setValue(.1); self.settling_delay.setValue(1.0)
            self._fresh_orm_defaults_applied=True
        if orm and isinstance(self.adapter,MockAdapter) and hasattr(self.adapter,"set_simulated_writes_enabled"):
            self.adapter.set_simulated_writes_enabled(True)
        self.dispersion_config_group.setVisible(dispersion)
        if hasattr(self,"live_dispersion_controls"):self.live_dispersion_controls.setVisible(dispersion)
        if hasattr(self,"dispersion_selection_row"):self.dispersion_selection_row.setVisible(dispersion)
        self.orm_config_group.setVisible(orm)
        self.bpm_noise_convention.setVisible(not dispersion and not orm)
        self.bpm_noise_convention_label.setVisible(not dispersion and not orm)
        if hasattr(self,"orm_corrector_group"): self.orm_corrector_group.setEnabled(bool(self.devices))
        if hasattr(self,"bpm_exclusions"): self.bpm_exclusions.setVisible(orm); self.bpm_exclusions_label.setVisible(orm)
        self._update_measurement_help()
        self._workspace_changed(self.tabs.currentIndex())
        for widget in getattr(self,"dispersion_plan_widgets",[]): widget.setVisible(dispersion)
        if hasattr(self,"results_tabs"):
            self._configure_result_tabs(kind)
            self.dispersion_display_row.setVisible(dispersion)
            self._configure_physical_dispersion_option()
        self._update_diagnostics_visibility()
        self._reset_measurement_display()
        if hasattr(self,"repeat_button"):self.repeat_button.setVisible(False)
        if hasattr(self,"repeat_button"):self.repeat_button.setText("Repeat ORM measurement" if orm else "Repeat measurement")
        if hasattr(self,"start_button"):self.start_button.setVisible(True)
        if hasattr(self,"validate_button"):
            self.validate_button.setEnabled(False); self.open_button.setEnabled(False)
        if hasattr(self,"live_plot"): self._reset_live_plot()
        if hasattr(self,"plan_values"): self.refresh_plan()
        backend=self.adapter_combo.currentData() if hasattr(self,"adapter_combo") else "mock"
        context=self.pysc_profile_combo.currentText() if backend=="pysc" else "PETRA III LIVE" if backend=="petra" else "pyAML" if backend=="pyaml" else "Mock"
        default_names={"bpm-noise","mock-dispersion","pysc-dispersion","mock-orm","pysc-small-orm","pyloco-bpm-noise","pyloco-dispersion","pyloco-orm"}
        if self.measurement_name.text() in default_names:
            kind="orm" if orm else "dispersion" if dispersion else "bpm_noise"
            names={"bpm_noise":"pyloco-bpm-noise","dispersion":"pyloco-dispersion","orm":"pyloco-orm"}
            labels={"bpm_noise":f"pyLOCO BPM noise — {context}","dispersion":f"pyLOCO dispersion — {context}","orm":f"pyLOCO ORM — {context}"}
            self.measurement_name.setText(names[kind]); self.measurement_label.setText(labels[kind])
        self._update_kick_preview()

    def _configure_physical_dispersion_option(self):
        if not hasattr(self,"dispersion_display") or self.measurement_type.currentData()!="dispersion":return
        item=self.dispersion_display.model().item(self.dispersion_display.findData("physical"))
        lattice_metadata=self._dispersion_lattice_metadata()
        at_available=find_spec("at") is not None
        available=at_available and self.reference_model is not None and lattice_metadata is not None and self.dispersion_direction.currentData()=="bipolar"
        item.setEnabled(available)
        reason="Accelerator Toolbox (AT) is not installed" if not at_available else "load a reference lattice with AT" if self.reference_model is None else "the reference lattice provides no verified slip factor" if lattice_metadata is None else "positive and negative RF states are required"
        item.setToolTip("" if available else f"Unavailable: {reason}.")
        self.dispersion_display_reason.setText("Optional AT calculation using the reference-lattice slip factor and recorded f+/f− readbacks." if available else f"Physical dispersion unavailable: {reason}.")
        raw=self.dispersion_display.currentData()=="raw"
        self.dispersion_display_label.setText("Showing: RF orbit difference" if raw else "Showing: physical dispersion")
        self.dispersion_display_reason.setVisible(not raw)
        if hasattr(self,"results_tabs"):
            for canvas in (self.x_plot,self.y_plot):
                index=self.results_tabs.indexOf(canvas)
                if index>=0:self.results_tabs.setTabEnabled(index,available)
        if not available and self.dispersion_display.currentData()=="physical":self.dispersion_display.setCurrentIndex(0)

    def _dispersion_lattice_metadata(self):
        """Return explicit AT/design-lattice slip-factor conventions, or None."""
        model=getattr(self,"reference_model",None)
        if model is not None:
            return {"alpha":model.momentum_compaction,"inverse_gamma_squared":model.inverse_gamma_squared,
                    "at_slip_factor":-model.slip_factor,"eta":model.slip_factor,
                    "source":model.source,"path":str(model.path),"sha256":model.checksum_sha256}
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
        raw=self.dispersion_display.currentData()=="raw"
        self.dispersion_display_label.setText("Showing: RF orbit difference" if raw else "Showing: physical dispersion")
        self.dispersion_display_reason.setVisible(not raw)
        if isinstance(self.result,DispersionResult):self._show_dispersion_result(self.result)
        target=self.raw_x_plot if raw else self.x_plot
        index=self.results_tabs.indexOf(target)
        if index>=0:self.results_tabs.setCurrentIndex(index)

    def _redraw_dispersion_plots(self,*_):
        if isinstance(self.result,DispersionResult):self._show_dispersion_result(self.result)

    def _update_diagnostics_visibility(self):
        if not hasattr(self, "stats_group"):
            return

        kind = self.measurement_type.currentData()

        has_bpm_result = kind == "bpm_noise" and isinstance(
            self.result, BpmNoiseResult
        )
        has_dispersion_result = kind == "dispersion" and isinstance(
            self.result, DispersionResult
        )
        has_orm_result = kind == "orm" and isinstance(
            self.result, ORMResult
        )
        has_result = (
            has_bpm_result
            or has_dispersion_result
            or has_orm_result
        )

        # Final diagnostics belong to completed measurements only.
        self.rf_diagnostics.setVisible(has_dispersion_result)
        self.restoration_label.setVisible(has_dispersion_result)

        self.orm_measurement_summary.setVisible(has_bpm_result or has_orm_result)
        self.orm_restoration_status.setVisible(has_orm_result)

        if kind == "bpm_noise":
            self.stats_group.setTitle("Statistics")
        elif kind == "dispersion":
            self.stats_group.setTitle("Statistics and RF diagnostics")
        else:
            self.stats_group.setTitle("Statistics and ORM diagnostics")

        # BPM-noise and ORM statistics appear only after completion.
        # Dispersion uses its dedicated dispersion_summary instead.
        self.stats_group.setVisible(
            has_bpm_result or has_orm_result
        )

        if hasattr(self, "dispersion_summary"):
            self.dispersion_summary.setVisible(
                has_dispersion_result
            )

        # Result controls must also stay out of the acquisition layout
        # until their corresponding result exists.
        if hasattr(self, "dispersion_display_row"):
            self.dispersion_display_row.setVisible(
                has_dispersion_result
            )

        if hasattr(self, "orm_column_row"):
            self.orm_column_row.setVisible(has_orm_result)

        if hasattr(self, "orm_view_row"):
            self.orm_view_row.setVisible(has_orm_result)
        if hasattr(self,"orm_element_info"):
            self.orm_element_info.setVisible(has_orm_result)

        if hasattr(self, "results_tabs"):
            self.results_tabs.setVisible(has_result)

        if hasattr(self,"bpm_review_group"):
            self.bpm_review_group.setVisible(has_bpm_result)

    def _configure_result_tabs(self,kind):
        # These are final-result controls; visibility is managed by
        # _update_diagnostics_visibility() after a result exists.
        self.orm_column_row.setVisible(False)
        self.orm_view_row.setVisible(False)
        while self.results_tabs.count():self.results_tabs.removeTab(0)
        if kind=="orm":
            entries=((self.x_plot,"Full ORM"),(self.y_plot,"Direct H→X"),(self.orm_vv_plot,"Direct V→Y"),(self.mean_x_plot,"Coupling V→X"),(self.mean_y_plot,"Coupling H→Y"),(self.orm_model_plot,"Model ORM"),(self.orm_difference_plot,"Measured − Model"),(self.orm_column_plot,"Selected ORM column"),(self.orm_kick_plot,"Kick diagnostics"))
        elif kind=="dispersion":entries=((self.raw_x_plot,"RF orbit difference Δx_RF [mm]"),(self.raw_y_plot,"RF orbit difference Δy_RF [mm]"),(self.rf_shift_x_plot,"RF-induced horizontal shifts"),(self.rf_shift_y_plot,"RF-induced vertical shifts"),(self.x_plot,"Physical Dx [mm] — optional AT"),(self.y_plot,"Physical Dy [mm] — optional AT"))
        else:entries=((self.x_plot,"Horizontal BPM noise"),(self.y_plot,"Vertical BPM noise"),(self.mean_x_plot,"Mean horizontal orbit"),(self.mean_y_plot,"Mean vertical orbit"))
        for widget,label in entries:
            index=self.results_tabs.addTab(widget,label)
            self.results_tabs.setTabToolTip(index,label)
        if kind in {"orm","dispersion"}:self.results_tabs.setCurrentIndex(0)

    def _update_measurement_help(self, *_):
        if not hasattr(self,"measurement_help_text"): return
        if self.measurement_type.currentData()=="bpm_noise":
            self.measurement_help_title.setText("BPM Noise")
            self.measurement_help_text.setText(
                "Measure records many orbit readings without changing the machine. This measurement is read-only.\n\n"
                "For every BPM:\n"
                "• σx = standard deviation of its horizontal readings\n"
                "• σy = standard deviation of its vertical readings\n\n"
                "The mean orbit is shown separately. The standard deviations can be used as measurement weights in pyLOCO."
            )
            ddof=int(self.bpm_noise_convention.currentData()) if hasattr(self,"bpm_noise_convention") else 0
            label="population" if ddof==0 else "sample"
            self.measurement_help_convention.setText(f"Current convention: {label} standard deviation, NumPy ddof={ddof}.\nArray layout: (reading, BPM); standard deviation axis: 0."); self.measurement_help_convention.setVisible(True)
            return
        if self.measurement_type.currentData()=="orm":
            self.measurement_help_title.setText("Orbit Response Matrix (ORM)")
            self.measurement_help_text.setText("Measure changes one corrector at a time and records the resulting X/Y orbit change. The original corrector value is restored and checked between states and after each column.\n\nFor bipolar acquisition, each ORM column is mean orbit at K₀ + ΔK/2 minus mean orbit at K₀ − ΔK/2. The FIT-ready matrix is stored as orbit difference in metres [m]; the actual corrector separation is recorded separately in radians. Normalized response [m/rad] is available only as a display analysis and does not replace the FIT matrix.\n\nMatrix order: X BPM rows, then Y BPM rows; H corrector columns, then V corrector columns.")
            direction=self.orm_direction.currentData() if hasattr(self,"orm_direction") else "bipolar"
            convention={"bipolar":"K+ = K0 + ΔK/2; K− = K0 − ΔK/2\nORM column = orbit(+) − orbit(−)","positive":"K+ = K0 + ΔK\nORM column = orbit(+) − reference","negative":"K− = K0 − ΔK\nORM column = orbit(−) − reference"}.get(direction,"")
            self.measurement_help_convention.setText("Current convention:\n"+convention); self.measurement_help_convention.setVisible(True); return
        self.measurement_help_title.setText("Dispersion / RF response")
        self.measurement_help_text.setText(
            "Measure records the orbit at reference, positive and negative RF frequencies, then restores and checks the original RF.\n\n"
            "The stored measured_eta_x/y datasets are the signed RF-induced orbit differences in metres [m] used directly by the current pyLOCO FIT importer. "
            "Physical dispersion is a separate derived quantity, also dimensionally in metres [m], calculated only with a verified RF-to-momentum conversion from an AT reference lattice; it does not replace the FIT datasets."
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
        if getattr(self,"_displayed_orm_result",None) is not None:self._show_orm_result(self._displayed_orm_result)

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
        centered("pyLOCO Measure records the machine data needed for LOCO analysis: BPM noise, RF response and orbit-response matrices.")
        layout.addSpacing(8)
        centered("pyLOCO\nStorage Ring Optics Correction",object_name="aboutTitle")
        centered("pyLOCO fits measured accelerator response data to an accelerator model to diagnose and correct optics errors.\n\npyLOCO Measure produces measurement files and sessions that can be consumed directly by pyLOCO.")
        centered("pyLOCO Suite workflow: Measure → Fit and analyze → Review and correct.")
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
            verification = (
                " • post-restoration orbit verification enabled"
                if self.verify_restored_orbit.isChecked()
                else ""
            )
            self.step_instruction.setText(
                f"Automatic bipolar RF scan: {separation:g} Hz total "
                f"(±{step:g} Hz) • reference → positive → negative.\n"
                f"RESTORE (not a measurement state) — RF restoration protected{verification}."
            )
            self.start_button.setText("Start automatic dispersion measurement"); return
        if not specs:
            self.step_instruction.setText("Enter the nominal RF frequency to create the guided manual sequence."); self.start_button.setText("Nominal RF required"); return
        if self.dispersion_step_index < len(specs):
            label,frequency,title=specs[self.dispersion_step_index]
            offset=frequency-self._nominal_rf_value()
            action="Set/confirm the machine at nominal RF." if label=="reference" else f"Set RF externally to {frequency:.3f} Hz (requested offset {offset:+.3f} Hz)."
            evidence="pyLOCO will verify the backend RF readback before acquisition." if AdapterCapability.RF_READ in self.adapter.capabilities else "Enter the actual applied frequency above; it will be recorded as operator-entered and not independently verified."
            self.step_instruction.setText(f"STEP {self.dispersion_step_index+1} OF {total} — {title}\n{action}\n{evidence}")
            self.start_button.setText("Confirm actual RF and measure orbit")
        else:
            evidence="pyLOCO will verify the RF readback." if AdapterCapability.RF_READ in self.adapter.capabilities else "Enter the actual restored frequency above; independent readback is unavailable."
            orbit=" A diagnostic restored orbit will then be acquired." if self.verify_restored_orbit.isChecked() else ""
            self.step_instruction.setText(f"STEP {total} OF {total} — Restore RF\nRestore RF externally to nominal frequency {self._nominal_rf_value():.3f} Hz. {evidence}{orbit}")
            self.start_button.setText("Confirm and verify RF restoration")

    def _manual_rf_evidence(self, label, requested):
        """Return actual RF, provenance, and independent-verification state without writing RF."""
        if AdapterCapability.RF_READ in self.adapter.capabilities:
            try:
                actual=float(self.adapter.get_rf_frequency())
            except Exception as exc:
                text=self.manual_actual_rf.text().strip()
                try:actual=float(text)
                except ValueError as parse_exc:raise ValueError(f"RF readback failed: {exc}. Enter the actual applied frequency after confirming it externally.") from parse_exc
                source="operator_entry_unverified"; verified=False
            else:source="backend_readback"; verified=True
        else:
            text=self.manual_actual_rf.text().strip()
            try:actual=float(text)
            except ValueError as exc:raise ValueError("Enter the actual applied RF frequency in Hz before confirming this step.") from exc
            source="operator_entry_unverified"; verified=False
        if not np.isfinite(actual) or actual<=0:
            raise ValueError("The actual applied RF frequency must be a finite positive value.")
        nominal=self._nominal_rf_value()
        tolerance=max(1e-6,min(0.1,max(0.01,self._rf_offset_hz()*0.01)))
        if label in {"reference","reference_after","restoration"} and abs(actual-nominal)>tolerance:
            raise ValueError(f"RF is not at nominal: actual {actual:.6f} Hz, nominal {nominal:.6f} Hz (tolerance {tolerance:g} Hz).")
        if label=="positive" and actual<=nominal:
            raise ValueError("Positive RF state must be above the nominal frequency.")
        if label=="negative" and actual>=nominal:
            raise ValueError("Negative RF state must be below the nominal frequency.")
        return actual,source,verified

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
        self._model_orm_cache=None
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
        if orm:
            orbit_states=5 if self.orm_direction.currentData()=="bipolar" else 4; writes=4 if self.orm_direction.currentData()=="bipolar" else 3
            estimate=(len(self.selected_hcorrectors)+len(self.selected_vcorrectors))*(orbit_states*max(0,self.readings.value()-1)*self.delay.value()+writes*self.settling_delay.value())
        else:estimate=state_count*per_state
        self.duration.setText(f"approximately {estimate:.2f} s")
        output=self._resolved_output_directory()
        try:self._load_kick_arrays() if orm else None; kick_valid=True
        except Exception:kick_valid=False
        automatic_rf=dispersion and self.rf_control_mode.currentData()=="automatic"
        requires_connection=self.adapter_combo.currentData() in ("pysc", "pyaml"); connection_ok=not requires_connection or self.connection_verified
        valid=bool(connection_ok and self.selected_devices and self.measurement_name.text().strip() and self.output_directory.text().strip() and (not dispersion or automatic_rf or self._nominal_rf_value() is not None) and (not automatic_rf or AdapterCapability.RF_WRITE in self.adapter.capabilities) and (not orm or (self.selected_hcorrectors or self.selected_vcorrectors) and kick_valid and AdapterCapability.WRITE in self.adapter.capabilities))
        direction={"bipolar":"Bipolar ±Δf","positive":"One-sided +Δf","negative":"One-sided −Δf"}.get(self.dispersion_direction.currentData(),"—")
        try:kh,kv=self._load_kick_arrays()
        except Exception:kh=kv=np.array([])
        descriptor=InterfaceRegistry().descriptor(self.adapter_combo.currentData()); adapter_text=descriptor.badge
        if descriptor.key == "pyaml":
            adapter_text = self.status_badge.text()
        nominal=self._nominal_rf_value(); offset=self._rf_offset_hz(); negative="Waiting for RF readback" if nominal is None else f"{nominal-offset:.3f} Hz"; positive="Waiting for RF readback" if nominal is None else f"{nominal+offset:.3f} Hz"
        separation=2*offset if self.dispersion_direction.currentData()=="bipolar" else offset
        values={"measurement":"ORM" if orm else "Dispersion" if dispersion else "BPM noise","bpms":str(len(self.selected_devices)),"h_correctors":str(len(self.selected_hcorrectors)),"v_correctors":str(len(self.selected_vcorrectors)),"total_correctors":str(len(self.selected_hcorrectors)+len(self.selected_vcorrectors)),"kick_source":"Per-corrector file" if self.orm_kick_mode.currentData()=="file" else "Common","h_kick":f"{np.min(kh)*1e6:g}–{np.max(kh)*1e6:g} µrad" if kh.size else "Invalid","v_kick":f"{np.min(kv)*1e6:g}–{np.max(kv)*1e6:g} µrad" if kv.size else "Invalid","scaled_orm":"Saved FIT matrix: orbit difference [m]"+("; normalized analysis view enabled" if self.orm_normalized_analysis.isChecked() else ""),"orm_direction":self.orm_direction.currentText(),"rf_mode":"Automatic backend RF with restoration protection" if automatic_rf else "Manual (no writes)","nominal_rf":"Read from backend" if automatic_rf and nominal is None else f"{nominal:.3f} Hz" if nominal else "Required","rf_step_±δf":f"{offset:g} Hz per side","bipolar_separation":f"{separation:g} Hz total","negative_rf":negative,"positive_rf":positive,"direction":direction,"readings":str(self.readings.value()),"delay":f"{self.delay.value():g} s","settling_delay":f"{self.settling_delay.value():g} s","rf_states":", ".join(spec[0] for spec in self._dispersion_state_specs()) or "backend-derived","est_duration":f"{estimate:.2f} s","adapter":adapter_text,"output":str(output)}
        values["measurement_i/o"] = "Temporary magnet writes" if orm else "Temporary RF writes" if automatic_rf else "Read only — orbit reads only"
        values["noise_definition"] = f"{'Population' if self.bpm_noise_convention.currentData()==0 else 'Sample'} σ per BPM (ddof={self.bpm_noise_convention.currentData()})" if not (orm or dispersion) else "—"
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
        if reasons:self.start_block_reason.setText("Cannot start: "+"; ".join(reasons)+".")
        elif automatic_rf:self.start_block_reason.setText(f"Ready for automatic dispersion through {descriptor.badge.split(' • ')[0]} • RF WRITES AUTHORIZED; BPM orbit acquisition is read only.")
        else:self.start_block_reason.setText("Ready to acquire through "+descriptor.badge+".")
        for widget in getattr(self,"orm_plan_widgets",[]):widget.setVisible(orm)
        for widget in getattr(self,"dispersion_plan_widgets",[]):widget.setVisible(dispersion)
        for widget in getattr(self,"settling_plan_widgets",[]):widget.setVisible(dispersion or orm)
        self._update_persistent_header()
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
        self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat(f"Starting — 0 / {self.readings.value()}"); self.reading_status.setText("Starting…"); self.elapsed.setText("Elapsed: 0.00 s"); self.remaining.setText(f"Remaining: ~{max(0,self.readings.value()-1)*self.delay.value():.2f} s"); self.samples.setText(f"Samples: 0 / {self.readings.value()}"); self._reset_live_plot(); self.log.clear(); self.log.appendPlainText(f"Starting BPM-noise measurement through {self.status_badge.text()}…")
        self.thread=QThread(self); self.worker=AcquisitionWorker(BpmNoiseAcquirer(self.adapter,self.selected_devices,ddof=int(self.bpm_noise_convention.currentData())),self.readings.value(),self.delay.value(),self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed): signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    def _start_orm(self):
        if isinstance(self.adapter,PETRAReadOnlyAdapter):QMessageBox.warning(self,"ORM unavailable","ORM needs write access. PETRA is read only."); return
        self._refresh_corrector_preview("hcor"); self._refresh_corrector_preview("vcor")
        try:kick_h,kick_v=self._load_kick_arrays(); acquirer=ORMAcquirer(self.adapter,self.selected_devices,self.selected_hcorrectors,self.selected_vcorrectors)
        except Exception as exc:QMessageBox.warning(self,"Cannot start ORM",str(exc)); return
        self._orm_has_completed_column = False
        self._model_orm_cache=None
        if self.compare_reference.isChecked() and self.reference_model is not None:
            try:
                self._model_orm_cache=model_orm(self.reference_model,self.selected_devices,self.selected_hcorrectors,self.selected_vcorrectors,kick_h,kick_v,scaled=False)
            except Exception as exc:self.statusBar().showMessage(f"Reference-model ORM unavailable: {exc}")
        self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat("Preparing ORM"); self.reading_status.setText("Preparing first corrector…"); self.log.clear(); self.log.appendPlainText(f"Starting ORM through {self.status_badge.text()}; each original setpoint will be restored.")
        options={"direction":self.orm_direction.currentData(),"scaled":False,"readings":self.readings.value(),"delay_seconds":self.delay.value(),"settling_delay_seconds":self.settling_delay.value()}
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
        if state_index==0:self._live_dispersion_reference=(np.asarray(x,dtype=float).copy(),np.asarray(y,dtype=float).copy())
        self.progress.setValue(round(100*overall/overall_total)); self.progress.setFormat(f"{label} — orbit {current} / {total}"); self.reading_status.setText(f"Automatic RF: {label} — reading {current} / {total}"); self.samples.setText(f"Samples: {overall} / {overall_total}"); self.elapsed.setText(f"Elapsed: {elapsed:.2f} s"); self._update_live_orbit(x,y,title=preview,dispersion=True)

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
        if event.get("event")=="intermediate_restore":
            self.reading_status.setText(f"Corrector {current} / {total} — restoring original K0 before negative kick")
            self.progress.setFormat(f"Corrector {current} / {total} — restore K0 before K−")
            self.log.appendPlainText(f"{event.get('device')}: positive orbit acquired; restoring original K0 before applying K−.")
            return
        if event.get("event")=="intermediate_restored":
            error=(float(event["readback"])-float(event["original"]))*1e6
            self.reading_status.setText(f"Corrector {current} / {total} — original K0 restored and verified; negative kick may proceed")
            self.log.appendPlainText(f"{event.get('device')}: intermediate K0 restoration VERIFIED (readback error {error:+.9g} µrad).")
            return
        if event.get("event")=="final_restored":
            error=(float(event["readback"])-float(event["original"]))*1e6
            self.reading_status.setText(f"Corrector {current} / {total} — final K0 restored; acquiring return orbit")
            self.log.appendPlainText(f"{event.get('device')}: final K0 restoration VERIFIED (readback error {error:+.9g} µrad); acquiring diagnostic return orbit.")
            return
        if event.get("event")=="orbit":
            reading=int(event["reading"]); readings=int(event["readings"]); direction=self.orm_direction.currentData()
            if direction=="bipolar":phase,phase_count={"reference before kick":(0,5),"+kick":(1,5),"restored reference before −kick":(2,5),"−kick":(3,5),"final restored reference":(4,5)}.get(state,(0,5))
            else:phase,phase_count={"reference before kick":(0,4),"+kick":(1,4),"−kick":(1,4),"reference":(2,4),"final restored reference":(3,4)}.get(state,(0,4))
            fraction=((current-1)+(phase+reading/readings)/phase_count)/total
            self.reading_status.setText(f"Corrector {current} / {total} — {event.get('plane')} — {event.get('device')} — {state} — orbit {reading} / {readings}")
            self.samples.setText(
                f"Orbit reading: {reading} / {readings}"
            )

            # Show individual orbit traces until the first complete ORM
            # column exists. Afterwards keep the evolving matrix visible.
            if not getattr(
                self,
                "_orm_has_completed_column",
                False,
            ):
                self._update_live_orbit(
                    event["x"],
                    event["y"],
                    title=f"{event.get('device')} — {state}",
                )
        else:fraction=(current-1)/total; self.reading_status.setText(f"Corrector {current} / {total} — {event.get('plane','')} — {event.get('device','')} — {state}")
        self.progress.setValue(round(100*fraction)); self.progress.setFormat(f"Corrector {current} / {total}"); elapsed=float(event.get("elapsed",0)); self.elapsed.setText(f"Elapsed: {elapsed:.2f} s")
        if event.get("event")=="column":
            column=np.asarray(event["column"]); self.log.appendPlainText(f"Completed ORM column {current}/{total}; restoration pending.")
            model=getattr(self,"_model_orm_cache",None)
            if model is not None and current-1<model.shape[1]:
                metrics=comparison_metrics(column,model[:,current-1]); self.log.appendPlainText(f"Model comparison available — cosine {metrics['cosine_similarity']:.6f}; relative difference {metrics['relative_norm_difference']:.3%}.")
            self._show_orm_progress(np.asarray(event["matrix"]),column)

    def _start_dispersion_step(self):
        if self._nominal_rf_value() is None and AdapterCapability.RF_READ in self.adapter.capabilities:
            try:
                nominal=float(self.adapter.get_rf_frequency())
                if np.isfinite(nominal) and nominal>0:
                    self.nominal_rf.setText(f"{nominal:.12f}")
                    self._manual_nominal_source="backend_readback"
                    self.nominal_rf_source.setText("Source: backend RF readback (manual control; no writes)")
            except Exception as exc:
                self.log.appendPlainText(f"Nominal RF readback unavailable: {exc}")
        specs=self._dispersion_state_specs()
        if not specs:
            QMessageBox.warning(self,"Nominal RF required","Enter an explicit nominal RF frequency before measuring dispersion."); return
        if self.dispersion_step_index >= len(specs):
            try:actual,source,verified=self._manual_rf_evidence("restoration",self._nominal_rf_value())
            except ValueError as exc:QMessageBox.warning(self,"RF restoration not confirmed",str(exc)); return
            self._restored_rf_readback=actual; self._manual_restoration_source=source; self._manual_restoration_verified=verified
            if not self.verify_restored_orbit.isChecked():return self._complete_dispersion()
            label="reference_after"; frequency=self._nominal_rf_value(); title="Restored reference orbit"
            self._manual_restoration_pending=True
            self.log.appendPlainText(f"Operator confirmed restored RF {actual:.6f} Hz ({source}); acquiring diagnostic restored orbit.")
            return self._launch_manual_dispersion_state(label,frequency,title,actual,source,verified)
        if self.dispersion_step_index==0:
            self.dispersion_states=[]; self.dispersion_started_at=monotonic(); self.log.clear(); self._manual_restoration_pending=False
        label,frequency,title=specs[self.dispersion_step_index]
        try:actual,source,verified=self._manual_rf_evidence(label,frequency)
        except ValueError as exc:QMessageBox.warning(self,"Actual RF not confirmed",str(exc)); return
        if label=="reference":self._manual_nominal_source=source
        self.log.appendPlainText(f"Operator confirmed {label}: requested {frequency:.6f} Hz; actual {actual:.6f} Hz ({source}).")
        self.manual_actual_rf.clear()
        self._launch_manual_dispersion_state(label,frequency,title,actual,source,verified)

    def _launch_manual_dispersion_state(self,label,frequency,title,actual,source,verified):
        self.cancel_event=Event(); self.result=None
        specs=self._dispersion_state_specs(); total=(len(specs)+(1 if label=="reference_after" else 0))*self.readings.value(); completed=min(self.dispersion_step_index,len(specs))*self.readings.value(); self.progress.setValue(round(100*completed/total)); self.progress.setFormat(f"RF state {min(self.dispersion_step_index+1,len(specs))} / {len(specs)} — 0 / {self.readings.value()}"); self.reading_status.setText(f"{title} — settling/acquiring")
        self.thread=QThread(self); self.worker=DispersionAcquisitionWorker(DispersionStateAcquirer(self.adapter,self.selected_devices),label,frequency,actual,source,verified,self.readings.value(),self.delay.value(),self.settling_delay.value(),self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.completed.connect(self._on_dispersion_state_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed): signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    @Slot(int,int,float,object,object)
    def _on_progress(self,current,total,elapsed,x,y):
        if self.measurement_type.currentData()=="dispersion":
            states=max(1,len(self._dispersion_state_specs())); overall=self.dispersion_step_index*total+current; overall_total=states*total; remaining=(elapsed/current)*(overall_total-overall) if current else 0.0; self.progress.setValue(round(100*overall/overall_total)); self.progress.setFormat(f"RF state {self.dispersion_step_index+1} / {states} — orbit {current} / {total}"); self.reading_status.setText(f"RF state {self.dispersion_step_index+1} / {states} — Reading {current} / {total}"); self.samples.setText(f"Samples: {overall} / {overall_total}")
            if self.dispersion_step_index==0:self._live_dispersion_reference=(np.asarray(x,dtype=float).copy(),np.asarray(y,dtype=float).copy())
        else:
            remaining=(elapsed/current)*(total-current) if current else 0.0; self.progress.setValue(round(100*current/total)); self.progress.setFormat(f"Reading {current} / {total}"); self.reading_status.setText(f"Reading {current} / {total}"); self.samples.setText(f"Samples: {current} / {total}")
        self.elapsed.setText(f"Elapsed: {elapsed:.2f} s"); self.remaining.setText(f"Remaining: ~{remaining:.2f} s")
        if current in (1,total) or current % max(1,total//20)==0:self._update_live_orbit(x,y,dispersion=self.measurement_type.currentData()=="dispersion")

    @Slot(object)
    def _on_completed(self,result):
        if isinstance(result,ORMResult):
            self.result=result; total=len(result.correctors); self.reading_status.setText("Done — all correctors restored"); self.progress.setValue(100); self.progress.setFormat(f"Done — {total} / {total} correctors"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Correctors: {total} / {total}"); self.log.appendPlainText("ORM complete. Saving…")
            try:self._save_result(result)
            except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
            self._show_result(result); self._update_diagnostics_visibility(); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("ORM saved and validated for pyLOCO."); self._update_workflow_tabs(); return
        if isinstance(result,DispersionResult):
            self.result=result; self._restored_rf_readback=float(self.adapter.get_rf_frequency()); total=sum(state.orbits_x_m.shape[0] for state in result.states); self.reading_status.setText("Done — RF restored"); self.progress.setValue(100); self.progress.setFormat("Done — RF restored"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {total} / {total}"); self.log.appendPlainText("RF measurement complete. Saving…")
            try:self._save_result(result)
            except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
            self.dispersion_display.setCurrentIndex(self.dispersion_display.findData("raw")); self._show_result(result); self._update_diagnostics_visibility(); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self._update_workflow_tabs(); return
        self.result=result; total=result.orbits_x_m.shape[0]; self.reading_status.setText("Done"); self.progress.setValue(100); self.progress.setFormat(f"Done — {total} / {total}"); self.elapsed.setText(f"Elapsed: {result.elapsed_seconds:.2f} s"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {total} / {total}"); self.log.appendPlainText("Measurement complete. Saving…")
        try: self._save_result(result)
        except Exception as exc: self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
        self._show_result(result); self._update_diagnostics_visibility(); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("Saved and validated for pyLOCO."); self._update_workflow_tabs()

    def repeat_measurement(self):
        """Re-run current settings only after read-only connection/restoration checks."""
        previous=self.result
        try:
            if self.adapter_combo.currentData()!="mock" and hasattr(self.adapter,"test_connection"):self.adapter.test_connection()
            if isinstance(previous,DispersionResult):
                if AdapterCapability.RF_READ in self.adapter.capabilities:
                    current=float(self.adapter.get_rf_frequency())
                    if current != previous.nominal_rf_hz:raise RuntimeError(f"RF is not exactly restored: {current!r} != {previous.nominal_rf_hz!r}")
            if isinstance(previous,ORMResult):
                if any(status!="restored" for status in previous.restoration_status):raise RuntimeError("One or more corrector restorations were not confirmed")
                if previous.direction=="bipolar" and any(status!="restored" for status in previous.intermediate_restoration_status):raise RuntimeError("One or more between-kick restorations were not confirmed")
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
        self.dispersion_states.append(state)
        if state.label=="reference_after":
            self._manual_restoration_pending=False
            self.log.appendPlainText(f"Measured {state.label}: {state.orbits_x_m.shape[0]} diagnostic orbit samples retained.")
            return self._complete_dispersion()
        self.dispersion_step_index+=1
        self.log.appendPlainText(f"Measured {state.label}: {state.orbits_x_m.shape[0]} orbit samples retained.")
        self.reading_status.setText("State complete — continue when the next RF state is confirmed")
        self._update_dispersion_guide(); self._update_workflow_tabs()

    def _complete_dispersion(self):
        specs=self._dispersion_state_specs()
        measurement_states=[state for state in self.dispersion_states if state.label!="reference_after"]
        if len(measurement_states)!=len(specs):
            QMessageBox.warning(self,"Incomplete dispersion","Measure every requested RF state before confirming restoration."); return
        if any(not state.operator_confirmed or not np.isfinite(state.actual_rf_hz) for state in measurement_states):
            QMessageBox.warning(self,"Unconfirmed RF state","Every dispersion state needs a confirmed, finite actual RF frequency. Nothing was saved."); return
        result=DispersionResult(
            tuple(self.selected_devices),tuple(self.dispersion_states),self.dispersion_direction.currentData(),
            self._nominal_rf_value(),self._rf_offset_hz(),"verified_readback" if self._manual_restoration_verified else "confirmed_by_operator_unverified",
            0.0 if self.dispersion_started_at is None else monotonic()-self.dispersion_started_at,
        )
        self.result=result; self.reading_status.setText("Completed — RF restoration manually confirmed"); self.progress.setValue(100); self.progress.setFormat("Completed — RF restored"); self.remaining.setText("Remaining: 0.00 s"); self.samples.setText(f"Samples: {len(specs)*self.readings.value()} / {len(specs)*self.readings.value()}")
        try:self._save_result(result)
        except Exception as exc:self.log.appendPlainText(f"Saving failed: {exc}"); QMessageBox.critical(self,"Save failed",str(exc)); return
        self.dispersion_display.setCurrentIndex(self.dispersion_display.findData("raw")); self._show_result(result); self._update_diagnostics_visibility(); self._set_completed_result_view(True); self.start_button.setVisible(False); self.repeat_button.setVisible(True); self.validate_button.setEnabled(True); self.open_button.setEnabled(True); self.log.appendPlainText("RF restoration manually confirmed. Dispersion saved and validated for pyLOCO."); self._update_workflow_tabs(); self.refresh_plan()

    def _finish_thread(self,*_):
        thread=self.thread
        if thread: thread.quit(); thread.wait(3000); thread.deleteLater()
        self.thread=None; self.worker=None; self._set_acquisition_running(False); self.refresh_plan()

    def _manual_rf_restore_instruction(self):
        nominal=self._nominal_rf_value()
        return f" Restore RF externally to nominal {nominal:.6f} Hz and verify it before leaving the machine." if nominal is not None else " Restore RF externally to its original nominal value and verify it before leaving the machine."

    def cancel_acquisition(self):
        self.cancel_event.set(); self.log.appendPlainText("Stopping…")
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="manual":self.log.appendPlainText(self._manual_rf_restore_instruction().strip())
    def _on_cancelled(self,message):
        instruction=self._manual_rf_restore_instruction() if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="manual" else ""
        self.progress.setValue(0); self.progress.setFormat("Cancelled"); self.reading_status.setText("Cancelled — no measurement saved."+instruction); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message+instruction); self._update_workflow_tabs()
    def _on_failed(self,message):
        instruction=self._manual_rf_restore_instruction() if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="manual" else ""
        self.progress.setValue(0); self.progress.setFormat("Failed"); self.reading_status.setText("Failed — no measurement saved."+instruction); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message+instruction); QMessageBox.critical(self,"Measurement failed",message+instruction); self._update_workflow_tabs()

    def closeEvent(self,event):
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="manual" and self.result is None and (self.dispersion_step_index or self.thread is not None):
            self.cancel_event.set()
            QMessageBox.warning(self,"Manual RF restoration required","The manual dispersion workflow is incomplete."+self._manual_rf_restore_instruction())
        super().closeEvent(event)

    def _reset_live_plot(self):
        self.live_plot.clear(); axis=self.live_plot.figure.add_subplot(111); axis.set_title("Orbit Preview"); axis.set_xlabel("BPM selection position"); axis.set_ylabel("Orbit [µm]"); axis.grid(True,alpha=.25); self.live_plot.apply_theme()

    def _update_live_orbit(self,x,y,*,title="Orbit preview — most recent reading",dispersion=False):
        x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); self._live_orbit_absolute=(x.copy(),y.copy(),title,dispersion)
        relative=bool(dispersion and self.live_dispersion_view.currentData()=="relative" and self._live_dispersion_reference is not None)
        if relative:x=x-self._live_dispersion_reference[0]; y=y-self._live_dispersion_reference[1]; title=title+" — change from reference"
        self.live_plot.clear(); axis=self.live_plot.figure.add_subplot(111); positions=np.arange(len(x)); sx=x*1e6; sy=y*1e6
        line_x=axis.plot(positions,sx,"o-",label="X",color="#12BFC4",markersize=3,picker=5)[0]; line_y=axis.plot(positions,sy,"s-",label="Y",color="#F59E42",markersize=3,picker=5)[0]
        axis.set_title(title); axis.set_xlabel("BPM index (zero-based)"); axis.set_ylabel("Orbit change [µm]" if relative else "Orbit [µm]"); axis.grid(True,alpha=.25); axis.legend(ncol=2); self.live_plot.apply_theme(); theme=self.live_plot.theme(); annotation=axis.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=6); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False); highlight_x=axis.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=5); highlight_y=axis.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=5)
        pinned_x=axis.scatter([],[],s=55,facecolors="#12BFC4",edgecolors="#8B5CF6",linewidths=1.8,zorder=4); pinned_y=axis.scatter([],[],s=55,facecolors="#F59E42",edgecolors="#8B5CF6",linewidths=1.8,zorder=4)
        self._live_dispersion_plot_state={"axis":axis,"lines":(("X",line_x,sx),("Y",line_y,sy)),"annotation":annotation,"highlight_x":highlight_x,"highlight_y":highlight_y,"pinned_x":pinned_x,"pinned_y":pinned_y,"pin_lines":[]}
        self._refresh_dispersion_highlights()

    def _redisplay_live_orbit(self,*_):
        cached=getattr(self,"_live_orbit_absolute",None)
        if cached is not None:self._update_live_orbit(cached[0],cached[1],title=cached[2],dispersion=cached[3])

    def _live_dispersion_index_from_event(self,event):
        state=self._live_dispersion_plot_state
        if state is None or event.inaxes is not state["axis"]:return None,None
        for plane,line,values in state["lines"]:
            contains,details=line.contains(event)
            if contains and details.get("ind"):return int(details["ind"][0]),plane
        return None,None

    def _dispersion_bpm_text(self,index,plane=None):
        if not 0<=index<len(self.selected_devices):return ""
        state=self._live_dispersion_plot_state; values={p:v for p,_line,v in state["lines"]} if state else {}
        prefix=f"{self.selected_devices[index].name}\nBPM index {index} (zero-based)"
        if plane:return f"{prefix}\n{plane} {values[plane][index]:.6g} µm"
        return f"{prefix} · X {values.get('X',[np.nan])[index]:.6g} µm · Y {values.get('Y',[np.nan])[index]:.6g} µm"

    def _on_live_dispersion_hover(self,event):
        state=self._live_dispersion_plot_state
        if state is None:return
        index,plane=self._live_dispersion_index_from_event(event); annotation=state["annotation"]
        if index is None:annotation.set_visible(False)
        else:
            values={p:v for p,_line,v in state["lines"]}; value=values[plane][index]; self._place_plot_annotation(annotation,state["axis"],index,value); annotation.set_text(self._dispersion_bpm_text(index,plane)); annotation.set_visible(True)
        self.live_plot.canvas.draw_idle()

    def _on_live_dispersion_click(self,event):
        index,_plane=self._live_dispersion_index_from_event(event)
        if index is None:return
        self._dispersion_selected_bpm_index=index; text=self._dispersion_bpm_text(index); self.live_dispersion_selection.setText(text); self.dispersion_selected_details.setText(text.replace("\n"," · ")); self.live_dispersion_pin.setEnabled(True); self.dispersion_pin_button.setEnabled(True); label="Remove highlight" if index in self._dispersion_pinned_bpm_indices else "Keep highlighted"; self.live_dispersion_pin.setText(label); self.dispersion_pin_button.setText(label); self._refresh_dispersion_highlights()

    def _toggle_live_dispersion_pin(self):
        index=self._dispersion_selected_bpm_index
        if index is None:return
        if index in self._dispersion_pinned_bpm_indices:
            self._dispersion_pinned_bpm_indices.remove(index); self._dispersion_selected_bpm_index=None
            self.live_dispersion_selection.setText("Click a BPM point to inspect it"); self.dispersion_selected_details.setText("Click a plotted BPM to inspect it")
            for button in (self.live_dispersion_pin,self.dispersion_pin_button):button.setEnabled(False); button.setText("Keep highlighted")
            self._refresh_dispersion_highlights(); return
        self._dispersion_pinned_bpm_indices.add(index)
        label="Remove highlight" if index in self._dispersion_pinned_bpm_indices else "Keep highlighted"; self.live_dispersion_pin.setText(label); self.dispersion_pin_button.setText(label); self._refresh_dispersion_highlights()

    def _clear_dispersion_pins(self):self._dispersion_pinned_bpm_indices.clear(); self._refresh_dispersion_highlights()

    def _clear_dispersion_selection(self):
        self._dispersion_selected_bpm_index=None; self._dispersion_pinned_bpm_indices.clear()
        self.live_dispersion_selection.setText("Click a BPM point to inspect it"); self.dispersion_selected_details.setText("Click a plotted BPM to inspect it")
        for button in (self.live_dispersion_pin,self.dispersion_pin_button):button.setEnabled(False); button.setText("Keep highlighted")
        self._refresh_dispersion_highlights()

    def _refresh_dispersion_highlights(self):
        states=([self._live_dispersion_plot_state] if self._live_dispersion_plot_state else [])+list(self._dispersion_plot_state.values())
        for state in states:
            for line in state.get("pin_lines",[]):line.remove()
            state["pin_lines"]=[state["axis"].axvline(i,color="#8B5CF6",linestyle="--",linewidth=1.0,alpha=.55,zorder=0) for i in sorted(self._dispersion_pinned_bpm_indices)]
            pins=sorted(self._dispersion_pinned_bpm_indices)
            if "pinned" in state:
                pin_values=state.get("active_values",state.get("value_by_index",{})); state["pinned"].set_offsets([[i,pin_values[i]] for i in pins] if pins else np.empty((0,2)))
            elif "pinned_x" in state:
                values={p:v for p,_line,v in state["lines"]}; state["pinned_x"].set_offsets([[i,values["X"][i]] for i in pins] if pins else np.empty((0,2))); state["pinned_y"].set_offsets([[i,values["Y"][i]] for i in pins] if pins else np.empty((0,2)))
            index=self._dispersion_selected_bpm_index
            if index is not None:
                if "highlight" in state:
                    selected_values=state.get("active_values",state.get("value_by_index",{})); state["highlight"].set_offsets([[index,selected_values[index]]])
                elif "highlight_x" in state:
                    values={p:v for p,_line,v in state["lines"]}; state["highlight_x"].set_offsets([[index,values["X"][index]]]); state["highlight_y"].set_offsets([[index,values["Y"][index]]])
            else:
                if "highlight" in state:state["highlight"].set_offsets(np.empty((0,2)))
                elif "highlight_x" in state:state["highlight_x"].set_offsets(np.empty((0,2))); state["highlight_y"].set_offsets(np.empty((0,2)))
            state.get("canvas",self.live_plot).canvas.draw_idle()

    def save_log(self):
        filename=QFileDialog.getSaveFileName(self,"Save measurement log","pyloco-measure.log","Log files (*.log *.txt)")[0]
        if filename: Path(filename).write_text(self.log.toPlainText()+"\n",encoding="utf-8")

    def open_output_folder(self):
        path = self.saved_measurement_path.parent if self.saved_measurement_path is not None else self._resolved_output_directory()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _save_result(self,result):
        output=self._resolved_output_directory(); output.mkdir(parents=True,exist_ok=True); safe=re.sub(r"[^A-Za-z0-9_.-]+","_",self.measurement_name.text().strip()).strip("_") or "measurement"; stamp=datetime.now().strftime("%Y%m%d-%H%M%S"); measurement=output/f"{safe}-{stamp}.h5"; is_dispersion=isinstance(result,DispersionResult)
        descriptor=InterfaceRegistry().descriptor(self.adapter_combo.currentData())
        identity,profile=self._machine_identity(); profile_metadata={}
        if descriptor.key == "pyaml":
            options = {key: editor.text().strip() for key, editor in self.pyaml_fields.items()}
            profile_metadata = {"pyaml_options": options, "pyaml_mode": options["mode"],
                                "backend_badge": self.status_badge.text(),
                                "backend_environment": options["mode"].upper(),
                                "backend_real_machine": options["mode"] == "live",
                                "magnet_writes_enabled": AdapterCapability.WRITE in self.adapter.capabilities,
                                "rf_writes_enabled": AdapterCapability.RF_WRITE in self.adapter.capabilities,
                                "corrector_control_unit": "rad" if AdapterCapability.WRITE in self.adapter.capabilities else None,
                                "corrector_unit_source": "pyAML strength.unit converted to SI rad by Measure" if AdapterCapability.WRITE in self.adapter.capabilities else "not verified"}
        if profile is not None:
            profile_metadata={"machine":profile.machine,"profile_key":profile.key,"profile":profile.scenario,"profile_label":profile.label,"profile_manifest":str(profile.manifest_path),"lattice_file":str(profile.resolve("lattice_file")),"lattice_provenance":profile.configuration.get("provenance",{}),"random_seed":profile.configuration.get("random_seed")}
        model_metadata=self.reference_model.provenance() if self.reference_model is not None else {"reference_model_available":False}
        actual_readings=(result.raw_state_a_m.shape[1] if isinstance(result,ORMResult) and result.raw_state_a_m.ndim==3 else self.readings.value())
        metadata={"measurement_name":self.measurement_name.text(),"measurement_label":self.measurement_label.text(),"acquisition_timestamp":datetime.now().astimezone().isoformat(),"operator_comments":self.comments.toPlainText(),"adapter":self.adapter_combo.currentText(),"backend_key":descriptor.key,"backend_badge":descriptor.badge,"backend_environment":descriptor.environment,"backend_real_machine":descriptor.real_machine,"machine_identity":identity,**profile_metadata,**model_metadata,"compare_with_reference_model":self.compare_reference.isChecked(),"pysc_machine_profile":self.pysc_profile_combo.currentData() if descriptor.key=="pysc" else None,"readings_per_state":actual_readings,"delay_seconds":self.delay.value(),"settling_delay_seconds":self.settling_delay.value() if isinstance(result,(ORMResult,DispersionResult)) else 0.0,"elapsed_seconds":result.elapsed_seconds}
        if isinstance(result,ORMResult):
            nh=len(result.horizontal_correctors); diagnostics={}
            if result.raw_state_a_m.ndim==3 and result.raw_state_a_m.shape[1]>1:
                diagnostics={"state_a_first_sample_minus_later_mean_m":result.raw_state_a_m[:,0,:]-np.mean(result.raw_state_a_m[:,1:,:],axis=1),"state_b_first_sample_minus_later_mean_m":result.raw_state_b_m[:,0,:]-np.mean(result.raw_state_b_m[:,1:,:],axis=1)}
            write_orm(measurement,response_matrix=result.response_matrix,bpm_names=[d.name for d in result.bpms],horizontal_corrector_names=[d.name for d in result.horizontal_correctors],vertical_corrector_names=[d.name for d in result.vertical_correctors],requested_kick_h_rad=result.requested_kicks_rad[:nh],requested_kick_v_rad=result.requested_kicks_rad[nh:],actual_kick_h_rad=result.effective_kicks_rad[:nh],actual_kick_v_rad=result.effective_kicks_rad[nh:],orbit_plus_m=result.raw_state_a_m,orbit_minus_m=result.raw_state_b_m,reference_before_m=result.raw_reference_before_m,reference_intermediate_m=result.raw_reference_intermediate_m,reference_final_m=result.raw_reference_final_m,scaled=result.scaled,direction=result.direction,original_setpoints_rad=result.original_setpoints_rad,requested_state_a_rad=result.requested_state_a_rad,requested_state_b_rad=result.requested_state_b_rad,actual_state_a_rad=result.actual_state_a_rad,actual_state_b_rad=result.actual_state_b_rad,intermediate_restored_setpoints_rad=result.intermediate_restored_setpoints_rad,intermediate_restoration_status=result.intermediate_restoration_status,final_setpoints_rad=result.final_setpoints_rad,timestamps_plus_s=result.timestamps_state_a_s,timestamps_minus_s=result.timestamps_state_b_s,restoration_status=result.restoration_status,diagnostics=diagnostics,metadata={**metadata,"orm_validation_convention":"historical_petra_bipolar_v3","kick_convention":"total bipolar delta_K; K+=K0+delta_K/2; restore/verify K0 and acquire diagnostic reference; K-=K0-delta_K/2; restore/verify K0 and acquire diagnostic reference","restored_orbit_role":"reference_before, reference_intermediate and reference_final are diagnostic-only and excluded from ORM calculation","subtraction_convention":"mean(positive)-mean(negative)","normalization_convention":"divide each column by actual K+ minus K- readback only when scaled","response_matrix_unit":"m/rad" if result.scaled else "m","row_order":"horizontal_bpms,vertical_bpms","column_order":"horizontal_correctors,vertical_correctors","transient_diagnostic":"first orbit sample minus mean of later samples for each applied corrector state; diagnostic only; no samples discarded","selected_bpm_names":[d.name for d in result.bpms],"selected_horizontal_corrector_names":[d.name for d in result.horizontal_correctors],"selected_vertical_corrector_names":[d.name for d in result.vertical_correctors]})
            model_matrix=getattr(self,"_model_orm_cache",None)
            if model_matrix is not None and self.reference_model is not None:
                store_reference_model_arrays(measurement,{"orm":model_matrix,"orm_difference":result.response_matrix-model_matrix},units={"orm":"m/rad" if result.scaled else "m","orm_difference":"m/rad" if result.scaled else "m"},provenance=self.reference_model.provenance())
        elif is_dispersion:
            automatic=self.rf_control_mode.currentData()=="automatic"; states=result.states; metadata.update({"rf_control_mode":"automatic" if automatic else "manual","nominal_rf_hz":result.nominal_rf_hz,"nominal_rf_source":"backend_readback" if automatic else self._manual_nominal_source,"requested_rf_offset_hz":result.requested_offset_hz,"direction":result.direction,"canonical_measured_eta_definition":"mean_orbit_negative - mean_orbit_positive; historical PETRA III pyLOCO RF-response convention; reference_before and reference_after are excluded","canonical_measured_eta_unit":"m","rf_difference_sign_convention":"negative_minus_positive","rf_normalized_response_unit":"m/Hz","actual_rf_available":bool(all(np.isfinite(s.actual_rf_hz) for s in states)),"rf_acquisition_order":[s.label for s in states],"rf_requested_frequencies_hz":[float(s.requested_rf_hz) for s in states],"rf_actual_frequencies_hz":[float(s.actual_rf_hz) for s in states],"rf_actual_frequency_sources":[s.actual_rf_source for s in states],"rf_readback_verified":[bool(s.rf_readback_verified) for s in states],"operator_confirmed_states":[bool(s.operator_confirmed) for s in states],"restoration_status":result.restoration_status,"rf_restoration_source":"backend_readback" if automatic else self._manual_restoration_source,"rf_restoration_readback_verified":True if automatic else self._manual_restoration_verified,"settling_delay_seconds":self.settling_delay.value(),"dispersion_measurement_states":"reference_before,positive,negative","rf_restoration_is_measurement_state":False,"verify_restored_orbit":any(state.label=="reference_after" for state in states),"reference_after_role":"post-restoration diagnostic only; excluded from physical dispersion and RF orbit difference"})
            state_rf={state.label:float(state.actual_rf_hz) for state in states}; restored=float(getattr(self,"_restored_rf_readback",np.nan)); metadata.update({"rf_original_hz":result.nominal_rf_hz,"rf_positive_hz":state_rf.get("positive",np.nan),"rf_negative_hz":state_rf.get("negative",np.nan),"rf_restored_hz":restored,"rf_restoration_difference_hz":restored-result.nominal_rf_hz if np.isfinite(restored) else np.nan})
            diagnostics={"raw_orbit_difference_x_m":result.measured_eta_x,"raw_orbit_difference_y_m":result.measured_eta_y}
            if all(state.orbits_x_m.shape[0]>1 for state in states):
                diagnostics.update({"state_first_sample_minus_later_mean_x_m":np.stack([state.orbits_x_m[0]-np.mean(state.orbits_x_m[1:],axis=0) for state in states]),"state_first_sample_minus_later_mean_y_m":np.stack([state.orbits_y_m[0]-np.mean(state.orbits_y_m[1:],axis=0) for state in states])})
                metadata["state_transition_diagnostic"]="first orbit sample minus mean of later samples for each RF state; diagnostic only, raw data and fitting inputs unchanged"
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
            write_dispersion(measurement,measured_eta_x=result.measured_eta_x,measured_eta_y=result.measured_eta_y,bpm_names=[d.name for d in result.devices],rf_frequency_hz=[s.requested_rf_hz for s in states],rf_setpoint_hz=[s.requested_rf_hz for s in states],rf_readback_hz=[s.actual_rf_hz for s in states],raw_orbits_x_m=np.stack([s.orbits_x_m for s in states]),raw_orbits_y_m=np.stack([s.orbits_y_m for s in states]),rf_step_hz=result.canonical_rf_step_hz,bidirectional=result.direction=="bipolar",metadata=metadata,diagnostics=diagnostics,state_labels=saved_labels,operator_confirmed=[s.operator_confirmed for s in states],rf_actual_sources=[s.actual_rf_source for s in states],rf_readback_verified=[s.rf_readback_verified for s in states],sample_timestamps_s=np.stack([s.timestamps_s for s in states]),restoration_status=result.restoration_status)
            if self.reference_model is not None and self.compare_reference.isChecked():
                try:
                    model_x,model_y=model_dispersion(self.reference_model,result.devices)
                    store_reference_model_arrays(measurement,{"dispersion_x_m":model_x,"dispersion_y_m":model_y},units={"dispersion_x_m":"m","dispersion_y_m":"m"},provenance=self.reference_model.provenance())
                except Exception:pass
        else:
            convention="population" if result.ddof==0 else "sample"; metadata.update({"readings":self.readings.value(),"selected_bpm_names":[d.name for d in result.devices],"bpm_noise_statistic":f"{convention} standard deviation across readings per BPM","bpm_noise_ddof":result.ddof,"bpm_noise_unit":"m"}); write_bpm_noise(measurement,noise_x_m=result.noise_x_m,noise_y_m=result.noise_y_m,bpm_names=[d.name for d in result.devices],raw_orbits_x_m=result.orbits_x_m,raw_orbits_y_m=result.orbits_y_m,ddof=result.ddof,metadata=metadata)
        validate_measurement_file(measurement); manifest=output/"measurement-session.pyloco-session.json"
        entries=[]
        role="orm" if isinstance(result,ORMResult) else "dispersion" if is_dispersion else "bpm_noise"

        # A newly opened Measure window or New project starts a genuinely
        # fresh logical measurement session. Do not inherit roles from an
        # older manifest that happens to exist in the output directory.
        fresh_session = getattr(self, "_fresh_measurement_session", True)

        if manifest.exists() and not fresh_session:
            entries=list(load_session(manifest,validate_files=False).files)
            entries=[e for e in entries if e.role!=role]

        options={"dataset":"response_matrix"} if isinstance(result,ORMResult) else {"horizontal_dataset":"measured_eta_x","vertical_dataset":"measured_eta_y"} if is_dispersion else {"horizontal_dataset":"Noise_BPMx","vertical_dataset":"Noise_BPMy"}
        entries.append(SessionFile(role,measurement.name,options))

        session=MeasurementSession(session_id=output.name or safe,files=tuple(entries),metadata={"label":self.measurement_label.text(),"updated":datetime.now().isoformat(timespec="seconds"),"machine_identity":identity,**profile_metadata})
        save_session(manifest,session)

        # From this point onward this window owns the new manifest, so
        # Dispersion/ORM may accumulate alongside the first measurement.
        self._fresh_measurement_session = False
        self.saved_measurement_path=measurement; self.saved_session_path=manifest; self.paths.setText(f"SAVED ✓\nMeasurement file: {measurement}\nSession manifest: {manifest}"); self.paths.setToolTip(str(measurement)); self.paths.setObjectName("savedPath"); self.paths.style().unpolish(self.paths); self.paths.style().polish(self.paths); self.open_output_button.setEnabled(True); self._refresh_review_page(); self.statusBar().showMessage(f"Saved: {measurement}")

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

    def _bpm_details_text(self, index):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult) or not 0 <= index < len(result.devices):return ""
        status=[]
        review_status=self._bpm_review_statuses.get(index,"Unreviewed")
        if review_status!="Unreviewed":status.append(review_status)
        if index in self._threshold_flagged_bpm_indices:status.append("threshold flag")
        suffix=f" · Status: {', '.join(status)}" if status else ""
        return (f"{result.devices[index].name} · BPM index {index} (zero-based) · "
                f"X noise {result.noise_x_m[index]*1e9:.3f} nm · "
                f"Y noise {result.noise_y_m[index]*1e9:.3f} nm{suffix}")

    def _bpm_tooltip_text(self,index):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult) or not 0 <= index < len(result.devices):return ""
        return (f"{result.devices[index].name}\n"
                f"BPM index {index} (zero-based)\n"
                f"X noise {result.noise_x_m[index]*1e9:.3f} nm   ·   Y noise {result.noise_y_m[index]*1e9:.3f} nm")

    @staticmethod
    def _place_plot_annotation(annotation,axis,x,y):
        xmin,xmax=axis.get_xlim(); ymin,ymax=axis.get_ylim()
        right=x > (xmin+xmax)/2; high=y > ymin+.72*(ymax-ymin)
        annotation.xy=(x,y); annotation.set_position((-12,-14) if right and high else (-12,14) if right else (12,-14) if high else (12,14)); annotation.set_horizontalalignment("right" if right else "left"); annotation.set_verticalalignment("top" if high else "bottom")

    def _populate_bpm_review_table(self,result):
        self._selected_bpm_index=None; self._manual_bad_bpm_indices.clear(); self._threshold_flagged_bpm_indices.clear(); self._pinned_bpm_indices.clear(); self._bpm_review_notes.clear(); self._saved_bpm_review_notes.clear(); self._bpm_review_statuses.clear()
        table=self.bpm_review_table; table.blockSignals(True); table.setSortingEnabled(False); table.setRowCount(len(result.devices))
        for index,device in enumerate(result.devices):
            short_name=device.name.rstrip("/").rsplit("/",1)[-1] or device.name
            values=(f"{short_name}  [{index}]",float(result.noise_x_m[index]*1e9),float(result.noise_y_m[index]*1e9),"","Unreviewed","")
            for column,value in enumerate(values):
                item=QTableWidgetItem(); item.setData(Qt.DisplayRole,value); item.setData(Qt.UserRole,index)
                if column==0:item.setData(Qt.UserRole+1,device.name); item.setToolTip(device.name)
                if column==3:item.setFlags(item.flags()|Qt.ItemIsUserCheckable); item.setCheckState(Qt.Unchecked)
                table.setItem(index,column,item)
        table.setSortingEnabled(True); table.blockSignals(False)
        self._refresh_bpm_review_table_states(); self.bpm_review_note.clear(); self.bpm_review_note.setEnabled(False); self.bpm_review_status.setEnabled(False); self.bpm_review_highlight.setEnabled(False); self._set_bpm_review_dirty(False)
        self.bpm_review_details.setText("Hover or click a BPM point to identify it. Indices are zero-based.")
        self._update_bpm_review_group_title()

    def _selected_bpm_review_changed(self,*_):
        if self._loading_bpm_review_panel:return
        index=self._selected_bpm_index
        if index is None:return
        status=self.bpm_review_status.currentText()
        if status=="Unreviewed":self._bpm_review_statuses.pop(index,None)
        else:self._bpm_review_statuses[index]=status
        self._manual_bad_bpm_indices={i for i,value in self._bpm_review_statuses.items() if value=="Bad"}
        note=self.bpm_review_note.text().strip()
        if note:self._bpm_review_notes[index]=note
        else:self._bpm_review_notes.pop(index,None)
        if self.bpm_review_highlight.isChecked():self._pinned_bpm_indices.add(index)
        else:self._pinned_bpm_indices.discard(index)
        self._refresh_bpm_pin_views(); self._update_bpm_review_group_title(); self.bpm_review_details.setText(self._bpm_details_text(index)); self._set_bpm_review_dirty(True)

    def _toggle_selected_bpm_pin(self):
        index=self._selected_bpm_index
        if index is None:return
        if index in self._pinned_bpm_indices:self._pinned_bpm_indices.remove(index)
        else:self._pinned_bpm_indices.add(index)
        self._refresh_bpm_pin_views(); self._set_bpm_review_dirty(True)

    def _clear_bpm_pins(self):
        if not self._pinned_bpm_indices:return
        self._pinned_bpm_indices.clear(); self._refresh_bpm_pin_views(); self._set_bpm_review_dirty(True)

    def _refresh_bpm_pin_views(self):
        for state in self._bpm_plot_state.values():
            for artist in state.get("pin_lines",[]):artist.remove()
            state["pin_lines"]=[]
            offsets=[]
            for index in sorted(self._pinned_bpm_indices):
                if index not in state["value_by_index"]:continue
                line=state["axis"].axvline(index,color="#8B5CF6",linestyle="--",linewidth=1.0,alpha=.45,zorder=0)
                state["pin_lines"].append(line); offsets.append([index,state["value_by_index"][index]])
            state["pinned"].set_offsets(np.asarray(offsets) if offsets else np.empty((0,2))); state["canvas"].canvas.draw_idle()
        self._refresh_bpm_review_table_states()

    def _bpm_review_index_for_row(self,row):
        item=self.bpm_review_table.item(row,0)
        return None if item is None else int(item.data(Qt.UserRole))

    def _filter_bpm_review_table(self,text):
        self._apply_bpm_review_filter()

    def _apply_bpm_review_filter(self,*_):
        query=self.bpm_review_search.text().strip().casefold(); status_filter=self.bpm_review_status_filter.currentData() or "all"; matches=0; total=self.bpm_review_table.rowCount()
        for row in range(self.bpm_review_table.rowCount()):
            index=self._bpm_review_index_for_row(row); item=self.bpm_review_table.item(row,0); name=item.text().casefold(); full_name=str(item.data(Qt.UserRole+1) or "").casefold()
            search_match=not query or query in name or query in full_name or query==str(index); status=self._bpm_review_statuses.get(index,"Unreviewed"); status_match=status_filter=="all" or status_filter=="bad_suspect" and status in {"Bad","Suspect"} or status==status_filter; visible=search_match and status_match; self.bpm_review_table.setRowHidden(row,not visible); matches+=int(visible)
        self.bpm_review_match_count.setText(f"{matches} of {total} BPMs"); self.bpm_review_empty.setVisible(total>0 and matches==0)

    def _bpm_review_table_selection_changed(self):
        rows=self.bpm_review_table.selectionModel().selectedRows()
        if rows:
            index=self._bpm_review_index_for_row(rows[0].row())
            if index is not None:self._select_bpm(index,source="table")

    def _bpm_review_item_changed(self,item):
        index=int(item.data(Qt.UserRole))
        if item.column()==3:
            if item.checkState()==Qt.Checked:self._pinned_bpm_indices.add(index)
            else:self._pinned_bpm_indices.discard(index)
            self._refresh_bpm_pin_views(); self._set_bpm_review_dirty(True)
            if index==self._selected_bpm_index:
                self._loading_bpm_review_panel=True; self.bpm_review_highlight.setChecked(index in self._pinned_bpm_indices); self._loading_bpm_review_panel=False

    def _refresh_bpm_review_table_states(self):
        table=self.bpm_review_table; table.blockSignals(True)
        for row in range(table.rowCount()):
            index=self._bpm_review_index_for_row(row)
            table.item(row,3).setCheckState(Qt.Checked if index in self._pinned_bpm_indices else Qt.Unchecked)
            name=table.item(row,0); result=getattr(self,"result",None)
            if isinstance(result,BpmNoiseResult):
                device_name=result.devices[index].name; short_name=device_name.rstrip("/").rsplit("/",1)[-1] or device_name; name.setText(f"{short_name}  [{index}]"); name.setToolTip(device_name); name.setData(Qt.UserRole+1,device_name)
            status=table.item(row,4); status.setText(self._bpm_review_statuses.get(index,"Unreviewed")); status.setToolTip("Threshold flag also active" if index in self._threshold_flagged_bpm_indices else "")
            note=self._bpm_review_notes.get(index,""); preview=note if len(note)<=32 else note[:31].rstrip()+"…"; note_item=table.item(row,5); note_item.setText(preview); note_item.setToolTip(note or "No review note")
        table.blockSignals(False); self.bpm_clear_pins_button.setEnabled(bool(self._pinned_bpm_indices)); self._apply_bpm_review_filter()

    def _threshold_review_changed(self,*_):
        self._update_bpm_review_flags(); self._set_bpm_review_dirty(True)

    def _update_bpm_review_flags(self,*_):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult):return
        tx=self.bpm_threshold_x.value(); ty=self.bpm_threshold_y.value()
        self._threshold_flagged_bpm_indices={i for i in range(len(result.devices)) if (tx>0 and result.noise_x_m[i]*1e9>tx) or (ty>0 and result.noise_y_m[i]*1e9>ty)}
        self._refresh_bpm_review_table_states(); self._update_bpm_review_group_title()
        if self._selected_bpm_index is not None:self.bpm_review_details.setText(self._bpm_details_text(self._selected_bpm_index))

    def _update_bpm_review_group_title(self):
        count=len({index for index,status in self._bpm_review_statuses.items() if status in {"Suspect","Bad"}}|self._threshold_flagged_bpm_indices)
        self.bpm_review_group.setTitle(f"BPM identification and review — {count} flagged, no automatic FIT exclusion")

    def _set_bpm_review_dirty(self,dirty):
        self._bpm_review_dirty=bool(dirty); self.bpm_review_save_state.setText("Unsaved review changes" if dirty else "All review changes saved"); self.bpm_review_save_state.setObjectName("validationMissing" if dirty else "validationOk"); self.bpm_review_save_state.style().unpolish(self.bpm_review_save_state); self.bpm_review_save_state.style().polish(self.bpm_review_save_state); self.bpm_save_review_button.setEnabled(bool(dirty))

    def _save_bpm_review(self):
        path=getattr(self,"saved_measurement_path",None)
        if path is None or not Path(path).exists():
            QMessageBox.critical(self,"Cannot save BPM review","The BPM-noise measurement file is not available. Review changes remain unsaved."); self._set_bpm_review_dirty(True); return False
        try:write_bpm_noise_review_annotations(path,manual_bad_bpm_indices=sorted(self._manual_bad_bpm_indices),threshold_flagged_bpm_indices=sorted(self._threshold_flagged_bpm_indices),pinned_bpm_indices=sorted(self._pinned_bpm_indices),review_notes=self._bpm_review_notes,review_statuses=self._bpm_review_statuses,horizontal_threshold_nm=self.bpm_threshold_x.value() or None,vertical_threshold_nm=self.bpm_threshold_y.value() or None)
        except Exception as exc:
            QMessageBox.critical(self,"Cannot save BPM review",f"The review could not be written to:\n{path}\n\n{exc}\n\nReview changes remain unsaved."); self._set_bpm_review_dirty(True); return False
        self._saved_bpm_review_notes=dict(self._bpm_review_notes); self._refresh_bpm_review_table_states(); self._set_bpm_review_dirty(False); self.statusBar().showMessage(f"BPM review saved: {path}"); return True

    def _load_bpm_review(self,path):
        data=read_bpm_noise_review_annotations(path); result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult):return
        if data["bpm_names"]!=[device.name for device in result.devices]:raise ValueError("Stored BPM review identity/order does not match the displayed measurement")
        self._pinned_bpm_indices=set(data["pinned_bpm_indices"]); self._threshold_flagged_bpm_indices=set(data["threshold_flagged_bpm_indices"]); self._bpm_review_notes=dict(data["review_notes"]); self._saved_bpm_review_notes=dict(self._bpm_review_notes); self._bpm_review_statuses=dict(data["review_statuses"]); self._manual_bad_bpm_indices={i for i,status in self._bpm_review_statuses.items() if status=="Bad"}
        self.bpm_threshold_x.blockSignals(True); self.bpm_threshold_y.blockSignals(True); self.bpm_threshold_x.setValue(data["horizontal_threshold_nm"] or 0); self.bpm_threshold_y.setValue(data["vertical_threshold_nm"] or 0); self.bpm_threshold_x.blockSignals(False); self.bpm_threshold_y.blockSignals(False)
        self._refresh_bpm_pin_views(); self._refresh_bpm_review_table_states(); self._update_bpm_review_group_title(); self._set_bpm_review_dirty(False)

    def _export_bpm_review_csv(self):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult):return
        path=QFileDialog.getSaveFileName(self,"Export BPM review","bpm-review.csv","CSV files (*.csv)")[0]
        if not path:return
        try:
            with open(path,"w",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream); writer.writerow(["bpm_name","original_zero_based_index","horizontal_noise_nm","vertical_noise_nm","highlighted","review_status","threshold_flagged","notes"])
                for index,device in enumerate(result.devices):writer.writerow([device.name,index,f"{result.noise_x_m[index]*1e9:.9g}",f"{result.noise_y_m[index]*1e9:.9g}",index in self._pinned_bpm_indices,self._bpm_review_statuses.get(index,"Unreviewed"),index in self._threshold_flagged_bpm_indices,self._bpm_review_notes.get(index,"")])
        except Exception as exc:QMessageBox.critical(self,"Cannot export BPM review",str(exc)); return
        self.statusBar().showMessage(f"BPM review CSV exported: {path}")

    def _bpm_index_from_event(self,plane,event):
        state=self._bpm_plot_state.get(plane)
        if state is None or event.inaxes is not state["axis"]:return None
        contains,details=state["points"].contains(event)
        if not contains or not details.get("ind"):return None
        return int(state["indices"][int(details["ind"][0])])

    def _on_bpm_plot_hover(self,plane,event):
        state=self._bpm_plot_state.get(plane)
        if state is None:return
        index=self._bpm_index_from_event(plane,event); annotation=state["annotation"]
        if index is None:
            annotation.set_visible(False); state["canvas"].canvas.draw_idle(); return
        value=state["value_by_index"][index]; self._place_plot_annotation(annotation,state["axis"],index,value); annotation.set_text(self._bpm_tooltip_text(index)); annotation.set_visible(True); state["canvas"].canvas.draw_idle()

    def _on_bpm_plot_click(self,plane,event):
        index=self._bpm_index_from_event(plane,event)
        if index is not None:self._select_bpm(index,source="plot")

    def _mean_details_text(self,index):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult) or not 0 <= index < len(result.devices):return ""
        return (f"{result.devices[index].name} · BPM index {index} (zero-based) · "
                f"mean X orbit {result.mean_x_m[index]*1e6:.3f} µm · "
                f"mean Y orbit {result.mean_y_m[index]*1e6:.3f} µm")

    def _mean_index_from_event(self,plane,event):
        state=self._mean_plot_state.get(plane)
        if state is None or event.inaxes is not state["axis"]:return None
        contains,details=state["points"].contains(event)
        if not contains or not details.get("ind"):return None
        return int(state["indices"][int(details["ind"][0])])

    def _on_mean_plot_hover(self,plane,event):
        state=self._mean_plot_state.get(plane)
        if state is None:return
        index=self._mean_index_from_event(plane,event); annotation=state["annotation"]
        if index is None:annotation.set_visible(False)
        else:
            value=state["value_by_index"][index]; self._place_plot_annotation(annotation,state["axis"],index,value); annotation.set_text(self._mean_details_text(index).replace(" · ","\n",1).replace(" · mean Y","\nmean Y")); annotation.set_visible(True)
        state["canvas"].canvas.draw_idle()

    def _on_mean_plot_click(self,plane,event):
        index=self._mean_index_from_event(plane,event)
        if index is None:return
        state=self._mean_plot_state[plane]; state["highlight"].set_offsets([[index,state["value_by_index"][index]]]); state["canvas"].canvas.draw_idle(); self.bpm_review_details.setText(self._mean_details_text(index)); self.bpm_review_group.setChecked(True)

    def _dispersion_index_from_event(self,key,event):
        state=self._dispersion_plot_state.get(key)
        if state is None or event.inaxes is not state["axis"]:return None
        series=state["series"] if "series" in state else ((state["label"],state["points"],state["value_by_index"]),)
        for label,points,values in series:
            contains,details=points.contains(event)
            if contains and details.get("ind"):
                index=int(state["indices"][int(details["ind"][0])]); state["active_label"]=label; state["active_values"]=values; return index
        return None

    def _dispersion_details_text(self,key,index):
        result=getattr(self,"result",None); state=self._dispersion_plot_state.get(key)
        if not isinstance(result,DispersionResult) or state is None:return ""
        label=state.get("active_label",state["label"]); values=state.get("active_values",state.get("value_by_index",{}))
        return f"{result.devices[index].name} · BPM index {index} (zero-based) · {label} {values[index]:.6g} {state['unit']}"

    def _on_dispersion_plot_hover(self,key,event):
        state=self._dispersion_plot_state.get(key)
        if state is None:return
        index=self._dispersion_index_from_event(key,event); annotation=state["annotation"]
        if index is None:annotation.set_visible(False)
        else:
            values=state.get("active_values",state.get("value_by_index",{})); value=values[index]; self._place_plot_annotation(annotation,state["axis"],index,value); annotation.set_text(self._dispersion_details_text(key,index).replace(" · ","\n")); annotation.set_visible(True)
        state["canvas"].canvas.draw_idle()

    def _on_dispersion_plot_click(self,key,event):
        index=self._dispersion_index_from_event(key,event)
        if index is None:return
        self._dispersion_selected_bpm_index=index; details=self._dispersion_details_text(key,index); self.live_dispersion_selection.setText(details); self.dispersion_selected_details.setText(details); self.live_dispersion_pin.setEnabled(True); self.dispersion_pin_button.setEnabled(True); label="Remove highlight" if index in self._dispersion_pinned_bpm_indices else "Keep highlighted"; self.live_dispersion_pin.setText(label); self.dispersion_pin_button.setText(label); self._refresh_dispersion_highlights(); self.statusBar().showMessage(details)

    def _select_bpm(self,index,*,source):
        result=getattr(self,"result",None)
        if not isinstance(result,BpmNoiseResult) or not 0 <= index < len(result.devices):return
        self._selected_bpm_index=index; self.bpm_review_details.setText(self._bpm_details_text(index)); self.bpm_review_group.setChecked(True)
        for state in self._bpm_plot_state.values():
            if index in state["value_by_index"]:state["highlight"].set_offsets([[index,state["value_by_index"][index]]])
            else:state["highlight"].set_offsets(np.empty((0,2)))
            state["canvas"].canvas.draw_idle()
        self._loading_bpm_review_panel=True; self.bpm_review_status.setEnabled(True); self.bpm_review_highlight.setEnabled(True); self.bpm_review_note.setEnabled(True); self.bpm_review_status.setCurrentText(self._bpm_review_statuses.get(index,"Unreviewed")); self.bpm_review_highlight.setChecked(index in self._pinned_bpm_indices); self.bpm_review_note.setText(self._bpm_review_notes.get(index,"")); self._loading_bpm_review_panel=False
        if source=="plot":
            for row in range(self.bpm_review_table.rowCount()):
                if self._bpm_review_index_for_row(row)==index and not self.bpm_review_table.isRowHidden(row):
                    self.bpm_review_table.selectRow(row); self.bpm_review_table.scrollToItem(self.bpm_review_table.item(row,0)); break

    def _show_result(self,result):
        if isinstance(result,ORMResult): return self._show_orm_result(result)
        if isinstance(result,DispersionResult): return self._show_dispersion_result(result)
        display={}; self._bpm_plot_state.clear(); self._mean_plot_state.clear()
        for plane,canvas,values,title,color in (("x",self.x_plot,result.noise_x_m,"Horizontal BPM noise σx","#11B7C1"),("y",self.y_plot,result.noise_y_m,"Vertical BPM noise σy","#F59E42")):
            configured=self._configured_bpm_noise(plane)
            scale,unit=self._noise_display_unit(values,configured); display[plane]=(scale,unit)
            canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(values)); shown=np.asarray(values)*scale
            ax.plot(positions,shown,color=color,linewidth=.65,alpha=.45,zorder=1)
            points=ax.scatter(positions,shown,color=color,s=18,alpha=.9,edgecolors="none",zorder=2,label=f"Measured σ{plane}",picker=5)
            highlight=ax.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=4)
            pinned=ax.scatter([],[],s=58,facecolors=color,edgecolors="#8B5CF6",linewidths=1.8,zorder=3)
            if configured is not None:
                configured_shown=configured*scale; ax.axhline(configured_shown,color="#D946EF",linestyle="--",linewidth=1.4,label=f"σconfigured = {configured_shown:g} {unit}",zorder=0); ax.legend(loc="best")
            ax.set_title(f"{title} [{unit}]"); ax.set_ylabel(f"Noise [{unit}]"); ax.set_xlabel("BPM index (zero-based)"); ax.grid(True,alpha=.25); canvas.apply_theme()
            theme=canvas.theme(); annotation=ax.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=5); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False)
            original_indices=np.arange(len(values),dtype=int)
            self._bpm_plot_state[plane]={"axis":ax,"canvas":canvas,"points":points,"highlight":highlight,"pinned":pinned,"pin_lines":[],"annotation":annotation,"indices":original_indices,"values":shown,"value_by_index":dict(zip(original_indices.tolist(),shown.tolist()))}
        for plane,canvas,values,title,color in (("x",self.mean_x_plot,result.mean_x_m,"Mean horizontal orbit","#11B7C1"),("y",self.mean_y_plot,result.mean_y_m,"Mean vertical orbit","#F59E42")):
            canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(values)); shown=np.asarray(values)*1e6; ax.plot(positions,shown,color=color,linewidth=.8,alpha=.6); points=ax.scatter(positions,shown,color=color,s=18,picker=5,zorder=2); highlight=ax.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=4); ax.set_title(title); ax.set_ylabel("Mean orbit [µm]"); ax.set_xlabel("BPM index (zero-based)"); ax.grid(True,alpha=.25); canvas.apply_theme(); theme=canvas.theme(); annotation=ax.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=5); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False); indices=np.arange(len(values),dtype=int); self._mean_plot_state[plane]={"axis":ax,"canvas":canvas,"points":points,"highlight":highlight,"annotation":annotation,"indices":indices,"value_by_index":dict(zip(indices.tolist(),shown.tolist()))}
        def text(label,values,plane):
            scale,unit=display[plane]; return label+": "+", ".join(f"{k} {v*scale:.3f} {unit}" for k,v in self._stats(values).items())
        self.summary_x.setText(text("Horizontal",result.noise_x_m,"x")); self.summary_y.setText(text("Vertical",result.noise_y_m,"y"))
        configured_values=[value for value in (self._configured_bpm_noise("x"),self._configured_bpm_noise("y")) if value is not None]
        common_scale,common_unit=self._noise_display_unit(
            np.concatenate((np.asarray(result.noise_x_m),np.asarray(result.noise_y_m))),
            max(configured_values) if configured_values else None,
        )
        convention="population" if result.ddof==0 else "sample"; self.orm_measurement_summary.setText(f"{len(result.devices)} BPMs · {result.orbits_x_m.shape[0]} readings · σ is the per-BPM {convention} standard deviation across readings (ddof={result.ddof}). Mean and RMS summarize the BPM σ values.")
        self.statistics_table.set_statistics((
            ("Horizontal (X)",{key:value*common_scale for key,value in self._stats(result.noise_x_m).items()}),
            ("Vertical (Y)",{key:value*common_scale for key,value in self._stats(result.noise_y_m).items()}),
        ),common_unit)
        self.rf_diagnostics.clear()
        self.restoration_label.clear()
        self._populate_bpm_review_table(result)
        path=getattr(self,"saved_measurement_path",None)
        if path is not None and Path(path).exists():
            try:self._load_bpm_review(path)
            except Exception as exc:self.bpm_review_save_state.setText(f"Review unavailable: {exc}"); self.bpm_review_save_state.setObjectName("validationMissing")
        self.bpm_review_group.setVisible(True)

    @staticmethod
    def _fit_colormap():
        app=QApplication.instance(); theme=app.property("pyLOCOThemePlot") if app is not None else None
        return theme.get("colormap","viridis") if isinstance(theme,dict) else "viridis"

    def _orm_colormap(self):
        return "RdBu_r" if self.orm_palette.currentData()=="red_blue" else "viridis"

    def _heatmap(self,canvas,data,title,unit,*,row_devices=None,column_devices=None):
        values=np.asarray(data); finite=np.abs(values[np.isfinite(values)]); limit=float(np.max(finite)) if finite.size else 1.0; limit=limit or 1.0; canvas.clear(); self._orm_plot_state.pop(canvas,None); surface=hasattr(self,"orm_view_mode") and self.orm_view_mode.currentData()=="3d"; ax=canvas.figure.add_subplot(111,projection="3d" if surface else None); cmap=self._orm_colormap()
        signed_limits={"vmin":-limit,"vmax":limit} if self.orm_palette.currentData()=="red_blue" else {}
        if surface:
            columns,rows=np.meshgrid(np.arange(values.shape[1]),np.arange(values.shape[0])); artist=ax.plot_surface(columns,rows,values,cmap=cmap,linewidth=0,antialiased=False,rcount=values.shape[0],ccount=values.shape[1],**signed_limits); ax.set_zlabel(f"Response [{unit}]")
        else:
            artist=ax.imshow(values,aspect="auto",origin="upper",cmap=cmap,interpolation="nearest",**signed_limits)
        ax.set_title(title); ax.set_xlabel("Corrector index"); ax.set_ylabel("BPM index")
        colorbar=canvas.figure.colorbar(artist,ax=ax,label=f"Response [{unit}]",shrink=.9,pad=.018,fraction=.045)
        # Keep zero explicitly identifiable for signed ORM data even when the
        # FIT-compatible viridis view uses the data range instead of a
        # zero-centred normalization.
        vmin,vmax=artist.norm.vmin,artist.norm.vmax
        if vmin is not None and vmax is not None and vmin<=0<=vmax:
            ticks=list(colorbar.get_ticks())
            if not any(np.isclose(tick,0.0) for tick in ticks):
                colorbar.set_ticks(sorted((*ticks,0.0)))
        canvas.apply_theme()
        if not surface and row_devices is not None and column_devices is not None:
            selected=ax.scatter([],[],s=115,facecolors="none",edgecolors="#D946EF",linewidths=2.2,zorder=5)
            self._orm_plot_state[canvas]={"axis":ax,"canvas":canvas,"values":values,"rows":tuple(row_devices),"columns":tuple(column_devices),"unit":unit,"selected":selected,"guides":[]}
        return ax

    def _orm_cell(self,canvas,event):
        state=self._orm_plot_state.get(canvas)
        if state is None or event.inaxes is not state["axis"] or event.xdata is None or event.ydata is None:return None
        column=int(round(event.xdata)); row=int(round(event.ydata))
        if not (0<=row<state["values"].shape[0] and 0<=column<state["values"].shape[1]):return None
        if not np.isfinite(state["values"][row,column]):return "incomplete"
        return state,row,column

    @staticmethod
    def _orm_cell_text(state,row,column):
        bpm_plane,bpm_index,bpm_name=state["rows"][row]; cor_plane,cor_index,cor_name=state["columns"][column]
        block=f"{cor_plane}→{bpm_plane}"
        return (f"BPM: {bpm_name} · index {bpm_index} (zero-based) · plane {bpm_plane}<br>"
                f"Corrector: {cor_name} · index {cor_index} (zero-based) · plane {cor_plane}<br>"
                f"Block: {block} · Response: <b>{state['values'][row,column]:+.7g} {state['unit']}</b>")

    def _on_orm_plot_hover(self,canvas,event):
        hit=self._orm_cell(canvas,event); state=self._orm_plot_state.get(canvas)
        if state is None:
            if event.inaxes is not None:self.orm_element_info.setText("Identity mapping is unavailable for this plot.")
            return
        if hit=="incomplete":self.orm_element_info.setText("This ORM cell has not been acquired yet; no response value or device identity is inferred."); return
        if hit is None:self.orm_element_info.setText(self._orm_selected_info or "Hover over an ORM cell or selected-column BPM point to identify it."); return
        state,row,column=hit; self.orm_element_info.setText(self._orm_cell_text(state,row,column))

    def _on_orm_plot_click(self,canvas,event):
        if getattr(canvas.toolbar,"mode",""):return
        hit=self._orm_cell(canvas,event)
        if hit is None or hit=="incomplete":return
        state,row,column=hit
        for guide in state["guides"]:guide.remove()
        state["guides"]=[state["axis"].axvline(column,color="#D946EF",linestyle="--",linewidth=.9,alpha=.65),state["axis"].axhline(row,color="#D946EF",linestyle="--",linewidth=.9,alpha=.65)]
        state["selected"].set_offsets([[column,row]]); details=self._orm_cell_text(state,row,column); self._orm_selected_info=details; self.orm_element_info.setText(details); canvas.canvas.draw_idle(); self.statusBar().showMessage(re.sub("<[^>]+>"," ",details))

    def _redraw_orm_result(self,*_):
        result=getattr(self,"_displayed_orm_result",None)
        if result is not None:self._show_orm_result(result)

    def _show_orm_progress(self, matrix, column):
        # During ORM acquisition there is no final self.result yet, but the
        # evolving matrix is scientifically useful and should remain visible.
        self.results_tabs.setVisible(True)

        # Keep only the compact evolving ORM view during acquisition.
        self.orm_column_row.setVisible(False)
        self.orm_view_row.setVisible(False)
        self.stats_group.setVisible(False)

        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        bpms=tuple(("X",i,d.name) for i,d in enumerate(self.selected_devices))+tuple(("Y",i,d.name) for i,d in enumerate(self.selected_devices)); nh=len(self.selected_hcorrectors); correctors=tuple(("H",i,d.name) for i,d in enumerate(self.selected_hcorrectors))+tuple(("V",nh+i,d.name) for i,d in enumerate(self.selected_vcorrectors))
        self._heatmap(
            self.x_plot,
            finite,
            "Evolving ORM heatmap",
            "m",
            row_devices=bpms,
            column_devices=correctors,
        )
        self.orm_element_info.setVisible(True)

        self.orm_column_plot.clear()
        self._orm_column_state=None
        ax = self.orm_column_plot.figure.add_subplot(111)
        nb = len(self.selected_devices)
        ax.plot(
            column[:nb] * 1e6,
            label="Horizontal BPMs",
        )
        ax.plot(
            column[nb:] * 1e6,
            label="Vertical BPMs",
        )
        ax.set_title("Current ORM column")
        ax.set_ylabel("Orbit difference [µm]")
        ax.legend()
        ax.grid(True, alpha=.25)
        self.orm_column_plot.apply_theme()

        # Keep the evolving full ORM as the active acquisition tab.
        index = self.results_tabs.indexOf(self.x_plot)
        if index >= 0:
            self.results_tabs.setCurrentIndex(index)

    def _show_orm_result(self,result):
        if getattr(self,"_displayed_orm_result",None) is not result:self._orm_selected_info=""
        stored_matrix=result.response_matrix; nb=len(result.bpms); nh=len(result.horizontal_correctors)
        analysis=bool(self.orm_normalized_analysis.isChecked()) and not result.scaled
        matrix=stored_matrix/result.effective_kicks_rad[np.newaxis,:] if analysis else stored_matrix; unit="m/rad" if result.scaled or analysis else "m"
        self._displayed_orm_matrix=matrix; self._displayed_orm_unit=unit
        bpm_rows=tuple(("X",i,d.name) for i,d in enumerate(result.bpms))+tuple(("Y",i,d.name) for i,d in enumerate(result.bpms)); corrector_columns=tuple(("H",i,d.name) for i,d in enumerate(result.horizontal_correctors))+tuple(("V",nh+i,d.name) for i,d in enumerate(result.vertical_correctors)); x_rows=bpm_rows[:nb]; y_rows=bpm_rows[nb:]; h_columns=corrector_columns[:nh]; v_columns=corrector_columns[nh:]
        full_ax=self._heatmap(self.x_plot,matrix,"Full orbit response matrix",unit,row_devices=bpm_rows,column_devices=corrector_columns); self._heatmap(self.y_plot,matrix[:nb,:nh],"H corrector → X BPM",unit,row_devices=x_rows,column_devices=h_columns); self._heatmap(self.mean_x_plot,matrix[:nb,nh:],"V corrector → X BPM (cross-plane)",unit,row_devices=x_rows,column_devices=v_columns); self._heatmap(self.mean_y_plot,matrix[nb:,:nh],"H corrector → Y BPM (cross-plane)",unit,row_devices=y_rows,column_devices=h_columns); self._heatmap(self.orm_vv_plot,matrix[nb:,nh:],"V corrector → Y BPM",unit,row_devices=y_rows,column_devices=v_columns)
        if self.orm_view_mode.currentData()=="2d":
            if nh and len(result.vertical_correctors):full_ax.axvline(nh-.5,color="white",linewidth=2,linestyle="--")
            full_ax.axhline(nb-.5,color="white",linewidth=2,linestyle="--")
            for x,y,label in (((nh-1)/2,(nb-1)/2,"H → X"),(nh+(len(result.vertical_correctors)-1)/2,(nb-1)/2,"V → X"),((nh-1)/2,nb+(nb-1)/2,"H → Y"),(nh+(len(result.vertical_correctors)-1)/2,nb+(nb-1)/2,"V → Y")):
                if x>=0:full_ax.text(x,y,label,ha="center",va="center",color="black",bbox={"facecolor":"white","edgecolor":"none","alpha":.7,"pad":2},fontsize=8,zorder=4)
        full_ax.set_xlabel("Corrector index"); full_ax.set_ylabel("BPM index"); self.x_plot.apply_theme()
        self._displayed_orm_result=result; self.orm_column_selector.blockSignals(True); self.orm_column_selector.clear()
        for index,device in enumerate(result.correctors):self.orm_column_selector.addItem(f"{index}: {device.name} ({device.plane})",index)
        self.orm_column_selector.setCurrentIndex(len(result.correctors)-1); self.orm_column_selector.blockSignals(False); self._update_selected_orm_column()
        self.orm_kick_plot.clear(); ax=self.orm_kick_plot.figure.add_subplot(111); pos=np.arange(len(result.correctors)); ax.plot(pos,result.requested_kicks_rad*1e6,"o-",label="Requested |ΔK|"); ax.plot(pos,np.abs(result.effective_kicks_rad)*1e6,"s--",label="Actual |ΔK|"); ax.set_title("Requested versus actual corrector kicks"); ax.set_ylabel("Kick [µrad]"); ax.legend(); ax.grid(True,alpha=.25); self.orm_kick_plot.apply_theme()
        values=np.asarray(matrix); display_scale,display_unit=(1e3,"mm") if unit=="m" else (1.0,unit); shown=values*display_scale
        self.summary_x.setText(f"<b>Measured ORM</b> &nbsp; Mean {np.mean(shown):.5g} &nbsp; RMS {np.sqrt(np.mean(shown**2)):.5g} &nbsp; Min {np.min(shown):.5g} &nbsp; Max {np.max(shown):.5g} {display_unit}")
        errors=(result.effective_kicks_rad-np.where(result.direction=="negative",-result.requested_kicks_rad,result.requested_kicks_rad))*1e6; self.summary_y.setText(f"<b>Kick error</b> &nbsp; RMS {np.sqrt(np.mean(errors**2)):.5g} &nbsp; Min {np.min(errors):.5g} &nbsp; Max {np.max(errors):.5g} µrad")
        self.statistics_table.set_statistics((("Measured ORM",self._stats(shown)),),display_unit)
        intermediate_ok=result.direction!="bipolar" or (all(status=="restored" for status in result.intermediate_restoration_status) and np.allclose(result.intermediate_restored_setpoints_rad,result.original_setpoints_rad,rtol=0,atol=1e-12)); restored=intermediate_ok and all(status=="restored" for status in result.restoration_status) and np.allclose(result.final_setpoints_rad,result.original_setpoints_rad,rtol=0,atol=1e-12); self.orm_restoration_status.setText("✓ ALL CORRECTORS RESTORED" if restored else "✗ CORRECTOR RESTORATION FAILED"); kicks=result.requested_kicks_rad*1e6; kick_text=f"{kicks[0]:g} µrad" if np.allclose(kicks,kicks[0]) else f"{np.min(kicks):g}–{np.max(kicks):g} µrad"; identity,_=self._machine_identity(); readings=result.raw_state_a_m.shape[1] if result.raw_state_a_m.ndim==3 else 1
        actual=np.abs(result.effective_kicks_rad)*1e6; actual_text=f"{actual[0]:g} µrad" if np.allclose(actual,actual[0]) else f"{np.min(actual):g}–{np.max(actual):g} µrad"
        summary_html=(f"<h3 style='color:#12BFC4;margin:0 0 8px 0'>Measurement summary</h3>"
                      f"<table cellspacing='0' cellpadding='4' width='100%'>"
                      f"<tr><td><b>Machine/profile</b></td><td colspan='3'>{escape(identity.replace(chr(10),' — '))}</td></tr>"
                      f"<tr><td>BPMs used: {nb}</td><td>H correctors used: {nh}</td><td>V correctors used: {len(result.vertical_correctors)}</td></tr>"
                      f"<tr><td>Requested separation: {kick_text}</td><td>Actual separation: {actual_text}</td><td>Readings/state: {readings}</td><td>Matrix: {matrix.shape[0]} × {matrix.shape[1]} [{unit}]</td></tr>"
                      f"<tr><td>Reading delay: {self.delay.value():g} s</td><td>Settling delay: {self.settling_delay.value():g} s</td><td colspan='2'>Storage: {'normalized response divided by actual separation' if result.scaled else 'orbit difference; not kick-normalized'}</td></tr></table>")
        self.orm_measurement_summary.setText(summary_html); self.restoration_label.setText("<b>Restore checks</b><br>"+" &nbsp; • &nbsp; ".join(f"{escape(device.name)}: {middle} / {final}" for device,middle,final in zip(result.correctors,result.intermediate_restoration_status,result.restoration_status)))
        if self.compare_reference.isChecked() and self.reference_model is not None:
            try:
                model_matrix=getattr(self,"_model_orm_cache",None)
                if model_matrix is None:model_matrix=model_orm(self.reference_model,result.bpms,result.horizontal_correctors,result.vertical_correctors,result.effective_kicks_rad[:nh],result.effective_kicks_rad[nh:],scaled=result.scaled)
                shown_model=model_matrix/result.effective_kicks_rad[np.newaxis,:] if analysis else model_matrix; difference=matrix-shown_model; self._heatmap(self.orm_model_plot,shown_model,"MODEL ORM — reference lattice",unit,row_devices=bpm_rows,column_devices=corrector_columns); self._heatmap(self.orm_difference_plot,difference,"Measured − MODEL ORM",unit,row_devices=bpm_rows,column_devices=corrector_columns)
                def comparison_row(label, measured_values, model_values):
                    metrics=comparison_metrics(measured_values,model_values)
                    def value(key):return f"{metrics[key]*display_scale:.5g}"
                    return (f"<tr><td><b>{label}</b></td><td>{value('rms_measured')}</td><td>{value('rms_model')}</td>"
                            f"<td>{value('rms_difference')}</td><td>{value('max_abs_difference')}</td></tr>")
                comparisons=(
                    comparison_row("Full ORM",matrix,shown_model),
                    comparison_row("Direct H→X",matrix[:nb,:nh],shown_model[:nb,:nh]),
                    comparison_row("Direct V→Y",matrix[nb:,nh:],shown_model[nb:,nh:]),
                    comparison_row("Coupling V→X",matrix[:nb,nh:],shown_model[:nb,nh:]),
                    comparison_row("Coupling H→Y",matrix[nb:,:nh],shown_model[nb:,:nh]),
                )
                comparison_html=(f"<br><h3 style='color:#12BFC4;margin:8px 0'>Measured vs reference model</h3>"
                                 f"<div style='margin-bottom:6px'>RMS and maximum values are shown in <b>{display_unit}</b>.</div>"
                                 f"<table cellspacing='0' cellpadding='4' width='100%'><tr><th align='left'>ORM block</th><th>Measured RMS</th><th>Model RMS</th><th>Difference RMS</th><th>Max |difference|</th></tr>{''.join(comparisons)}</table>")
                self.orm_measurement_summary.setText(summary_html+comparison_html)
            except Exception as exc:
                message=f"Reference-model comparison unavailable: {exc}"
                self.orm_measurement_summary.setText(self.orm_measurement_summary.text()+f"\n{message}")
                for canvas,title in ((self.orm_model_plot,"MODEL ORM"),(self.orm_difference_plot,"Measured − MODEL ORM")):
                    canvas.clear(); axis=canvas.figure.add_subplot(111); axis.set_title(title); axis.text(.5,.5,message,ha="center",va="center",wrap=True,transform=axis.transAxes); axis.set_axis_off(); canvas.apply_theme()

    def _orm_transaction_text(self,result,index):
        device=result.correctors[index]; scale=1e6; error=(result.final_setpoints_rad[index]-result.original_setpoints_rad[index])*scale; intermediate=result.intermediate_restored_setpoints_rad[index]; intermediate_error=(intermediate-result.original_setpoints_rad[index])*scale
        reference=np.mean(result.raw_reference_before_m[index],axis=0); final_return=(np.mean(result.raw_reference_final_m[index],axis=0)-reference)*scale; nb=len(result.bpms); return_x=final_return[:nb]; return_y=final_return[nb:]
        if result.direction=="bipolar":
            middle_return=(np.mean(result.raw_reference_intermediate_m[index],axis=0)-reference)*scale; middle_x=middle_return[:nb]; middle_y=middle_return[nb:]
            middle_text=f"Intermediate orbit return: X RMS {np.sqrt(np.mean(middle_x**2)):.6g} µm / max {np.max(np.abs(middle_x)):.6g} µm; Y RMS {np.sqrt(np.mean(middle_y**2)):.6g} µm / max {np.max(np.abs(middle_y)):.6g} µm\n"
        else:middle_text="Intermediate orbit return: not applicable for a one-sided measurement\n"
        return (f"Corrector transaction — {device.name} ({device.plane})\n"
                f"Backend identifier: {device.identifier}\n"
                f"Original K0: {result.original_setpoints_rad[index]*scale:+.9g} µrad; requested K+/K−: {result.requested_state_a_rad[index]*scale:+.9g} / {result.requested_state_b_rad[index]*scale:+.9g} µrad\n"
                f"Readback K+: {result.actual_state_a_rad[index]*scale:+.9g} µrad\n"
                f"Intermediate restore K+ → K0: {intermediate*scale:+.9g} µrad; error: {intermediate_error:+.9g} µrad — {result.intermediate_restoration_status[index]}\n"
                f"{middle_text}"
                f"Readback K−: {result.actual_state_b_rad[index]*scale:+.9g} µrad; actual K+−K− separation: {result.effective_kicks_rad[index]*scale:+.9g} µrad\n"
                f"Final restore K− → K0: {result.final_setpoints_rad[index]*scale:+.9g} µrad; error: {error:+.9g} µrad — {result.restoration_status[index]}\n"
                f"Final orbit return: X RMS {np.sqrt(np.mean(return_x**2)):.6g} µm / max {np.max(np.abs(return_x)):.6g} µm; Y RMS {np.sqrt(np.mean(return_y**2)):.6g} µm / max {np.max(np.abs(return_y)):.6g} µm\n"
                f"Restored-reference orbits are diagnostic-only and excluded from the ORM calculation.\n"
                f"Matrix convention: orbit(+) − orbit(−); rows X BPM then Y BPM; columns H then V; {'normalized by actual separation' if result.scaled else 'raw orbit difference' }.")

    def _update_selected_orm_column(self,*_):
        result=getattr(self,"_displayed_orm_result",None); index=self.orm_column_selector.currentData()
        if result is None or index is None:return
        index=int(index); nb=len(result.bpms); display_matrix=getattr(self,"_displayed_orm_matrix",result.response_matrix); display_unit=getattr(self,"_displayed_orm_unit","m/rad" if result.scaled else "m"); column=display_matrix[:,index]; scale,unit=(1.0,"m/rad") if display_unit=="m/rad" else (1e6,"µm"); self.orm_column_plot.clear(); ax=self.orm_column_plot.figure.add_subplot(111); x_values=column[:nb]*scale; y_values=column[nb:]*scale; x_line=ax.plot(x_values,"o-",markersize=3,label="X BPM response",picker=5)[0]; y_line=ax.plot(y_values,"s-",markersize=3,label="Y BPM response",picker=5)[0]; selected=ax.scatter([],[],s=105,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=5); device=result.correctors[index]; ax.set_title(f"Corrector {index} ({'H' if index<len(result.horizontal_correctors) else 'V'})"); ax.set_xlabel("BPM index"); ax.set_ylabel(f"Response [{unit}]"); ax.legend(); ax.grid(True,alpha=.25); self.orm_column_plot.apply_theme()
        corrector_plane="H" if index<len(result.horizontal_correctors) else "V"; self.orm_column_corrector_info.setText(f"Corrector {index} (zero-based) · {corrector_plane} · {device.name}")
        self._orm_column_state={"axis":ax,"corrector":device,"corrector_index":index,"corrector_plane":corrector_plane,"bpms":result.bpms,"series":(("X",x_line,np.asarray(x_values)),("Y",y_line,np.asarray(y_values))),"unit":unit,"selected":selected}
        self.rf_diagnostics.setText(self._orm_transaction_text(result,index))

    def _orm_column_hit(self,event):
        state=self._orm_column_state
        if state is None or event.inaxes is not state["axis"] or event.xdata is None or event.ydata is None:return None
        bpm_index=int(round(event.xdata))
        if not 0<=bpm_index<len(state["bpms"]):return None
        plane,line,values=min(state["series"],key=lambda item:abs(float(event.ydata)-item[2][bpm_index]))
        return state,bpm_index,plane,float(values[bpm_index])

    @staticmethod
    def _orm_column_text(state,bpm_index,plane,value):
        bpm=state["bpms"][bpm_index]; block=f"{state['corrector_plane']}→{plane}"
        return (f"BPM: {bpm.name} · index {bpm_index} (zero-based) · plane {plane}<br>"
                f"Corrector: {state['corrector'].name} · index {state['corrector_index']} (zero-based) · plane {state['corrector_plane']}<br>"
                f"Block: {block} · Response: <b>{value:+.7g} {state['unit']}</b>")

    def _on_orm_column_hover(self,event):
        hit=self._orm_column_hit(event)
        if hit is None:self.orm_element_info.setText(self._orm_selected_info or "Hover over an ORM cell or selected-column BPM point to identify it."); return
        self.orm_element_info.setText(self._orm_column_text(*hit))

    def _on_orm_column_click(self,event):
        if getattr(self.orm_column_plot.toolbar,"mode",""):return
        hit=self._orm_column_hit(event)
        if hit is None:return
        state,bpm_index,plane,value=hit; state["selected"].set_offsets([[bpm_index,value]]); details=self._orm_column_text(state,bpm_index,plane,value); self._orm_selected_info=details; self.orm_element_info.setText(details); self.orm_column_plot.canvas.draw_idle()

    def _show_dispersion_result(self,result):
        self._dispersion_plot_state.clear()
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
            canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(values)); shown=np.asarray(values)*scale
            if self.dispersion_connect_points.isChecked():ax.plot(positions,shown,color=color,linewidth=.75,alpha=.55,zorder=1)
            points=ax.scatter(positions,shown,color=color,s=20,picker=5,zorder=2); highlight=ax.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=4); pinned=ax.scatter([],[],s=55,facecolors=color,edgecolors="#8B5CF6",linewidths=1.8,zorder=3); ax.set_title(f"{title} [{unit}]"); label="Physical dispersion" if title.startswith("Physical") else "RF orbit difference"; ax.set_ylabel(f"{label} [{unit}]"); ax.set_xlabel("BPM index (zero-based)"); ax.grid(True,alpha=.25); canvas.apply_theme(); theme=canvas.theme(); annotation=ax.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=5); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False); indices=np.arange(len(values),dtype=int); key="raw_x" if canvas is self.raw_x_plot else "raw_y" if canvas is self.raw_y_plot else "physical_x" if canvas is self.x_plot else "physical_y"; self._dispersion_plot_state[key]={"axis":ax,"canvas":canvas,"points":points,"highlight":highlight,"pinned":pinned,"annotation":annotation,"indices":indices,"value_by_index":dict(zip(indices.tolist(),shown.tolist())),"label":label,"unit":unit,"pin_lines":[]}
        self._refresh_dispersion_highlights()
        model_comparison=""
        model_comparison_html="<b>Not calculated.</b><br>Enable <i>Compare measurements with reference model</i> before or after the measurement."
        if physical_x is not None and self.compare_reference.isChecked() and self.reference_model is not None:
            try:
                model_x,model_y=model_dispersion(self.reference_model,result.devices)
                for canvas,measured,model_values,plane in ((self.x_plot,physical_x,model_x,"X"),(self.y_plot,physical_y,model_y,"Y")):
                    ax=canvas.figure.axes[0]; ax.plot(model_values*1e3,"--",linewidth=1.5,label=f"MODEL D{plane.lower()} — reference lattice"); ax.plot((measured-model_values)*1e3,":",linewidth=1.2,label="Measured − MODEL"); ax.legend(); canvas.apply_theme()
                mx=comparison_metrics(physical_x,model_x); my=comparison_metrics(physical_y,model_y)
                model_comparison=(f"\nReference-model comparison\n"
                                  f"X: RMS measured/model/difference {mx['rms_measured']*1e3:.4g} / {mx['rms_model']*1e3:.4g} / {mx['rms_difference']*1e3:.4g} mm; max |difference| {mx['max_abs_difference']*1e3:.4g} mm\n"
                                  f"Y: RMS measured/model/difference {my['rms_measured']*1e3:.4g} / {my['rms_model']*1e3:.4g} / {my['rms_difference']*1e3:.4g} mm; max |difference| {my['max_abs_difference']*1e3:.4g} mm\n"
                                  f"Source: {self.reference_model.source}")
                rows=[]
                for plane,metrics in (("Dₓ",mx),("Dᵧ",my)):
                    rows.append(
                        f"<tr><td><b>{plane}</b></td><td>{metrics['rms_measured']*1e3:.4g}</td>"
                        f"<td>{metrics['rms_model']*1e3:.4g}</td><td>{metrics['rms_difference']*1e3:.4g}</td>"
                        f"<td>{metrics['max_abs_difference']*1e3:.4g}</td></tr>"
                    )
                model_comparison_html=(
                    "<table cellspacing='8'><tr><th>Plane</th><th>Measured RMS [mm]</th><th>Model RMS [mm]</th>"
                    "<th>Difference RMS [mm]</th><th>Max |difference| [mm]</th></tr>"+"".join(rows)+"</table>"
                    f"<small>Reference lattice: {self.reference_model.source}</small>"
                )
            except Exception as exc:
                model_comparison=f"\nReference-model comparison unavailable: {exc}"
                model_comparison_html=f"<b>Unavailable.</b><br>{exc}"
        elif self.compare_reference.isChecked():
            model_comparison_html="<b>Unavailable.</b><br>Physical dispersion and a loaded AT reference lattice are required."
        self.dispersion_model_comparison.setText(model_comparison_html)
        for canvas,plane,title in ((self.mean_x_plot,"x","Raw and mean horizontal orbit by RF state"),(self.mean_y_plot,"y","Raw and mean vertical orbit by RF state")):
            canvas.clear(); ax=canvas.figure.add_subplot(111)
            colors=("#64748B","#F59E42","#11B7C1","#D946EF")
            offset=result.requested_offset_hz
            display_labels={"reference":"Reference RF (0 Hz)","reference_after":"Restored RF (0 Hz)","positive":f"Positive RF (+{offset:g} Hz)","negative":f"Negative RF (−{offset:g} Hz)"}
            series=[]; positions=np.arange(len(result.devices)); indices=np.arange(len(result.devices),dtype=int)
            for state,color in zip(result.states,colors):
                raw=getattr(state,f"orbits_{plane}_m")*1e6
                for sample in raw: ax.plot(sample,color=color,alpha=.10,linewidth=.7)
                shown=getattr(state,f"mean_{plane}_m")*1e6; label=display_labels.get(state.label,state.label); points=ax.plot(positions,shown,"o-",label=label,color=color,markersize=3,picker=5)[0]; series.append((label,points,dict(zip(indices.tolist(),shown.tolist()))))
            highlight=ax.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=5); pinned=ax.scatter([],[],s=55,facecolors=colors[0],edgecolors="#8B5CF6",linewidths=1.8,zorder=4); ax.set_title(title); ax.set_ylabel("Orbit [µm]"); ax.set_xlabel("BPM index (zero-based)"); ax.grid(True,alpha=.25); ax.legend(); canvas.apply_theme(); theme=canvas.theme(); annotation=ax.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=6); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False); key=f"states_{plane}"; self._dispersion_plot_state[key]={"axis":ax,"canvas":canvas,"series":series,"points":series[0][1],"highlight":highlight,"pinned":pinned,"annotation":annotation,"indices":indices,"label":series[0][0],"unit":"µm","active_values":series[0][2],"pin_lines":[]}
        symmetry=[]
        if plus is not None and minus is not None:
            for canvas,plane,title in ((self.rf_shift_x_plot,"x","Horizontal RF-induced orbit shift"),(self.rf_shift_y_plot,"y","Vertical RF-induced orbit shift")):
                shift_plus=getattr(plus,f"mean_{plane}_m")-getattr(reference,f"mean_{plane}_m")
                shift_minus=getattr(minus,f"mean_{plane}_m")-getattr(reference,f"mean_{plane}_m")
                canvas.clear(); ax=canvas.figure.add_subplot(111); positions=np.arange(len(result.devices))
                if self.dispersion_connect_points.isChecked():
                    ax.plot(positions,shift_plus*1e6,linewidth=.75,alpha=.55,color="#F59E42")
                    ax.plot(positions,shift_minus*1e6,linewidth=.75,alpha=.55,color="#11B7C1")
                plus_label=f"{plane}(f+) − {plane}(f₀)"; minus_label=f"{plane}(f−) − {plane}(f₀)"; plus_values=shift_plus*1e6; minus_values=shift_minus*1e6; plus_points=ax.scatter(positions,plus_values,s=18,color="#F59E42",label=plus_label,picker=5); minus_points=ax.scatter(positions,minus_values,s=18,color="#11B7C1",label=minus_label,picker=5)
                highlight=ax.scatter([],[],s=90,facecolors="none",edgecolors="#D946EF",linewidths=2,zorder=5); pinned=ax.scatter([],[],s=55,facecolors="#F59E42",edgecolors="#8B5CF6",linewidths=1.8,zorder=4); ax.axhline(0,color="#64748B",linewidth=.8); ax.set_title(title); ax.set_ylabel("Orbit shift [µm]"); ax.set_xlabel("BPM index (zero-based)"); ax.grid(True,alpha=.25); ax.legend(); canvas.apply_theme(); theme=canvas.theme(); annotation=ax.annotate("",xy=(0,0),xytext=(12,14),textcoords="offset points",bbox={"boxstyle":"round,pad=0.45","facecolor":theme["face"],"edgecolor":theme["spine"],"alpha":.97},color=theme["text"],fontsize=9,zorder=6); annotation.set_gid("bpm-identification-tooltip"); annotation.set_visible(False); indices=np.arange(len(result.devices),dtype=int); key=f"shift_{plane}"; plus_map=dict(zip(indices.tolist(),plus_values.tolist())); minus_map=dict(zip(indices.tolist(),minus_values.tolist())); self._dispersion_plot_state[key]={"axis":ax,"canvas":canvas,"series":((plus_label,plus_points,plus_map),(minus_label,minus_points,minus_map)),"points":plus_points,"highlight":highlight,"pinned":pinned,"annotation":annotation,"indices":indices,"label":plus_label,"unit":"µm","active_values":plus_map,"pin_lines":[]}
                mismatch=(shift_plus+shift_minus)*1e6
                correlation=float(np.corrcoef(shift_plus,-shift_minus)[0,1]) if np.std(shift_plus)>0 and np.std(shift_minus)>0 else float("nan")
                symmetry.append(f"{plane.upper()} RMS mismatch {np.sqrt(np.mean(mismatch**2)):.3f} µm, correlation {correlation:.6f}")
        else:
            for canvas in (self.rf_shift_x_plot,self.rf_shift_y_plot):
                canvas.clear(); ax=canvas.figure.add_subplot(111); ax.text(.5,.5,"Positive and negative RF states are required.",ha="center",va="center",transform=ax.transAxes); ax.set_axis_off(); canvas.apply_theme()
        self._refresh_dispersion_highlights()
        display_physical=self.dispersion_display.currentData()=="physical" and physical_x is not None
        values_x,values_y=(physical_x,physical_y) if display_physical else (result.measured_eta_x,result.measured_eta_y)
        scale,unit=(1e3,"mm")
        titles=("Physical Dx","Physical Dy") if display_physical else ("RF orbit difference Δx_RF","RF orbit difference Δy_RF")
        def text(label,values): return label+": "+", ".join(f"{k} {v*scale:.3f} {unit}" for k,v in self._stats(values).items())
        self.summary_x.setText(text(titles[0],values_x)); self.summary_y.setText(text(titles[1],values_y))
        self.statistics_table.set_statistics((
            (titles[0],{key:value*scale for key,value in self._stats(values_x).items()}),
            (titles[1],{key:value*scale for key,value in self._stats(values_y).items()}),
        ),unit)
        self.rf_response_stats.set_statistics((("Horizontal Δx_RF",{key:value*1e3 for key,value in self._stats(result.measured_eta_x).items()}),("Vertical Δy_RF",{key:value*1e3 for key,value in self._stats(result.measured_eta_y).items()})),"mm")
        if physical_x is not None:
            self.physical_stats.set_statistics((("Horizontal Dₓ",{key:value*1e3 for key,value in self._stats(physical_x).items()}),("Vertical Dᵧ",{key:value*1e3 for key,value in self._stats(physical_y).items()})),"mm"); self.physical_stats.setVisible(True); self.physical_unavailable.clear(); self.physical_unavailable.setVisible(False)
        else:
            self.physical_stats.clear_statistics(); self.physical_stats.setVisible(False); self.physical_unavailable.setText("Unavailable: a verified lattice slip factor and bipolar RF readbacks are required."); self.physical_unavailable.setVisible(True)
        readbacks={state.label:state.actual_rf_hz for state in result.states}; restored=getattr(self,"_restored_rf_readback",np.nan); difference=restored-result.nominal_rf_hz
        automatic=self.rf_control_mode.currentData()=="automatic"
        if automatic:self.restoration_label.setText(f"RF readbacks — original: {result.nominal_rf_hz:.6f} Hz; negative: {readbacks.get('negative',np.nan):.6f} Hz; positive: {readbacks.get('positive',np.nan):.6f} Hz; restored: {restored:.6f} Hz; restoration difference: {difference:+g} Hz ({result.restoration_status}).")
        else:self.restoration_label.setText(f"RF restoration: {result.restoration_status}. Readback verification: not available. Canonical step: {result.canonical_rf_step_hz:g} Hz.")
        if lattice_metadata is None:self.rf_diagnostics.setText("Physical dispersion unavailable: the backend provides no verified 4D/design-lattice slip factor. The pyLOCO-compatible RF orbit difference remains available.")
        elif result.direction!="bipolar":self.rf_diagnostics.setText("Physical dispersion requires recorded positive and negative RF states; the one-sided RF orbit response remains available.")
        else:
            if deltas is None:deltas=relative_momentum_deviation([readbacks["negative"],readbacks["positive"]],result.nominal_rf_hz,eta); delta_span=float(deltas[0]-deltas[1])
            diagnostic=f"Momentum diagnostics — αc: {lattice_metadata['alpha']:.12g}; 1/γ²: {lattice_metadata['inverse_gamma_squared']:.12g}; η=αc−1/γ²: {eta:.12g}; AT get_slip_factor(): {lattice_metadata['at_slip_factor']:.12g}; δ−: {deltas[0]:+.12g}; δ+: {deltas[1]:+.12g}; δ− − δ+: {delta_span:+.12g}. Convention: {MOMENTUM_RELATION}."
            transition=[]
            for state in result.states:
                if state.orbits_x_m.shape[0]>1:
                    dx=(state.orbits_x_m[0]-np.mean(state.orbits_x_m[1:],axis=0))*1e6; dy=(state.orbits_y_m[0]-np.mean(state.orbits_y_m[1:],axis=0))*1e6
                    transition.append(f"{state.label}: X RMS {np.sqrt(np.mean(dx**2)):.3f} µm / Y RMS {np.sqrt(np.mean(dy**2)):.3f} µm")
            if transition:diagnostic+="\nRF-state transition diagnostic (first sample minus later-sample mean; no samples excluded): "+"; ".join(transition)+"."
            if any(state.label=="reference_after" for state in result.states):
                before=result.state("reference"); after=result.state("reference_after"); dx=(after.mean_x_m-before.mean_x_m)*1e6; dy=(after.mean_y_m-before.mean_y_m)*1e6
                diagnostic+=f"\nRestored-orbit diagnostic (excluded from dispersion): ΔX reference-after − reference-before: RMS {np.sqrt(np.mean(dx**2)):.3f} µm / max {np.max(np.abs(dx)):.3f} µm; ΔY: RMS {np.sqrt(np.mean(dy**2)):.3f} µm / max {np.max(np.abs(dy)):.3f} µm."
            self.rf_diagnostics.setText(diagnostic+model_comparison)
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
        if session.is_complete_for_loco:
            roles = {entry.role for entry in session.files}
            available = ", ".join(
                label for role, label in (
                    ("orm", "ORM"),
                    ("bpm_noise", "BPM noise"),
                    ("dispersion", "Dispersion"),
                )
                if role in roles
            )
            message = (
                f"Measurement schema {info['schema_version']} is valid."
                f"{sign}\n"
                "Session is complete for pyLOCO fitting.\n"
                f"Available measurements: {available}.\n"
                "Measurement order does not matter."
            )
        else:
            message = (
                f"Measurement schema {info['schema_version']} is valid."
                f"{sign}\n"
                "Session is incomplete for fitting; missing: "
                + ", ".join(session.missing_roles)
                + ".\nMeasurement order does not matter."
            )

        QMessageBox.information(
            self,
            "Valid for pyLOCO",
            message,
        )

    def explain_open(self):
        session=load_session(self.saved_session_path)
        try:ok,detail=launch_suite_application("fit","--measurement-session",str(self.saved_session_path))
        except Exception as exc:QMessageBox.warning(self,"Cannot open pyLOCO Fit",str(exc)); return
        if not ok:QMessageBox.warning(self,"Cannot open pyLOCO Fit",detail); return
        missing=", ".join(session.missing_roles) or "none"; self.statusBar().showMessage(f"Opening session in pyLOCO Fit ({detail}); missing roles: {missing}")

    def _collect_project(self):
        self.project.metadata["pyaml_options"] = {key: editor.text().strip() for key, editor in self.pyaml_fields.items()}
        self._store_active_selection()
        h=self.corrector_selection_widgets["hcor"]; v=self.corrector_selection_widgets["vcor"]
        identity,profile=self._machine_identity(); profile_meta={} if profile is None else {"machine":profile.machine,"profile":profile.scenario,"profile_key":profile.key,"profile_manifest":str(profile.manifest_path),"lattice_file":str(profile.resolve("lattice_file")),"provenance":profile.configuration.get("provenance",{}),"random_seed":profile.configuration.get("random_seed")}
        return MeasureProject(measurement_type=self.measurement_type.currentData(),measurement_name=self.measurement_name.text(),measurement_label=self.measurement_label.text(),operator_comments=self.comments.toPlainText(),adapter=self.adapter_combo.currentText(),pysc_profile=self.pysc_profile_combo.currentData(),bpm_selection_method=self.selection_method.currentData(),bpm_manual=self.manual_input.text(),bpm_names_file=self.names_file.text(),excluded_bpm_positions=self.bpm_exclusions.text(),hcor_selection_method=h["method"].currentData(),hcor_manual=h["manual"].text(),hcor_names_file=h["file"].text(),vcor_selection_method=v["method"].currentData(),vcor_manual=v["manual"].text(),vcor_names_file=v["file"].text(),excluded_hcor_positions=h["exclusion"].text(),excluded_vcor_positions=v["exclusion"].text(),measurement_selections=self._selection_states,readings=self.readings.value(),bpm_noise_ddof=int(self.bpm_noise_convention.currentData()),delay_seconds=self.delay.value(),settling_delay_seconds=self.settling_delay.value(),verify_restored_orbit=self.verify_restored_orbit.isChecked(),rf_control_mode=self.rf_control_mode.currentData(),nominal_rf_hz=self._nominal_rf_value(),nominal_rf_source="manual",rf_step_hz=self.rf_step.value(),dispersion_direction=self.dispersion_direction.currentData(),orm_direction=self.orm_direction.currentData(),orm_kick_mode=self.orm_kick_mode.currentData(),orm_horizontal_kick_rad=self.orm_hkick.value()*1e-6,orm_vertical_kick_rad=self.orm_vkick.value()*1e-6,orm_kick_file=self.orm_kick_file.text(),orm_scaled=False,reference_model_path=self.reference_model_path.text(),reference_model_source="automatic" if self.adapter_combo.currentData()=="pysc" else "user",compare_with_reference_model=self.compare_reference.isChecked(),output_directory=self.output_directory.text(),theme=self.project.theme,metadata={**self.project.metadata,"machine_identity":identity,"profile":profile_meta,"orm_normalized_analysis":self.orm_normalized_analysis.isChecked()})

    def _load_project_widgets(self):
        self.pyaml_magnet_writes.setChecked(True)
        self.pyaml_rf_writes.setChecked(True)
        for key, editor in self.pyaml_fields.items():
            editor.setText(self.project.metadata.get("pyaml_options", {}).get(key, {"mode": "live", "bpm_group": "BPM", "hcor_group": "HCorr", "vcor_group": "VCorr", "rf_name": "RF"}.get(key, "")))
        self.pysc_profile_combo.setCurrentIndex(max(0,self.pysc_profile_combo.findData(self.project.pysc_profile)))
        self.adapter_combo.setCurrentIndex(max(0,self.adapter_combo.findText(self.project.adapter)))
        if self.adapter_combo.currentData() == "pyaml":
            self._invalidate_pyaml()
        p=self.project; self._selection_states=p.measurement_selections; self._active_selection_kind=p.measurement_type; self.measurement_type.blockSignals(True); self.measurement_type.setCurrentIndex(max(0,self.measurement_type.findData(p.measurement_type))); self.measurement_type.blockSignals(False); self._restore_selection(p.measurement_type); self.measurement_name.setText(p.measurement_name); self.measurement_label.setText(p.measurement_label); self.comments.setPlainText(p.operator_comments); self.readings.setValue(p.readings); self.bpm_noise_convention.setCurrentIndex(max(0,self.bpm_noise_convention.findData(p.bpm_noise_ddof))); self.delay.setValue(p.delay_seconds); self.settling_delay.setValue(p.settling_delay_seconds); self.verify_restored_orbit.setChecked(p.verify_restored_orbit); self.nominal_rf.setText("" if p.nominal_rf_hz is None else f"{p.nominal_rf_hz:g}"); self.rf_step.setValue(p.rf_step_hz); self.dispersion_direction.setCurrentIndex(max(0,self.dispersion_direction.findData(p.dispersion_direction))); self.orm_direction.setCurrentIndex(max(0,self.orm_direction.findData(p.orm_direction))); self.orm_kick_mode.setCurrentIndex(max(0,self.orm_kick_mode.findData(p.orm_kick_mode))); self.orm_hkick.setValue(p.orm_horizontal_kick_rad*1e6); self.orm_vkick.setValue(p.orm_vertical_kick_rad*1e6); self.orm_kick_file.setText(p.orm_kick_file); self.orm_normalized_analysis.setChecked(bool(p.orm_scaled or p.metadata.get("orm_normalized_analysis",False)))
        self.reference_model_path.setText(p.reference_model_path); self.compare_reference.setChecked(p.compare_with_reference_model)
        if p.reference_model_path:self._load_reference_model(p.reference_model_path,"Project reference model")
        self.output_directory.setText(p.output_directory); self.apply_theme(p.theme); self.refresh_preview(); self._measurement_type_changed(); self._update_machine_identity()

    def new_project(self):
        self.project = MeasureProject()
        self.project_path = None
        self._load_project_widgets()
        self._reset_measurement_display()
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
        self.project.theme=key; app=QApplication.instance(); apply_application_theme(app,theme_for_key(key),"teal")
        self.setStyleSheet(measure_stylesheet(key)); self.theme_button.setText("☀ Light" if key=="dark" else "🌙 Dark")
        for plot in (getattr(self,"live_plot",None),getattr(self,"x_plot",None),getattr(self,"y_plot",None),getattr(self,"mean_x_plot",None),getattr(self,"mean_y_plot",None),getattr(self,"rf_shift_x_plot",None),getattr(self,"rf_shift_y_plot",None),getattr(self,"orm_vv_plot",None),getattr(self,"orm_column_plot",None),getattr(self,"orm_kick_plot",None),getattr(self,"orm_model_plot",None),getattr(self,"orm_difference_plot",None)):
            if plot is not None: plot.apply_theme()
    def toggle_theme(self): self.apply_theme("light" if self.project.theme=="dark" else "dark")
