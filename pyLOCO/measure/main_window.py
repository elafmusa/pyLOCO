"""pyLOCO Measure GUI for structured, read-only measurement acquisition."""
from __future__ import annotations

from types import SimpleNamespace
from copy import deepcopy

import re
import csv
import json
import shutil
from fnmatch import fnmatchcase
from html import escape
from datetime import datetime
from pathlib import Path
from importlib.util import find_spec
from threading import Event
from time import monotonic, sleep
from typing import Sequence
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QObject, QThread, Qt, Signal, Slot, QSize, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QInputDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QMenu,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QDoubleSpinBox, QSizePolicy, QTabWidget, QTableWidget, QTableWidgetItem, QToolBar, QToolButton,
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
from pyLOCO.data_schema.session import SESSION_FILE_TYPE
from pyLOCO.gui.branding import DISPLAY_ASSET, application_icon, set_asset
from pyLOCO.gui.file_dialogs import FileDialog as QFileDialog
from pyLOCO.gui.screenshots import install_window_screenshot_actions
from pyLOCO.gui.control_search import install_control_search
from pyLOCO.gui.project_info import (
    PROJECT_ACKNOWLEDGEMENTS, PROJECT_CONTRIBUTORS, PROJECT_DOCUMENTATION,
    PROJECT_ISSUES, PROJECT_LICENSE, PROJECT_PAPER_TITLE, PROJECT_PAPER_URL,
    PROJECT_REPOSITORY, bibtex_text, citation_text,
)
from pyLOCO.gui import __version__ as PYLOCO_VERSION
from pyLOCO.data_schema import SCHEMA_VERSION as MEASUREMENT_SCHEMA_VERSION
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.measurement_mapping import (
    find_exact_device_mappings, is_unvalidated_mapping_template, load_document,
    inventory_fingerprint, mapping_counts, mapping_template, ordered_ordinals,
    python_mapping_helper_script, validate_mapping,
)
from pyLOCO.gui.mapping_table import (CopyableMappingTable, configure_mapping_table,
                                      identity_match_label)
from pyLOCO.machine_mapping import (is_document as is_machine_mapping,
    is_template as is_machine_mapping_template, measurement_projection,
    complete_template as complete_machine_mapping_template,
    editable_template as editable_machine_mapping_template,
    merge_measurement, pyaml_configuration_inventory, reconcile_configured,
    template as machine_mapping_template)
from pyLOCO.gui.lattice_loader import MultipleMatLatticesError, mat_lattice_candidates
from pyLOCO.gui.themes import ACCENTS, accent_for_key, apply_application_theme, theme_for_key
from pyLOCO.gui.suite import launch_suite_application,present_single_about_dialog
from .acquisition import (
    AcquisitionCancelled, BpmDevice, BpmNoiseAcquirer, BpmNoiseResult, CorrectorDevice,
    DispersionResult, DispersionStateAcquirer, DispersionStateResult, ORMAcquirer, ORMAcquisitionError,
    ORMInterrupted, ORMResult,
)
from .project import MeasureProject, load_measure_project, save_measure_project
from .full_session import (MANIFEST_NAME, file_sha256 as session_file_sha256, open_full_session_bundle,
                           save_full_session_bundle, save_full_session_state)
from .automatic import AutomaticDispersionAcquirer
from .dispersion import MOMENTUM_RELATION, physical_dispersion, relative_momentum_deviation
from pyLOCO.reference_model import comparison_metrics, load_reference_model, model_dispersion, model_orm, reference_model_for_pysc, store_reference_model_arrays


def _device_filter_terms(text):
    """Return nonempty comma-separated search terms in case-insensitive form."""
    return tuple(term.strip().casefold() for term in str(text or "").split(",") if term.strip())


def _device_matches_filter(name, identifier, text):
    """Match any plain substring or shell-style wildcard term without reordering."""
    terms=_device_filter_terms(text)
    if not terms:return True
    fields=(str(name or "").casefold(),str(identifier or "").casefold())
    combined=" ".join(fields)
    for term in terms:
        if any(char in term for char in "*?["):
            if any(fnmatchcase(field,term) for field in fields) or fnmatchcase(combined,term):return True
        elif term in combined:return True
    return False


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
    completed = Signal(object); cancelled = Signal(object); failed = Signal(object)

    def __init__(self, acquirer, options, cancel_event):
        super().__init__(); self.acquirer=acquirer; self.options=options; self.cancel_event=cancel_event

    @Slot()
    def run(self):
        try:
            result=self.acquirer.acquire(cancel_event=self.cancel_event,
                                         progress=lambda *args:self.progress.emit(*args),
                                         status=lambda phase,details:self.status.emit(phase,details), **self.options)
        except AcquisitionCancelled as exc:self.cancelled.emit(exc)
        except Exception as exc:self.failed.emit(exc)
        else:self.completed.emit(result)


class ConnectionTestWorker(QObject):
    completed=Signal(object); failed=Signal(object)
    def __init__(self,operation):super().__init__(); self.operation=operation
    @Slot()
    def run(self):
        try:self.completed.emit(self.operation())
        except Exception as exc:self.failed.emit(exc)


class ORMAcquisitionWorker(QObject):
    event=Signal(object); completed=Signal(object); cancelled=Signal(object); failed=Signal(object)
    def __init__(self,acquirer,kick_h,kick_v,options,cancel_event): super().__init__(); self.acquirer=acquirer; self.kick_h=kick_h; self.kick_v=kick_v; self.options=options; self.cancel_event=cancel_event
    @Slot()
    def run(self):
        try:self.completed.emit(self.acquirer.acquire(self.kick_h,self.kick_v,cancel_event=self.cancel_event,progress=self.event.emit,**self.options))
        except AcquisitionCancelled as exc:self.cancelled.emit(exc)
        except Exception as exc:self.failed.emit(exc)


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
        self._rf_restoration_unresolved = False
        self._rf_restoration_diagnostics = None
        self._orm_restoration_unresolved = False
        self._orm_restoration_diagnostics = None
        self._active_acquisition_kind = None
        self._measurement_results = {"bpm_noise": None, "dispersion": None, "orm": None}
        self._measurement_result_paths = {"bpm_noise": None, "dispersion": None, "orm": None}
        self._measurement_restore_status = {
            role: {"present": False, "validated": False, "restored": False, "error": ""}
            for role in ("bpm_noise", "dispersion", "orm")
        }
        self._measurement_view_states = {}
        self._connection_timeout_ms = 15000
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
        self.measurement_device_alignment = None
        self.machine_mapping_document = None
        self._pending_mapping_revalidation = None
        self.last_mapping_template_path: Path | None = None
        self._pending_full_bundle = None
        self._session_dirty = True

        # Every newly opened Measure window starts a new logical
        # measurement session. Existing files on disk are preserved,
        # but their roles must not leak into this session.
        self._fresh_measurement_session = True
        self.cancel_event = Event(); self.thread: QThread | None = None; self.worker = None
        self.connection_verified = True
        self._build_ui(); self._measurement_type_changed(); self._update_machine_identity()
        active_theme=QApplication.instance().property("pyLOCOTheme") if QApplication.instance() is not None else None
        self.apply_theme(str(active_theme or self.project.theme)); self.refresh_preview(); self.refresh_plan()

    def _screenshot_session_directory(self):
        """Return the active saved session/project directory, if one exists."""
        if self.saved_session_path is not None:
            return self.saved_session_path.parent
        if self.project_path is not None:
            return self.project_path.parent
        return None

    def _build_ui(self):
        self.menuBar().setNativeMenuBar(True)
        file_menu=self.menuBar().addMenu("File")
        # Native macOS menus live outside the window. The offscreen Qt test
        # platform otherwise reserves an artificial in-window menu-bar row.
        if QApplication.platformName()=="offscreen":self.menuBar().setMaximumHeight(0)
        self.session_new_action=QAction("New Session",self); self.session_new_action.triggered.connect(self.new_project)
        self.session_open_action=QAction("Open Session…",self); self.session_open_action.triggered.connect(self.open_full_measurement_session)
        self.session_save_action=QAction("Save Session",self); self.session_save_action.triggered.connect(self.save_active_session); self.session_save_action.setEnabled(True)
        self.session_save_as_action=QAction("Save Session As…",self); self.session_save_as_action.triggered.connect(self.save_session_as); self.session_save_as_action.setEnabled(True)
        self.session_export_action=QAction("Export Portable Session…",self); self.session_export_action.triggered.connect(self.export_portable_session); self.session_export_action.setEnabled(False)
        for action in (self.session_new_action,self.session_open_action,self.session_save_action,self.session_save_as_action,self.session_export_action):file_menu.addAction(action)
        self.session_open_folder_action=QAction("Open Portable Session Folder…",self); self.session_open_folder_action.triggered.connect(self.open_full_measurement_session_folder); file_menu.insertAction(self.session_save_action,self.session_open_folder_action)
        templates=file_menu.addMenu("Configuration Templates")
        open_template=QAction("Open Measure Configuration…",self); open_template.triggered.connect(self.open_project)
        save_template=QAction("Save Measure Configuration As…",self); save_template.triggered.connect(self.save_project_as)
        templates.addAction(open_template); templates.addAction(save_template)
        toolbar = QToolBar("Measure toolbar"); toolbar.setObjectName("mainToolbar"); toolbar.setMovable(False); self.addToolBar(toolbar)
        self.brand_title = QLabel("pyLOCO  MEASURE")
        self.brand_title.setObjectName("measureBrand")
        self.brand_title.setAccessibleName("pyLOCO Measure")
        toolbar.addWidget(self.brand_title)
        toolbar.addSeparator()
        self.session_toolbar_buttons={}
        for text,handler in (("New",self.new_project),("Open",self.open_full_measurement_session),("Save",self.save_active_session),("Open in FIT",self.explain_open)):
            button=QPushButton(text); button.clicked.connect(handler); toolbar.addWidget(button); self.session_toolbar_buttons[text]=button
        self.session_toolbar_buttons["Save"].setEnabled(True); self.session_toolbar_buttons["Open in FIT"].setEnabled(False)
        self.screenshot_controller=install_window_screenshot_actions(
            self,toolbar,application_slug="measure",
            directory_provider=self._screenshot_session_directory,file_menu=file_menu,
        )
        self.control_search=install_control_search(self, toolbar)
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
        self.measurement_tab_indices={"bpm_noise":2,"dispersion":3,"orm":4}
        self.measurement_tab_hosts={}
        for kind,label in (("bpm_noise","BPM Noise"),("dispersion","RF Dispersion"),("orm","ORM")):
            host=QWidget(); host_layout=QVBoxLayout(host); host_layout.setContentsMargins(0,0,0,0)
            self.measurement_tab_hosts[kind]=(host,host_layout); self.tabs.addTab(host,label)
        self.measurement_tab_hosts["bpm_noise"][1].addWidget(measurement)
        saved=QWidget(); saved_layout=QVBoxLayout(saved); saved_layout.setContentsMargins(20,18,20,18); saved_layout.setSpacing(12)
        saved_title=QLabel("Session overview"); saved_title.setObjectName("pageTitle"); saved_layout.addWidget(saved_title)
        saved_explanation=QLabel(
            "New creates a complete Measure session workspace. Open restores a saved full session. Save stores the configuration, reference lattice and measurement references without duplicating HDF5 files. File → Export Portable Session packages approved files for relocation."
        )
        saved_explanation.setObjectName("infoText"); saved_explanation.setWordWrap(True); saved_layout.addWidget(saved_explanation)
        self.review_machine_identity=QLabel(); self.review_machine_identity.setObjectName("planValue"); self.review_machine_identity.setWordWrap(True); saved_layout.addWidget(self.review_machine_identity)
        review_group=QGroupBox("Review"); self.review_summary_group=review_group; review_layout=QVBoxLayout(review_group)
        self.review_status=QLabel("No completed measurement yet."); self.review_status.setObjectName("runState"); self.review_status.setWordWrap(True)
        self.review_details=QLabel("Complete a measurement to review its saved data and validation status."); self.review_details.setObjectName("summaryCard"); self.review_details.setWordWrap(True); self.review_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.review_paths=QLabel("Measurement file: —\nSession manifest: —"); self.review_paths.setWordWrap(True); self.review_paths.setTextInteractionFlags(Qt.TextSelectableByMouse)
        review_layout.addWidget(self.review_status); review_layout.addWidget(self.review_details); review_layout.addWidget(self.review_paths); saved_layout.addWidget(review_group)
        roles_group=QGroupBox("Measurements in this session"); self.review_roles_group=roles_group; roles_layout=QGridLayout(roles_group); roles_layout.setVerticalSpacing(9); roles_layout.setHorizontalSpacing(14)
        self.role_status_labels={}; self.role_manage_actions={}; self.role_manage_buttons={}
        for row,(role,label) in enumerate((("bpm_noise","BPM noise"),("dispersion","RF dispersion"),("orm","ORM"))):
            roles_layout.addWidget(QLabel(label),row,0); status=QLabel("Missing"); status.setTextInteractionFlags(Qt.TextSelectableByMouse); self.role_status_labels[role]=status; roles_layout.addWidget(status,row,1)
            manage=QToolButton(); manage.setText("Manage…"); manage.setPopupMode(QToolButton.InstantPopup); manage.setMinimumHeight(30)
            menu=QMenu(manage); load=menu.addAction("Load existing measurement…"); replace_action=menu.addAction("Replace measurement…"); remove=menu.addAction("Remove from session")
            load.triggered.connect(lambda _=False,r=role:self._load_measurement_role(r,replace=False)); replace_action.triggered.connect(lambda _=False,r=role:self._load_measurement_role(r,replace=True)); remove.triggered.connect(lambda _=False,r=role:self._remove_measurement_role(r))
            manage.setMenu(menu); self.role_manage_buttons[role]=manage; self.role_manage_actions[role]={"load":load,"replace":replace_action,"remove":remove}; roles_layout.addWidget(manage,row,2)
        roles_layout.setColumnStretch(1,1); saved_layout.addWidget(roles_group)
        review_actions=QHBoxLayout(); review_actions.setSpacing(10); self.review_validate_button=QPushButton("Validate measurements for pyLOCO"); self.review_validate_button.setEnabled(False); self.review_validate_button.clicked.connect(self.validate_saved); self.review_open_button=QPushButton("Open in FIT"); self.review_open_button.setObjectName("measurePrimary"); self.review_open_button.setEnabled(False); self.review_open_button.setToolTip("Open this loaded full measurement session in pyLOCO FIT"); self.review_open_button.clicked.connect(self.explain_open); back_button=QPushButton("Back to active measurement"); back_button.clicked.connect(lambda:self.tabs.setCurrentIndex(self.measurement_tab_indices[self._active_selection_kind])); review_actions.addWidget(self.review_validate_button); review_actions.addWidget(self.review_open_button); review_actions.addWidget(back_button); review_actions.addStretch(1); saved_layout.addLayout(review_actions)
        saved_layout.addStretch()
        self.review_scroll=self._scroll_page(saved)
        saved.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)
        saved.setMinimumHeight(saved.sizeHint().height())
        self.tabs.addTab(self.review_scroll,"Review && Save")
        self.review_tab_index=5
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.setCentralWidget(root); self.statusBar().showMessage("Mock ready — read only"); self._update_workflow_tabs()
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self._mark_session_dirty)
        for widget in self.findChildren(QPlainTextEdit):
            widget.textChanged.connect(self._mark_session_dirty)
        for widget in self.findChildren(QCheckBox):
            widget.toggled.connect(self._mark_session_dirty)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._mark_session_dirty)
        spin_boxes = (*self.findChildren(QSpinBox), *self.findChildren(QDoubleSpinBox))
        for widget in spin_boxes:
            widget.valueChanged.connect(self._mark_session_dirty)

    def _mark_session_dirty(self, *_):
        self._session_dirty = True
        if hasattr(self, "session_save_action"):
            self.session_save_action.setEnabled(True)
            self.session_toolbar_buttons["Save"].setEnabled(True)
        self._update_session_title()

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
        active_kind=self._active_acquisition_kind or self.measurement_type.currentData()
        for kind,index in getattr(self,"measurement_tab_indices",{}).items():
            self.tabs.setTabEnabled(index,not running or kind==active_kind)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.cancel_button.setText("Cancel measurement" if running else "Cancel")
        self.repeat_button.setEnabled(not running)
        self.measurement_type.setEnabled(not running)
        if running:
            kind={"orm":"ORM","dispersion":"RF dispersion","bpm_noise":"BPM noise"}.get(self._active_acquisition_kind,"Measurement")
            self.statusBar().showMessage(f"ACQUIRING {kind.upper()} • {self.status_badge.text()}")
            if hasattr(self,"start_block_reason"):
                io="temporary corrector writes authorized; BPM acquisition is read only" if self._active_acquisition_kind=="orm" else "temporary RF writes authorized; BPM acquisition is read only" if self._active_acquisition_kind=="dispersion" and self.rf_control_mode.currentData()=="automatic" else "read-only acquisition"
                self.start_block_reason.setText(f"{kind} acquisition in progress — {io}.")
        else:
            self.statusBar().showMessage(self.status_badge.text())

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
        self.connection_progress=QProgressBar(); self.connection_progress.setRange(0,0); self.connection_progress.setTextVisible(True); self.connection_progress.setFormat("Testing connection…"); self.connection_progress.setVisible(False); grid.addWidget(self.connection_progress,1,2)
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
            editor.textChanged.connect(self._pyaml_configuration_changed)
        self.pyaml_magnet_writes = QCheckBox("Enable corrector magnet writes (ORM)")
        self.pyaml_rf_writes = QCheckBox("Enable RF frequency writes (dispersion)")
        for checkbox in (self.pyaml_magnet_writes, self.pyaml_rf_writes):
            checkbox.setChecked(True)
            pyaml_form.addRow(checkbox)
            # Authorization changes require reconnecting the adapter, but do
            # not alter lattice/device identity or invalidate Machine Mapping.
            checkbox.toggled.connect(lambda _checked:self._invalidate_pyaml(preserve_mapping=True))
        self.pyaml_orbit_correction = QCheckBox("Correct orbit after each ORM column")
        self.pyaml_orbit_correction.setToolTip("Use the measured corrector and its opposite-plane partner to return to the pre-kick orbit. Corrections remain applied.")
        pyaml_form.addRow(self.pyaml_orbit_correction)
        self.pyaml_orbit_correction.toggled.connect(lambda _checked:self._invalidate_pyaml(preserve_mapping=True))
        self.pyaml_simulator_identity_status=QLabel("Unavailable until pyAML provides verified runtime simulator identity.")
        self.pyaml_simulator_identity_status.setWordWrap(True); self.pyaml_simulator_identity_status.setObjectName("infoText"); pyaml_form.addRow("Simulator identity",self.pyaml_simulator_identity_status)
        pyaml_form.addRow(QLabel("Mode: design or live. RF plant defaults to RF. Writes default to enabled; uncheck to disable. Live connections require confirmation."))
        self.pyaml_group.setVisible(False)
        layout.addWidget(self.pyaml_group)
        reference_group=QGroupBox("Optional reference model — requires Accelerator Toolbox (AT)"); reference_layout=QFormLayout(reference_group); self._configure_form(reference_layout)
        self.reference_model_path=QLineEdit(); self.reference_model_path.setPlaceholderText("Automatic from pySC profile, or select an authoritative nominal .mat for LIVE")
        reference_row=QWidget(); rr=QHBoxLayout(reference_row); rr.setContentsMargins(0,0,0,0); rr.addWidget(self.reference_model_path,1); choose=QPushButton("Change Reference Lattice…"); choose.clicked.connect(self._choose_reference_model); rr.addWidget(choose)
        self.reference_model_summary=QLabel("No reference model loaded."); self.reference_model_summary.setObjectName("referenceModelCard"); self.reference_model_summary.setWordWrap(True); self.reference_model_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reference_model_variable=QLabel("Not applicable"); self.reference_model_variable.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reference_model_element_count=QLabel("Not available"); self.reference_model_element_count.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.change_reference_variable_button=QPushButton("Change MAT variable…"); self.change_reference_variable_button.setEnabled(False); self.change_reference_variable_button.clicked.connect(self._change_reference_mat_variable)
        variable_row=QWidget(); vrl=QHBoxLayout(variable_row); vrl.setContentsMargins(0,0,0,0); vrl.addWidget(self.reference_model_variable,1); vrl.addWidget(self.change_reference_variable_button)
        self.reference_model_requirement=QLabel("AT is installed — reference-model calculations are available." if find_spec("at") is not None else "AT is not installed — reference-model calculations and physical dispersion are unavailable."); self.reference_model_requirement.setWordWrap(True)
        self.compare_reference=QCheckBox("Compare measurements with reference model (diagnostic only)"); self.compare_reference.setChecked(True)
        self.compare_reference.toggled.connect(self._reference_comparison_changed)
        self.verify_alignment_button=QPushButton("Find Machine Mapping")
        self.verify_alignment_button.clicked.connect(self._verify_device_alignment)
        self.load_mapping_button=QPushButton("Load Machine Mapping…"); self.load_mapping_button.clicked.connect(self._load_device_mapping)
        self.mapping_template_button=QPushButton("Create Machine Mapping template…"); self.mapping_template_button.clicked.connect(self._save_mapping_template)
        self.mapping_python_helper_button=QPushButton("Python Mapping Helper…"); self.mapping_python_helper_button.clicked.connect(self._show_python_mapping_helper)
        self.save_mapping_button=QPushButton("Save validated Machine Mapping…"); self.save_mapping_button.setEnabled(False); self.save_mapping_button.clicked.connect(self._save_validated_mapping)
        self.alignment_details_button=QPushButton("Machine Mapping details…"); self.alignment_details_button.setEnabled(False)
        self.alignment_details_button.clicked.connect(self._show_alignment_details)
        alignment_actions=QWidget(); aal=QGridLayout(alignment_actions); aal.setContentsMargins(0,0,0,0); aal.addWidget(self.verify_alignment_button,0,0); aal.addWidget(self.load_mapping_button,0,1); aal.addWidget(self.mapping_template_button,1,0); aal.addWidget(self.mapping_python_helper_button,1,1); aal.addWidget(self.save_mapping_button,2,0); aal.addWidget(self.alignment_details_button,0,2,3,1); aal.setColumnStretch(3,1)
        self.device_alignment_status=QLabel("Load a reference lattice and connect pyAML, then find or load an exact Machine Mapping."); self.device_alignment_status.setWordWrap(True); self.device_alignment_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        reference_layout.addRow("Requirement",self.reference_model_requirement); reference_layout.addRow("Reference lattice",reference_row); reference_layout.addRow("MAT lattice variable",variable_row); reference_layout.addRow("Lattice elements",self.reference_model_element_count); reference_layout.addRow("MODEL — not live machine data",self.reference_model_summary); reference_layout.addRow("Machine Mapping",alignment_actions); reference_layout.addRow("Mapping status",self.device_alignment_status); reference_layout.addRow("",self.compare_reference)
        layout.addWidget(reference_group); return self._scroll_page(content)

    def _choose_reference_model(self):
        path=QFileDialog.getOpenFileName(self,"Select authoritative reference lattice","","AT lattice (*.mat);;All files (*)")[0]
        if path:self._load_reference_model(path,"User-selected reference model")

    def _choose_reference_mat_candidate(self,candidates,current=""):
        labels=[f"{candidate.variable} — {candidate.element_count} elements" for candidate in candidates]
        current_index=next((index for index,candidate in enumerate(candidates) if candidate.variable==current),0)
        choice,accepted=QInputDialog.getItem(self,"Select MAT lattice variable","Valid Accelerator Toolbox lattice variables in this MAT file:",labels,current_index,False)
        if not accepted:return None
        return candidates[labels.index(choice)]

    def _load_reference_model(self,path,source,mat_key=None):
        try:
            candidate_model=load_reference_model(path,source=source,mat_key=mat_key)
        except MultipleMatLatticesError as exc:
            selected=self._choose_reference_mat_candidate(exc.candidates,mat_key or "")
            if selected is None:return False
            try:candidate_model=load_reference_model(path,source=source,mat_key=selected.variable)
            except Exception as nested:
                QMessageBox.warning(self,"Reference model unavailable",str(nested)); return False
        except Exception as exc:
            QMessageBox.warning(self,"Reference model unavailable",str(exc)); return False
        self._model_orm_cache=None
        self._invalidate_device_alignment("Reference lattice or MAT variable changed — verify device alignment again.")
        self.reference_model=candidate_model
        self.reference_model_path.setText(str(candidate_model.path))
        self.reference_model_variable.setText(candidate_model.mat_variable or "Not applicable")
        self.change_reference_variable_button.setEnabled(candidate_model.path.suffix.lower()==".mat")
        self._display_reference_model()
        self._refresh_completed_reference_comparison()
        return True

    def _change_reference_mat_variable(self):
        if self.reference_model is None or self.reference_model.path.suffix.lower()!=".mat":return
        try:candidates=mat_lattice_candidates(self.reference_model.path)
        except Exception as exc:QMessageBox.warning(self,"Unable to inspect MAT variables",str(exc)); return
        if not candidates:
            QMessageBox.warning(self,"Unable to inspect MAT variables","The selected file contains no valid Accelerator Toolbox lattice variables."); return
        selected=self._choose_reference_mat_candidate(candidates,self.reference_model.mat_variable or "")
        if selected is None or selected.variable==self.reference_model.mat_variable:return
        self._load_reference_model(self.reference_model.path,self.reference_model.source,selected.variable)

    def _alignment_inventory(self):
        return {
            "bpm":[device.name for device in self.selected_devices],
            "horizontal_corrector":[device.name for device in self.selected_hcorrectors],
            "vertical_corrector":[device.name for device in self.selected_vcorrectors],
        }

    def _pyaml_inventory_bundle(self):
        """Merge file-backed and connected public pyAML device inventories."""
        if self.adapter_combo.currentData()!="pyaml":return {},{}
        file_inventory={}; file_metadata={}
        path=Path(self.pyaml_fields["config_path"].text()).expanduser()
        if path.is_file():
            try:
                import yaml
                document=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                file_inventory,file_metadata=pyaml_configuration_inventory(document)
            except Exception:pass
        runtime_inventory={}; runtime_metadata={}
        provider=getattr(self.adapter,"machine_mapping_inventory",None)
        if self.connection_verified and callable(provider):
            try:runtime_inventory,runtime_metadata=provider()
            except Exception:pass
        roles=("bpm","horizontal_corrector","vertical_corrector","normal_quadrupole",
               "skew_quadrupole","sextupole","octupole","dipole")
        merged={role:list(file_inventory.get(role,())) for role in roles}
        for role in roles:
            for name in runtime_inventory.get(role,()):
                if name not in merged[role]:merged[role].append(name)
        metadata=dict(file_metadata)
        for key,values in runtime_metadata.items():
            combined=dict(metadata.get(key,{}))
            combined.update({name:value for name,value in values.items() if value not in (None,"")})
            metadata[key]=combined
        return merged,metadata

    def _pyaml_named_inventories(self):
        """Read every supported physical device configured in pyAML."""
        return self._pyaml_inventory_bundle()[0]

    def _pyaml_inventory_metadata(self):
        return self._pyaml_inventory_bundle()[1]

    def _pyaml_fitted_magnet_inventory(self):
        inventories=self._pyaml_named_inventories()
        return {role:list(inventories.get(role,())) for role in
                ("normal_quadrupole","skew_quadrupole","sextupole","octupole","dipole")}

    def _complete_machine_mapping_template(self):
        if self.reference_model is None:raise ValueError("Load the authoritative AT reference lattice first.")
        inventories=self._pyaml_named_inventories()
        configured_total=sum(len(names) for names in inventories.values())
        if not configured_total:
            raise ValueError(
                "No configured pyAML device identities were discovered. Test the connection first, or select "
                "a configuration that exposes public devices/arrays; an empty Machine Mapping template was not created."
            )
        measured={role:list(inventories.get(role,())) for role in
                  ("bpm","horizontal_corrector","vertical_corrector")}
        document=complete_machine_mapping_template(self.reference_model.ring,measured,
            self._pyaml_fitted_magnet_inventory(),self._pyaml_inventory_metadata())
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        document["reference_lattice"]={"path":str(self.reference_model.path),
            "sha256":self.reference_model.checksum_sha256,
            "element_count":self.reference_model.element_count,
            "mat_variable":self.reference_model.mat_variable or ""}
        document["machine_profile"]={"backend":"pyaml",
            "configuration_path":str(configuration),
            "configuration_sha256":session_file_sha256(configuration) if configuration.is_file() else "",
            "mode":self.pyaml_fields["mode"].text().strip()}
        return document

    def _ensure_complete_machine_mapping(self):
        """Rebuild and validate the master from the active pyAML configuration.

        This is the single serialization gate used by mapping export and full
        session save.  It deliberately does not derive scope from the selected
        measurement devices, so a legacy 896-row mapping is upgraded instead
        of being re-exported as though it were the complete machine mapping.
        """
        if self.adapter_combo.currentData()!="pyaml":
            return self.machine_mapping_document or self.measurement_device_alignment
        if self.reference_model is None:raise ValueError("Load the authoritative AT reference lattice first.")
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        if not configuration.is_file():raise ValueError("The active pyAML configuration file is unavailable.")
        proposal=reconcile_configured(self.machine_mapping_document,self._complete_machine_mapping_template(),
                                      ring=self.reference_model.ring)
        configured=self._pyaml_named_inventories()
        measurement_inventory={role:list(configured.get(role,())) for role in
            ("bpm","horizontal_corrector","vertical_corrector")}
        checked=find_exact_device_mappings(self.reference_model.ring,measurement_inventory,
            lattice_path=self.reference_model.path,configuration_path=configuration,
            provenance="Measure complete configured pyAML inventory matched to authoritative AT lattice")
        master=merge_measurement(proposal,checked)
        expected=sum(len(names) for names in configured.values())
        if len(master.get("mappings",()))!=expected:
            raise ValueError(f"Complete Machine Mapping serialization mismatch: expected {expected} configured devices, got {len(master.get('mappings',()))}.")
        self.machine_mapping_document=master
        return master

    @staticmethod
    def _master_mapping_counts(document):
        roles={}
        for row in (document or {}).get("mappings",()):
            role=str(row.get("role","") or "unclassified"); roles[role]=roles.get(role,0)+1
        return len((document or {}).get("mappings",())),roles

    def _invalidate_device_alignment(self,reason="Device inventory changed — validate the device mapping again."):
        self.measurement_device_alignment=None; self._model_orm_cache=None
        if hasattr(self,"device_alignment_status"):
            self.device_alignment_status.setText(reason)
            self.alignment_details_button.setEnabled(False)
            if hasattr(self,"save_mapping_button"):self.save_mapping_button.setEnabled(False)
        self._update_reference_comparison_enabled()

    def _update_reference_comparison_enabled(self):
        if not hasattr(self,"compare_reference"):return
        required=self.adapter_combo.currentData()=="pyaml"
        counts=mapping_counts(self.measurement_device_alignment) if self.measurement_device_alignment else {}
        ready=(not required or bool(counts) and counts["mapped"]==counts["total"])
        self.compare_reference.setEnabled(ready)
        if required and not ready:self.compare_reference.setToolTip("Validate the exact pyAML device mapping before calculating a reference-model comparison.")
        else:self.compare_reference.setToolTip("Read-only diagnostic comparison; acquisition and saved measurements are unchanged.")

    def _verify_device_alignment(self):
        if self.adapter_combo.currentData()!="pyaml":
            self.device_alignment_status.setText("Built-in pySC profiles retain their profile-bound lattice/device resolver.")
            self._update_reference_comparison_enabled(); return
        if self.reference_model is None:
            self._invalidate_device_alignment("Cannot verify: load the authoritative AT reference lattice first."); return
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        if not configuration.is_file():
            self._invalidate_device_alignment("Cannot verify: the active pyAML configuration file is unavailable."); return
        inventory=self._alignment_inventory()
        try:
            document=find_exact_device_mappings(self.reference_model.ring,inventory,
                lattice_path=self.reference_model.path,configuration_path=configuration,
                provenance="Measure exact AT Device identity")
            self._ensure_complete_machine_mapping()
            counts=mapping_counts(document)
            master_total,_roles=self._master_mapping_counts(self.machine_mapping_document)
            measurement_total=sum(len(self._pyaml_named_inventories().get(role,())) for role in
                                  ("bpm","horizontal_corrector","vertical_corrector"))
            if counts["mapped"]!=counts["total"]:
                self.measurement_device_alignment=document
                self.device_alignment_status.setText(
                    f"Mapping incomplete — {counts['mapped']}/{counts['total']} mapped; "
                    f"{counts['unresolved']} unresolved, {counts['ambiguous']} ambiguous, "
                    f"{counts['duplicate']} duplicate, {counts['invalid']} invalid. Comparison remains disabled.")
            else:
                self.measurement_device_alignment=document
                self.device_alignment_status.setText(
                    f"Machine Mapping: {master_total:,} configured devices · Measurement Mapping: {measurement_total:,} available BPM/H/V devices · "
                    f"Selected measurement subset: {counts['mapped']}/{counts['total']} identities matched by the recorded exact Device, exact FamName, or legacy-suffix method "
                    "(including validated legacy component-suffix interpretation where recorded). "
                    "Measured BPM/corrector order will be preserved in model calculations.")
            self.alignment_details_button.setEnabled(True)
            self.save_mapping_button.setEnabled(counts["mapped"]==counts["total"])
        except Exception as exc:
            self._invalidate_device_alignment(f"Device mapping unavailable: {exc}"); return
        self._update_reference_comparison_enabled()
        if self.compare_reference.isEnabled():self._refresh_completed_reference_comparison()

    def _mapping_context(self):
        if self.reference_model is None:raise ValueError("Load the authoritative AT reference lattice first.")
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        if self.adapter_combo.currentData()=="pyaml" and not configuration.is_file():
            raise ValueError("The active pyAML configuration file is unavailable.")
        return self._alignment_inventory(),configuration

    @staticmethod
    def _write_json_atomic(path,document):
        target=Path(path).expanduser().resolve(); target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+".tmp")
        temporary.write_text(json.dumps(document,indent=2)+"\n",encoding="utf-8")
        temporary.replace(target)

    def _load_device_mapping(self):
        path=QFileDialog.getOpenFileName(self,"Load Machine Mapping","","pyLOCO Machine Mapping (*.json *.yaml *.yml);;All files (*)")[0]
        if not path:return
        try:
            inventory,configuration=self._mapping_context(); source=load_document(path)
            common=is_machine_mapping(source); projected=measurement_projection(source) if common else source
            template=is_machine_mapping_template(source) if common else is_unvalidated_mapping_template(source)
            if common:
                full=self._pyaml_named_inventories(); full_inventory={role:list(full.get(role,())) for role in ("bpm","horizontal_corrector","vertical_corrector")}
                # Legacy shared files may contain only the measurement subset.
                # Reconcile them with the active configured inventory, then
                # validate every configured measurement device afresh.
                proposal=reconcile_configured(source,self._complete_machine_mapping_template(),
                                              ring=self.reference_model.ring)
                projected=measurement_projection(proposal)
                if template:projected={**projected,"mapping_provenance":"Externally completed complete Machine Mapping template"}
                full_checked=validate_mapping(projected,self.reference_model.ring,full_inventory,lattice_path=self.reference_model.path,
                    configuration_path=configuration,require_provenance=False,
                    accept_explicit_ordinals=template)
                master=merge_measurement(proposal,full_checked)
                wanted={(role,name) for role,names in inventory.items() for name in names}
                selected={**full_checked,"mappings":[row for row in full_checked["mappings"]
                    if (row.get("role"),row.get("measured_name")) in wanted]}
                checked=validate_mapping(selected,self.reference_model.ring,inventory,lattice_path=self.reference_model.path,
                    configuration_path=configuration,require_provenance=False)
                self.machine_mapping_document=master
            else:
                if template:projected={**projected,"mapping_provenance":"Externally completed mapping template, validated against the active lattice and inventory"}
                checked=validate_mapping(projected,self.reference_model.ring,inventory,lattice_path=self.reference_model.path,
                    configuration_path=configuration,require_provenance=bool(projected.get("schema")) and not template,
                    accept_explicit_ordinals=template)
                self.machine_mapping_document=merge_measurement(None,checked)
            counts=mapping_counts(checked); self.measurement_device_alignment=checked
            master_total,_roles=self._master_mapping_counts(self.machine_mapping_document)
            measurement_total=sum(len(self._pyaml_named_inventories().get(role,())) for role in
                                  ("bpm","horizontal_corrector","vertical_corrector"))
            kind="Filled template validated" if template else "Loaded mapping"
            self.device_alignment_status.setText(
                f"{kind} · Machine Mapping: {master_total:,} configured devices · "
                f"Measurement Mapping: {measurement_total:,} available BPM/H/V devices · "
                f"selected subset {counts['mapped']}/{counts['total']} mapped; "
                f"{counts['unresolved']} unresolved, {counts['ambiguous']} ambiguous, "
                f"{counts['duplicate']} duplicate, {counts['invalid']} invalid.")
            self.alignment_details_button.setEnabled(True); self.save_mapping_button.setEnabled(counts["mapped"]==counts["total"])
            self._update_reference_comparison_enabled()
        except Exception as exc:QMessageBox.warning(self,"Cannot load Machine Mapping",str(exc))

    def _save_mapping_template(self):
        path=QFileDialog.getSaveFileName(self,"Create Machine Mapping template","machine-mapping-template.json","JSON (*.json)")[0]
        if not path:return
        try:self._write_json_atomic(path,editable_machine_mapping_template(
            self._complete_machine_mapping_template()))
        except Exception as exc:QMessageBox.warning(self,"Cannot create mapping template",str(exc)); return
        self.last_mapping_template_path=Path(path).expanduser().resolve()
        self.statusBar().showMessage(f"Created mapping template: {path}")

    def _show_python_mapping_helper(self):
        dialog=QDialog(self); dialog.setWindowTitle("Python Mapping Helper"); dialog.setMinimumSize(820,620); dialog.resize(980,760); dialog.setSizeGripEnabled(True)
        layout=QVBoxLayout(dialog); form=QFormLayout(); layout.addLayout(form)
        template=QLineEdit(str(self.last_mapping_template_path or "")); lattice=QLineEdit(str(self.reference_model.path) if self.reference_model is not None else "")
        variable=QLineEdit(self.reference_model.mat_variable or "" if self.reference_model is not None else "")
        default_parent=self.last_mapping_template_path.parent if self.last_mapping_template_path is not None else Path.cwd()
        output=QLineEdit(str(default_parent/"machine-mapping-completed.json"))
        form.addRow("Mapping template JSON",template); form.addRow("AT lattice",lattice); form.addRow("MAT variable",variable); form.addRow("Output JSON",output)
        editor=QPlainTextEdit(); editor.setLineWrapMode(QPlainTextEdit.NoWrap); layout.addWidget(editor,1)
        def refresh_script(*_):editor.setPlainText(python_mapping_helper_script(template.text(),lattice.text(),variable.text(),output.text()))
        for field in (template,lattice,variable,output):field.textChanged.connect(refresh_script)
        refresh_script()
        buttons=QHBoxLayout(); copy=QPushButton("Copy Script"); save=QPushButton("Save Script…"); close=QPushButton("Close")
        def copy_script():
            QApplication.clipboard().setText(editor.toPlainText()); copy.setText("Copied ✓"); QTimer.singleShot(1800,lambda: copy.setText("Copy Script"))
        def save_script():
            path=QFileDialog.getSaveFileName(dialog,"Save Python mapping helper","fill-measurement-device-mapping.py","Python (*.py)")[0]
            if path:Path(path).expanduser().write_text(editor.toPlainText(),encoding="utf-8")
        copy.clicked.connect(copy_script); save.clicked.connect(save_script); close.clicked.connect(dialog.accept)
        buttons.addWidget(copy); buttons.addWidget(save); buttons.addStretch(1); buttons.addWidget(close); layout.addLayout(buttons)
        self.mapping_helper_dialog=dialog; self.mapping_helper_template=template; self.mapping_helper_lattice=lattice; self.mapping_helper_variable=variable; self.mapping_helper_output=output; self.mapping_helper_editor=editor; self.mapping_helper_copy=copy; self.mapping_helper_save=save
        dialog.exec()

    def _save_validated_mapping(self):
        if self.measurement_device_alignment is None:return
        try:
            inventory,configuration=self._mapping_context()
            checked=validate_mapping(self.measurement_device_alignment,self.reference_model.ring,inventory,
                lattice_path=self.reference_model.path,configuration_path=configuration,require_provenance=True)
            counts=mapping_counts(checked)
            if counts["mapped"]!=counts["total"]:raise ValueError("Only a complete, currently valid mapping can be saved as validated.")
        except Exception as exc:QMessageBox.warning(self,"Cannot save validated mapping",str(exc)); return
        try:shared=self._ensure_complete_machine_mapping()
        except Exception as exc:QMessageBox.warning(self,"Cannot save validated mapping",str(exc)); return
        path=QFileDialog.getSaveFileName(self,"Save validated Machine Mapping","machine-mapping.json","JSON (*.json)")[0]
        if not path:return
        if Path(path).exists() and QMessageBox.question(self,"Replace mapping file?",f"Replace the existing mapping file?\n{path}",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Yes:return
        try:self._write_json_atomic(path,shared)
        except Exception as exc:QMessageBox.warning(self,"Cannot save validated mapping",str(exc)); return
        total,_roles=self._master_mapping_counts(shared)
        self.statusBar().showMessage(f"Saved complete Machine Mapping — {total:,} configured devices: {path}")

    def _validated_alignment_ordinals(self):
        if self.adapter_combo.currentData()!="pyaml":return None
        if self.reference_model is None or self.measurement_device_alignment is None:
            raise ValueError("Validate the exact device mapping before comparing with the reference model")
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        checked=validate_mapping(self.measurement_device_alignment,self.reference_model.ring,
            self._alignment_inventory(),lattice_path=self.reference_model.path,
            configuration_path=configuration,require_provenance=True)
        counts=mapping_counts(checked)
        if counts["mapped"]!=counts["total"]:
            raise ValueError(f"Device mapping is incomplete: {counts['mapped']}/{counts['total']} mapped")
        return {role:ordered_ordinals(checked,role) for role in ("bpm","horizontal_corrector","vertical_corrector")}

    def _model_mapping_kwargs(self):
        ordinals=self._validated_alignment_ordinals()
        if ordinals is None:return {}
        return {
            "bpm_ordinals":ordinals["bpm"],
            "horizontal_corrector_ordinals":ordinals["horizontal_corrector"],
            "vertical_corrector_ordinals":ordinals["vertical_corrector"],
        }

    def _reference_model_provenance(self):
        provenance=dict(self.reference_model.provenance())
        if self.measurement_device_alignment:
            counts=mapping_counts(self.measurement_device_alignment)
            provenance.update({
                "device_alignment_schema":self.measurement_device_alignment.get("schema"),
                "device_alignment_lattice_sha256":self.measurement_device_alignment.get("lattice",{}).get("sha256"),
                "device_alignment_configuration_sha256":self.measurement_device_alignment.get("configuration",{}).get("sha256"),
                "device_alignment_inventory_sha256":self.measurement_device_alignment.get("inventory",{}).get("sha256"),
                "device_alignment_mapped":counts.get("mapped",0),
                "device_alignment_total":counts.get("total",0),
            })
        return provenance

    def _show_alignment_details(self):
        if self.measurement_device_alignment is None:return
        # Revalidate the raw records before presenting a document as
        # validated. This distinguishes source corruption from table sorting.
        if self.reference_model is not None and self.adapter_combo.currentData()=="pyaml":
            try:
                inventory,configuration=self._mapping_context()
                selected_document=validate_mapping(self.measurement_device_alignment,self.reference_model.ring,inventory,
                    lattice_path=self.reference_model.path,configuration_path=configuration,require_provenance=True)
            except Exception as exc:
                self._invalidate_device_alignment(f"Mapping integrity failure: {exc}")
                QMessageBox.warning(self,"Cannot display validated mapping",f"The stored mapping records failed revalidation and were not presented as validated:\n{exc}")
                return
            self.measurement_device_alignment=selected_document
        try:document=self._ensure_complete_machine_mapping()
        except Exception as exc:QMessageBox.warning(self,"Cannot display Machine Mapping",str(exc)); return
        common=is_machine_mapping(document)
        dialog=QDialog(self); dialog.setWindowTitle("Complete Machine Mapping — configured pyAML devices"); dialog.setWindowFlag(Qt.WindowMaximizeButtonHint,True); dialog.setSizeGripEnabled(True); dialog.setMinimumSize(900,520); dialog.resize(1250,700)
        layout=QVBoxLayout(dialog)
        total,roles=self._master_mapping_counts(document)
        measurement_total=sum(roles.get(role,0) for role in ("bpm","horizontal_corrector","vertical_corrector"))
        explanation=QLabel(f"Machine Mapping: {total:,} configured devices. Measurement Mapping: {measurement_total:,} available BPM/H/V devices. Rows come from the selected pyAML configuration; AT fields come from the checksummed reference lattice. Measurement selections are stored separately.")
        explanation.setWordWrap(True); layout.addWidget(explanation)
        role_filter=QComboBox(); role_filter.addItem("All configured device roles","")
        for role in sorted(roles):role_filter.addItem(f"{role.replace('_',' ').title()} ({roles[role]:,})",role)
        layout.addWidget(role_filter,0,Qt.AlignLeft)
        table=CopyableMappingTable(0,9); table.setHorizontalHeaderLabels(("Measured device (pyAML)","Role","AT FamName","AT Device","AT index (0-based)","Match method","Component / steering capability","Mapping status","Evidence / problem"))
        configure_mapping_table(table,(
            "Exact device name from the selected pyAML measurement inventory.","Measurement role and plane.","FamName attribute on the selected AT lattice element.","Device attribute on the selected AT lattice element; distinct from a pyAML control name.","AT element index (0-based) in the active lattice.","Recorded identity_match method; never inferred by this dialog.","Required component and whether role compatibility passed.","Validated mapping state.","Detailed identity and steering evidence or actionable problem."
        ))
        rows=document.get("mappings",())
        def populate():
            selected_role=role_filter.currentData(); visible=[item for item in rows if not selected_role or item.get("role")==selected_role]
            table.setSortingEnabled(False); table.setRowCount(len(visible))
            for row,item in enumerate(visible):
                status=str(item.get("status","") or "Not available")
                capability=item.get("component") or "BPM position"
                evidence=item.get("evidence") or item.get("identity_evidence") or "Exact configured identity checked against active lattice"
                measured_name=item.get("pyaml_device") if common else item.get("measured_name")
                lattice_name=item.get("lattice_fam_name") if common else item.get("lattice_name")
                if not common:
                    evidence=item.get("reason") or item.get("identity_evidence") or evidence
                    capability=(item.get("component") or "BPM identity")+(" — validated" if status=="mapped" else " — not validated")
                values=(measured_name or "Not available",item.get("role") or "Not available",
                    lattice_name or "Not available",item.get("lattice_device") or "Not available",
                    item.get("lattice_ordinal") if item.get("lattice_ordinal") is not None else "Not available",
                    identity_match_label(item.get("identity_match")),capability,status.replace("_"," ").title(),evidence)
                record_identity=(row,str(item.get("role","") or ""),str(measured_name or ""),item.get("lattice_ordinal"))
                for column,value in enumerate(values):
                    cell=QTableWidgetItem(str(value)); cell.setData(Qt.UserRole,record_identity); cell.setToolTip(str(value)); table.setItem(row,column,cell)
            table.setSortingEnabled(True)
        role_filter.currentIndexChanged.connect(populate); populate()
        layout.addWidget(table,1); self.alignment_details_dialog=dialog; self.alignment_details_table=table
        buttons=QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons); dialog.exec()

    def _reference_comparison_changed(self,*_):
        self._model_orm_cache=None
        self._refresh_completed_reference_comparison()

    def _refresh_completed_reference_comparison(self):
        """Refresh diagnostic model plots when a model is loaded after acquisition."""
        result=getattr(self,"result",None)
        if isinstance(result,ORMResult):
            self._show_orm_result(result)
        elif isinstance(result,DispersionResult):
            self._show_dispersion_result(result)

    def _display_reference_model(self):
        model=self.reference_model; self.reference_model_path.setText(str(model.path))
        self.reference_model_element_count.setText(f"{len(model.ring):,}")
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
            f"<div><b>Lattice:</b> {escape(model.path.name)} &nbsp; <b>Elements:</b> {len(model.ring):,} &nbsp; <b>SHA-256:</b> {escape(model.checksum_sha256[:12])}…</div>"
            f"<div><b>File selection:</b> {selection}</div><br>"
            f"<table cellspacing='0' cellpadding='4' width='100%'><tr><th align='left'>Quantity</th><th align='left'>Value</th><th align='left'>Source</th></tr>{body}</table>")

    def _load_profile_reference_model(self):
        self._model_orm_cache=None
        try:self.reference_model=reference_model_for_pysc(self.pysc_profile_combo.currentData())
        except Exception as exc:self.reference_model=None; self.reference_model_element_count.setText("Not available"); self.reference_model_summary.setText(f"Automatic profile reference model unavailable: {exc}"); return
        self.reference_model_path.setText(str(self.reference_model.path))
        self.reference_model_variable.setText(self.reference_model.mat_variable or "Not applicable")
        self.change_reference_variable_button.setEnabled(False)
        self._display_reference_model()

    def _bpms_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.devices_identity=QLabel("Machine inventory"); self.devices_identity.setObjectName("planValue"); self.devices_identity.setWordWrap(True); layout.addWidget(self.devices_identity)
        self.device_selection_configuration_group=QGroupBox("Device selection configuration")
        self.device_selection_configuration_group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
        selection_config_layout=QHBoxLayout(self.device_selection_configuration_group)
        selection_config_text=QLabel("Reuse the active measurement's ordered BPM, H-corrector and V-corrector selections, filters and exclusions.")
        selection_config_text.setWordWrap(True); selection_config_text.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)
        self.save_device_selection_button=QPushButton("Save selection…"); self.save_device_selection_button.clicked.connect(self._save_selection_configuration)
        self.load_device_selection_button=QPushButton("Load selection…"); self.load_device_selection_button.clicked.connect(self._load_selection_configuration)
        selection_config_layout.addWidget(selection_config_text,1); selection_config_layout.addWidget(self.save_device_selection_button); selection_config_layout.addWidget(self.load_device_selection_button)
        layout.addWidget(self.device_selection_configuration_group)
        group=QGroupBox("BPMs — selection for the active measurement"); group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum); form=QFormLayout(group); self._configure_form(form)
        self.selection_method=QComboBox(); self.selection_method.addItem("All available BPMs","all"); self.selection_method.addItem("Load names file","names_file"); self.selection_method.addItem("Manual names / positions","manual"); self.selection_method.currentIndexChanged.connect(self._selection_method_changed); form.addRow("Selection method",self.selection_method)
        self.names_row=QWidget(); nr=QHBoxLayout(self.names_row); nr.setContentsMargins(0,0,0,0); self.names_file=QLineEdit(); browse=QPushButton("Browse…"); browse.clicked.connect(self.browse_names); nr.addWidget(self.names_file,1); nr.addWidget(browse); form.addRow("BPM names file",self.names_row); self.names_label=form.labelForField(self.names_row)
        self.manual_input=QLineEdit(); self.manual_input.setPlaceholderText("BPM-001, BPM-004 or positions 0, 3"); form.addRow("Manual selection",self.manual_input); self.manual_label=form.labelForField(self.manual_input)
        self.bpm_exclusions=QLineEdit(); self.bpm_exclusions.setPlaceholderText("Selected-list positions to exclude, e.g. 1, 4"); form.addRow("Excluded BPM positions",self.bpm_exclusions); self.bpm_exclusions_label=form.labelForField(self.bpm_exclusions); self.bpm_exclusions.textChanged.connect(self.refresh_preview)
        self.bpm_search=QLineEdit(); self.bpm_search.setPlaceholderText("Name/identifier patterns, comma separated (for example c04, c05 or SF*)…"); self.bpm_search.textChanged.connect(self._filter_bpm_table); form.addRow("Search / filter",self.bpm_search)
        self.bpm_visibility=QComboBox(); self.bpm_visibility.addItem("All devices","all"); self.bpm_visibility.addItem("Selected only","selected"); self.bpm_visibility.addItem("Unselected only","unselected"); self.bpm_visibility.currentIndexChanged.connect(self._filter_bpm_table); form.addRow("Show devices",self.bpm_visibility)
        quick=QHBoxLayout(); select_all=QPushButton("Select All"); select_all.clicked.connect(self._select_all_bpms); clear=QPushButton("Clear"); clear.clicked.connect(self._clear_bpms); select_filtered=QPushButton("Replace with filtered"); select_filtered.clicked.connect(self._select_filtered_bpms); self.bpm_select_only_highlighted_button=QPushButton("Select only highlighted"); self.bpm_select_only_highlighted_button.setEnabled(False); self.bpm_select_only_highlighted_button.setToolTip("Replace the checked BPM selection with the highlighted rows"); self.bpm_select_only_highlighted_button.clicked.connect(lambda:self._select_only_highlighted("bpm")); self.bpm_add_highlighted_button=QPushButton("Add highlighted to selection"); self.bpm_add_highlighted_button.setEnabled(False); self.bpm_add_highlighted_button.setToolTip("Keep checked BPMs and add highlighted rows without duplicates"); self.bpm_add_highlighted_button.clicked.connect(lambda:self._select_highlighted("bpm")); preview=QPushButton("Refresh"); preview.clicked.connect(self.refresh_preview); quick.addWidget(select_all); quick.addWidget(clear); quick.addWidget(select_filtered); quick.addWidget(self.bpm_select_only_highlighted_button); quick.addWidget(self.bpm_add_highlighted_button); quick.addStretch(); quick.addWidget(preview); form.addRow("",quick)
        subset=QHBoxLayout(); self.bpm_subset_n=NoWheelSpinBox(); self.bpm_subset_n.setRange(1,100000); self.bpm_subset_n.setValue(20); first_n=QPushButton("First N"); first_n.clicked.connect(lambda:self._select_device_subset("bpm","first",self.bpm_subset_n.value())); uniform_n=QPushButton("Uniform N"); uniform_n.clicked.connect(lambda:self._select_device_subset("bpm","uniform",self.bpm_subset_n.value())); subset.addWidget(self.bpm_subset_n); subset.addWidget(first_n); subset.addWidget(uniform_n); subset.addStretch(); form.addRow("Selection helpers",subset); layout.addWidget(group)
        table_group=QGroupBox("Selection preview"); table_group.setMinimumHeight(350); tl=QVBoxLayout(table_group); self.preview_table=QTableWidget(0,3); self.preview_table.setMinimumHeight(260); self.preview_table.setHorizontalHeaderLabels(["Selection position","BPM name","Adapter/device identifier"]); self.preview_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents); self.preview_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents); self.preview_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch); self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers); self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection); tl.addWidget(self.preview_table); self.selection_message=QLabel(); tl.addWidget(self.selection_message); layout.addWidget(table_group)
        self.manual_input.textChanged.connect(self.refresh_preview)
        self.preview_table.itemSelectionChanged.connect(self._filter_bpm_table)
        self.preview_table.itemChanged.connect(self._bpm_checkbox_changed)
        self.orm_corrector_group=QGroupBox("Correctors — ORM selection"); cg=QVBoxLayout(self.orm_corrector_group); self.corrector_selection_widgets={}
        self.corrector_diagnostic_legend=QLabel("SP = Setpoint; RBV = Readback Value. Kick/strength and electrical current are distinct quantities; units are never inferred from one another.")
        self.corrector_diagnostic_legend.setObjectName("infoText"); self.corrector_diagnostic_legend.setWordWrap(True)
        self.corrector_diagnostic_legend.setToolTip("Table values come from the named control-system components. Hover a column heading for its source and unit policy.")
        cg.addWidget(self.corrector_diagnostic_legend)
        demo=QHBoxLayout(); one=QPushButton("Demo: 1 H + 1 V"); one.clicked.connect(self._select_demo_one_each); small=QPushButton("Demo: small uniform ORM"); small.clicked.connect(self._select_demo_small_orm); demo.addWidget(one); demo.addWidget(small); demo.addStretch(); cg.addLayout(demo)
        for key,title,devices in (("hcor","Horizontal correctors",self.horizontal_correctors),("vcor","Vertical correctors",self.vertical_correctors)):
            box=QGroupBox(title); form=QFormLayout(box); self._configure_form(form); method=QComboBox(); method.addItem("All available","all"); method.addItem("Load names file","names_file"); method.addItem("Manual names / positions","manual")
            manual=QLineEdit(); manual.setPlaceholderText("Exact names, positions, or comma-separated wildcards such as SF*, SJ*"); file_edit=QLineEdit(); browse=QPushButton("Browse…"); file_row=QWidget(); fr=QHBoxLayout(file_row); fr.setContentsMargins(0,0,0,0); fr.addWidget(file_edit,1); fr.addWidget(browse)
            exclusion=QLineEdit(); exclusion.setPlaceholderText("Selected-list positions to exclude, e.g. 1, 4")
            search=QLineEdit(); search.setPlaceholderText("Name/identifier patterns, comma separated (for example c04, c05 or SF*)…"); visibility=QComboBox(); visibility.addItem("All devices","all"); visibility.addItem("Selected only","selected"); visibility.addItem("Unselected only","unselected"); preview=QPushButton("Refresh"); table=QTableWidget(0,8); table.setMinimumHeight(150); table.setHorizontalHeaderLabels(["Selection position","Device name","Adapter/device identifier","Plane","Kick setpoint","Kick readback","Current setpoint","Current readback"]); table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)
            diagnostic_header_tooltips=(
                "KICK.SP — requested kick/strength setpoint. The displayed numeric value remains in the unit supplied by the component adapter.",
                "KICK.RBV — independently reported kick/strength readback value. It is not an electrical-current value.",
                "CURRENT.SP — electrical-current setpoint. It is not converted to or presented as kick/strength.",
                "CURRENT.RBV — independently reported electrical-current readback value. It is not converted to or presented as kick/strength.",
            )
            for column,tooltip in enumerate(diagnostic_header_tooltips,start=4):
                table.horizontalHeaderItem(column).setToolTip(tooltip+" Units are shown only when authoritative component metadata are available; otherwise no unit is asserted.")
            table.setSelectionBehavior(QAbstractItemView.SelectRows); table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            controls=QHBoxLayout(); all_button=QPushButton("Select All"); all_button.clicked.connect(lambda _=False,k=key:self._select_all_correctors(k)); clear_button=QPushButton("Clear"); clear_button.clicked.connect(lambda _=False,k=key:self._clear_correctors(k)); filtered=QPushButton("Replace with filtered"); filtered.clicked.connect(lambda _=False,k=key:self._select_filtered_correctors(k)); select_only_highlighted=QPushButton("Select only highlighted"); select_only_highlighted.setEnabled(False); select_only_highlighted.setToolTip("Replace the checked corrector selection with the highlighted rows"); select_only_highlighted.clicked.connect(lambda _=False,k=key:self._select_only_highlighted(k)); add_highlighted=QPushButton("Add highlighted to selection"); add_highlighted.setEnabled(False); add_highlighted.setToolTip("Keep checked correctors and add highlighted rows without duplicates"); add_highlighted.clicked.connect(lambda _=False,k=key:self._select_highlighted(k)); controls.addWidget(all_button); controls.addWidget(clear_button); controls.addWidget(filtered); controls.addWidget(select_only_highlighted); controls.addWidget(add_highlighted)
            if key=="vcor":
                self.copy_h_to_v_button=QPushButton("Copy H selection to V…"); self.copy_h_to_v_button.clicked.connect(self._copy_h_selection_to_v); controls.addWidget(self.copy_h_to_v_button)
            controls.addStretch(); controls.addWidget(preview)
            subset=QHBoxLayout(); nbox=NoWheelSpinBox(); nbox.setRange(1,100000); nbox.setValue(5); first_n=QPushButton("First N"); first_n.clicked.connect(lambda _=False,k=key,n=nbox:self._select_device_subset(k,"first",n.value())); uniform_n=QPushButton("Uniform N"); uniform_n.clicked.connect(lambda _=False,k=key,n=nbox:self._select_device_subset(k,"uniform",n.value())); subset.addWidget(nbox); subset.addWidget(first_n); subset.addWidget(uniform_n); subset.addStretch()
            count=QLabel(); count.setObjectName("planValue"); empty=QLabel(); empty.setObjectName("infoText"); empty.setWordWrap(True); empty.setVisible(False)
            form.addRow("Selection method",method); form.addRow("Search / filter",search); form.addRow("Show devices",visibility); form.addRow("Names file",file_row); file_label=form.labelForField(file_row); form.addRow("Manual selection",manual); manual_label=form.labelForField(manual); form.addRow("Excluded positions",exclusion); form.addRow("",controls); form.addRow("Selection helpers",subset); form.addRow("Selected / available",count); form.addRow(table); form.addRow("",empty)
            data={"method":method,"manual":manual,"file":file_edit,"file_row":file_row,"file_label":file_label,"manual_label":manual_label,"exclusion":exclusion,"search":search,"visibility":visibility,"table":table,"devices":devices,"count":count,"empty":empty,"n":nbox,"select_only_highlighted":select_only_highlighted,"add_highlighted":add_highlighted}
            self.corrector_selection_widgets[key]=data
            method.currentIndexChanged.connect(lambda _=0,k=key:self._corrector_method_changed(k)); preview.clicked.connect(lambda _=False,k=key:self._refresh_corrector_preview(k)); search.textChanged.connect(lambda _="",k=key:self._filter_corrector_table(k)); visibility.currentIndexChanged.connect(lambda _=0,k=key:self._filter_corrector_table(k)); table.itemSelectionChanged.connect(lambda k=key:self._filter_corrector_table(k)); table.itemChanged.connect(lambda item,k=key:self._corrector_checkbox_changed(k,item)); browse.clicked.connect(lambda _=False,e=file_edit:self._browse_into(e)); exclusion.textChanged.connect(self.refresh_plan)
            cg.addWidget(box); self._corrector_method_changed(key); self._refresh_corrector_preview(key)
        layout.addWidget(self.orm_corrector_group)
        self.device_selection_reviewed=QCheckBox("I reviewed the ordered BPM, H-corrector and V-corrector selections")
        self.device_selection_reviewed.setToolTip("This confirms the selection workflow only; reference-model mapping remains a separate validation.")
        self.device_selection_reviewed.toggled.connect(lambda *_:self._update_workflow_tabs())
        layout.addWidget(self.device_selection_reviewed)
        self._selection_method_changed(); return self._scroll_page(content)

    @staticmethod
    def _load_repository_names(filename):
        path=Path(__file__).resolve().parents[2]/"Examples"/"PETRAIII"/"data"/filename
        return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())

    def _adapter_changed(self,*_):
        key=self.adapter_combo.currentData(); petra=key=="petra"; pysc=key=="pysc"
        self.pyaml_group.setVisible(key == "pyaml")
        self.reference_model=None
        if hasattr(self,"reference_model_path"):
            self.reference_model_path.clear(); self.reference_model_variable.setText("Not applicable"); self.reference_model_element_count.setText("Not available"); self.change_reference_variable_button.setEnabled(False)
            self.reference_model_summary.setText("Not available — select an authoritative reference lattice." if petra else "No reference model loaded.")
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

    def _pyaml_configuration_changed(self,*_):
        if hasattr(self,"pyaml_simulator_identity_status"):
            self.pyaml_simulator_identity_status.setText("Configuration changed — Test connection to identify the active runtime and restoration evidence.")
        self._invalidate_pyaml()

    def _set_pyaml_simulator_identity(self,adapter):
        identity=getattr(adapter,"simulator_identity",None)
        verified=bool(isinstance(identity,dict) and identity.get("verified") is True)
        if verified:
            evidence="Adapter value matched via fresh strength.get() — independent readback unavailable." if getattr(adapter,"allow_get_only_restoration",False) else "Independent readback capability will be used where available."
            self.pyaml_simulator_identity_status.setText(f"Verified {identity['backend']} simulator · instance {identity['instance_id']} · {identity['evidence']}. Restoration evidence: {evidence}")
        elif getattr(adapter,"allow_get_only_restoration",False):
            self.pyaml_simulator_identity_status.setText("pyAML strength.get() is available for adapter-value restoration checks. Independent readback is unavailable. This weaker evidence is intended for the approved ESRF simulator workflow and does not authorize real-machine operation.")
        else:
            self.pyaml_simulator_identity_status.setText("No authoritative runtime simulator identity/capability was provided by pyAML. Real-machine or unknown-target ORM remains blocked without an independent readback.")

    def _invalidate_pyaml(self, *_, preserve_mapping=False):
        if self.adapter_combo.currentData() != "pyaml":
            return
        if not preserve_mapping:
            self._invalidate_device_alignment("pyAML configuration changed — verify device alignment again.")
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
        previous_loading=self._loading_selection_state; self._loading_selection_state=True
        try:
            if hasattr(self, "corrector_selection_widgets"):
                for kind in ("hcor", "vcor"):
                    self.corrector_selection_widgets[kind]["devices"] = ()
                    self._refresh_corrector_preview(kind, read_diagnostics=False)
            self.refresh_preview()
        finally:self._loading_selection_state=previous_loading
        self._reset_selection_states_for_inventory()
        self._update_machine_identity()
        self._sync_rf_presentation()

    def _connect_pyaml(self):
        self._invalidate_pyaml()
        options = {key: editor.text().strip() for key, editor in self.pyaml_fields.items()}
        options.update(allow_writes=self.pyaml_magnet_writes.isChecked(),
                       allow_rf_writes=self.pyaml_rf_writes.isChecked(),
                       corrector_values_in_radians=True,
                       orbit_correction_enabled=self.pyaml_orbit_correction.isChecked())
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
        self._set_pyaml_simulator_identity(self.adapter)
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
        if getattr(self,"connection_thread",None) is not None:
            self.statusBar().showMessage("A connection test is already running or cleaning up."); return
        key=self.adapter_combo.currentData()
        if key=="mock":self.statusBar().showMessage("Mock ready — no external connection"); return
        options=None
        if key=="pyaml":
            candidate=self.measurement_device_alignment if self._pending_full_bundle is not None else None
            self._invalidate_pyaml()
            # Hold the offline-validated mapping outside active GUI state while
            # the new adapter is being constructed. It is restored only as a
            # candidate for exact connected-inventory revalidation below.
            self._pending_mapping_revalidation=candidate
            options={name:editor.text().strip() for name,editor in self.pyaml_fields.items()}; options.update(allow_writes=self.pyaml_magnet_writes.isChecked(),allow_rf_writes=self.pyaml_rf_writes.isChecked(),corrector_values_in_radians=True,orbit_correction_enabled=self.pyaml_orbit_correction.isChecked())
            if options["mode"]=="live" and (options["allow_writes"] or options["allow_rf_writes"]):
                answer=QMessageBox.question(self,"Enable live pyAML writes?","Measurements may change configured corrector magnets and/or RF frequency. Enable selected writes?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
                if answer!=QMessageBox.Yes:return
            def operation():
                session=InterfaceRegistry(pyaml_options=options).create("pyaml"); result=session.adapter.test_connection()
                return {"key":key,"session":session,"result":result,"bpms":tuple(BpmDevice(**item) for item in session.adapter.list_devices("bpm")),"hcor":tuple(CorrectorDevice(**item) for item in session.adapter.list_devices("hcor")),"vcor":tuple(CorrectorDevice(**item) for item in session.adapter.list_devices("vcor")),"options":options}
        else:
            adapter=self.adapter
            def operation():return {"key":key,"result":adapter.test_connection()}
        self._connection_test_timed_out=False; self.test_connection_button.setEnabled(False); self.connection_progress.setVisible(True); self.machine_info["connection"].setText("Testing connection…"); self._set_connection_state(False,"TESTING")
        self.connection_thread=QThread(self); self.connection_worker=ConnectionTestWorker(operation); self.connection_worker.moveToThread(self.connection_thread); self.connection_thread.started.connect(self.connection_worker.run); self.connection_worker.completed.connect(self._connection_test_completed); self.connection_worker.failed.connect(self._connection_test_failed)
        self.connection_worker.completed.connect(self._finish_connection_test); self.connection_worker.failed.connect(self._finish_connection_test)
        QTimer.singleShot(self._connection_timeout_ms,self._connection_test_timeout); self.connection_thread.start()

    def _connection_test_completed(self,payload):
        if getattr(self,"_connection_test_timed_out",False):return
        key=payload["key"]; result=payload["result"]
        if key=="pyaml":
            self.adapter=payload["session"].adapter; self.devices=payload["bpms"]; self.horizontal_correctors=payload["hcor"]; self.vertical_correctors=payload["vcor"]
            self._set_pyaml_simulator_identity(self.adapter)
            self.selected_devices=self.devices; self.selected_hcorrectors=self.horizontal_correctors; self.selected_vcorrectors=self.vertical_correctors
            for kind,devices in (("hcor",self.horizontal_correctors),("vcor",self.vertical_correctors)):self.corrector_selection_widgets[kind]["devices"]=devices
            self.project.metadata["pyaml_options"]={name:value for name,value in payload["options"].items() if name not in ("allow_writes","allow_rf_writes")}
            self.machine_info["connection"].setText("Connected — read only"); self.machine_info["bpm_orbit"].setText(f"{result['bpms']} BPMs")
            for name in ("corrector_readback","calibration","rf_readback"):self.machine_info[name].setText(str(result[name]))
            try:
                rf_readback=float(result["rf_readback"])
                self.nominal_rf.setText(f"{rf_readback:.12f}" if np.isfinite(rf_readback) else "")
            except (TypeError,ValueError):
                self.nominal_rf.clear()
            if self.adapter_combo.currentData()=="pyaml" and self.reference_model is None:
                provider=getattr(self.adapter,"theoretical_lattice_provenance",None)
                provenance=provider() if callable(provider) else None
                if provenance:
                    self._load_reference_model(provenance["path"],provenance["source"],provenance.get("mat_variable") or None)
                else:
                    self.reference_model_summary.setText("pyAML exposes no resolvable file-backed theoretical lattice through its public design Simulator.lattice metadata. Select the authoritative lattice manually; measurement remains available.")
            if self._pending_full_bundle is None:
                self._reset_selection_states_for_inventory()
            else:
                self.measurement_device_alignment=self._pending_mapping_revalidation
                try:self._restore_full_bundle_inventory(self._pending_full_bundle)
                except Exception as exc:
                    self._reset_selection_states_for_inventory(); self._invalidate_device_alignment(f"Saved device selection is incompatible: {exc}")
                finally:self._pending_mapping_revalidation=None
            self.refresh_preview()
            writable=AdapterCapability.WRITE in self.adapter.capabilities; rf_writable=AdapterCapability.RF_WRITE in self.adapter.capabilities
            if rf_writable:
                self.rf_control_mode.setCurrentIndex(self.rf_control_mode.findData("automatic"))
            access=f"Correctors: {'WRITE' if writable else 'read only'}; RF: {'WRITE' if rf_writable else 'read only'}"
            self.machine_info["access"].setText(access)
            self.subtitle.setText(f"pyAML {payload['options']['mode']} connected — {access}")
            self.status_badge.setText("pyAML • WRITES ENABLED" if writable or rf_writable else "pyAML • READ ONLY")
            self.measurement_type.model().item(self.measurement_type.findData("orm")).setEnabled(writable); self.orm_unavailable_label.setVisible(not writable)
            self._sync_rf_presentation()
        elif key=="pysc":
            self.machine_info["connection"].setText(f"Connected — {self.pysc_profile_combo.currentText()}"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(str(result["corrector_readback"])); self.machine_info["rf_readback"].setText(f"available — {result['rf_readback']:.6f} Hz"); self.nominal_rf.setText(f"{result['rf_readback']:.12f}"); self._load_profile_reference_model()
        else:
            self.machine_info["connection"].setText("Connected — read only"); self.machine_info["bpm_orbit"].setText(f"available — {result['bpms']} BPMs"); self.machine_info["corrector_readback"].setText(result["corrector_readback"]); self.machine_info["calibration"].setText(result["calibration"]); self.machine_info["rf_readback"].setText(result["rf_readback"])
        self._set_connection_state(True,"CONNECTED"); self.statusBar().showMessage("Connection test succeeded")
        self._verify_unresolved_orm_restoration()
        diagnostics=self._rf_restoration_diagnostics or {}
        if self._rf_restoration_unresolved and "rf_readback" in result and diagnostics:
            try:resolved=abs(float(result["rf_readback"])-float(diagnostics["original_rf_hz"]))<=float(diagnostics["tolerance_hz"])
            except Exception:resolved=False
            if resolved:
                self._rf_restoration_unresolved=False; self.statusBar().showMessage("Connection verified; original RF readback restored. RF-write block cleared.")

    def _connection_test_failed(self,exc):
        if getattr(self,"_connection_test_timed_out",False):return
        self._pending_mapping_revalidation=None
        self._set_connection_state(False,"DISCONNECTED"); self.machine_info["connection"].setText("Unavailable"); QMessageBox.warning(self,"Connection unavailable",str(exc))

    def _connection_test_timeout(self):
        if getattr(self,"connection_thread",None) is not None and self.connection_thread.isRunning():
            self._connection_test_timed_out=True
            self._pending_mapping_revalidation=None
            self._set_connection_state(False,"TIMEOUT"); self.machine_info["connection"].setText("Connection test timed out; waiting for backend cleanup…"); self.statusBar().showMessage("Connection test timed out — no stale connection was accepted")
            self.connection_progress.setVisible(False); self.test_connection_button.setEnabled(True)

    def _finish_connection_test(self,*_):
        thread=self.connection_thread; thread.quit(); thread.wait(3000); thread.deleteLater(); self.connection_thread=None; self.connection_worker=None; self.connection_progress.setVisible(False); self.test_connection_button.setEnabled(True); self.refresh_plan()

    def _measurement_page(self):
        content=QWidget(); layout=QVBoxLayout(content); layout.setContentsMargins(22,22,22,22); layout.setAlignment(Qt.AlignTop)
        self.measurement_machine_identity=QLabel(); self.measurement_machine_identity.setObjectName("planValue"); self.measurement_machine_identity.setWordWrap(True); layout.addWidget(self.measurement_machine_identity)
        # The combo remains the internal single source of truth used by the
        # acquisition code and project schema. Users navigate through the three
        # visible workspace tabs rather than a second, hidden-in-page selector.
        self.measurement_type=QComboBox(self); self.measurement_type.addItem("BPM Noise","bpm_noise"); self.measurement_type.addItem("RF Dispersion","dispersion"); self.measurement_type.addItem("ORM","orm"); self.measurement_type.currentIndexChanged.connect(self._measurement_type_changed); self.measurement_type.setVisible(False)
        self.orm_unavailable_label=QLabel("ORM needs corrector-write authorization. This measurement tab is unavailable for a read-only backend."); self.orm_unavailable_label.setWordWrap(True); self.orm_unavailable_label.setVisible(False); layout.addWidget(self.orm_unavailable_label)
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
        self.rf_restoration_tolerance=NoWheelDoubleSpinBox(); self.rf_restoration_tolerance.setRange(0,1000); self.rf_restoration_tolerance.setDecimals(6); self.rf_restoration_tolerance.setValue(.1); self.rf_restoration_tolerance.setSuffix(" Hz")
        self.rf_restoration_tolerance.setToolTip("Maximum absolute difference between the original independent RF readback and the final restored readback. This does not weaken missing-readback checks.")
        self.verify_restored_orbit=QCheckBox("Verify restored orbit"); self.verify_restored_orbit.setChecked(True)
        self.verify_restored_orbit.setToolTip("After exact RF restoration, acquire a read-only reference orbit for comparison. This orbit is never used in the dispersion calculation.")
        self.rf_safety=QLabel("READ ONLY — pyLOCO Measure will guide RF changes but will never write RF."); self.rf_safety.setWordWrap(True)
        self.rf_states_explanation=QLabel("Applied states: f₀ + 100 Hz / f₀ − 100 Hz\nTotal separation: Δ = 200 Hz"); self.rf_states_explanation.setWordWrap(True)
        for label,widget in (("RF control mode",self.rf_control_mode),("Nominal RF frequency",nominal_widget),("Actual applied RF",self.manual_actual_rf),("Total RF separation Δ",self.rf_step),("Applied RF states",self.rf_states_explanation),("Measurement direction",self.dispersion_direction),("Settling delay after RF change",self.settling_delay),("RF restoration tolerance",self.rf_restoration_tolerance),("Restoration check",self.verify_restored_orbit),("",self.rf_safety)): dispersion_form.addRow(label,widget)
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
        self.orm_state_explanation=QLabel(
            "Applied states: K₀ + ΔK/2 and K₀ − ΔK/2\n"
            "Effective readback separation: K⁺ − K⁻ = +ΔK"
        ); self.orm_state_explanation.setWordWrap(True)
        self.orm_settling_delay=NoWheelDoubleSpinBox(); self.orm_settling_delay.setRange(0,3600); self.orm_settling_delay.setDecimals(3); self.orm_settling_delay.setValue(1.0); self.orm_settling_delay.setSuffix(" s")
        self.orm_settling_delay.setToolTip("Wait after every corrector transition, including restoration, before checking readback or acquiring BPM data.")
        orm_form.addRow("ORM output format",self.orm_output_format); orm_form.addRow("Optional analysis",self.orm_normalized_analysis); orm_form.addRow("Measurement direction",self.orm_direction); orm_form.addRow("Corrector kick mode",self.orm_kick_mode); orm_form.addRow("Settling delay after corrector change",self.orm_settling_delay); orm_form.addRow("Total horizontal bipolar kick Δ",self.orm_hkick); self.orm_hkick_label=orm_form.labelForField(self.orm_hkick); orm_form.addRow("Total vertical bipolar kick Δ",self.orm_vkick); self.orm_vkick_label=orm_form.labelForField(self.orm_vkick); orm_form.addRow("Applied corrector states",self.orm_state_explanation); orm_form.addRow("Kick arrays file",self.orm_kick_file_row); self.orm_kick_file_label=orm_form.labelForField(self.orm_kick_file_row)
        self.kick_preview=QTableWidget(0,5); self.kick_preview.setMinimumHeight(180); self.kick_preview.setHorizontalHeaderLabels(["Position","Plane","Corrector","Effective kick [µrad]","Source"]); self.kick_preview.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch); orm_form.addRow("Effective kicks",self.kick_preview)
        for widget in (self.orm_direction,self.orm_kick_mode,self.orm_settling_delay,self.orm_hkick,self.orm_vkick,self.orm_kick_file,self.orm_normalized_analysis):
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
        # Native macOS tab-scroll arrows can overlap the first and last tabs,
        # making scientific views inaccessible. Fit every tab into the bar and
        # retain each complete title in its tooltip.
        result_tab_bar.setExpanding(True)
        result_tab_bar.setUsesScrollButtons(False)
        result_tab_bar.setElideMode(Qt.ElideRight)
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
        if hasattr(self,"preview_table") and not self._loading_selection_state:self.refresh_preview()

    def _filter_bpm_table(self, text=""):
        if not hasattr(self,"preview_table"):return
        query=self.bpm_search.text(); mode=self.bpm_visibility.currentData() if hasattr(self,"bpm_visibility") else "all"; selected={device.name for device in self.selected_devices}; visible=0
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}.get(self.selection_method.currentData(),"selection")
        for row in range(self.preview_table.rowCount()):
            name=self.preview_table.item(row,1).text() if self.preview_table.item(row,1) else ""; identifier=self.preview_table.item(row,2).text() if self.preview_table.item(row,2) else ""; selected_row=name in selected
            hidden=not _device_matches_filter(name,identifier,query) or mode=="selected" and not selected_row or mode=="unselected" and selected_row
            self.preview_table.setRowHidden(row,hidden); visible+=not hidden
        if hasattr(self,"selection_message"):
            highlighted=len(self.preview_table.selectionModel().selectedRows()) if self.preview_table.selectionModel() is not None else 0
            if hasattr(self,"bpm_select_only_highlighted_button"):
                self.bpm_select_only_highlighted_button.setEnabled(highlighted>0); self.bpm_add_highlighted_button.setEnabled(highlighted>0)
            prefix=f"Highlighted: {highlighted} • {len(selected)} / {len(self.devices)} BPMs selected for measurement. "
            error=getattr(self,"_bpm_selection_error","")
            if error:self.selection_message.setText(prefix+f"Selection is not valid: {error}")
            elif mode=="selected" and not selected:self.selection_message.setText(prefix+"No BPMs are selected. Choose All devices to highlight devices, or use Clear to return to All devices automatically.")
            elif visible==0:self.selection_message.setText(prefix+"No BPM devices match the current search and visibility filters.")
            else:self.selection_message.setText(prefix+f"Shown by filter: {visible}. Highlighted rows do not change the checked selection. Source: {source}.")

    def _select_all_bpms(self):
        self.selection_method.setCurrentIndex(self.selection_method.findData("all")); self.refresh_preview()

    def _bpm_checkbox_changed(self,item):
        if item is None or item.column()!=1:return
        highlighted=self._highlighted_device_names(self.preview_table)
        checked={self.preview_table.item(row,1).text() for row in range(self.preview_table.rowCount())
                 if self.preview_table.item(row,1) is not None and self.preview_table.item(row,1).checkState()==Qt.Checked}
        names=[device.name for device in self.devices if device.name in checked]
        self.selection_method.blockSignals(True); self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.selection_method.blockSignals(False)
        self.manual_input.blockSignals(True); self.manual_input.setText(", ".join(names)); self.manual_input.blockSignals(False)
        self.bpm_exclusions.blockSignals(True); self.bpm_exclusions.clear(); self.bpm_exclusions.blockSignals(False)
        self._selection_method_changed()
        self._restore_table_highlights(self.preview_table,highlighted); self._filter_bpm_table()

    @staticmethod
    def _highlighted_device_names(table):
        return tuple(table.item(index.row(),1).text() for index in table.selectionModel().selectedRows() if table.item(index.row(),1))

    @staticmethod
    def _restore_table_highlights(table,names):
        wanted=set(names); table.clearSelection()
        for row in range(table.rowCount()):
            item=table.item(row,1)
            if item is not None and item.text() in wanted:
                table.selectionModel().select(table.model().index(row,0),QItemSelectionModel.Select|QItemSelectionModel.Rows)

    def _clear_bpms(self):
        highlighted=self._highlighted_device_names(self.preview_table)
        if self.bpm_visibility.currentData()=="selected":self.bpm_visibility.setCurrentIndex(self.bpm_visibility.findData("all"))
        self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.clear(); self.refresh_preview()
        self._restore_table_highlights(self.preview_table,highlighted); self._filter_bpm_table()

    def _select_filtered_bpms(self):
        query=self.bpm_search.text()
        matches=[device.name for device in self.devices if _device_matches_filter(device.name,device.identifier,query)]
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
        search=self.bpm_search if key=="bpm" else self.corrector_selection_widgets[key]["search"]
        query=search.text()
        candidates=tuple(device for device in devices if _device_matches_filter(device.name,device.identifier,query))
        selected=tuple(candidates[:min(count,len(candidates))]) if mode=="first" else self._uniform_devices(candidates,count)
        method.setCurrentIndex(method.findData("manual")); manual.setText(", ".join(device.name for device in selected))
        self.refresh_preview() if key=="bpm" else self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_all_correctors(self,key):
        data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(data["method"].findData("all")); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _corrector_checkbox_changed(self,key,item):
        if item is None or item.column()!=1:return
        data=self.corrector_selection_widgets[key]; table=data["table"]; highlighted=self._highlighted_device_names(table)
        checked={table.item(row,1).text() for row in range(table.rowCount())
                 if table.item(row,1) is not None and table.item(row,1).checkState()==Qt.Checked}
        names=[device.name for device in data["devices"] if device.name in checked]
        data["method"].blockSignals(True); data["method"].setCurrentIndex(data["method"].findData("manual")); data["method"].blockSignals(False)
        data["manual"].blockSignals(True); data["manual"].setText(", ".join(names)); data["manual"].blockSignals(False)
        data["exclusion"].blockSignals(True); data["exclusion"].clear(); data["exclusion"].blockSignals(False)
        self._corrector_method_changed(key); self._refresh_corrector_preview(key,read_diagnostics=False)
        self._restore_table_highlights(table,highlighted); self._filter_corrector_table(key)

    def _clear_correctors(self,key):
        data=self.corrector_selection_widgets[key]; highlighted=self._highlighted_device_names(data["table"])
        if data["visibility"].currentData()=="selected":data["visibility"].setCurrentIndex(data["visibility"].findData("all"))
        data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].clear(); self._refresh_corrector_preview(key,read_diagnostics=False)
        self._restore_table_highlights(data["table"],highlighted); self._filter_corrector_table(key)

    def _select_filtered_correctors(self,key):
        data=self.corrector_selection_widgets[key]; query=data["search"].text(); matches=[device.name for device in data["devices"] if _device_matches_filter(device.name,device.identifier,query)]
        data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(matches)); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_highlighted(self,key):
        table=self.preview_table if key=="bpm" else self.corrector_selection_widgets[key]["table"]
        highlighted={table.item(index.row(),1).text() for index in table.selectionModel().selectedRows() if table.item(index.row(),1)}
        if not highlighted:return
        if key=="bpm":
            current={device.name for device in self.selected_devices}; names=[device.name for device in self.devices if device.name in current|highlighted]
            self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.setText(", ".join(names)); self.refresh_preview()
        else:
            data=self.corrector_selection_widgets[key]; current={device.name for device in (self.selected_hcorrectors if key=="hcor" else self.selected_vcorrectors)}; names=[device.name for device in data["devices"] if device.name in current|highlighted]
            data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(names)); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_only_highlighted(self,key):
        table=self.preview_table if key=="bpm" else self.corrector_selection_widgets[key]["table"]
        highlighted={table.item(index.row(),1).text() for index in table.selectionModel().selectedRows() if table.item(index.row(),1)}
        if not highlighted:
            self.statusBar().showMessage("Highlight at least one device before replacing the measurement selection."); return
        if key=="bpm":
            names=[device.name for device in self.devices if device.name in highlighted]
            self.selection_method.setCurrentIndex(self.selection_method.findData("manual")); self.manual_input.setText(", ".join(names)); self.refresh_preview()
        else:
            data=self.corrector_selection_widgets[key]; names=[device.name for device in data["devices"] if device.name in highlighted]
            data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(names)); self._refresh_corrector_preview(key,read_diagnostics=False)

    def _select_demo_one_each(self):
        self._select_device_subset("hcor","uniform",1); self._select_device_subset("vcor","uniform",1)

    def _select_demo_small_orm(self):
        self._select_device_subset("bpm","uniform",20); self._select_device_subset("hcor","uniform",5); self._select_device_subset("vcor","uniform",5)

    def _h_to_v_proposal_from_document(self,document):
        """Resolve H B1L to V A1L only through a shared exact-identity mapping."""
        rows=tuple(document.get("mappings",()))
        horizontal={row.get("measured_name"):row for row in rows if row.get("role")=="horizontal_corrector"}
        vertical_by_ordinal={}
        for row in rows:
            if row.get("role")=="vertical_corrector" and row.get("component")=="A1L" and type(row.get("lattice_ordinal")) is int:
                vertical_by_ordinal.setdefault(row["lattice_ordinal"],[]).append(row)
        preview=[]; matched=set()
        for device in self.selected_hcorrectors:
            source=horizontal.get(device.name)
            if not source or source.get("status")!="mapped" or source.get("component")!="B1L":
                preview.append({"horizontal":device.name,"vertical":"","status":"Missing","reason":(source or {}).get("reason","No exact B1L mapping")}); continue
            candidates=vertical_by_ordinal.get(source.get("lattice_ordinal"),[])
            valid=[row for row in candidates if row.get("status")=="mapped"]
            if len(candidates)>1 or len(valid)>1:
                preview.append({"horizontal":device.name,"vertical":"","status":"Ambiguous","reason":"More than one A1L device maps to the same AT element"}); continue
            if len(valid)!=1:
                reason=candidates[0].get("reason","No validated A1L counterpart") if candidates else "No A1L device maps to the same AT element"
                preview.append({"horizontal":device.name,"vertical":"","status":"Missing","reason":reason}); continue
            target=valid[0]; matched.add(target["measured_name"])
            preview.append({"horizontal":device.name,"vertical":target["measured_name"],"status":"Matched","reason":f"Exact Device identity; AT index {source['lattice_ordinal']} (0-based; B1L → A1L)"})
        ordered=tuple(device for device in self.vertical_correctors if device.name in matched)
        return {"rows":preview,"devices":ordered}

    def _authoritative_h_to_v_pairs(self):
        """Return optional exact pairs published by the active backend.

        ``corrector_pairings()`` is an optional adapter capability.  Its data
        come from the connected backend, never from editable GUI labels.  A
        mapping or an iterable of ``(horizontal, vertical)`` pairs is accepted;
        repeated horizontal identities remain visible as ambiguity.
        """
        provider=getattr(self.adapter,"corrector_pairings",None)
        if not callable(provider):return {}
        supplied=provider(); pairs={}
        entries=supplied.items() if isinstance(supplied,dict) else supplied
        for entry in entries:
            if isinstance(entry,dict):horizontal=entry.get("horizontal"); vertical=entry.get("vertical")
            else:
                try:horizontal,vertical=entry
                except (TypeError,ValueError) as exc:raise ValueError("Backend corrector pairing metadata must contain exact H/V identity pairs.") from exc
            if not isinstance(horizontal,str) or not isinstance(vertical,str) or not horizontal or not vertical:
                raise ValueError("Backend corrector pairing metadata contains an empty or non-text device identity.")
            pairs.setdefault(horizontal,[]).append(vertical)
        return pairs

    def _h_to_v_proposal_from_inventory(self):
        """Pair selected H controls to the connected V inventory without AT.

        Authoritative backend pairs take precedence.  Otherwise only the ESRF
        terminal-component convention ``shared-B1L`` → ``shared-A1L`` is
        accepted.  This is device-selection evidence, not validated AT mapping.
        """
        authoritative=self._authoritative_h_to_v_pairs()
        vertical_by_name={}
        for device in self.vertical_correctors:vertical_by_name.setdefault(device.name,[]).append(device)
        preview=[]; matched=set()
        for device in self.selected_hcorrectors:
            source="backend_pairing" if device.name in authoritative else "esrf_terminal_component"
            targets=list(authoritative.get(device.name,()))
            if not targets:
                match=re.fullmatch(r"(.+)-B1L",device.name)
                if match:targets=[match.group(1)+"-A1L"]
                else:
                    preview.append({"horizontal":device.name,"vertical":"","status":"Unmatched","pairing":"unrecognized",
                                    "reason":"No authoritative backend pair and the H identity does not end with the recognized -B1L terminal component. Select V devices manually or load an AT lattice for optional lattice-based pairing."}); continue
            if len(set(targets))!=1 or len(targets)!=1:
                preview.append({"horizontal":device.name,"vertical":"","status":"Ambiguous","pairing":source,
                                "reason":"Authoritative backend metadata provides more than one V identity for this H corrector."}); continue
            target_name=targets[0]; candidates=vertical_by_name.get(target_name,())
            if len(candidates)>1:
                preview.append({"horizontal":device.name,"vertical":"","status":"Ambiguous","pairing":source,
                                "reason":f"Exact V identity {target_name!r} occurs {len(candidates)} times in the connected V inventory."}); continue
            if len(candidates)!=1:
                preview.append({"horizontal":device.name,"vertical":"","status":"Unmatched","pairing":source,
                                "reason":f"Exact proposed V identity {target_name!r} is not present in the connected V inventory."}); continue
            matched.add(target_name)
            evidence=("Authoritative backend H/V pairing metadata" if source=="backend_pairing" else
                      "Exact shared ESRF device identity with terminal component -B1L → -A1L")
            preview.append({"horizontal":device.name,"vertical":target_name,"status":"Matched","pairing":source,
                            "reason":evidence+"; selection pairing only, not validated AT mapping."})
        ordered=tuple(device for device in self.vertical_correctors if device.name in matched)
        return {"rows":preview,"devices":ordered}

    def _h_to_v_proposal(self):
        proposal=self._h_to_v_proposal_from_inventory()
        # Preserve the former AT path as an optional fallback, but only for an
        # unrecognized naming convention.  Recognized missing/ambiguous pairs
        # must not be silently substituted.
        unresolved={row["horizontal"] for row in proposal["rows"] if row.get("pairing")=="unrecognized"}
        if not unresolved or self.reference_model is None:return proposal
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        inventory={"bpm":[],"horizontal_corrector":[device.name for device in self.selected_hcorrectors if device.name in unresolved],"vertical_corrector":[device.name for device in self.vertical_correctors]}
        try:
            document=find_exact_device_mappings(self.reference_model.ring,inventory,lattice_path=self.reference_model.path,
                                                configuration_path=configuration if configuration.is_file() else None,
                                                provenance="Measure optional AT H-to-V counterpart preview")
            lattice=self._h_to_v_proposal_from_document(document)
        except Exception as exc:
            for row in proposal["rows"]:
                if row.get("pairing")=="unrecognized":row["reason"]+=f" AT fallback was unavailable: {exc}"
            return proposal
        lattice_rows={row["horizontal"]:row for row in lattice["rows"]}
        for index,row in enumerate(proposal["rows"]):
            fallback=lattice_rows.get(row["horizontal"])
            if row.get("pairing")=="unrecognized" and fallback and fallback["status"]=="Matched":
                proposal["rows"][index]={**fallback,"pairing":"at_lattice",
                                         "reason":fallback["reason"]+"; optional AT-lattice pairing used."}
        matched={row["vertical"] for row in proposal["rows"] if row["status"]=="Matched"}
        proposal["devices"]=tuple(device for device in self.vertical_correctors if device.name in matched)
        return proposal

    def _show_h_to_v_preview(self,proposal):
        dialog=QDialog(self); dialog.setWindowTitle("Copy horizontal corrector selection to vertical correctors"); dialog.resize(900,520)
        layout=QVBoxLayout(dialog); summary=QLabel(f"{len(proposal['devices'])} unique V counterparts found for {len(self.selected_hcorrectors)} selected H correctors. Pairing uses authoritative backend metadata, exact ESRF terminal components, or an optional AT fallback; it is not validated AT mapping. No positional or name-similarity fallback is used."); summary.setWordWrap(True); layout.addWidget(summary)
        table=QTableWidget(len(proposal["rows"]),4); table.setHorizontalHeaderLabels(("Selected H corrector (B1L)","Proposed V corrector (A1L)","Status","Evidence / problem")); table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row,item in enumerate(proposal["rows"]):
            for column,key in enumerate(("horizontal","vertical","status","reason")):table.setItem(row,column,QTableWidgetItem(str(item[key]) or "—"))
        table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch); table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch); table.horizontalHeader().setSectionResizeMode(3,QHeaderView.Stretch); layout.addWidget(table)
        if self.corrector_selection_widgets["vcor"]["exclusion"].text().strip():
            warning=QLabel("Vertical positional exclusions will be cleared because the ordered V inventory is changing; review exclusions after applying."); warning.setObjectName("warningText"); warning.setWordWrap(True); layout.addWidget(warning)
        actions=QDialogButtonBox(); replace=actions.addButton("Replace vertical selection",QDialogButtonBox.AcceptRole); add=actions.addButton("Add to vertical selection",QDialogButtonBox.ActionRole); cancel=actions.addButton(QDialogButtonBox.Cancel); layout.addWidget(actions)
        choice={"mode":None}; replace.clicked.connect(lambda:(choice.update(mode="replace"),dialog.accept())); add.clicked.connect(lambda:(choice.update(mode="add"),dialog.accept())); cancel.clicked.connect(dialog.reject)
        return choice["mode"] if dialog.exec()==QDialog.Accepted else None

    def _copy_h_selection_to_v(self):
        try:proposal=self._h_to_v_proposal()
        except Exception as exc:QMessageBox.warning(self,"Cannot copy H selection to V",str(exc)); return
        mode=self._show_h_to_v_preview(proposal)
        if mode not in {"replace","add"}:return
        proposed={device.name for device in proposal["devices"]}; current={device.name for device in self.selected_vcorrectors} if mode=="add" else set()
        ordered=[device.name for device in self.vertical_correctors if device.name in proposed|current]
        data=self.corrector_selection_widgets["vcor"]; data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText(", ".join(ordered)); data["exclusion"].clear(); self._refresh_corrector_preview("vcor",read_diagnostics=False)
        unmatched=sum(item["status"]!="Matched" for item in proposal["rows"]); self.statusBar().showMessage(f"Applied {len(proposal['devices'])} unique V counterparts; {unmatched} H correctors had no unique counterpart. Device selection pairing does not validate AT alignment; the V selection is now independent.")

    def _select_first_corrector(self,key):
        data=self.corrector_selection_widgets[key]
        data["method"].setCurrentIndex(data["method"].findData("manual")); data["manual"].setText("0"); self._refresh_corrector_preview(key)

    def _filter_corrector_table(self,key):
        data=self.corrector_selection_widgets[key]; query=data["search"].text(); mode=data["visibility"].currentData(); table=data["table"]; selected={device.name for device in (self.selected_hcorrectors if key=="hcor" else self.selected_vcorrectors)}; visible=0
        for row in range(table.rowCount()):
            name=table.item(row,1).text() if table.item(row,1) else ""; identifier=table.item(row,2).text() if table.item(row,2) else ""; selected_row=name in selected
            hidden=not _device_matches_filter(name,identifier,query) or mode=="selected" and not selected_row or mode=="unselected" and selected_row
            table.setRowHidden(row,hidden); visible+=not hidden
        highlighted=len(table.selectionModel().selectedRows()) if table.selectionModel() is not None else 0
        data["select_only_highlighted"].setEnabled(highlighted>0); data["add_highlighted"].setEnabled(highlighted>0)
        data["count"].setText(f"{len(selected)} / {len(data['devices'])} selected for measurement • {highlighted} highlighted • {visible} shown by filter")
        if mode=="selected" and not selected:data["empty"].setText("No correctors are selected. Choose All devices to highlight devices, or use Clear to return to All devices automatically."); data["empty"].setVisible(True)
        elif visible==0:data["empty"].setText("No devices match the current search and visibility filters."); data["empty"].setVisible(True)
        else:data["empty"].setVisible(False)

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
            selected=[]; invalid=[]; selected_names=set(); used_pattern=False
            for token in tokens:
                if token in lookup:
                    if token not in selected_names:selected.append(lookup[token]); selected_names.add(token)
                    continue
                if any(character in token for character in "*?["):
                    used_pattern=True
                    matches=[device for device in devices if fnmatchcase(device.name,token)]
                    if not matches:invalid.append(f"unmatched pattern {token!r}"); continue
                    for device in matches:
                        if device.name not in selected_names:selected.append(device); selected_names.add(device.name)
                    continue
                try:index=int(token)
                except ValueError:invalid.append(token); continue
                if 0<=index<len(devices):
                    device=devices[index]
                    if device.name not in selected_names:selected.append(device); selected_names.add(device.name)
                else:invalid.append(token)
            if invalid: raise ValueError("Unknown corrector names/positions: "+", ".join(invalid))
            if used_pattern:selected=[device for device in devices if device.name in selected_names]
        excluded=self._positions(data["exclusion"].text()); invalid=sorted(pos for pos in excluded if pos<0 or pos>=len(selected))
        if invalid: raise ValueError("Excluded positions outside selection: "+", ".join(map(str,invalid)))
        return tuple(device for pos,device in enumerate(selected) if pos not in excluded)

    def _corrector_method_changed(self,key):
        data=self.corrector_selection_widgets[key]; method=data["method"].currentData(); data["file_row"].setVisible(method=="names_file"); data["file_label"].setVisible(method=="names_file"); data["manual"].setVisible(method=="manual"); data["manual_label"].setVisible(method=="manual")

    @staticmethod
    def _corrector_diagnostic_cell(entry, field):
        if entry is None:
            return "Not read yet",f"{field} has not been queried. Select the device and use Refresh after connecting."
        if not isinstance(entry,dict):
            try:return f"{float(entry):.8g}",f"{field} was read from the adapter; no authoritative unit metadata were returned."
            except Exception:return "Read failed",f"{field} returned an unsupported value: {entry!r}"
        status=entry.get("status")
        if status=="unsupported":return "Not supported",f"{field} is not exposed independently by this adapter. {entry.get('error','')}".strip()
        if status=="failed":return "Read failed",f"{field} read failed: {entry.get('error','unknown error')}"
        if status!="available":return "Not read yet",f"{field} has not been queried."
        value=entry.get("value"); unit=str(entry.get("unit") or "").strip()
        try:text=f"{float(value):.8g}"+(f" {unit}" if unit else "")
        except Exception:return "Read failed",f"{field} returned an invalid value: {value!r}"
        source="authoritative component metadata" if unit else "adapter read (unit metadata unavailable)"
        return text,f"{field}: {value!r}{(' '+unit) if unit else ''}; source: {source}."

    def _refresh_corrector_preview(self,key,read_diagnostics=True):
        self._model_orm_cache=None
        data=self.corrector_selection_widgets[key]; devices=tuple(data["devices"])
        try:selected=self._resolve_correctors(key)
        except Exception as exc:data["table"].setRowCount(0); data["table"].setToolTip(str(exc)); return
        previous=self.selected_hcorrectors if key=="hcor" else self.selected_vcorrectors
        if (not self._loading_selection_state and
                tuple(device.name for device in previous)!=tuple(device.name for device in selected)):
            self._invalidate_device_alignment("Corrector inventory changed — verify device alignment again.")
            if hasattr(self,"device_selection_reviewed"):self.device_selection_reviewed.setChecked(False)
        if key=="hcor":self.selected_hcorrectors=selected
        else:self.selected_vcorrectors=selected
        table=data["table"]; table.blockSignals(True); table.setRowCount(len(devices)); table.setToolTip(f"{len(selected)} selected; checkboxes define the measurement inventory, while highlighting rows does not change it")
        selected_positions={device.name:position for position,device in enumerate(selected)}
        for row,device in enumerate(devices):
            diagnostics=None
            should_read=device.name in selected_positions and read_diagnostics and (isinstance(self.adapter,PETRAReadOnlyAdapter) or self.connection_verified)
            if should_read and callable(getattr(self.adapter,"read_corrector_diagnostics",None)):
                try:diagnostics=self.adapter.read_corrector_diagnostics(device.name)
                except Exception as exc:
                    diagnostics={field:{"status":"failed","error":str(exc)} for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")}
            elif should_read:
                diagnostics={field:{"status":"unsupported","error":"The active adapter has no corrector-diagnostics interface."} for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")}
            diagnostic_cells=[self._corrector_diagnostic_cell(None if diagnostics is None else diagnostics.get(field),field) for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")]
            values=(selected_positions.get(device.name,"—"),device.name,device.identifier,device.plane,*(cell[0] for cell in diagnostic_cells))
            for col,value in enumerate(values):
                item=QTableWidgetItem(str(value)); table.setItem(row,col,item)
                if col==1:item.setFlags(item.flags()|Qt.ItemIsUserCheckable); item.setCheckState(Qt.Checked if device.name in selected_positions else Qt.Unchecked)
                if col>=4:item.setToolTip(diagnostic_cells[col-4][1])
        table.blockSignals(False)
        self._filter_corrector_table(key)
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}[data["method"].currentData()]; table.setToolTip((table.toolTip()+"\n" if table.toolTip() else "")+f"Source: {source}; {len(selected)} retained")
        # Correctors are ORM inputs even when prepared before ORM is selected.
        # Store them directly in the ORM-specific state so changing measurement
        # type cannot replace the user's prepared selection with old defaults.
        if not self._loading_selection_state:
            self._selection_states["orm"][key]={
                "method":data["method"].currentData(),
                "manual":data["manual"].text(),
                "names_file":data["file"].text(),
                "excluded_positions":data["exclusion"].text(), "search":data["search"].text(),
                "resolved_names":[device.name for device in selected],
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
        def values(method,manual,names_file,excluded,search,resolved):
            return {"method":method.currentData(),"manual":manual.text(),"names_file":names_file.text(),"excluded_positions":excluded.text(),"search":search.text(),"resolved_names":[device.name for device in resolved]}
        result={"bpm":values(self.selection_method,self.manual_input,self.names_file,self.bpm_exclusions,self.bpm_search,self.selected_devices)}
        for key in ("hcor","vcor"):
            data=self.corrector_selection_widgets[key]
            result[key]=values(data["method"],data["manual"],data["file"],data["exclusion"],data["search"],self.selected_hcorrectors if key=="hcor" else self.selected_vcorrectors)
        return result

    def _save_selection_configuration(self):
        """Save selections using the existing Measure-project schema."""
        path=QFileDialog.getSaveFileName(self,"Save device selection configuration","device-selection.pyloco-measure.json","pyLOCO Measure (*.pyloco-measure.json)")[0]
        if not path:return
        self._store_active_selection()
        project=MeasureProject(measurement_type=self._active_selection_kind,
                               measurement_selections=self._selection_states,
                               metadata={"purpose":"device_selection_configuration"})
        save_measure_project(path,project)
        self.statusBar().showMessage(f"Device selections saved in Measure project {path}")

    def _load_selection_configuration(self):
        path=QFileDialog.getOpenFileName(self,"Load device selection configuration","","pyLOCO Measure (*.pyloco-measure.json *.json)")[0]
        if not path:return
        try:
            project=load_measure_project(path); state=project.measurement_selections.get(self._active_selection_kind,{})
            inventories={"bpm":self.devices,"hcor":self.horizontal_correctors,"vcor":self.vertical_correctors}
            missing=[]; duplicates=[]
            for key,devices in inventories.items():
                names=list(state.get(key,{}).get("resolved_names",()))
                if len(names)!=len(set(names)):duplicates.extend(f"{key}: {name}" for name in names if names.count(name)>1)
                available={device.name for device in devices}; missing.extend(f"{key}: {name}" for name in names if name not in available)
            if duplicates or missing:
                details=[]
                if duplicates:details.append("Duplicate saved identities:\n"+"\n".join(sorted(set(duplicates))))
                if missing:details.append("Devices missing from the current machine inventory:\n"+"\n".join(missing))
                raise ValueError("\n\n".join(details))
        except Exception as exc:
            QMessageBox.warning(self,"Selection configuration incompatible",str(exc)); return
        answer=QMessageBox.question(self,"Replace current device selection?",f"Replace the current {self._active_selection_kind.replace('_',' ')} BPM/H/V selection with the validated selection from:\n{path}?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes:return
        self._selection_states[self._active_selection_kind]={key:dict(value) for key,value in state.items()}
        self._restore_selection(self._active_selection_kind)
        self.statusBar().showMessage("Device selection loaded and validated against the current inventory")

    def _store_active_selection(self):
        if not self._loading_selection_state and hasattr(self,"selection_method"):
            self._selection_states[self._active_selection_kind]=self._selection_snapshot()

    def _measurement_view_snapshot(self):
        """Return serializable per-measurement controls shared by the one UI."""
        return {
            "readings": self.readings.value(), "delay_seconds": self.delay.value(),
            "measurement_name": self.measurement_name.text(), "measurement_label": self.measurement_label.text(),
            "operator_comments": self.comments.toPlainText(), "bpm_noise_ddof": self.bpm_noise_convention.currentData(),
            "rf_control_mode": self.rf_control_mode.currentData(), "nominal_rf_hz": self.nominal_rf.text(),
            "rf_step_hz": self.rf_step.value(), "dispersion_direction": self.dispersion_direction.currentData(),
            "settling_delay_seconds": self.settling_delay.value(),
            "rf_restoration_tolerance_hz": self.rf_restoration_tolerance.value(),
            "verify_restored_orbit": self.verify_restored_orbit.isChecked(),
            "orm_direction": self.orm_direction.currentData(), "orm_kick_mode": self.orm_kick_mode.currentData(),
            "orm_settling_delay_seconds": self.orm_settling_delay.value(),
            "orm_horizontal_kick_urad": self.orm_hkick.value(), "orm_vertical_kick_urad": self.orm_vkick.value(),
            "orm_kick_file": self.orm_kick_file.text(), "orm_normalized_analysis": self.orm_normalized_analysis.isChecked(),
        }

    @staticmethod
    def _set_combo_data(widget, value):
        index=widget.findData(value)
        if index >= 0:widget.setCurrentIndex(index)

    def _store_measurement_view_state(self, kind):
        if hasattr(self,"readings") and kind in ("bpm_noise","dispersion","orm"):
            self._measurement_view_states[kind]=self._measurement_view_snapshot()
            expected={"bpm_noise":BpmNoiseResult,"dispersion":DispersionResult,"orm":ORMResult}[kind]
            # A shared measurement workspace changes ``self.result`` and
            # ``saved_measurement_path`` when tabs change.  Never let that
            # transient active state overwrite another role's cached result or
            # HDF5 path (for example assigning the BPM-noise file to ORM).
            if isinstance(self.result,expected):
                self._measurement_results[kind]=self.result
            path=self.saved_measurement_path
            if path is not None:
                try:
                    if validate_measurement_file(path)["kind"]==kind:
                        self._measurement_result_paths[kind]=Path(path)
                except Exception:
                    pass

    def _restore_measurement_view_state(self, kind):
        state=self._measurement_view_states.get(kind)
        if not state:return False
        self.readings.setValue(int(state.get("readings",self.readings.value())))
        self.delay.setValue(float(state.get("delay_seconds",self.delay.value())))
        self.measurement_name.setText(str(state.get("measurement_name",self.measurement_name.text())))
        self.measurement_label.setText(str(state.get("measurement_label",self.measurement_label.text())))
        self.comments.setPlainText(str(state.get("operator_comments",self.comments.toPlainText())))
        self._set_combo_data(self.bpm_noise_convention,state.get("bpm_noise_ddof"))
        self._set_combo_data(self.rf_control_mode,state.get("rf_control_mode"))
        self.nominal_rf.setText(str(state.get("nominal_rf_hz",self.nominal_rf.text())))
        self.rf_step.setValue(float(state.get("rf_step_hz",self.rf_step.value())))
        self._set_combo_data(self.dispersion_direction,state.get("dispersion_direction"))
        self.settling_delay.setValue(float(state.get("settling_delay_seconds",self.settling_delay.value())))
        self.rf_restoration_tolerance.setValue(float(state.get("rf_restoration_tolerance_hz",self.rf_restoration_tolerance.value())))
        self.verify_restored_orbit.setChecked(bool(state.get("verify_restored_orbit",self.verify_restored_orbit.isChecked())))
        self._set_combo_data(self.orm_direction,state.get("orm_direction")); self._set_combo_data(self.orm_kick_mode,state.get("orm_kick_mode"))
        self.orm_settling_delay.setValue(float(state.get("orm_settling_delay_seconds",self.orm_settling_delay.value())))
        self.orm_hkick.setValue(float(state.get("orm_horizontal_kick_urad",self.orm_hkick.value())))
        self.orm_vkick.setValue(float(state.get("orm_vertical_kick_urad",self.orm_vkick.value())))
        self.orm_kick_file.setText(str(state.get("orm_kick_file",self.orm_kick_file.text())))
        self.orm_normalized_analysis.setChecked(bool(state.get("orm_normalized_analysis",self.orm_normalized_analysis.isChecked())))
        return True

    def _attach_measurement_workspace(self,kind):
        if hasattr(self,"measurement_tab_hosts") and kind in self.measurement_tab_hosts:
            self.measurement_tab_hosts[kind][1].addWidget(self.measurement_splitter)

    def _restore_selection(self,kind):
        state=self._selection_states.get(kind) or MeasureProject().measurement_selections[kind]
        # Correctors are ORM inputs.  Keep one authoritative H/V selection
        # while switching between BPM-noise, dispersion and ORM workspaces;
        # per-tab defaults must not replace it or invalidate a saved mapping.
        orm_state=self._selection_states.get("orm") or MeasureProject().measurement_selections["orm"]
        self._loading_selection_state=True
        try:
            bpm=state["bpm"]; self.selection_method.setCurrentIndex(max(0,self.selection_method.findData(bpm["method"]))); self.manual_input.setText(bpm.get("manual","")); self.names_file.setText(bpm.get("names_file","")); self.bpm_exclusions.setText(bpm.get("excluded_positions","")); self.bpm_search.setText(bpm.get("search",""))
            for key in ("hcor","vcor"):
                settings=orm_state[key]; data=self.corrector_selection_widgets[key]; data["method"].setCurrentIndex(max(0,data["method"].findData(settings["method"]))); data["manual"].setText(settings.get("manual","")); data["file"].setText(settings.get("names_file","")); data["exclusion"].setText(settings.get("excluded_positions","")); data["search"].setText(settings.get("search","")); self._corrector_method_changed(key); self._refresh_corrector_preview(key,read_diagnostics=False)
            self._selection_method_changed(); self.refresh_preview()
        finally:self._loading_selection_state=False

    def _reset_selection_states_for_inventory(self):
        defaults=MeasureProject().measurement_selections
        self._selection_states={kind:{device:dict(settings) for device,settings in selection.items()} for kind,selection in defaults.items()}
        self._active_selection_kind=self.measurement_type.currentData() if hasattr(self,"measurement_type") else "bpm_noise"
        if hasattr(self,"selection_method"):self._restore_selection(self._active_selection_kind)

    def _reset_measurement_display(self, *, clear_session=True):
        """Reset transient acquisition/result UI without deleting saved files."""
        self.result = None
        if clear_session:
            self.saved_measurement_path = None
            self.saved_session_path = None
            self._measurement_results={"bpm_noise":None,"dispersion":None,"orm":None}
            self._measurement_result_paths={"bpm_noise":None,"dispersion":None,"orm":None}
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
        elif index==getattr(self,"review_tab_index",5):title="Review & Save"
        else:
            reverse={value:key for key,value in getattr(self,"measurement_tab_indices",{}).items()}
            requested=reverse.get(index)
            if requested is not None and self.thread is not None and requested!=self._active_acquisition_kind:
                active=self.measurement_tab_indices.get(self._active_acquisition_kind,index)
                self.tabs.blockSignals(True); self.tabs.setCurrentIndex(active); self.tabs.blockSignals(False)
                self.statusBar().showMessage("Measurement in progress — finish or cancel it before opening another measurement workspace.")
                return
            if requested is not None and hasattr(self,"measurement_type"):
                target=self.measurement_type.findData(requested)
                if target >= 0 and target!=self.measurement_type.currentIndex():self.measurement_type.setCurrentIndex(target)
                self._attach_measurement_workspace(requested)
            kind=requested or (self.measurement_type.currentData() if hasattr(self,"measurement_type") else "bpm_noise")
            title={"dispersion":"Dispersion measurement","orm":"Orbit response matrix measurement"}.get(kind,"BPM noise measurement")
        self.heading.setText(title)
        self.setWindowTitle("pyLOCO Measure" if index<2 else f"pyLOCO Measure — {title.replace(' measurement','').title()}")
        if index==getattr(self,"review_tab_index",5):self._refresh_review_page()
        if hasattr(self,"plan_values"):self.refresh_plan()

    def _refresh_review_page(self):
        if not hasattr(self,"review_status"):return
        session_exists=self.saved_session_path is not None
        save_enabled=not session_exists or self._session_dirty
        self.session_save_action.setEnabled(save_enabled)
        self.session_save_as_action.setEnabled(True); self.session_export_action.setEnabled(session_exists)
        self.session_toolbar_buttons["Save"].setEnabled(save_enabled)
        if not session_exists:
            self.review_validate_button.setEnabled(False); self.review_open_button.setEnabled(False)
            self.session_toolbar_buttons["Open in FIT"].setEnabled(False)
            self.review_status.setText("No measurements acquired yet")
            self.review_details.setText("<b>BPM noise</b> &nbsp; Missing<br><b>Dispersion</b> &nbsp; Missing<br><b>ORM</b> &nbsp; Missing<br><b>FIT readiness</b> &nbsp; Incomplete")
            self.review_paths.setText("Active measurement HDF5: —\nFull session manifest: Not saved yet")
            for label in self.role_status_labels.values():label.setText("Missing")
            for actions in self.role_manage_actions.values():actions["load"].setEnabled(True); actions["replace"].setEnabled(False); actions["remove"].setEnabled(False)
            return
        complete=False
        try:
            session=load_session(self.saved_session_path,validate_files=False)
            roles={entry.role for entry in session.files}
            complete=session.is_complete_for_loco
            completeness="Complete for FIT" if session.is_complete_for_loco else "Missing for FIT: "+", ".join(session.missing_roles)
        except Exception as exc:
            roles=set(); completeness=f"Session could not be read: {exc}"
            session=None
        latest_role=session.files[-1].role if session is not None and session.files else None
        kind={"orm":"ORM","dispersion":"Dispersion","bpm_noise":"BPM noise"}.get(latest_role,"Restored session")
        latest_result=self._measurement_results.get(latest_role) if latest_role is not None else None
        restoration="Not applicable"
        if isinstance(latest_result,DispersionResult):restoration=latest_result.restoration_status.replace("_"," ").title()
        elif isinstance(latest_result,ORMResult):
            statuses=tuple(latest_result.restoration_status)
            restoration="All correctors restored" if statuses and all(status in {"restored","confirmed_by_operator"} for status in statuses) else "Check restoration status"
        has_measurements=bool(roles)
        self.review_validate_button.setEnabled(has_measurements); self.review_open_button.setEnabled(has_measurements)
        self.session_toolbar_buttons["Open in FIT"].setEnabled(has_measurements)
        def role_state(role):
            state=self._measurement_restore_status.get(role,{})
            if role not in roles:return "Missing —"
            if state.get("restored"):return "File present · validated · result restored ✓"
            if state.get("validated"):return "File present · validated · plot unavailable: "+str(state.get("error") or "unknown reason")
            return "File present · validation failed: "+str(state.get("error") or "not validated")
        role_rows="".join(
            f"<tr><td><b>{label}</b></td><td>{role_state(role)}</td></tr>"
            for role,label in (("bpm_noise","BPM noise"),("dispersion","Dispersion"),("orm","ORM"))
        )
        restored_complete=complete and all(self._measurement_restore_status.get(role,{}).get("restored") for role in ("bpm_noise","dispersion","orm"))
        mapping_ready=self.adapter_combo.currentData()!="pyaml" or self.measurement_device_alignment is not None
        if restored_complete and mapping_ready:self.review_status.setText("✓ Measurement session restored and ready for FIT")
        elif complete:self.review_status.setText("Measurement files complete; review restoration and device mapping status")
        elif not has_measurements:self.review_status.setText("✓ Session configuration saved — no measurements acquired yet")
        else:self.review_status.setText(f"✓ {kind} saved — measurement session incomplete")
        self.review_details.setText(
            f"<table cellspacing='0' cellpadding='3'>{role_rows}</table>"
            f"<br><b>Latest measurement</b> &nbsp; {kind+' — files and manifest validated' if has_measurements else 'None'}"
            f"<br><b>Restoration</b> &nbsp; {restoration}"
            f"<br><b>FIT readiness</b> &nbsp; {completeness}; "
            f"{'device mapping available' if mapping_ready else 'device mapping requires validation'}"
        )
        display_path=self.saved_measurement_path
        if display_path is None and session is not None:
            display_path=next(iter(session.resolve(self.saved_session_path).values()),None)
        self.review_paths.setText(f"Active measurement HDF5: {display_path or '—'}\nFull session manifest: {self.saved_session_path}")
        entries={entry.role:entry for entry in session.files} if session is not None else {}
        for role,label in self.role_status_labels.items():
            entry=entries.get(role); label.setText(f"Available — {Path(entry.path).name}" if entry else "Missing")
            actions=self.role_manage_actions[role]; actions["load"].setEnabled(entry is None); actions["replace"].setEnabled(entry is not None); actions["remove"].setEnabled(entry is not None)

    @staticmethod
    def _h5_strings(dataset):
        return tuple(value.decode() if isinstance(value,bytes) else str(value) for value in dataset[:])

    def _restore_saved_result(self,kind,path):
        """Rebuild plot inputs read-only from recorded HDF5 arrays."""
        import h5py
        validate_measurement_file(path)
        with h5py.File(path,"r") as handle:
            metadata=json.loads(handle["metadata/json"][()]) if "metadata/json" in handle else {}
            names=self._h5_strings(handle["names/bpms"])
            bpms=tuple(BpmDevice(name,"saved HDF5 X","saved HDF5 Y") for name in names)
            if kind=="bpm_noise":
                return BpmNoiseResult(bpms,np.asarray(handle["raw/orbits_x_m"]),np.asarray(handle["raw/orbits_y_m"]),float(metadata.get("elapsed_seconds",0.0)),int(handle.attrs.get("noise_standard_deviation_ddof",0)))
            if kind=="dispersion":
                saved_labels=self._h5_strings(handle["raw/state_labels"])
                # Current acquisitions record both reference_before and
                # reference_after.  DispersionResult uses the pre-perturbation
                # reference under its historical canonical label "reference".
                labels=tuple("reference" if label=="reference_before" else label for label in saved_labels)
                requested=np.asarray(handle["raw/rf_setpoint_hz"]); actual=np.asarray(handle["raw/rf_readback_hz"]); raw_x=np.asarray(handle["raw/orbits_x_m"]); raw_y=np.asarray(handle["raw/orbits_y_m"])
                direction=str(metadata.get("dispersion_direction",handle.attrs.get("direction","bipolar")))
                required={"reference","negative","positive"} if direction=="bipolar" else {"reference",direction}
                if not required.issubset(labels):raise ValueError("historical RF-state labels required to reconstruct the dispersion plots were not recorded")
                confirmed=np.asarray(handle["raw/operator_confirmed"]) if "raw/operator_confirmed" in handle else np.ones(len(labels),dtype=bool)
                verified=np.asarray(handle["raw/rf_readback_verified"]) if "raw/rf_readback_verified" in handle else np.zeros(len(labels),dtype=bool)
                sources=self._h5_strings(handle["raw/rf_actual_sources"]) if "raw/rf_actual_sources" in handle else tuple("not recorded" for _ in labels)
                timestamps=np.asarray(handle["raw/sample_timestamps_s"]) if "raw/sample_timestamps_s" in handle else None; states=[]
                for index,label in enumerate(labels):
                    x=raw_x[index]; y=raw_y[index]
                    if x.ndim==1:x=x[np.newaxis,:]; y=y[np.newaxis,:]
                    states.append(DispersionStateResult(label,float(requested[index]),float(actual[index]),bool(confirmed[index]),x,y,timestamps[index] if timestamps is not None else np.asarray([],dtype=float),names,sources[index],bool(verified[index])))
                nominal=next(state.actual_rf_hz for state in states if state.label=="reference"); offset=abs(float(handle.attrs.get("rf_step_hz",metadata.get("rf_step_hz",0.0))))/(2 if direction=="bipolar" else 1)
                result=DispersionResult(bpms,tuple(states),direction,nominal,offset,str(handle.attrs.get("restoration_status","not_recorded")),float(metadata.get("elapsed_seconds",0.0)))
                for dataset,calculated in (("measured_eta_x",result.measured_eta_x),("measured_eta_y",result.measured_eta_y)):
                    if dataset in handle and not np.allclose(np.asarray(handle[dataset]),calculated,rtol=1e-10,atol=1e-15,equal_nan=True):
                        raise ValueError(f"stored {dataset} is inconsistent with the recorded RF states/sign convention")
                return result
            if kind=="orm":
                h_names=self._h5_strings(handle["names/horizontal_correctors"]); v_names=self._h5_strings(handle["names/vertical_correctors"]); h=tuple(CorrectorDevice(name,"saved HDF5 setpoint","saved HDF5 readback","H") for name in h_names); v=tuple(CorrectorDevice(name,"saved HDF5 setpoint","saved HDF5 readback","V") for name in v_names); count=len(h)+len(v)
                required=("setpoints/original","setpoints/requested_state_a","setpoints/requested_state_b","setpoints/actual_state_a","setpoints/actual_state_b","setpoints/intermediate_restored","setpoints/final","setpoints/restoration_status","setpoints/intermediate_restoration_status","raw/reference_before_m","raw/reference_intermediate_m","raw/reference_final_m")
                missing=[name for name in required if name not in handle]
                if missing:raise ValueError("historical ORM diagnostics required to reconstruct all plots were not recorded: "+", ".join(missing))
                raw_a=np.asarray(handle["raw/orbit_plus_m"]); raw_b=np.asarray(handle["raw/orbit_minus_m"])
                def array(name):return np.asarray(handle[name])
                def strings(name):return self._h5_strings(handle[name])
                requested=np.concatenate((np.asarray(handle["kicks/horizontal/requested"]),np.asarray(handle["kicks/vertical/requested"]))); actual=np.concatenate((np.asarray(handle["kicks/horizontal/actual"]),np.asarray(handle["kicks/vertical/actual"])))
                return ORMResult(bpms,h,v,np.asarray(handle["response_matrix"]),str(handle.attrs.get("direction","bipolar")),bool(handle.attrs.get("scaled",False)),requested,actual,array("setpoints/original"),array("setpoints/requested_state_a"),array("setpoints/requested_state_b"),array("setpoints/actual_state_a"),array("setpoints/actual_state_b"),array("setpoints/intermediate_restored"),strings("setpoints/intermediate_restoration_status"),array("setpoints/final"),raw_a,raw_b,array("raw/reference_before_m"),array("raw/reference_intermediate_m"),array("raw/reference_final_m"),np.asarray(handle["raw/timestamps_plus_s"]) if "raw/timestamps_plus_s" in handle else np.empty((0,0)),np.asarray(handle["raw/timestamps_minus_s"]) if "raw/timestamps_minus_s" in handle else np.empty((0,0)),strings("setpoints/restoration_status"),float(metadata.get("elapsed_seconds",0.0)))
        raise ValueError("unsupported saved measurement kind")

    def _measurement_type_changed(self, *_):
        if not hasattr(self,"measurement_type"): return
        kind=self.measurement_type.currentData()
        if kind!=self._active_selection_kind:
            old_kind=self._active_selection_kind
            self._store_active_selection(); self._store_measurement_view_state(old_kind)
            self._active_selection_kind=kind; self._restore_selection(kind)
            restored_state=self._restore_measurement_view_state(kind)
        else:restored_state=bool(self._measurement_view_states.get(kind))
        self._attach_measurement_workspace(kind)
        dispersion=kind=="dispersion"; orm=kind=="orm"
        if orm and not restored_state and not getattr(self,"_fresh_orm_defaults_applied",False):
            # Match the established PETRA III ORM procedure for a fresh GUI
            # workflow. Loaded projects subsequently restore their own values.
            self.readings.setValue(10); self.delay.setValue(.1); self.orm_settling_delay.setValue(1.0)
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
        self._reset_measurement_display(clear_session=False)
        self.result=self._measurement_results.get(kind)
        self.saved_measurement_path=self._measurement_result_paths.get(kind)
        if self.result is None and self.saved_measurement_path is not None:
            try:self.result=self._restore_saved_result(kind,self.saved_measurement_path); self._measurement_results[kind]=self.result
            except Exception as exc:self.statusBar().showMessage(f"{kind.replace('_',' ').title()} plot not available in saved data: {exc}")
        if self.result is not None:
            self._show_result(self.result); self._set_completed_result_view(True)
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
            # A default filename stem may legitimately accompany an operator's
            # custom label.  Refresh only labels that are still application
            # defaults; otherwise opening a saved session silently discards
            # the configured scientific description.
            current_label=self.measurement_label.text().strip()
            default_label=(not current_label or current_label.startswith("pyLOCO BPM noise — ")
                           or current_label.startswith("pyLOCO dispersion — ")
                           or current_label.startswith("pyLOCO ORM — "))
            self.measurement_name.setText(names[kind])
            if default_label:self.measurement_label.setText(labels[kind])
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
            "The stored measured_eta_x/y datasets are RF-induced orbit differences in metres [m] used directly by the current pyLOCO FIT importer. "
            "Physical dispersion is a separate derived quantity, also dimensionally in metres [m], calculated only with a verified RF-to-momentum conversion from an AT reference lattice; it does not replace the FIT datasets."
        )
        convention={
            "bipolar":"RF orbit difference = mean(+Δf/2) − mean(−Δf/2)\ncanonical total RF separation = f+ − f− = +Δf\nphysical dispersion uses the matching momentum-difference sign",
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
        direction=self.orm_direction.currentData()
        if direction=="bipolar":
            self.orm_hkick_label.setText("Total horizontal bipolar separation ΔK")
            self.orm_vkick_label.setText("Total vertical bipolar separation ΔK")
            self.orm_state_explanation.setText(
                "Applied states: K₀ + ΔK/2 and K₀ − ΔK/2\n"
                "Effective readback separation: K⁺ − K⁻ = +ΔK"
            )
        elif direction=="positive":
            self.orm_hkick_label.setText("Horizontal one-sided kick +ΔK")
            self.orm_vkick_label.setText("Vertical one-sided kick +ΔK")
            self.orm_state_explanation.setText(
                "Applied states: K₀ + ΔK and reference K₀\n"
                "Effective readback separation: K⁺ − K₀ = +ΔK"
            )
        else:
            self.orm_hkick_label.setText("Horizontal one-sided kick −ΔK")
            self.orm_vkick_label.setText("Vertical one-sided kick −ΔK")
            self.orm_state_explanation.setText(
                "Applied states: K₀ − ΔK and reference K₀\n"
                "Effective readback separation: K⁻ − K₀ = −ΔK"
            )
        self._update_kick_preview(); self._update_measurement_help(); self.refresh_plan()
        if getattr(self,"_displayed_orm_result",None) is not None:self._show_orm_result(self._displayed_orm_result)

    def _update_kick_preview(self):
        if not hasattr(self,"kick_preview"):return
        try:h,v=self._load_kick_arrays(); error=""
        except Exception as exc:h=np.array([]); v=np.array([]); error=str(exc)
        sign=-1.0 if self.orm_direction.currentData()=="negative" else 1.0
        rows=list(zip(("Horizontal",)*len(h),self.selected_hcorrectors,sign*h))+list(zip(("Vertical",)*len(v),self.selected_vcorrectors,sign*v)); self.kick_preview.setRowCount(len(rows)); self.kick_preview.setToolTip(error)
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
        if len({d.name for d in result}) != len(result): raise ValueError("The BPM selection contains duplicates")
        return tuple(result)

    def refresh_preview(self):
        self._model_orm_cache=None
        try:selected=self._resolve_selection(); self._bpm_selection_error=""
        except Exception as exc:selected=(); self._bpm_selection_error=str(exc)
        if (not self._loading_selection_state and
                tuple(device.name for device in self.selected_devices)!=tuple(device.name for device in selected)):
            self._invalidate_device_alignment("BPM inventory changed — verify device alignment again.")
            if hasattr(self,"device_selection_reviewed"):self.device_selection_reviewed.setChecked(False)
        self.selected_devices=tuple(selected); self.preview_table.blockSignals(True); self.preview_table.setRowCount(len(self.devices)); selected_positions={device.name:position for position,device in enumerate(selected)}
        for row,device in enumerate(self.devices):
            for column,value in enumerate((selected_positions.get(device.name,"—"),device.name,device.identifier)):
                item=QTableWidgetItem(str(value)); self.preview_table.setItem(row,column,item)
                if column==1:item.setFlags(item.flags()|Qt.ItemIsUserCheckable); item.setCheckState(Qt.Checked if device.name in selected_positions else Qt.Unchecked)
        self.preview_table.blockSignals(False)
        source={"all":"adapter discovery","names_file":"names file","manual":"manual selection"}.get(self.selection_method.currentData(),"selection")
        if not self._loading_selection_state:
            self._selection_states[self._active_selection_kind]["bpm"]={
                "method":self.selection_method.currentData(), "manual":self.manual_input.text(),
                "names_file":self.names_file.text(), "excluded_positions":self.bpm_exclusions.text(),
                "search":self.bpm_search.text(), "resolved_names":[device.name for device in selected],
            }
        self.preview_table.setToolTip(f"Source: {source}; checked devices define the measurement inventory, while highlighted rows are only for selection helpers.")
        self._filter_bpm_table(self.bpm_search.text()); self.refresh_plan(); self._update_workflow_tabs()

    def refresh_plan(self, *_):
        if not hasattr(self,"plan_values"): return
        kind=self.measurement_type.currentData(); dispersion=kind=="dispersion"; orm=kind=="orm"; state_count=len(self._dispersion_state_specs()) if dispersion else 1
        active_settling=self.orm_settling_delay.value() if orm else self.settling_delay.value()
        per_state=max(0,self.readings.value()-1)*self.delay.value()+(active_settling if (dispersion or orm) else 0)
        if orm:
            orbit_states=5 if self.orm_direction.currentData()=="bipolar" else 4; writes=4 if self.orm_direction.currentData()=="bipolar" else 3
            estimate=(len(self.selected_hcorrectors)+len(self.selected_vcorrectors))*(orbit_states*max(0,self.readings.value()-1)*self.delay.value()+writes*self.orm_settling_delay.value())
        else:estimate=state_count*per_state
        self.duration.setText(f"approximately {estimate:.2f} s")
        output=self._resolved_output_directory()
        try:self._load_kick_arrays() if orm else None; kick_valid=True
        except Exception:kick_valid=False
        automatic_rf=dispersion and self.rf_control_mode.currentData()=="automatic"
        requires_connection=self.adapter_combo.currentData() in ("pysc", "pyaml"); connection_ok=not requires_connection or self.connection_verified
        rf_safe=not (automatic_rf and self._rf_restoration_unresolved)
        orm_safe=not (orm and self._orm_restoration_unresolved)
        valid=bool(connection_ok and rf_safe and orm_safe and self.selected_devices and self.measurement_name.text().strip() and self.output_directory.text().strip() and (not dispersion or automatic_rf or self._nominal_rf_value() is not None) and (not automatic_rf or AdapterCapability.RF_WRITE in self.adapter.capabilities) and (not orm or (self.selected_hcorrectors or self.selected_vcorrectors) and kick_valid and AdapterCapability.WRITE in self.adapter.capabilities))
        direction={"bipolar":"Bipolar ±Δf","positive":"One-sided +Δf","negative":"One-sided −Δf"}.get(self.dispersion_direction.currentData(),"—")
        try:kh,kv=self._load_kick_arrays()
        except Exception:kh=kv=np.array([])
        descriptor=InterfaceRegistry().descriptor(self.adapter_combo.currentData()); adapter_text=descriptor.badge
        if descriptor.key == "pyaml":
            adapter_text = self.status_badge.text()
        nominal=self._nominal_rf_value(); offset=self._rf_offset_hz(); negative="Waiting for RF readback" if nominal is None else f"{nominal-offset:.3f} Hz"; positive="Waiting for RF readback" if nominal is None else f"{nominal+offset:.3f} Hz"
        separation=2*offset if self.dispersion_direction.currentData()=="bipolar" else offset
        values={"measurement":"ORM" if orm else "Dispersion" if dispersion else "BPM noise","bpms":str(len(self.selected_devices)),"h_correctors":str(len(self.selected_hcorrectors)),"v_correctors":str(len(self.selected_vcorrectors)),"total_correctors":str(len(self.selected_hcorrectors)+len(self.selected_vcorrectors)),"kick_source":"Per-corrector file" if self.orm_kick_mode.currentData()=="file" else "Common","h_kick":f"{np.min(kh)*1e6:g}–{np.max(kh)*1e6:g} µrad" if kh.size else "Invalid","v_kick":f"{np.min(kv)*1e6:g}–{np.max(kv)*1e6:g} µrad" if kv.size else "Invalid","scaled_orm":"Saved FIT matrix: orbit difference [m]"+("; normalized analysis view enabled" if self.orm_normalized_analysis.isChecked() else ""),"orm_direction":self.orm_direction.currentText(),"rf_mode":"Automatic backend RF with restoration protection" if automatic_rf else "Manual (no writes)","nominal_rf":"Read from backend" if automatic_rf and nominal is None else f"{nominal:.3f} Hz" if nominal else "Required","rf_step_±δf":f"{offset:g} Hz per side","bipolar_separation":f"{separation:g} Hz total","negative_rf":negative,"positive_rf":positive,"direction":direction,"readings":str(self.readings.value()),"delay":f"{self.delay.value():g} s","settling_delay":f"{active_settling:g} s","rf_states":", ".join(spec[0] for spec in self._dispersion_state_specs()) or "backend-derived","est_duration":f"{estimate:.2f} s","adapter":adapter_text,"output":str(output)}
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
        if automatic_rf and self._rf_restoration_unresolved:reasons.append("RF restoration is unresolved — stop writes and verify RF before continuing")
        if orm and self._orm_restoration_unresolved:reasons.append("corrector restoration is unresolved — stop writes and verify affected correctors before continuing")
        if orm and not (self.selected_hcorrectors or self.selected_vcorrectors):reasons.append("select at least one corrector")
        if orm and not kick_valid:reasons.append("corrector kick settings are invalid")
        if reasons:self.start_block_reason.setText("Cannot start: "+"; ".join(reasons)+".")
        elif automatic_rf:self.start_block_reason.setText(f"Ready for automatic dispersion through {adapter_text.split(' • ')[0]} • RF WRITES AUTHORIZED; BPM orbit acquisition is read only.")
        elif orm:self.start_block_reason.setText(f"Ready to acquire ORM through {adapter_text} — temporary corrector writes authorized; BPM orbit acquisition is read only.")
        else:self.start_block_reason.setText("Ready to acquire through "+adapter_text+".")
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
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="automatic" and self._rf_restoration_unresolved:
            QMessageBox.critical(self,"RF writes blocked","A previous RF restoration could not be verified. Stop further machine writes and check the RF frequency, then reconnect and verify the RF readback."); return
        if self.measurement_type.currentData()=="orm": return self._start_orm()
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="automatic": return self._start_automatic_dispersion()
        if self.measurement_type.currentData()=="dispersion": return self._start_dispersion_step()
        self._active_acquisition_kind="bpm_noise"; self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat(f"Starting — 0 / {self.readings.value()}"); self.reading_status.setText("Starting…"); self.elapsed.setText("Elapsed: 0.00 s"); self.remaining.setText(f"Remaining: ~{max(0,self.readings.value()-1)*self.delay.value():.2f} s"); self.samples.setText(f"Samples: 0 / {self.readings.value()}"); self._reset_live_plot(); self.log.clear(); self.log.appendPlainText(f"Starting BPM-noise measurement through {self.status_badge.text()}…")
        self.thread=QThread(self); self.worker=AcquisitionWorker(BpmNoiseAcquirer(self.adapter,self.selected_devices,ddof=int(self.bpm_noise_convention.currentData())),self.readings.value(),self.delay.value(),self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._on_progress); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed): signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    def _start_orm(self):
        if self._orm_restoration_unresolved:
            QMessageBox.critical(self,"Corrector writes blocked","A previous ORM restoration could not be verified. Stop further machine writes and check the affected correctors, then reconnect and verify the available restoration evidence."); return
        if isinstance(self.adapter,PETRAReadOnlyAdapter):QMessageBox.warning(self,"ORM unavailable","ORM needs write access. PETRA is read only."); return
        if self.adapter_combo.currentData()=="pyaml":
            for device in self.selected_hcorrectors+self.selected_vcorrectors:
                resolver=getattr(self.adapter,"corrector_restoration_evidence",None)
                evidence=resolver(device.name) if callable(resolver) else {"level":"unavailable"}
                if evidence.get("level")=="independent":continue
                if evidence.get("level")=="adapter_value":continue
                QMessageBox.warning(self,"Cannot start ORM",f"No usable corrector restoration verification is available for {device.name}. The pyAML strength accessor must provide readback() or get(). ORM writes remain blocked."); return
        self._refresh_corrector_preview("hcor"); self._refresh_corrector_preview("vcor")
        try:kick_h,kick_v=self._load_kick_arrays(); acquirer=ORMAcquirer(self.adapter,self.selected_devices,self.selected_hcorrectors,self.selected_vcorrectors)
        except Exception as exc:QMessageBox.warning(self,"Cannot start ORM",str(exc)); return
        self._orm_has_completed_column = False
        self._model_orm_cache=None
        if self.compare_reference.isChecked() and self.reference_model is not None:
            try:
                self._model_orm_cache=model_orm(self.reference_model,self.selected_devices,self.selected_hcorrectors,self.selected_vcorrectors,kick_h,kick_v,scaled=False,**self._model_mapping_kwargs())
            except Exception as exc:self.statusBar().showMessage(f"Reference-model ORM unavailable: {exc}")
        self._active_acquisition_kind="orm"; self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat("Preparing ORM"); self.reading_status.setText("Preparing first corrector…"); self.log.clear(); self.log.appendPlainText(f"Starting ORM through {self.status_badge.text()}; each original setpoint will be restored.")
        if self.adapter_combo.currentData()=="pyaml" and getattr(self.adapter,"allow_get_only_restoration",False):
            self.log.appendPlainText("Corrector restoration evidence: adapter strength.get() values will be checked; independent readback unavailable. Approved ESRF simulator workflow only — not authorization for real-machine operation.")
        options={"direction":self.orm_direction.currentData(),"scaled":False,"readings":self.readings.value(),"delay_seconds":self.delay.value(),"settling_delay_seconds":self.orm_settling_delay.value()}
        self.thread=QThread(self); self.worker=ORMAcquisitionWorker(acquirer,kick_h,kick_v,options,self.cancel_event); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.event.connect(self._on_orm_event); self.worker.completed.connect(self._on_completed); self.worker.cancelled.connect(self._on_cancelled); self.worker.failed.connect(self._on_failed)
        for signal in (self.worker.completed,self.worker.cancelled,self.worker.failed):signal.connect(self._finish_thread)
        self._set_acquisition_running(True); self.thread.start()

    def _start_automatic_dispersion(self):
        try:acquirer=AutomaticDispersionAcquirer(self.adapter,self.selected_devices)
        except Exception as exc:QMessageBox.warning(self,"Automatic RF unavailable",str(exc)); return
        self._active_acquisition_kind="dispersion"; self.cancel_event=Event(); self.result=None; self.progress.setValue(0); self.progress.setFormat("Preparing RF scan"); self.reading_status.setText("Reading original RF frequency…"); self.log.clear(); self.log.appendPlainText(f"Starting automatic dispersion through {self.status_badge.text()}; original RF is restored in cleanup.")
        options={"rf_step_hz":self._rf_offset_hz(),"readings":self.readings.value(),"delay_seconds":self.delay.value(),"direction":self.dispersion_direction.currentData(),"settling_delay_seconds":self.settling_delay.value(),"verify_restored_orbit":self.verify_restored_orbit.isChecked(),"restoration_tolerance_hz":self.rf_restoration_tolerance.value()}
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
        self._active_acquisition_kind="dispersion"; self.cancel_event=Event(); self.result=None
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
            evidence_levels={self.adapter.corrector_restoration_evidence(device.name).get("level") for device in result.correctors}
            if evidence_levels=={"independent"}:
                self.log.appendPlainText("Correctors restored — independent readback matched.")
            else:
                self.log.appendPlainText("Corrector restored — adapter strength.get() value matched; independent readback unavailable.")
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
        self.thread=None; self.worker=None; self._set_acquisition_running(False); self._active_acquisition_kind=None; self.refresh_plan()

    def _manual_rf_restore_instruction(self):
        nominal=self._nominal_rf_value()
        return f" Restore RF externally to nominal {nominal:.6f} Hz and verify it before leaving the machine." if nominal is not None else " Restore RF externally to its original nominal value and verify it before leaving the machine."

    def cancel_acquisition(self):
        self.cancel_event.set(); self.log.appendPlainText("Stopping…")
        if self.measurement_type.currentData()=="dispersion" and self.rf_control_mode.currentData()=="manual":self.log.appendPlainText(self._manual_rf_restore_instruction().strip())
    @staticmethod
    def _restoration_diagnostic_text(event):
        data=getattr(event,"restoration_diagnostics",None) or {}
        if not data:return ""
        if data.get("orbit_correction"):
            rows = []
            for correction in data["orbit_correction"]:
                rows.append(f"Orbit correction for {correction['corrector']}: {correction['status']}; "
                            f"correctors={correction['correctors']}; strengths before={correction['before_native']}; "
                            f"strengths after={correction.get('after_native')}; units={correction['units']}; "
                            f"error={correction.get('error', '')}; rollback={correction.get('rollback', [])}")
            text = "\n" + "\n".join(rows) + "\nSuccessful earlier corrections remain applied."
            if "correctors" in data:
                remainder = dict(data)
                remainder.pop("orbit_correction")
                text += MeasureMainWindow._restoration_diagnostic_text(SimpleNamespace(restoration_diagnostics=remainder))
            return text
        if "correctors" in data:
            rows=[]
            for item in data.get("correctors",()):
                rows.append(f"{item.get('name')} ({item.get('plane')}): Set command issued={item.get('set_command_issued')}; Evidence={item.get('verification_source','adapter read')}; Independent readback verified={item.get('independent_readback_verified',False)}; Original setpoint={item.get('original_setpoint_rad')} rad; Original readback={item.get('original_readback_rad')} rad; Restoration target={item.get('attempted_restoration_rad')} rad; Final readback={item.get('final_readback_rad')} rad; Tolerance={item.get('tolerance_rad')} rad; Attempts={item.get('attempts')}; Status={item.get('status')}"+(f"; Error={item.get('error')}" if item.get('error') else ""))
            return "\nCorrector restoration diagnostics (SI kick angle in rad; converted only from verified angular component metadata):\n"+"\n".join(rows)
        return (f" Original setpoint={data.get('original_rf_hz')} Hz; Restoration target={data.get('attempted_restoration_hz')} Hz; "
                f"Final readback={data.get('final_readback_hz')} Hz; Absolute difference={data.get('absolute_difference_hz')} Hz; "
                f"Tolerance={data.get('tolerance_hz')} Hz; Status={data.get('status')}")

    def _verify_unresolved_orm_restoration(self):
        if not self._orm_restoration_unresolved:return True
        diagnostics=self._orm_restoration_diagnostics or {}; inventory={device.name:device for device in self.horizontal_correctors+self.vertical_correctors}
        if diagnostics.get("orbit_correction") and "correctors" not in diagnostics:
            from pyLOCO.control_system.pyaml_adapter import PyAMLAdapter
            if not isinstance(self.adapter, PyAMLAdapter) or not self.adapter.orbit_correction_enabled:
                return False
            verified = self.adapter.verify_orbit_correction_restoration(diagnostics["orbit_correction"][-1])
            if verified:
                self._orm_restoration_unresolved = False
                diagnostics["status"] = "restored"
                self.statusBar().showMessage("Orbit-correction rollback targets verified; corrector-write block cleared.")
            return verified
        rows=diagnostics.get("correctors",()); verified=bool(rows)
        for item in rows:
            device=inventory.get(item.get("name"))
            if device is None or device.readback_channel!=item.get("readback_channel"):
                verified=False; continue
            try:final=float(self.adapter.read(device.readback_channel).value)
            except Exception:verified=False; continue
            item["recovery_readback_rad"]=final
            item["recovery_verified"]=bool(np.isfinite(final) and np.isclose(final,float(item["original_readback_rad"]),rtol=0,atol=float(item["tolerance_rad"])))
            verified=verified and item["recovery_verified"]
        if verified:
            self._orm_restoration_unresolved=False; diagnostics["status"]="restored"; self.statusBar().showMessage("Connection verified; affected corrector readbacks match their original values. Corrector-write block cleared.")
        return verified

    def _on_cancelled(self,message):
        event=message; message=str(message); restoration=getattr(event,"restoration_status",None)
        operation=self._event_operation(event)
        instruction=self._manual_rf_restore_instruction() if operation=="dispersion" and self.rf_control_mode.currentData()=="manual" else ""
        if operation=="orm":
            if restoration=="restored":
                level=(getattr(event,"restoration_diagnostics",None) or {}).get("verification_level")
                message=("ORM measurement cancelled. All correctors restored and independently verified." if level=="independent" else "ORM measurement cancelled. Corrector restored — adapter strength.get() value matched; independent readback unavailable.")
            elif restoration=="not_required":message="ORM measurement cancelled before a corrector change; restoration was not required."
            elif restoration:
                self._orm_restoration_unresolved=True; self._orm_restoration_diagnostics=getattr(event,"restoration_diagnostics",None); message="ORM measurement cancelled. Corrector restoration could not be verified. Stop further machine writes and check the affected correctors."
                QMessageBox.critical(self,"Corrector restoration not verified",message+self._restoration_diagnostic_text(event))
        elif restoration=="restored":
            message="Measurement cancelled. Original RF frequency restored and verified."
        elif restoration:
            self._rf_restoration_unresolved=True; self._rf_restoration_diagnostics=getattr(event,"restoration_diagnostics",None); message="Measurement cancelled. RF restoration could not be verified. Stop further machine writes and check the RF frequency."
            QMessageBox.critical(self,"RF restoration not verified",message+self._restoration_diagnostic_text(event))
        self.progress.setValue(0); self.progress.setFormat("Cancelled"); self.reading_status.setText(message+instruction); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message+self._restoration_diagnostic_text(event)+instruction); self._update_workflow_tabs(); self.refresh_plan()
    def _on_failed(self,message):
        event=message; message=str(message); restoration=getattr(event,"restoration_status",None)
        operation=self._event_operation(event)
        instruction=self._manual_rf_restore_instruction() if operation=="dispersion" and self.rf_control_mode.currentData()=="manual" else ""
        if operation=="orm":
            if restoration not in (None,"restored","not_required"):
                self._orm_restoration_unresolved=True; self._orm_restoration_diagnostics=getattr(event,"restoration_diagnostics",None); message="ORM measurement failed. Corrector restoration could not be verified. Stop further machine writes and check the affected correctors."
            elif restoration=="restored":message="ORM measurement failed, but all affected correctors were restored according to the recorded verification evidence."
        elif restoration and restoration!="restored":
            self._rf_restoration_unresolved=True; self._rf_restoration_diagnostics=getattr(event,"restoration_diagnostics",None); message="RF restoration could not be verified. Stop further machine writes and check the RF frequency."
        diagnostic=self._restoration_diagnostic_text(event)
        self.progress.setValue(0); self.progress.setFormat("Failed"); self.reading_status.setText("Failed — no measurement saved. "+message+instruction); self.remaining.setText("Remaining: —"); self.log.appendPlainText(message+diagnostic+instruction); QMessageBox.critical(self,"Measurement failed",message+diagnostic+instruction); self._update_workflow_tabs(); self.refresh_plan()

    def _event_operation(self,event):
        """Classify terminal worker events without consulting mutable GUI state."""
        diagnostics=getattr(event,"restoration_diagnostics",None) or {}
        if isinstance(event,(ORMInterrupted,ORMAcquisitionError)) or "correctors" in diagnostics:return "orm"
        if "original_rf_hz" in diagnostics:return "dispersion"
        return self._active_acquisition_kind or self.measurement_type.currentData()

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
            hit_indices=details.get("ind")
            if contains and hit_indices is not None:
                indices=np.asarray(hit_indices).reshape(-1)
                if indices.size>0:return int(indices[0]),plane
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
            if isinstance(result, ORMResult) and result.orbit_correction_diagnostics:
                options["orbit_correction_enabled"] = True
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
        model_metadata=self._reference_model_provenance() if self.reference_model is not None else {"reference_model_available":False}
        actual_readings=(result.raw_state_a_m.shape[1] if isinstance(result,ORMResult) and result.raw_state_a_m.ndim==3 else self.readings.value())
        metadata={"measurement_name":self.measurement_name.text(),"measurement_label":self.measurement_label.text(),"acquisition_timestamp":datetime.now().astimezone().isoformat(),"operator_comments":self.comments.toPlainText(),"adapter":self.adapter_combo.currentText(),"backend_key":descriptor.key,"backend_badge":descriptor.badge,"backend_environment":descriptor.environment,"backend_real_machine":descriptor.real_machine,"machine_identity":identity,**profile_metadata,**model_metadata,"compare_with_reference_model":self.compare_reference.isChecked(),"pysc_machine_profile":self.pysc_profile_combo.currentData() if descriptor.key=="pysc" else None,"readings_per_state":actual_readings,"delay_seconds":self.delay.value(),"settling_delay_seconds":self.orm_settling_delay.value() if isinstance(result,ORMResult) else self.settling_delay.value() if isinstance(result,DispersionResult) else 0.0,"elapsed_seconds":result.elapsed_seconds}
        if isinstance(result,ORMResult):
            nh=len(result.horizontal_correctors); diagnostics={}
            if result.orbit_correction_diagnostics:
                metadata["orbit_correction_enabled"] = True
                metadata["orbit_correction"] = list(result.orbit_correction_diagnostics)
                metadata["final_reference_role"] = "post-correction orbit; nominal restoration strengths recorded before correction"
            if result.raw_state_a_m.ndim==3 and result.raw_state_a_m.shape[1]>1:
                diagnostics={"state_a_first_sample_minus_later_mean_m":result.raw_state_a_m[:,0,:]-np.mean(result.raw_state_a_m[:,1:,:],axis=1),"state_b_first_sample_minus_later_mean_m":result.raw_state_b_m[:,0,:]-np.mean(result.raw_state_b_m[:,1:,:],axis=1)}
            bipolar=result.direction=="bipolar"
            if bipolar:
                validation_convention="historical_petra_bipolar_v3"
                kick_convention="total bipolar delta_K; K+=K0+delta_K/2; restore/verify K0 and acquire diagnostic reference; K-=K0-delta_K/2; restore/verify K0 and acquire diagnostic reference"
                restored_orbit_role="reference_before, reference_intermediate and reference_final are diagnostic-only and excluded from ORM calculation"
                subtraction_convention="mean(positive)-mean(negative)"
            else:
                sign="+" if result.direction=="positive" else "-"
                state_name="positive" if result.direction=="positive" else "negative"
                validation_convention="pyloco_one_sided_v1"
                kick_convention=f"one-sided delta_K; applied state K=K0{sign}delta_K; reference state K0; final restore/verify K0"
                restored_orbit_role="reference_before and reference_final are diagnostic-only and excluded from ORM calculation; no intermediate reference is acquired in one-sided mode"
                subtraction_convention=f"mean({state_name})-mean(reference)"
            write_orm(measurement,response_matrix=result.response_matrix,bpm_names=[d.name for d in result.bpms],horizontal_corrector_names=[d.name for d in result.horizontal_correctors],vertical_corrector_names=[d.name for d in result.vertical_correctors],requested_kick_h_rad=result.requested_kicks_rad[:nh],requested_kick_v_rad=result.requested_kicks_rad[nh:],actual_kick_h_rad=result.effective_kicks_rad[:nh],actual_kick_v_rad=result.effective_kicks_rad[nh:],orbit_plus_m=result.raw_state_a_m,orbit_minus_m=result.raw_state_b_m,reference_before_m=result.raw_reference_before_m,reference_intermediate_m=result.raw_reference_intermediate_m if bipolar else None,reference_final_m=result.raw_reference_final_m,scaled=result.scaled,direction=result.direction,original_setpoints_rad=result.original_setpoints_rad,requested_state_a_rad=result.requested_state_a_rad,requested_state_b_rad=result.requested_state_b_rad,actual_state_a_rad=result.actual_state_a_rad,actual_state_b_rad=result.actual_state_b_rad,intermediate_restored_setpoints_rad=result.intermediate_restored_setpoints_rad if bipolar else None,intermediate_restoration_status=result.intermediate_restoration_status,final_setpoints_rad=result.final_setpoints_rad,timestamps_plus_s=result.timestamps_state_a_s,timestamps_minus_s=result.timestamps_state_b_s,restoration_status=result.restoration_status,diagnostics=diagnostics,metadata={**metadata,"orm_validation_convention":validation_convention,"kick_convention":kick_convention,"restored_orbit_role":restored_orbit_role,"subtraction_convention":subtraction_convention,"normalization_convention":"divide each column by actual applied-state minus reference-state readback" if not bipolar else "divide each column by actual K+ minus K- readback only when scaled","response_matrix_unit":"m/rad" if result.scaled else "m","row_order":"horizontal_bpms,vertical_bpms","column_order":"horizontal_correctors,vertical_correctors","transient_diagnostic":"first orbit sample minus mean of later samples for each applied corrector state; diagnostic only; no samples discarded","selected_bpm_names":[d.name for d in result.bpms],"selected_horizontal_corrector_names":[d.name for d in result.horizontal_correctors],"selected_vertical_corrector_names":[d.name for d in result.vertical_correctors]})
            model_matrix=getattr(self,"_model_orm_cache",None)
            if model_matrix is not None and self.reference_model is not None:
                store_reference_model_arrays(measurement,{"orm":model_matrix,"orm_difference":result.response_matrix-model_matrix},units={"orm":"m/rad" if result.scaled else "m","orm_difference":"m/rad" if result.scaled else "m"},provenance=self._reference_model_provenance())
        elif is_dispersion:
            automatic=self.rf_control_mode.currentData()=="automatic"; states=result.states; metadata.update({"rf_control_mode":"automatic" if automatic else "manual","nominal_rf_hz":result.nominal_rf_hz,"nominal_rf_source":"backend_readback" if automatic else self._manual_nominal_source,"requested_rf_offset_hz":result.requested_offset_hz,"requested_total_rf_separation_hz":result.canonical_rf_step_hz if result.direction=="bipolar" else abs(result.requested_offset_hz),"direction":result.direction,"canonical_measured_eta_definition":"mean_orbit_positive - mean_orbit_negative; canonical positive total RF separation; reference_before and reference_after are excluded","canonical_measured_eta_unit":"m","rf_difference_sign_convention":"positive_minus_negative","rf_normalized_response_unit":"m/Hz","actual_rf_available":bool(all(np.isfinite(s.actual_rf_hz) for s in states)),"rf_acquisition_order":[s.label for s in states],"rf_requested_frequencies_hz":[float(s.requested_rf_hz) for s in states],"rf_actual_frequencies_hz":[float(s.actual_rf_hz) for s in states],"rf_actual_frequency_sources":[s.actual_rf_source for s in states],"rf_readback_verified":[bool(s.rf_readback_verified) for s in states],"operator_confirmed_states":[bool(s.operator_confirmed) for s in states],"restoration_status":result.restoration_status,"rf_restoration_source":"backend_readback" if automatic else self._manual_restoration_source,"rf_restoration_readback_verified":True if automatic else self._manual_restoration_verified,"rf_restoration_tolerance_hz":self.rf_restoration_tolerance.value(),"settling_delay_seconds":self.settling_delay.value(),"dispersion_measurement_states":"reference_before,positive,negative","rf_restoration_is_measurement_state":False,"verify_restored_orbit":any(state.label=="reference_after" for state in states),"reference_after_role":"post-restoration diagnostic only; excluded from physical dispersion and RF orbit difference"})
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
                deltas=relative_momentum_deviation([plus.actual_rf_hz,minus.actual_rf_hz],result.nominal_rf_hz,eta); dx,span=physical_dispersion(plus.mean_x_m,minus.mean_x_m,*deltas); dy,_=physical_dispersion(plus.mean_y_m,minus.mean_y_m,*deltas)
                metadata.update({"momentum_compaction_factor":lattice_metadata["alpha"],"relativistic_correction_inverse_gamma_squared":lattice_metadata["inverse_gamma_squared"],"at_slip_factor":lattice_metadata["at_slip_factor"],"slip_factor_eta":eta,"slip_factor_convention":"AT ring.get_slip_factor() = 1/gamma^2 - alpha_c; eta = alpha_c - 1/gamma^2 = -AT slip factor","momentum_relation":MOMENTUM_RELATION,"delta_positive":float(deltas[0]),"delta_negative":float(deltas[1]),"delta_span":float(span),"delta_span_definition":"delta_positive - delta_negative","physical_dispersion_definition":"(mean_orbit_positive - mean_orbit_negative) / (delta_positive - delta_negative)","physical_dispersion_unit":"m"})
                diagnostics.update({"physical_dispersion_x_m":dx,"physical_dispersion_y_m":dy,"relative_momentum_deviation":deltas})
            saved_labels=["reference_before" if s.label=="reference" else s.label for s in states]
            write_dispersion(measurement,measured_eta_x=result.measured_eta_x,measured_eta_y=result.measured_eta_y,bpm_names=[d.name for d in result.devices],rf_frequency_hz=[s.requested_rf_hz for s in states],rf_setpoint_hz=[s.requested_rf_hz for s in states],rf_readback_hz=[s.actual_rf_hz for s in states],raw_orbits_x_m=np.stack([s.orbits_x_m for s in states]),raw_orbits_y_m=np.stack([s.orbits_y_m for s in states]),rf_step_hz=result.canonical_rf_step_hz,bidirectional=result.direction=="bipolar",metadata=metadata,diagnostics=diagnostics,state_labels=saved_labels,operator_confirmed=[s.operator_confirmed for s in states],rf_actual_sources=[s.actual_rf_source for s in states],rf_readback_verified=[s.rf_readback_verified for s in states],sample_timestamps_s=np.stack([s.timestamps_s for s in states]),restoration_status=result.restoration_status)
            if self.reference_model is not None and self.compare_reference.isChecked():
                try:
                    mapping=self._model_mapping_kwargs()
                    model_x,model_y=model_dispersion(self.reference_model,result.devices,bpm_ordinals=mapping.get("bpm_ordinals"))
                    store_reference_model_arrays(measurement,{"dispersion_x_m":model_x,"dispersion_y_m":model_y},units={"dispersion_x_m":"m","dispersion_y_m":"m"},provenance=self._reference_model_provenance())
                except Exception:pass
        else:
            convention="population" if result.ddof==0 else "sample"; metadata.update({"readings":self.readings.value(),"selected_bpm_names":[d.name for d in result.devices],"bpm_noise_statistic":f"{convention} standard deviation across readings per BPM","bpm_noise_ddof":result.ddof,"bpm_noise_unit":"m"}); write_bpm_noise(measurement,noise_x_m=result.noise_x_m,noise_y_m=result.noise_y_m,bpm_names=[d.name for d in result.devices],raw_orbits_x_m=result.orbits_x_m,raw_orbits_y_m=result.orbits_y_m,ddof=result.ddof,metadata=metadata)
        validate_measurement_file(measurement)
        if not getattr(self,"_fresh_measurement_session",True) and self.saved_session_path is not None:
            manifest=self.saved_session_path
        else:
            manifest=output/"measurement-session.pyloco-session.json"
            if manifest.exists():manifest=output/f"measurement-session-{stamp}.pyloco-session.json"
        entries=[]
        role="orm" if isinstance(result,ORMResult) else "dispersion" if is_dispersion else "bpm_noise"

        # A newly opened Measure window or New project starts a genuinely
        # fresh logical measurement session. Do not inherit roles from an
        # older manifest that happens to exist in the output directory.
        fresh_session = getattr(self, "_fresh_measurement_session", True)

        previous_session=None
        if manifest.exists() and not fresh_session:
            previous_session=load_session(manifest,validate_files=False); entries=list(previous_session.files)
            if any(entry.role==role for entry in entries) and self.isVisible():
                answer=QMessageBox.question(self,"Replace active measurement?",f"This session already contains {role.replace('_',' ')}. Replace its active reference with the newly acquired file?\n\nCancel keeps the new HDF5 file separately and leaves the session reference unchanged.",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)
                if answer!=QMessageBox.Yes:
                    self.statusBar().showMessage(f"New {role.replace('_',' ')} HDF5 kept separately; active session unchanged."); return
            entries=[e for e in entries if e.role!=role]

        options={"dataset":"response_matrix"} if isinstance(result,ORMResult) else {"horizontal_dataset":"measured_eta_x","vertical_dataset":"measured_eta_y"} if is_dispersion else {"horizontal_dataset":"Noise_BPMx","vertical_dataset":"Noise_BPMy"}
        options.update({"sha256":session_file_sha256(measurement),"size":measurement.stat().st_size})
        try:measurement_reference=measurement.relative_to(manifest.parent).as_posix()
        except ValueError:raise ValueError("Measurement output must remain inside the resumed session directory")
        entries.append(SessionFile(role,measurement_reference,options))

        session_metadata=dict(previous_session.metadata) if previous_session is not None else {}
        session_metadata.update({"label":self.measurement_label.text(),"updated":datetime.now().isoformat(timespec="seconds"),"machine_identity":identity,**profile_metadata})
        full=session_metadata.get("full_measure_session")
        if isinstance(full,dict):full["roles"]={name:name in {entry.role for entry in entries} for name in ("bpm_noise","dispersion","orm")}
        session=MeasurementSession(session_id=previous_session.session_id if previous_session is not None else output.name or safe,files=tuple(entries),metadata=session_metadata)
        save_session(manifest,session)

        # From this point onward this window owns the new manifest, so
        # Dispersion/ORM may accumulate alongside the first measurement.
        self._fresh_measurement_session = False
        self.saved_measurement_path=measurement; self.saved_session_path=manifest
        self._measurement_results[role]=result; self._measurement_result_paths[role]=measurement
        # A freshly written result is already resident in memory and the file
        # has just passed schema validation above.  Keep Review & Save's
        # validation/restoration cache in sync without requiring a close/open
        # cycle merely to display the correct status.
        self._measurement_restore_status[role]={
            "present":True,"validated":True,"restored":True,"error":""
        }
        self._session_dirty=True; self._update_session_title(); self.paths.setText(f"SAVED ✓\nMeasurement file: {measurement}\nSession manifest: {manifest}"); self.paths.setToolTip(str(measurement)); self.paths.setObjectName("savedPath"); self.paths.style().unpolish(self.paths); self.paths.style().polish(self.paths); self.open_output_button.setEnabled(True); self._refresh_review_page(); self.statusBar().showMessage(f"Saved: {measurement}")

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

    def _heatmap(self,canvas,data,title,unit,*,row_devices=None,column_devices=None,row_axis_label="BPM index"):
        values=np.asarray(data); finite=np.abs(values[np.isfinite(values)]); limit=float(np.max(finite)) if finite.size else 1.0; limit=limit or 1.0; canvas.clear(); self._orm_plot_state.pop(canvas,None); surface=hasattr(self,"orm_view_mode") and self.orm_view_mode.currentData()=="3d"; ax=canvas.figure.add_subplot(111,projection="3d" if surface else None); cmap=self._orm_colormap()
        signed_limits={"vmin":-limit,"vmax":limit} if self.orm_palette.currentData()=="red_blue" else {}
        if surface:
            columns,rows=np.meshgrid(np.arange(values.shape[1]),np.arange(values.shape[0])); artist=ax.plot_surface(columns,rows,values,cmap=cmap,linewidth=0,antialiased=False,rcount=values.shape[0],ccount=values.shape[1],**signed_limits); ax.set_zlabel(f"Response [{unit}]")
        else:
            # Explicit half-bin boundaries keep pixel centres aligned with
            # their zero-based scientific indices. Masked cells are pending,
            # not measured zeros, and receive an intentionally neutral color.
            rendered=np.ma.masked_invalid(values)
            from matplotlib import colormaps
            display_cmap=colormaps.get_cmap(cmap).with_extremes(bad="#596174")
            artist=ax.imshow(rendered,aspect="auto",origin="upper",cmap=display_cmap,
                             interpolation="nearest",
                             extent=(-.5,values.shape[1]-.5,values.shape[0]-.5,-.5),
                             **signed_limits)
        ax.set_title(title); ax.set_xlabel("Corrector index"); ax.set_ylabel(row_axis_label)
        if not surface and values.shape[1] <= 20:
            ax.set_xticks(np.arange(values.shape[1]))
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
        column=int(np.floor(event.xdata+.5)); row=int(np.floor(event.ydata+.5))
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
        acquired=int(np.count_nonzero(np.any(np.isfinite(finite),axis=0)))
        total=int(finite.shape[1])
        ax=self._heatmap(
            self.x_plot,
            finite,
            f"Evolving ORM heatmap — {acquired}/{total} corrector columns acquired",
            "m",
            row_devices=bpms,
            column_devices=correctors,
            row_axis_label=f"ORM row index (0–{len(self.selected_devices)-1}: X; {len(self.selected_devices)}–{2*len(self.selected_devices)-1}: Y)",
        )
        nb=len(self.selected_devices)
        if nh and len(self.selected_vcorrectors):ax.axvline(nh-.5,color="white",linewidth=1.6,linestyle="--")
        ax.axhline(nb-.5,color="white",linewidth=1.6,linestyle="--")
        if acquired < total:
            ax.text(.995,.015,"Gray = not acquired",transform=ax.transAxes,ha="right",va="bottom",
                    fontsize=8,color="white",bbox={"facecolor":"#303548","edgecolor":"none","alpha":.85,"pad":2})
        self.x_plot.apply_theme()
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
        full_ax=self._heatmap(self.x_plot,matrix,"Full orbit response matrix",unit,row_devices=bpm_rows,column_devices=corrector_columns,row_axis_label=f"ORM row index (0–{nb-1}: X; {nb}–{2*nb-1}: Y)"); self._heatmap(self.y_plot,matrix[:nb,:nh],"H corrector → X BPM",unit,row_devices=x_rows,column_devices=h_columns); self._heatmap(self.mean_x_plot,matrix[:nb,nh:],"V corrector → X BPM (cross-plane)",unit,row_devices=x_rows,column_devices=v_columns); self._heatmap(self.mean_y_plot,matrix[nb:,:nh],"H corrector → Y BPM (cross-plane)",unit,row_devices=y_rows,column_devices=h_columns); self._heatmap(self.orm_vv_plot,matrix[nb:,nh:],"V corrector → Y BPM",unit,row_devices=y_rows,column_devices=v_columns)
        if self.orm_view_mode.currentData()=="2d":
            if nh and len(result.vertical_correctors):full_ax.axvline(nh-.5,color="white",linewidth=2,linestyle="--")
            full_ax.axhline(nb-.5,color="white",linewidth=2,linestyle="--")
            for x,y,label in (((nh-1)/2,(nb-1)/2,"H → X"),(nh+(len(result.vertical_correctors)-1)/2,(nb-1)/2,"V → X"),((nh-1)/2,nb+(nb-1)/2,"H → Y"),(nh+(len(result.vertical_correctors)-1)/2,nb+(nb-1)/2,"V → Y")):
                if x>=0:full_ax.text(x,y,label,ha="center",va="center",color="black",bbox={"facecolor":"white","edgecolor":"none","alpha":.7,"pad":2},fontsize=8,zorder=4)
        full_ax.set_xlabel("Corrector index"); self.x_plot.apply_theme()
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
        if result.orbit_correction_diagnostics:
            self.orm_restoration_status.setText("✓ NOMINAL RESTORES VERIFIED; ORBIT CORRECTIONS RETAINED" if restored else "✗ CORRECTOR RESTORATION FAILED")
            summary_html += "<br>Orbit correction applied after each column; corrected strengths remain applied. Restore checks refer to nominal restoration before correction."
        self.orm_measurement_summary.setText(summary_html); self.restoration_label.setText("<b>Restore checks</b><br>"+" &nbsp; • &nbsp; ".join(f"{escape(device.name)}: {middle} / {final}" for device,middle,final in zip(result.correctors,result.intermediate_restoration_status,result.restoration_status)))
        if self.compare_reference.isChecked() and self.reference_model is not None:
            try:
                model_matrix=getattr(self,"_model_orm_cache",None)
                if model_matrix is None:model_matrix=model_orm(self.reference_model,result.bpms,result.horizontal_correctors,result.vertical_correctors,result.effective_kicks_rad[:nh],result.effective_kicks_rad[nh:],scaled=result.scaled,**self._model_mapping_kwargs())
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
        correction_text = ""
        if result.orbit_correction_diagnostics:
            correction = result.orbit_correction_diagnostics[index]
            correction_text = (f"Orbit correction after nominal restore: {correction['status']}; "
                               f"correctors: {', '.join(correction['correctors'])}; "
                               f"strengths before: {correction['before_native']}; "
                               f"retained strengths: {correction['after_native']}; units: {correction['units']}. "
                               "Final orbit return is measured after correction.\n")
        return (f"Corrector transaction — {device.name} ({device.plane})\n"
                f"{correction_text}"
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
                mapping=self._model_mapping_kwargs()
                model_x,model_y=model_dispersion(self.reference_model,result.devices,bpm_ordinals=mapping.get("bpm_ordinals"))
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
        if not hasattr(self,"tabs") or self.tabs.count()<6:return
        machine_done=self.connection_verified
        bpms_done=bool(self.selected_devices) and bool(self.device_selection_reviewed.isChecked())
        review_done=self.saved_session_path is not None and not self._session_dirty
        roles=set()
        if self.saved_session_path is not None:
            try:roles={entry.role for entry in load_session(self.saved_session_path,validate_files=False).files}
            except Exception:pass
        completed={kind:False for kind in ("bpm_noise","dispersion","orm")}
        for kind in completed:
            if self._measurement_results.get(kind) is not None:completed[kind]=True; continue
            path=self._measurement_result_paths.get(kind)
            if kind in roles and path is not None:
                try:completed[kind]=validate_measurement_file(path)["kind"]==kind
                except Exception:completed[kind]=False
        labels=((machine_done,"Machine"),(bpms_done,"Devices"),(completed["bpm_noise"],"BPM Noise"),
                (completed["dispersion"],"RF Dispersion"),(completed["orm"],"ORM"),(review_done,"Review && Save"))
        for index,(done,label) in enumerate(labels): self.tabs.setTabText(index,("✓ " if done else "")+label)
        if self.thread is None and hasattr(self,"measurement_type"):
            for kind,index in self.measurement_tab_indices.items():
                item=self.measurement_type.model().item(self.measurement_type.findData(kind))
                self.tabs.setTabEnabled(index,item is None or item.isEnabled())

    def validate_saved(self):
        if self.saved_session_path is None:return
        session=load_session(self.saved_session_path)
        measurement=self.saved_measurement_path or next(iter(session.resolve(self.saved_session_path).values()),None)
        if measurement is None:return
        info=validate_measurement_file(measurement)
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
        # Acquisition initially creates a lightweight measurement manifest.
        # Before handing it to FIT, persist the active Measure project,
        # selections, lattice provenance, and complete Machine Mapping into
        # that same manifest.  HDF5 files are referenced in place and are not
        # copied or rewritten.
        if hasattr(self,"_persist_current_full_session"):
            try:self._persist_current_full_session()
            except Exception as exc:
                QMessageBox.warning(self,"Cannot prepare session for FIT",str(exc)); return
        session=load_session(self.saved_session_path)
        try:ok,detail=launch_suite_application("fit","--measurement-session",str(self.saved_session_path))
        except Exception as exc:QMessageBox.warning(self,"Cannot open pyLOCO Fit",str(exc)); return
        if not ok:QMessageBox.warning(self,"Cannot open pyLOCO Fit",detail); return
        missing=", ".join(session.missing_roles) or "none"; self.statusBar().showMessage(f"Opening session in pyLOCO Fit ({detail}); missing roles: {missing}")

    def _collect_project(self):
        self.project.metadata["pyaml_options"] = {**{key: editor.text().strip() for key, editor in self.pyaml_fields.items()}, "orbit_correction_enabled": self.pyaml_orbit_correction.isChecked(), "requested_magnet_writes": self.pyaml_magnet_writes.isChecked(), "requested_rf_writes": self.pyaml_rf_writes.isChecked()}
        self.project.metadata["device_selection_reviewed"]=bool(self.device_selection_reviewed.isChecked())
        self._store_active_selection(); self._store_measurement_view_state(self._active_selection_kind)
        reverse={value:key for key,value in getattr(self,"measurement_tab_indices",{}).items()}
        active_index=self.tabs.currentIndex()
        workspace=reverse.get(active_index,"machine" if active_index==0 else "devices" if active_index==1 else "review" if active_index==getattr(self,"review_tab_index",5) else self._active_selection_kind)
        h=self.corrector_selection_widgets["hcor"]; v=self.corrector_selection_widgets["vcor"]
        identity,profile=self._machine_identity(); profile_meta={} if profile is None else {"machine":profile.machine,"profile":profile.scenario,"profile_key":profile.key,"profile_manifest":str(profile.manifest_path),"lattice_file":str(profile.resolve("lattice_file")),"provenance":profile.configuration.get("provenance",{}),"random_seed":profile.configuration.get("random_seed")}
        active_settling=self.orm_settling_delay.value() if self.measurement_type.currentData()=="orm" else self.settling_delay.value()
        return MeasureProject(measurement_type=self.measurement_type.currentData(),measurement_name=self.measurement_name.text(),measurement_label=self.measurement_label.text(),operator_comments=self.comments.toPlainText(),adapter=self.adapter_combo.currentText(),pysc_profile=self.pysc_profile_combo.currentData(),bpm_selection_method=self.selection_method.currentData(),bpm_manual=self.manual_input.text(),bpm_names_file=self.names_file.text(),excluded_bpm_positions=self.bpm_exclusions.text(),hcor_selection_method=h["method"].currentData(),hcor_manual=h["manual"].text(),hcor_names_file=h["file"].text(),vcor_selection_method=v["method"].currentData(),vcor_manual=v["manual"].text(),vcor_names_file=v["file"].text(),excluded_hcor_positions=h["exclusion"].text(),excluded_vcor_positions=v["exclusion"].text(),measurement_selections=self._selection_states,readings=self.readings.value(),bpm_noise_ddof=int(self.bpm_noise_convention.currentData()),delay_seconds=self.delay.value(),settling_delay_seconds=active_settling,verify_restored_orbit=self.verify_restored_orbit.isChecked(),rf_control_mode=self.rf_control_mode.currentData(),nominal_rf_hz=self._nominal_rf_value(),nominal_rf_source="manual",rf_step_hz=self.rf_step.value(),rf_restoration_tolerance_hz=self.rf_restoration_tolerance.value(),dispersion_direction=self.dispersion_direction.currentData(),orm_direction=self.orm_direction.currentData(),orm_kick_mode=self.orm_kick_mode.currentData(),orm_horizontal_kick_rad=self.orm_hkick.value()*1e-6,orm_vertical_kick_rad=self.orm_vkick.value()*1e-6,orm_kick_file=self.orm_kick_file.text(),orm_scaled=False,reference_model_path=self.reference_model_path.text(),reference_model_source="automatic" if self.adapter_combo.currentData()=="pysc" else "user",reference_model_mat_variable=self.reference_model.mat_variable if self.reference_model is not None and self.reference_model.mat_variable else "",compare_with_reference_model=self.compare_reference.isChecked(),output_directory=self.output_directory.text(),theme=self.project.theme,metadata={**self.project.metadata,"machine_identity":identity,"profile":profile_meta,"orm_normalized_analysis":self.orm_normalized_analysis.isChecked(),"measurement_view_states":deepcopy(self._measurement_view_states),"active_workspace":workspace})

    def _load_project_widgets(self):
        # Restore requested capabilities, never connection/write authorization.
        # Test Connection still presents the target and requires confirmation.
        saved_pyaml=self.project.metadata.get("pyaml_options", {})
        for widget,value in ((self.pyaml_magnet_writes,bool(saved_pyaml.get("requested_magnet_writes",False))),(self.pyaml_rf_writes,bool(saved_pyaml.get("requested_rf_writes",False)))):
            widget.blockSignals(True); widget.setChecked(value); widget.blockSignals(False)
        self.pyaml_orbit_correction.setChecked(bool(saved_pyaml.get("orbit_correction_enabled", False)))
        self.device_selection_reviewed.blockSignals(True); self.device_selection_reviewed.setChecked(bool(self.project.metadata.get("device_selection_reviewed",False))); self.device_selection_reviewed.blockSignals(False)
        self.pyaml_simulator_identity_status.setText("Project opened — Test connection to confirm pyAML capabilities and restoration evidence.")
        for key, editor in self.pyaml_fields.items():
            editor.setText(self.project.metadata.get("pyaml_options", {}).get(key, {"mode": "live", "bpm_group": "BPM", "hcor_group": "HCorr", "vcor_group": "VCorr", "rf_name": "RF"}.get(key, "")))
        self.pysc_profile_combo.setCurrentIndex(max(0,self.pysc_profile_combo.findData(self.project.pysc_profile)))
        self.adapter_combo.setCurrentIndex(max(0,self.adapter_combo.findText(self.project.adapter)))
        if self.adapter_combo.currentData() == "pyaml":
            self._invalidate_pyaml()
        p=self.project; self._selection_states=p.measurement_selections; self._measurement_view_states=deepcopy(p.metadata.get("measurement_view_states",{})); self._active_selection_kind=p.measurement_type; self.measurement_type.blockSignals(True); self.measurement_type.setCurrentIndex(max(0,self.measurement_type.findData(p.measurement_type))); self.measurement_type.blockSignals(False); self._restore_selection(p.measurement_type); self.measurement_name.setText(p.measurement_name); self.measurement_label.setText(p.measurement_label); self.comments.setPlainText(p.operator_comments); self.readings.setValue(p.readings); self.bpm_noise_convention.setCurrentIndex(max(0,self.bpm_noise_convention.findData(p.bpm_noise_ddof))); self.delay.setValue(p.delay_seconds); self.settling_delay.setValue(p.settling_delay_seconds); self.orm_settling_delay.setValue(p.settling_delay_seconds); self.rf_restoration_tolerance.setValue(p.rf_restoration_tolerance_hz); self.verify_restored_orbit.setChecked(p.verify_restored_orbit); self.nominal_rf.setText("" if p.nominal_rf_hz is None else f"{p.nominal_rf_hz:g}"); self.rf_step.setValue(p.rf_step_hz); self.dispersion_direction.setCurrentIndex(max(0,self.dispersion_direction.findData(p.dispersion_direction))); self.orm_direction.setCurrentIndex(max(0,self.orm_direction.findData(p.orm_direction))); self.orm_kick_mode.setCurrentIndex(max(0,self.orm_kick_mode.findData(p.orm_kick_mode))); self.orm_hkick.setValue(p.orm_horizontal_kick_rad*1e6); self.orm_vkick.setValue(p.orm_vertical_kick_rad*1e6); self.orm_kick_file.setText(p.orm_kick_file); self.orm_normalized_analysis.setChecked(bool(p.orm_scaled or p.metadata.get("orm_normalized_analysis",False))); self._measurement_view_states[p.measurement_type]=self._measurement_view_snapshot()
        self.reference_model_path.setText(p.reference_model_path); self.compare_reference.setChecked(p.compare_with_reference_model)
        if p.reference_model_path:self._load_reference_model(p.reference_model_path,"Project reference model",p.reference_model_mat_variable or None)
        self.output_directory.setText(p.output_directory); self.apply_theme(p.theme)
        # Project hydration may occur while pyAML is intentionally
        # disconnected and its live inventory is empty.  Rebuilding the table
        # must not write that temporary empty preview back into the saved BPM
        # selection (and consequently invalidate its mapping fingerprint).
        self._loading_selection_state=True
        try:self.refresh_preview()
        finally:self._loading_selection_state=False
        if p.measurement_type=="orm":self._fresh_orm_defaults_applied=True
        self._measurement_type_changed(); self._update_machine_identity()
        workspace=p.metadata.get("active_workspace",p.measurement_type)
        target={"machine":0,"devices":1,"review":getattr(self,"review_tab_index",5),**self.measurement_tab_indices}.get(workspace,self.measurement_tab_indices[p.measurement_type])
        self.tabs.setCurrentIndex(target)

    def new_project(self):
        self.project = MeasureProject()
        self.project_path = None
        self._load_project_widgets()
        self._reset_measurement_display()
        self._fresh_measurement_session=True; self._pending_full_bundle=None
        self._session_dirty=True; self._update_session_title()
    def open_project(self):
        path=QFileDialog.getOpenFileName(self,"Open pyLOCO Measure project","","pyLOCO Measure (*.pyloco-measure.json *.json)")[0]
        if not path:return
        try:
            payload=json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError,UnicodeError,json.JSONDecodeError) as exc:
            QMessageBox.warning(self,"Cannot open Measure project",f"The selected file is not a readable pyLOCO Measure project:\n{exc}"); return
        if isinstance(payload,dict) and payload.get("file_type")==SESSION_FILE_TYPE:
            QMessageBox.information(self,"Measurement session selected","This file is a measurement-session manifest, not a Measure project. Use Open to restore the full measurement session and its available measurements."); return
        try:self.project=load_measure_project(path)
        except Exception as exc:QMessageBox.warning(self,"Cannot open Measure project",str(exc)); return
        self.project_path=Path(path).resolve(); self._load_project_widgets(); self.statusBar().showMessage(f"Opened {path}")
    def save_project(self):
        if self.project_path is None: return self.save_project_as()
        self.project=self._collect_project(); save_measure_project(self.project_path,self.project); self.statusBar().showMessage(f"Saved {self.project_path}")
    def save_project_as(self):
        path=QFileDialog.getSaveFileName(self,"Save pyLOCO Measure project",f"{self.measurement_type.currentData().replace('_','-')}.pyloco-measure.json","pyLOCO Measure (*.pyloco-measure.json)")[0]
        if not path:return
        self.project_path=Path(path).resolve(); self.save_project()

    def _update_session_title(self):
        name=self.saved_session_path.parent.name if self.saved_session_path is not None else "Untitled Measure Session"
        self.setWindowTitle(f"pyLOCO Measure — {name}{' *' if self._session_dirty else ''}")

    def save_active_session(self):
        suggested=(self.saved_session_path if self.saved_session_path is not None else Path(self.output_directory.text()).expanduser()/"measurement-session.pyloco-session.json")
        target_text=QFileDialog.getSaveFileName(self,"Save Full Measurement Session — choose destination",str(suggested),"Measurement session (*.pyloco-session.json)")[0]
        if not target_text:return False
        target=Path(target_text).expanduser().resolve(); current=self.saved_session_path
        if current is not None and target==current:
            if QMessageBox.question(self,"Save to current session?",f"Update the session metadata at the current location?\n{target}\n\nExisting HDF5 measurements will not be rewritten.",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Yes:return False
        elif target.exists():
            QMessageBox.warning(self,"Session already exists","Save will not silently overwrite another session. Choose a new path, or explicitly choose the current session location."); return False
        elif current is not None:
            source=load_session(current,validate_files=True); resolved=source.resolve(current); entries=[]
            try:
                for entry in source.files:entries.append(SessionFile(entry.role,resolved[entry.role].relative_to(target.parent).as_posix(),dict(entry.metadata)))
            except ValueError:
                QMessageBox.information(self,"Use Export Portable Session","The selected destination cannot reference the existing HDF5 files relatively. Use File → Export Portable Session… when dependencies must be copied."); return False
            save_session(target,MeasurementSession(source.session_id,tuple(entries),deepcopy(source.metadata)))
            self.saved_session_path=target
        created_manifest=False
        if self.saved_session_path is None:
            try:
                session_id=f"measure-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
                save_session(target,MeasurementSession(session_id,(),{"created":datetime.now().astimezone().isoformat()}))
            except Exception as exc:QMessageBox.warning(self,"Cannot create session",str(exc)); return False
            self.saved_session_path=target; self._fresh_measurement_session=False; created_manifest=True
        try:bundle=self._persist_current_full_session()
        except Exception as exc:
            if created_manifest:
                self.saved_session_path.unlink(missing_ok=True); self.saved_session_path=None; self._fresh_measurement_session=True
            QMessageBox.warning(self,"Cannot save session",str(exc)); return False
        self.statusBar().showMessage(f"Saved session workspace without copying measurement HDF5 files: {bundle.manifest_path}")
        return True

    def _persist_current_full_session(self):
        """Persist the active handoff state without modifying measurement data."""
        if self.saved_session_path is None:
            raise ValueError("No active measurement session is available to save.")
        serialized_mapping=(self._ensure_complete_machine_mapping()
            if self.machine_mapping_document is not None or self.measurement_device_alignment is not None else None)
        bundle=save_full_session_state(self.saved_session_path,project=self._collect_project(),
            reference_lattice=self.reference_model.path if self.reference_model is not None else None,
            reference_mat_variable=self.reference_model.mat_variable if self.reference_model is not None else "",
            reference_element_count=len(self.reference_model.ring) if self.reference_model is not None else None,
            mapping=serialized_mapping)
        self.project=bundle.project; self.project_path=bundle.project_path; self._pending_full_bundle=bundle
        self._measurement_result_paths={role:bundle.measurement_paths.get(role) for role in ("bpm_noise","dispersion","orm")}
        self._session_dirty=False; self._update_session_title(); self._refresh_review_page(); self._update_workflow_tabs()
        return bundle

    def save_session_as(self):
        return self.save_active_session()

    @staticmethod
    def _measurement_identities(path):
        import h5py
        def strings(dataset):return [item.decode() if isinstance(item,bytes) else str(item) for item in dataset[:]]
        with h5py.File(path,"r") as handle:
            result={"bpm":strings(handle["names/bpms"])} if "names/bpms" in handle else {"bpm":[]}
            if "names/horizontal_correctors" in handle:result["horizontal_corrector"]=strings(handle["names/horizontal_correctors"])
            if "names/vertical_correctors" in handle:result["vertical_corrector"]=strings(handle["names/vertical_correctors"])
        return result

    def _load_measurement_role(self,role,*,replace=False):
        path_text=QFileDialog.getOpenFileName(self,f"{'Replace' if replace else 'Load'} {role.replace('_',' ').title()}","","pyLOCO measurement (*.h5 *.hdf5);;All files (*)")[0]
        if not path_text:return False
        source=Path(path_text).expanduser().resolve()
        try:
            info=validate_measurement_file(source)
            if info["kind"]!=role:raise ValueError(f"Selected file contains {info['kind']}, not {role}.")
            incoming=self._measurement_identities(source)
        except Exception as exc:QMessageBox.warning(self,"Invalid measurement file",str(exc)); return False
        if self.saved_session_path is None:
            self.saved_session_path=source.parent/"measurement-session.pyloco-session.json"
            if self.saved_session_path.exists():
                QMessageBox.warning(self,"Session manifest already exists","Open the existing session first instead of overwriting its manifest."); self.saved_session_path=None; return False
            session=MeasurementSession(source.parent.name or "measure-session",())
        else:session=load_session(self.saved_session_path,validate_files=True)
        entries={entry.role:entry for entry in session.files}
        if role in entries and not replace:
            QMessageBox.information(self,"Measurement already available",f"{role.replace('_',' ').title()} is already associated with this session. Use Replace Measurement… to change it."); return False
        incompatible=[]
        for existing_role,existing_path in session.resolve(self.saved_session_path).items():
            if existing_role==role:continue
            existing=self._measurement_identities(existing_path)
            if existing.get("bpm") and incoming.get("bpm") and existing["bpm"]!=incoming["bpm"]:incompatible.append(f"BPM identity/order differs from {existing_role}")
        if incompatible and QMessageBox.question(self,"Measurement inventory differs","\n".join(incompatible)+"\n\nAssociate this file anyway and invalidate device mapping/FIT readiness?",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Yes:return False
        root=self.saved_session_path.parent
        try:relative=source.relative_to(root).as_posix(); target=source
        except ValueError:
            target=root/"measurements"/f"{session_file_sha256(source)[:12]}-{source.name}"; target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            relative=target.relative_to(root).as_posix()
        entries[role]=SessionFile(role,relative,{"sha256":session_file_sha256(target),"size":target.stat().st_size,"imported":True})
        metadata=deepcopy(session.metadata); full=metadata.get("full_measure_session")
        if isinstance(full,dict):full["roles"]={name:name in entries for name in ("bpm_noise","dispersion","orm")}
        updated=MeasurementSession(session.session_id,tuple(entries[name] for name in ("bpm_noise","dispersion","orm") if name in entries),metadata)
        save_session(self.saved_session_path,updated); self.saved_measurement_path=target; self._fresh_measurement_session=False; self.result=None
        self._measurement_results[role]=None; self._measurement_result_paths[role]=target
        if incompatible or role=="orm":self.measurement_device_alignment=None; self._invalidate_device_alignment("Measurement inventory changed — verify exact device alignment again.")
        self._session_dirty=True; self._update_session_title(); self._refresh_review_page(); self._update_workflow_tabs(); return True

    def _remove_measurement_role(self,role):
        if self.saved_session_path is None:return False
        session=load_session(self.saved_session_path,validate_files=True); entries=tuple(entry for entry in session.files if entry.role!=role)
        if len(entries)==len(session.files):return False
        if QMessageBox.question(self,"Remove measurement from session?",f"Remove the {role.replace('_',' ')} reference from this session? The HDF5 file will not be deleted.",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)!=QMessageBox.Yes:return False
        metadata=deepcopy(session.metadata); full=metadata.get("full_measure_session"); present={entry.role for entry in entries}
        if isinstance(full,dict):full["roles"]={name:name in present for name in ("bpm_noise","dispersion","orm")}
        save_session(self.saved_session_path,MeasurementSession(session.session_id,entries,metadata))
        self.saved_measurement_path=next(iter(MeasurementSession(session.session_id,entries).resolve(self.saved_session_path).values()),None)
        self._measurement_results[role]=None; self._measurement_result_paths[role]=None
        if role=="orm":self.measurement_device_alignment=None; self._invalidate_device_alignment("ORM removed — device alignment and FIT readiness are invalid.")
        self._session_dirty=True; self._update_session_title(); self._refresh_review_page(); self._update_workflow_tabs(); return True

    def export_portable_session(self):
        if self.saved_session_path is None:
            QMessageBox.warning(self,"Cannot export portable session","Acquire or open at least one measurement first."); return
        suggested=f"measurement-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pyloco-measure-session"
        destination=QFileDialog.getSaveFileName(self,"Export Portable Measurement Session",suggested,"Full Measure session (*.pyloco-measure-session)")[0]
        if not destination:return
        include_yaml=False
        config=str(self.pyaml_fields["config_path"].text()).strip() if self.adapter_combo.currentData()=="pyaml" else ""
        if config:
            include_yaml=QMessageBox.question(self,"Include approved pyAML configuration?","By default the bundle records the external pyAML configuration path and checksum only. Include this file only if it is approved, contains no credentials or private endpoints, and may be redistributed.",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes
        try:
            serialized_mapping=(self._ensure_complete_machine_mapping()
                if self.machine_mapping_document is not None or self.measurement_device_alignment is not None else None)
            bundle=save_full_session_bundle(destination,project=self._collect_project(),session_manifest=self.saved_session_path,
                reference_lattice=self.reference_model.path if self.reference_model is not None else None,
                reference_mat_variable=self.reference_model.mat_variable if self.reference_model is not None else "",
                reference_element_count=len(self.reference_model.ring) if self.reference_model is not None else None,
                mapping=serialized_mapping,include_pyaml_configuration=include_yaml)
        except Exception as exc:QMessageBox.warning(self,"Cannot export portable session",str(exc)); return
        message=f"Exported portable measurement session: {bundle.root}"
        if bundle.warnings:message+=" — dependencies require attention: "+" ".join(bundle.warnings)
        self.statusBar().showMessage(message)

    def open_full_measurement_session_folder(self):
        directory=QFileDialog.getExistingDirectory(self,"Open Portable Measurement Session Folder","")
        if directory:self.open_full_measurement_session(directory)

    def _clear_session_role_caches(self):
        self.result=None; self.saved_measurement_path=None
        self._measurement_results={"bpm_noise":None,"dispersion":None,"orm":None}
        self._measurement_result_paths={"bpm_noise":None,"dispersion":None,"orm":None}
        self._measurement_restore_status={role:{"present":False,"validated":False,"restored":False,"error":""} for role in ("bpm_noise","dispersion","orm")}

    def _restore_session_measurements(self, paths):
        """Validate and reconstruct every available result independently."""
        for role in ("bpm_noise","dispersion","orm"):
            path=paths.get(role)
            status={"present":path is not None,"validated":False,"restored":False,"error":""}
            if path is not None:
                try:
                    details=validate_measurement_file(path)
                    if details["kind"]!=role:raise ValueError(f"file contains {details['kind']}, not {role}")
                    status["validated"]=True
                    self._measurement_results[role]=self._restore_saved_result(role,path)
                    self._measurement_result_paths[role]=Path(path)
                    status["restored"]=True
                except Exception as exc:
                    status["error"]=str(exc)
            self._measurement_restore_status[role]=status

    def _restore_offline_measurement_inventory(self):
        """Restore HDF5 identity/order for disconnected, read-only review."""
        available={role:self._measurement_identities(path) for role,path in self._measurement_result_paths.items() if path is not None}
        bpm_orders=[tuple(item.get("bpm",())) for item in available.values() if item.get("bpm")]
        if bpm_orders and any(order!=bpm_orders[0] for order in bpm_orders[1:]):
            raise ValueError("BPM identity/order differs between saved measurement files")
        orm=available.get("orm",{})
        bpm_names=bpm_orders[0] if bpm_orders else ()
        h_names=tuple(orm.get("horizontal_corrector",()))
        v_names=tuple(orm.get("vertical_corrector",()))
        self.devices=tuple(BpmDevice(name,"saved HDF5 X","saved HDF5 Y") for name in bpm_names)
        self.horizontal_correctors=tuple(CorrectorDevice(name,"saved HDF5 setpoint","saved HDF5 adapter value","H") for name in h_names)
        self.vertical_correctors=tuple(CorrectorDevice(name,"saved HDF5 setpoint","saved HDF5 adapter value","V") for name in v_names)
        self.selected_devices=self.devices; self.selected_hcorrectors=self.horizontal_correctors; self.selected_vcorrectors=self.vertical_correctors
        for widget_name,items in (("bpm_table",self.devices),("hcor_table",self.horizontal_correctors),("vcor_table",self.vertical_correctors)):
            widget=getattr(self,widget_name,None)
            if widget is not None and hasattr(widget,"devices"):widget.devices=items

    def open_full_measurement_session(self,selected_path=None):
        if isinstance(selected_path,bool):selected_path=None
        filename=str(selected_path or QFileDialog.getOpenFileName(self,"Open Full Measurement Session","","Measurement session (*.pyloco-session.json *.json);;All files (*)")[0])
        if not filename:return
        selected=Path(filename).expanduser().resolve(); manifest=selected/MANIFEST_NAME if selected.is_dir() else selected
        try:session_probe=load_session(manifest,validate_files=False)
        except Exception as exc:QMessageBox.warning(self,"Cannot open measurement session",str(exc)); return
        is_modern="full_measure_session" in session_probe.metadata
        if not is_modern:
            try:session=load_session(manifest,validate_files=True); paths=session.resolve(manifest)
            except Exception as exc:QMessageBox.warning(self,"Cannot open legacy measurement session",str(exc)); return
            # Legacy manifests contain measurements only. Clear prior full-session
            # configuration so it cannot be mistaken for legacy provenance.
            self.project=MeasureProject(); self.project_path=None; self._pending_full_bundle=None; self._load_project_widgets(); self._clear_session_role_caches()
            self.saved_session_path=manifest; self.saved_measurement_path=next(iter(paths.values()),None); self._fresh_measurement_session=False
            self._measurement_result_paths.update(paths)
            self._restore_session_measurements(paths)
            self._restore_offline_measurement_inventory()
            self._measurement_type_changed()
            self._session_dirty=False; self._update_session_title(); self._refresh_review_page(); self._update_workflow_tabs()
            QMessageBox.information(self,"Older measurement session loaded","Measurement files were restored for review and FIT, but this older manifest contains no Measure project, lattice, selections or mapping. Those historical settings cannot be recovered; load authoritative dependencies manually before continuing acquisition."); return
        try:bundle=open_full_session_bundle(manifest)
        except Exception as exc:
            # Presence of the modern marker is authoritative. Integrity or
            # schema failures must never be reinterpreted as a legacy session.
            QMessageBox.warning(self,"Cannot open full session",f"The modern full session failed integrity validation and was not opened:\n{exc}"); return
        self._clear_session_role_caches()
        self.project=bundle.project; self.project_path=bundle.project_path; self._pending_full_bundle=bundle; self._load_project_widgets()
        # Requested write preferences were restored above, but authorization is
        # always absent until the operator confirms a new Test Connection.
        self.connection_verified=False
        self.saved_session_path=bundle.manifest_path; self.saved_measurement_path=next(iter(bundle.measurement_paths.values()),None); self._fresh_measurement_session=False
        self._measurement_result_paths.update(bundle.measurement_paths)
        self._restore_session_measurements(bundle.measurement_paths)
        open_warnings=list(bundle.warnings)
        try:
            self._restore_offline_measurement_inventory()
        except Exception as exc:
            open_warnings.append(f"Measurement inventories are incompatible: {exc}")
        self._selection_states=deepcopy(bundle.project.measurement_selections)
        self._active_selection_kind=self.measurement_type.currentData()
        self._restore_selection(self._active_selection_kind)
        self._session_dirty=False; self._update_session_title()
        self.measurement_device_alignment=None
        self.machine_mapping_document=bundle.mapping if bundle.mapping is not None and is_machine_mapping(bundle.mapping) else None
        mapping_note=self._restore_offline_bundle_mapping(bundle)
        self._measurement_type_changed()
        # Widget synchronization and offline mapping reconstruction are part of
        # opening the saved session, not operator edits.
        self._session_dirty=False; self._update_session_title()
        self._refresh_review_page(); self._update_workflow_tabs()
        state="Loaded for disconnected review. "+mapping_note
        if open_warnings:state+=". "+" ".join(open_warnings)
        self.statusBar().showMessage(state); QMessageBox.information(self,"Full measurement session opened",state+"\nNo machine connection was opened and all write authorizations are disabled.")

    def _restore_offline_bundle_mapping(self,bundle):
        """Validate saved mapping, or rebuild it from checksummed offline inputs."""
        if self.reference_model is None:
            self._invalidate_device_alignment("Session measurements restored; reference lattice is unavailable for mapping validation.")
            return "Device mapping requires an authoritative reference lattice."
        configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
        if self.adapter_combo.currentData()=="pyaml" and not configuration.is_file():
            self._invalidate_device_alignment("Session measurements restored; saved pyAML configuration is unavailable for mapping validation.")
            return "Device mapping requires the saved pyAML configuration."
        inventory=self._alignment_inventory()
        try:
            source=bundle.mapping
            if source is not None:
                projected=measurement_projection(source) if is_machine_mapping(source) else source
                if is_machine_mapping(source):
                    wanted={(role,name) for role,names in inventory.items() for name in names}
                    projected=deepcopy(projected)
                    projected["mappings"]=[row for row in projected.get("mappings",())
                        if (row.get("role"),row.get("measured_name")) in wanted]
                    projected["inventory"]={"sha256":inventory_fingerprint(inventory),
                        "counts":{role:len(names) for role,names in inventory.items()}}
                checked=validate_mapping(projected,self.reference_model.ring,inventory,
                    lattice_path=self.reference_model.path,configuration_path=configuration if configuration.is_file() else None,
                    require_provenance=True)
                origin="Saved mapping restored and validated offline"
            else:
                checked=find_exact_device_mappings(self.reference_model.ring,inventory,
                    lattice_path=self.reference_model.path,configuration_path=configuration if configuration.is_file() else None,
                    provenance="Reconstructed offline from saved HDF5 identities and checksummed session dependencies")
                origin="No mapping saved; exact mapping reconstructed and validated offline"
            counts=mapping_counts(checked)
            if counts["mapped"]!=counts["total"]:
                raise ValueError(f"mapping is incomplete ({counts['mapped']}/{counts['total']} identities mapped)")
            self.measurement_device_alignment=checked
            self.machine_mapping_document=(deepcopy(source) if source is not None and is_machine_mapping(source)
                else merge_measurement(None,checked))
            self.device_alignment_status.setText(f"{origin} — {counts['mapped']}/{counts['total']} identities.")
            self.alignment_details_button.setEnabled(True); self.save_mapping_button.setEnabled(True)
            self._update_reference_comparison_enabled()
            return origin+". A future live connection will revalidate the connected inventory."
        except Exception as exc:
            self._invalidate_device_alignment(f"Measurements restored; mapping needs validation: {exc}")
            return f"Device mapping requires validation: {exc}"

    def _restore_full_bundle_inventory(self,bundle):
        inventories={"bpm":self.devices,"hcor":self.horizontal_correctors,"vcor":self.vertical_correctors}; state=bundle.project.measurement_selections; missing=[]
        for measurement_kind,measurement in state.items():
            # Corrector inventory is scientifically relevant only to ORM.
            # Dispersion/BPM-noise projects may retain harmless default
            # corrector selections that must not invalidate reconnection.
            keys=("bpm","hcor","vcor") if measurement_kind=="orm" else ("bpm",)
            for key in keys:
                devices=inventories[key]
                available={device.name for device in devices}; missing.extend(name for name in measurement[key].get("resolved_names",()) if name not in available)
        if missing:raise ValueError("missing identities: "+", ".join(dict.fromkeys(missing)))
        self._selection_states=deepcopy(state); self._active_selection_kind=self.measurement_type.currentData(); self._restore_selection(self._active_selection_kind)
        if self.measurement_device_alignment is not None and self.reference_model is not None:
            configuration=Path(self.pyaml_fields["config_path"].text()).expanduser()
            if self.adapter_combo.currentData()=="pyaml" and not configuration.is_file():raise ValueError("the saved pyAML configuration is unavailable; mapping cannot be revalidated")
            orm=state["orm"]; inventory={"bpm":list(orm["bpm"].get("resolved_names",())),"horizontal_corrector":list(orm["hcor"].get("resolved_names",())),"vertical_corrector":list(orm["vcor"].get("resolved_names",()))}
            checked=validate_mapping(self.measurement_device_alignment,self.reference_model.ring,inventory,lattice_path=self.reference_model.path,configuration_path=configuration if configuration.is_file() else None,require_provenance=True)
            counts=mapping_counts(checked)
            if counts["mapped"]!=counts["total"]:raise ValueError("saved device mapping is incomplete")
            self.measurement_device_alignment=checked
            if not is_machine_mapping(self.machine_mapping_document or {}):
                self.machine_mapping_document=merge_measurement(None,checked)
            self.device_alignment_status.setText(f"Restored and revalidated — {counts['mapped']}/{counts['total']} exact identities; connected inventory matches the saved session.")
            self.alignment_details_button.setEnabled(True); self.save_mapping_button.setEnabled(True); self._update_reference_comparison_enabled()

    def apply_theme(self,key):
        self.project.theme=key; app=QApplication.instance(); apply_application_theme(app,theme_for_key(key),"teal")
        self.setStyleSheet(measure_stylesheet(key)); self.theme_button.setText("☀ Light" if key=="dark" else "🌙 Dark")
        for plot in (getattr(self,"live_plot",None),getattr(self,"x_plot",None),getattr(self,"y_plot",None),getattr(self,"mean_x_plot",None),getattr(self,"mean_y_plot",None),getattr(self,"rf_shift_x_plot",None),getattr(self,"rf_shift_y_plot",None),getattr(self,"orm_vv_plot",None),getattr(self,"orm_column_plot",None),getattr(self,"orm_kick_plot",None),getattr(self,"orm_model_plot",None),getattr(self,"orm_difference_plot",None)):
            if plot is not None: plot.apply_theme()
    def toggle_theme(self): self.apply_theme("light" if self.project.theme=="dark" else "dark")
