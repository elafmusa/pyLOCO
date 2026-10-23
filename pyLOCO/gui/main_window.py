"""Main window for the pyLOCO GUI.

The GUI manages project state, lattice metadata, imported measurement files,
backend-compatible LOCO configuration, and responsive execution monitoring.
"""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import threading
import numpy as np
from html import escape
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QObject, QMetaObject, QRect, QSettings, QSize, Qt, QThread, QUrl, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QDoubleValidator, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QLayout,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QRadioButton,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QProgressDialog,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QApplication,
)
from pyLOCO.gui.file_dialogs import FileDialog as QFileDialog
from pyLOCO.gui.lattice_loader import (
    NoMatLatticeError, load_project_lattice, mat_lattice_candidates,
)

from .fit_workflow import (
    FitRecipe, FitStage, execute_workflow, file_sha256, inspect_measurement_files, preflight_recipe,
)
from .measurement_mapping import (
    find_exact_device_mappings, inventory_fingerprint as measurement_inventory_fingerprint,
    load_document as load_measurement_mapping_document, mapping_counts as measurement_mapping_counts,
    mapping_template as measurement_mapping_template, measurement_inventory,
    ordered_ordinals as measurement_mapping_ordered_ordinals,
    validate_mapping as validate_measurement_mapping,
)
from .mapping_table import CopyableMappingTable, configure_mapping_table, identity_match_label

from .backend import LocoRunError, LocoRunRequest, LocoRunResult, run_loco_request, _load_bad_bpm_positions, _load_measurements
from .machine_detection import detect_machine_elements
from .measurement_metadata import IMPORT_HINTS, inspect_measurement_metadata, measurement_display_fields
from .branding import DISPLAY_ASSET, application_icon, set_asset, wordmark_html
from .models.project import (
    CompletedRunReference, ImportedDataset, LatticeSelection, LocoConfiguration, MachineElementsConfig, ProjectMetadata,
    json_safe, load_cmstep_npz, load_example_project_data, measurement_options_from_config, resolve_element_name_file,
    resolve_example_machine_elements,
)
from .widgets.project_explorer import ProjectExplorer
from .widgets.orm_comparison import OrmComparisonWindow
from .widgets.reference_comparison import ReferenceComparisonWindow
from .reference_comparison import calculate_reference_comparison
from .widgets.waiting_games import WaitingGamesDialog
from .themes import ACCENTS, DEFAULT_ACCENT_KEY, THEMES, apply_application_theme, configure_item_view, theme_for_key
from .results.results_workspace import ResultsWorkspace
from .suite import inspect_measurement_session,launch_suite_application,present_single_about_dialog

APP_STYLESHEET = """
* { font-size: 13px; }
QMainWindow, QDialog { background: #1E1E2E; color: #DDE3F0; }
QMenuBar, QMenu, QToolBar#mainToolbar, QStatusBar { background: #25283A; color: #E7EAF3; border: 0; }
QMenuBar::item:selected, QMenu::item:selected { background: #3B315A; color: #FFFFFF; }
QMenu { border: 1px solid #3C4058; padding: 6px; }
QToolBar#mainToolbar { border-bottom: 1px solid #3C4058; spacing: 10px; padding: 8px 12px; }
QToolButton, QPushButton { background: #2F3347; border: 1px solid #4A4F68; border-radius: 6px; color: #F4F6FB; font-weight: 600; padding: 7px 12px; }
QToolButton:hover, QPushButton:hover { background: #3B315A; border-color: #8A63D2; }
QToolButton:pressed, QPushButton:pressed, QToolButton:checked { background: #8A63D2; border-color: #A78BFA; color: #FFFFFF; }
QPushButton:disabled, QToolButton:disabled { background: #25283A; color: #737993; border-color: #34384D; }
QTabWidget::pane { background: #1E1E2E; border: 1px solid #34384D; border-radius: 10px; padding-top: 8px; }
QTabBar::tab { background: #25283A; border: 1px solid #34384D; border-bottom: 0; border-top-left-radius: 8px; border-top-right-radius: 8px; color: #BFC7D8; margin-right: 4px; padding: 10px 18px; }
QTabBar::tab:selected { background: #2A2D3E; color: #FFFFFF; border-color: #8A63D2; font-weight: 700; }
QDockWidget::title { background: #2A2D3E; color: #FFFFFF; font-weight: 700; padding: 8px 10px; border-bottom: 1px solid #8A63D2; }
QTreeWidget#projectExplorerTree, QTreeView, QTableView, QListWidget, QTextEdit { background: #202334; alternate-background-color: #25283A; border: 1px solid #3C4058; border-radius: 8px; color: #DDE3F0; selection-background-color: #5E45A0; selection-color: #FFFFFF; }
QHeaderView::section { background: #2A2D3E; color: #E7EAF3; border: 0; border-right: 1px solid #3C4058; padding: 6px; font-weight: 700; }
QGroupBox { background: #2A2D3E; border: 1px solid #3C4058; border-radius: 10px; color: #E7EAF3; margin: 10px; padding: 14px; }
QGroupBox::title { color: #C4B5FD; font-weight: 700; subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QLabel { color: #DDE3F0; }
QLabel#statusPill { background: #312A4A; border: 1px solid #8A63D2; border-radius: 10px; color: #EDE9FE; font-weight: 700; padding: 4px 11px; }
QLabel#pageTitle { color: #FFFFFF; font-size: 24px; font-weight: 750; }
QLabel#validationOk { color: #6EE7B7; font-weight: 700; }
QLabel#validationMissing { color: #FBBF24; font-weight: 700; }
QWidget#placeholderPageCard, QWidget#dashboardCard { background: #2A2D3E; border: 1px solid #3C4058; border-radius: 14px; }
QLabel#placeholderTitle { color: #FFFFFF; font-size: 26px; font-weight: 750; }
QLabel#placeholderDescription, QLabel#dashboardCardText { color: #BFC7D8; font-size: 14px; }
QLabel#dashboardCardTitle { color: #C4B5FD; font-size: 15px; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #202334; color: #F4F6FB; border: 1px solid #4A4F68; border-radius: 7px; padding: 6px 8px; selection-background-color: #8A63D2; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #A78BFA; }
QComboBox::drop-down { border: 0; width: 24px; }
QComboBox QAbstractItemView { background: #25283A; color: #F4F6FB; border: 1px solid #8A63D2; selection-background-color: #5E45A0; }
QCheckBox, QRadioButton { color: #DDE3F0; spacing: 8px; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid #6B7280; background: #202334; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #8A63D2; border-color: #C4B5FD; }
QProgressBar { background: #202334; color: #FFFFFF; border: 1px solid #3C4058; border-radius: 7px; text-align: center; }
QProgressBar::chunk { background: #8A63D2; border-radius: 7px; }
QScrollArea { background: #1E1E2E; border: 0; }
QScrollBar:vertical, QScrollBar:horizontal { background: #1E1E2E; width: 12px; height: 12px; }
QScrollBar::handle { background: #4A4F68; border-radius: 6px; }
QScrollBar::handle:hover { background: #8A63D2; }
"""

MACHINE_CONFIGURATION_SCHEMA = "pyloco-machine-configuration/v1"

# Exact main logo preserved from the approved pre-resizing GUI version.
# The toolbar intentionally uses the compact clickable wordmark instead.
LOGO_PATH = Path(__file__).with_name("assets") / "pyloco_logo_pre_resize_version.png"
from .project_info import (PROJECT_ACKNOWLEDGEMENTS,PROJECT_CONTRIBUTORS,PROJECT_DOCUMENTATION,PROJECT_ISSUES,PROJECT_LICENSE,PROJECT_PAPER_TITLE,PROJECT_PAPER_URL,PROJECT_REPOSITORY,bibtex_text,citation_text)


class AspectRatioPixmapLabel(QLabel):
    """A pixmap label that rescales its image when layouts compress it."""

    def __init__(self, pixmap: QPixmap, maximum_width: int, minimum_width: int = 240) -> None:
        super().__init__()
        self._source_pixmap = pixmap
        self._maximum_width = maximum_width
        self._minimum_width = min(minimum_width, maximum_width)
        source_width = max(1, pixmap.width())
        self._aspect_ratio = pixmap.height() / source_width
        self.setAlignment(Qt.AlignCenter)
        self.setMaximumWidth(maximum_width)
        self.setMinimumSize(
            self._minimum_width, round(self._minimum_width * self._aspect_ratio)
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._update_pixmap()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(
            self._maximum_width, round(self._maximum_width * self._aspect_ratio)
        )

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(
            self._minimum_width, round(self._minimum_width * self._aspect_ratio)
        )

    def _update_pixmap(self) -> None:
        if not self._source_pixmap.isNull() and self.width() > 0 and self.height() > 0:
            pixel_ratio = self.devicePixelRatioF()
            target = QSize(
                max(1, round(self.width() * pixel_ratio)),
                max(1, round(self.height() * pixel_ratio)),
            )
            rendered = self._source_pixmap.scaled(
                target, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            rendered.setDevicePixelRatio(pixel_ratio)
            self.setPixmap(rendered)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_header_branding()
        self._update_pixmap()


class ClickableBrandLabel(QLabel):
    """Compact pyLOCO wordmark used as the toolbar's About control."""

    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setText(
            '<span style="font-size:28px; font-weight:800; color:#5B00E6;">py</span>'
            '<span style="font-size:28px; font-weight:800; color:#002B73;">LOCO</span>'
        )
        self.setTextFormat(Qt.RichText)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("About pyLOCO and scientific resources")
        self.setAccessibleName("About pyLOCO")
        self.setContentsMargins(16, 0, 16, 0)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that accepts and displays scientific notation."""

    def __init__(self) -> None:
        super().__init__()
        self._validator = QDoubleValidator(self)
        self._validator.setNotation(QDoubleValidator.ScientificNotation)

    def validate(self, text: str, pos: int):  # type: ignore[override]
        suffix = self.suffix()
        candidate = text.strip()
        if suffix and candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
        if candidate in {"", "+", "-", ".", "+.", "-."}:
            return QDoubleValidator.Intermediate, text, pos
        state, _, _ = self._validator.validate(candidate, pos)
        try:
            value = float(candidate)
        except ValueError:
            return state, text, pos
        if self.minimum() <= value <= self.maximum():
            return state, text, pos
        return QDoubleValidator.Invalid, text, pos

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        suffix = self.suffix()
        candidate = text.strip()
        if suffix and candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)].strip()
        return float(candidate)

    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        return f"{value:.{self.decimals()}g}"

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        # Trackpad/mouse-wheel gestures belong to the surrounding settings
        # page. Numeric values are changed by typing or the step buttons.
        event.ignore()


class ScrollSafeSpinBox(QSpinBox):
    """Integer spin box that never consumes page-scroll gestures."""

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()


class FamilyWeightEditor(QWidget):
    """Compact family/weight table used for constraint exceptions."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Family index", "Weight"])
        configure_item_view(self.table)
        self.table.itemChanged.connect(self.changed.emit)
        buttons = QHBoxLayout()
        add = QPushButton("Add")
        remove = QPushButton("Remove")
        add.clicked.connect(self.add_row)
        remove.clicked.connect(self.remove_selected)
        buttons.addWidget(add); buttons.addWidget(remove); buttons.addStretch(1)
        layout.addWidget(self.table); layout.addLayout(buttons)

    def add_row(self, family: int | None = None, weight: float = 1.0) -> None:
        row = self.table.rowCount(); self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("" if family is None else str(family)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{weight:.6g}"))
        self.changed.emit()

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self.changed.emit()

    def set_mapping(self, values: dict[int, float]) -> None:
        self.table.blockSignals(True); self.table.setRowCount(0)
        for family, weight in sorted(values.items()):
            self.add_row(int(family), float(weight))
        self.table.blockSignals(False)

    def mapping(self) -> dict[int, float]:
        values: dict[int, float] = {}
        for row in range(self.table.rowCount()):
            family_item, weight_item = self.table.item(row, 0), self.table.item(row, 1)
            if family_item is None or not family_item.text().strip():
                continue
            family = int(family_item.text())
            weight = float(weight_item.text()) if weight_item and weight_item.text().strip() else 1.0
            if family < 0 or family in values:
                raise ValueError("Constraint family indices must be unique non-negative integers.")
            values[family] = weight
        return values


class LocoRunWorker(QObject):
    log = Signal(str)
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(object)
    svd_selection_requested = Signal(object)

    def __init__(self, request: LocoRunRequest) -> None:
        super().__init__()
        self.request = request
        self.cancel_requested = False
        self._svd_event: threading.Event | None = None
        self._svd_response = None

    def request_svd_selection(self, singular_values, iteration_tag):
        event = threading.Event()
        self._svd_event = event
        self._svd_response = None
        default_rank = int(
            self.request.backend_mapping.get("LOCOOptions", {}).get("cut_", 0) or 0
        )
        self.svd_selection_requested.emit(
            {
                "singular_values": singular_values,
                "iteration_tag": str(iteration_tag),
                "default_rank": default_rank,
            }
        )
        while not event.wait(0.1):
            if self.cancel_requested:
                return None
        return self._svd_response

    def provide_svd_selection(self, indices) -> None:
        self._svd_response = indices
        if self._svd_event is not None:
            self._svd_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = run_loco_request(
                self.request,
                log_callback=self.log.emit,
                cancel_callback=lambda: self.cancel_requested,
                svd_selection_callback=self.request_svd_selection,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            import traceback

            self.failed.emit(LocoRunError(str(exc), traceback.format_exc(), self.cancel_requested))
        else:
            self.finished.emit(result)


class ReferenceComparisonWorker(QObject):
    """Calculate a reference response without entering the LOCO optimizer."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, request: LocoRunRequest) -> None:
        super().__init__(); self.request = request; self.cancel_requested = False

    @Slot()
    def run(self) -> None:
        try:
            result = calculate_reference_comparison(
                self.request, cancelled=lambda: self.cancel_requested,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class FitWorkflowWorker(LocoRunWorker):
    """Run a recipe serially while preserving the existing per-stage backend."""

    def __init__(self, request, recipe, session_path, resume_session=None) -> None:
        super().__init__(request); self.recipe = recipe; self.session_path = session_path; self.resume_session = resume_session

    @Slot()
    @Slot()
    def run(self) -> None:
        try:
            def runner(request, **callbacks):
                return run_loco_request(
                    request, log_callback=callbacks.get("log_callback"),
                    progress_callback=callbacks.get("progress_callback"),
                    cancel_callback=lambda: self.cancel_requested,
                    svd_selection_callback=self.request_svd_selection,
                )
            session = execute_workflow(
                self.request, self.recipe, session_path=self.session_path, runner=runner,
                log_callback=self.log.emit, progress_callback=self.progress.emit,
                resume_session=self.resume_session,
            )
            last = session.checkpoints[-1]; summary = json.loads((Path(last.results_dir) / "summary.json").read_text())
            output_files = [str(path) for path in Path(last.results_dir).iterdir() if path.is_file()]
            self.finished.emit(LocoRunResult(
                last.results_dir, float(summary.get("runtime_seconds") or 0.0),
                list(summary.get("chi2_history") or []), output_files,
            ))
        except Exception as exc:
            import traceback
            self.failed.emit(LocoRunError(str(exc), traceback.format_exc(), self.cancel_requested))


class SVDSelectionDialog(QDialog):
    """Choose retained singular values without relying on worker-thread stdin."""

    def __init__(self, singular_values, iteration_tag: str, default_rank: int, parent=None) -> None:
        super().__init__(parent)
        import numpy as np

        values = np.asarray(singular_values, dtype=float).ravel()
        self.setWindowTitle("Interactive SVD selection")
        self.resize(620, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Select the singular values to retain for {iteration_tag or 'this solver step'}."
        ))
        layout.addWidget(QLabel(
            "The solver ordering is preserved. At least one singular value must be selected."
        ))

        self.table = QTableWidget(len(values), 3, self)
        self.table.setHorizontalHeaderLabels(["Keep", "Index", "Singular value / max"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        maximum = float(np.max(values)) if len(values) else 1.0
        keep_count = min(default_rank, len(values)) if default_rank > 0 else len(values)
        for row, value in enumerate(values):
            keep = QTableWidgetItem()
            keep.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            keep.setCheckState(Qt.Checked if row < keep_count else Qt.Unchecked)
            self.table.setItem(row, 0, keep)
            index = QTableWidgetItem(str(row))
            index.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, index)
            normalized = QTableWidgetItem(f"{value / maximum:.6e}")
            normalized.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 2, normalized)
        layout.addWidget(self.table, 1)

        quick = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_none = QPushButton("Clear selection")
        select_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        select_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        quick.addWidget(select_all)
        quick.addWidget(select_none)
        quick.addStretch(1)
        layout.addLayout(quick)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, state) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)

    def selected_indices(self) -> list[int]:
        return [
            row for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.Checked
        ]

    def _accept_if_valid(self) -> None:
        if not self.selected_indices():
            QMessageBox.warning(self, "No singular values selected", "Select at least one singular value.")
            return
        self.accept()


ELEMENT_ROLES = {
    "bpm_ords": ("BPMs", "bpm"),
    "horizontal_corrector_ords": ("Horizontal correctors", "hcor"),
    "vertical_corrector_ords": ("Vertical correctors", "vcor"),
    "normal_quadrupole_ords": ("Normal quadrupoles", "quad"),
    "skew_quadrupole_ords": ("Skew quadrupoles", "skew"),
    "quadrupole_tilt_ords": ("Quadrupole tilts", "tilt"),
    "cavity_ords": ("RF cavities", "cavity"),
}


class BrandToolButton(QToolButton):
    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(max(180, hint.width()), hint.height())


class ElementSelectionDialog(QDialog):
    """Select and preview lattice ordinals for one machine-element role."""

    def __init__(self, parent, role_key: str, current: list[int], selection_state: dict | None = None) -> None:
        super().__init__(parent)
        self.role_key = role_key
        self.role_label, self.role_kind = ELEMENT_ROLES[role_key]
        self.setWindowTitle(f"Select {self.role_label}")
        self.resize(760, 560)
        self._lattice = parent._load_current_lattice()
        self.selected_ords = list(current)
        self.selection_state = dict(selection_state or {})
        self._generated_ords = list(self.selection_state.get("generated_ords", current))
        self._restoring_selection = True

        layout = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        self.type_radio = QRadioButton("AT element type")
        self.pattern_radio = QRadioButton("Family/name pattern")
        self.name_file_radio = QRadioButton("Load name file")
        self.file_radio = QRadioButton("Load index file")
        self.manual_radio = QRadioButton("Manual indices")
        for button in (self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
            mode_row.addWidget(button)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        form = QFormLayout(); self._method_form=form
        self.type_edit = QLineEdit(self._default_type_name())
        self.pattern_syntax = QComboBox()
        self.pattern_syntax.addItem("AT wildcard patterns", "at_wildcard")
        self.pattern_syntax.addItem("Python regular expressions", "python_regex")
        self.pattern_edit = QPlainTextEdit(self._default_pattern())
        self.pattern_edit.setMaximumHeight(105)
        self.pattern_edit.setPlaceholderText("One pattern per line, for example:\nQD3[AE]*\nQF4[ABDE]*\nQD5[BD]*")
        self.pattern_match_summary = QLabel("Enter one pattern per line, then update the preview.")
        self.pattern_match_summary.setObjectName("patternMatchSummary")
        self.pattern_match_summary.setWordWrap(True)
        self.pattern_widget = QWidget()
        pattern_layout = QVBoxLayout(self.pattern_widget)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.addWidget(self.pattern_syntax)
        pattern_layout.addWidget(self.pattern_edit)
        pattern_layout.addWidget(self.pattern_match_summary)
        self.file_edit = QLineEdit()
        self.file_button = QPushButton("Browse…")
        self.file_button.clicked.connect(self._browse_index_file)
        self.file_row_widget=QWidget(); file_row = QHBoxLayout(self.file_row_widget); file_row.setContentsMargins(0,0,0,0); file_row.addWidget(self.file_edit); file_row.addWidget(self.file_button)
        self.name_file_edit = QLineEdit()
        self.name_attribute = QComboBox()
        for label, value in (("Auto-detect attribute", "auto"), ("CommonName", "CommonName"), ("FamName", "FamName"), ("Name", "Name"), ("name", "name")):
            self.name_attribute.addItem(label, value)
        self.name_file_button = QPushButton("Browse…")
        self.name_file_button.clicked.connect(self._browse_name_file)
        self.name_file_row_widget=QWidget(); name_file_row = QHBoxLayout(self.name_file_row_widget); name_file_row.setContentsMargins(0,0,0,0); name_file_row.addWidget(self.name_file_edit); name_file_row.addWidget(self.name_file_button)
        self.manual_edit = QPlainTextEdit(", ".join(str(i) for i in current))
        self.manual_edit.setMaximumHeight(105)
        self.manual_edit.setPlaceholderText("Enter integer lattice ordinals separated by commas, spaces, or new lines.")
        form.addRow("AT class/type contains", self.type_edit)
        form.addRow("Family/name patterns", self.pattern_widget)
        self.pattern_edit.setToolTip(
            "Enter one pattern per line. AT wildcard mode calls at.get_refpts() for every line. "
            "Python regular-expression mode searches the displayed CommonName/FamName/Name text instead."
        )
        form.addRow("Element-name file", self.name_file_row_widget)
        form.addRow("Name attribute", self.name_attribute)
        form.addRow("Index file", self.file_row_widget)
        form.addRow("Manual lattice ordinals", self.manual_edit)
        layout.addLayout(form)

        self.provenance_summary = QLabel()
        self.provenance_summary.setObjectName("elementSelectionProvenanceSummary")
        self.provenance_summary.setWordWrap(True)
        layout.addWidget(self.provenance_summary)

        preview_row=QHBoxLayout(); preview_button = QPushButton("Update preview"); preview_button.clicked.connect(self._preview)
        self.selection_count=QLabel(); self.selection_count.setStyleSheet("font-weight:700")
        preview_row.addWidget(preview_button); preview_row.addWidget(self.selection_count); preview_row.addStretch(); layout.addLayout(preview_row)
        self.validation_message = QLabel()
        self.validation_message.setObjectName("selectionValidationMessage")
        self.validation_message.setWordWrap(True)
        self.validation_message.hide()
        layout.addWidget(self.validation_message)
        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Search/filter elements…")
        self.search_edit.textChanged.connect(self._filter_rows)
        filter_row.addWidget(self.search_edit, 1)
        for text, callback in (("Select All", self._select_all), ("Clear", self._clear_all),
                               ("Select Filtered", self._select_filtered)):
            button = QPushButton(text); button.clicked.connect(callback); filter_row.addWidget(button)
        layout.addLayout(filter_row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use / position", "Lattice ordinal", "Element name", "AT class/type", "Family name", "Current value"])
        configure_item_view(self.table)
        self.table.setMinimumHeight(250)
        header=self.table.horizontalHeader(); header.setSectionResizeMode(0,QHeaderView.ResizeToContents); header.setSectionResizeMode(1,QHeaderView.ResizeToContents); header.setSectionResizeMode(2,QHeaderView.Stretch); header.setSectionResizeMode(3,QHeaderView.ResizeToContents); header.setSectionResizeMode(4,QHeaderView.Stretch); header.setSectionResizeMode(5,QHeaderView.Stretch)
        self.table.itemChanged.connect(self._update_selection_count)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for button in (self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
            button.toggled.connect(self._update_method_controls)
        self._restore_selection_state(current)
        self._set_preview_with_final(self._generated_ords, current)
        self._verify_restored_selection()
        self._restoring_selection = False
        self._update_method_controls()

    def _restore_selection_state(self, current: list[int]) -> None:
        method = self.selection_state.get("method")
        if method == "manual":  # Early GUI provenance used this shorter name.
            method = "manual_indices"
            legacy_indices = self.selection_state.get("indices", current)
            self.selection_state = {
                "version": 1,
                "method": method,
                "manual_text": ", ".join(map(str, legacy_indices)),
                "generated_ords": list(map(int, legacy_indices)),
                "resolved_ords": list(current),
            }
            self._generated_ords = list(map(int, legacy_indices))
        if not method:
            if not current:
                self.manual_radio.setChecked(True)
                self._generated_ords = []
                self.provenance_summary.setText(
                    "Method: Manual indices · Source: — · Resolved: 0 · Manual additions: 0 · "
                    "Manual removals: 0 · Final: 0"
                )
                return
            for button in (self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
                button.setAutoExclusive(False); button.setChecked(False); button.setAutoExclusive(True)
            self._generated_ords = list(current)
            self.provenance_summary.setText(
                f"Selection method not recorded — imported lattice indices. Final selection: {len(current)} elements."
            )
            return
        if method == "at_element_type":
            self.type_radio.setChecked(True)
            self.type_edit.setText(str(self.selection_state.get("type_name", "")))
        elif method in {"at_wildcard", "python_regex"}:
            self.pattern_radio.setChecked(True)
            self.pattern_syntax.setCurrentIndex(self.pattern_syntax.findData(method))
            self.pattern_edit.setPlainText(str(self.selection_state.get("patterns", "")))
        elif method == "name_file":
            self.name_file_radio.setChecked(True)
            self.name_file_edit.setText(str(self.selection_state.get("path", "")))
            index = self.name_attribute.findData(self.selection_state.get("name_attribute", "auto"))
            self.name_attribute.setCurrentIndex(max(0, index))
        elif method == "index_file":
            self.file_radio.setChecked(True)
            self.file_edit.setText(str(self.selection_state.get("path", "")))
        elif method == "manual_indices":
            self.manual_radio.setChecked(True)
            self.manual_edit.setPlainText(str(self.selection_state.get("manual_text", ", ".join(map(str, current)))))
        else:
            self.selection_state = {}
            self._restore_selection_state(current)
            return
        self._update_provenance_summary(current)

    def _method_and_source(self) -> tuple[str, str]:
        if self.type_radio.isChecked():
            return "at_element_type", self.type_edit.text()
        if self.pattern_radio.isChecked():
            method = str(self.pattern_syntax.currentData())
            return method, self.pattern_edit.toPlainText()
        if self.name_file_radio.isChecked():
            return "name_file", self.name_file_edit.text()
        if self.file_radio.isChecked():
            return "index_file", self.file_edit.text()
        if self.manual_radio.isChecked():
            return "manual_indices", self.manual_edit.toPlainText()
        return "legacy", "Imported lattice indices"

    def _lattice_identity(self) -> dict:
        if self._lattice is None:
            return {}
        digest = hashlib.sha256()
        for ordinal, element in self._iter_elements():
            identity = (
                ordinal,
                self._element_class(element),
                getattr(element, "CommonName", None),
                getattr(element, "FamName", None),
                getattr(element, "Name", None),
                getattr(element, "name", None),
            )
            digest.update(repr(identity).encode("utf-8"))
            digest.update(b"\0")
        return {"element_count": len(self._lattice), "element_identity_sha256": digest.hexdigest()}

    def _update_provenance_summary(self, final: list[int] | None = None, mismatch: str = "") -> None:
        final_values = list(self.selected_ords if final is None else final)
        generated = set(map(int, self._generated_ords))
        final_set = set(map(int, final_values))
        method, source = self._method_and_source()
        labels = {
            "at_element_type": "AT element type", "at_wildcard": "AT wildcard patterns",
            "python_regex": "Python regular expressions", "name_file": "Loaded name file",
            "index_file": "Loaded index file", "manual_indices": "Manual indices",
            "legacy": "Not recorded",
        }
        source_text = " ".join(source.split()) or "—"
        text = (
            f"Method: {labels.get(method, method)} · Source: {source_text} · "
            f"Resolved: {len(generated)} · Manual additions: {len(final_set - generated)} · "
            f"Manual removals: {len(generated - final_set)} · Final: {len(final_set)}"
        )
        if mismatch:
            text += f"\nSelection verification warning: {mismatch} Stored final ordinals remain unchanged."
        self.provenance_summary.setText(text)

    def _verify_restored_selection(self) -> None:
        if not self.selection_state:
            if self._lattice is not None:
                invalid = [value for value in self.selected_ords if value < 0 or value >= len(self._lattice)]
                if invalid:
                    self._update_provenance_summary(self.selected_ords, f"{len(invalid)} stored ordinal(s) are outside the loaded lattice.")
            return
        try:
            self._validate_array(self.selected_ords)
        except Exception as exc:
            self._update_provenance_summary(self.selected_ords, f"The stored final ordinals are invalid for this lattice: {exc}")
            return
        recorded_final = self.selection_state.get("resolved_ords")
        if recorded_final is not None and list(map(int, recorded_final)) != list(map(int, self.selected_ords)):
            self._update_provenance_summary(
                self.selected_ords,
                "The provenance record does not match the authoritative saved final ordinals.",
            )
            return
        recorded_lattice = self.selection_state.get("lattice_identity") or {}
        current_lattice = self._lattice_identity()
        if recorded_lattice and recorded_lattice != current_lattice:
            self._update_provenance_summary(
                self.selected_ords,
                "The loaded lattice identity differs from the lattice used to save this selection.",
            )
            return
        stored_generated = list(map(int, self.selection_state.get("generated_ords", [])))
        try:
            resolved_now = self._candidate_indices()
            self._validate_array(resolved_now)
        except Exception as exc:
            self._generated_ords = stored_generated
            self._update_provenance_summary(self.selected_ords, f"The recorded method cannot be resolved on this lattice: {exc}")
            return
        self._generated_ords = stored_generated
        if resolved_now != stored_generated:
            self._update_provenance_summary(
                self.selected_ords,
                f"The recorded method now resolves to {len(resolved_now)} elements instead of the stored {len(stored_generated)}.",
            )
        else:
            self._update_provenance_summary(self.selected_ords)

    def _update_method_controls(self) -> None:
        def show_field(field,visible):
            field.setVisible(visible); field.setEnabled(visible)
            label=self._method_form.labelForField(field)
            if label is not None:label.setVisible(visible)
        show_field(self.type_edit,self.type_radio.isChecked())
        show_field(self.pattern_widget,self.pattern_radio.isChecked())
        name_visible=self.name_file_radio.isChecked()
        show_field(self.name_file_row_widget,name_visible)
        show_field(self.name_attribute,name_visible)
        for widget in (self.name_file_edit, self.name_attribute, self.name_file_button):
            widget.setVisible(name_visible); widget.setEnabled(name_visible)
        file_visible=self.file_radio.isChecked(); show_field(self.file_row_widget,file_visible)
        for widget in (self.file_edit, self.file_button):widget.setVisible(file_visible); widget.setEnabled(file_visible)
        show_field(self.manual_edit,self.manual_radio.isChecked())

    def _default_type_name(self) -> str:
        return {"bpm": "Monitor", "hcor": "Corrector", "vcor": "Corrector", "quad": "Quadrupole", "skew": "Quadrupole", "tilt": "Quadrupole", "cavity": "RFCavity"}[self.role_kind]

    def _default_pattern(self) -> str:
        return {"bpm": "*BPM*", "hcor": "*HCM*", "vcor": "*VCM*", "quad": "Q*", "skew": "S*", "tilt": "Q*", "cavity": "*CAV*"}[self.role_kind]

    def _iter_elements(self):
        return list(enumerate(self._lattice or []))

    def _element_name(self, elem) -> str:
        values = []
        for attribute in ("CommonName", "FamName", "Name", "name"):
            value = getattr(elem, attribute, None)
            if value is not None and str(value) not in values:
                values.append(str(value))
        return " / ".join(values)

    def _element_class(self, elem) -> str:
        return type(elem).__name__

    def _browse_index_file(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load lattice-index array", "", "Index arrays (*.npy *.npz *.h5 *.hdf5 *.txt);;All files (*)")[0]
        if filename:
            self.file_edit.setText(filename); self.file_radio.setChecked(True); self._preview()

    def _browse_name_file(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load element-name list", "", "Text files (*.txt *.list *.dat);;All files (*)")[0]
        if filename:
            self.name_file_edit.setText(filename); self.name_file_radio.setChecked(True); self._preview()

    def _candidate_indices(self) -> list[int]:
        if self.type_radio.isChecked():
            needle = self.type_edit.text().strip().lower()
            return [i for i, e in self._iter_elements() if needle and needle in self._element_class(e).lower()]
        if self.pattern_radio.isChecked():
            patterns = [line.strip() for line in self.pattern_edit.toPlainText().splitlines() if line.strip()]
            if not patterns:
                self.pattern_match_summary.setText("No patterns entered · combined: 0 unique elements")
                return []
            matches_by_pattern = []
            syntax = self.pattern_syntax.currentData()
            if syntax == "at_wildcard":
                import at
                for pattern_text in patterns:
                    if pattern_text.count("[") != pattern_text.count("]"):
                        raise ValueError(f"Invalid AT wildcard {pattern_text!r}: unmatched '[' or ']'.")
                    if pattern_text.startswith("re:"):
                        raise ValueError("The 're:' prefix is not used in AT wildcard mode; select Python regular expressions instead.")
                    refs = np.asarray(at.get_refpts(self._lattice, pattern_text))
                    values = [int(value) for value in (np.flatnonzero(refs) if refs.dtype == np.bool_ else refs.ravel())]
                    matches_by_pattern.append((pattern_text, values))
            else:
                for pattern_text in patterns:
                    expression = pattern_text[3:] if pattern_text.startswith("re:") else pattern_text
                    try:
                        pattern = re.compile(expression, re.I)
                    except re.error as exc:
                        raise ValueError(f"Invalid Python regular expression {pattern_text!r}: {exc}") from exc
                    values = [i for i, e in self._iter_elements() if pattern.search(self._element_name(e))]
                    matches_by_pattern.append((pattern_text, values))
            combined = sorted({ordinal for _pattern, values in matches_by_pattern for ordinal in values})
            details = " · ".join(f"{pattern}: {len(values)}" for pattern, values in matches_by_pattern)
            self.pattern_match_summary.setText(
                f"{details} · combined: {len(combined)} unique element{'s' if len(combined) != 1 else ''}"
            )
            return combined
        if self.name_file_radio.isChecked():
            if not self._lattice:
                raise ValueError("Load a lattice before selecting elements from a name file.")
            return resolve_element_name_file(
                self._lattice,
                Path(self.name_file_edit.text()).expanduser(),
                self.name_attribute.currentData() or "auto",
            )
        if self.file_radio.isChecked():
            return self._load_index_file(Path(self.file_edit.text()).expanduser())
        text = self.manual_edit.toPlainText().strip()
        if not text:
            raise ValueError("Enter at least one lattice ordinal before previewing the selection.")
        tokens = [token for token in re.split(r"[,\s]+", text) if token]
        invalid = [token for token in tokens if re.fullmatch(r"[-+]?\d+", token) is None]
        if invalid:
            raise ValueError("Manual lattice ordinals must be integers; invalid value(s): " + ", ".join(invalid))
        return [int(token) for token in tokens]

    def _load_index_file(self, path: Path) -> list[int]:
        import numpy as np
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".npz":
            data = np.load(path); arr = data[next(iter(data.files))]
        elif suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as h5:
                first = next(iter(h5.keys())); arr = h5[first][()]
        else:
            arr = np.loadtxt(path)
        return self._validate_array(arr)

    def _validate_array(self, arr) -> list[int]:
        import numpy as np
        a = np.asarray(arr)
        if a.ndim != 1:
            raise ValueError("Selection indices must be a one-dimensional array.")
        if not np.issubdtype(a.dtype, np.integer):
            if np.issubdtype(a.dtype, np.floating) and np.all(a == np.floor(a)):
                a = a.astype(int)
            else:
                raise ValueError("Selection indices must be integers.")
        values = [int(v) for v in a.tolist()]
        if len(set(values)) != len(values):
            raise ValueError("Selection indices must be unique.")
        if self._lattice is None:
            raise ValueError("Load a lattice before previewing a manual element selection.")
        if any(v < 0 or v >= len(self._lattice) for v in values):
            raise ValueError(f"Selection indices must be within lattice range [0, {len(self._lattice)-1}].")
        self._validate_compatible(values)
        return values

    def _validate_compatible(self, values: list[int]) -> None:
        if self._lattice is None:
            return
        if self.role_kind in {"hcor", "vcor"}:
            from pyLOCO.response_matrix import corrector_steering_compatibility
            calculator_widget = getattr(self.parent(), "rm_calculator", None)
            calculator = calculator_widget.currentData() if calculator_widget is not None else "Linear"
            plane = 0 if self.role_kind == "hcor" else 1
            for ordinal in values:
                element = self._lattice[ordinal]
                compatible, detail = corrector_steering_compatibility(
                    element, plane, calculator=calculator
                )
                if not compatible:
                    name = self._element_name(element) or "unnamed element"
                    raise ValueError(
                        f"Ordinal {ordinal} ({name}, {self._element_class(element)}) has no supported "
                        f"{'horizontal' if plane == 0 else 'vertical'} steering representation for the "
                        f"{calculator} calculator: {detail}"
                    )
            return
        if self.role_kind in {"quad", "skew"}:
            default_attribute = "PolynomB" if self.role_kind == "quad" else "PolynomA"
            attribute_widget = getattr(
                self.parent(),
                "params_quads_attr" if self.role_kind == "quad" else "params_skew_attr",
                None,
            )
            index_widget = getattr(
                self.parent(),
                "params_quads_attr_index" if self.role_kind == "quad" else "params_skew_attr_index",
                None,
            )
            attribute = attribute_widget.text().strip() if attribute_widget is not None else default_attribute
            component = index_widget.value() if index_widget is not None else 1
            if "[" in attribute and attribute.endswith("]"):
                attribute, component_text = attribute[:-1].split("[", 1)
                try:
                    component = int(component_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid fitted component {attribute_widget.text()!r}; expected a name such as "
                        f"{default_attribute}[1]."
                    ) from exc
            for ordinal in values:
                element = self._lattice[ordinal]
                coefficients = getattr(element, attribute, None)
                try:
                    compatible = coefficients is not None and 0 <= int(component) < len(coefficients)
                except TypeError:
                    compatible = coefficients is not None and int(component) == 0
                if not compatible:
                    name = self._element_name(element) or "unnamed element"
                    raise ValueError(
                        f"Ordinal {ordinal} ({name}, {self._element_class(element)}) cannot be used as "
                        f"{self.role_label}: the selected FIT calculation requires {attribute}[{component}], "
                        "but that component is unavailable. Multipole container elements are supported when "
                        "they expose the required coefficient."
                    )
            return
        required = self._default_type_name().lower()
        for v in values:
            cls = self._element_class(self._lattice[v]).lower()
            if required not in cls:
                raise ValueError(f"Ordinal {v} is a {cls}, not compatible with {self.role_label}.")

    def _preview(self) -> bool:
        try:
            values = self._validate_array(self._candidate_indices())
        except Exception as exc:
            self.validation_message.setText(f"Invalid selection: {exc}")
            self.validation_message.show()
            QMessageBox.warning(self, "Invalid selection", str(exc))
            return False
        self.validation_message.clear()
        self.validation_message.hide()
        self._generated_ords = list(values)
        self.selected_ords = values
        self._set_preview(values)
        self._update_provenance_summary(values)
        return True

    def _set_preview(self, values: list[int]) -> None:
        self._set_preview_with_final(values, values)

    def _set_preview_with_final(self, generated: list[int], final: list[int]) -> None:
        final_set = set(map(int, final))
        # Preserve explicit/manual and file ordering. Pattern resolution is
        # already sorted by lattice ordinal in _candidate_indices().
        values = list(dict.fromkeys(map(int, generated)))
        values.extend(int(value) for value in final if int(value) not in set(values))
        self.table.setRowCount(len(values))
        for row, ordinal in enumerate(values):
            elem = self._lattice[ordinal] if self._lattice is not None and 0 <= ordinal < len(self._lattice) else None
            use = QTableWidgetItem(str(row)); use.setCheckState(Qt.Checked if ordinal in final_set else Qt.Unchecked); use.setData(Qt.UserRole, int(ordinal)); self.table.setItem(row, 0, use)
            name = str(
                getattr(elem, "CommonName", None)
                or getattr(elem, "Name", None)
                or getattr(elem, "FamName", "")
            ) if elem else ""
            family = str(getattr(elem, "FamName", "")) if elem else ""
            value = self._current_value(elem) if elem else ""
            cells = [ordinal, name, self._element_class(elem) if elem else "", family, value]
            for col, value in enumerate(cells, start=1):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self._filter_rows()
        self._update_selection_count()

    def _update_selection_count(self, *_args) -> None:
        if not hasattr(self,"selection_count"):return
        selected_ordinals = [
            int(self.table.item(row, 0).data(Qt.UserRole))
            for row in range(self.table.rowCount())
            if self.table.item(row, 0) and self.table.item(row, 0).checkState() == Qt.Checked
        ]
        self.selection_count.setText(f"{len(selected_ordinals)} selected / {self.table.rowCount()} in preview")
        if hasattr(self, "provenance_summary") and not getattr(self, "_restoring_selection", False):
            self._update_provenance_summary(selected_ordinals)

    def _current_value(self, elem) -> str:
        import numpy as np
        if self.role_kind == "skew":
            values = getattr(elem, "PolynomA", ())
            return f"PolynomA[1] = {float(values[1]):.9g}" if len(values) > 1 else "Not available"
        if self.role_kind == "tilt":
            matrix = getattr(elem, "R1", None)
            return f"tilt = {float(np.arctan2(matrix[0, 2], matrix[0, 0])):.9g} rad" if getattr(matrix, "shape", None) == (6, 6) else "tilt = 0 rad"
        if self.role_kind == "quad":
            values = getattr(elem, "PolynomB", ())
            return f"PolynomB[1] = {float(values[1]):.9g} m⁻²" if len(values) > 1 else "Not available"
        return "—"

    def _filter_rows(self) -> None:
        needle = self.search_edit.text().strip().lower() if hasattr(self, "search_edit") else ""
        for row in range(self.table.rowCount()):
            text = " ".join(self.table.item(row, col).text() for col in range(self.table.columnCount()) if self.table.item(row, col))
            self.table.setRowHidden(row, bool(needle and needle not in text.lower()))

    def _set_checks(self, checked: bool, *, filtered_only: bool = False) -> None:
        for row in range(self.table.rowCount()):
            if not filtered_only or not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _select_all(self): self._set_checks(True)
    def _clear_all(self): self._set_checks(False)
    def _select_filtered(self): self._set_checks(True, filtered_only=True)

    def _accept_if_valid(self) -> None:
        selected = [int(self.table.item(row, 0).data(Qt.UserRole)) for row in range(self.table.rowCount())
                    if self.table.item(row, 0).checkState() == Qt.Checked]
        if not selected:
            QMessageBox.warning(self, "No elements selected", "Select at least one element."); return
        self.selected_ords = self._validate_array(selected)
        generated = list(dict.fromkeys(map(int, self._generated_ords)))
        final_set = set(self.selected_ords)
        method, _source = self._method_and_source()
        state = {
            "version": 1,
            "method": method,
            "generated_ords": generated,
            "resolved_ords": list(self.selected_ords),
            "manual_additions": sorted(final_set - set(generated)),
            "manual_removals": sorted(set(generated) - final_set),
            "lattice_identity": self._lattice_identity(),
        }
        if method == "at_element_type":
            state["type_name"] = self.type_edit.text()
        elif method in {"at_wildcard", "python_regex"}:
            state["patterns"] = self.pattern_edit.toPlainText()
        elif method == "name_file":
            state.update(path=self.name_file_edit.text(), name_attribute=self.name_attribute.currentData() or "auto")
        elif method == "index_file":
            state["path"] = self.file_edit.text()
        elif method == "manual_indices":
            state["manual_text"] = self.manual_edit.toPlainText()
        else:
            state = {}
        self.selection_state = state
        self.accept()


class FamilyGroupDialog(QDialog):
    """Load and inspect explicit physical-element groups."""
    def __init__(self, parent, role_key: str, selected: list[int], current: list[list[int]]):
        super().__init__(parent); self.setWindowTitle("Explicit family groups"); self.resize(820, 560)
        self.role_key=role_key; self.selected=list(selected); self.groups=[list(group) for group in current]
        layout=QVBoxLayout(self)
        note=QLabel("Family membership is explicit. Names are displayed for inspection only and never used to infer groups."); note.setWordWrap(True); layout.addWidget(note)
        row=QHBoxLayout(); self.path=QLineEdit(); browse=QPushButton("Load family groups…"); browse.clicked.connect(self._browse); row.addWidget(self.path,1); row.addWidget(browse); layout.addLayout(row)
        self.table=QTableWidget(0,4); self.table.setHorizontalHeaderLabels(["Parameter", "Members", "Lattice ordinals", "Element names"]); configure_item_view(self.table); layout.addWidget(self.table,1)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel); buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self._populate()

    def _browse(self):
        import numpy as np
        filename=QFileDialog.getOpenFileName(self,"Load explicit family groups","","NumPy arrays (*.npy *.npz);;All files (*)")[0]
        if not filename:return
        try:
            data=np.load(filename,allow_pickle=True)
            if hasattr(data,"files"):data=data[data.files[0]]
            from .models.project import validate_element_groups
            self.groups=validate_element_groups(data.tolist(),self.selected,len(self.parent()._load_current_lattice()),ELEMENT_ROLES[self.role_key][0])
        except Exception as exc:QMessageBox.warning(self,"Invalid family groups",str(exc));return
        self.path.setText(filename); self._populate()

    def _populate(self):
        lattice=self.parent()._load_current_lattice(); self.table.setRowCount(len(self.groups))
        for row,group in enumerate(self.groups):
            names=[ElementSelectionDialog._element_name(None,lattice[index]) for index in group]
            for col,value in enumerate((row,len(group),", ".join(map(str,group)),"; ".join(names))):self.table.setItem(row,col,QTableWidgetItem(str(value)))

    def _accept(self):
        if not self.groups:QMessageBox.warning(self,"No family groups","Load explicit groups before choosing family parameterization.");return
        self.accept()


class ExclusionSelectionDialog(QDialog):
    """Select exclusions by selected-list position while showing lattice identity."""

    def __init__(self, parent, title: str, ordinals: list[int], excluded: list[int]) -> None:
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(720, 560)
        self.ordinals = list(ordinals)
        self.excluded_positions = sorted(set(int(value) for value in excluded))
        lattice = parent._load_current_lattice()
        layout = QVBoxLayout(self)
        note = QLabel("Check rows to exclude. ‘Selected-list position’ is zero-based within the current component selection; ‘lattice ordinal’ identifies the corresponding pyAT element.")
        note.setWordWrap(True); layout.addWidget(note)
        actions = QHBoxLayout(); load_button = QPushButton("Load exclusion file…")
        load_button.setToolTip("Supported: .txt, .npy, .npz, .h5/.hdf5, .mat; values are zero-based selected-list positions.")
        load_button.clicked.connect(self._load_file)
        clear_button = QPushButton("Clear all"); clear_button.clicked.connect(self._clear)
        actions.addWidget(load_button); actions.addWidget(clear_button); actions.addStretch(1); layout.addLayout(actions)
        self.table = QTableWidget(len(self.ordinals), 4)
        self.table.setHorizontalHeaderLabels(["Exclude", "Selected-list position", "Lattice ordinal", "Element name / class"])
        configure_item_view(self.table)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for row, ordinal in enumerate(self.ordinals):
            check = QTableWidgetItem(); check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Checked if row in self.excluded_positions else Qt.Unchecked)
            elem = lattice[ordinal] if lattice is not None and 0 <= ordinal < len(lattice) else None
            name = str(getattr(elem, "FamName", getattr(elem, "name", ""))) if elem is not None else ""
            cls = type(elem).__name__ if elem is not None else ""
            self.table.setItem(row, 0, check); self.table.setItem(row, 1, QTableWidgetItem(str(row)))
            self.table.setItem(row, 2, QTableWidgetItem(str(ordinal)))
            self.table.setItem(row, 3, QTableWidgetItem(f"{name} · {cls}".strip(" ·")))
        layout.addWidget(self.table, 1); self.counts = QLabel(); layout.addWidget(self.counts)
        self.table.itemChanged.connect(self._update_counts); self._update_counts()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def _selected(self) -> list[int]:
        return [row for row in range(self.table.rowCount()) if self.table.item(row, 0).checkState() == Qt.Checked]

    def _update_counts(self, *_args) -> None:
        count = len(self._selected())
        self.counts.setText(f"{len(self.ordinals)} selected · {count} excluded · {len(self.ordinals)-count} retained")

    def _clear(self) -> None:
        for row in range(self.table.rowCount()): self.table.item(row, 0).setCheckState(Qt.Unchecked)

    def _load_file(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load exclusion positions", "", "Position arrays (*.txt *.npy *.npz *.h5 *.hdf5 *.mat);;All files (*)")[0]
        if not filename: return
        try:
            values = _load_bad_bpm_positions({"bad_bpms": filename})
            positions = [] if values is None else [int(value) for value in values]
            if len(positions) != len(set(positions)) or any(value < 0 or value >= len(self.ordinals) for value in positions):
                raise ValueError(f"Positions must be unique and within 0..{max(0, len(self.ordinals)-1)}.")
            selected = set(positions)
            for row in range(self.table.rowCount()):
                self.table.item(row, 0).setCheckState(Qt.Checked if row in selected else Qt.Unchecked)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid exclusion file", str(exc))

    def _accept(self) -> None:
        self.excluded_positions = self._selected(); self.accept()


class MainWindow(QMainWindow):
    """Top-level pyLOCO GUI window for project management and data import."""

    def __init__(self) -> None:
        super().__init__()
        self.project = ProjectMetadata()
        self._loading_config = False
        self.setObjectName("pyLocoMainWindow")
        self.setWindowTitle("pyLOCO GUI")
        self.setWindowIcon(application_icon())
        self.resize(1320, 860)
        # Keep the top-level window freely resizable. Individual pages reflow
        # or provide their own scrolling when the window becomes very small.
        self.setMinimumSize(360, 240)
        self._settings = QSettings()
        self._geometry_save_timer = QTimer(self)
        self._geometry_save_timer.setSingleShot(True)
        self._geometry_save_timer.setInterval(350)
        self._geometry_save_timer.timeout.connect(self._save_window_layout)
        self._startup_geometry_restored = False
        self._restoring_startup_geometry = True
        self.current_theme = theme_for_key(self._settings.value("appearance/theme", "dark"))
        self.current_accent = str(self._settings.value("fit/appearance/accent", DEFAULT_ACCENT_KEY))
        apply_application_theme(QApplication.instance(), self.current_theme, self.current_accent)
        saved_rect_values = tuple(
            self._settings.value(f"window/{key}", None)
            for key in ("x", "y", "width", "height")
        )
        self._startup_normal_geometry = (
            QRect(*(int(value) for value in saved_rect_values))
            if all(value is not None for value in saved_rect_values)
            else self._settings.value("window/normal_geometry")
        )
        self._startup_geometry = self._settings.value("window/geometry")
        saved_mode = str(self._settings.value("workflow/mode", "Basic"))
        self.project.mode = saved_mode if saved_mode in {"Basic", "Advanced"} else "Basic"

        self._mode_label = QLabel("Basic mode")
        self._mode_label.setObjectName("statusPill")
        self._project_label = QLabel()
        self._workflow_label = QLabel("Workflow: Project")
        self._backend_label = QLabel("Backend: unchanged")
        self._validation_label = QLabel()
        self._project_explorer = ProjectExplorer()
        self._run_thread: QThread | None = None
        self._run_worker: LocoRunWorker | None = None
        self._run_preparing = False
        self._run_started_at = 0.0
        self._last_loco_result = None
        self._waiting_games_dialog: WaitingGamesDialog | None = None
        self._run_cancel_requested = False
        self._orm_comparison_windows = []
        self._reference_comparison_windows = []
        self._reference_comparison_thread = None
        self._reference_comparison_worker = None
        self._reference_comparison_progress = None
        self._reference_comparison_start_signature = ""
        self.fit_recipe = FitRecipe()
        self._active_fit_stage = -1
        self._resume_fit_session = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._workspace = self._create_workspace()
        # The Fit page is scrollable; its wide form size hint must not prevent
        # the dock divider from reallocating space within the current window.
        self._workspace.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        saved_results_tab = int(self._settings.value("results/tab", 0))
        self.results_workspace.tabs.setCurrentIndex(
            min(max(saved_results_tab, 0), self.results_workspace.tabs.count() - 1)
        )
        self.results_workspace.tabs.currentChanged.connect(
            lambda index: self._settings.setValue("results/tab", index)
        )
        parameter_splitter_state = self._settings.value("results/parameter_splitter")
        if parameter_splitter_state is not None:
            self.results_workspace.parameters.content_splitter.restoreState(parameter_splitter_state)
        self.results_workspace.parameters.content_splitter.splitterMoved.connect(
            lambda *_: self._settings.setValue(
                "results/parameter_splitter",
                self.results_workspace.parameters.content_splitter.saveState(),
            )
        )

        self.setCentralWidget(self._workspace)
        self._configure_responsive_layouts()
        self.addDockWidget(Qt.LeftDockWidgetArea, self._project_explorer)
        saved_window_state = self._settings.value("window/state")
        if saved_window_state is not None:
            self.restoreState(saved_window_state)
        else:
            self.resizeDocks([self._project_explorer], [320], Qt.Horizontal)
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._workspace.currentChanged.connect(self._on_tab_changed)
        self._project_explorer.navigate_requested.connect(self._navigate_from_explorer)
        self._refresh_ui("Ready — create or open a project")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._confirm_discard_changes():
            event.ignore()
            return
        if self._reference_comparison_worker is not None:
            self._reference_comparison_worker.cancel_requested = True
        if self._reference_comparison_thread is not None:
            # response_matrix itself is not interruptible; wait for its next
            # safe boundary so Qt never destroys a running worker thread.
            self._reference_comparison_thread.quit()
            self._reference_comparison_thread.wait()
        self._save_window_layout()
        self._settings.setValue("workflow/mode", self.project.mode)
        self._settings.sync()
        event.accept()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._startup_geometry_restored:
            self._startup_geometry_restored = True
            normal_geometry = self._startup_normal_geometry
            geometry = self._startup_geometry
            if normal_geometry is not None:
                QTimer.singleShot(0, lambda: self._restore_movable_geometry(normal_geometry))
                QTimer.singleShot(250, lambda: self._finish_startup_geometry(normal_geometry))
            elif geometry is not None:
                # Restore after all toolbars/docks exist; macOS can otherwise
                # reposition a window while applying their size hints.
                QTimer.singleShot(0, lambda: self._restore_legacy_geometry(geometry))
                QTimer.singleShot(250, lambda: self._finish_startup_geometry(self.normalGeometry()))
            else:
                self._restoring_startup_geometry = False

    def _restore_movable_geometry(self, geometry) -> None:
        """Restore a normal window rectangle that remains draggable."""
        self.showNormal()
        self.resize(geometry.size())
        self.move(geometry.topLeft())

    def _finish_startup_geometry(self, geometry) -> None:
        """Reapply placement after the native macOS window has settled."""
        if geometry is not None and geometry.isValid():
            self._restore_movable_geometry(geometry)
        self._restoring_startup_geometry = False
        self._save_window_layout()

    def _restore_legacy_geometry(self, geometry) -> None:
        """Migrate old saved geometry without retaining maximized/fullscreen state."""
        self.restoreGeometry(geometry)
        normal = self.normalGeometry()
        self.showNormal()
        if normal.isValid():
            self.setGeometry(normal)

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        timer = getattr(self, "_geometry_save_timer", None)
        if timer is not None and self.isVisible() and not self._restoring_startup_geometry:
            timer.start()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Keep the central workspace usable while the whole window shrinks."""
        super().resizeEvent(event)
        timer = getattr(self, "_geometry_save_timer", None)
        if timer is not None and self.isVisible() and not self._restoring_startup_geometry:
            timer.start()
        explorer = getattr(self, "_project_explorer", None)
        if (
            explorer is not None
            and explorer.isVisible()
            and not explorer.isFloating()
            and self.width() < 760
        ):
            maximum_sidebar = max(explorer.minimumWidth(), int(self.width() * 0.38))
            if explorer.width() > maximum_sidebar:
                self.resizeDocks([explorer], [maximum_sidebar], Qt.Horizontal)

    def _save_window_layout(self) -> None:
        """Persist window position, size, and dock layout during the session."""
        if not hasattr(self, "_settings"):
            return
        normal = self.normalGeometry() if (self.isMaximized() or self.isFullScreen()) else self.geometry()
        if normal.isValid():
            self._settings.setValue("window/normal_geometry", normal)
            self._settings.setValue("window/x", normal.x())
            self._settings.setValue("window/y", normal.y())
            self._settings.setValue("window/width", normal.width())
            self._settings.setValue("window/height", normal.height())
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())
        self._settings.sync()

    def _confirm_discard_changes(self) -> bool:
        if not self.project.modified:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes", "Save changes to the current project?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            self.save_project()
            return self.project.is_saved
        return True

    def _confirm_open_with_unsaved_changes(self) -> bool:
        """Offer an optional save before Open Project without discarding state."""
        if not self.project.modified:
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Unsaved changes — open FIT project")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText("The current FIT project has unsaved changes.")
        dialog.setInformativeText(
            "Saving is optional. Continue without saving keeps the current project in memory until you choose "
            "whether to retain its measurements in the next step. No project or measurement file is deleted."
        )
        save = dialog.addButton("Save current project…", QMessageBox.AcceptRole)
        proceed = dialog.addButton("Continue without saving", QMessageBox.ActionRole)
        cancel = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(proceed)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is save:
            return self.save_project()
        if clicked is proceed:
            return True
        if clicked is cancel:
            return False
        return False

    def _create_workspace(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        self.dashboard_name = QLineEdit(self.project.name)
        self.dashboard_name.editingFinished.connect(self._rename_project)
        self.dashboard_description = QLineEdit(self.project.description)
        self.dashboard_description.setPlaceholderText("Optional project description")
        self.dashboard_description.editingFinished.connect(self._update_project_description)
        self.dashboard_summary = QLabel()
        self.recent_list = QListWidget()
        configure_item_view(self.recent_list)
        tabs.addTab(self._project_page(), "Project")
        tabs.addTab(self._machine_page(), "Machine Components")
        tabs.addTab(self._measurements_page(), "Measurements")
        tabs.addTab(self._fit_page(), "Fit")
        self.results_page = self._results_page()
        tabs.addTab(self.results_page, "Results")
        return tabs


    def _results_page(self) -> QWidget:
        self.results_workspace = ResultsWorkspace()
        self.run_status_label = self.results_workspace.run_status_label
        self.run_elapsed_label = self.results_workspace.run_elapsed_label
        self.run_progress = self.results_workspace.run_progress
        self.run_output_dir = self.results_workspace.run_output_dir
        self.run_log = self.results_workspace.log.text
        self.cancel_loco_button = self.results_workspace.cancel_button
        self.cancel_loco_button.clicked.connect(self.cancel_loco_run)
        self.results_workspace.waiting_games_button.clicked.connect(self._open_waiting_games)
        self.results_workspace.open_correct_requested.connect(self.open_correct_app)
        self.results_workspace.orm.comparison_requested.connect(self.compare_orms)
        return self.results_workspace

    def _project_page(self) -> QWidget:
        page = self._page("Project Dashboard")
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); body = QVBoxLayout(content); body.setContentsMargins(0, 0, 8, 12); body.setSpacing(12); body.setSizeConstraint(QLayout.SetMinimumSize)
        scroll.setWidget(content); page.layout().addWidget(scroll, 1)
        self.dashboard_logo_button = QToolButton()
        self.dashboard_logo_button.setObjectName("dashboardLogoButton")
        self.dashboard_logo_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_logo_button.setToolTip("Open pyLOCO information and scientific resources")
        self.dashboard_logo_button.setAccessibleName("pyLOCO logo and information")
        self.dashboard_logo_button.clicked.connect(self._show_about_dialog)
        self.dashboard_logo_button.setStyleSheet(
            "QToolButton { background: transparent; border: 0; border-radius: 10px; padding: 4px; "
            "min-width: 270px; max-width: 270px; min-height: 180px; max-height: 180px; }"
            "QToolButton:hover { background: rgba(126, 87, 194, 0.10); }"
            "QToolButton::menu-indicator { image: none; width: 0px; }"
        )
        logo_layout = QHBoxLayout(self.dashboard_logo_button)
        logo_layout.setContentsMargins(4, 4, 4, 4)
        self.dashboard_logo = QLabel(); self.dashboard_logo.setFixedSize(270, 180)
        set_asset(self.dashboard_logo, QSize(270, 180), DISPLAY_ASSET, crop_transparency=False, theme_key=self.current_theme.key)
        self.dashboard_logo.setAttribute(Qt.WA_TransparentForMouseEvents)
        logo_layout.addWidget(self.dashboard_logo)
        # Apply this after styling so the global tool-button theme cannot
        # replace the exact dimensions from the earlier GUI version.
        self.dashboard_logo_button.setFixedSize(278, 188)
        body.addWidget(self.dashboard_logo_button, 0, Qt.AlignHCenter)
        suite=QGroupBox("pyLOCO Suite — MEASURE → FIT → CORRECT"); suite_layout=QHBoxLayout(suite)
        for title,description,color,slot in (("1. MEASURE","Acquire BPM noise, dispersion and ORM measurements","#12BFC4",self.open_measure_app),("2. FIT","Fit measured response data and reconstruct optics errors","#496FD8",None),("3. CORRECT","Review, scale and validate fitted machine corrections","#E88B22",self.open_correct_app)):
            card=QWidget(); card.setMinimumHeight(125); card_layout=QVBoxLayout(card); heading=QLabel(title); heading.setStyleSheet(f"font-size:14pt;font-weight:800;color:{color}"); detail=QLabel(description); detail.setWordWrap(True); detail.setMinimumHeight(38); card_layout.addWidget(heading); card_layout.addWidget(detail)
            if slot is not None:
                button=QPushButton(f"Open pyLOCO {title.split('. ',1)[1].title()}"); button.clicked.connect(slot)
            else:
                button=QPushButton("Current application"); button.setEnabled(False)
            card_layout.addWidget(button)
            suite_layout.addWidget(card,1)
        suite.setMinimumHeight(160); body.addWidget(suite)
        form = QFormLayout()
        form.addRow("Project name", self.dashboard_name)
        form.addRow("Description", self.dashboard_description)
        for text, slot in (
            ("New Project", self.new_project),
            ("Open Project…", self.open_project),
            ("Save Project", self.save_project),
            ("Save Project As…", self.save_project_as),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            form.addRow(button)
            if text == "Open Project…":
                self.dashboard_open_project_button = button
        group = QGroupBox("Project state")
        group.setLayout(form)
        body.addWidget(group)
        body.addWidget(self.dashboard_summary)
        body.addWidget(QLabel("Recent projects"))
        self.recent_list.setMinimumHeight(100)
        self.recent_list.setToolTip(
            "Double-click to open a project. Select a path and press Command+C/Ctrl+C, "
            "or use Copy selected path."
        )
        body.addWidget(self.recent_list)
        recent_actions = QHBoxLayout()
        self.copy_recent_path_button = QPushButton("Copy selected path")
        self.copy_recent_path_button.setEnabled(False)
        self.copy_recent_path_button.clicked.connect(self._copy_selected_recent_project_path)
        recent_actions.addWidget(self.copy_recent_path_button)
        recent_actions.addStretch(1)
        body.addLayout(recent_actions)
        body.addStretch(1)
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.text()))
        )
        self.recent_list.currentItemChanged.connect(
            lambda current, _previous: self.copy_recent_path_button.setEnabled(current is not None)
        )
        copy_recent_action = QAction("Copy project path", self.recent_list)
        copy_recent_action.setShortcut(QKeySequence.Copy)
        copy_recent_action.setShortcutContext(Qt.WidgetShortcut)
        copy_recent_action.triggered.connect(self._copy_selected_recent_project_path)
        self.recent_list.addAction(copy_recent_action)
        self.recent_list.setContextMenuPolicy(Qt.ActionsContextMenu)
        return page

    @Slot()
    def _copy_selected_recent_project_path(self) -> bool:
        item = self.recent_list.currentItem()
        if item is None:
            self.statusBar().showMessage("Select a recent project path to copy.", 4000)
            return False
        path = item.text()
        QApplication.clipboard().setText(path)
        self.statusBar().showMessage(f"Project path copied: {path}", 5000)
        return True

    def _logo_block(self, maximum_width: int) -> QVBoxLayout:
        """Build a centered, aspect-ratio-preserving pyLOCO logo block."""
        layout = QVBoxLayout()
        logo = AspectRatioPixmapLabel(QPixmap(str(LOGO_PATH)), maximum_width)
        logo.setObjectName("pyLocoLogo")
        layout.addWidget(logo, 0, Qt.AlignHCenter)
        return layout

    def _build_brand_menu(self) -> QMenu:
        """Information menu restored from the pre-resizing GUI branding."""
        menu = QMenu(self)
        menu.setObjectName("brandMenu")
        menu.addSection("pyLOCO — Storage Ring Optics Correction")
        version_action = menu.addAction(f"Version {self._package_version()}"); version_action.setEnabled(False)
        menu.addSeparator(); menu.addAction("About pyLOCO", self._show_about_dialog)
        menu.addAction(
            "Documentation",
            lambda: QDesktopServices.openUrl(QUrl(PROJECT_DOCUMENTATION)),
        )
        menu.addAction(
            "Scientific reference / methodology",
            lambda: QDesktopServices.openUrl(QUrl(PROJECT_PAPER_URL)),
        )
        menu.addSeparator()
        menu.addAction("Copy citation", lambda: QApplication.clipboard().setText(self._software_citation()))
        menu.addAction("Copy BibTeX", lambda: QApplication.clipboard().setText(self._software_bibtex()))
        menu.addSeparator()
        menu.addAction(
            "Repository / Source code",
            lambda: QDesktopServices.openUrl(QUrl(PROJECT_REPOSITORY)),
        )
        menu.addAction("Report an issue", lambda: QDesktopServices.openUrl(QUrl(PROJECT_ISSUES)))
        return menu

    @staticmethod
    def _package_version() -> str:
        from . import __version__
        return __version__

    def _software_citation(self) -> str:
        return citation_text()

    def _software_bibtex(self) -> str:
        return bibtex_text()

    def _machine_page(self) -> QWidget:
        page = self._page("Machine Lattice")
        scroll = QScrollArea()
        scroll.setObjectName("machineComponentsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        content.setObjectName("machineComponentsContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 8, 12)
        content_layout.setSpacing(14)
        content_layout.setSizeConstraint(QLayout.SetMinimumSize)
        scroll.setWidget(content)
        page.layout().addWidget(scroll, 1)
        self.machine_lattice_intro = QLabel(
            "Load or select machine elements for LOCO. BPMs and correctors define the measured devices; "
            "exclusions remove selected devices from fitting. Mapping links measured devices to AT lattice elements."
        )
        self.machine_lattice_intro.setWordWrap(True)
        content_layout.addWidget(self.machine_lattice_intro)
        self.lattice_path = QLabel("No lattice selected")
        self.lattice_path.setWordWrap(True)
        self.lattice_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lattice_type = QLabel("—")
        self.lattice_variable = QLabel("Not applicable")
        self.lattice_elements = QLabel("Unknown", content)
        choose = QPushButton("Select lattice/model file…")
        choose.setMaximumWidth(260)
        choose.clicked.connect(self.select_lattice)
        self.change_mat_variable_button = QPushButton("Change MAT variable…")
        self.change_mat_variable_button.setMaximumWidth(220)
        self.change_mat_variable_button.clicked.connect(self.change_mat_variable)
        form = QFormLayout()
        selection_header=QWidget(); selection_layout=QHBoxLayout(selection_header); selection_layout.setContentsMargins(0,0,0,0)
        selection_title=QLabel("Reference lattice"); selection_title.setStyleSheet("font-weight:700")
        selection_layout.addWidget(selection_title); selection_layout.addStretch(); selection_layout.addWidget(choose)
        form.addRow(selection_header)
        form.addRow("Path", self.lattice_path)
        form.addRow("Type", self.lattice_type)
        variable_row = QWidget(); variable_layout = QHBoxLayout(variable_row); variable_layout.setContentsMargins(0, 0, 0, 0)
        variable_layout.addWidget(self.lattice_variable, 1); variable_layout.addWidget(self.change_mat_variable_button)
        form.addRow("MAT lattice variable", variable_row)
        form.addRow("Number of lattice elements", self.lattice_elements)
        self.reference_model_info=QLabel("Not available"); self.reference_model_info.setObjectName("fitReferenceModelCard"); self.reference_model_info.setWordWrap(True); self.reference_model_info.setTextInteractionFlags(Qt.TextSelectableByMouse); self.reference_model_info.setMargin(14)
        form.addRow(self.reference_model_info)
        group = QGroupBox("Lattice selection and metadata")
        group.setLayout(form)
        content_layout.addWidget(group)

        self.machine_config_group = QGroupBox("Reusable machine configuration")
        machine_config_layout = QVBoxLayout(self.machine_config_group)
        machine_config_help = QLabel(
            "Save or load only lattice-validated element selections, selection provenance, family groups and "
            "measurement exclusions. Measurements, FIT stages, solver settings and output paths are not changed."
        )
        machine_config_help.setWordWrap(True)
        machine_config_layout.addWidget(machine_config_help)
        machine_config_buttons = QHBoxLayout()
        self.save_machine_configuration_button = QPushButton("Save machine configuration…")
        self.load_machine_configuration_button = QPushButton("Load machine configuration…")
        self.save_machine_configuration_button.clicked.connect(self.save_machine_configuration)
        self.load_machine_configuration_button.clicked.connect(self.load_machine_configuration)
        machine_config_buttons.addWidget(self.save_machine_configuration_button)
        machine_config_buttons.addWidget(self.load_machine_configuration_button)
        machine_config_buttons.addStretch(1)
        machine_config_layout.addLayout(machine_config_buttons)
        content_layout.addWidget(self.machine_config_group)

        self.element_count_labels = {}
        self.element_preview_tables = {}
        self.machine_elements_group = QGroupBox("Machine Elements")
        elements_layout = QVBoxLayout(self.machine_elements_group)
        elements_layout.setContentsMargins(16, 20, 16, 16)
        elements_layout.setSpacing(8)
        self.machine_elements_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.machine_elements_help = QLabel(
            "Define BPMs, correctors, quadrupoles and RF cavities after loading the lattice. "
            "Open Edit/Select to check or uncheck individual elements."
        )
        self.machine_elements_help.setWordWrap(True)
        self.machine_elements_help.setToolTip(
            "Edit/Select changes the saved selected-element list. Normal and skew quadrupoles can be "
            "included or excluded directly with those checkboxes; no separate quadrupole-exclusion field is needed."
        )
        elements_layout.addWidget(self.machine_elements_help)
        self.element_row_widgets = {}
        self.element_edit_buttons = {}
        self.element_group_buttons = {}
        for key, (label, _kind) in ELEMENT_ROLES.items():
            row_widget = QWidget()
            row_widget.setObjectName("machineElementRow")
            row_widget.setMinimumHeight(44)
            row_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(8, 5, 8, 5)
            row.setSpacing(14)
            component_label = QLabel(label)
            component_label.setMinimumWidth(180)
            component_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            row.addWidget(component_label)
            count = QLabel("0 selected")
            count.setMinimumWidth(100)
            count.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.element_count_labels[key] = count
            row.addWidget(count, 1)
            button = QPushButton("Edit/Select…")
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setFixedSize(126, 34)
            button.clicked.connect(lambda checked=False, role=key: self.edit_element_selection(role))
            row.addWidget(button, 0, Qt.AlignVCenter)
            if key in {"normal_quadrupole_ords", "skew_quadrupole_ords", "quadrupole_tilt_ords"}:
                group_button=QPushButton("Family groups…"); group_button.setFixedSize(126,34)
                group_button.clicked.connect(lambda checked=False, role=key:self.edit_family_groups(role))
                row.addWidget(group_button,0,Qt.AlignVCenter); self.element_group_buttons[key]=group_button
            self.element_row_widgets[key] = row_widget
            self.element_edit_buttons[key] = button
            elements_layout.addWidget(row_widget)
            table = QTableWidget(0, 4)
            table.setHorizontalHeaderLabels(["Selection position", "Lattice ordinal", "Element name(s)", "Element class"])
            configure_item_view(table)
            table.setMinimumHeight(90)
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.element_preview_tables[key] = table
            elements_layout.addWidget(table)
        content_layout.addWidget(self.machine_elements_group)
        exclusion_group = QGroupBox("BPM and corrector exclusions")
        exclusion_form = QFormLayout(exclusion_group)
        exclusion_form.setContentsMargins(16, 22, 16, 16)
        exclusion_form.setHorizontalSpacing(18)
        exclusion_form.setVerticalSpacing(12)
        exclusion_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        exclusion_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        exclusion_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        exclusion_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.exclusion_help = QLabel(
            "Enter zero-based positions within the selected BPM, H-corrector or V-corrector list — "
            "not AT lattice ordinals. Example: BPM position 27 is the 28th BPM in the selected BPM list."
        )
        self.exclusion_help.setWordWrap(True)
        self.exclusion_help.setToolTip(
            "These exclusions keep the original selected device list unchanged, but omit the corresponding "
            "measurement rows or columns from the fit. To change which devices are selected, use Edit/Select above."
        )
        exclusion_form.addRow(self.exclusion_help)
        self.bad_bpm_positions_edit = QLineEdit(); self.bad_bpm_positions_edit.setPlaceholderText("selected-list positions, e.g. 0, 4, 17")
        self.hcor_exclusions_edit = QLineEdit(); self.vcor_exclusions_edit = QLineEdit()
        self.exclusion_counts = QLabel(); self.exclusion_counts.setWordWrap(True); self.exclusion_counts.setMinimumHeight(36)
        for edit, label in ((self.bad_bpm_positions_edit, "Excluded BPM positions"), (self.hcor_exclusions_edit, "Excluded H-corrector positions"), (self.vcor_exclusions_edit, "Excluded V-corrector positions")):
            edit.setMinimumHeight(34)
            edit.setToolTip(
                "Enter zero-based positions in this selected device list, not AT lattice ordinals. "
                "The selected device identities remain saved; matching measurement rows or columns are omitted from FIT."
            )
            exclusion_form.addRow(label, edit); edit.editingFinished.connect(self._store_exclusions)
        exclusion_form.addRow("Counts", self.exclusion_counts)
        content_layout.addWidget(exclusion_group)
        measurement_mapping_group = QGroupBox("Measurement-device mapping")
        measurement_mapping_layout = QVBoxLayout(measurement_mapping_group)
        mapping_intro = QLabel(
            "Measured identities come from imported HDF5 metadata. AT FamName, AT Device and zero-based ordinal "
            "come from the active reference lattice. The recorded match method and steering compatibility are "
            "reviewed separately; array position is never identity. "
            "Find machine mappings verifies identities only — it does not change the FIT selections."
        )
        mapping_intro.setWordWrap(True); measurement_mapping_layout.addWidget(mapping_intro)
        self.measurement_mapping_steps = QLabel(
            "<b>Step 1 — Find device mappings</b><br>Match measured BPMs and correctors with the reference AT lattice.<br>"
            "<b>Step 2 — Review results</b><br>Check matched devices and resolve unmatched or ambiguous entries.<br>"
            "<b>Step 3 — Use mappings in FIT</b><br>Preview and confirm the BPM and corrector selections for the measured ORM.<br>"
            "<b>Step 4 — Validate</b><br>Verify the complete configuration before running LOCO."
        )
        self.measurement_mapping_steps.setWordWrap(True)
        self.measurement_mapping_steps.setObjectName("mappingHelpText")
        measurement_mapping_layout.addWidget(self.measurement_mapping_steps)
        mapping_actions = QHBoxLayout()
        self.find_measurement_mappings_button = QPushButton("Find machine mappings")
        self.create_measurement_mapping_template_button = QPushButton("Create mapping template…")
        self.load_measurement_mapping_button = QPushButton("Load completed mapping…")
        self.save_measurement_mapping_button = QPushButton("Save validated mapping…")
        self.find_measurement_mappings_button.clicked.connect(self.find_measurement_mappings)
        self.create_measurement_mapping_template_button.clicked.connect(self.create_measurement_mapping_template)
        self.load_measurement_mapping_button.clicked.connect(self.load_measurement_mapping)
        self.save_measurement_mapping_button.clicked.connect(self.save_validated_measurement_mapping)
        for button in (self.find_measurement_mappings_button, self.create_measurement_mapping_template_button,
                       self.load_measurement_mapping_button, self.save_measurement_mapping_button):
            mapping_actions.addWidget(button)
        mapping_actions.addStretch(1); measurement_mapping_layout.addLayout(mapping_actions)
        self.measurement_mapping_feedback = QLabel("Not run yet — load modern Measure data and a reference lattice.")
        self.measurement_mapping_feedback.setWordWrap(True); self.measurement_mapping_feedback.setObjectName("mappingHelpText")
        measurement_mapping_layout.addWidget(self.measurement_mapping_feedback)
        self.use_measurement_mappings_button = QPushButton("Use mapped devices in FIT…")
        self.use_measurement_mappings_button.setEnabled(False)
        self.use_measurement_mappings_button.clicked.connect(self.use_measurement_mappings_for_fit)
        measurement_mapping_layout.addWidget(self.use_measurement_mappings_button, 0, Qt.AlignLeft)
        mapping_filters = QHBoxLayout(); self.measurement_mapping_search = QLineEdit()
        self.measurement_mapping_search.setPlaceholderText("Search measured device, AT FamName or AT Device")
        self.measurement_mapping_filter = QComboBox()
        self.measurement_mapping_filter.addItem("Needs attention", "attention")
        self.measurement_mapping_filter.addItem("All mappings", "all")
        self.measurement_mapping_filter.addItem("Validated only", "validated")
        self.measurement_mapping_search.textChanged.connect(self._refresh_measurement_mapping_table)
        self.measurement_mapping_filter.currentIndexChanged.connect(self._refresh_measurement_mapping_table)
        mapping_filters.addWidget(self.measurement_mapping_search, 1); mapping_filters.addWidget(self.measurement_mapping_filter)
        measurement_mapping_layout.addLayout(mapping_filters)
        self.measurement_mapping_table = CopyableMappingTable(0, 9)
        self.measurement_mapping_table.setHorizontalHeaderLabels(
            ["Measured device (HDF5)", "Role / plane", "AT FamName", "AT Device", "AT ordinal (0-based)", "Match method", "Component / steering capability", "Mapping status", "Evidence / problem"]
        )
        configure_mapping_table(self.measurement_mapping_table,(
            "Exact ordered identity stored in imported ORM HDF5 metadata.","Measured role and plane.","FamName attribute from the selected AT lattice element.","Device attribute from the selected AT lattice element.","Zero-based ordinal in the active AT lattice.","Recorded identity_match method, including explicit legacy suffix interpretation.","Required component and independently validated physical steering capability.","Revalidated mapping state.","Separate identity and steering evidence or an actionable validation error."
        ))
        self.measurement_mapping_table.setMinimumHeight(170)
        measurement_mapping_layout.addWidget(self.measurement_mapping_table)
        self.measurement_mapping_table_hint = QLabel(
            "No mapping records yet. Select Find machine mappings after loading measurements and a reference lattice."
        )
        self.measurement_mapping_table_hint.setWordWrap(True)
        self.measurement_mapping_table_hint.setObjectName("mappingHelpText")
        measurement_mapping_layout.addWidget(self.measurement_mapping_table_hint)
        mapping_help = QGroupBox("Mapping workflow and technical details"); mapping_help.setCheckable(True); mapping_help.setChecked(False)
        mapping_help_layout = QVBoxLayout(mapping_help)
        mapping_help_text = QLabel(
            "Find machine mappings uses exact AT Device or FamName identities, with explicit legacy suffix compatibility. Create writes JSON; Load accepts JSON, "
            "YAML and YML. For unresolved entries, fill in only lattice_ordinal. Roles, AT identity, B1L/A1L "
            "component, units and validation status are derived again when loaded. A typed ordinal is only a proposal. "
            "Validated mappings are bound to the lattice checksum, machine configuration checksum and ordered inventory."
        )
        mapping_help_text.setWordWrap(True); mapping_help_layout.addWidget(mapping_help_text)
        mapping_help_text.setVisible(False); mapping_help.toggled.connect(mapping_help_text.setVisible)
        measurement_mapping_layout.addWidget(mapping_help)
        content_layout.addWidget(measurement_mapping_group)
        content_layout.addStretch(1)
        return page

    def _measurements_page(self) -> QWidget:
        page = self._page("Measurement Import")
        self.measurement_role = QComboBox()
        self.measurement_role.addItems(
            ["orm", "dispersion", "bpm_noise", "other"]
        )
        import_button = QPushButton("Import HDF5, MAT, NumPy…")
        import_button.clicked.connect(self.import_measurement)
        session_button=QPushButton("Open Measurement Session…"); session_button.clicked.connect(self.open_measurement_session)
        self.measurement_list = QTableWidget(0, 6)
        self.measurement_list.setHorizontalHeaderLabels(
            ["Measurement role", "Filename", "Date", "Time", "Status", "Details"]
        )
        self.measurement_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.measurement_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.measurement_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.measurement_list.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.measurement_list.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.measurement_list.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.measurement_list.verticalHeader().setDefaultSectionSize(40)
        self.measurement_list.verticalHeader().setMinimumSectionSize(40)
        configure_item_view(self.measurement_list)
        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset role"))
        row.addWidget(self.measurement_role)
        row.addWidget(import_button)
        row.addWidget(session_button)
        group = QGroupBox("File import")
        layout = QVBoxLayout(group)
        layout.addLayout(row)
        self.measurement_hint = QLabel(); self.measurement_hint.setWordWrap(True)
        self.active_measurement_session_label = QLabel("Active measurement session: none")
        self.active_measurement_session_label.setObjectName("activeMeasurementSession")
        self.active_measurement_session_label.setWordWrap(True)
        self.active_measurement_session_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.measurement_metadata_label = QLabel("No measurement metadata loaded."); self.measurement_metadata_label.setWordWrap(True)
        self.measurement_file_status = QLabel("No measurement files imported.")
        self.measurement_file_status.setObjectName("measurementFileStatus")
        self.measurement_file_status.setWordWrap(True)
        self.locate_missing_measurement_button = QPushButton("Locate missing measurement…")
        self.locate_missing_measurement_button.clicked.connect(self.locate_missing_measurement)
        self.locate_missing_measurement_button.setEnabled(False)
        file_status_row = QHBoxLayout()
        file_status_row.addWidget(self.measurement_file_status, 1)
        file_status_row.addWidget(self.locate_missing_measurement_button)
        layout.addWidget(self.measurement_hint); layout.addWidget(self.active_measurement_session_label); layout.addWidget(self.measurement_metadata_label)
        layout.addLayout(file_status_row)
        layout.addWidget(self.measurement_list)
        page.layout().addWidget(group)
        self.measurement_role.currentTextChanged.connect(self._update_measurement_hint)
        self._update_measurement_hint(self.measurement_role.currentText())
        return page


    def _fit_page(self) -> QWidget:
        page = self._page("LOCO Configuration")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.fit_scroll = scroll
        container = QWidget()
        layout = QVBoxLayout(container)

        self.fit_configuration_actions = QGroupBox("Reusable FIT configuration")
        action_layout = QVBoxLayout(self.fit_configuration_actions)
        action_help = QLabel(
            "Import an existing fitting configuration or export the current settings for reuse. "
            "This action covers the current working stage's LOCO settings, not the multi-stage recipe. "
            "Use Load fitting stages below for a complete stage sequence."
        )
        action_help.setWordWrap(True)
        action_layout.addWidget(action_help)
        action_row = QHBoxLayout()
        self.import_configuration_button = QPushButton("Import configuration…")
        self.import_configuration_button.clicked.connect(self.import_loco_configuration)
        self.export_configuration_button = QPushButton("Export configuration…")
        self.export_configuration_button.clicked.connect(self.export_loco_configuration)
        self.review_configuration_button = QPushButton("Review configuration")
        self.review_configuration_button.clicked.connect(self._scroll_to_fit_summary)
        action_row.addWidget(self.import_configuration_button)
        action_row.addWidget(self.export_configuration_button)
        action_row.addStretch(1)
        action_row.addWidget(self.review_configuration_button)
        action_layout.addLayout(action_row)
        layout.addWidget(self.fit_configuration_actions)

        self.pre_fit_checks_group = QGroupBox("Pre-FIT measurement checks")
        pre_fit_layout = QVBoxLayout(self.pre_fit_checks_group)
        pre_fit_help = QLabel(
            "Compare the imported measurements with the selected reference lattice before fitting. "
            "This read-only FIT check does not acquire data, run the optimizer or change measurements. "
            "After changing measurements, lattice, selections or exclusions, use Validate first. The "
            "comparison also rechecks compatibility and refuses incompatible inputs."
        )
        pre_fit_help.setWordWrap(True)
        pre_fit_layout.addWidget(pre_fit_help)
        self.compare_reference_button = QPushButton("Compare imported measurements with initial model…")
        self.compare_reference_button.clicked.connect(self.compare_measurements_with_reference)
        self.compare_reference_button.setToolTip(
            "Calculate a read-only reference ORM and RF response before fitting; no optimizer or machine writes."
        )
        pre_fit_row = QHBoxLayout(); pre_fit_row.addWidget(self.compare_reference_button); pre_fit_row.addStretch(1)
        pre_fit_layout.addLayout(pre_fit_row)
        layout.addWidget(self.pre_fit_checks_group)

        workflow_group = QGroupBox("Multi-stage fitting workflow")
        workflow_layout = QVBoxLayout(workflow_group)
        self.fit_workflow_table = QTableWidget(0, 5)
        self.fit_workflow_table.setHorizontalHeaderLabels(
            ["Stage", "Name", "Parameters fitted in this stage", "Iterations", "Starting model"]
        )
        configure_item_view(self.fit_workflow_table)
        self.fit_workflow_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.fit_workflow_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.fit_workflow_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.fit_workflow_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.fit_workflow_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        # Keep workflow rows compact.  The Starting-model editor is placed in
        # a centred wrapper below rather than being stretched to the cell.
        self.fit_workflow_table.verticalHeader().setMinimumSectionSize(38)
        self.fit_workflow_table.verticalHeader().setDefaultSectionSize(38)
        self.fit_workflow_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fit_workflow_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.fit_workflow_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.fit_workflow_table.itemSelectionChanged.connect(self._select_fit_stage)
        self.fit_workflow_table.itemChanged.connect(self._on_fit_workflow_item_changed)
        self.fit_workflow_table.itemDoubleClicked.connect(self._on_fit_workflow_item_double_clicked)
        workflow_layout.addWidget(self.fit_workflow_table)
        self.fit_workflow_hint = QLabel(
            "Selecting a stage edits its configuration; it does not limit execution. Run LOCO executes the entire "
            "validated recipe automatically. Double-click a stage name or iteration count to edit it, use the "
            "Starting model selector for inheritance, and hover a row for its full setting summary."
        )
        self.fit_workflow_hint.setWordWrap(True)
        self.fit_workflow_hint.setObjectName("fitWorkflowExecutionHelp")
        workflow_layout.addWidget(self.fit_workflow_hint)
        self.fit_stage_editor_label = QLabel()
        self.fit_stage_editor_label.setObjectName("dashboardCardTitle")
        self.fit_stage_editor_label.setWordWrap(True)
        workflow_layout.addWidget(self.fit_stage_editor_label)
        self.fit_empty_state_label = QLabel(
            "No fitting stages configured. Add the first FIT stage to begin. "
            "A new stage uses pyLOCO's documented initial defaults and must be validated before running."
        )
        self.fit_empty_state_label.setWordWrap(True)
        self.fit_empty_state_label.setObjectName("validationMissing")
        self.fit_empty_state_label.setToolTip(
            "Initial Stage 1 defaults: Levenberg–Marquardt; 1 outer and 10 inner iterations; "
            "scaled solve; threshold SVD at 1e-7; outlier rejection at 10 sigma; normalization, "
            "dispersion and constraints off. Default fitted groups are normal quadrupoles, H/V BPM gains, "
            "H/V corrector calibration and horizontal-corrector energy shift. Validate and review these "
            "choices before running."
        )
        self.fit_empty_state_label.setVisible(False)
        workflow_layout.addWidget(self.fit_empty_state_label)
        stage_buttons = QHBoxLayout(); session_buttons = QHBoxLayout()
        self.fit_stage_buttons = {}
        for text, slot in (("+ Add stage", self._add_fit_stage), ("Duplicate stage", self._duplicate_fit_stage),
                           ("Remove", self._remove_fit_stage), ("Move up", lambda: self._move_fit_stage(-1)),
                           ("Move down", lambda: self._move_fit_stage(1)),
                           ("Review fitting sequence", self.preview_fit_workflow)):
            button = QPushButton(text); button.clicked.connect(slot); stage_buttons.addWidget(button)
            self.fit_stage_buttons[text] = button
        self.clear_fit_configuration_button = QPushButton("Clear FIT configuration…")
        self.clear_fit_configuration_button.clicked.connect(self.clear_fit_configuration)
        stage_buttons.addWidget(self.clear_fit_configuration_button)
        stage_buttons.addStretch(1); workflow_layout.addLayout(stage_buttons)
        for text, slot in (("Save fitting stages…", self.save_fit_recipe),
                           ("Load fitting stages…", self.load_fit_recipe),
                           ("Save complete fitting session…", self.save_fit_run_session),
                           ("Resume previous fitting run…", self.resume_fit_run)):
            button = QPushButton(text); button.clicked.connect(slot); session_buttons.addWidget(button)
            if text.startswith("Save fitting stages"):
                self.save_fit_recipe_button = button
            if text.startswith("Save complete"):
                self.save_fit_run_session_button = button
            if text.startswith("Resume previous"):
                self.resume_fit_run_button = button
        session_buttons.addStretch(1); workflow_layout.addLayout(session_buttons)
        workflow_help = QLabel(
            "Fitting stages store the reusable sequence and settings. A complete fitting session also stores "
            "the selected lattice, measurements and completed-stage checkpoints."
        )
        workflow_help.setWordWrap(True); workflow_layout.addWidget(workflow_help)
        self.compatibility_status_label = QLabel()
        self.compatibility_status_label.setWordWrap(True)
        self.compatibility_status_label.setObjectName("validationMissing")
        self.compatibility_status_label.setVisible(False)
        workflow_layout.addWidget(self.compatibility_status_label)
        layout.addWidget(workflow_group)

        self.rm_calculator = QComboBox()
        self.rm_calculator.addItem("Linear (transfer matrix)", "Linear")
        self.rm_calculator.addItem("Analytical (uncoupled optics)", "Analytical")
        self.rm_calculator.addItem("Tracking", "Numerical")
        self.rm_calculator.setToolTip("Choose the backend ORM implementation. Analytical uses the uncoupled beta/phase formula; Tracking uses numerical closed-orbit perturbations.")
        self.rm_dispersion = QCheckBox("Include dispersion/RF response column")
        self.rm_dispersion.setToolTip("Append the response to an RF frequency shift to the ORM.")
        # Keep the legacy attribute as a compatibility alias.  There is only one
        # user-facing dispersion switch: the Response Matrix checkbox above.
        self.loco_include_dispersion = self.rm_dispersion
        self.loco_hor_dispersion_weight = self._double_spin(0.0, 1e9, 1.0, 6)
        self.loco_ver_dispersion_weight = self._double_spin(0.0, 1e9, 1.0, 6)
        dispersion_weight_help = (
            "Controls the relative contribution of this plane's dispersion "
            "residuals to the LOCO fit."
        )
        self.loco_hor_dispersion_weight.setToolTip(
            "Horizontal dispersion weight. " + dispersion_weight_help
        )
        self.loco_ver_dispersion_weight.setToolTip(
            "Vertical dispersion weight. " + dispersion_weight_help
        )
        self.dispersion_weight_controls = QWidget()
        self.dispersion_weight_controls.setObjectName("dispersionWeightControls")
        dispersion_weight_form = QFormLayout(self.dispersion_weight_controls)
        dispersion_weight_form.setContentsMargins(24, 2, 0, 4)
        dispersion_weight_form.setVerticalSpacing(8)
        dispersion_weight_form.addRow(
            "Horizontal dispersion weight", self.loco_hor_dispersion_weight
        )
        dispersion_weight_form.addRow(
            "Vertical dispersion weight", self.loco_ver_dispersion_weight
        )
        self.rm_coupling = QCheckBox("Include coupling ORM terms")
        self.rm_coupling.setToolTip("Include cross-plane response blocks in the ORM.")
        self.rm_bidirectional = QCheckBox("Bidirectional (+/- delta kick)")
        self.rm_bidirectional.setToolTip("Compute the ORM using positive and negative perturbations (central difference) instead of a single perturbation. This generally improves numerical accuracy.")
        self.rm_vectorized = QCheckBox("Use vectorized response calculation")
        self.rm_vectorized.setToolTip("Use the backend vectorized ORM path where available.")
        # Retain the legacy response-matrix kick values as non-user-facing
        # configuration state.  The authoritative corrector-step controls are
        # presented together below in the Corrector Steps group.
        self.rm_dkick_h = self._double_spin(0.0, 1e-1, 1e-5, 9, " rad")
        self.rm_dkick_v = self._double_spin(0.0, 1e-1, 1e-5, 9, " rad")
        self.rm_dkick_h.setParent(container); self.rm_dkick_v.setParent(container)
        self.rm_dkick_h.hide(); self.rm_dkick_v.hide()
        self.rm_rf_step = self._double_spin(-1e9, 1e9, -3000.0, 9, " Hz")
        self.rm_delta_coupling = self._double_spin(-1.0, 1.0, 1e-6, 9)
        self.rm_bpm_ords = QLineEdit()
        self.rm_cm_ords = QLineEdit()
        self.rm_cav_ords = QLineEdit()
        self.rm_hcm_coupling = QLineEdit()
        self.rm_vcm_coupling = QLineEdit()
        self.rm_frequency = QLineEdit()
        self.rm_harm_number = QLineEdit()
        self.rm_rf_attr = QLineEdit("Frequency")
        self.rm_fixedpath = QCheckBox("Apply fixed-circumference correction")
        self.rm_fixedpath.setToolTip(
            "Modify corrector-response columns by the dispersive momentum correction required to keep the closed-orbit path length at the reference circumference. "
            "This is separate from fitting the RF-dispersion column. This setting applies to the selected stage."
        )
        self.rm_log_info = QCheckBox("log_info")
        self.rm_rf_step.setToolTip("RF frequency step in Hz. Positive and negative shifts are supported and the sign is preserved.")
        self.rm_delta_coupling.setToolTip("Small dimensionless delta used to evaluate corrector coupling terms; scientific notation is accepted.")
        rm_form = QFormLayout()
        for label, widget in (
            ("Response Matrix Calculator", self.rm_calculator),
            ("RF frequency step", self.rm_rf_step),
            ("Coupling delta (dimensionless)", self.rm_delta_coupling),
        ):
            rm_form.addRow(label, widget)
        rm_form.addRow(self.rm_dispersion)
        rm_form.addRow(self.dispersion_weight_controls)
        for widget in (self.rm_coupling, self.rm_bidirectional, self.rm_vectorized, self.rm_fixedpath, self.rm_log_info):
            rm_form.addRow(widget)
        rm_group = QGroupBox("Response Matrix")
        rm_group.setLayout(rm_form)
        layout.addWidget(rm_group)

        self.solver_algorithm = QComboBox()
        self.solver_algorithm.addItem("Levenberg–Marquardt", "lm")
        self.solver_algorithm.addItem("Gauss–Newton", "gn")
        self.solver_n_iter = self._spin(1, 100, 1)
        self.solver_lm_iter = self._spin(0, 100, 10)
        self.solver_lambda = self._double_spin(0.0, 1e9, 1e-3, 9)
        self.solver_max_lambda = self._double_spin(0.0, 1e9, 15.0, 3)
        self.solver_scaled = QCheckBox("Scaled Levenberg–Marquardt")
        self.solver_scaled.setToolTip(
            "Enable the scaled-variable formulation of the Levenberg–Marquardt "
            "algorithm to improve conditioning when fit parameters have very different magnitudes."
        )
        solver_form = QFormLayout()
        for label, widget in (
            ("Algorithm", self.solver_algorithm),
            ("Outer iterations", self.solver_n_iter),
            ("LM inner iterations", self.solver_lm_iter),
            ("Starting lambda", self.solver_lambda),
            ("Maximum lambda", self.solver_max_lambda),
        ):
            solver_form.addRow(label, widget)
        solver_form.addRow(self.solver_scaled)
        solver_group = QGroupBox("Solver")
        solver_group.setLayout(solver_form)
        self._lm_only_controls = (self.solver_lm_iter, self.solver_lambda, self.solver_max_lambda, self.solver_scaled)
        layout.addWidget(solver_group)

        self.svd_method = QComboBox()
        for label, method in (
            ("Threshold", "threshold"),
            ("Rank", "rank"),
            ("User-defined cut", "user_input"),
            ("Interactive", "interactive"),
        ):
            self.svd_method.addItem(label, method)
        self.svd_threshold = self._double_spin(0.0, 1.0, 1e-7, 10)
        self.svd_rank = self._spin(0, 100000, 500)
        self.svd_parameter_label = QLabel("Threshold")
        self.svd_parameter_input = QWidget()
        svd_parameter_layout = QHBoxLayout(self.svd_parameter_input)
        svd_parameter_layout.setContentsMargins(0, 0, 0, 0)
        svd_parameter_layout.addWidget(self.svd_threshold)
        svd_parameter_layout.addWidget(self.svd_rank)
        self.svd_plot = QCheckBox("Show SVD plot")
        svd_form = QFormLayout()
        svd_form.addRow("Selection method", self.svd_method)
        svd_form.addRow(self.svd_parameter_label, self.svd_parameter_input)
        svd_form.addRow(self.svd_plot)
        svd_group = QGroupBox("SVD")
        svd_group.setLayout(svd_form)
        layout.addWidget(svd_group)

        self.quad_jacobian_calculator = QComboBox()
        self.quad_jacobian_calculator.addItems(["Numerical", "Analytical"])
        self.skew_jacobian_calculator = QComboBox()
        self.skew_jacobian_calculator.addItems(["Numerical", "Analytical"])
        self.analytical_thick_quadrupole = QCheckBox("Thick normal quadrupoles")
        self.analytical_thick_steerers = QCheckBox("Thick steerers for normal Jacobian")
        self.analytical_verbose = QCheckBox("Verbose normal analytical calculation")
        self.analytical_use_mp = QCheckBox("Multiprocessing for normal analytical calculation")
        self.analytical_implementation = QComboBox()
        self.analytical_implementation.addItem("Vectorized (production)", "vectorized")
        self.analytical_implementation.addItem("Legacy (reference)", "legacy")
        self.analytical_dispersion_calculator = QComboBox()
        self.analytical_dispersion_calculator.addItem("Same as model ORM (compatible)", None)
        for calculator in ("Linear", "Analytical", "Tracking"):
            self.analytical_dispersion_calculator.addItem(calculator, calculator)
        self.analytical_thick_skew = QCheckBox("Thick skew quadrupoles")
        self.analytical_skew_thick_steerers = QCheckBox("Thick steerers for skew Jacobian")
        self.analytical_skew_verbose = QCheckBox("Verbose skew analytical calculation")
        self.analytical_skew_use_mp = QCheckBox("Multiprocessing for skew analytical calculation")
        self.skew_analytical_implementation = QComboBox()
        self.skew_analytical_implementation.addItem("Vectorized (production)", "vectorized")
        self.skew_analytical_implementation.addItem("Legacy (reference)", "legacy")
        self.skew_analytical_dispersion_calculator = QComboBox()
        self.skew_analytical_dispersion_calculator.addItem("Same as model ORM (compatible)", None)
        for calculator in ("Linear", "Analytical", "Tracking"):
            self.skew_analytical_dispersion_calculator.addItem(calculator, calculator)
        self.skew_analytical_dispersion_worker = QComboBox()
        self.skew_analytical_dispersion_worker.addItem("Legacy full ORM", "legacy_full_orm")
        self.skew_analytical_dispersion_worker.addItem("RF column only", "rf_only")

        normal_options_form = QFormLayout()
        normal_options_form.addRow("Implementation", self.analytical_implementation)
        normal_options_form.addRow(
            "Dispersion calculator", self.analytical_dispersion_calculator
        )
        for widget in (
            self.analytical_thick_quadrupole,
            self.analytical_thick_steerers,
            self.analytical_verbose,
            self.analytical_use_mp,
        ):
            normal_options_form.addRow(widget)
        self.normal_analytical_options = QGroupBox("Normal analytical options")
        self.normal_analytical_options.setLayout(normal_options_form)

        skew_options_form = QFormLayout()
        skew_options_form.addRow("Implementation", self.skew_analytical_implementation)
        skew_options_form.addRow("Dispersion calculator", self.skew_analytical_dispersion_calculator)
        skew_options_form.addRow("Dispersion worker", self.skew_analytical_dispersion_worker)
        for widget in (
            self.analytical_thick_skew,
            self.analytical_skew_thick_steerers,
            self.analytical_skew_verbose,
            self.analytical_skew_use_mp,
        ):
            skew_options_form.addRow(widget)
        self.skew_analytical_options = QGroupBox("Skew analytical options")
        self.skew_analytical_options.setLayout(skew_options_form)

        jacobian_form = QFormLayout()
        jacobian_form.addRow("Normal quadrupole Jacobian", self.quad_jacobian_calculator)
        jacobian_form.addRow(self.normal_analytical_options)
        jacobian_form.addRow("Skew quadrupole Jacobian", self.skew_jacobian_calculator)
        jacobian_form.addRow(self.skew_analytical_options)
        self.fixed_dk = QLineEdit()
        self.fixed_dk.setPlaceholderText("Automatic per parameter (None)")
        self.fixed_dk.setToolTip(
            "Blank means no fixed normal-quadrupole step. For a numerical Jacobian, pyLOCO starts "
            "from 0.001 × each fitted K (or 0.001 when K is zero), then adapts the step when automatic adjustment is enabled."
        )
        self.fixed_delta_skew = self._double_spin(-1.0, 1.0, 1e-3, 9)
        self.fixed_delta_q_tilt = self._double_spin(-1.0, 1.0, 1e-6, 9)
        self.perturbation_group=QGroupBox("Numerical perturbation steps")
        perturbation_form=QFormLayout(self.perturbation_group)
        for label, widget in (("Normal quadrupole step", self.fixed_dk), ("Skew strength step [lattice units]", self.fixed_delta_skew), ("Quadrupole tilt step [rad]", self.fixed_delta_q_tilt)):
            perturbation_form.addRow(label, widget)
        self.normal_step_resolution = QLabel()
        self.normal_step_resolution.setWordWrap(True)
        perturbation_form.addRow("Resolved normal step", self.normal_step_resolution)
        self.skew_step_status = QLabel(); self.tilt_step_status = QLabel()
        perturbation_form.addRow("Skew step status", self.skew_step_status)
        perturbation_form.addRow("Tilt step status", self.tilt_step_status)
        jacobian_form.addRow(self.perturbation_group)
        self.jacobian_group = QGroupBox("Jacobian calculation")
        self.jacobian_group.setLayout(jacobian_form)
        layout.addWidget(self.jacobian_group)

        self.outlier_enabled = QCheckBox("Reject outliers")
        self.outlier_sigma = self._double_spin(0.0, 1e6, 10.0, 3)
        self.norm_enabled = QCheckBox("Apply normalization")
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["component", "global", "none"])
        self.auto_delta = QCheckBox("Automatically optimize Jacobian step sizes")
        self.loco_fixedpath = QCheckBox("Keep path length fixed")
        self.loco_individuals = QCheckBox("individuals")
        self.loco_remove_coupling = QCheckBox("Exclude cross-plane ORM responses from the fit")
        self.loco_plot_fit_parameters = QCheckBox("Generate fitted-parameter plots")
        rej_layout = QVBoxLayout()
        outlier_group = QGroupBox("Outlier rejection")
        outlier_form = QFormLayout(outlier_group)
        outlier_form.addRow(self.outlier_enabled)
        outlier_form.addRow("Sigma cut", self.outlier_sigma)
        self.outlier_enabled.setToolTip("Remove residual entries beyond the two-pass sigma threshold before solving; removed entries are also excluded from χ².")
        normalization_group = QGroupBox("Residual normalization")
        normalization_form = QFormLayout(normalization_group)
        normalization_form.addRow(self.norm_enabled)
        normalization_form.addRow("Normalization mode", self.norm_mode)
        self.norm_enabled.setToolTip("Rescale weighted Jacobian columns before solving; component mode treats parameter blocks separately.")
        calculation_group = QGroupBox("Calculation behavior")
        calculation_form = QFormLayout(calculation_group)
        self.loco_individuals.hide()  # compatibility state; superseded by explicit modes
        self.auto_delta.setToolTip("Adapt finite-difference perturbations to obtain a numerically useful ORM change.")
        self.loco_fixedpath.setText("Apply fixed-circumference correction")
        self.loco_fixedpath.setToolTip("Modify corrector-response columns by the dispersive momentum correction required by a fixed ring circumference. This does not add the RF-dispersion column. Applies to this stage.")
        self.loco_remove_coupling.setToolTip("Exclude H→Y and V→X from residuals, Jacobian, outlier selection and χ² for this stage. Raw measurement arrays remain unchanged.")
        self.loco_plot_fit_parameters.setToolTip("Generate parameter-history plots; this changes output only, not fitted values.")
        self.loco_fixedpath.hide()  # compatibility mirror; the Response Matrix control is authoritative in the UI
        for widget in (self.auto_delta, self.loco_remove_coupling, self.loco_plot_fit_parameters):
            calculation_form.addRow(widget)
        rej_layout.addWidget(outlier_group); rej_layout.addWidget(normalization_group); rej_layout.addWidget(calculation_group)
        rej_group = QGroupBox("Fit data treatment and diagnostics")
        rej_group.setLayout(rej_layout)
        layout.addWidget(rej_group)

        self.constraint_enabled = QCheckBox("Enable constraints")
        self.constraint_quad_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_skew_sigma = self._double_spin(0.0, 1e12, 0.0, 6)
        self.constraint_quad_weights = QLineEdit()
        self.constraint_skew_weights = QLineEdit()
        self.constraint_quad_mask = QLineEdit()
        self.constraint_skew_mask = QLineEdit()
        self.constraint_quad_sigma_mode = QComboBox()
        self.constraint_quad_sigma_mode.addItem("Absolute σ", "absolute")
        self.constraint_quad_sigma_mode.addItem("Relative σ × |K|", "relative")
        self.constraint_quad_relative_sigma = self._double_spin(1e-15, 1.0, 1e-4, 10)
        self.constraint_quad_minimum_sigma = self._double_spin(0.0, 1.0, 1e-12, 12)
        self.constraint_quad_default_weight = self._double_spin(0.0, 1e12, 1.0, 8)
        self.constraint_quad_selected_families = QLineEdit()
        self.constraint_quad_selected_families.setPlaceholderText("e.g. 12, 27, 35")
        self.constraint_quad_selected_weight = self._double_spin(0.0, 1e12, 1.0, 8)
        self.constraint_quad_exceptions = FamilyWeightEditor()
        self.constraint_skew_default_weight = self._double_spin(0.0, 1e12, 1.0, 8)
        self.constraint_skew_selected_families = QLineEdit()
        self.constraint_skew_selected_families.setPlaceholderText("e.g. 0, 3")
        self.constraint_skew_selected_weight = self._double_spin(0.0, 1e12, 1.0, 8)
        self.constraint_skew_exceptions = FamilyWeightEditor()
        constraint_form = QFormLayout()
        constraint_form.addRow("Quadrupole sigma definition", self.constraint_quad_sigma_mode)
        constraint_form.addRow("Quadrupole sigma", self.constraint_quad_sigma)
        constraint_form.addRow("Relative quadrupole sigma", self.constraint_quad_relative_sigma)
        constraint_form.addRow("Minimum quadrupole sigma", self.constraint_quad_minimum_sigma)
        constraint_form.addRow("Default quadrupole weight", self.constraint_quad_default_weight)
        constraint_form.addRow("Selected quadrupole families", self.constraint_quad_selected_families)
        constraint_form.addRow("Common selected-family weight", self.constraint_quad_selected_weight)
        constraint_form.addRow("Quadrupole weight exceptions", self.constraint_quad_exceptions)
        constraint_form.addRow("Skew sigma", self.constraint_skew_sigma)
        constraint_form.addRow("Default skew weight", self.constraint_skew_default_weight)
        constraint_form.addRow("Selected skew families", self.constraint_skew_selected_families)
        constraint_form.addRow("Common selected-skew weight", self.constraint_skew_selected_weight)
        constraint_form.addRow("Skew weight exceptions", self.constraint_skew_exceptions)
        constraint_form.addRow("Quadrupole weights", self.constraint_quad_weights)
        constraint_form.addRow("Skew weights", self.constraint_skew_weights)
        constraint_form.addRow("Quadrupole mask", self.constraint_quad_mask)
        constraint_form.addRow("Skew mask", self.constraint_skew_mask)
        self.constraint_details = QWidget()
        self.constraint_details.setLayout(constraint_form)
        constraint_group = QGroupBox("Constraints")
        constraint_layout = QVBoxLayout(constraint_group)
        constraint_layout.addWidget(self.constraint_enabled)
        constraint_layout.addWidget(self.constraint_details)
        layout.addWidget(constraint_group)

        self.parameter_checks = {}
        self.parameter_group = QGroupBox("Parameter Selection")
        param_layout = QVBoxLayout(self.parameter_group)
        self.parameterization_radios = {}
        for key, label in (("quads", "Normal quadrupoles"), ("skew_quads", "Skew quadrupoles"), ("quads_tilt", "Quadrupole tilts")):
            box=QGroupBox(label); row=QHBoxLayout(box); check=QCheckBox("Fit"); individual=QRadioButton("Individually"); family=QRadioButton("By explicit family/group"); individual.setChecked(True)
            self.parameter_checks[key]=check; self.parameterization_radios[key]=(individual,family)
            row.addWidget(check); row.addStretch(1); row.addWidget(individual); row.addWidget(family); param_layout.addWidget(box)
        for key, label in (
            ("hbpm_gain", "Horizontal BPM gains"), ("vbpm_gain", "Vertical BPM gains"),
            ("hbpm_coupling", "Horizontal BPM coupling"), ("vbpm_coupling", "Vertical BPM coupling"),
            ("hcor_cal", "Horizontal corrector calibration"), ("vcor_cal", "Vertical corrector calibration"),
            ("hcor_coupling", "Horizontal corrector coupling"), ("vcor_coupling", "Vertical corrector coupling"),
            ("HCMEnergyShift", "Horizontal corrector energy shifts"), ("VCMEnergyShift", "Vertical corrector energy shifts"),
            ("delta_rf", "RF frequency shift"),
        ):
            check = QCheckBox(label)
            self.parameter_checks[key] = check
            param_layout.addWidget(check)
        self.cmstep_mode = QComboBox()
        self.cmstep_mode.addItem("Uniform", "uniform")
        self.cmstep_mode.addItem("Load from file", "file")
        # Hidden compatibility storage.  This is edited through the structured
        # policy dialog and must never become a parentless top-level window.
        self.params_init_policy = QLineEdit(self)
        self.params_init_policy.setPlaceholderText("Uses DEFAULT_INIT_POLICY unless overridden")
        self.params_init_policy.setToolTip(
            "Advanced Python-literal dictionary. Supported values: ones, zeros, cmstep:h, cmstep:v, "
            "rfstep, quads:K, quads:attr, quads:Attribute[index], and tilts:zeros."
        )
        self.params_init_policy.hide()
        self.params_init_policy_summary = QLabel("Built-in defaults")
        self.params_init_policy_summary.setWordWrap(True)
        self.params_init_policy_edit = QPushButton("Edit policies…")
        self.params_init_policy_edit.clicked.connect(self._edit_initialization_policies)
        self.params_init_policy_row = QWidget()
        init_policy_row_layout = QHBoxLayout(self.params_init_policy_row)
        init_policy_row_layout.setContentsMargins(0, 0, 0, 0)
        init_policy_row_layout.addWidget(self.params_init_policy_summary, 1)
        init_policy_row_layout.addWidget(self.params_init_policy_edit)
        self.params_cmstep_h = self._double_spin(-1e-1, 1e-1, 1e-5, 9, " rad")
        self.params_cmstep_v = self._double_spin(-1e-1, 1e-1, 1e-5, 9, " rad")
        for spin, plane in ((self.params_cmstep_h, "Horizontal"), (self.params_cmstep_v, "Vertical")):
            spin.setSingleStep(1e-5)
            spin.setToolTip(
                f"{plane} corrector kick step in radians. Scientific notation such as 1e-4 is accepted; "
                "100 µrad equals 1e-4 rad. The mouse wheel changes this value only while the control has focus."
            )
        self.params_cmstep_file = QLineEdit()
        self.params_cmstep_browse = QPushButton("Browse…")
        self.params_cmstep_browse.clicked.connect(self._browse_cmstep_file)
        self.params_cmstep_file_row = QWidget()
        cmstep_file_layout = QHBoxLayout(self.params_cmstep_file_row)
        cmstep_file_layout.setContentsMargins(0, 0, 0, 0)
        cmstep_file_layout.addWidget(self.params_cmstep_file, 1)
        cmstep_file_layout.addWidget(self.params_cmstep_browse)
        self.params_cmstep_h_label = QLabel("Horizontal kick step")
        self.params_cmstep_v_label = QLabel("Vertical kick step")
        self.params_cmstep_file_label = QLabel("Corrector-step .npz file")
        self.params_cmstep_resolved = QWidget()
        resolved_layout = QFormLayout(self.params_cmstep_resolved)
        resolved_layout.setContentsMargins(0, 2, 0, 0)
        self.params_cmstep_resolved_h = QLabel("No corrector-step file loaded")
        self.params_cmstep_resolved_v = QLabel("No corrector-step file loaded")
        self.params_cmstep_resolved_h.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.params_cmstep_resolved_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
        resolved_layout.addRow("Resolved horizontal values", self.params_cmstep_resolved_h)
        resolved_layout.addRow("Resolved vertical values", self.params_cmstep_resolved_v)
        self.params_rfstep = self._double_spin(-1e9, 1e9, -3000.0, 9)
        # Hidden compatibility storage.  The user-facing editor is the modal
        # dialog opened by ``params_init_edit`` below.
        self.params_init = QLineEdit(self)
        self.params_init.setPlaceholderText("Optional dictionary, e.g. {'hbpm_gain': 1.0}")
        self.params_init.setToolTip(
            "Advanced Python-literal dictionary keyed by parameter block. A scalar is broadcast; "
            "an array must match that block's parameter count. Explicit values override the policy for that block."
        )
        self.params_init.hide()
        self.params_init_summary = QLabel("Optional — no explicit overrides")
        self.params_init_summary.setWordWrap(True)
        self.params_init_edit = QPushButton("Edit optional overrides…")
        self.params_init_edit.clicked.connect(self._edit_explicit_initial_values)
        self.params_init_row = QWidget(); params_init_row_layout = QHBoxLayout(self.params_init_row)
        params_init_row_layout.setContentsMargins(0, 0, 0, 0)
        params_init_row_layout.addWidget(self.params_init_summary, 1); params_init_row_layout.addWidget(self.params_init_edit)
        self.params_quads_attr = QLineEdit("PolynomB")
        self.params_quads_attr_index = self._spin(0, 100, 1)
        self.params_skew_attr = QLineEdit("PolynomA")
        self.params_skew_attr_index = self._spin(0, 100, 1)
        self.params_tilt_attr_r1 = QLineEdit("R1")
        self.params_tilt_attr_r2 = QLineEdit("R2")
        self.params_tilt_method = QLineEdit("set")
        cm_group = QGroupBox("Corrector Steps")
        cm_form = QFormLayout(cm_group)
        cm_form.addRow("Corrector step mode", self.cmstep_mode)
        cm_form.addRow(self.params_cmstep_h_label, self.params_cmstep_h)
        cm_form.addRow(self.params_cmstep_v_label, self.params_cmstep_v)
        cm_form.addRow(self.params_cmstep_file_label, self.params_cmstep_file_row)
        cm_form.addRow(self.params_cmstep_resolved)
        cm_form.addRow("Initialization RF step [Hz]", self.params_rfstep)
        param_layout.addWidget(cm_group)

        init_group = QGroupBox("Fit Initialization")
        self.fit_init_group = init_group
        param_form = QFormLayout(init_group)
        init_help = QLabel(
            "Policies create the initial vector for this stage. Explicit initial values override a policy block. "
            "When a stage resumes, matching values from the previous fit state then replace the newly initialized blocks."
        )
        init_help.setWordWrap(True)
        param_form.addRow(init_help)
        param_form.addRow("Initialization policies", self.params_init_policy_row)
        param_form.addRow("Explicit initial-value overrides", self.params_init_row)
        for label, widget in (("Normal quadrupole attribute", self.params_quads_attr), ("Normal quadrupole attribute index", self.params_quads_attr_index), ("Skew quadrupole attribute", self.params_skew_attr), ("Skew quadrupole attribute index", self.params_skew_attr_index), ("Tilt R1 attribute", self.params_tilt_attr_r1), ("Tilt R2 attribute", self.params_tilt_attr_r2), ("Tilt update method", self.params_tilt_method)):
            param_form.addRow(label, widget)
        param_layout.addWidget(init_group)
        self._advanced_form_rows = (
            (cm_form, self.params_cmstep_h),
            (cm_form, self.params_cmstep_v),
            (cm_form, self.params_cmstep_file_row),
            (cm_form, self.params_cmstep_resolved),
            (cm_form, self.params_rfstep),
        )
        # Parameter choices determine which Jacobian controls are relevant, so
        # present them before the calculation details without changing either
        # the stored configuration or backend mapping.
        layout.insertWidget(layout.indexOf(self.jacobian_group), self.parameter_group)

        output_group = QGroupBox("Output & Saving")
        output_form = QFormLayout(output_group)
        self.output_directory_edit = QLineEdit()
        self.output_directory_browse = QPushButton("Browse…")
        self.output_directory_browse.clicked.connect(self._browse_output_directory)
        output_row = QWidget(); output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.addWidget(self.output_directory_edit, 1)
        output_row_layout.addWidget(self.output_directory_browse)
        self.run_name_edit = QLineEdit(); self.run_name_edit.setPlaceholderText("Defaults to project name + timestamp")
        self.save_jacobian_check = QCheckBox("Save Jacobian")
        output_form.addRow("Output directory", output_row)
        output_form.addRow("Run name", self.run_name_edit)
        output_form.addRow(self.save_jacobian_check)
        layout.addWidget(output_group)

        self.resume_current = QRadioButton("Start from current model")
        self.resume_previous = QRadioButton("Resume from previous LOCO state")
        self.resume_current.setChecked(True)
        self.resume_directory = QLineEdit()
        self.resume_browse = QPushButton("Browse…")
        self.resume_browse.clicked.connect(self._browse_resume_directory)
        resume_path_widget = QWidget(); resume_path_layout = QHBoxLayout(resume_path_widget)
        resume_path_layout.setContentsMargins(0, 0, 0, 0)
        resume_path_layout.addWidget(self.resume_directory, 1); resume_path_layout.addWidget(self.resume_browse)
        self.resume_ring_file = QLineEdit("ring_pyloco.mat")
        self.resume_fit_dict_file = QLineEdit("fit_dict.pkl")
        self.resume_fit_results_file = QLineEdit("fit_results.npy")
        self.resume_metadata = QLabel("No previous state selected.")
        self.resume_metadata.setWordWrap(True)
        resume_group = QGroupBox("Initialization / Resume")
        resume_form = QFormLayout(resume_group)
        resume_form.addRow(self.resume_current); resume_form.addRow(self.resume_previous)
        resume_form.addRow("Previous run or results directory", resume_path_widget)
        resume_form.addRow("Fitted lattice file", self.resume_ring_file)
        resume_form.addRow("Fit dictionary file", self.resume_fit_dict_file)
        resume_form.addRow("Fit history file", self.resume_fit_results_file)
        resume_form.addRow("State metadata", self.resume_metadata)
        layout.addWidget(resume_group)

        self.fixed_group = QGroupBox("RF and Momentum Compaction")
        fixed_form = QFormLayout(self.fixed_group)
        self.fixed_frequency = QLineEdit("499664399.4230182")
        self.fixed_harm_number = self._spin(1, 1000000, 3840)
        self.fixed_rfstep = self._double_spin(-1e9, 1e9, -3000.0, 9)
        self.mcf_source = QComboBox()
        self.mcf_source.addItem("Automatic — from reference lattice (AT/pyAT)", "automatic")
        self.mcf_source.addItem("Manual — user-defined value", "user")
        self.mcf_user_value = QLineEdit()
        self.mcf_user_value.setToolTip(
            "Dimensionless momentum compaction factor αc. The existing backend accepts any finite value."
        )
        self.mcf_automatic_summary = QLabel()
        self.mcf_automatic_summary.setWordWrap(True)
        self.mcf_automatic_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for label, widget in (("RF frequency [Hz]", self.fixed_frequency), ("Harmonic number", self.fixed_harm_number), ("RF frequency attribute", self.rm_rf_attr), ("rfStep", self.fixed_rfstep), ("Momentum compaction source", self.mcf_source)):
            fixed_form.addRow(label, widget)
        self.mcf_automatic_label = QLabel("Automatic momentum compaction")
        fixed_form.addRow(self.mcf_automatic_label, self.mcf_automatic_summary)
        self.mcf_user_value_label = QLabel("Momentum compaction factor (αc)")
        fixed_form.addRow(self.mcf_user_value_label, self.mcf_user_value)
        layout.addWidget(self.fixed_group)

        self.summary_group = QGroupBox("Configuration summary")
        summary_layout = QVBoxLayout(self.summary_group)
        self.fit_current_stage_summary = QLabel()
        self.fit_current_stage_summary.setWordWrap(True)
        self.fit_current_stage_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.fit_all_stages_summary = QLabel()
        self.fit_all_stages_summary.setWordWrap(True)
        self.fit_all_stages_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.fit_scientific_summary = QLabel()
        self.fit_scientific_summary.setWordWrap(True)
        self.fit_scientific_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self.fit_current_stage_summary)
        summary_layout.addWidget(self.fit_all_stages_summary)
        summary_layout.addWidget(self.fit_scientific_summary)
        self.fit_backend_group = QGroupBox("Backend options (technical)")
        self.fit_backend_group.setCheckable(True)
        self.fit_backend_group.setChecked(False)
        backend_layout = QVBoxLayout(self.fit_backend_group)
        backend_help = QLabel(
            "Exact options passed to the backend. These names are retained for expert diagnostics and reproducibility."
        )
        backend_help.setWordWrap(True)
        self.fit_backend_options = QPlainTextEdit()
        self.fit_backend_options.setReadOnly(True)
        self.fit_backend_options.setMinimumHeight(180)
        # Compatibility alias for tests and downstream GUI helpers that read
        # the former combined summary widget.
        self.fit_summary = self.fit_backend_options
        copy_backend = QPushButton("Copy backend options")
        copy_backend.clicked.connect(
            lambda: QApplication.clipboard().setText(self.fit_backend_options.toPlainText())
        )
        self.fit_backend_contents = QWidget()
        backend_contents_layout = QVBoxLayout(self.fit_backend_contents)
        backend_contents_layout.setContentsMargins(0, 0, 0, 0)
        backend_contents_layout.addWidget(backend_help)
        backend_contents_layout.addWidget(self.fit_backend_options)
        backend_contents_layout.addWidget(copy_backend, 0, Qt.AlignLeft)
        backend_layout.addWidget(self.fit_backend_contents)
        self.fit_backend_contents.setVisible(False)
        self.fit_backend_group.toggled.connect(self.fit_backend_contents.setVisible)
        summary_layout.addWidget(self.fit_backend_group)
        layout.addWidget(self.summary_group)
        layout.addStretch(1)
        scroll.setWidget(container)
        page.layout().addWidget(scroll, 1)
        self._load_config_to_widgets()
        self._connect_fit_controls()
        self._add_fit_stage(name="Stage 1", start_from="original_model", mark_modified=False)
        return page

    def _stage_configuration_snapshot(self) -> dict:
        cfg = self._collect_loco_configuration()
        return {"gui_config": json_safe(__import__("dataclasses").asdict(cfg)),
                "backend_mapping": json_safe(cfg.to_backend_mapping())}

    def _store_active_fit_stage(self) -> None:
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages) and not self._loading_config:
            self.fit_recipe.stages[self._active_fit_stage].configuration = self._stage_configuration_snapshot()

    def _validate_active_stage_widget_consistency(self) -> None:
        """Fail closed if visible fitted groups differ from the active recipe stage."""
        if not (0 <= self._active_fit_stage < len(self.fit_recipe.stages)):
            return
        visible = self._collect_loco_configuration().parameters.fit_list()
        stage = self.fit_recipe.stages[self._active_fit_stage]
        mapping = stage.configuration.get("backend_mapping", stage.configuration)
        stored = list(mapping.get("FitInitConfig", {}).get("fit_list", []))
        if visible != stored:
            raise ValueError(
                "The visible Parameter Selection and active FIT stage disagree. "
                f"Visible groups: {visible or ['none']}; stored stage groups: {stored or ['none']}. "
                "Review the stage selection and edit a fitted-parameter checkbox before running again."
            )

    def _configuration_for_fit_stage(self, stage: FitStage) -> LocoConfiguration:
        """Return an editable GUI configuration for new and legacy recipes.

        Early FIT recipes stored only the backend constructor mapping.  Those
        recipes have always been executable, but the editor previously kept
        showing the last GUI configuration when their rows were selected.
        Merge their complete backend mapping onto the current project defaults
        so every stage is inspectable and can be edited without changing its
        continuation semantics.
        """
        stored = stage.configuration
        gui_data = stored.get("gui_config")
        if gui_data:
            config = LocoConfiguration.from_dict(gui_data)
            return self._apply_machine_override_to_config(config)

        mapping = stored.get("backend_mapping", stored)
        data = json_safe(__import__("dataclasses").asdict(self.project.loco_config))

        def update_known(section: str, values: dict) -> None:
            target = data[section]
            for key, value in values.items():
                if key in target:
                    target[key] = str(value) if isinstance(target[key], str) and value is not None else value

        options = dict(mapping.get("LOCOOptions", {}))
        for section in ("solver", "svd", "rejection"):
            update_known(section, options)

        fit_init = dict(mapping.get("FitInitConfig", {}))
        selected = set(fit_init.pop("fit_list", options.get("fit_list", [])) or [])
        parameter_names = tuple(self.parameter_checks) if hasattr(self, "parameter_checks") else ()
        for name in parameter_names:
            data["parameters"][name] = name in selected
        update_known("parameters", fit_init)
        if "individuals" in fit_init:
            data["parameters"]["individuals"] = bool(fit_init["individuals"])

        for section, backend_key in (
            ("machine_elements", "MachineElements"),
            ("response_matrix", "RMConfig"),
            ("constraints", "ConstraintConfig"),
            ("fixed_parameters", "FixedParameters"),
        ):
            update_known(section, dict(mapping.get(backend_key, {})))

        momentum = dict(mapping.get("MomentumCompaction", {}))
        if momentum:
            data["mcf_source"] = momentum.get("source", data["mcf_source"])
            value = momentum.get("value")
            data["mcf_user_value"] = "" if value is None else str(value)
        output = dict(mapping.get("Output", {}))
        data["output_directory"] = output.get("directory", data["output_directory"])
        data["run_name"] = output.get("run_name", data["run_name"])
        if "BadBPMPositions" in mapping:
            data["bad_bpm_positions"] = list(mapping["BadBPMPositions"])
        excluded = dict(mapping.get("ExcludedCorrectorPositions", {}))
        if excluded:
            data["excluded_horizontal_corrector_positions"] = list(excluded.get("horizontal", []))
            data["excluded_vertical_corrector_positions"] = list(excluded.get("vertical", []))
        if "Resume" in mapping:
            update_known("resume", dict(mapping["Resume"]))
        return self._apply_machine_override_to_config(LocoConfiguration.from_dict(data))

    @staticmethod
    def _machine_override_from_config(config: LocoConfiguration) -> dict:
        return {
            "machine_elements": __import__("dataclasses").asdict(config.machine_elements),
            "element_selection_state": deepcopy(config.element_selection_state),
            "bad_bpm_positions": list(config.bad_bpm_positions),
            "excluded_horizontal_corrector_positions": list(config.excluded_horizontal_corrector_positions),
            "excluded_vertical_corrector_positions": list(config.excluded_vertical_corrector_positions),
        }

    def _apply_machine_override_to_config(self, config: LocoConfiguration) -> LocoConfiguration:
        if self.project.recipe_machine_policy != "current" or not self.project.machine_configuration_override:
            return config
        override = self.project.machine_configuration_override
        config.machine_elements = MachineElementsConfig(**deepcopy(override["machine_elements"]))
        config.element_selection_state = deepcopy(override.get("element_selection_state", {}))
        config.bad_bpm_positions = list(override.get("bad_bpm_positions", []))
        config.excluded_horizontal_corrector_positions = list(override.get("excluded_horizontal_corrector_positions", []))
        config.excluded_vertical_corrector_positions = list(override.get("excluded_vertical_corrector_positions", []))
        config._sync_response_matrix_elements()
        return config

    def _apply_machine_override_to_recipe(self, recipe: FitRecipe, override: dict) -> None:
        elements = deepcopy(override["machine_elements"])
        excluded = {
            "horizontal": list(override.get("excluded_horizontal_corrector_positions", [])),
            "vertical": list(override.get("excluded_vertical_corrector_positions", [])),
        }
        for stage in recipe.stages:
            stored = stage.configuration
            gui_data = stored.get("gui_config")
            if gui_data is not None:
                config = LocoConfiguration.from_dict(gui_data)
                config.machine_elements = MachineElementsConfig(**deepcopy(elements))
                config.element_selection_state = deepcopy(override.get("element_selection_state", {}))
                config.bad_bpm_positions = list(override.get("bad_bpm_positions", []))
                config.excluded_horizontal_corrector_positions = list(excluded["horizontal"])
                config.excluded_vertical_corrector_positions = list(excluded["vertical"])
                config._sync_response_matrix_elements()
                stored["gui_config"] = json_safe(__import__("dataclasses").asdict(config))
            mapping = stored.get("backend_mapping", stored)
            mapping["MachineElements"] = deepcopy(elements)
            mapping["BadBPMPositions"] = list(override.get("bad_bpm_positions", []))
            mapping["ExcludedCorrectorPositions"] = deepcopy(excluded)
            rm = mapping.setdefault("RMConfig", {})
            rm["bpm_ords"] = list(elements["bpm_ords"])
            rm["cm_ords"] = [
                list(elements["horizontal_corrector_ords"]),
                list(elements["vertical_corrector_ords"]),
            ]
            rm["cav_ords"] = list(elements["cavity_ords"])

    def _refresh_recipe_machine_override(self) -> None:
        if self.project.recipe_machine_policy != "current":
            return
        override = self._machine_override_from_config(self.project.loco_config)
        self.project.machine_configuration_override = deepcopy(override)
        self._apply_machine_override_to_recipe(self.fit_recipe, override)

    def _refresh_fit_workflow_table(self, select: int | None = None) -> None:
        table = self.fit_workflow_table
        table.blockSignals(True); table.setRowCount(len(self.fit_recipe.stages))
        completed = int((self.project.fit_run_session or {}).get("completed_stages", 0))
        for row, stage in enumerate(self.fit_recipe.stages):
            stage_config = self._configuration_for_fit_stage(stage)
            detail_lines = stage_config.summary_lines()
            detail_lines.append(
                "Dispersion weights: "
                f"horizontal={stage_config.rejection.hor_dispersion_weight:g}, "
                f"vertical={stage_config.rejection.ver_dispersion_weight:g}"
            )
            detail_lines.append(
                "Family groups: "
                f"normal={len(stage_config.machine_elements.normal_quadrupole_groups)}, "
                f"skew={len(stage_config.machine_elements.skew_quadrupole_groups)}, "
                f"tilt={len(stage_config.machine_elements.quadrupole_tilt_groups)}"
            )
            detail_lines.append(
                "Fitting exclusions: "
                f"{len(stage_config.bad_bpm_positions)} BPM, "
                f"{len(stage_config.excluded_horizontal_corrector_positions)} H corrector, "
                f"{len(stage_config.excluded_vertical_corrector_positions)} V corrector"
            )
            stage_details = "\n".join(detail_lines)
            mapping = stage.configuration.get("backend_mapping", stage.configuration)
            options = mapping.get("LOCOOptions", {})
            fit_list = mapping.get("FitInitConfig", {}).get("fit_list") or options.get("fit_list") or []
            inheritance = {
                "original_model": "Start from reference lattice",
                "previous_stage": "Continue from previous stage",
                "saved_result": "Continue from saved fitted result",
            }.get(stage.start_from, stage.start_from)
            values = (f"{'✓' if row < completed else '○'} {row + 1}", stage.name,
                      ", ".join(fit_list) if fit_list else "None",
                      options.get("nIter", 1), inheritance)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in (1, 3):
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    edit_hint = (
                        "Double-click to edit the stage name."
                        if column == 1 else "Double-click to enter a positive outer-iteration count."
                    )
                    item.setToolTip(f"{edit_hint}\n\n{stage_details}")
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setToolTip(stage_details)
                table.setItem(row, column, item)
            starting_model = QComboBox(table)
            starting_model.addItem("Start from reference lattice", "original_model")
            if row > 0:
                starting_model.addItem("Continue from previous stage", "previous_stage")
            if stage.start_from == "saved_result":
                starting_model.addItem("Continue from saved fitted result", "saved_result")
            index = starting_model.findData(stage.start_from)
            starting_model.setCurrentIndex(max(0, index))
            starting_model.setEnabled(row > 0 or stage.start_from == "saved_result")
            starting_model.setMinimumWidth(245)
            starting_model.setMaximumWidth(310)
            starting_model.setMinimumHeight(0)
            starting_model.setMaximumHeight(28)
            starting_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            starting_model.setStyleSheet("QComboBox { min-height: 0px; max-height: 28px; padding: 1px 8px; }")
            starting_model.setObjectName("fitStageStartingModelEditor")
            starting_model.setToolTip(
                "Stage 1 must start from the reference lattice. Later stages may start from the reference "
                "lattice or continue from the fitted state of the previous stage.\n\n" + stage_details
            )
            starting_model.currentIndexChanged.connect(
                lambda _index, stage_row=row, editor=starting_model:
                self._on_fit_stage_starting_model_changed(stage_row, str(editor.currentData()))
            )
            editor_container = QWidget(table)
            editor_container.setObjectName("fitStageStartingModelCell")
            editor_container.setFixedHeight(36)
            editor_layout = QHBoxLayout(editor_container)
            editor_layout.setContentsMargins(4, 3, 4, 3)
            editor_layout.setSpacing(0)
            editor_layout.addWidget(starting_model, 0, Qt.AlignVCenter)
            table.setCellWidget(row, 4, editor_container)
            table.setRowHeight(row, table.verticalHeader().defaultSectionSize())
        table.blockSignals(False)
        if select is not None and 0 <= select < table.rowCount(): table.selectRow(select)
        self._refresh_fit_stage_editor_label()
        self._update_fit_stage_empty_state()

    def _update_fit_stage_empty_state(self) -> None:
        """Keep empty-workflow guidance and editing controls synchronized."""
        if not hasattr(self, "fit_empty_state_label"):
            return
        has_stages = bool(self.fit_recipe.stages)
        self.fit_empty_state_label.setVisible(not has_stages)
        self.fit_stage_buttons["+ Add stage"].setText(
            "+ Add stage" if has_stages else "+ Add first FIT stage"
        )
        for label in ("Duplicate stage", "Remove", "Review fitting sequence"):
            self.fit_stage_buttons[label].setEnabled(has_stages)
        for label in ("Move up", "Move down"):
            self.fit_stage_buttons[label].setEnabled(len(self.fit_recipe.stages) > 1)
        self.clear_fit_configuration_button.setEnabled(has_stages)
        if hasattr(self, "save_fit_recipe_button"):
            self.save_fit_recipe_button.setEnabled(has_stages)
        if hasattr(self, "save_fit_run_session_button"):
            self.save_fit_run_session_button.setEnabled(has_stages)
        if hasattr(self, "_fit_configuration_widgets"):
            for widget in self._fit_configuration_widgets:
                widget.setEnabled(has_stages)

    def _refresh_fit_stage_editor_label(self) -> None:
        if not hasattr(self, "fit_stage_editor_label"):
            return
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages):
            stage = self.fit_recipe.stages[self._active_fit_stage]
            self.fit_stage_editor_label.setText(
                f"Editing stage {self._active_fit_stage + 1}: {stage.name} — "
                "the FIT controls below belong to this stage"
            )
        else:
            self.fit_stage_editor_label.setText(
                "No fitting stages configured — the FIT controls below are inactive"
            )

    def _record_fit_stage_edit(self, reason: str) -> None:
        """Persist the edited stage and invalidate stale execution bindings."""
        self.project.fit_recipe = self.fit_recipe.to_dict()
        self.project.completed_run = CompletedRunReference()
        self.project.fit_run_session = {}
        self._resume_fit_session = None
        self.project.compatibility_validation = {
            "required": True,
            "status": "validation_required",
            "issues": [reason],
        }
        self.project.modified = True

    @Slot(QTableWidgetItem)
    def _on_fit_workflow_item_changed(self, item: QTableWidgetItem) -> None:
        row, column = item.row(), item.column()
        if not (0 <= row < len(self.fit_recipe.stages)) or column not in (1, 3):
            return
        if row != self._active_fit_stage:
            self.fit_workflow_table.selectRow(row)
        stage = self.fit_recipe.stages[row]
        if column == 1:
            value = item.text().strip()
            if not value:
                QMessageBox.warning(self, "Invalid stage name", "A FIT stage name cannot be empty.")
                self._refresh_fit_workflow_table(row)
                return
            stage.name = value
            reason = f"Stage {row + 1} name changed; validate before running."
        else:
            try:
                value = int(item.text().strip())
                if value < 1:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid iteration count",
                    "Iterations must be a positive whole number (1 or greater).",
                )
                self._refresh_fit_workflow_table(row)
                return
            stored = stage.configuration
            gui_data = stored.get("gui_config")
            if gui_data is not None:
                gui_data.setdefault("solver", {})["nIter"] = value
            mapping = stored.get("backend_mapping", stored)
            mapping.setdefault("LOCOOptions", {})["nIter"] = value
            if row == self._active_fit_stage:
                self._loading_config = True
                try:
                    self.solver_n_iter.setValue(value)
                    self.project.loco_config.solver.nIter = value
                finally:
                    self._loading_config = False
            reason = f"Stage {row + 1} iteration count changed; validate before running."
        self._record_fit_stage_edit(reason)
        self._refresh_fit_workflow_table(row)
        self._refresh_ui(reason)

    def _on_fit_stage_starting_model_changed(self, row: int, start_from: str) -> None:
        if not (0 <= row < len(self.fit_recipe.stages)):
            return
        if row == 0 and start_from != "original_model":
            QMessageBox.warning(
                self, "Invalid starting model",
                "Stage 1 must start from the reference lattice because no previous fitted stage exists.",
            )
            self._refresh_fit_workflow_table(0)
            return
        if start_from not in {"original_model", "previous_stage", "saved_result"}:
            return
        if row != self._active_fit_stage:
            self.fit_workflow_table.selectRow(row)
        stage = self.fit_recipe.stages[row]
        if stage.start_from == start_from:
            return
        stage.start_from = start_from
        if start_from != "saved_result":
            stage.saved_result = ""
        reason = f"Stage {row + 1} starting model changed; validate before running."
        self._record_fit_stage_edit(reason)
        self._refresh_fit_workflow_table(row)
        self._refresh_ui(reason)

    def _on_fit_workflow_item_double_clicked(self, item: QTableWidgetItem) -> None:
        # Editable columns are handled by the table editor. Retain the existing
        # result-opening shortcut only on the stage-number/status column.
        if item.column() == 0:
            self._inspect_fit_stage_result(item)

    def _add_fit_stage(self, checked=False, *, name: str | None = None,
                       start_from: str | None = None, mark_modified: bool = True) -> None:
        if hasattr(self, "solver_n_iter"):
            configuration = self._stage_configuration_snapshot()
        else:
            configuration = {"backend_mapping": self.project.loco_config.to_backend_mapping()}
        index = len(self.fit_recipe.stages)
        self.fit_recipe.stages.append(FitStage(name or f"Stage {index + 1}", configuration,
                                               start_from or ("original_model" if index == 0 else "previous_stage")))
        self._active_fit_stage = index
        if mark_modified:
            self._record_fit_workflow_structure_change()
        else:
            self.project.fit_recipe = self.fit_recipe.to_dict()
        self._refresh_fit_workflow_table(index)
        if mark_modified:
            self._refresh_ui(f"Added FIT stage {index + 1}; validate before running")

    def _select_fit_stage(self) -> None:
        rows = self.fit_workflow_table.selectionModel().selectedRows() if self.fit_workflow_table.selectionModel() else []
        if not rows: return
        selected = rows[0].row()
        if selected == self._active_fit_stage:
            self._refresh_fit_stage_editor_label()
            return
        self._store_active_fit_stage(); self._active_fit_stage = selected
        self.project.loco_config = self._configuration_for_fit_stage(self.fit_recipe.stages[selected])
        self._load_config_to_widgets()
        self._refresh_fit_stage_editor_label()

    def _duplicate_fit_stage(self) -> None:
        self._store_active_fit_stage()
        if not self.fit_recipe.stages: return
        source = deepcopy(self.fit_recipe.stages[max(0, self._active_fit_stage)])
        source.name += " copy"; source.start_from = "previous_stage" if self.fit_recipe.stages else "original_model"
        self.fit_recipe.stages.insert(self._active_fit_stage + 1, source)
        self._active_fit_stage += 1
        self._record_fit_workflow_structure_change()
        self._refresh_fit_workflow_table(self._active_fit_stage)
        self._refresh_ui(f"Duplicated FIT stage {self._active_fit_stage + 1}; validate before running")

    def _record_fit_workflow_structure_change(self) -> None:
        """Invalidate active run state after adding/removing/reordering stages."""
        self.project.fit_recipe = self.fit_recipe.to_dict()
        self.project.completed_run = CompletedRunReference()
        self.project.fit_run_session = {}
        self._resume_fit_session = None
        self.project.compatibility_validation = {
            "required": True,
            "status": "validation_required",
            "issues": ["Fitting stages changed; run compatibility validation again."],
        }
        self.project.modified = True

    def _confirm_clear_fit_configuration(self, exclusions: dict[str, list[int]]) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Clear FIT configuration?")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText("Clear every fitting stage and return the FIT editor to its initial state?")
        exclusion_count = sum(len(values) for values in exclusions.values())
        dialog.setInformativeText(
            "Cleared:\n"
            "• all FIT stages and stage-specific fitted parameters\n"
            "• solver, SVD, weights, constraints, initialization and numerical-step settings\n"
            "• active FIT results/checkpoint bindings\n"
            f"• {exclusion_count} BPM/corrector fitting exclusion position(s)\n\n"
            "Preserved:\n"
            "• reference lattice and selected MAT variable\n"
            "• complete machine-element selections and selection provenance\n"
            "• validated measurement-device mapping evidence\n"
            "• measurement files, arrays, identities, checksums and session provenance\n\n"
            "The original project file is not changed unless you later save over it. "
            "Use Save Project As… to keep an adapted copy."
        )
        if exclusion_count:
            dialog.setDetailedText(
                "Fitting exclusions use positions in the selected measured-device lists. "
                "They will be cleared only after this confirmation and can be reviewed or re-entered later.\n\n"
                f"BPM positions: {exclusions['bpm']}\n"
                f"Horizontal-corrector positions: {exclusions['horizontal_corrector']}\n"
                f"Vertical-corrector positions: {exclusions['vertical_corrector']}"
            )
        clear_button = dialog.addButton("Clear FIT configuration", QMessageBox.DestructiveRole)
        cancel_button = dialog.addButton("Cancel — keep current configuration", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is clear_button

    def _initial_fit_configuration(self, current: LocoConfiguration) -> LocoConfiguration:
        """Return documented FIT defaults while preserving machine identity state."""
        initial = LocoConfiguration()
        initial.machine_elements = deepcopy(current.machine_elements)
        initial.element_selection_state = deepcopy(current.element_selection_state)
        initial._sync_response_matrix_elements()

        # Acquisition facts are not fitting choices. Restore them from the
        # imported file metadata when available instead of inheriting a stage.
        orm = (self.project.measurements.get("orm").options
               if self.project.measurements.get("orm") else {})
        for option, rm_field, cm_field in (
            ("dkick_h", "dkick_h", "horizontal"),
            ("dkick_v", "dkick_v", "vertical"),
        ):
            value = orm.get(option)
            if value is not None:
                numeric = float(value)
                setattr(initial.response_matrix, rm_field, numeric)
                setattr(initial.parameters.cmstep, cm_field, numeric)
        if "bidirectional" in orm:
            initial.response_matrix.bidirectional = bool(orm["bidirectional"])
        dispersion = (self.project.measurements.get("dispersion").options
                      if self.project.measurements.get("dispersion") else {})
        if dispersion.get("rf_step_hz") is not None:
            step = float(dispersion["rf_step_hz"])
            initial.response_matrix.rfStep = step
            initial.parameters.rfStep = step
            initial.fixed_parameters.rfstep = step
        return initial

    @Slot()
    def clear_fit_configuration(self) -> bool:
        current = self._collect_loco_configuration()
        exclusions = {
            "bpm": list(current.bad_bpm_positions),
            "horizontal_corrector": list(current.excluded_horizontal_corrector_positions),
            "vertical_corrector": list(current.excluded_vertical_corrector_positions),
        }
        if not self._confirm_clear_fit_configuration(exclusions):
            return False

        old_recipe = self.fit_recipe
        initial = self._initial_fit_configuration(current)
        self.fit_recipe = FitRecipe(
            name="FIT workflow",
            machine_identity=old_recipe.machine_identity,
            reference_model_checksum=old_recipe.reference_model_checksum,
            element_identities=deepcopy(old_recipe.element_identities),
            data_requirements=deepcopy(old_recipe.data_requirements),
            stages=[],
        )
        self.project.loco_config = initial
        self.project.fit_recipe = self.fit_recipe.to_dict()
        self.project.recipe_machine_policy = "current"
        self.project.machine_configuration_override = self._machine_override_from_config(initial)
        self.project.completed_run = CompletedRunReference()
        self.project.fit_run_session = {}
        self._resume_fit_session = None
        self._active_fit_stage = -1
        self.project.compatibility_validation = {
            "required": True,
            "status": "validation_required",
            "issues": ["No fitting stages are configured. Add and validate a FIT stage before running."],
        }
        self.project.modified = True
        self._load_config_to_widgets()
        self._refresh_fit_workflow_table()
        self._refresh_ui("FIT configuration cleared; lattice, measurements and machine setup preserved")
        return True

    def _remove_fit_stage(self) -> None:
        if not self.fit_recipe.stages:
            self.statusBar().showMessage("No FIT stage is configured"); return
        if len(self.fit_recipe.stages) == 1:
            self.clear_fit_configuration()
            return
        self.fit_recipe.stages.pop(self._active_fit_stage)
        self._active_fit_stage = min(self._active_fit_stage, len(self.fit_recipe.stages) - 1)
        self.fit_recipe.stages[0].start_from = "original_model"
        # Selecting the same numeric row after deletion does not emit a stage
        # change: the next stage has moved into that row.  Load that surviving
        # stage explicitly so the editor cannot retain the deleted stage's
        # controls.
        surviving_stage = self.fit_recipe.stages[self._active_fit_stage]
        self.project.loco_config = self._configuration_for_fit_stage(surviving_stage)
        self._load_config_to_widgets()
        self._record_fit_workflow_structure_change()
        self._refresh_fit_workflow_table(self._active_fit_stage)
        self._refresh_ui(
            f"Removed FIT stage; now editing stage {self._active_fit_stage + 1}: {surviving_stage.name}"
        )

    def _move_fit_stage(self, delta: int) -> None:
        self._store_active_fit_stage(); old = self._active_fit_stage; new = old + delta
        if old < 0 or new < 0 or new >= len(self.fit_recipe.stages): return
        self.fit_recipe.stages[old], self.fit_recipe.stages[new] = self.fit_recipe.stages[new], self.fit_recipe.stages[old]
        self.fit_recipe.stages[0].start_from = "original_model"; self._active_fit_stage = new
        self._record_fit_workflow_structure_change()
        self._refresh_fit_workflow_table(new)
        self._refresh_ui(f"Moved FIT stage to position {new + 1}; validate before running")

    def save_fit_recipe(self) -> None:
        self._store_active_fit_stage()
        filename = QFileDialog.getSaveFileName(self, "Save FIT recipe", "fit-recipe.json", "FIT recipe (*.json)")[0]
        if filename:
            try: self.fit_recipe.save(filename)
            except Exception as exc: QMessageBox.warning(self, "Cannot save FIT recipe", str(exc))

    def _choose_recipe_machine_policy(self) -> str | None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Machine elements for imported fitting stages")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText("Choose which machine-element selections the imported stages should use.")
        dialog.setInformativeText(
            "Use machine elements stored in the recipe restores each stage exactly as saved.\n\n"
            "Keep current machine elements and exclusions applies the current BPM, corrector, quadrupole, "
            "RF-cavity and exclusion selections to every imported stage. Stage-specific parameters, weights, "
            "optimizer settings, constraints and handoff remain unchanged."
        )
        use_recipe = dialog.addButton("Use machine elements stored in the recipe", QMessageBox.AcceptRole)
        keep_current = dialog.addButton("Keep current machine elements and exclusions", QMessageBox.ActionRole)
        cancel = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(keep_current)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is use_recipe:
            return "recipe"
        if clicked is keep_current:
            return "current"
        if clicked is cancel:
            return None
        return None

    def _preflight_imported_recipe(self, recipe: FitRecipe) -> list[str]:
        lattice_path = self.project.resolve_path(self.project.lattice.path)
        if not lattice_path.is_file():
            return ["Select an existing reference lattice before loading fitting stages."]
        lattice = self._load_current_lattice()
        if lattice is None:
            return ["The selected reference lattice could not be loaded."]
        names = [self._element_identity(element)[0] for element in lattice]
        measurements = {
            role: str(self.project.resolve_path(dataset.path))
            for role, dataset in self.project.measurements.items()
        }
        options = {role: deepcopy(dataset.options) for role, dataset in self.project.measurements.items()}
        report = preflight_recipe(
            recipe,
            lattice_path=str(lattice_path),
            measurement_identity=deepcopy(self.project.measurement_session),
            lattice_element_names=names,
            measurements=measurements,
            measurement_options=options,
            measurement_device_mapping=deepcopy(self.project.measurement_device_mapping),
            lattice_mat_variable=self.project.lattice.mat_variable or None,
        )
        return list(report["errors"])

    def load_fit_recipe(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load FIT recipe", "", "FIT recipe (*.json)")[0]
        if not filename: return
        try:
            candidate = FitRecipe.load(filename)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot load FIT recipe", str(exc)); return
        policy = self._choose_recipe_machine_policy()
        if policy is None:
            return
        current_config = self._collect_loco_configuration()
        override = self._machine_override_from_config(current_config) if policy == "current" else {}
        if policy == "current":
            self._apply_machine_override_to_recipe(candidate, override)
        errors = self._preflight_imported_recipe(candidate)
        if errors:
            QMessageBox.warning(
                self, "Fitting stages are incompatible",
                "The fitting stages were not loaded, and the existing project was left unchanged:\n\n"
                + "\n".join(f"• {error}" for error in errors),
            )
            return
        self.fit_recipe = candidate
        self.project.recipe_machine_policy = policy
        self.project.machine_configuration_override = deepcopy(override)
        self._resume_fit_session = None
        self._active_fit_stage = -1; self._refresh_fit_workflow_table(0); self._select_fit_stage()
        # A recipe changes FIT settings only.  Measurement bindings remain part
        # of the open project and are persisted again by the normal Save action.
        self.project.modified = True
        self._mark_compatibility_stale("Fitting stages changed; run compatibility validation again.")
        self._refresh_ui(f"Loaded fitting stages from {Path(filename).name}; measurement bindings preserved")

    def preview_fit_workflow(self) -> None:
        """Show a compact, read-only recipe summary before execution."""
        self._store_active_fit_stage()
        lines = [f"FIT workflow — {self.fit_recipe.name}", ""]
        for number, stage in enumerate(self.fit_recipe.stages, 1):
            mapping = stage.configuration.get("backend_mapping", stage.configuration)
            options = mapping.get("LOCOOptions", {})
            fit_list = list(mapping.get("FitInitConfig", {}).get("fit_list") or options.get("fit_list") or [])
            elements = mapping.get("MachineElements", {})
            mode = {
                "original_model": "starts from original reference model",
                "previous_stage": "inherits previous fitted state; uses this stage's configuration",
                "saved_result": f"starts from saved result {stage.saved_result}",
            }.get(stage.start_from, stage.start_from)
            lines.extend((f"{number}. {stage.name} — {options.get('nIter', 1)} iteration(s)",
                          f"   Start: {mode}",
                          f"   Fit: {', '.join(fit_list) if fit_list else 'no parameters'}"))
            for label, key, group_key, individual in (
                ("Normal quadrupoles", "normal_quadrupole_ords", "normal_quadrupole_groups",
                 bool(mapping.get("FitInitConfig", {}).get("individuals", True))),
                ("Skew quadrupoles", "skew_quadrupole_ords", "skew_quadrupole_groups",
                 bool(options.get("skew_individuals", True))),
                ("Quadrupole tilts", "quadrupole_tilt_ords", "quadrupole_tilt_groups",
                 bool(options.get("tilt_individuals", True))),
            ):
                physical = len(elements.get(key) or [])
                parameters = physical if individual else len(elements.get(group_key) or [])
                if physical:
                    lines.append(f"   {label}: {physical} physical → {parameters} {'individual' if individual else 'group'} parameter(s)")
            lines.append("")
        QMessageBox.information(self, "Preview full FIT workflow", "\n".join(lines).rstrip())

    def save_fit_run_session(self) -> None:
        source = Path(self.project.fit_run_session.get("path", "")) if self.project.fit_run_session else Path()
        if not source.is_file():
            self.statusBar().showMessage("Run a FIT stage before saving the session"); return
        filename = QFileDialog.getSaveFileName(self, "Save FIT run/session", "fit-run-session.json", "FIT run session (*.json)")[0]
        if filename:
            try: Path(filename).write_bytes(source.read_bytes())
            except OSError as exc: QMessageBox.warning(self, "Cannot save FIT run/session", str(exc))

    def resume_fit_run(self) -> None:
        from .fit_workflow import FitRunSession
        if not self._compatibility_is_validated():
            QMessageBox.information(
                self, "Resume disabled",
                "Validate the current measurements and FIT configuration before resuming a previous run.",
            )
            return
        filename = QFileDialog.getOpenFileName(self, "Resume FIT run", "", "FIT run session (*.json)")[0]
        if not filename: return
        try:
            session = FitRunSession.load(filename)
            current_lattice = str(self.project.resolve_path(self.project.lattice.path)) if self.project.lattice.path else ""
            if current_lattice and session.lattice_checksum != file_sha256(current_lattice):
                raise ValueError("The current reference lattice does not match this FIT run/session.")
            current_variable = self.project.lattice.mat_variable or ""
            if session.lattice_mat_variable and session.lattice_mat_variable != current_variable:
                raise ValueError(
                    "The selected MAT lattice variable does not match this FIT run/session."
                )
            current_measurements = {key: str(self.project.resolve_path(value.path)) for key, value in self.project.measurements.items()}
            if current_measurements and current_measurements != session.measurements:
                raise ValueError("The current measurement binding does not match this FIT run/session.")
            missing = [item for checkpoint in session.checkpoints for item in checkpoint.validate_files()]
            if missing: raise ValueError("Incomplete continuation checkpoint: " + ", ".join(missing))
            recipe_data = deepcopy(session.recipe); recipe_data["stages"] = [FitStage(**item) for item in recipe_data.get("stages", [])]
            self.fit_recipe = FitRecipe(**recipe_data)
            if self.project.recipe_machine_policy == "current" and self.project.machine_configuration_override:
                self._apply_machine_override_to_recipe(self.fit_recipe, self.project.machine_configuration_override)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot resume FIT run", str(exc)); return
        self.project.fit_run_session = {"path": str(Path(filename).resolve()), "completed_stages": len(session.checkpoints)}
        self._resume_fit_session = session
        self._active_fit_stage = -1; self._refresh_fit_workflow_table(min(len(session.checkpoints), len(self.fit_recipe.stages)-1)); self._select_fit_stage()

    def _inspect_fit_stage_result(self, item) -> None:
        from .fit_workflow import FitRunSession
        path = self.project.fit_run_session.get("path", "") if self.project.fit_run_session else ""
        if not path or not Path(path).is_file(): return
        try:
            session = FitRunSession.load(path); checkpoint = session.checkpoints[item.row()]
            self.results_workspace.load_results(checkpoint.results_dir)
            self._workspace.setCurrentWidget(self.results_page)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open stage result", str(exc))

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = ScrollSafeSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spin(
        self, minimum: float, maximum: float, value: float, decimals: int, suffix: str = ""
    ) -> QDoubleSpinBox:
        spin = ScientificDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setValue(value)
        spin.setSingleStep(abs(value) or 1.0)
        return spin

    def _page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 28)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        return page

    def _configure_responsive_layouts(self) -> None:
        """Apply consistent reflow rules without changing scientific controls."""
        for form in self.findChildren(QFormLayout):
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)

    def _create_actions(self) -> None:
        self.new_project_action = QAction("New", self)
        self.new_project_action.setShortcut(QKeySequence.New)
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action = QAction("Open…", self)
        self.open_project_action.setShortcut(QKeySequence.Open)
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action = QAction("Save", self)
        self.save_project_action.setShortcut(QKeySequence.Save)
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action = QAction("Save As…", self)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.validate_project_action = QAction("Validate", self)
        self.validate_project_action.triggered.connect(self.validate_project)
        self.run_loco_action = QAction("▶ Run LOCO", self)
        self.run_loco_action.triggered.connect(self.run_loco)
        self.compare_reference_action = QAction("Compare imported measurements with initial model…", self)
        self.compare_reference_action.setToolTip(
            "Calculate a read-only reference ORM and RF response before fitting; no optimizer or machine writes."
        )
        self.compare_reference_action.triggered.connect(self.compare_measurements_with_reference)
        self.compare_orms_action = QAction("Compare measurements with fitted model…", self)
        self.compare_orms_action.setToolTip(
            "Compare the measurements recorded by the selected FIT stage and iteration with that fitted model. "
            "RF response is handled separately from corrector ORM columns."
        )
        self.compare_orms_action.triggered.connect(self.compare_orms)
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        self.basic_mode_action = QAction("Basic", self, checkable=True, checked=self.project.mode == "Basic")
        self.advanced_mode_action = QAction("Advanced", self, checkable=True, checked=self.project.mode == "Advanced")
        self.mode_action_group = QActionGroup(self, exclusive=True)
        self.mode_action_group.addAction(self.basic_mode_action)
        self.mode_action_group.addAction(self.advanced_mode_action)
        self.mode_action_group.triggered.connect(self._on_mode_changed)
        self.theme_action_group = QActionGroup(self, exclusive=True)
        self.theme_actions = {}
        for key, theme in THEMES.items():
            action = QAction(theme.display_name, self, checkable=True)
            action.setData(key)
            action.setChecked(key == self.current_theme.key)
            self.theme_action_group.addAction(action)
            self.theme_actions[key] = action
        self.theme_action_group.triggered.connect(self._on_theme_changed)
        self.toggle_theme_action = QAction(self)
        self.toggle_theme_action.triggered.connect(self._toggle_theme)
        self._update_toggle_theme_action()
        self.float_explorer_action = QAction(
            "Move Project Explorer to Separate Window", self, checkable=True
        )
        self.float_explorer_action.triggered.connect(
            self._set_project_explorer_floating
        )
        self._project_explorer.topLevelChanged.connect(
            self.float_explorer_action.setChecked
        )
        self.about_action = QAction("About pyLOCO GUI", self)
        self.about_action.triggered.connect(self._show_about_dialog)
        self.open_measure_action=QAction("Open pyLOCO Measure",self); self.open_measure_action.triggered.connect(self.open_measure_app)
        self.open_correct_action=QAction("Open pyLOCO Correct",self); self.open_correct_action.triggered.connect(self.open_correct_app)
        self.open_session_action=QAction("Open Measurement Session…",self); self.open_session_action.triggered.connect(self.open_measurement_session)

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.save_project_as_action,
        ):
            file_menu.addAction(action)
        self.recent_menu = file_menu.addMenu("Recent Projects")
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        project_menu = self.menuBar().addMenu("&Project")
        project_menu.addAction(self.validate_project_action)
        project_menu.addAction(self.run_loco_action)
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.compare_orms_action)
        suite_menu=self.menuBar().addMenu("pyLOCO &Suite"); suite_menu.addAction(self.open_session_action); suite_menu.addSeparator(); suite_menu.addAction(self.open_measure_action); suite_menu.addAction(self.open_correct_action)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._project_explorer.toggleViewAction())
        view_menu.addAction(self.float_explorer_action)
        mode_menu = view_menu.addMenu("Workflow Mode")
        mode_menu.addAction(self.basic_mode_action)
        mode_menu.addAction(self.advanced_mode_action)
        theme_menu = view_menu.addMenu("Theme")
        for action in self.theme_actions.values():
            theme_menu.addAction(action)
        settings_menu = self.menuBar().addMenu("&Settings")
        appearance_menu = settings_menu.addMenu("Appearance")
        for action in self.theme_actions.values():
            appearance_menu.addAction(action)
        self.menuBar().addMenu("&Help").addAction(self.about_action)

    def _set_project_explorer_floating(self, floating: bool) -> None:
        """Move the explorer between a free window and the left dock area."""
        self._project_explorer.setFloating(floating)
        if not floating:
            self.addDockWidget(Qt.LeftDockWidgetArea, self._project_explorer)
        self._project_explorer.show()

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for action in (self.new_project_action, self.open_project_action, self.save_project_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.validate_project_action)
        toolbar.addAction(self.run_loco_action)
        toolbar.addSeparator()
        toolbar.addAction(self.basic_mode_action)
        toolbar.addAction(self.advanced_mode_action)
        toolbar.addSeparator()
        toolbar.addAction(self.toggle_theme_action)
        self.accent_combo=QComboBox(); self.accent_combo.setToolTip("Shared suite accent color")
        for key,values in ACCENTS.items():self.accent_combo.addItem(values[0],key)
        self.accent_combo.setCurrentIndex(max(0,self.accent_combo.findData(self.current_accent)))
        self.accent_combo.currentIndexChanged.connect(self._on_accent_changed); toolbar.addWidget(self.accent_combo)
        toolbar_spacer = QWidget(); toolbar_spacer.setObjectName("toolbarSpacer")
        toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(toolbar_spacer)
        self.header_brand = BrandToolButton(toolbar); self.header_brand.setObjectName("headerBrandButton")
        self.header_brand.setMinimumWidth(180)
        brand_layout = QHBoxLayout(self.header_brand); brand_layout.setContentsMargins(12, 4, 12, 4)
        self.header_brand_label = QLabel(wordmark_html(self.current_theme.key)); self.header_brand_label.setTextFormat(Qt.RichText)
        self.header_brand_label.setMinimumWidth(180)
        self.header_brand_label.setAttribute(Qt.WA_TransparentForMouseEvents); brand_layout.addWidget(self.header_brand_label)
        self.header_brand.setCursor(Qt.PointingHandCursor); self.header_brand.setToolTip("About pyLOCO"); self.header_brand.clicked.connect(self._show_about_dialog)
        self.header_brand_action = toolbar.addWidget(self.header_brand)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        run_button = toolbar.widgetForAction(self.run_loco_action)
        if run_button is not None:
            run_button.setObjectName("primaryToolbarAction")

    def _create_status_bar(self) -> None:
        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(True)
        status_bar.addWidget(self._project_label, 1)
        status_bar.addWidget(self._workflow_label, 1)
        status_bar.addWidget(self._validation_label, 1)
        status_bar.addPermanentWidget(self._backend_label)
        status_bar.addPermanentWidget(self._mode_label)
        self.setStatusBar(status_bar)


    def _connect_fit_controls(self) -> None:
        widgets = [
            self.rm_calculator, self.rm_dispersion, self.rm_coupling, self.rm_bidirectional,
            self.quad_jacobian_calculator, self.skew_jacobian_calculator,
            self.analytical_thick_quadrupole, self.analytical_thick_steerers,
            self.analytical_verbose, self.analytical_use_mp,
            self.analytical_implementation,
            self.analytical_dispersion_calculator,
            self.analytical_thick_skew, self.analytical_skew_thick_steerers,
            self.analytical_skew_verbose, self.analytical_skew_use_mp,
            self.skew_analytical_implementation,
            self.skew_analytical_dispersion_calculator,
            self.skew_analytical_dispersion_worker,
            self.rm_vectorized, self.rm_dkick_h, self.rm_dkick_v, self.rm_rf_step,
            self.rm_delta_coupling, self.rm_fixedpath, self.rm_log_info, self.solver_algorithm, self.solver_n_iter,
            self.solver_lm_iter, self.solver_lambda, self.solver_max_lambda,
            self.solver_scaled, self.svd_method, self.svd_threshold, self.svd_rank,
            self.svd_plot, self.outlier_enabled, self.outlier_sigma, self.norm_enabled,
            self.norm_mode, self.loco_hor_dispersion_weight,
            self.loco_ver_dispersion_weight, self.auto_delta, self.loco_fixedpath, self.loco_individuals,
            self.loco_remove_coupling, self.loco_plot_fit_parameters, self.constraint_enabled,
            self.constraint_quad_sigma, self.constraint_skew_sigma,
            self.constraint_quad_weights, self.constraint_skew_weights,
            self.constraint_quad_sigma_mode, self.constraint_quad_relative_sigma,
            self.constraint_quad_minimum_sigma, self.constraint_quad_default_weight,
            self.constraint_quad_selected_families, self.constraint_quad_selected_weight,
            self.constraint_skew_default_weight, self.constraint_skew_selected_families,
            self.constraint_skew_selected_weight,
            self.cmstep_mode, self.params_init_policy, self.params_cmstep_h, self.params_cmstep_v,
            self.params_cmstep_file, self.params_cmstep_browse, self.params_rfstep, self.params_init, self.params_quads_attr, self.params_quads_attr_index,
            self.params_skew_attr, self.params_skew_attr_index, self.params_tilt_attr_r1,
            self.params_tilt_attr_r2, self.params_tilt_method, self.fixed_frequency, self.fixed_harm_number,
            self.fixed_rfstep, self.fixed_dk, self.fixed_delta_skew, self.fixed_delta_q_tilt, self.mcf_source, self.mcf_user_value,
            self.output_directory_edit, self.run_name_edit, self.save_jacobian_check,
        ] + list(self.parameter_checks.values()) + [radio for pair in self.parameterization_radios.values() for radio in pair]
        self._fit_configuration_widgets = list(dict.fromkeys(widgets))
        for widget in widgets:
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_fit_config_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_fit_config_changed)
            elif isinstance(widget, QRadioButton):
                widget.toggled.connect(self._on_fit_config_changed)
        self.solver_algorithm.currentIndexChanged.connect(self._update_solver_scaled_availability)
        self.svd_method.currentIndexChanged.connect(self._update_svd_input_availability)
        self.cmstep_mode.currentIndexChanged.connect(self._update_cmstep_input_availability)
        self.params_cmstep_file.textChanged.connect(self._refresh_cmstep_resolved_values)
        self.params_init_policy.textChanged.connect(self._refresh_initialization_policy_summary)
        self.params_init.textChanged.connect(self._refresh_explicit_initial_values_summary)
        self.fixed_dk.textChanged.connect(self._refresh_normal_step_resolution)
        self.auto_delta.toggled.connect(self._refresh_normal_step_resolution)
        self.parameter_checks["quads"].toggled.connect(self._refresh_normal_step_resolution)
        self.parameter_checks["quads"].toggled.connect(self._update_jacobian_option_availability)
        self.quad_jacobian_calculator.currentTextChanged.connect(self._refresh_normal_step_resolution)
        self.parameter_checks["skew_quads"].toggled.connect(self._refresh_numerical_step_status)
        self.parameter_checks["skew_quads"].toggled.connect(self._update_jacobian_option_availability)
        self.parameter_checks["quads_tilt"].toggled.connect(self._refresh_numerical_step_status)
        self.skew_jacobian_calculator.currentTextChanged.connect(self._refresh_numerical_step_status)
        self.quad_jacobian_calculator.currentTextChanged.connect(
            self._update_jacobian_option_availability
        )
        self.skew_jacobian_calculator.currentTextChanged.connect(
            self._update_jacobian_option_availability
        )
        self.mcf_source.currentIndexChanged.connect(self._update_mcf_source_visibility)
        self.constraint_quad_exceptions.changed.connect(self._on_fit_config_changed)
        self.constraint_skew_exceptions.changed.connect(self._on_fit_config_changed)
        self.rm_dispersion.toggled.connect(self._update_context_controls)
        self.constraint_enabled.toggled.connect(self._update_context_controls)
        self.rm_fixedpath.toggled.connect(self.loco_fixedpath.setChecked)
        self.resume_current.toggled.connect(self._on_fit_config_changed)
        self.resume_previous.toggled.connect(self._on_fit_config_changed)
        self.resume_previous.toggled.connect(self._update_resume_availability)
        self.resume_directory.textChanged.connect(self._on_fit_config_changed)
        self.resume_directory.textChanged.connect(self._update_resume_availability)
        for widget in (self.resume_ring_file, self.resume_fit_dict_file, self.resume_fit_results_file):
            widget.textChanged.connect(self._on_fit_config_changed)
            widget.textChanged.connect(self._update_resume_availability)
        self._update_context_controls()

    def _update_context_controls(self) -> None:
        dispersion = self.rm_dispersion.isChecked()
        self.dispersion_weight_controls.setVisible(dispersion)
        self.loco_hor_dispersion_weight.setEnabled(dispersion)
        self.loco_ver_dispersion_weight.setEnabled(dispersion)
        enabled = self.constraint_enabled.isChecked()
        self.constraint_details.setVisible(enabled)
        for widget in (
            self.constraint_quad_sigma_mode, self.constraint_quad_sigma,
            self.constraint_quad_relative_sigma, self.constraint_quad_minimum_sigma,
            self.constraint_quad_default_weight, self.constraint_quad_selected_families,
            self.constraint_quad_selected_weight, self.constraint_quad_exceptions,
            self.constraint_skew_sigma, self.constraint_skew_default_weight,
            self.constraint_skew_selected_families, self.constraint_skew_selected_weight,
            self.constraint_skew_exceptions,
            self.constraint_quad_weights, self.constraint_skew_weights,
            self.constraint_quad_mask, self.constraint_skew_mask,
        ):
            widget.setEnabled(enabled)

    def _refresh_initialization_policy_summary(self) -> None:
        import ast
        raw = self.params_init_policy.text().strip()
        if not raw:
            self.params_init_policy_summary.setText("Built-in defaults for all parameter groups")
            return
        try:
            value = ast.literal_eval(raw)
            if not isinstance(value, dict):
                raise ValueError
            self.params_init_policy_summary.setText(f"{len(value)} policy entries · click Edit policies to inspect")
        except (SyntaxError, ValueError):
            self.params_init_policy_summary.setText("Invalid policy dictionary — review required")

    def _refresh_normal_step_resolution(self) -> None:
        if not self.parameter_checks["quads"].isChecked():
            text = "Not used in this stage: normal quadrupoles are not fitted."
        elif self.quad_jacobian_calculator.currentText() == "Analytical":
            text = "Not used: the normal-quadrupole Jacobian is analytical."
        elif self.fixed_dk.text().strip():
            text = f"Fixed configured value: {self.fixed_dk.text().strip()} (lattice quadrupole-strength units)."
        else:
            adaptive = "adaptive refinement enabled" if self.auto_delta.isChecked() else "adaptive refinement disabled"
            text = (
                "Per fitted parameter: initial ΔK = 0.001 × K; if K = 0, ΔK = 0.001 "
                f"(lattice quadrupole-strength units); {adaptive}."
            )
        self.normal_step_resolution.setText(text)

    def _refresh_numerical_step_status(self) -> None:
        skew_used = self.parameter_checks["skew_quads"].isChecked() and self.skew_jacobian_calculator.currentText() == "Numerical"
        tilt_used = self.parameter_checks["quads_tilt"].isChecked()
        self.fixed_delta_skew.setEnabled(skew_used)
        self.fixed_delta_q_tilt.setEnabled(tilt_used)
        self.skew_step_status.setText("Used by this stage" if skew_used else "Not used in this stage")
        self.tilt_step_status.setText("Used by this stage" if tilt_used else "Not used in this stage")

    def _refresh_explicit_initial_values_summary(self) -> None:
        import ast
        raw = str(self.params_init.text() or "").strip()
        if not raw:
            self.params_init_summary.setText("Optional — no explicit overrides")
            return
        try:
            value = ast.literal_eval(raw)
            if not isinstance(value, dict): raise ValueError
            self.params_init_summary.setText(f"Overrides {len(value)} parameter group(s)")
        except (SyntaxError, ValueError):
            self.params_init_summary.setText("Invalid override dictionary — review required")

    def _edit_explicit_initial_values(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle("Explicit initial-value overrides — advanced")
        dialog.setObjectName("explicitInitialValuesDialog")
        dialog.setModal(True)
        dialog.resize(720, 360)
        dialog.setMinimumSize(560, 300)
        layout = QVBoxLayout(dialog)
        help_label = QLabel(
            "Optional. Values entered here override the initialization policy for the named parameter group. "
            "Use a Python-style dictionary; a scalar is broadcast and a list must match the group size. "
            "Example: {'hbpm_gain': 1.0, 'hcor_cal': [0.0001, 0.0001]}."
        )
        help_label.setObjectName("explicitInitialValuesHelp")
        help_label.setWordWrap(True); layout.addWidget(help_label)
        editor = QPlainTextEdit(); editor.setPlainText(self.params_init.text()); editor.setPlaceholderText("{'hbpm_gain': 1.0}")
        editor.setObjectName("explicitInitialValuesEditor")
        editor.setMinimumHeight(150); layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons); buttons.rejected.connect(dialog.reject)

        def accept() -> None:
            from pyLOCO.gui.models.project import ParameterSelectionConfig
            candidate = editor.toPlainText().strip()
            try: ParameterSelectionConfig(init=candidate).init_value()
            except ValueError as exc:
                QMessageBox.warning(dialog, "Invalid initial-value overrides", str(exc)); return
            self.params_init.setText(candidate); dialog.accept()

        buttons.accepted.connect(accept); dialog.exec()

    def _edit_initialization_policies(self) -> None:
        import ast
        from pyLOCO.config import BLOCK_ORDER, DEFAULT_INIT_POLICY

        dialog = QDialog(self); dialog.setWindowTitle("Initialization policies — advanced")
        layout = QVBoxLayout(dialog)
        help_label = QLabel(
            "For a new stage, these sources construct its initial parameter vector. When continuing, the previous fitted lattice and calibration state are loaded first, and matching fitted parameter values take precedence."
        )
        help_label.setWordWrap(True); layout.addWidget(help_label)
        raw = self.params_init_policy.text().strip()
        try: current = ast.literal_eval(raw) if raw else dict(DEFAULT_INIT_POLICY)
        except (SyntaxError, ValueError): current = dict(DEFAULT_INIT_POLICY)
        table = QTableWidget(len(BLOCK_ORDER), 2)
        table.setHorizontalHeaderLabels(["Parameter group", "Initialization source"])
        configure_item_view(table); table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents); table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        source_labels = (
            ("Start at 1", "ones"), ("Start at 0", "zeros"),
            ("Horizontal corrector kick steps", "cmstep:h"), ("Vertical corrector kick steps", "cmstep:v"),
            ("Configured RF frequency step", "rfstep"), ("Values from lattice K", "quads:K"),
            ("Configured lattice attribute", "quads:attr"), ("Zero tilt", "tilts:zeros"),
        )
        policy_combos = {}
        for row, key in enumerate(BLOCK_ORDER):
            table.setItem(row, 0, QTableWidgetItem(key))
            combo = QComboBox()
            for label, value in source_labels: combo.addItem(label, value)
            value = current.get(key, DEFAULT_INIT_POLICY.get(key, "zeros"))
            index = combo.findData(value)
            if index < 0: combo.addItem(f"Custom lattice source: {value}", value); index = combo.count() - 1
            combo.setCurrentIndex(index); table.setCellWidget(row, 1, combo); policy_combos[key] = combo
        table.setMinimumSize(700, 390); layout.addWidget(table)
        raw_toggle = QCheckBox("Show raw policy dictionary (advanced)")
        editor = QPlainTextEdit(); editor.setPlainText(raw or repr(dict(current))); editor.setVisible(False); editor.setMinimumHeight(150)
        raw_toggle.toggled.connect(editor.setVisible); layout.addWidget(raw_toggle); layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons); buttons.rejected.connect(dialog.reject)

        def accept() -> None:
            try:
                value = (ast.literal_eval(editor.toPlainText().strip()) if raw_toggle.isChecked()
                         else {key: combo.currentData() for key, combo in policy_combos.items()})
                if not isinstance(value, dict): raise ValueError("Policy configuration must be a dictionary.")
                unknown = sorted(set(value) - set(BLOCK_ORDER))
                if unknown: raise ValueError(f"Unsupported policy key: {unknown[0]}")
                allowed = {"ones", "zeros", "cmstep:h", "cmstep:v", "rfstep", "quads:K", "quads:attr", "tilts:zeros"}
                invalid = [item for item in value.values() if item not in allowed and not (isinstance(item, str) and item.startswith("quads:") and len(item) > 6)]
                if invalid: raise ValueError(f"Unsupported initialization policy: {invalid[0]!r}")
            except (SyntaxError, ValueError) as exc:
                QMessageBox.warning(dialog, "Invalid initialization policies", str(exc)); return
            self.params_init_policy.setText(repr(value)); dialog.accept()

        buttons.accepted.connect(accept); dialog.exec()

    def _set_calculator_value(self, calculator: str) -> None:
        aliases = {"linear": "Linear", "analytical": "Analytical", "numerical": "Numerical", "tracking": "Numerical"}
        backend_value = aliases.get(str(calculator).strip().lower(), calculator)
        index = self.rm_calculator.findData(backend_value)
        if index >= 0:
            self.rm_calculator.setCurrentIndex(index)

    def _update_jacobian_option_availability(self) -> None:
        """Show analytical-only controls without changing their stored values."""
        normal_used = self.parameter_checks["quads"].isChecked()
        skew_used = self.parameter_checks["skew_quads"].isChecked()
        self.quad_jacobian_calculator.setEnabled(normal_used)
        self.skew_jacobian_calculator.setEnabled(skew_used)
        self.quad_jacobian_calculator.setToolTip(
            "" if normal_used else "Not used in this stage because normal quadrupoles are not fitted."
        )
        self.skew_jacobian_calculator.setToolTip(
            "" if skew_used else "Not used in this stage because skew quadrupoles are not fitted."
        )
        normal_analytical = normal_used and self.quad_jacobian_calculator.currentText() == "Analytical"
        skew_analytical = skew_used and self.skew_jacobian_calculator.currentText() == "Analytical"
        self.normal_analytical_options.setVisible(normal_analytical)
        self.normal_analytical_options.setEnabled(normal_analytical)
        self.skew_analytical_options.setVisible(skew_analytical)
        self.skew_analytical_options.setEnabled(skew_analytical)
        self.fixed_dk.setEnabled(
            normal_used and self.quad_jacobian_calculator.currentText() == "Numerical"
        )
        self._refresh_normal_step_resolution()
        self._refresh_numerical_step_status()

    def _automatic_momentum_compaction(self) -> tuple[float | None, str | None]:
        """Return lattice-derived αc for display, without changing FIT inputs."""
        if not self.project.lattice.path:
            return None, None
        signature = (self.project.lattice.path, self.project.lattice.mat_variable)
        if getattr(self, "_mcf_model_cache_signature", None) == signature:
            return self._mcf_model_cache_value
        try:
            from pyLOCO.reference_model import load_reference_model
            model = load_reference_model(
                self.project.resolve_path(self.project.lattice.path),
                source="FIT reference lattice",
                mat_key=self.project.lattice.mat_variable or None,
            )
            result = (float(model.momentum_compaction), None)
        except Exception as exc:
            result = (None, str(exc))
        self._mcf_model_cache_signature = signature
        self._mcf_model_cache_value = result
        return result

    def _update_mcf_source_visibility(self) -> None:
        """Show either the model-derived explanation or the manual αc editor."""
        automatic = (self.mcf_source.currentData() or "automatic") == "automatic"
        self.mcf_automatic_label.setVisible(automatic)
        self.mcf_automatic_summary.setVisible(automatic)
        self.mcf_user_value_label.setVisible(not automatic)
        self.mcf_user_value.setVisible(not automatic)
        self.mcf_user_value.setEnabled(not automatic)
        if automatic:
            value, error = self._automatic_momentum_compaction()
            self.mcf_automatic_summary.setToolTip(error or "")
            if value is not None:
                self.mcf_automatic_summary.setText(
                    f"Calculated from the selected reference lattice using Accelerator Toolbox (pyAT): "
                    f"αc = {value:.9g} (dimensionless)."
                )
            elif error:
                self.mcf_automatic_summary.setText(
                    "Automatic αc is unavailable because the selected reference lattice could not be evaluated."
                )
            else:
                self.mcf_automatic_summary.setText(
                    "Select a reference lattice to calculate the dimensionless momentum compaction factor αc using Accelerator Toolbox (pyAT)."
                )

    def _set_solver_algorithm_value(self, algorithm: str) -> None:
        index = self.solver_algorithm.findData(algorithm)
        if index >= 0:
            self.solver_algorithm.setCurrentIndex(index)
        self._update_solver_scaled_availability()

    def _selected_solver_algorithm(self) -> str:
        return self.solver_algorithm.currentData() or self.solver_algorithm.currentText()

    def _update_solver_scaled_availability(self) -> None:
        is_lm = self._selected_solver_algorithm() == "lm"
        for widget in self._lm_only_controls:
            widget.setVisible(is_lm); widget.setEnabled(is_lm)

    def _selected_svd_method(self) -> str:
        return self.svd_method.currentData() or self.svd_method.currentText()

    def _set_svd_method_value(self, method: str) -> None:
        index = self.svd_method.findData(method)
        if index >= 0:
            self.svd_method.setCurrentIndex(index)
        else:
            self.svd_method.setCurrentText(method)
        self._update_svd_input_availability()

    def _update_svd_input_availability(self) -> None:
        method = self._selected_svd_method()
        parameter_labels = {
            "threshold": "Threshold",
            "rank": "Rank",
            "user_input": "Number of singular values to keep",
            "interactive": "Number of singular values to keep",
        }
        self.svd_parameter_label.setText(parameter_labels.get(method, "SVD parameter"))
        self.svd_threshold.setVisible(method == "threshold")
        self.svd_threshold.setEnabled(method == "threshold")
        self.svd_rank.setVisible(method in {"rank", "user_input", "interactive"})
        self.svd_rank.setEnabled(method in {"rank", "user_input"})
        self.svd_parameter_input.setEnabled(method != "interactive")
        self.svd_plot.setEnabled(method != "interactive")
        if method == "interactive":
            self.svd_plot.setToolTip(
                "Interactive selection is shown in a Qt dialog during the run."
            )
        else:
            self.svd_plot.setToolTip("")

    def _update_cmstep_input_availability(self) -> None:
        is_uniform = (self.cmstep_mode.currentData() or "uniform") == "uniform"
        advanced = self.project.mode == "Advanced"
        for widget in (self.params_cmstep_h_label, self.params_cmstep_h,
                       self.params_cmstep_v_label, self.params_cmstep_v):
            widget.setVisible(advanced and is_uniform); widget.setEnabled(is_uniform)
        for widget in (self.params_cmstep_file_label, self.params_cmstep_file_row,
                       self.params_cmstep_file, self.params_cmstep_browse,
                       self.params_cmstep_resolved):
            widget.setVisible(advanced and not is_uniform); widget.setEnabled(not is_uniform)
        if not is_uniform:
            self._refresh_cmstep_resolved_values()

    @staticmethod
    def _format_resolved_steps(values) -> str:
        import numpy as np
        array = np.asarray(values, dtype=float).ravel()
        if not array.size:
            return "No values"
        if np.allclose(array, array[0], rtol=1e-12, atol=0.0):
            return f"{array[0]:.8g} rad from file ({array.size} value{'s' if array.size != 1 else ''})"
        return (
            f"{array.size} values from file · min {np.min(array):.8g} rad · "
            f"max {np.max(array):.8g} rad"
        )

    def _refresh_cmstep_resolved_values(self) -> None:
        path_text = self.params_cmstep_file.text().strip()
        if not path_text:
            horizontal = vertical = "No corrector-step file selected"
        else:
            try:
                source = self.project.resolve_path(path_text)
                horizontal_values, vertical_values = load_cmstep_npz(source)
                horizontal = self._format_resolved_steps(horizontal_values)
                vertical = self._format_resolved_steps(vertical_values)
            except Exception as exc:
                horizontal = vertical = f"Unable to resolve loaded values: {exc}"
        self.params_cmstep_resolved_h.setText(horizontal)
        self.params_cmstep_resolved_v.setText(vertical)


    @Slot()
    def _browse_cmstep_file(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Load corrector-step file",
            "",
            "NumPy archives (*.npz);;All files (*)",
        )[0]
        if filename:
            self.params_cmstep_file.setText(filename)

    def _browse_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select LOCO output directory", self.output_directory_edit.text()
        )
        if directory:
            self.output_directory_edit.setText(directory)

    @Slot()
    def _browse_resume_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select previous LOCO run or results directory")
        if directory:
            self.resume_directory.setText(directory)
            self.resume_previous.setChecked(True)

    @staticmethod
    def _integer_list(text: str, label: str) -> list[int]:
        stripped = text.strip()
        if not stripped:
            return []
        try:
            values = [int(value) for value in re.split(r"[\s,;]+", stripped) if value]
        except ValueError as exc:
            raise ValueError(f"{label} must contain integer family indices.") from exc
        if any(value < 0 for value in values) or len(values) != len(set(values)):
            raise ValueError(f"{label} must contain unique non-negative indices.")
        return values

    def _update_resume_availability(self) -> None:
        compatibility_ready = self._compatibility_is_validated()
        self.resume_previous.setEnabled(compatibility_ready)
        enabled = compatibility_ready and self.resume_previous.isChecked()
        for widget in (self.resume_directory, self.resume_browse, self.resume_ring_file,
                       self.resume_fit_dict_file, self.resume_fit_results_file):
            widget.setEnabled(enabled)
        if not compatibility_ready:
            self.resume_metadata.setText("Resume disabled until measurement compatibility validation passes.")
            return
        if not enabled:
            self.resume_metadata.setText("Start from the currently selected lattice model.")
            return
        resume = self.project.loco_config.resume
        resume.enabled = True
        resume.directory = self.resume_directory.text()
        resume.ring_file = self.resume_ring_file.text() or "ring_pyloco.mat"
        resume.fit_dict_file = self.resume_fit_dict_file.text() or "fit_dict.pkl"
        resume.fit_results_file = self.resume_fit_results_file.text()
        errors = resume.validation_messages()
        if errors:
            self.resume_metadata.setText("⚠ " + "\n⚠ ".join(errors))
            return
        metadata = resume.metadata()
        details = [f"Source: {metadata.get('source', resume.directory)}"]
        if metadata.get("previous_iterations") is not None:
            details.append(f"Previous iterations: {metadata['previous_iterations']}")
        if metadata.get("previous_final_chi2") is not None:
            details.append(f"Previous final χ²: {float(metadata['previous_final_chi2']):.4e}")
        if metadata.get("fit_list"):
            details.append("Previous fit blocks: " + ", ".join(metadata["fit_list"]))
        self.resume_metadata.setText("✓ " + "\n".join(details))

    def _load_config_to_widgets(self) -> None:
        self._loading_config = True
        cfg = self.project.loco_config
        self.bad_bpm_positions_edit.setText(", ".join(map(str, cfg.bad_bpm_positions)))
        self.hcor_exclusions_edit.setText(", ".join(map(str, cfg.excluded_horizontal_corrector_positions)))
        self.vcor_exclusions_edit.setText(", ".join(map(str, cfg.excluded_vertical_corrector_positions)))
        self._set_calculator_value(cfg.response_matrix.calculator)
        self.rm_dispersion.setChecked(
            cfg.response_matrix.includeDispersion or cfg.rejection.includeDispersion
        )
        self.rm_coupling.setChecked(cfg.response_matrix.coupling_orm)
        self.rm_bidirectional.setChecked(cfg.response_matrix.bidirectional)
        self.rm_vectorized.setChecked(cfg.response_matrix.NewVectorizedMethod)
        self.rm_dkick_h.setValue(cfg.response_matrix.dkick_h)
        self.rm_dkick_v.setValue(cfg.response_matrix.dkick_v)
        self.rm_rf_step.setValue(cfg.response_matrix.rfStep)
        self.rm_delta_coupling.setValue(cfg.response_matrix.delta_coupling)
        self.rm_hcm_coupling.setText(cfg.response_matrix.HCMCoupling)
        self.rm_vcm_coupling.setText(cfg.response_matrix.VCMCoupling)
        self.rm_frequency.setText(cfg.response_matrix.Frequency)
        self.rm_harm_number.setText(cfg.response_matrix.HarmNumber)
        self.rm_rf_attr.setText(cfg.response_matrix.RFAttr)
        self._refresh_element_selection_ui()
        self.rm_fixedpath.setChecked(cfg.response_matrix.fixedpathlength)
        self.rm_log_info.setChecked(cfg.response_matrix.log_info)
        self._set_solver_algorithm_value(cfg.solver.algorithm)
        self.solver_n_iter.setValue(cfg.solver.nIter)
        self.solver_lm_iter.setValue(cfg.solver.nLMIter)
        self.solver_lambda.setValue(cfg.solver.Starting_Lambda)
        self.solver_max_lambda.setValue(cfg.solver.max_lm_lambda)
        self.solver_scaled.setChecked(cfg.solver.scaled)
        self._set_svd_method_value(cfg.svd.svd_selection_method)
        self.svd_threshold.setValue(cfg.svd.svd_threshold)
        self.svd_rank.setValue(cfg.svd.cut_)
        self.svd_plot.setChecked(cfg.svd.show_svd_plot)
        self.outlier_enabled.setChecked(cfg.rejection.outlier_rejection)
        self.outlier_sigma.setValue(cfg.rejection.sigma_outlier)
        self.norm_enabled.setChecked(cfg.rejection.apply_normalization)
        self.norm_mode.setCurrentText(cfg.rejection.normalization_mode)
        self.loco_hor_dispersion_weight.setValue(cfg.rejection.hor_dispersion_weight)
        self.loco_ver_dispersion_weight.setValue(cfg.rejection.ver_dispersion_weight)
        self.auto_delta.setChecked(cfg.rejection.auto_correct_delta)
        self.loco_fixedpath.setChecked(cfg.rejection.fixedpathlength)
        self.loco_individuals.setChecked(cfg.parameters.individuals)
        self.loco_remove_coupling.setChecked(cfg.rejection.remove_coupling_)
        self.loco_plot_fit_parameters.setChecked(cfg.rejection.plot_fit_parameters)
        self.quad_jacobian_calculator.setCurrentText(cfg.rejection.quad_jacobian_calculator)
        self.skew_jacobian_calculator.setCurrentText(cfg.rejection.skew_jacobian_calculator)
        self.analytical_thick_quadrupole.setChecked(cfg.rejection.analytical_thick_quadrupole)
        self.analytical_thick_steerers.setChecked(cfg.rejection.analytical_thick_steerers)
        self.analytical_verbose.setChecked(cfg.rejection.analytical_verbose)
        self.analytical_use_mp.setChecked(cfg.rejection.analytical_use_mp)
        implementation_index = self.analytical_implementation.findData(
            cfg.rejection.analytical_implementation
        )
        self.analytical_implementation.setCurrentIndex(max(0, implementation_index))
        dispersion_calculator_index = self.analytical_dispersion_calculator.findData(
            cfg.rejection.analytical_dispersion_calculator
        )
        self.analytical_dispersion_calculator.setCurrentIndex(
            max(0, dispersion_calculator_index)
        )
        self.analytical_thick_skew.setChecked(cfg.rejection.analytical_thick_skew)
        self.analytical_skew_thick_steerers.setChecked(cfg.rejection.analytical_skew_thick_steerers)
        self.analytical_skew_verbose.setChecked(cfg.rejection.analytical_skew_verbose)
        self.analytical_skew_use_mp.setChecked(cfg.rejection.analytical_skew_use_mp)
        skew_implementation_index = self.skew_analytical_implementation.findData(
            cfg.rejection.skew_analytical_implementation
        )
        self.skew_analytical_implementation.setCurrentIndex(
            max(0, skew_implementation_index)
        )
        index = self.skew_analytical_dispersion_calculator.findData(
            cfg.rejection.skew_analytical_dispersion_calculator
        )
        self.skew_analytical_dispersion_calculator.setCurrentIndex(max(0, index))
        index = self.skew_analytical_dispersion_worker.findData(
            cfg.rejection.skew_analytical_dispersion_worker
        )
        self.skew_analytical_dispersion_worker.setCurrentIndex(max(0, index))
        self._update_jacobian_option_availability()
        self.constraint_enabled.setChecked(cfg.constraints.enable)
        self.constraint_quad_sigma.setValue(cfg.constraints.quad_sigma)
        self.constraint_skew_sigma.setValue(cfg.constraints.skew_sigma)
        self.constraint_quad_weights.setText(cfg.constraints.quad_weights)
        self.constraint_skew_weights.setText(cfg.constraints.skew_weights)
        self.constraint_quad_mask.setText(cfg.constraints.quad_mask)
        self.constraint_skew_mask.setText(cfg.constraints.skew_mask)
        self.constraint_quad_sigma_mode.setCurrentIndex(max(0, self.constraint_quad_sigma_mode.findData(cfg.constraints.quad_sigma_mode)))
        self.constraint_quad_relative_sigma.setValue(cfg.constraints.quad_relative_sigma)
        self.constraint_quad_minimum_sigma.setValue(cfg.constraints.quad_minimum_sigma)
        self.constraint_quad_default_weight.setValue(cfg.constraints.quad_default_weight)
        self.constraint_quad_selected_families.setText(", ".join(map(str, cfg.constraints.quad_selected_families)))
        self.constraint_quad_selected_weight.setValue(cfg.constraints.quad_selected_weight)
        self.constraint_quad_exceptions.set_mapping(cfg.constraints.quad_weighted_families)
        self.constraint_skew_default_weight.setValue(cfg.constraints.skew_default_weight)
        self.constraint_skew_selected_families.setText(", ".join(map(str, cfg.constraints.skew_selected_families)))
        self.constraint_skew_selected_weight.setValue(cfg.constraints.skew_selected_weight)
        self.constraint_skew_exceptions.set_mapping(cfg.constraints.skew_weighted_families)
        for name, check in self.parameter_checks.items():
            check.setChecked(bool(getattr(cfg.parameters, name)))
        modes={"quads":cfg.parameters.individuals,"skew_quads":cfg.rejection.skew_individuals,"quads_tilt":cfg.rejection.tilt_individuals}
        for key,is_individual in modes.items():
            individual,family=self.parameterization_radios[key]; individual.setChecked(is_individual); family.setChecked(not is_individual)
        self.cmstep_mode.setCurrentIndex(max(0, self.cmstep_mode.findData(cfg.parameters.CMstep_mode)))
        self.params_init_policy.setText(cfg.parameters.init_policy)
        self.params_cmstep_h.setValue(float(cfg.parameters.CMstep_h))
        self.params_cmstep_v.setValue(float(cfg.parameters.CMstep_v))
        self.params_cmstep_file.setText(cfg.parameters.CMstep_file)
        self.params_rfstep.setValue(cfg.parameters.rfStep)
        self._refresh_measurement_mapping_table()
        self._refresh_measurement_mapping_feedback()
        self.params_init.setText(cfg.parameters.init)
        self.params_quads_attr.setText(cfg.parameters.quads_attr)
        self.params_quads_attr_index.setValue(cfg.parameters.quads_attr_index)
        self.params_skew_attr.setText(cfg.parameters.skew_attr)
        self.params_skew_attr_index.setValue(cfg.parameters.skew_attr_index)
        self.params_tilt_attr_r1.setText(cfg.parameters.quads_tilt_attr_R1)
        self.params_tilt_attr_r2.setText(cfg.parameters.quads_tilt_attr_R2)
        self.params_tilt_method.setText(cfg.parameters.quads_tilt_method)
        self.fixed_frequency.setText(str(cfg.fixed_parameters.Frequency))
        self.fixed_harm_number.setValue(cfg.fixed_parameters.HarmNumber)
        self.fixed_rfstep.setValue(cfg.fixed_parameters.rfstep)
        self.fixed_dk.setText(cfg.fixed_parameters.dk)
        self.fixed_delta_skew.setValue(cfg.fixed_parameters.delta_skew)
        self.fixed_delta_q_tilt.setValue(cfg.fixed_parameters.delta_q_tilt)
        self.mcf_source.setCurrentIndex(max(0, self.mcf_source.findData(cfg.mcf_source)))
        self.mcf_user_value.setText(cfg.mcf_user_value)
        self._update_mcf_source_visibility()
        self.output_directory_edit.setText(cfg.output_directory)
        self.run_name_edit.setText(cfg.run_name)
        self.save_jacobian_check.setChecked(cfg.rejection.save_jacobians)
        self.resume_previous.setChecked(cfg.resume.enabled)
        self.resume_current.setChecked(not cfg.resume.enabled)
        self.resume_directory.setText(cfg.resume.directory)
        self.resume_ring_file.setText(cfg.resume.ring_file)
        self.resume_fit_dict_file.setText(cfg.resume.fit_dict_file)
        self.resume_fit_results_file.setText(cfg.resume.fit_results_file)
        self._update_resume_availability()
        self._apply_mode_visibility()
        self._update_svd_input_availability()
        self._update_cmstep_input_availability()
        self._refresh_initialization_policy_summary()
        self._refresh_explicit_initial_values_summary()
        self._update_jacobian_option_availability()
        self._refresh_normal_step_resolution()
        self._refresh_numerical_step_status()
        self._update_fit_summary()
        self._update_exclusion_counts()
        self._loading_config = False

    def _collect_loco_configuration(self) -> LocoConfiguration:
        # Preserve advanced values hidden in Basic mode and all forward-compatible
        # source YAML fields while updating only controls visible to the user.
        cfg = deepcopy(self.project.loco_config)
        cfg.output_directory = self.project.loco_config.output_directory
        cfg.response_matrix.calculator = self.rm_calculator.currentData() or self.rm_calculator.currentText()
        cfg.response_matrix.includeDispersion = self.rm_dispersion.isChecked()
        cfg.response_matrix.coupling_orm = self.rm_coupling.isChecked()
        cfg.response_matrix.bidirectional = self.rm_bidirectional.isChecked()
        cfg.response_matrix.NewVectorizedMethod = self.rm_vectorized.isChecked()
        cfg.response_matrix.dkick_h = self.rm_dkick_h.value()
        cfg.response_matrix.dkick_v = self.rm_dkick_v.value()
        cfg.response_matrix.rfStep = self.rm_rf_step.value()
        cfg.response_matrix.delta_coupling = self.rm_delta_coupling.value()
        cfg.machine_elements = self.project.loco_config.machine_elements
        cfg._sync_response_matrix_elements()
        cfg.response_matrix.fixedpathlength = self.rm_fixedpath.isChecked()
        cfg.response_matrix.log_info = self.rm_log_info.isChecked()
        cfg.response_matrix.HCMCoupling = self.rm_hcm_coupling.text()
        cfg.response_matrix.VCMCoupling = self.rm_vcm_coupling.text()
        cfg.response_matrix.Frequency = self.rm_frequency.text()
        cfg.response_matrix.HarmNumber = self.rm_harm_number.text()
        cfg.response_matrix.RFAttr = self.rm_rf_attr.text()
        cfg.solver.algorithm = self._selected_solver_algorithm()
        cfg.solver.nIter = self.solver_n_iter.value()
        cfg.solver.nLMIter = self.solver_lm_iter.value()
        cfg.solver.Starting_Lambda = self.solver_lambda.value()
        cfg.solver.max_lm_lambda = self.solver_max_lambda.value()
        cfg.solver.scaled = self.solver_scaled.isChecked() and cfg.solver.algorithm == "lm"
        cfg.svd.svd_selection_method = self._selected_svd_method()
        cfg.svd.svd_threshold = self.svd_threshold.value()
        cfg.svd.cut_ = self.svd_rank.value()
        cfg.svd.show_svd_plot = self.svd_plot.isChecked()
        cfg.rejection.outlier_rejection = self.outlier_enabled.isChecked()
        cfg.rejection.sigma_outlier = self.outlier_sigma.value()
        cfg.rejection.apply_normalization = self.norm_enabled.isChecked()
        cfg.rejection.normalization_mode = self.norm_mode.currentText()
        cfg.rejection.includeDispersion = self.rm_dispersion.isChecked()
        cfg.response_matrix.includeDispersion = cfg.rejection.includeDispersion
        cfg.rejection.hor_dispersion_weight = self.loco_hor_dispersion_weight.value()
        cfg.rejection.ver_dispersion_weight = self.loco_ver_dispersion_weight.value()
        cfg.rejection.auto_correct_delta = self.auto_delta.isChecked()
        cfg.rejection.fixedpathlength = self.loco_fixedpath.isChecked() or self.rm_fixedpath.isChecked()
        cfg.response_matrix.fixedpathlength = cfg.rejection.fixedpathlength
        cfg.rejection.individuals = self.parameterization_radios["quads"][0].isChecked()
        cfg.rejection.remove_coupling_ = self.loco_remove_coupling.isChecked()
        cfg.rejection.plot_fit_parameters = self.loco_plot_fit_parameters.isChecked()
        cfg.rejection.quad_jacobian_calculator = self.quad_jacobian_calculator.currentText()
        cfg.rejection.skew_jacobian_calculator = self.skew_jacobian_calculator.currentText()
        cfg.rejection.analytical_thick_quadrupole = self.analytical_thick_quadrupole.isChecked()
        cfg.rejection.analytical_thick_steerers = self.analytical_thick_steerers.isChecked()
        cfg.rejection.analytical_verbose = self.analytical_verbose.isChecked()
        cfg.rejection.analytical_use_mp = self.analytical_use_mp.isChecked()
        cfg.rejection.analytical_implementation = self.analytical_implementation.currentData()
        cfg.rejection.analytical_dispersion_calculator = (
            self.analytical_dispersion_calculator.currentData()
        )
        cfg.rejection.analytical_thick_skew = self.analytical_thick_skew.isChecked()
        cfg.rejection.analytical_skew_thick_steerers = self.analytical_skew_thick_steerers.isChecked()
        cfg.rejection.analytical_skew_verbose = self.analytical_skew_verbose.isChecked()
        cfg.rejection.analytical_skew_use_mp = self.analytical_skew_use_mp.isChecked()
        cfg.rejection.skew_analytical_implementation = (
            self.skew_analytical_implementation.currentData()
        )
        cfg.rejection.skew_analytical_dispersion_calculator = (
            self.skew_analytical_dispersion_calculator.currentData()
        )
        cfg.rejection.skew_analytical_dispersion_worker = (
            self.skew_analytical_dispersion_worker.currentData()
        )
        cfg.constraints.enable = self.constraint_enabled.isChecked()
        cfg.constraints.quad_sigma = self.constraint_quad_sigma.value()
        cfg.constraints.skew_sigma = self.constraint_skew_sigma.value()
        cfg.constraints.quad_weights = self.constraint_quad_weights.text()
        cfg.constraints.skew_weights = self.constraint_skew_weights.text()
        cfg.constraints.quad_mask = self.constraint_quad_mask.text()
        cfg.constraints.skew_mask = self.constraint_skew_mask.text()
        cfg.constraints.quad_sigma_mode = self.constraint_quad_sigma_mode.currentData() or "absolute"
        cfg.constraints.quad_relative_sigma = self.constraint_quad_relative_sigma.value()
        cfg.constraints.quad_minimum_sigma = self.constraint_quad_minimum_sigma.value()
        cfg.constraints.quad_default_weight = self.constraint_quad_default_weight.value()
        cfg.constraints.quad_selected_families = self._integer_list(self.constraint_quad_selected_families.text(), "Selected quadrupole families")
        cfg.constraints.quad_selected_weight = self.constraint_quad_selected_weight.value()
        cfg.constraints.quad_weighted_families = self.constraint_quad_exceptions.mapping()
        cfg.constraints.skew_default_weight = self.constraint_skew_default_weight.value()
        cfg.constraints.skew_selected_families = self._integer_list(self.constraint_skew_selected_families.text(), "Selected skew families")
        cfg.constraints.skew_selected_weight = self.constraint_skew_selected_weight.value()
        cfg.constraints.skew_weighted_families = self.constraint_skew_exceptions.mapping()
        for name, check in self.parameter_checks.items():
            setattr(cfg.parameters, name, check.isChecked())
        cfg.parameters.individuals = self.parameterization_radios["quads"][0].isChecked()
        cfg.rejection.skew_individuals = self.parameterization_radios["skew_quads"][0].isChecked()
        cfg.rejection.tilt_individuals = self.parameterization_radios["quads_tilt"][0].isChecked()
        cfg.parameters.CMstep_mode = self.cmstep_mode.currentData() or "uniform"
        cfg.parameters.init_policy = self.params_init_policy.text()
        cfg.parameters.CMstep_h = self.params_cmstep_h.value()
        cfg.parameters.CMstep_v = self.params_cmstep_v.value()
        cfg.parameters.CMstep_file = self.params_cmstep_file.text()
        cfg.parameters.rfStep = self.params_rfstep.value()
        cfg.parameters.init = self.params_init.text()
        cfg.parameters.quads_attr = self.params_quads_attr.text()
        cfg.parameters.quads_attr_index = self.params_quads_attr_index.value()
        cfg.parameters.skew_attr = self.params_skew_attr.text()
        cfg.parameters.skew_attr_index = self.params_skew_attr_index.value()
        cfg.parameters.quads_tilt_attr_R1 = self.params_tilt_attr_r1.text()
        cfg.parameters.quads_tilt_attr_R2 = self.params_tilt_attr_r2.text()
        cfg.parameters.quads_tilt_method = self.params_tilt_method.text()
        cfg.fixed_parameters.Frequency = self.fixed_frequency.text()
        cfg.fixed_parameters.HarmNumber = self.fixed_harm_number.value()
        cfg.fixed_parameters.rfstep = self.fixed_rfstep.value()
        cfg.fixed_parameters.dk = self.fixed_dk.text()
        cfg.fixed_parameters.delta_skew = self.fixed_delta_skew.value()
        cfg.fixed_parameters.delta_q_tilt = self.fixed_delta_q_tilt.value()
        cfg.mcf_source = self.mcf_source.currentData() or "automatic"
        cfg.mcf_user_value = self.mcf_user_value.text()
        cfg.output_directory = self.output_directory_edit.text().strip()
        cfg.run_name = self.run_name_edit.text().strip()
        cfg.rejection.save_jacobians = self.save_jacobian_check.isChecked()
        cfg.bad_bpm_positions = self._parse_position_text(self.bad_bpm_positions_edit.text())
        cfg.excluded_horizontal_corrector_positions = self._parse_position_text(self.hcor_exclusions_edit.text())
        cfg.excluded_vertical_corrector_positions = self._parse_position_text(self.vcor_exclusions_edit.text())
        cfg.resume.enabled = self.resume_previous.isChecked()
        cfg.resume.directory = self.resume_directory.text()
        cfg.resume.ring_file = self.resume_ring_file.text() or "ring_pyloco.mat"
        cfg.resume.fit_dict_file = self.resume_fit_dict_file.text() or "fit_dict.pkl"
        cfg.resume.fit_results_file = self.resume_fit_results_file.text()
        return cfg

    @Slot()
    def _on_fit_config_changed(self) -> None:
        if self._loading_config:
            return
        try:
            self.project.loco_config = self._collect_loco_configuration()
        except ValueError as exc:
            self._validation_label.setText(f"Validation: {exc}")
            self._validation_label.setObjectName("validationMissing")
            return
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages):
            self.fit_recipe.stages[self._active_fit_stage].configuration = {
                "gui_config": json_safe(__import__("dataclasses").asdict(self.project.loco_config)),
                "backend_mapping": json_safe(self.project.loco_config.to_backend_mapping()),
            }
            self._record_fit_stage_edit(
                f"Stage {self._active_fit_stage + 1} FIT settings changed; validate before running."
            )
            self._refresh_fit_workflow_table(self._active_fit_stage)
        else:
            self.project.modified = True
        self._update_fit_summary()
        self._refresh_ui("LOCO configuration updated")

    def _update_fit_summary(self) -> None:
        if not hasattr(self, "fit_backend_options"):
            return
        cfg = self.project.loco_config
        backend = json.dumps(json_safe(cfg.to_backend_mapping()), indent=2)
        self.fit_backend_options.setPlainText(backend)

        group_labels = {
            "quads": "Normal quadrupole strengths",
            "skew_quads": "Skew quadrupole strengths",
            "quads_tilt": "Quadrupole tilts",
            "hbpm_gain": "Horizontal BPM gains", "vbpm_gain": "Vertical BPM gains",
            "hbpm_coupling": "Horizontal BPM coupling", "vbpm_coupling": "Vertical BPM coupling",
            "hcor_cal": "Horizontal corrector calibration", "vcor_cal": "Vertical corrector calibration",
            "hcor_coupling": "Horizontal corrector coupling", "vcor_coupling": "Vertical corrector coupling",
            "HCMEnergyShift": "Horizontal corrector energy shifts",
            "VCMEnergyShift": "Vertical corrector energy shifts", "delta_rf": "RF frequency shift",
        }
        fitted = [group_labels.get(key, key) for key in cfg.parameters.fit_list()]
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages):
            stage = self.fit_recipe.stages[self._active_fit_stage]
            starting = {
                "original_model": "Reference lattice",
                "previous_stage": "Previous fitted stage",
                "saved_result": "Saved fitted result",
            }.get(stage.start_from, stage.start_from)
            self.fit_current_stage_summary.setText(
                f"<h3 style='color:#A98BFF;margin:4px 0'>Current stage overview</h3>"
                f"<b>Stage {self._active_fit_stage + 1}: {escape(stage.name)}</b><br>"
                f"Starting model: {escape(starting)} · Outer iterations: {cfg.solver.nIter}<br>"
                f"Fitted groups: {escape(', '.join(fitted) if fitted else 'None selected')}"
            )
        else:
            self.fit_current_stage_summary.setText(
                "<h3 style='color:#A98BFF;margin:4px 0'>Current stage overview</h3>"
                "No fitting stage is currently configured."
            )

        stage_rows = []
        for index, stage in enumerate(self.fit_recipe.stages):
            stage_cfg = self._configuration_for_fit_stage(stage)
            stage_groups = [group_labels.get(key, key) for key in stage_cfg.parameters.fit_list()]
            starting = {
                "original_model": "Reference lattice",
                "previous_stage": "Previous stage",
                "saved_result": "Saved result",
            }.get(stage.start_from, stage.start_from)
            differences = (
                f"{stage_cfg.response_matrix.calculator} ORM; "
                f"{stage_cfg.solver.algorithm.upper()} solver; "
                f"SVD {stage_cfg.svd.svd_selection_method}"
            )
            stage_rows.append(
                "<tr>"
                f"<td>{index + 1}</td><td><b>{escape(stage.name)}</b></td>"
                f"<td>{escape(', '.join(stage_groups) if stage_groups else 'None')}</td>"
                f"<td>{stage_cfg.solver.nIter}</td><td>{escape(starting)}</td>"
                f"<td>{escape(differences)}</td></tr>"
            )
        if stage_rows:
            self.fit_all_stages_summary.setText(
                "<h3 style='color:#A98BFF;margin:12px 0 4px'>All stages overview</h3>"
                "<table cellspacing='0' cellpadding='5' width='100%'>"
                "<tr><th>#</th><th align='left'>Stage</th><th align='left'>Fitted groups</th>"
                "<th>Iterations</th><th align='left'>Starting model</th><th align='left'>Relevant settings</th></tr>"
                + "".join(stage_rows) + "</table>"
            )
        else:
            self.fit_all_stages_summary.setText(
                "<h3 style='color:#A98BFF;margin:12px 0 4px'>All stages overview</h3>No stages configured."
            )

        magnet_rows = []
        for label, enabled, ords, groups, individual in (
            ("Normal quadrupoles", cfg.parameters.quads, cfg.machine_elements.normal_quadrupole_ords,
             cfg.machine_elements.normal_quadrupole_groups, cfg.parameters.individuals),
            ("Skew quadrupoles", cfg.parameters.skew_quads, cfg.machine_elements.skew_quadrupole_ords,
             cfg.machine_elements.skew_quadrupole_groups, cfg.rejection.skew_individuals),
            ("Quadrupole tilts", cfg.parameters.quads_tilt, cfg.machine_elements.quadrupole_tilt_ords,
             cfg.machine_elements.quadrupole_tilt_groups, cfg.rejection.tilt_individuals),
        ):
            active_count = (len(ords) if individual else len(groups)) if enabled else 0
            state = (
                f"Fitted; {active_count} active fitting parameter{'s' if active_count != 1 else ''}"
                if enabled else "Inactive; 0 active fitting parameters"
            )
            magnet_rows.append(
                f"<tr><td><b>{label}</b></td><td>{state}; {len(ords)} physical magnets selected</td></tr>"
            )
        mcf_text = (
            "Automatic from reference lattice (AT/pyAT)"
            if cfg.mcf_source == "automatic" else f"Manual αc = {escape(cfg.mcf_user_value or 'not entered')} (dimensionless)"
        )
        readable_rows = (
            f"<tr><td><b>Response matrix</b></td><td>{escape(cfg.response_matrix.calculator)}; "
            f"dispersion {'included' if cfg.response_matrix.includeDispersion else 'not included'}; "
            f"coupling {'included' if cfg.response_matrix.coupling_orm else 'not included'}</td></tr>"
            f"<tr><td><b>Solver</b></td><td>{escape(cfg.solver.algorithm.upper())}; "
            f"{cfg.solver.nIter} outer iterations; {cfg.solver.nLMIter} LM inner iterations</td></tr>"
            f"<tr><td><b>SVD selection</b></td><td>{escape(cfg.svd.svd_selection_method)}; "
            f"threshold {cfg.svd.svd_threshold:g}; rank {cfg.svd.cut_}</td></tr>"
            f"<tr><td><b>Momentum compaction αc</b></td><td>{mcf_text}</td></tr>"
            f"<tr><td><b>Measurement exclusions</b></td><td>{len(cfg.bad_bpm_positions)} BPM positions; "
            f"{len(cfg.excluded_horizontal_corrector_positions)} horizontal and "
            f"{len(cfg.excluded_vertical_corrector_positions)} vertical corrector positions</td></tr>"
            + "".join(magnet_rows)
        )
        self.fit_scientific_summary.setText(
            "<h3 style='color:#A98BFF;margin:12px 0 4px'>Detailed scientific settings</h3>"
            "<table cellspacing='0' cellpadding='5' width='100%'>" + readable_rows + "</table>"
        )

    @Slot()
    def _scroll_to_fit_summary(self) -> None:
        """Bring the final review section into view without changing settings."""
        self.fit_scroll.ensureWidgetVisible(self.summary_group, 0, 12)
        self.summary_group.setFocus(Qt.OtherFocusReason)

    @Slot()
    def import_loco_configuration(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Import LOCO configuration",
            "",
            "Configuration (*.json *.yaml *.yml);;JSON (*.json);;YAML (*.yaml *.yml)",
        )[0]
        if not filename:
            return
        try:
            config, measurements, lattice = load_example_project_data(filename)
            candidate_lattice = None
            if lattice:
                source = Path(lattice)
                if not source.exists():
                    raise ValueError(f"Configured lattice file does not exist: {source}")
                mat_variable = ""
                if source.suffix.lower() == ".mat" and Path(filename).suffix.lower() in {".yaml", ".yml"}:
                    import yaml
                    imported = yaml.safe_load(Path(filename).read_text(encoding="utf-8")) or {}
                    lattice_spec = imported.get("lattice") or {}
                    mat_variable = str(lattice_spec.get("use") or lattice_spec.get("mat_key") or "")
                loaded_lattice = load_project_lattice(source, mat_variable or None)
                candidate_lattice = LatticeSelection(
                    path=str(source), file_type=source.suffix.lower().lstrip("."),
                    mat_variable=mat_variable, element_count=len(loaded_lattice),
                )
                resolved = resolve_example_machine_elements(filename, loaded_lattice)
                if any(getattr(resolved, key) for key in ELEMENT_ROLES):
                    config.machine_elements = resolved
                    config._sync_response_matrix_elements()
            candidate_measurements = {}
            for role, value in measurements.items():
                source = Path(value)
                if not source.exists():
                    raise ValueError(f"Configured {role.replace('_', ' ')} file does not exist: {source}")
                candidate_measurements[role] = ImportedDataset(
                    role=role, path=str(source), file_type=source.suffix.lower().lstrip("."),
                    size_bytes=source.stat().st_size,
                    options=measurement_options_from_config(config.source_config).get(role, {}),
                )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        affected = [
            "• Replace the current stage's LOCO, solver, parameter, weighting and exclusion settings.",
            "• Keep the existing multi-stage recipe and stage order.",
        ]
        affected.append(
            "• Replace the reference lattice declared by the imported file."
            if candidate_lattice is not None else
            "• Keep the current reference lattice (none is declared by the imported file)."
        )
        affected.append(
            f"• Replace/add these measurement bindings: {', '.join(sorted(candidate_measurements))}."
            if candidate_measurements else
            "• Keep all current measurement bindings (none are declared by the imported file)."
        )
        if self.project.modified:
            affected.insert(0, "The current project has unsaved changes.")
        answer = QMessageBox.question(
            self,
            "Import FIT configuration?",
            "This import does not load a multi-stage recipe. It will make the following changes:\n\n"
            + "\n".join(affected)
            + "\n\nContinue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self.project.loco_config = config
        if candidate_lattice is not None:
            self.project.lattice = candidate_lattice
        self.project.measurements.update(candidate_measurements)
        self.project.modified = True
        self._mark_compatibility_stale("Imported configuration changed; run compatibility validation again.")
        self._load_config_to_widgets()
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages):
            self.fit_recipe.stages[self._active_fit_stage].configuration = self._stage_configuration_snapshot()
            self.project.fit_recipe = self.fit_recipe.to_dict()
        self._refresh_ui(f"Imported LOCO configuration {Path(filename).name}")

    @Slot()
    def export_loco_configuration(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self,
            "Export LOCO configuration",
            f"{self.project.name}-loco-config.json",
            "JSON (*.json);;YAML (*.yaml *.yml)",
        )[0]
        if not filename:
            return
        self.project.loco_config = self._collect_loco_configuration()
        try:
            target = self.project.loco_config.save(filename)
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._refresh_ui(f"Exported LOCO configuration {target.name}")

    @Slot()
    def new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        recent = self.project.recent_projects
        self.project = ProjectMetadata(recent_projects=recent)
        self.dashboard_name.setText(self.project.name)
        self.dashboard_description.setText(self.project.description)
        self._load_config_to_widgets()
        self.fit_recipe = FitRecipe(); self._active_fit_stage = -1
        self._add_fit_stage(name="Stage 1", start_from="original_model")
        # A new project always starts at its dashboard.  Keeping the previous
        # workspace selected can leave an empty Results/Fit view on screen,
        # which makes the freshly-created project appear broken.
        self._workspace.setCurrentIndex(0)
        self._refresh_ui("New project created")

    def _choose_open_project_measurement_policy(self, candidate: ProjectMetadata) -> str | None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Open FIT project — measurement selection")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText("This FIT window already contains measurement data.")
        dialog.setInformativeText(
            "Keep current measurements (recommended) loads the selected project's lattice, machine elements, "
            "exclusions and FIT configuration while retaining the active measurement files and session manifest.\n\n"
            "Use measurements saved in the project opens the selected project exactly as saved."
        )
        keep = dialog.addButton("Keep current measurements (recommended)", QMessageBox.AcceptRole)
        saved = dialog.addButton("Use measurements saved in the project", QMessageBox.ActionRole)
        cancel = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(keep)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is keep:
            return "keep"
        if clicked is saved:
            return "saved"
        if clicked is cancel:
            return None
        return None

    @staticmethod
    def _measurement_binding_signature(project: ProjectMetadata) -> tuple:
        return (
            tuple((role, str(project.resolve_path(dataset.path))) for role, dataset in project.measurements.items()),
            str((project.measurement_session or {}).get("session_id", "")),
            str((project.measurement_session or {}).get("manifest", "")),
        )

    def _compatibility_is_validated(self, project: ProjectMetadata | None = None) -> bool:
        state = (project or self.project).compatibility_validation or {}
        return not state.get("required") or state.get("status") == "validated"

    def _mark_compatibility_stale(self, reason: str) -> None:
        state = self.project.compatibility_validation or {}
        if not state.get("required"):
            return
        state = deepcopy(state)
        state.update(status="validation_required", issues=[reason])
        self.project.compatibility_validation = state

    @staticmethod
    def _stage_machine_signatures(recipe: FitRecipe) -> list[tuple]:
        signatures = []
        for stage in recipe.stages:
            mapping = stage.configuration.get("backend_mapping", stage.configuration)
            elements = mapping.get("MachineElements", {})
            excluded = mapping.get("ExcludedCorrectorPositions", {})
            signatures.append((
                tuple(elements.get("bpm_ords") or []),
                tuple(elements.get("horizontal_corrector_ords") or []),
                tuple(elements.get("vertical_corrector_ords") or []),
                tuple(elements.get("normal_quadrupole_ords") or []),
                tuple(elements.get("skew_quadrupole_ords") or []),
                tuple(elements.get("quadrupole_tilt_ords") or []),
                tuple(mapping.get("BadBPMPositions") or []),
                tuple(excluded.get("horizontal") or []),
                tuple(excluded.get("vertical") or []),
            ))
        return signatures

    def _choose_adapted_machine_policy(self, recipe: FitRecipe) -> str | None:
        signatures = self._stage_machine_signatures(recipe)
        if len(set(signatures)) <= 1:
            return "current"
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Review stage-specific machine selections")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText("The saved fitting stages use different machine-element selections.")
        dialog.setInformativeText(
            "Preserve stage-specific selections keeps each stage exactly as saved. Applying a shared machine "
            "configuration replaces those selections in every stage and requires explicit confirmation."
        )
        preserve = dialog.addButton("Preserve stage-specific selections", QMessageBox.AcceptRole)
        shared = dialog.addButton("Apply shared project machine configuration", QMessageBox.ActionRole)
        cancel = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(preserve)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is preserve:
            return "recipe"
        if clicked is shared:
            return "current"
        if clicked is cancel:
            return None
        return None

    def _measurement_requirements_from_active_files(self, candidate: ProjectMetadata, recipe: FitRecipe) -> tuple[dict | None, list[str]]:
        paths = {role: str(candidate.resolve_path(dataset.path)) for role, dataset in candidate.measurements.items()}
        options = {role: deepcopy(dataset.options) for role, dataset in candidate.measurements.items()}
        old = recipe.data_requirements or {}
        payload = inspect_measurement_files(
            paths, options,
            trusted_hashes=old.get("measurement_sha256", {}),
            trusted_identity=old,
        )
        if payload["errors"]:
            return None, list(payload["errors"])
        files = payload["files"]
        orm = files.get("orm", {})
        required = ("bpm_names", "horizontal_corrector_names", "vertical_corrector_names")
        if any(not orm.get(key) for key in required):
            return None, ["The active ORM does not provide verified BPM and corrector identities."]
        requirements = deepcopy(old)
        for key in required:
            requirements[key] = list(orm[key])
        requirements["measurement_sha256"] = {
            role: entry.get("sha256", "") for role, entry in files.items() if entry.get("sha256")
        }
        requirements["units"] = {
            **dict(old.get("units", {})),
            **{role: entry.get("unit") for role, entry in files.items() if entry.get("unit")},
        }
        dispersion = files.get("dispersion", {})
        if dispersion.get("rf_step_hz") is not None:
            requirements["rf_step_hz"] = float(dispersion["rf_step_hz"])
        return requirements, []

    def _reconcile_retained_measurement_mapping(self, candidate: ProjectMetadata) -> list[str]:
        """Revalidate a saved mapping against retained measurements and lattice.

        Opening a FIT project with ``Keep current measurements`` deliberately
        keeps the project's scientific configuration.  Its mapping may only be
        kept as validated evidence after rebuilding the active HDF5 inventory
        and checking every saved source/target identity and fingerprint.  No
        positional or fuzzy remapping is performed here.
        """

        paths = {
            role: str(candidate.resolve_path(dataset.path))
            for role, dataset in candidate.measurements.items()
        }
        options = {
            role: deepcopy(dataset.options)
            for role, dataset in candidate.measurements.items()
        }
        saved_mapping = candidate.measurement_device_mapping or {}
        payload = inspect_measurement_files(paths, options)
        if payload["errors"]:
            if not saved_mapping and all(
                "identities are absent" in error.lower()
                for error in payload["errors"]
            ):
                # Historical projects without identity-bearing HDF5 files use
                # recipe-pinned checksums in the established validator below.
                candidate.measurement_device_mapping = {}
                return []
            return list(payload["errors"])
        inventory = measurement_inventory(payload)
        if not all(inventory.values()):
            # Legacy measurements without authoritative identities continue
            # through their existing compatibility checks.  They cannot be
            # promoted to a validated device mapping here.
            candidate.measurement_device_mapping = {}
            return []

        if not saved_mapping:
            # Established legacy projects predate explicit mapping evidence and
            # retain their existing identity/order checks.  Projects that opt
            # into compatibility validation must fail closed when that evidence
            # is absent.
            if not (candidate.compatibility_validation or {}).get("required"):
                return []
            return [
                "Measurement-device mapping is missing for the retained ordered BPM/H/V inventory; "
                "run Find machine mappings and validate before fitting."
            ]

        lattice_path = candidate.resolve_path(candidate.lattice.path) if candidate.lattice.path else Path()
        if not lattice_path.is_file():
            return ["The selected project's reference lattice is unavailable."]
        try:
            lattice = load_project_lattice(lattice_path, candidate.lattice.mat_variable or None)
        except Exception as exc:
            return [f"The selected project's reference lattice could not be loaded: {exc}"]

        orm_metadata = (options.get("orm", {}).get("acquisition_metadata") or {})
        pyaml_options = orm_metadata.get("pyaml_options") or {}
        raw_configuration = pyaml_options.get("config_path") or ""
        configuration = Path(raw_configuration).expanduser() if raw_configuration else None
        if configuration is not None and not configuration.is_absolute():
            configuration = candidate.resolve_path(configuration)
        if configuration is not None and not configuration.is_file():
            configuration = None
        try:
            candidate.measurement_device_mapping = validate_measurement_mapping(
                saved_mapping,
                lattice,
                inventory,
                lattice_path=lattice_path,
                configuration_path=configuration,
                require_provenance=True,
            )
        except Exception as exc:
            candidate.measurement_device_mapping = {}
            return [
                "Measurement-device mapping was invalidated for the retained measurements: "
                f"{exc}. Run Find machine mappings and validate before fitting."
            ]
        return []

    def _recipe_from_project(self, project: ProjectMetadata) -> FitRecipe | None:
        raw = project.fit_recipe or {}
        if not raw or not raw.get("stages"):
            return None
        data = deepcopy(raw)
        data["stages"] = [FitStage(**item) for item in data.get("stages", [])]
        return FitRecipe(**data)

    def _fit_run_measurements_are_stale(self, project: ProjectMetadata) -> bool:
        binding = project.fit_run_session or {}
        raw_path = binding.get("path")
        if not raw_path:
            return False
        session_path = project.resolve_path(raw_path)
        if not session_path.is_file():
            return True
        try:
            from .fit_workflow import FitRunSession
            session = FitRunSession.load(session_path)
        except Exception:
            return True
        expected_paths = {
            role: str(project.resolve_path(dataset.path))
            for role, dataset in project.measurements.items()
        }
        actual_paths = {}
        for role, path in session.measurements.items():
            value = Path(path).expanduser()
            actual_paths[role] = str((value if value.is_absolute() else session_path.parent / value).resolve())
        if actual_paths != expected_paths:
            return True
        project_session = str((project.measurement_session or {}).get("session_id", ""))
        run_session = str((session.measurement_identity or {}).get("session_id", ""))
        if project_session and run_session and project_session != run_session:
            return True
        for checkpoint in session.checkpoints:
            checkpoint_session = str((checkpoint.measurement_identity or {}).get("session_id", ""))
            if project_session and checkpoint_session and checkpoint_session != project_session:
                return True
        return False

    def _validate_retained_measurements(self, candidate: ProjectMetadata, *,
                                        comparison_only: bool = False) -> list[str]:
        """Validate a candidate-project/current-measurement combination before activation."""

        errors: list[str] = []
        missing = [
            f"{role}: {candidate.resolve_path(dataset.path)}"
            for role, dataset in candidate.measurements.items()
            if not candidate.resolve_path(dataset.path).is_file()
        ]
        if missing:
            return ["Retained measurement file is missing: " + item for item in missing]
        recipe = self._recipe_from_project(candidate)
        lattice_path = candidate.resolve_path(candidate.lattice.path) if candidate.lattice.path else Path()
        if not lattice_path.is_file():
            return ["The selected project's reference lattice is unavailable."]
        if recipe is not None:
            recipe_errors = recipe.validate()
            if recipe_errors:
                return [
                    "MALFORMED_PROJECT_CONFIGURATION: " + error
                    for error in recipe_errors
                ]
            requirements = recipe.data_requirements or {}
            required_identities = {
                "BPM identities": requirements.get("bpm_names") or [],
                "horizontal-corrector identities": requirements.get("horizontal_corrector_names") or [],
                "vertical-corrector identities": requirements.get("vertical_corrector_names") or [],
            }
            missing_identity = [label for label, values in required_identities.items() if not values]
            pinned_hashes = requirements.get("measurement_sha256") or {}
            missing_hashes = [role for role in ("orm", "dispersion", "bpm_noise") if not pinned_hashes.get(role)]
            if (missing_identity or missing_hashes) and not candidate.measurement_device_mapping:
                stage_counts = []
                for position, stage in enumerate(recipe.stages, 1):
                    mapping = stage.configuration.get("backend_mapping", stage.configuration)
                    elements = mapping.get("MachineElements", {})
                    stage_counts.append(
                        f"stage {position} ({stage.name}): "
                        f"{len(elements.get('bpm_ords') or [])} BPMs, "
                        f"{len(elements.get('horizontal_corrector_ords') or [])} H and "
                        f"{len(elements.get('vertical_corrector_ords') or [])} V correctors"
                    )
                return [
                    "MISSING_MEASUREMENT_PROVENANCE: The selected project's fitting recipe does not record "
                    "the measurement identities and pinned file checksums required to validate replacement "
                    "measurements. Missing: "
                    + ", ".join(missing_identity + (["checksums for " + ", ".join(missing_hashes)] if missing_hashes else []))
                    + ". Recorded stage selections: " + "; ".join(stage_counts)
                ]
            lattice = None
            try:
                lattice = load_project_lattice(lattice_path, candidate.lattice.mat_variable or None)
                names = [self._element_identity(element)[0] for element in lattice]
                report = preflight_recipe(
                    recipe,
                    lattice_path=str(lattice_path),
                    measurement_identity=deepcopy(candidate.measurement_session),
                    lattice_element_names=names,
                    measurements={role: str(candidate.resolve_path(dataset.path)) for role, dataset in candidate.measurements.items()},
                    measurement_options={role: deepcopy(dataset.options) for role, dataset in candidate.measurements.items()},
                    measurement_device_mapping=deepcopy(candidate.measurement_device_mapping),
                    lattice_mat_variable=candidate.lattice.mat_variable or None,
                    comparison_only=comparison_only,
                )
                errors.extend(report["errors"])
            except Exception as exc:
                errors.append(f"Compatibility validation failed: {exc}")
            return errors
        try:
            measured = _load_measurements(
                {role: str(candidate.resolve_path(dataset.path)) for role, dataset in candidate.measurements.items()},
                {role: deepcopy(dataset.options) for role, dataset in candidate.measurements.items()},
            )
        except Exception as exc:
            return [f"Measurement files could not be validated: {exc}"]
        elements = candidate.loco_config.machine_elements
        nbpm = len(elements.bpm_ords)
        nh = len(elements.horizontal_corrector_ords)
        nv = len(elements.vertical_corrector_ords)
        orm = np.asarray(measured.get("orm")) if measured.get("orm") is not None else None
        if orm is not None and orm.shape != (2 * nbpm, nh + nv):
            errors.append(
                f"Retained ORM shape {orm.shape} does not match the selected {nbpm} BPMs and "
                f"{nh} H/{nv} V correctors; expected {(2 * nbpm, nh + nv)}."
            )
        for key, label in (("noise_x", "horizontal BPM noise"), ("noise_y", "vertical BPM noise"),
                           ("eta_x", "horizontal dispersion"), ("eta_y", "vertical dispersion")):
            values = measured.get(key)
            if values is not None and np.asarray(values).shape != (nbpm,):
                errors.append(f"Retained {label} shape {np.asarray(values).shape} does not match {nbpm} selected BPMs.")
        orm_options = candidate.measurements.get("orm")
        if orm_options:
            options = orm_options.options
            if options.get("scaled") is True or options.get("response_matrix_unit") not in (None, "", "m"):
                errors.append("Retained ORM is not the required unscaled orbit difference in metres [m].")
        identity_sets = {
            role: dataset.options.get("bpm_names")
            for role, dataset in candidate.measurements.items()
            if dataset.options.get("bpm_names")
        }
        if identity_sets:
            first_role, first = next(iter(identity_sets.items()))
            if len(first) != nbpm:
                errors.append(f"Retained {first_role} contains {len(first)} BPM identities, expected {nbpm}.")
            for role, names in identity_sets.items():
                if names != first:
                    errors.append(f"BPM identities/order differ between retained {first_role} and {role} files.")
        return errors

    def _project_open_diagnostics(self, candidate: ProjectMetadata, errors: list[str]) -> str:
        recipe = self._recipe_from_project(candidate)
        requirements = recipe.data_requirements if recipe is not None else {}
        lines = [
            f"Selected project: {candidate.path or 'Path not recorded'}",
            f"Reference lattice: {candidate.resolve_path(candidate.lattice.path) if candidate.lattice.path else 'Not configured'}",
            f"Fitting stages: {len(recipe.stages) if recipe is not None else 0}",
            "Required identities: "
            f"{len(requirements.get('bpm_names') or [])} BPMs, "
            f"{len(requirements.get('horizontal_corrector_names') or [])} H correctors, "
            f"{len(requirements.get('vertical_corrector_names') or [])} V correctors",
            "",
            "Active measurement files:",
        ]
        for role, dataset in self.project.measurements.items():
            path = self.project.resolve_path(dataset.path)
            lines.append(f"- {role}: {path} ({'available' if path.is_file() else 'missing'})")
        lines.extend(("", "Validation failures:"))
        lines.extend(
            "- " + error.removeprefix("MALFORMED_PROJECT_CONFIGURATION: ").removeprefix(
                "MISSING_MEASUREMENT_PROVENANCE: "
            )
            for error in errors
        )
        return "\n".join(lines)

    def _project_open_error_content(self, candidate: ProjectMetadata, errors: list[str]) -> dict[str, str]:
        malformed = any(error.startswith("MALFORMED_PROJECT_CONFIGURATION:") for error in errors)
        missing_provenance = any(error.startswith("MISSING_MEASUREMENT_PROVENANCE:") for error in errors)
        missing_file = any("missing" in error.lower() or "does not exist" in error.lower() for error in errors)
        internal = any("validation failed" in error.lower() for error in errors)
        if malformed:
            title = "Unable to open FIT project"
            summary = "The selected project contains an incomplete or invalid FIT-stage configuration."
            action = "Correct the saved project configuration or select another FIT project."
        elif missing_provenance:
            title = "Measurements cannot be verified"
            summary = "The project is editable, but it lacks the identities and checksums needed to verify these measurements."
            action = "Open for editing to review the configuration, or use measurements saved in this project."
        elif missing_file:
            title = "Unable to open FIT project"
            summary = "A required project, lattice or measurement file is unavailable."
            action = "Locate the missing file or select another FIT project."
        elif internal:
            title = "Unable to open FIT project"
            summary = "An internal error prevented measurement compatibility validation."
            action = "Review the technical details and keep the current project open."
        else:
            title = "Measurements are incompatible"
            summary = "The currently loaded measurements are not compatible with the selected FIT project."
            action = "Select a compatible project or use the measurements saved in this project."
        return {
            "title": title,
            "summary": summary,
            "information": f"{action}\n\nThe current project and all measurement bindings remain unchanged.",
            "details": self._project_open_diagnostics(candidate, errors),
        }

    def _show_project_open_validation_error(self, candidate: ProjectMetadata, errors: list[str]) -> None:
        content = self._project_open_error_content(candidate, errors)
        dialog = QMessageBox(self)
        dialog.setWindowTitle(content["title"])
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(content["summary"])
        dialog.setInformativeText(content["information"])
        dialog.setDetailedText(content["details"])
        dialog.setStandardButtons(QMessageBox.Ok)
        for button in dialog.buttons():
            if "detail" in button.text().lower():
                button.setText("Show technical details…")
        dialog.exec()

    @staticmethod
    def _compatibility_errors_are_editable(errors: list[str]) -> bool:
        if any(error.startswith("MALFORMED_PROJECT_CONFIGURATION:") for error in errors):
            return False
        hard_markers = (
            "reference lattice is unavailable",
            "reference lattice could not be loaded",
            "compatibility validation failed",
            "unsupported recipe schema",
            "configuration is incomplete",
        )
        return bool(errors) and not any(
            marker in error.lower() for error in errors for marker in hard_markers
        )

    def _choose_open_for_editing(self, candidate: ProjectMetadata, errors: list[str]) -> bool:
        content = self._project_open_error_content(candidate, errors)
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Measurements are incompatible")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText("The measurements and saved FIT configuration are not currently compatible.")
        dialog.setInformativeText(
            "Open an unsaved working copy to adjust machine selections and FIT settings. Fitting and resume "
            "remain disabled until validation passes.\n\nCancel keeps the current project unchanged."
        )
        dialog.setDetailedText(content["details"])
        edit = dialog.addButton("Open for editing — fitting disabled", QMessageBox.AcceptRole)
        cancel = dialog.addButton("Cancel — keep everything unchanged", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        return dialog.clickedButton() is edit

    def _prepare_incompatible_working_copy(
        self, candidate: ProjectMetadata, errors: list[str], source_filename: str
    ) -> bool:
        recipe = self._recipe_from_project(candidate)
        if recipe is None:
            return False
        policy = self._choose_adapted_machine_policy(recipe)
        if policy is None:
            return False
        requirements, inspection_errors = self._measurement_requirements_from_active_files(candidate, recipe)
        if requirements is not None:
            recipe.data_requirements = requirements
        if policy == "current":
            override = self._machine_override_from_config(candidate.loco_config)
            self._apply_machine_override_to_recipe(recipe, override)
            candidate.machine_configuration_override = deepcopy(override)
        else:
            candidate.machine_configuration_override = {}
        candidate.recipe_machine_policy = policy
        candidate.fit_recipe = recipe.to_dict()
        updated_errors = self._validate_retained_measurements(candidate)
        combined_errors = updated_errors or inspection_errors or errors
        candidate.completed_run = CompletedRunReference()
        candidate.fit_run_session = {}
        candidate.compatibility_validation = {
            "required": True,
            "status": "incompatible",
            "issues": list(combined_errors),
            "source_project": str(Path(source_filename).expanduser().resolve()),
            "measurement_signature": json_safe(self._measurement_binding_signature(candidate)),
        }
        # Detach the editable copy from the original filename while retaining
        # its base directory for portable lattice and configuration references.
        candidate.path = ""
        candidate.modified = True
        return True

    @staticmethod
    def _project_load_error_content(filename: str, exc: Exception) -> dict[str, str]:
        return {
            "title": "Unable to open FIT project",
            "summary": "The selected file could not be read as a FIT project.",
            "information": (
                "Check that this is a valid pyLOCO FIT project. The current project and measurement bindings "
                "remain unchanged."
            ),
            "details": f"Selected file: {filename}\nException type: {type(exc).__name__}\nException: {exc}",
        }

    def _show_project_load_error(self, filename: str, exc: Exception) -> None:
        content = self._project_load_error_content(filename, exc)
        dialog = QMessageBox(self)
        dialog.setWindowTitle(content["title"])
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(content["summary"])
        dialog.setInformativeText(content["information"])
        dialog.setDetailedText(content["details"])
        dialog.setStandardButtons(QMessageBox.Ok)
        for button in dialog.buttons():
            if "detail" in button.text().lower():
                button.setText("Show technical details…")
        dialog.exec()

    @Slot()
    def open_project(self, path: Path | None = None) -> None:
        filename = (
            str(path)
            if path
            else QFileDialog.getOpenFileName(
                self,
                "Open pyLOCO project",
                "",
                "pyLOCO Project (*.pyloco.json);;JSON (*.json)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )[0]
        )
        if not filename:
            return
        try:
            candidate = ProjectMetadata.load(filename)
        except Exception as exc:
            self._show_project_load_error(filename, exc)
            return
        if not self._confirm_open_with_unsaved_changes():
            self.statusBar().showMessage("Open project cancelled; current project unchanged.", 5000)
            return
        existing_recent = list(self.project.recent_projects)
        retained = bool(self.project.measurements)
        policy = self._choose_open_project_measurement_policy(candidate) if retained else "saved"
        if policy is None:
            self.statusBar().showMessage("Open project cancelled; current project unchanged.", 5000)
            return
        measurements_changed = False
        stale_run_binding = False
        opened_for_editing = False
        if policy == "keep":
            current_signature = self._measurement_binding_signature(self.project)
            saved_signature = self._measurement_binding_signature(candidate)
            candidate.measurements = deepcopy(self.project.measurements)
            candidate.measurement_session = deepcopy(self.project.measurement_session)
            measurements_changed = current_signature != saved_signature
            try:
                mapping_errors = self._reconcile_retained_measurement_mapping(candidate)
                errors = mapping_errors + self._validate_retained_measurements(candidate)
            except Exception as exc:
                self._show_project_load_error(filename, exc)
                return
            if errors:
                if not self._compatibility_errors_are_editable(errors):
                    self._show_project_open_validation_error(candidate, errors)
                    return
                if not self._choose_open_for_editing(candidate, errors):
                    self.statusBar().showMessage("Open project cancelled; current project unchanged.", 5000)
                    return
                if not self._prepare_incompatible_working_copy(candidate, errors, filename):
                    self.statusBar().showMessage("Open project cancelled; current project unchanged.", 5000)
                    return
                opened_for_editing = True
            stale_run_binding = self._fit_run_measurements_are_stale(candidate) if not opened_for_editing else True
            if (measurements_changed or stale_run_binding) and not opened_for_editing:
                candidate.completed_run = CompletedRunReference()
                candidate.fit_run_session = {}
                candidate.compatibility_validation = {
                    "required": True,
                    "status": "validation_required",
                    "issues": [
                        "Current measurements were retained; exact mapping and FIT compatibility "
                        "must be validated before running."
                    ],
                    "measurement_signature": json_safe(
                        self._measurement_binding_signature(candidate)
                    ),
                }
                candidate.modified = True
        previous_project = self.project
        previous_recipe = deepcopy(self.fit_recipe)
        previous_stage = self._active_fit_stage
        self.project = candidate
        for recent in reversed(existing_recent):
            self.project.add_recent_project(recent)
        self.project.add_recent_project(filename)
        try:
            self.dashboard_name.setText(self.project.name)
            self.dashboard_description.setText(self.project.description)
            self._load_config_to_widgets()
            self._load_project_fit_recipe()
        except Exception as exc:
            # Opening is transactional: a malformed persisted widget value
            # must not leave a new project model behind an unchanged/empty UI.
            self.project = previous_project
            self.fit_recipe = previous_recipe
            self._active_fit_stage = previous_stage
            self.dashboard_name.setText(self.project.name)
            self.dashboard_description.setText(self.project.description)
            try:
                self._load_config_to_widgets()
                self._refresh_fit_workflow_table(
                    previous_stage if 0 <= previous_stage < len(previous_recipe.stages) else None
                )
            except Exception:
                pass
            self._show_project_load_error(filename, exc)
            self.statusBar().showMessage(
                "Project could not be restored; current project remains unchanged.", 8000
            )
            return
        completed = self.project.completed_run
        fit_session = self.project.fit_run_session or {}
        session_path = fit_session.get("path")
        resolved_session = self.project.resolve_path(session_path) if session_path else None
        if resolved_session is not None and resolved_session.is_file():
            preferred = self.project.resolve_path(completed.results_dir) if completed.results_dir else None
            self.results_workspace.load_workflow_session(
                resolved_session, preferred_result=preferred
            )
            self._workspace.setCurrentWidget(self.results_page)
        elif completed.results_dir:
            results_dir = self.project.resolve_path(completed.results_dir)
            if results_dir.exists():
                self.results_workspace.load_results(
                    results_dir, runtime=completed.elapsed_seconds
                )
                self._workspace.setCurrentWidget(self.results_page)
        suffix = (
            " as an unsaved working copy; fitting disabled until compatibility validation passes"
            if opened_for_editing else
            " with current measurements; previous run/checkpoint bindings cleared"
            if policy == "keep" and (measurements_changed or stale_run_binding) else
            " with current measurements" if policy == "keep" else ""
        )
        elements = self.project.loco_config.machine_elements
        summary = (
            f"{len(self.fit_recipe.stages)} fitting stage(s) • {len(elements.bpm_ords)} BPMs • "
            f"{len(elements.horizontal_corrector_ords)} H/{len(elements.vertical_corrector_ords)} V correctors"
        )
        if policy == "keep":
            retained_detail = (
                " • Opened as an unsaved working copy; fitting disabled until compatibility validation passes"
                if opened_for_editing else
                " • Previous run/checkpoint bindings cleared"
                if measurements_changed or stale_run_binding else ""
            )
            status = (
                f"Project configuration loaded: {Path(filename).name} • Current measurements preserved"
                f"{retained_detail} • {summary}"
            )
        else:
            status = f"Opened {filename}{suffix} • {summary}"
        self._refresh_ui(status)
        if policy == "keep":
            # The operator opened this project specifically to reuse its
            # lattice and machine/FIT configuration with the active data.
            # Show the imported machine selections immediately instead of
            # leaving the GUI on the previous Measurements page.
            machine_index = next(
                (index for index in range(self._workspace.count())
                 if self._workspace.tabText(index) == "Machine Components"),
                -1,
            )
            if machine_index >= 0:
                self._workspace.setCurrentIndex(machine_index)
            self.statusBar().showMessage(status, 8000)

    def _launch_suite(self,application,*arguments):
        try:ok,detail=launch_suite_application(application,*arguments)
        except Exception as exc:QMessageBox.warning(self,"pyLOCO Suite launch failed",str(exc)); return False
        if not ok:QMessageBox.warning(self,"pyLOCO Suite launch failed",detail); return False
        self.statusBar().showMessage(f"Opening pyLOCO {application.title()} ({detail})…"); return True

    @Slot()
    def open_measure_app(self):self._launch_suite("measure")

    @Slot()
    def open_correct_app(self):
        results_dir=None; iteration=None
        if getattr(self.results_workspace,"loader",None) is not None:
            results_dir=Path(self.results_workspace.loader.result_dir)
            iteration=self.results_workspace.loader.iteration
        arguments=[]
        if results_dir is not None:
            arguments.extend(("--results",str(results_dir)))
            if iteration is not None:arguments.extend(("--iteration",str(iteration)))
        return self._launch_suite("correct",*arguments)

    @Slot()
    def open_measurement_session(self,path: Path|None=None):
        filename=str(path) if path else QFileDialog.getOpenFileName(self,"Open Measurement Session","","pyLOCO Measurement Session (*.pyloco-session.json *.json)")[0]
        if not filename:return False
        try:handoff=inspect_measurement_session(filename)
        except Exception as exc:QMessageBox.warning(self,"Cannot import Measurement Session",str(exc)); return False
        if self.project.measurements:
            answer=QMessageBox.question(self,"Import Measure session?","Keep the current FIT recipe and replace its measurement bindings with this session?\n\nFitting stages, quadrupole selections, weights, SVD and optimizer settings remain unchanged. Cancel makes no changes.",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)
            if answer!=QMessageBox.Yes:return False
        self.project.measurements={role:ImportedDataset(role,str(source),source.suffix.lower().lstrip("."),source.stat().st_size,deepcopy(handoff.options[role])) for role,source in handoff.files.items()}
        self.project.measurement_session=deepcopy(handoff.provenance); self.project.measurement_session["manifest"]=str(handoff.manifest)
        self.project.measurement_device_mapping={}
        self._mark_compatibility_stale("Measurement session changed; validation is required.")
        orm=handoff.options.get("orm",{}); dispersion=handoff.options.get("dispersion",{})
        if orm:
            self.project.loco_config.response_matrix.bidirectional=bool(orm.get("bidirectional",False))
            for key,plane in (("requested_kick_h_rad","horizontal"),("requested_kick_v_rad","vertical")):
                values=np.asarray(orm.get(key,()),dtype=float)
                if values.size and np.allclose(values,values[0]):
                    value=float(abs(values[0])); setattr(self.project.loco_config.response_matrix,"dkick_h" if plane=="horizontal" else "dkick_v",value); setattr(self.project.loco_config.parameters.cmstep,plane,value)
            self.project.loco_config.element_selection_state["measurement_session_names"]={"bpms":orm.get("bpm_names",[]),"horizontal_correctors":orm.get("horizontal_corrector_names",[]),"vertical_correctors":orm.get("vertical_corrector_names",[])}
        if dispersion:
            step=float(dispersion["rf_step_hz"]); self.project.loco_config.response_matrix.rfStep=step; self.project.loco_config.parameters.rfStep=step; self.project.loco_config.fixed_parameters.rfstep=step; self.project.loco_config.response_matrix.includeDispersion=True; self.project.loco_config.rejection.includeDispersion=True
        full=(handoff.provenance.get("session_metadata") or {}).get("full_measure_session")
        mapping_ready=False
        if isinstance(full,dict):
            lattice_info=full.get("reference_lattice") or {}; relative=lattice_info.get("path")
            if relative:
                lattice_path=(handoff.manifest.parent/str(relative)).resolve()
                expected=str(lattice_info.get("sha256", ""))
                if not lattice_path.is_file() or (expected and file_sha256(lattice_path)!=expected):
                    QMessageBox.warning(self,"Full Measure session lattice unavailable","The saved reference lattice is missing or its checksum changed. Measurements were imported, but lattice mapping was not reused.")
                else:
                    replace_lattice=True
                    if self.project.lattice.path:
                        current=self.project.resolve_path(self.project.lattice.path)
                        try:same=current.is_file() and file_sha256(current)==expected
                        except OSError:same=False
                        if not same:
                            replace_lattice=QMessageBox.question(self,"Reference lattice differs","The imported Measure session uses a different reference lattice. Replace the current FIT lattice while preserving the FIT recipe?\n\nCancel leaves the current FIT project unchanged except for the already imported measurements.",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)==QMessageBox.Yes
                    if replace_lattice:
                        self.project.lattice=LatticeSelection(path=str(lattice_path),file_type=lattice_path.suffix.lower().lstrip("."),element_count=lattice_info.get("element_count"),mat_variable=str(lattice_info.get("mat_variable", "")))
                        saved_mapping=full.get("mapping") or {}
                        if saved_mapping and orm:
                            inventory=measurement_inventory({"files":{"orm":orm}}); configuration=None
                            config_info=full.get("pyaml_configuration") or {}; config_value=config_info.get("path") if config_info.get("included") else config_info.get("external_path")
                            if config_value:
                                candidate=(handoff.manifest.parent/str(config_value)).resolve() if not Path(str(config_value)).is_absolute() else Path(str(config_value)).expanduser().resolve()
                                if candidate.is_file() and (not config_info.get("sha256") or file_sha256(candidate)==config_info.get("sha256")):configuration=candidate
                            requires_config=bool((saved_mapping.get("machine_profile") or {}).get("configuration_sha256"))
                            if not requires_config or configuration is not None:
                                try:
                                    ring=load_project_lattice(lattice_path,self.project.lattice.mat_variable or None)
                                    self.project.measurement_device_mapping=validate_measurement_mapping(saved_mapping,ring,inventory,lattice_path=lattice_path,configuration_path=configuration,require_provenance=True)
                                    mapping_ready=measurement_mapping_counts(self.project.measurement_device_mapping)["mapped"]==measurement_mapping_counts(self.project.measurement_device_mapping)["total"]
                                except Exception as exc:
                                    self.project.measurement_device_mapping={}; self._append_run_log(f"Saved Measure mapping requires revalidation: {exc}")
                            else:
                                self.project.measurement_device_mapping={}; self._append_run_log("Saved Measure mapping requires the checksummed pyAML configuration before it can be reused.")
        self.project.modified=True; self._load_config_to_widgets(); self._refresh_ui(f"Imported Measurement Session {handoff.session_id}")
        if mapping_ready:self.use_measurement_mappings_for_fit()
        available="\n".join(f"✓ {role.replace('_',' ').title()}" for role in handoff.available_roles) or "—"
        missing="\n".join(f"✗ {role.replace('_',' ').title()}" for role in handoff.missing_roles) or "None"
        message=f"Measurement Session: {handoff.session_id}\n\nAvailable:\n{available}\n\nMissing:\n{missing}"
        if "orm" in handoff.missing_roles:message+="\n\nA LOCO fit cannot run until an ORM is supplied. Existing measurements were imported without fabrication."
        self.statusBar().showMessage(f"Measurement session imported • {handoff.session_id}")
        if "orm" in handoff.missing_roles:self._append_run_log(message)
        return True

    def _snapshot_project_from_widgets(self) -> None:
        """Make the project model an exact snapshot of persistable GUI state."""
        self.project.name = (
            self.dashboard_name.text().strip() or "Untitled LOCO Project"
        )
        self.project.description = self.dashboard_description.text().strip()
        self.project.loco_config = self._collect_loco_configuration()
        self._store_active_fit_stage()
        self.project.fit_recipe = self.fit_recipe.to_dict()
        self.project.fit_recipe = self.fit_recipe.to_dict()

    def _load_project_fit_recipe(self) -> None:
        raw = self.project.fit_recipe
        explicit_empty = bool(raw) and "stages" in raw and not raw.get("stages")
        if raw:
            try:
                data = deepcopy(raw); data["stages"] = [FitStage(**item) for item in data.get("stages", [])]
                self.fit_recipe = FitRecipe(**data)
            except Exception:
                self.fit_recipe = FitRecipe()
        else:
            self.fit_recipe = FitRecipe()
        if (self.fit_recipe.stages and self.project.recipe_machine_policy == "current"
                and self.project.machine_configuration_override):
            self._apply_machine_override_to_recipe(self.fit_recipe, self.project.machine_configuration_override)
        if not self.fit_recipe.stages:
            self._active_fit_stage = -1
            if explicit_empty:
                self._refresh_fit_workflow_table()
            else:
                self._add_fit_stage(name="Stage 1", start_from="original_model", mark_modified=False)
        else:
            self._active_fit_stage = -1; self._refresh_fit_workflow_table(0); self._select_fit_stage()

    @Slot()
    def save_project(self) -> bool:
        if not self.project.path:
            return self.save_project_as()
        return self._save_project_to(self.project.path)

    def _save_project_to(self, filename: str | Path) -> bool:
        try:
            self._snapshot_project_from_widgets()
            saved_path = self.project.save(filename)
        except Exception as exc:
            message = f"Project could not be saved: {exc}"
            self._last_save_status = message
            QMessageBox.warning(self, "Project could not be saved", message)
            self.statusBar().showMessage(message)
            return False
        incomplete = bool(self.project.validation_messages())
        message = (
            f"Project saved — FIT configuration incomplete: {saved_path}"
            if incomplete else f"Project saved successfully: {saved_path}"
        )
        self._last_save_status = message
        self._refresh_ui(message)
        return True

    @Slot()
    def save_project_as(self) -> bool:
        filename = QFileDialog.getSaveFileName(
            self,
            "Save pyLOCO project",
            self.project.path or f"{self.project.name}.pyloco.json",
            "pyLOCO Project (*.pyloco.json)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename:
            self._last_save_status = "Project not saved — no filename selected."
            self.statusBar().showMessage(self._last_save_status)
            return False
        return self._save_project_to(filename)


    def _load_current_lattice(self):
        """Load the currently selected AT lattice for element detection/preview."""

        if not self.project.lattice.path:
            return None
        try:
            path = self.project.resolve_path(self.project.lattice.path)
            ring = load_project_lattice(path, self.project.lattice.mat_variable or None)
            if path.suffix.lower() == ".mat" and not self.project.lattice.mat_variable:
                candidates = mat_lattice_candidates(path)
                if len(candidates) == 1:
                    self.project.lattice.mat_variable = candidates[0].variable
                    self.project.lattice.element_count = candidates[0].element_count
            return ring
        except Exception:
            return None

    def _element_preview_rows(self, ords: list[int], lattice=None) -> list[tuple[int, int, str, str]]:
        if lattice is None:
            lattice = self._load_current_lattice()
        rows = []
        for position, ordinal in enumerate(ords):
            elem = lattice[ordinal] if lattice and 0 <= ordinal < len(lattice) else None
            name = ElementSelectionDialog._element_name(None, elem) if elem else ""
            cls = type(elem).__name__ if elem else ""
            rows.append((position, ordinal, name, cls))
        return rows

    def _refresh_element_selection_ui(self) -> None:
        if not hasattr(self, "element_count_labels"):
            return
        elements = self.project.loco_config.machine_elements
        advanced = self.project.mode == "Advanced"
        signature = (
            self.project.lattice.path,
            self.project.lattice.mat_variable,
            advanced,
            tuple((key, tuple(getattr(elements, key))) for key in ELEMENT_ROLES),
        )
        if getattr(self, "_element_preview_signature", None) == signature:
            return
        self._element_preview_signature = signature
        # Loading a large AT lattice is comparatively expensive.  Reuse one
        # load for every element preview instead of loading it independently
        # for BPMs, correctors, cavities, normal quads, skews, and tilts.
        lattice = self._load_current_lattice() if advanced else None
        for key in ELEMENT_ROLES:
            values = list(getattr(elements, key))
            self.element_count_labels[key].setText(f"{len(values)} selected")
            table = self.element_preview_tables[key]
            table.setVisible(advanced)
            rows = self._element_preview_rows(values, lattice) if advanced else []
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(str(value)))
        self._update_exclusion_counts()

    def _refresh_reference_model_info(self) -> None:
        if not hasattr(self,"reference_model_info"):return
        path=self.project.lattice.path
        signature=(path,self.project.lattice.mat_variable)
        if getattr(self,"_reference_model_info_path",None)==signature:return
        self._reference_model_info_path=signature
        if not path:self.reference_model_info.setText("Not available");return
        try:
            from pyLOCO.reference_model import load_reference_model
            model=load_reference_model(
                self.project.resolve_path(path), source="FIT reference lattice",
                mat_key=self.project.lattice.mat_variable or None,
            )
            harmonic=str(model.harmonic_number) if model.harmonic_number is not None else "Not available"
            rf=f"{model.nominal_rf_hz/1e6:.9f} MHz" if model.nominal_rf_hz is not None else "Not available"
            residual=f"{model.rf_harmonic_residual_hz:.6g} Hz" if model.rf_harmonic_residual_hz is not None else "Not available"
            rows=(
                ("Beam energy",f"{model.energy_ev/1e9:.6g} GeV","Reference lattice"),
                ("Number of lattice elements",str(model.element_count),"Loaded AT lattice"),
                ("Circumference",f"{model.circumference_m:.6f} m","AT lattice geometry"),
                ("Revolution frequency f<sub>rev</sub>",f"{model.revolution_frequency_hz:.6f} Hz","c / circumference"),
                ("Harmonic number h",harmonic,model.harmonic_number_source),
                ("RF frequency f<sub>RF</sub>",rf,model.rf_source),
                ("RF–revolution frequency consistency [Hz]",residual,"f<sub>RF</sub> − h·f<sub>rev</sub>; reference-lattice consistency, not a measured accelerator RF error"),
                ("Fractional tunes Qx / Qy",f"{model.tune[0]:.6f} / {model.tune[1]:.6f}","AT linear optics"),
                ("Chromaticity ξx / ξy",f"{model.chromaticity[0]:.6f} / {model.chromaticity[1]:.6f}","AT linear optics"),
                ("Momentum compaction α<sub>c</sub>",f"{model.momentum_compaction:.9g}","AT model"),
                ("Relativistic term 1/γ²",f"{model.inverse_gamma_squared:.9g}","Lattice energy"),
                ("Slip factor η (pyAT convention)",f"{model.at_slip_factor:.9g}","1/γ² − α<sub>c</sub>; ring.get_slip_factor()"),
            )
            body="".join(f"<tr><td><b>{name}</b></td><td>{value}</td><td style='color:#9CA3AF'>{source}</td></tr>" for name,value,source in rows)
            self.reference_model_info.setText(
                "<h3 style='color:#A98BFF;margin:0 0 8px 0'>Reference model</h3>"
                "<div style='margin-bottom:8px'>Calculated from the loaded lattice with Accelerator Toolbox (AT).</div>"
                f"<table cellspacing='0' cellpadding='5' width='100%'><tr><th align='left'>Quantity</th><th align='left'>Value</th><th align='left'>Source</th></tr>{body}</table>"
            )
            self.reference_model_info.setToolTip(
                "RF–revolution frequency consistency is the reference cavity RF frequency minus "
                "the harmonic-number multiple of c divided by the loaded lattice circumference. "
                "Near zero indicates internally consistent reference-lattice RF settings; it is "
                "not a measured accelerator RF error.\n\n"
                f"Advanced RF calculation: η_RF = αc − 1/γ² = −η_pyAT = {model.rf_momentum_coefficient:.9g}. "
                "This coefficient remains available to the signed RF-to-momentum conversion; it is not a second slip-factor result.\n\n"
                f"Lattice details\nFile: {model.path.name}\nSHA-256: {model.checksum_sha256}\n"
                f"Source / provenance: {model.source}"
            )
            if hasattr(self, "mcf_automatic_summary"):
                self._update_mcf_source_visibility()
        except Exception as exc:self.reference_model_info.setText(f"Not available — {exc}")

    @staticmethod
    def _parse_position_text(text: str) -> list[int]:
        values = [int(value) for value in re.findall(r"\d+", text)]
        if len(values) != len(set(values)):
            raise ValueError("Exclusion positions must be unique.")
        return values

    def _update_exclusion_counts(self) -> None:
        if not hasattr(self, "exclusion_counts"):
            return
        cfg = self.project.loco_config
        total = len(cfg.machine_elements.bpm_ords); excluded = len(cfg.bad_bpm_positions)
        self.exclusion_counts.setText(
            f"BPMs: {total} selected, {excluded} excluded, {max(0, total-excluded)} retained; "
            f"H correctors: {len(cfg.machine_elements.horizontal_corrector_ords)-len(cfg.excluded_horizontal_corrector_positions)} retained; "
            f"V correctors: {len(cfg.machine_elements.vertical_corrector_ords)-len(cfg.excluded_vertical_corrector_positions)} retained"
        )

    @Slot()
    def _store_exclusions(self) -> None:
        try:
            cfg = self.project.loco_config
            cfg.bad_bpm_positions = self._parse_position_text(self.bad_bpm_positions_edit.text())
            cfg.excluded_horizontal_corrector_positions = self._parse_position_text(self.hcor_exclusions_edit.text())
            cfg.excluded_vertical_corrector_positions = self._parse_position_text(self.vcor_exclusions_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid exclusion", str(exc)); return
        self._refresh_recipe_machine_override()
        self._store_active_fit_stage()
        self._record_fit_stage_edit("Measurement exclusions changed; validation is required.")
        self._update_exclusion_counts()
        self._refresh_fit_workflow_table(self._active_fit_stage)
        self._refresh_ui("Measurement exclusions updated; validate before running")

    @Slot()
    def edit_element_selection(self, role_key: str) -> None:
        elements = self.project.loco_config.machine_elements
        current = list(getattr(elements, role_key))
        selection_state = self.project.loco_config.element_selection_state.get(role_key)
        dialog = ElementSelectionDialog(self, role_key, current, selection_state)
        if dialog.exec() != QDialog.Accepted:
            return
        setattr(elements, role_key, dialog.selected_ords)
        if dialog.selection_state:
            self.project.loco_config.element_selection_state[role_key] = dialog.selection_state
        else:
            self.project.loco_config.element_selection_state.pop(role_key, None)
        group_key={"normal_quadrupole_ords":"normal_quadrupole_groups","skew_quadrupole_ords":"skew_quadrupole_groups","quadrupole_tilt_ords":"quadrupole_tilt_groups"}.get(role_key)
        if group_key:setattr(elements,group_key,[])
        self.project.loco_config._sync_response_matrix_elements()
        self._refresh_recipe_machine_override()
        self._mark_compatibility_stale("Machine-element selections changed; validation is required.")
        self.project.modified = True
        self._refresh_element_selection_ui()
        self._refresh_measurement_mapping_table()
        self._refresh_measurement_mapping_feedback()
        self._update_fit_summary()
        self._refresh_ui(f"Updated {ELEMENT_ROLES[role_key][0]} selection")

    @Slot()
    def edit_family_groups(self, role_key: str) -> None:
        elements=self.project.loco_config.machine_elements
        group_key={"normal_quadrupole_ords":"normal_quadrupole_groups","skew_quadrupole_ords":"skew_quadrupole_groups","quadrupole_tilt_ords":"quadrupole_tilt_groups"}[role_key]
        dialog=FamilyGroupDialog(self,role_key,list(getattr(elements,role_key)),list(getattr(elements,group_key)))
        if dialog.exec()!=QDialog.Accepted:return
        setattr(elements,group_key,dialog.groups)
        self._refresh_recipe_machine_override()
        self._store_active_fit_stage()
        self._record_fit_stage_edit("Machine-element groups changed; validation is required.")
        self._refresh_fit_workflow_table(self._active_fit_stage)
        self._update_fit_summary()
        self._refresh_ui(f"Updated {ELEMENT_ROLES[role_key][0]} family groups")

    @staticmethod
    def _element_identity(element) -> tuple[str, str]:
        name = str(
            getattr(element, "CommonName", None)
            or getattr(element, "FamName", None)
            or getattr(element, "Name", None)
            or getattr(element, "name", "")
        )
        return name, type(element).__name__

    def _machine_configuration_snapshot(self, config: LocoConfiguration | None = None) -> dict:
        config = config or self._collect_loco_configuration()
        lattice_path = self.project.resolve_path(self.project.lattice.path)
        if not lattice_path.is_file():
            raise ValueError("Select an existing reference lattice before saving a machine configuration.")
        lattice = self._load_current_lattice()
        if lattice is None:
            raise ValueError("The selected reference lattice could not be loaded.")
        elements = __import__("dataclasses").asdict(config.machine_elements)
        identities = {}
        for role_key in ELEMENT_ROLES:
            values = list(getattr(config.machine_elements, role_key))
            identities[role_key] = [
                {"ordinal": ordinal, "name": self._element_identity(lattice[ordinal])[0],
                 "element_class": self._element_identity(lattice[ordinal])[1]}
                for ordinal in values
            ]
        return {
            "schema": MACHINE_CONFIGURATION_SCHEMA,
            "reference_lattice": {
                "filename": lattice_path.name,
                "sha256": file_sha256(lattice_path),
                "element_count": len(lattice),
                "mat_variable": self.project.lattice.mat_variable,
            },
            "machine_elements": elements,
            "element_selection_state": deepcopy(config.element_selection_state),
            "selected_element_identities": identities,
            "measurement_exclusions": {
                "bpm_positions": list(config.bad_bpm_positions),
                "horizontal_corrector_positions": list(config.excluded_horizontal_corrector_positions),
                "vertical_corrector_positions": list(config.excluded_vertical_corrector_positions),
            },
        }

    @Slot()
    def save_machine_configuration(self) -> bool:
        filename = QFileDialog.getSaveFileName(
            self, "Save machine configuration", "machine-configuration.pyloco-machine.json",
            "pyLOCO machine configuration (*.pyloco-machine.json *.json)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename:
            return False
        try:
            payload = self._machine_configuration_snapshot()
            target = Path(filename).expanduser()
            if not str(target).endswith(".json"):
                target = target.with_suffix(".pyloco-machine.json")
            target.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Machine configuration could not be saved", str(exc))
            return False
        self.statusBar().showMessage(f"Saved machine configuration: {target.name}", 5000)
        return True

    def _validated_machine_configuration(self, filename: str | Path) -> tuple[MachineElementsConfig, dict, dict]:
        source = Path(filename).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != MACHINE_CONFIGURATION_SCHEMA:
            raise ValueError(f"Unsupported machine-configuration schema: {payload.get('schema')!r}")
        lattice_path = self.project.resolve_path(self.project.lattice.path)
        if not lattice_path.is_file():
            raise ValueError("Select the reference lattice before loading a machine configuration.")
        lattice = self._load_current_lattice()
        reference = payload.get("reference_lattice") or {}
        if int(reference.get("element_count", -1)) != len(lattice):
            raise ValueError(
                f"Reference-lattice element count differs: configuration has {reference.get('element_count')}, "
                f"loaded lattice has {len(lattice)}."
            )
        if str(reference.get("sha256", "")) != file_sha256(lattice_path):
            raise ValueError("Reference-lattice checksum does not match the loaded lattice.")
        recorded_variable = str(reference.get("mat_variable") or "")
        if recorded_variable and recorded_variable != self.project.lattice.mat_variable:
            raise ValueError(
                f"Machine configuration uses MAT variable {recorded_variable!r}, but the active project uses "
                f"{self.project.lattice.mat_variable!r}."
            )
        try:
            elements = MachineElementsConfig(**dict(payload["machine_elements"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid machine-elements payload: {exc}") from exc
        selection_state = deepcopy(payload.get("element_selection_state") or {})
        recorded = payload.get("selected_element_identities") or {}
        for role_key in ELEMENT_ROLES:
            values = list(getattr(elements, role_key))
            dialog = ElementSelectionDialog(self, role_key, values, selection_state.get(role_key))
            try:
                dialog._validate_array(values)
            finally:
                dialog.close()
            identities = recorded.get(role_key)
            if identities is None or len(identities) != len(values):
                raise ValueError(f"Stored identities are incomplete for {ELEMENT_ROLES[role_key][0]}.")
            for expected, ordinal in zip(identities, values):
                actual_name, actual_class = self._element_identity(lattice[ordinal])
                if int(expected.get("ordinal", -1)) != ordinal or expected.get("name") != actual_name or expected.get("element_class") != actual_class:
                    raise ValueError(
                        f"Stored identity for {ELEMENT_ROLES[role_key][0]} ordinal {ordinal} does not match "
                        "the loaded lattice."
                    )
        for selected_key, group_key in (
            ("normal_quadrupole_ords", "normal_quadrupole_groups"),
            ("skew_quadrupole_ords", "skew_quadrupole_groups"),
            ("quadrupole_tilt_ords", "quadrupole_tilt_groups"),
        ):
            selected = set(getattr(elements, selected_key))
            for group in getattr(elements, group_key):
                if not group or not set(group).issubset(selected):
                    raise ValueError(f"A stored {group_key.replace('_', ' ')} entry is empty or references an unselected element.")
        exclusions = dict(payload.get("measurement_exclusions") or {})
        normalized = {
            "bpm_positions": [int(value) for value in exclusions.get("bpm_positions", [])],
            "horizontal_corrector_positions": [int(value) for value in exclusions.get("horizontal_corrector_positions", [])],
            "vertical_corrector_positions": [int(value) for value in exclusions.get("vertical_corrector_positions", [])],
        }
        for key, maximum in (
            ("bpm_positions", len(elements.bpm_ords)),
            ("horizontal_corrector_positions", len(elements.horizontal_corrector_ords)),
            ("vertical_corrector_positions", len(elements.vertical_corrector_ords)),
        ):
            values = normalized[key]
            if len(values) != len(set(values)) or any(value < 0 or value >= maximum for value in values):
                raise ValueError(f"Stored {key.replace('_', ' ')} are duplicated or outside the selected-list range.")
        return elements, selection_state, normalized

    @Slot()
    def load_machine_configuration(self) -> bool:
        filename = QFileDialog.getOpenFileName(
            self, "Load machine configuration", "",
            "pyLOCO machine configuration (*.pyloco-machine.json *.json)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename:
            return False
        try:
            elements, selection_state, exclusions = self._validated_machine_configuration(filename)
        except Exception as exc:
            QMessageBox.warning(self, "Machine configuration is incompatible", str(exc))
            return False
        counts = [f"{ELEMENT_ROLES[key][0]}: {len(getattr(elements, key))}" for key in ELEMENT_ROLES]
        preview = (
            "The following lattice-validated selections will replace the current machine selections:\n\n"
            + "\n".join(counts)
            + f"\n\nExcluded selected-list positions: {len(exclusions['bpm_positions'])} BPM, "
              f"{len(exclusions['horizontal_corrector_positions'])} H corrector, "
              f"{len(exclusions['vertical_corrector_positions'])} V corrector."
            + "\n\nMeasurements, FIT settings, stages and output paths will remain unchanged."
        )
        if QMessageBox.question(
            self, "Load machine configuration?", preview,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return False
        self.project.loco_config.machine_elements = elements
        self.project.loco_config.element_selection_state = selection_state
        self.project.loco_config.bad_bpm_positions = exclusions["bpm_positions"]
        self.project.loco_config.excluded_horizontal_corrector_positions = exclusions["horizontal_corrector_positions"]
        self.project.loco_config.excluded_vertical_corrector_positions = exclusions["vertical_corrector_positions"]
        self.project.loco_config._sync_response_matrix_elements()
        self._refresh_recipe_machine_override()
        self._mark_compatibility_stale("Machine configuration changed; validation is required.")
        self.project.modified = True
        self._load_config_to_widgets()
        self._refresh_ui(f"Loaded machine configuration {Path(filename).name}")
        return True

    def _measurement_mapping_context(self):
        lattice_path = self.project.resolve_path(self.project.lattice.path) if self.project.lattice.path else Path()
        if not lattice_path.is_file():
            raise ValueError("Select an existing reference lattice first.")
        lattice = self._load_current_lattice()
        if lattice is None:
            raise ValueError("The selected reference lattice could not be loaded.")
        paths = {role: str(self.project.resolve_path(dataset.path)) for role, dataset in self.project.measurements.items()}
        options = {role: deepcopy(dataset.options) for role, dataset in self.project.measurements.items()}
        payload = inspect_measurement_files(paths, options)
        if payload["errors"]:
            raise ValueError("\n".join(payload["errors"]))
        inventory = measurement_inventory(payload)
        if not all(inventory.values()):
            raise ValueError("The current ORM does not contain ordered BPM and H/V-corrector identities.")
        orm_options = options.get("orm", {})
        metadata = orm_options.get("acquisition_metadata") or {}
        pyaml_options = metadata.get("pyaml_options") or {}
        raw_configuration = pyaml_options.get("config_path") or ""
        configuration = Path(raw_configuration).expanduser() if raw_configuration else None
        if configuration is not None and not configuration.is_absolute():
            configuration = self.project.resolve_path(configuration)
        if configuration is not None and not configuration.is_file():
            configuration = None
        return lattice, lattice_path, payload, inventory, configuration

    @Slot()
    def find_measurement_mappings(self) -> bool:
        try:
            lattice, lattice_path, _payload, inventory, configuration = self._measurement_mapping_context()
            saved_mapping = self.project.measurement_device_mapping or {}
            if saved_mapping.get("reference_lattice") and saved_mapping.get("inventory"):
                try:
                    mapping = validate_measurement_mapping(
                        saved_mapping, lattice, inventory, lattice_path=lattice_path,
                        configuration_path=configuration, require_provenance=True,
                    )
                    mapping["mapping_provenance"] = "saved project mapping revalidated against current inputs"
                except ValueError:
                    mapping = find_exact_device_mappings(
                        lattice, inventory, lattice_path=lattice_path, configuration_path=configuration,
                        provenance="exact AT Device/FamName identity from selected lattice and Measure inventory",
                    )
            else:
                mapping = find_exact_device_mappings(
                    lattice, inventory, lattice_path=lattice_path, configuration_path=configuration,
                    provenance="exact AT Device/FamName identity from selected lattice and Measure inventory",
                )
        except Exception as exc:
            self.measurement_mapping_feedback.setText(f"Mapping could not be completed — {exc}")
            return False
        self.project.measurement_device_mapping = mapping
        self.project.modified = True
        self.project.compatibility_validation = {
            "required": True, "status": "validation_required",
            "issues": ["Measurement-device mapping changed; validation is required."],
        }
        self._select_measurement_mapping_filter_after_operation(mapping)
        self._refresh_measurement_mapping_table(); self._refresh_measurement_mapping_feedback()
        self._refresh_ui("Measurement-device mapping changed; validate before running FIT")
        return measurement_mapping_counts(mapping)["mapped"] == measurement_mapping_counts(mapping)["total"]

    def _validated_mapping_fit_proposal(self) -> dict:
        """Return an ordered, revalidated BPM/H/V selection proposal.

        The order is the Measure HDF5 inventory order.  No ordinal sorting or
        position-based identity inference is performed here.
        """
        lattice, lattice_path, payload, inventory, configuration = self._measurement_mapping_context()
        mapping = self.project.measurement_device_mapping or {}
        counts = measurement_mapping_counts(mapping)
        if not counts["total"] or counts["mapped"] != counts["total"]:
            raise ValueError("Every measured BPM and H/V corrector must be mapped and validated first.")
        mapping = validate_measurement_mapping(
            mapping, lattice, inventory, lattice_path=lattice_path,
            configuration_path=configuration, require_provenance=True,
        )
        roles = {
            "bpm_ords": "bpm",
            "horizontal_corrector_ords": "horizontal_corrector",
            "vertical_corrector_ords": "vertical_corrector",
        }
        proposed = {}
        for field, role in roles.items():
            values = measurement_mapping_ordered_ordinals(mapping, role)
            if values is None:
                raise ValueError(f"The {role.replace('_', ' ')} mapping is incomplete or invalid.")
            proposed[field] = list(values)
        expected_shape = (2 * len(proposed["bpm_ords"]),
                          len(proposed["horizontal_corrector_ords"]) + len(proposed["vertical_corrector_ords"]))
        measured_shape = tuple((payload.get("files", {}).get("orm", {}) or {}).get("shape", ()))
        if measured_shape and measured_shape != expected_shape:
            raise ValueError(
                f"The measured ORM shape {measured_shape} does not match the mapped inventory; "
                f"expected {expected_shape}."
            )
        current = self.project.loco_config.machine_elements
        current_ords = {field: list(getattr(current, field)) for field in roles}
        names_state = self.project.loco_config.element_selection_state.get("measurement_session_names", {})
        inventory_matches = (
            list(names_state.get("bpms", ())) == list(inventory["bpm"])
            and list(names_state.get("horizontal_correctors", ())) == list(inventory["horizontal_corrector"])
            and list(names_state.get("vertical_correctors", ())) == list(inventory["vertical_corrector"])
        )
        same_ordered_selection = all(current_ords[field] == proposed[field] for field in roles)
        exclusions = {
            "bpm": list(self.project.loco_config.bad_bpm_positions),
            "horizontal_corrector": list(self.project.loco_config.excluded_horizontal_corrector_positions),
            "vertical_corrector": list(self.project.loco_config.excluded_vertical_corrector_positions),
        }
        exclusions_present = any(exclusions.values())
        preserve_exclusions = not exclusions_present or inventory_matches or same_ordered_selection
        stage_selections = []
        stage_exclusions = []
        for stage in self.fit_recipe.stages:
            config = self._configuration_for_fit_stage(stage)
            stage_selections.append({field: list(getattr(config.machine_elements, field)) for field in roles})
            stage_exclusions.append((
                tuple(config.bad_bpm_positions),
                tuple(config.excluded_horizontal_corrector_positions),
                tuple(config.excluded_vertical_corrector_positions),
            ))
        distinct_stage_selections = len({
            (tuple(item["bpm_ords"]), tuple(item["horizontal_corrector_ords"]),
             tuple(item["vertical_corrector_ords"])) for item in stage_selections
        })
        return {
            "mapping": mapping, "inventory": inventory, "current": current_ords, "proposed": proposed,
            "expected_shape": expected_shape, "measured_shape": measured_shape or expected_shape,
            "exclusions": exclusions, "preserve_exclusions": preserve_exclusions,
            "stage_count": len(self.fit_recipe.stages), "distinct_stage_selections": distinct_stage_selections,
            "distinct_stage_exclusions": len(set(stage_exclusions)),
        }

    def _confirm_mapping_fit_proposal(self, proposal: dict) -> bool:
        labels = (("BPM", "bpm_ords"), ("H corrector", "horizontal_corrector_ords"),
                  ("V corrector", "vertical_corrector_ords"))
        comparisons = "\n".join(
            f"{label}: {len(proposal['current'][field])} → {len(proposal['proposed'][field])}"
            for label, field in labels
        )
        stage_text = (
            f"All {proposal['stage_count']} FIT stages will receive the same measured-device selections."
            if proposal["stage_count"] != 1 else "The FIT stage will receive the measured-device selections."
        )
        if proposal["distinct_stage_selections"] > 1:
            stage_text += (
                " The stages currently use different BPM/corrector selections; applying this proposal will "
                "replace those differences. Review this explicitly before continuing."
            )
        if proposal["distinct_stage_exclusions"] > 1:
            stage_text += (
                " The stages also contain different BPM/corrector exclusions; applying this proposal will "
                "replace those differences with the reviewed exclusions shown here."
            )
        exclusion_text = (
            "Existing exclusions remain valid for the same ordered inventory and will be preserved."
            if proposal["preserve_exclusions"] else
            "Existing exclusions are tied to a different or unverified ordered inventory and will be reset."
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Use validated mappings in FIT?")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText("Preview the measured-device selections before applying them.")
        dialog.setInformativeText(
            f"{comparisons}\n\nMeasured ORM: {proposal['measured_shape'][0]} × {proposal['measured_shape'][1]}\n"
            f"Proposed model ORM: {proposal['expected_shape'][0]} × {proposal['expected_shape'][1]}\n\n"
            f"{stage_text}\n\n{exclusion_text}\n\n"
            "Normal/skew quadrupole and tilt selections remain unchanged; legacy magnet ordinals may still need "
            "revalidation after a lattice-version change.\n\n"
            "Measurement arrays and the original saved project remain unchanged. The adapted project will require validation."
        )
        details = []
        for label, field in labels:
            details.append(f"{label} current ordinals:\n{proposal['current'][field]}\n")
            details.append(f"{label} proposed ordinals in measurement order:\n{proposal['proposed'][field]}\n")
        if any(proposal["exclusions"].values()):
            details.append(f"Current selected-list exclusions:\n{proposal['exclusions']}\n")
        dialog.setDetailedText("\n".join(details))
        apply_label = (
            f"Apply to all {proposal['stage_count']} stages"
            if proposal["stage_count"] > 1 else "Apply mapped selections"
        )
        apply_button = dialog.addButton(apply_label, QMessageBox.AcceptRole)
        cancel_button = dialog.addButton("Cancel — make no changes", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is apply_button

    @Slot()
    def use_measurement_mappings_for_fit(self) -> bool:
        try:
            proposal = self._validated_mapping_fit_proposal()
        except Exception as exc:
            QMessageBox.warning(self, "Mapped devices cannot be used", str(exc))
            self._refresh_measurement_mapping_feedback()
            return False
        if not self._confirm_mapping_fit_proposal(proposal):
            return False

        # Build the complete replacement before mutating live state.
        config = deepcopy(self.project.loco_config)
        elements = deepcopy(config.machine_elements)
        for field, values in proposal["proposed"].items():
            setattr(elements, field, list(values))
        config.machine_elements = elements
        mapping = proposal["mapping"]
        reference = mapping.get("reference_lattice") or {}
        inventory_hash = (mapping.get("inventory") or {}).get("sha256", "")
        role_names = {
            "bpm_ords": "bpm", "horizontal_corrector_ords": "horizontal_corrector",
            "vertical_corrector_ords": "vertical_corrector",
        }
        for field, role in role_names.items():
            values = list(proposal["proposed"][field])
            config.element_selection_state[field] = {
                "version": 1, "method": "validated_measurement_mapping",
                "mapping_schema": mapping.get("schema", ""),
                "mapping_provenance": mapping.get("mapping_provenance", ""),
                "inventory_sha256": inventory_hash,
                "lattice_sha256": reference.get("sha256", ""),
                "measured_role": role,
                "measured_names": list(proposal["inventory"][role]),
                "generated_ords": values, "resolved_ords": values,
                "manual_additions": [], "manual_removals": [],
            }
        config.element_selection_state["measurement_session_names"] = {
            "bpms": list(proposal["inventory"]["bpm"]),
            "horizontal_correctors": list(proposal["inventory"]["horizontal_corrector"]),
            "vertical_correctors": list(proposal["inventory"]["vertical_corrector"]),
        }
        if not proposal["preserve_exclusions"]:
            config.bad_bpm_positions = []
            config.excluded_horizontal_corrector_positions = []
            config.excluded_vertical_corrector_positions = []
        config._sync_response_matrix_elements()
        override = self._machine_override_from_config(config)
        recipe = deepcopy(self.fit_recipe)
        self._apply_machine_override_to_recipe(recipe, override)

        self.project.loco_config = config
        self.fit_recipe = recipe
        self.project.fit_recipe = recipe.to_dict()
        self.project.recipe_machine_policy = "current"
        self.project.machine_configuration_override = deepcopy(override)
        self.project.completed_run = CompletedRunReference()
        self.project.fit_run_session = {}
        self.project.measurement_device_mapping = mapping
        self.project.compatibility_validation = {
            "required": True, "status": "validation_required",
            "issues": ["Mapped measurement devices were applied to FIT selections; validation is required."],
            "measurement_signature": json_safe(self._measurement_binding_signature(self.project)),
        }
        self.project.modified = True
        self._load_config_to_widgets()
        self._refresh_fit_workflow_table(self._active_fit_stage if self._active_fit_stage >= 0 else 0)
        self._refresh_measurement_mapping_feedback()
        self._refresh_ui("Mapped device selections applied. Next: Validate FIT configuration.")
        self.statusBar().showMessage("Selections applied. Next: Validate FIT configuration.", 8000)
        return True

    @Slot()
    def create_measurement_mapping_template(self) -> bool:
        try:
            _lattice, _lattice_path, _payload, inventory, _configuration = self._measurement_mapping_context()
        except Exception as exc:
            QMessageBox.warning(self, "Mapping template unavailable", str(exc)); return False
        filename = QFileDialog.getSaveFileName(
            self, "Create measurement-device mapping template", "measurement-device-mapping.json",
            "JSON mapping (*.json)", options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename: return False
        target = Path(filename).expanduser()
        if target.suffix.lower() != ".json": target = target.with_suffix(".json")
        target.resolve().write_text(json.dumps(measurement_mapping_template(inventory), indent=2), encoding="utf-8")
        self.measurement_mapping_feedback.setText(
            f"Template saved — not validated: {target.name}. Fill in only lattice_ordinal for unresolved devices, then load it."
        )
        return True

    @Slot()
    def load_measurement_mapping(self) -> bool:
        filename = QFileDialog.getOpenFileName(
            self, "Load completed measurement-device mapping", "",
            "Mapping files (*.json *.yaml *.yml)", options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename: return False
        try:
            lattice, lattice_path, _payload, inventory, configuration = self._measurement_mapping_context()
            document = load_measurement_mapping_document(filename)
            strict = bool(document.get("reference_lattice") and document.get("inventory"))
            mapping = validate_measurement_mapping(
                document, lattice, inventory, lattice_path=lattice_path,
                configuration_path=configuration, require_provenance=strict,
            )
            mapping["mapping_provenance"] = f"loaded and revalidated from {Path(filename).name}"
        except Exception as exc:
            QMessageBox.warning(self, "Measurement mapping is incompatible", str(exc)); return False
        self.project.measurement_device_mapping = mapping
        self.project.modified = True
        self.project.compatibility_validation = {
            "required": True, "status": "validation_required",
            "issues": ["Measurement-device mapping changed; validation is required."],
        }
        self._select_measurement_mapping_filter_after_operation(mapping)
        self._refresh_measurement_mapping_table(); self._refresh_measurement_mapping_feedback()
        self._refresh_ui("Measurement-device mapping loaded; validate before running FIT")
        return True

    @Slot()
    def save_validated_measurement_mapping(self) -> bool:
        mapping = self.project.measurement_device_mapping or {}
        counts = measurement_mapping_counts(mapping)
        if not counts["total"] or counts["mapped"] != counts["total"]:
            QMessageBox.warning(self, "Mapping is incomplete", "Resolve and validate every measured device before saving a reusable mapping.")
            return False
        try:
            lattice, lattice_path, _payload, inventory, configuration = self._measurement_mapping_context()
            mapping = validate_measurement_mapping(mapping, lattice, inventory, lattice_path=lattice_path,
                                                  configuration_path=configuration, require_provenance=True)
        except Exception as exc:
            QMessageBox.warning(self, "Mapping could not be revalidated", str(exc)); return False
        filename = QFileDialog.getSaveFileName(
            self, "Save validated measurement-device mapping", "measurement-device-mapping.validated.json",
            "JSON mapping (*.json)", options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename: return False
        target = Path(filename).expanduser()
        if target.suffix.lower() != ".json": target = target.with_suffix(".json")
        target.resolve().write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        self.project.measurement_device_mapping = mapping
        self.measurement_mapping_feedback.setText(f"Validated reusable mapping saved — {counts['mapped']} devices: {target.name}")
        return True

    def _refresh_measurement_mapping_feedback(self) -> None:
        if not hasattr(self, "measurement_mapping_feedback"): return
        mapping = self.project.measurement_device_mapping or {}
        counts = measurement_mapping_counts(mapping)
        if not counts["total"]:
            ready = bool(self.project.measurements.get("orm") and self.project.lattice.path)
            self.measurement_mapping_feedback.setText(
                "Ready. Step 1: Find device mappings."
                if ready else "Load measurements and a reference lattice to begin."
            )
            self.save_measurement_mapping_button.setEnabled(False)
            self.use_measurement_mappings_button.setEnabled(False)
            return
        unresolved = counts["total"] - counts["mapped"]
        applied = all(
            (self.project.loco_config.element_selection_state.get(field) or {}).get("method")
            == "validated_measurement_mapping"
            for field in ("bpm_ords", "horizontal_corrector_ords", "vertical_corrector_ords")
        )
        if unresolved:
            text = (
                f"Mapping incomplete — {counts['mapped']} mapped, {counts['unresolved']} unresolved, "
                f"{counts['ambiguous']} ambiguous, {counts['duplicate']} duplicate and {counts['invalid']} invalid. "
                "Resolve the unmatched devices first."
            )
        elif applied:
            text = (
                f"{counts['mapped']}/{counts['total']} devices mapped and selections applied. "
                "Next: Validate FIT configuration."
            )
        else:
            text = f"{counts['mapped']}/{counts['total']} devices mapped. Next: Use mappings in FIT."
        self.measurement_mapping_feedback.setText(text)
        self.save_measurement_mapping_button.setEnabled(not unresolved)
        self.use_measurement_mappings_button.setEnabled(not unresolved)

    def _select_measurement_mapping_filter_after_operation(self, mapping: dict) -> None:
        """Choose the useful post-operation view without disturbing later user choices."""
        counts = measurement_mapping_counts(mapping)
        mode = "all" if counts["total"] and counts["mapped"] == counts["total"] else "attention"
        index = self.measurement_mapping_filter.findData(mode)
        if index >= 0:
            self.measurement_mapping_filter.setCurrentIndex(index)

    def _refresh_measurement_mapping_table(self, *_args) -> None:
        if not hasattr(self, "measurement_mapping_table"): return
        query = self.measurement_mapping_search.text().strip().lower()
        mode = self.measurement_mapping_filter.currentData()
        rows = (self.project.measurement_device_mapping or {}).get("mappings", ())
        table = self.measurement_mapping_table; table.setSortingEnabled(False); table.setRowCount(0)
        role_labels = {"bpm": "BPM", "horizontal_corrector": "H corrector", "vertical_corrector": "V corrector"}
        displayed = 0
        for item in rows:
            status = item.get("status", "unresolved")
            if mode == "attention" and status == "mapped": continue
            if mode == "validated" and status != "mapped": continue
            searchable = " ".join(str(item.get(key, "")) for key in ("measured_name", "lattice_name", "lattice_device", "reason")).lower()
            if query and query not in searchable: continue
            capability=(item.get("component") or "BPM identity")+(" — validated" if status=="mapped" else " — not validated")
            values = (item.get("measured_name", "") or "Not available", role_labels.get(item.get("role"), item.get("role", "")) or "Not available",
                      item.get("lattice_name") or "Not available", item.get("lattice_device") or "Not available",
                      item.get("lattice_ordinal") if item.get("lattice_ordinal") is not None else "Not available",
                      identity_match_label(item.get("identity_match")), capability,
                      "Validated" if status == "mapped" else status.capitalize(), item.get("reason", "") or "Not available")
            row = table.rowCount(); table.insertRow(row)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value)); cell.setToolTip(str(value)); table.setItem(row, column, cell)
            displayed += 1
        table.setSortingEnabled(True)
        if not rows:
            self.measurement_mapping_table_hint.setText(
                "No mapping records yet. Select Find machine mappings after loading measurements and a reference lattice."
            )
            self.measurement_mapping_table_hint.setVisible(True)
        elif not displayed:
            self.measurement_mapping_table_hint.setText("No mapping records match the current search and filter.")
            self.measurement_mapping_table_hint.setVisible(True)
        else:
            self.measurement_mapping_table_hint.setVisible(False)

    @Slot()
    def select_lattice(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Select lattice/model file",
            "",
            "Model files (*.mat *.h5 *.hdf5 *.npy *.npz *.json *.yaml *.yml);;All files (*)",
        )[0]
        if not filename:
            return
        path = Path(filename).expanduser().resolve()
        try:
            variable = ""
            if path.suffix.lower() == ".mat":
                candidates = mat_lattice_candidates(path)
                if not candidates:
                    raise NoMatLatticeError(
                        f"{path.name} contains no variable that Accelerator Toolbox can load as a valid lattice."
                    )
                if len(candidates) == 1:
                    selected = candidates[0]
                else:
                    selected = self._choose_mat_lattice_candidate(candidates)
                    if selected is None:
                        return
                variable = selected.variable
            lattice = load_project_lattice(path, variable or None)
            selection = LatticeSelection(
                path=str(path), file_type=path.suffix.lower().lstrip("."),
                element_count=len(lattice), mat_variable=variable,
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Unable to load reference lattice",
                f"The selected file was not loaded and the active project is unchanged.\n\n{exc}",
            )
            return
        self.project.lattice = selection
        self.project.modified = True
        self._mark_compatibility_stale("Reference lattice changed; validation is required.")
        self._element_preview_signature = None
        self._reference_model_info_path = None
        self._refresh_element_selection_ui()
        self._refresh_ui(f"Selected lattice {path.name}")

    def _choose_mat_lattice_candidate(self, candidates, current: str = ""):
        labels = [f"{candidate.variable} — {candidate.element_count} elements" for candidate in candidates]
        current_index = next(
            (index for index, candidate in enumerate(candidates) if candidate.variable == current), 0
        )
        choice, accepted = QInputDialog.getItem(
            self, "Select MAT lattice variable",
            "Valid Accelerator Toolbox lattice variables in this MAT file:",
            labels, current_index, False,
        )
        if not accepted:
            return None
        return candidates[labels.index(choice)]

    @Slot()
    def change_mat_variable(self) -> None:
        if not self.project.lattice.path:
            return
        path = self.project.resolve_path(self.project.lattice.path)
        if path.suffix.lower() != ".mat":
            return
        try:
            candidates = mat_lattice_candidates(path)
            if not candidates:
                raise NoMatLatticeError(
                    f"{path.name} contains no valid Accelerator Toolbox lattice variables."
                )
            # Unlike initial automatic detection, this is an explicit advanced
            # action.  Always show the selector—even when there is only one
            # valid candidate—so clicking the button gives visible feedback
            # and lets the operator confirm which MAT variable is in use.
            selected = self._choose_mat_lattice_candidate(
                candidates, self.project.lattice.mat_variable
            )
            if selected is None or selected.variable == self.project.lattice.mat_variable:
                return
            lattice = load_project_lattice(path, selected.variable)
        except Exception as exc:
            QMessageBox.warning(self, "Unable to change MAT variable", str(exc))
            return
        self.project.lattice.mat_variable = selected.variable
        self.project.lattice.element_count = len(lattice)
        self.project.modified = True
        self._mark_compatibility_stale("Reference-lattice MAT variable changed; validation is required.")
        self._element_preview_signature = None
        self._reference_model_info_path = None
        self._refresh_ui(f"Selected MAT variable {selected.variable}")

    @Slot()
    def import_measurement(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Import measurement file",
            "",
            "Measurement files (*.h5 *.hdf5 *.mat *.npy *.npz);;HDF5 (*.h5 *.hdf5);;MAT (*.mat);;NumPy (*.npy *.npz)",
        )[0]
        if filename:
            source = Path(filename)
            role = self.measurement_role.currentText()
            path = self._store_imported_measurement(source, role)
            metadata = inspect_measurement_metadata(path, role)
            self.project.measurements[role] = ImportedDataset(
                role=role,
                path=str(path),
                file_type=path.suffix.lower().lstrip("."),
                size_bytes=path.stat().st_size,
                options=metadata,
            )
            self._apply_measurement_metadata(role, metadata)
            self._mark_compatibility_stale(f"{role.replace('_', ' ').title()} measurement changed; validation is required.")
            self.project.modified = True
            self._refresh_ui(f"Imported {role}: {path.name}")

    def _missing_measurement_roles(self) -> list[tuple[str, ImportedDataset]]:
        """Return missing external measurement references in stored role order."""

        return [
            (role, dataset)
            for role, dataset in self.project.measurements.items()
            if not self.project.resolve_path(dataset.path).is_file()
        ]

    @Slot()
    def locate_missing_measurement(self) -> None:
        """Replace one missing external reference without changing measurement data."""

        missing = self._missing_measurement_roles()
        if not missing:
            self.measurement_file_status.setText("All imported measurement files are available.")
            self.locate_missing_measurement_button.setEnabled(False)
            return
        role, dataset = missing[0]
        filename = QFileDialog.getOpenFileName(
            self,
            f"Locate missing {role.replace('_', ' ')} measurement",
            str(Path(dataset.path).expanduser().parent),
            "Measurement files (*.h5 *.hdf5 *.mat *.npy *.npz);;All files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if not filename:
            self.statusBar().showMessage("Measurement location unchanged — no file selected.", 4000)
            return
        replacement = Path(filename).expanduser().resolve()
        try:
            detected = inspect_measurement_metadata(replacement, role)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Measurement could not be loaded",
                f"The selected file could not replace the missing {role} measurement: {exc}",
            )
            return
        # Preserve project-specific options while refreshing facts that can be
        # inspected from the replacement file.  The scientific arrays remain
        # external and are never copied into the project JSON.
        options = deepcopy(dataset.options)
        options.update(detected)
        dataset.path = str(replacement)
        dataset.file_type = replacement.suffix.lower().lstrip(".")
        dataset.size_bytes = replacement.stat().st_size
        dataset.options = options
        self._apply_measurement_metadata(role, detected)
        self._mark_compatibility_stale(f"{role.replace('_', ' ').title()} measurement path changed; validation is required.")
        self.project.modified = True
        self._refresh_ui(f"Located {role} measurement: {replacement.name}")

    def _update_measurement_hint(self, role: str) -> None:
        self.measurement_hint.setText(IMPORT_HINTS.get(role, IMPORT_HINTS["other"]))

    def _apply_measurement_metadata(self, role: str, metadata: dict) -> None:
        applied = []
        if role == "orm":
            for key, widget in (("dkick_h", self.rm_dkick_h), ("dkick_v", self.rm_dkick_v)):
                if key in metadata:
                    widget.setValue(float(metadata[key])); applied.append(f"{key}={metadata[key]:g} rad")
        elif role == "dispersion" and "rf_step_hz" in metadata:
            value = float(metadata["rf_step_hz"])
            self.rm_rf_step.setValue(value); self.params_rfstep.setValue(value); self.fixed_rfstep.setValue(value)
            applied.append(f"RF step={value:g} Hz")
        if "bidirectional" in metadata:
            self.rm_bidirectional.setChecked(bool(metadata["bidirectional"])); applied.append(f"bidirectional={bool(metadata['bidirectional'])}")
        datasets = metadata.get("available_datasets") or metadata.get("datasets") or []
        if isinstance(datasets, dict):
            datasets = list(datasets.values())
        if datasets: applied.append("datasets=" + ", ".join(map(str, datasets)))
        self.measurement_metadata_label.setText(
            "Detected/applied measurement metadata: " + ", ".join(applied) if applied else
            "No recognized measurement settings were found. FIT settings were not changed."
        )


    def _store_imported_measurement(self, source: Path, role: str) -> Path:
        """Copy imported measurement data into the project folder when possible."""

        if not self.project.path:
            return source
        project_dir = Path(self.project.path).expanduser().resolve().parent
        data_dir = project_dir / "measurements"
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / f"{role}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def validate_project(self) -> None:
        self.project.loco_config = self._collect_loco_configuration()
        self._store_active_fit_stage()
        # Compatibility must be checked against the edits currently shown in
        # the stage editor, not the recipe snapshot last loaded from disk.
        self.project.fit_recipe = self.fit_recipe.to_dict()
        structural = [
            message for message in self.project.validation_messages()
            if not message.startswith("Measurement compatibility must be validated")
        ]
        if structural:
            if self.project.compatibility_validation.get("required"):
                self.project.compatibility_validation.update(status="validation_required", issues=list(structural))
            QMessageBox.information(
                self, "FIT configuration is incomplete",
                "Complete the required project settings before compatibility can be validated.\n\n"
                + "\n".join(f"• {message}" for message in structural[:6]),
            )
            self._refresh_ui("Validation incomplete")
            return
        try:
            errors = self._validate_retained_measurements(self.project)
        except Exception as exc:
            if self.project.compatibility_validation.get("required"):
                self.project.compatibility_validation.update(
                    status="internal_error", issues=[f"Compatibility validation failed: {exc}"]
                )
            self._show_project_load_error(self.project.path or "Unsaved adapted project", exc)
            self._refresh_ui("Compatibility validation could not be completed")
            return
        if errors:
            if self.project.compatibility_validation.get("required"):
                self.project.compatibility_validation.update(status="incompatible", issues=list(errors))
            self._show_project_open_validation_error(self.project, errors)
            self._refresh_ui("Compatibility issues remain unresolved")
            return
        if self.project.compatibility_validation.get("required"):
            self.project.compatibility_validation.update(
                status="validated", issues=[],
                measurement_signature=json_safe(self._measurement_binding_signature(self.project)),
                lattice_checksum=file_sha256(self.project.resolve_path(self.project.lattice.path)),
            )
            self.project.modified = True
        QMessageBox.information(self, "Project validation", "Compatibility validation passed. Run LOCO is enabled.")
        self._refresh_ui("Compatibility validation passed")

    def _rename_project(self) -> None:
        value = self.dashboard_name.text().strip() or "Untitled LOCO Project"
        if value == self.project.name:
            return
        self.project.name = value
        self.project.modified = True
        self._refresh_ui("Project renamed")

    def _update_project_description(self) -> None:
        value = self.dashboard_description.text().strip()
        if value == self.project.description:
            return
        self.project.description = value
        self.project.modified = True
        self._refresh_ui("Project description updated")

    def _apply_theme_selection(self, theme_key: str | None) -> None:
        self.current_theme = theme_for_key(theme_key)
        apply_application_theme(QApplication.instance(), self.current_theme, self.current_accent)
        self._settings.setValue("appearance/theme", self.current_theme.key)
        self._settings.sync()
        for key, action in self.theme_actions.items():
            action.setChecked(key == self.current_theme.key)
        self._update_toggle_theme_action()
        for window in self._orm_comparison_windows:
            if hasattr(window, "apply_theme"):
                window.apply_theme(self.current_theme)
        if hasattr(self, "results_workspace"):
            self.results_workspace.apply_theme()
        self._refresh_branding_assets()
        self._refresh_ui(f"{self.current_theme.display_name} theme selected")

    def _on_accent_changed(self) -> None:
        key = self.accent_combo.currentData()
        if not key:return
        self.current_accent=str(key); self._settings.setValue("fit/appearance/accent",self.current_accent); self._settings.sync()
        apply_application_theme(QApplication.instance(),self.current_theme,self.current_accent)
        if hasattr(self,"results_workspace"):self.results_workspace.apply_theme()
        self._refresh_ui(f"{self.accent_combo.currentText()} accent selected")

    def _update_header_branding(self) -> None:
        if hasattr(self, "header_brand_action"):
            visible = self.width() >= 1300
            self.header_brand_action.setVisible(visible); self.header_brand.setVisible(visible)

    def _refresh_branding_assets(self) -> None:
        if hasattr(self, "header_brand_label"): self.header_brand_label.setText(wordmark_html(self.current_theme.key))
        if hasattr(self, "dashboard_logo"): set_asset(self.dashboard_logo, QSize(270, 180), DISPLAY_ASSET, crop_transparency=False, theme_key=self.current_theme.key)

    def _update_toggle_theme_action(self) -> None:
        if self.current_theme.key == "dark":
            self.toggle_theme_action.setText("☀️ Light")
            self.toggle_theme_action.setToolTip("Switch to the Light theme")
        else:
            self.toggle_theme_action.setText("🌙 Dark")
            self.toggle_theme_action.setToolTip("Switch to the Dark theme")


    def _apply_mode_visibility(self) -> None:
        if not hasattr(self, "fixed_group"):
            return
        advanced = self.project.mode == "Advanced"
        widgets = (
            self.rm_fixedpath, self.rm_log_info,
            self.loco_remove_coupling, self.loco_plot_fit_parameters,
            self.params_cmstep_h, self.params_cmstep_v, self.params_cmstep_file_row, self.params_rfstep,
            self.params_quads_attr, self.params_quads_attr_index, self.params_skew_attr,
            self.params_skew_attr_index, self.params_tilt_attr_r1, self.params_tilt_attr_r2,
            self.params_tilt_method, self.fixed_group,
        )
        for widget in widgets:
            widget.setVisible(advanced)
        # Raw Python-literal storage is intentionally never shown.  Showing a
        # parentless QLineEdit made it appear as the blank macOS window seen in
        # the bug report; users edit these values through the guided dialogs.
        self.params_init_policy.hide()
        self.params_init.hide()
        self.loco_fixedpath.setVisible(False)
        # Legacy configuration compatibility only: the generic Individuals
        # checkbox has been replaced by the three explicit parameterization
        # controls and must never become a free-floating visible widget.
        self.loco_individuals.setVisible(False)
        self.fit_init_group.setVisible(advanced)
        for form, field in self._advanced_form_rows:
            if hasattr(form, "setRowVisible"):
                form.setRowVisible(field, advanced)
            else:
                label = form.labelForField(field)
                if label is not None:
                    label.setVisible(advanced)
                field.setVisible(advanced)
        self._update_cmstep_input_availability()
        if hasattr(self, "results_workspace"):
            self.results_workspace.set_mode(self.project.mode)

    @Slot()
    def _toggle_theme(self) -> None:
        next_theme = "light" if self.current_theme.key == "dark" else "dark"
        self._apply_theme_selection(next_theme)

    @Slot(QAction)
    def _on_theme_changed(self, action: QAction) -> None:
        self._apply_theme_selection(action.data())

    @Slot(QAction)
    def _on_mode_changed(self, action: QAction) -> None:
        self.project.mode = action.text()
        self.project.modified = True
        self._settings.setValue("workflow/mode", self.project.mode)
        self._mode_label.setText(f"{self.project.mode} mode")
        self._apply_mode_visibility()
        self._refresh_ui(f"{self.project.mode} mode selected")

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        self._workflow_label.setText(f"Workflow: {self._workspace.tabText(index)}")

    def _measurement_table_details(self, role: str, dataset: ImportedDataset) -> str:
        """Return a compact, read-only shape/unit summary for the import table."""
        path = self.project.resolve_path(dataset.path)
        if not path.is_file():
            return "File unavailable"
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            return f"{dataset.file_type.upper()} · {dataset.size_bytes:,} bytes"
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                candidates: list[tuple[str, tuple[int, ...], str]] = []
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        raw_unit = obj.attrs.get("unit", obj.attrs.get("units", ""))
                        if isinstance(raw_unit, bytes):
                            raw_unit = raw_unit.decode(errors="replace")
                        candidates.append((name, tuple(obj.shape), str(raw_unit)))
                handle.visititems(collect)
                preferred = {
                    "orm": ("response_matrix", "orm"),
                    "dispersion": ("measured_eta_x", "eta_x", "measured_eta_y", "eta_y"),
                    "bpm_noise": ("Noise_BPMx", "noise_x", "Noise_BPMy", "noise_y"),
                }.get(role, ())
                selected = [item for item in candidates if item[0].split("/")[-1] in preferred]
                if not selected:
                    selected = candidates[:2]
                parts = []
                for name, shape, unit in selected[:2]:
                    dimensions = " × ".join(str(value) for value in shape) or "scalar"
                    parts.append(f"{name.split('/')[-1]}: {dimensions}" + (f" [{unit}]" if unit else ""))
                return " · ".join(parts) if parts else "No numeric datasets"
        except Exception as exc:
            return f"Details unavailable: {exc}"

    @staticmethod
    def _format_measurement_scalar(value: float) -> str:
        return f"{float(value):g}".replace("-", "−")

    def _active_measurement_metadata_summary(self) -> str:
        """Summarize retained acquisition metadata without changing FIT settings."""
        if not self.project.measurements:
            return "No measurements loaded."
        options = {
            role: dataset.options or {}
            for role, dataset in self.project.measurements.items()
        }
        parts: list[str] = []
        identity_sources = [
            value.get("bpm_names") or [] for value in options.values()
            if value.get("bpm_names")
        ]
        if identity_sources:
            bpm_count = len(identity_sources[0])
            ordered = all(list(names) == list(identity_sources[0]) for names in identity_sources[1:])
            parts.append(f"{bpm_count} ordered BPM identities" + ("" if ordered else " (order differs between files)"))

        orm = options.get("orm", {})
        if orm:
            acquisition = orm.get("acquisition_metadata") or {}
            h_names = orm.get("horizontal_corrector_names") or []
            v_names = orm.get("vertical_corrector_names") or []
            unit = orm.get("response_matrix_unit") or acquisition.get("response_matrix_unit") or "not recorded"
            matrix_kind = "normalized" if orm.get("scaled") else "unscaled"
            orm_text = f"ORM: {len(h_names)} H + {len(v_names)} V correctors, {matrix_kind} [{unit}]"
            actual = np.concatenate((
                np.asarray(orm.get("actual_kick_h_rad") or (), dtype=float).ravel(),
                np.asarray(orm.get("actual_kick_v_rad") or (), dtype=float).ravel(),
            ))
            if actual.size:
                magnitude = np.abs(actual) * 1e6
                if np.allclose(magnitude, magnitude[0]):
                    orm_text += f", actual bipolar separation {self._format_measurement_scalar(magnitude[0])} µrad"
                else:
                    orm_text += (
                        f", actual bipolar separations {self._format_measurement_scalar(np.min(magnitude))}–"
                        f"{self._format_measurement_scalar(np.max(magnitude))} µrad"
                    )
            parts.append(orm_text)

        dispersion = options.get("dispersion", {})
        if dispersion:
            acquisition = dispersion.get("acquisition_metadata") or {}
            unit = acquisition.get("canonical_measured_eta_unit") or "m"
            step = dispersion.get("rf_step_hz")
            definition = dispersion.get("measured_eta_definition") or acquisition.get("rf_difference_sign_convention")
            text = "Dispersion RF orbit difference"
            if step is not None:
                text += f": signed RF step {self._format_measurement_scalar(step)} Hz"
            if definition:
                text += f", {str(definition).replace('_', ' ')}"
            text += f" [{unit}]"
            parts.append(text)

        noise = options.get("bpm_noise", {})
        if noise:
            acquisition = noise.get("acquisition_metadata") or {}
            datasets = noise.get("datasets") or {}
            horizontal = noise.get("horizontal_dataset") or datasets.get("horizontal") or "horizontal dataset not recorded"
            vertical = noise.get("vertical_dataset") or datasets.get("vertical") or "vertical dataset not recorded"
            unit = acquisition.get("bpm_noise_unit") or ("m" if noise.get("units_convention") == "SI" else "not recorded")
            parts.append(f"BPM noise: {horizontal} / {vertical} [{unit}]")

        if not parts:
            return (
                "Measurement files are loaded, but no recognized acquisition metadata summary is available. "
                "FIT compatibility validation remains required."
            )
        return "Measurement metadata available — " + "; ".join(parts) + "."

    def _show_measurement_details(self, role: str) -> None:
        dataset = self.project.measurements.get(role)
        if dataset is None:
            QMessageBox.information(self, "Measurement details", "This measurement is no longer loaded.")
            return
        resolved = self.project.resolve_path(dataset.path)
        fields = measurement_display_fields(resolved, dataset.options)
        details = self._measurement_table_details(role, dataset)
        QMessageBox.information(
            self,
            f"{role.replace('_', ' ').title()} measurement details",
            f"Filename: {Path(dataset.path).name}\n"
            f"Directory: {resolved.parent}\n"
            f"Full path: {resolved}\n"
            f"Status: {'Loaded' if resolved.is_file() else 'Missing'}\n\n"
            f"Datasets and dimensions: {details}\n"
            f"Acquisition date: {fields['date']}\n"
            f"Acquisition time: {fields['time']}\n"
            f"Time source: {fields['timestamp_source']}\n"
            f"Machine/profile: {fields['machine_profile']}",
        )

    def _invalidate_stale_reference_comparisons(self) -> None:
        """Close comparison windows that no longer describe this project."""
        if not getattr(self, "_reference_comparison_windows", None):
            return
        try:
            signature = self._reference_request_signature(LocoRunRequest.from_project(self.project))
        except Exception:
            signature = ""
        for window in list(self._reference_comparison_windows):
            if getattr(window, "_project_signature", None) != signature:
                window.close()

    def _refresh_ui(self, message: str) -> None:
        self._invalidate_stale_reference_comparisons()
        suffix = " *" if self.project.modified else ""
        filename = Path(self.project.path).name if self.project.path else "Not saved"
        state = "Unsaved changes" if self.project.modified else ("Saved" if self.project.path else "New project")
        self._project_label.setText(f"Project: {self.project.name} — {filename} · {state}")
        self._project_label.setToolTip(
            f"Project file: {self.project.path or 'No project file selected'}\nState: {state}"
        )
        self.setWindowTitle(f"pyLOCO GUI — {self.project.name}{suffix}")
        has_fit_stages = bool(self.fit_recipe.stages)
        gui_complete = self.project.is_complete and has_fit_stages
        self._validation_label.setText(
            "Validation: complete"
            if gui_complete
            else "Validation: incomplete"
        )
        self._validation_label.setObjectName(
            "validationOk" if gui_complete else "validationMissing"
        )
        self.run_loco_action.setEnabled(
            gui_complete
            and self._run_thread is None
            and not self._run_preparing
        )
        compatibility = self.project.compatibility_validation or {}
        compatibility_ready = self._compatibility_is_validated()
        if hasattr(self, "resume_fit_run_button"):
            self.resume_fit_run_button.setEnabled(compatibility_ready and has_fit_stages)
        if hasattr(self, "resume_previous"):
            self.resume_previous.setEnabled(compatibility_ready and has_fit_stages)
        if hasattr(self, "compatibility_status_label"):
            if compatibility.get("required"):
                status = compatibility.get("status", "validation_required")
                issues = list(compatibility.get("issues") or [])
                if status == "validated":
                    self.compatibility_status_label.setObjectName("validationOk")
                    self.compatibility_status_label.setText("Compatibility validated — fitting and resume are enabled.")
                else:
                    self.compatibility_status_label.setObjectName("validationMissing")
                    first = (
                        issues[0]
                        .removeprefix("MISSING_MEASUREMENT_PROVENANCE: ")
                        .removeprefix("MALFORMED_PROJECT_CONFIGURATION: ")
                        if issues else "Validation is required."
                    )
                    remaining = f" · {len(issues) - 1} additional issue(s)" if len(issues) > 1 else ""
                    self.compatibility_status_label.setText(
                        f"Editing mode — fitting disabled. {first}{remaining} Use Validate after making changes."
                    )
                self.compatibility_status_label.setToolTip("\n".join(issues))
                self.compatibility_status_label.setVisible(True)
                self.compatibility_status_label.style().unpolish(self.compatibility_status_label)
                self.compatibility_status_label.style().polish(self.compatibility_status_label)
            else:
                self.compatibility_status_label.setVisible(False)
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        if (
            self.project.lattice.path
            and self.project.lattice.file_type.lower() == "mat"
            and not self.project.lattice.mat_variable
        ):
            # Backward-compatible upgrade for projects saved before MAT
            # variable provenance existed. Only a unique valid candidate is
            # accepted; ambiguous files still require explicit user choice.
            self._load_current_lattice()
        self.lattice_path.setText(self.project.lattice.path or "No lattice selected")
        self.lattice_type.setText(self.project.lattice.file_type or "—")
        is_mat = bool(self.project.lattice.path) and self.project.lattice.file_type.lower() == "mat"
        self.lattice_variable.setText(
            self.project.lattice.mat_variable or ("Not selected" if is_mat else "Not applicable")
        )
        self.change_mat_variable_button.setEnabled(is_mat)
        self.lattice_elements.setText(
            str(self.project.lattice.element_count)
            if self.project.lattice.element_count
            else "Unknown"
        )
        self._refresh_reference_model_info()
        self._refresh_element_selection_ui()
        self.measurement_list.clearContents()
        self.measurement_list.setRowCount(len(self.project.measurements))
        for row,(role, dataset) in enumerate(sorted(self.project.measurements.items())):
            resolved = self.project.resolve_path(dataset.path)
            fields=measurement_display_fields(resolved,dataset.options)
            available = resolved.is_file()
            details = self._measurement_table_details(role, dataset)
            values=(role.replace("_", " ").title(), Path(dataset.path).name,
                    fields["date"], fields["time"], "Loaded" if available else "Missing")
            for column,value in enumerate(values):
                item=QTableWidgetItem(str(value))
                item.setToolTip(
                    f"{resolved}\n{details}\n"
                    f"Acquisition date/time: {fields['date']} {fields['time']}\n"
                    f"Machine/profile: {fields['machine_profile']}"
                )
                self.measurement_list.setItem(row,column,item)
            details_cell = QWidget()
            details_layout = QHBoxLayout(details_cell)
            details_layout.setContentsMargins(5, 3, 5, 3)
            details_button = QToolButton()
            details_button.setText("Details…")
            details_button.setObjectName("measurementDetailsButton")
            details_button.setFixedHeight(30)
            details_button.setStyleSheet(
                "QToolButton { min-height: 0px; max-height: 30px; padding: 2px 9px; }"
            )
            details_button.setToolTip("Show the full path, directory, dimensions, units and acquisition metadata.")
            details_button.clicked.connect(
                lambda checked=False, measurement_role=role: self._show_measurement_details(measurement_role)
            )
            details_layout.addWidget(details_button, 0, Qt.AlignCenter)
            self.measurement_list.setCellWidget(row, 5, details_cell)
        session = self.project.measurement_session or {}
        session_name = str(session.get("session_id") or session.get("name") or "Not recorded")
        manifest = str(session.get("manifest") or "Not recorded")
        if self.project.measurements:
            session_text = f"Active measurement session: {session_name}"
            if manifest != "Not recorded":
                session_text += f" · Manifest: {Path(manifest).name}"
            self.active_measurement_session_label.setText(session_text)
            self.active_measurement_session_label.setToolTip(
                f"Session: {session_name}\nManifest: {manifest}"
            )
        else:
            self.active_measurement_session_label.setText("Active measurement session: none")
            self.active_measurement_session_label.setToolTip("")
        self.measurement_metadata_label.setText(self._active_measurement_metadata_summary())
        missing_measurements = self._missing_measurement_roles()
        if missing_measurements:
            details = "; ".join(f"{role}: {dataset.path}" for role, dataset in missing_measurements)
            self.measurement_file_status.setText(
                "Missing measurement file reference(s): " + details
                + ". Locate each file to restore the saved binding; its contents will not be embedded in the project."
            )
            self.locate_missing_measurement_button.setEnabled(True)
        elif self.project.measurements:
            self.measurement_file_status.setText("All imported measurement files are available.")
            self.locate_missing_measurement_button.setEnabled(False)
        else:
            self.measurement_file_status.setText("No measurement files imported.")
            self.locate_missing_measurement_button.setEnabled(False)
        self.recent_list.clear()
        self.recent_list.addItems(self.project.recent_projects)
        self.recent_menu.clear()
        for recent in self.project.recent_projects:
            action = self.recent_menu.addAction(recent)
            action.triggered.connect(
                lambda checked=False, value=recent: self.open_project(Path(value))
            )
        missing = self.project.validation_messages()
        if not has_fit_stages and not any("FIT stage" in item for item in missing):
            missing.append("At least one FIT stage must be configured before running LOCO.")
        self.dashboard_summary.setText(
            "Project complete. Run LOCO is enabled."
            if not missing
            else "Missing inputs:\n" + "\n".join(f"• {m}" for m in missing)
        )
        self._project_explorer.update_project(self.project)
        self.statusBar().showMessage(message, 4000)



    @Slot()
    def run_loco(self) -> None:
        if self._run_thread is not None or self._run_preparing:
            self.statusBar().showMessage("FIT is already running")
            return
        self._run_preparing = True
        self._run_cancel_requested = False
        self._set_waiting_game_status("running")
        self._run_started_at = __import__("time").monotonic()
        self.results_workspace.begin_run()
        self.results_workspace.update_progress({
            "phase": "preflight",
            "message": "Preparing inputs and checking the FIT workflow…",
            "workflow_fraction": 0.0,
        })
        self.run_loco_action.setEnabled(False)
        self._workspace.setCurrentIndex(self._workspace.indexOf(self.results_page))
        self._elapsed_timer.start(500)
        self.statusBar().showMessage("Preparing LOCO run…")
        # Return to Qt before loading the lattice or preparing the backend.
        # This guarantees that the Results page is painted and responsive
        # before any potentially expensive FIT setup begins.
        QTimer.singleShot(0, self._prepare_and_start_loco)

    @Slot()
    def _prepare_and_start_loco(self) -> None:
        try:
            self._validate_active_stage_widget_consistency()
            self._store_active_fit_stage()
            if not self.fit_recipe.stages:
                raise ValueError("No FIT stages are configured. Add and validate a FIT stage before running LOCO.")
            messages = self.project.validation_messages()
            if messages:
                raise ValueError("Missing required inputs:\n\n" + "\n".join(messages))
            self.project.loco_config = self._collect_loco_configuration()
            request = LocoRunRequest.from_project(self.project)
            workflow = self.fit_recipe if len(self.fit_recipe.stages) > 1 else None
            if workflow is not None:
                lattice = self._load_current_lattice() or []
                names = [str(getattr(element, "CommonName", None) or getattr(element, "FamName", None) or getattr(element, "Name", "")) for element in lattice]
                report = preflight_recipe(
                    workflow, lattice_path=request.lattice_path,
                    measurement_identity=request.measurement_session, lattice_element_names=names,
                    measurements=request.measurements,
                    measurement_options=request.measurement_options,
                    measurement_device_mapping=deepcopy(self.project.measurement_device_mapping),
                    lattice_mat_variable=request.lattice_mat_variable or None,
                )
                if not report["compatible"]:
                    raise ValueError("\n".join(report["errors"]))
        except Exception as exc:
            import traceback

            self._run_preparing = False
            self.results_workspace.fail_run(cancelled=False)
            self.results_workspace.run_status_label.setText("Run preparation failed")
            self.results_workspace.run_iteration_label.setText("Not started")
            self._append_run_log(traceback.format_exc())
            self._elapsed_timer.stop()
            self.cancel_loco_button.setEnabled(False)
            self._set_waiting_game_status("failed")
            self.run_loco_action.setEnabled(self.project.is_complete and bool(self.fit_recipe.stages))
            self.statusBar().showMessage("LOCO run could not be prepared")
            QMessageBox.critical(self, "Cannot start LOCO", f"Run preparation failed:\n\n{exc}")
            return
        if workflow is not None and workflow.stages:
            # Paint the workflow context before starting the worker thread.
            # On macOS, spawning numerical-Jacobian workers can temporarily
            # delay queued Qt signals; without this synchronous update the UI
            # misleadingly remains at "Preparing workflow" and 0%.
            self._on_loco_progress({
                "phase": "stage_start",
                "message": f"Starting {workflow.stages[0].name}.",
                "workflow_stage": 1,
                "workflow_stages": len(workflow.stages),
                "stage_name": workflow.stages[0].name,
                "stage_fraction": 0.0,
                "workflow_fraction": 0.0,
            })
        self._run_thread = QThread(self)
        if workflow is None:
            self._run_worker = LocoRunWorker(request)
        else:
            session_root = Path(self.project.path).parent if self.project.path else Path.cwd()
            self._run_worker = FitWorkflowWorker(
                request, deepcopy(workflow), session_root / "fit-run-session.json",
                resume_session=self._resume_fit_session,
            )
        self._run_worker.moveToThread(self._run_thread)
        self._run_worker.log.connect(self._append_run_log)
        self._run_worker.progress.connect(self._on_loco_progress)
        self._run_worker.svd_selection_requested.connect(self._on_svd_selection_requested)
        self._run_worker.finished.connect(self._on_loco_finished)
        self._run_worker.failed.connect(self._on_loco_failed)
        self._run_worker.finished.connect(self._cleanup_run_thread)
        self._run_worker.failed.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_worker.deleteLater)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()
        # QThread.started is emitted by the QThread object, whose affinity is
        # the GUI thread.  PySide may consequently execute a Python override
        # in the GUI despite moveToThread().  Queue the named Qt slot directly
        # on the worker object after its event loop has started.
        QMetaObject.invokeMethod(self._run_worker, "run", Qt.QueuedConnection)
        self._run_preparing = False
        self._refresh_ui("LOCO run started")

    @Slot(object)
    def _on_svd_selection_requested(self, request: dict) -> None:
        worker = self._run_worker
        if worker is None:
            return
        dialog = SVDSelectionDialog(
            request["singular_values"],
            request.get("iteration_tag", ""),
            request.get("default_rank", 0),
            self,
        )
        if dialog.exec() == QDialog.Accepted:
            worker.provide_svd_selection(dialog.selected_indices())
            self._append_run_log(
                f"Interactive SVD: retained {len(dialog.selected_indices())} singular value(s)."
            )
        else:
            worker.cancel_requested = True
            worker.provide_svd_selection(None)
            self._append_run_log("Interactive SVD selection cancelled by the user.")

    @Slot()
    def cancel_loco_run(self) -> None:
        if self._run_worker is not None:
            self._run_cancel_requested = True
            self._run_worker.cancel_requested = True
            self.cancel_loco_button.setEnabled(False)
            self._set_waiting_game_status("cancelled")
            self._append_run_log("Cancellation requested. Stopping at the next safe calculation checkpoint…")

    @Slot(str)
    def _append_run_log(self, message: str) -> None:
        self.results_workspace.append_log(message)

    @Slot(object)
    def _on_loco_progress(self, event: dict) -> None:
        if "workflow_stage" in event:
            self._append_run_log(
                f"Stage {event['workflow_stage']}/{event['workflow_stages']} — {event['stage_name']}"
            )
        self.results_workspace.update_progress(event)

    @Slot(object)
    def _on_loco_finished(self, result) -> None:
        if isinstance(self._run_worker, FitWorkflowWorker):
            self.project.fit_run_session = {"path": str(self._run_worker.session_path),
                                            "completed_stages": len(self._run_worker.recipe.stages)}
            self._resume_fit_session = None
            self._refresh_fit_workflow_table(self._active_fit_stage)
        self._append_run_log("Saved outputs:\n" + "\n".join(result.output_files))
        self.results_workspace.complete_run(result)
        if isinstance(self._run_worker, FitWorkflowWorker):
            self.results_workspace.load_workflow_session(
                self._run_worker.session_path, preferred_result=result.results_dir
            )
        self.project.completed_run = CompletedRunReference(
            results_dir=str(result.results_dir),
            elapsed_seconds=float(result.elapsed_seconds),
            status="completed",
        )
        self.project.modified = True
        self._project_explorer.set_result(self.results_workspace.loader)
        self._project_explorer.update_project(self.project)
        self._last_loco_result = result
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self._set_waiting_game_status("cancelled" if self._run_cancel_requested else "completed")
        self.statusBar().showMessage(f"FIT complete • {result.results_dir}")

    @Slot(str)
    def _navigate_from_explorer(self, target: str) -> None:
        if target in {"Machine", "Machine Components", "Measurements", "Fit"}:
            if target == "Machine":
                target = "Machine Components"
            index = next((i for i in range(self._workspace.count()) if self._workspace.tabText(i) == target), -1)
            if index >= 0:
                self._workspace.setCurrentIndex(index)
            return
        if not target.startswith("Results:"):
            return
        result_page = next((i for i in range(self._workspace.count()) if self._workspace.tabText(i) == "Results"), -1)
        if result_page >= 0:
            self._workspace.setCurrentIndex(result_page)
        label = target.split(":", 1)[1]
        index = next((i for i in range(self.results_workspace.tabs.count()) if self.results_workspace.tabs.tabText(i) == label), -1)
        if index >= 0 and self.results_workspace.tabs.isTabVisible(index):
            self.results_workspace.tabs.setCurrentIndex(index)

    @Slot(object)
    def _on_loco_failed(self, error: LocoRunError) -> None:
        partial_results = self.results_workspace.fail_run(cancelled=error.cancelled)
        if partial_results is not None:
            elapsed = None
            if self._run_started_at is not None:
                elapsed = __import__("time").monotonic() - self._run_started_at
            self.project.completed_run = CompletedRunReference(
                results_dir=str(partial_results), elapsed_seconds=elapsed,
                status="cancelled" if error.cancelled else "failed"
            )
            self.project.modified = True
            self._project_explorer.set_result(self.results_workspace.loader)
            self._project_explorer.update_project(self.project)
        self._append_run_log(error.message if error.cancelled else error.traceback)
        self._set_waiting_game_status("cancelled" if self._run_cancel_requested else "failed")
        if error.cancelled:
            self.statusBar().showMessage("FIT stopped safely • completed stages kept")
        else:
            QMessageBox.critical(self, "LOCO failed", f"The backend reported an error:\n\n{error.message}")

    @Slot()
    def _open_waiting_games(self) -> None:
        if self._waiting_games_dialog is None:
            self._waiting_games_dialog = WaitingGamesDialog(self)
        self._waiting_games_dialog.set_loco_status(
            "cancelled" if self._run_cancel_requested else "running"
        )
        self._waiting_games_dialog.show()
        self._waiting_games_dialog.raise_()
        self._waiting_games_dialog.activateWindow()

    def _set_waiting_game_status(self, state: str) -> None:
        if self._waiting_games_dialog is not None:
            self._waiting_games_dialog.set_loco_status(state)

    @Slot()
    def _cleanup_run_thread(self) -> None:
        self._elapsed_timer.stop()
        self.cancel_loco_button.setEnabled(False)
        if self._run_thread is not None:
            self._run_thread.quit()
            self._run_thread.wait()
        self._run_thread = None
        self._run_worker = None
        self._run_preparing = False
        self.run_loco_action.setEnabled(self.project.is_complete)
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self._refresh_ui("LOCO run finished")

    @Slot()
    def _update_elapsed_time(self) -> None:
        if self._run_started_at:
            elapsed = __import__("time").monotonic() - self._run_started_at
            self.run_elapsed_label.setText(self.results_workspace.format_elapsed(elapsed))


    @staticmethod
    def _reference_request_signature(request: LocoRunRequest) -> str:
        payload = {
            "lattice_path": request.lattice_path,
            "lattice_mat_variable": request.lattice_mat_variable,
            "measurements": request.measurements,
            "measurement_options": request.measurement_options,
            "measurement_session": request.measurement_session,
            "backend_mapping": request.backend_mapping,
        }
        encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @Slot()
    def compare_measurements_with_reference(self) -> None:
        """Start the read-only pre-FIT reference calculation in a worker."""
        if self._reference_comparison_thread is not None:
            self.statusBar().showMessage("Reference-model comparison is already running.", 4000)
            return
        try:
            self._validate_active_stage_widget_consistency()
            if self.fit_recipe.stages:
                self._store_active_fit_stage()
            self.project.loco_config = self._collect_loco_configuration()
            if not self.project.lattice.path:
                raise ValueError("Select a reference lattice first.")
            if "orm" not in self.project.measurements:
                raise ValueError("Import an ORM measurement first.")
            errors = self._validate_retained_measurements(self.project, comparison_only=True)
            if errors:
                raise ValueError(
                    "The imported measurements are not ready for comparison with the initial model.\n\n"
                    + "\n".join(errors)
                    + "\n\nThe FIT execution configuration was not evaluated by this check."
                )
            request = LocoRunRequest.from_project(self.project)
        except Exception as exc:
            QMessageBox.warning(self, "Pre-FIT comparison not ready", str(exc))
            return

        self._reference_comparison_start_signature = self._reference_request_signature(request)
        self._reference_comparison_progress = QProgressDialog(
            "Calculating the reference ORM and RF response…", "Cancel", 0, 0, self,
        )
        self._reference_comparison_progress.setWindowTitle("Reference-model comparison")
        self._reference_comparison_progress.setWindowModality(Qt.NonModal)
        self._reference_comparison_progress.setMinimumDuration(0)
        self._reference_comparison_thread = QThread(self)
        self._reference_comparison_worker = ReferenceComparisonWorker(request)
        self._reference_comparison_worker.moveToThread(self._reference_comparison_thread)
        self._reference_comparison_progress.canceled.connect(self._cancel_reference_comparison)
        self._reference_comparison_worker.finished.connect(self._on_reference_comparison_finished)
        self._reference_comparison_worker.failed.connect(self._on_reference_comparison_failed)
        self._reference_comparison_worker.finished.connect(self._cleanup_reference_comparison)
        self._reference_comparison_worker.failed.connect(self._cleanup_reference_comparison)
        self._reference_comparison_thread.finished.connect(self._reference_comparison_worker.deleteLater)
        self._reference_comparison_thread.finished.connect(self._reference_comparison_thread.deleteLater)
        self._reference_comparison_thread.start()
        QMetaObject.invokeMethod(self._reference_comparison_worker, "run", Qt.QueuedConnection)
        self.compare_reference_action.setEnabled(False)
        self.compare_reference_button.setEnabled(False)

    @Slot()
    def _cancel_reference_comparison(self) -> None:
        if self._reference_comparison_worker is not None:
            self._reference_comparison_worker.cancel_requested = True
            self.statusBar().showMessage(
                "Cancellation requested; the active AT calculation will finish at its next safe boundary.", 5000
            )

    @Slot(object)
    def _on_reference_comparison_finished(self, result) -> None:
        try:
            current = LocoRunRequest.from_project(self.project)
            if self._reference_request_signature(current) != self._reference_comparison_start_signature:
                self.statusBar().showMessage(
                    "Reference comparison discarded because the active project changed during calculation.", 6000
                )
                return
        except Exception:
            self.statusBar().showMessage("Reference comparison discarded because the active project changed.", 6000)
            return
        window = ReferenceComparisonWindow(result, self)
        window._project_signature = self._reference_comparison_start_signature
        self._reference_comparison_windows.append(window)
        window.destroyed.connect(
            lambda _obj=None, w=window: self._reference_comparison_windows.remove(w)
            if w in self._reference_comparison_windows else None
        )
        window.show()
        self.statusBar().showMessage("Reference-model comparison complete — measurements were unchanged.", 5000)

    @Slot(str)
    def _on_reference_comparison_failed(self, message: str) -> None:
        if "cancelled" in message.lower():
            self.statusBar().showMessage("Reference-model comparison cancelled.", 4000)
        else:
            QMessageBox.warning(self, "Reference comparison failed", message)

    @Slot()
    def _cleanup_reference_comparison(self) -> None:
        if self._reference_comparison_progress is not None:
            self._reference_comparison_progress.close()
        if self._reference_comparison_thread is not None:
            self._reference_comparison_thread.quit(); self._reference_comparison_thread.wait()
        self._reference_comparison_progress = None
        self._reference_comparison_worker = None
        self._reference_comparison_thread = None
        self.compare_reference_action.setEnabled(True)
        self.compare_reference_button.setEnabled(True)


    def _can_compare_orms(self) -> bool:
        loader = getattr(self.results_workspace, "loader", None)
        return loader is not None and loader.fitted_orm is not None

    def _latest_model_orm_path(self) -> Path | None:
        names = ("loco_results.npz", "model_orm_initial.h5")
        if self._last_loco_result is not None:
            result_dir = Path(self._last_loco_result.results_dir)
            for name in names:
                candidate = result_dir / name
                if candidate.exists():
                    return candidate
        if not self.project.path:
            return None
        results_root = Path(self.project.path).expanduser().resolve().parent / "results"
        if not results_root.exists():
            return None
        candidates = [path for name in names for path in results_root.glob(f"*/{name}")]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _load_measured_orm_for_comparison(self):
        import h5py
        import numpy as np

        dataset = self.project.measurements["orm"]
        path = self.project.resolve_path(dataset.path)
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as handle:
                if "response_matrix" in handle:
                    measured_orm = np.asarray(handle["response_matrix"])
                else:
                    keys = list(handle.keys())
                    if not keys:
                        raise ValueError(f"ORM measurement file {path} contains no datasets.")
                    measured_orm = np.asarray(handle[keys[0]])
        elif suffix == ".npy":
            measured_orm = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                key = "orm" if "orm" in archive else archive.files[0]
                measured_orm = np.asarray(archive[key])
        else:
            raise ValueError(f"Unsupported ORM comparison file type: {suffix}")

        bad_bpm_positions = _load_bad_bpm_positions(
            {
                key: str(self.project.resolve_path(dataset.path))
                for key, dataset in self.project.measurements.items()
            }
        )
        if bad_bpm_positions is None:
            return measured_orm

        from pyLOCO.pyloco import remove_bad_bpms

        total_bpms = measured_orm.shape[0] // 2
        if measured_orm.shape[0] != total_bpms * 2:
            raise ValueError(
                "Measured ORM must have an even number of rows before applying the Bad BPM list; "
                f"got shape {measured_orm.shape}."
            )
        cleaned_orm, _removed = remove_bad_bpms(
            measured_orm,
            bad_bpm_positions,
            total_bpms=total_bpms,
            axis=0,
            input_type="positions",
        )
        return cleaned_orm

    def _load_model_orm_for_comparison(self):
        import numpy as np

        path = self._latest_model_orm_path()
        if path is None:
            raise ValueError("No initial or final model ORM result was found.")
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(path, "r") as handle:
                if "response_matrix" not in handle:
                    raise ValueError(f"{path} does not contain a response_matrix dataset.")
                return np.asarray(handle["response_matrix"])
        with np.load(path, allow_pickle=True) as archive:
            if "orm_model" not in archive:
                raise ValueError(f"{path} does not contain an orm_model array.")
            return np.asarray(archive["orm_model"])

    @Slot()
    def compare_orms(self) -> None:
        try:
            loader = getattr(self.results_workspace, "loader", None)
            if loader is None:
                raise ValueError(
                    "Select a completed FIT stage and iteration in Results before opening this comparison."
                )
            measured_orm, model_orm, context = loader.orm_comparison_inputs()
            if self.results_workspace.stage_selector.isVisible():
                context["stage_name"] = self.results_workspace.stage_selector.currentText()
            window = OrmComparisonWindow(measured_orm, model_orm, self, context=context)
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            QMessageBox.warning(
                self, "Fitted-model comparison unavailable",
                f"{exc}\n\nThe selected result and current project were not modified."
            )
            return
        self._orm_comparison_windows.append(window)
        window.destroyed.connect(lambda _obj=None, w=window: self._orm_comparison_windows.remove(w) if w in self._orm_comparison_windows else None)
        window.show()

    def _build_about_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("About pyLOCO")
        dialog.setWindowIcon(self.windowIcon())
        dialog.setModal(True); dialog.resize(580, 680); dialog.setMinimumSize(430, 480)
        outer = QVBoxLayout(dialog)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(28, 22, 28, 22)
        logo = QLabel()
        set_asset(logo, QSize(360, 240), DISPLAY_ASSET, crop_transparency=False, theme_key=self.current_theme.key)
        layout.addWidget(logo, 0, Qt.AlignHCenter)

        def centered(text: str, *, rich: bool = False, object_name: str = "") -> QLabel:
            label = QLabel(text); label.setAlignment(Qt.AlignCenter); label.setWordWrap(True)
            if object_name: label.setObjectName(object_name)
            if rich: label.setTextFormat(Qt.RichText); label.setOpenExternalLinks(True)
            layout.addWidget(label); return label

        centered("S T O R A G E   R I N G   O P T I C S   C O R R E C T I O N", object_name="aboutTagline")
        layout.addSpacing(8)
        centered("pyLOCO — Storage Ring Optics Correction", object_name="aboutTitle")
        centered(f"Version {self._package_version()}")
        layout.addSpacing(8)
        centered("Scientific software for linear-optics correction workflows in storage rings.")
        centered("pyLOCO Suite: Measure acquires structured machine data, Fit reconstructs optics/model errors, and Correct reviews and prepares fitted machine corrections.")
        layout.addSpacing(10)
        centered(f"Contributors: {PROJECT_CONTRIBUTORS}")
        centered(f"With thanks to: {PROJECT_ACKNOWLEDGEMENTS}")
        centered(f"License: {PROJECT_LICENSE}")
        layout.addSpacing(10)
        centered(f"<i>{PROJECT_PAPER_TITLE}</i><br>IPAC’26, paper WEP5011", rich=True)
        centered(
            f'<a href="{PROJECT_REPOSITORY}">Repository / Source code</a><br>'
            f'<a href="{PROJECT_DOCUMENTATION}">Documentation</a> · '
            f'<a href="{PROJECT_PAPER_URL}">Scientific reference / methodology</a>',
            rich=True,
        )

        link_actions = (
            ("Documentation", PROJECT_DOCUMENTATION), ("Methodology", PROJECT_PAPER_URL),
            ("Source code", PROJECT_REPOSITORY), ("Report issue", PROJECT_ISSUES),
        )
        for pair in (link_actions[:2], link_actions[2:]):
            links = QHBoxLayout(); links.addStretch(1)
            for text, url in pair:
                button = QPushButton(text)
                button.clicked.connect(lambda _checked=False, value=url: QDesktopServices.openUrl(QUrl(value)))
                links.addWidget(button)
            links.addStretch(1); layout.addLayout(links)

        copy_actions = QHBoxLayout()
        copy_citation = QPushButton("Copy citation")
        copy_citation.clicked.connect(lambda: QApplication.clipboard().setText(self._software_citation()))
        copy_bibtex = QPushButton("Copy BibTeX")
        copy_bibtex.clicked.connect(lambda: QApplication.clipboard().setText(self._software_bibtex()))
        copy_actions.addStretch(1); copy_actions.addWidget(copy_citation); copy_actions.addWidget(copy_bibtex); copy_actions.addStretch(1)
        layout.addLayout(copy_actions); layout.addStretch(1)
        scroll.setWidget(content); outer.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)
        return dialog

    def _show_about_dialog(self) -> None:
        present_single_about_dialog(self,self._build_about_dialog)
