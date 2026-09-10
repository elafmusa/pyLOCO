"""Main window for the pyLOCO GUI.

The GUI manages project state, lattice metadata, imported measurement files,
backend-compatible LOCO configuration, and responsive execution monitoring.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import numpy as np
from html import escape
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import QObject, QRect, QSettings, QSize, Qt, QThread, QUrl, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QDoubleValidator, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
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

from .fit_workflow import FitRecipe, FitStage, execute_workflow, file_sha256, preflight_recipe

from .backend import LocoRunError, LocoRunRequest, LocoRunResult, run_loco_request, _load_bad_bpm_positions
from .machine_detection import detect_machine_elements
from .measurement_metadata import IMPORT_HINTS, inspect_measurement_metadata, measurement_display_fields
from .branding import DISPLAY_ASSET, application_icon, set_asset, wordmark_html
from .models.project import (
    CompletedRunReference, ImportedDataset, LatticeSelection, LocoConfiguration, ProjectMetadata,
    json_safe, load_cmstep_npz, load_example_project_data, measurement_options_from_config, resolve_element_name_file,
    resolve_example_machine_elements,
)
from .widgets.project_explorer import ProjectExplorer
from .widgets.orm_comparison import OrmComparisonWindow
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


class FitWorkflowWorker(LocoRunWorker):
    """Run a recipe serially while preserving the existing per-stage backend."""

    def __init__(self, request, recipe, session_path, resume_session=None) -> None:
        super().__init__(request); self.recipe = recipe; self.session_path = session_path; self.resume_session = resume_session

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

    def __init__(self, parent, role_key: str, current: list[int]) -> None:
        super().__init__(parent)
        self.role_key = role_key
        self.role_label, self.role_kind = ELEMENT_ROLES[role_key]
        self.setWindowTitle(f"Select {self.role_label}")
        self.resize(760, 560)
        self._lattice = parent._load_current_lattice()
        self.selected_ords = list(current)

        layout = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        self.type_radio = QRadioButton("AT element type")
        self.pattern_radio = QRadioButton("Family/name pattern")
        self.name_file_radio = QRadioButton("Load name file")
        self.file_radio = QRadioButton("Load index file")
        self.manual_radio = QRadioButton("Manual indices")
        self.manual_radio.setChecked(True)
        for button in (self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
            mode_row.addWidget(button)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        form = QFormLayout(); self._method_form=form
        self.type_edit = QLineEdit(self._default_type_name())
        self.pattern_edit = QLineEdit(self._default_pattern())
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
        form.addRow("Family/name regex", self.pattern_edit)
        form.addRow("Element-name file", self.name_file_row_widget)
        form.addRow("Name attribute", self.name_attribute)
        form.addRow("Index file", self.file_row_widget)
        form.addRow("Manual lattice ordinals", self.manual_edit)
        layout.addLayout(form)

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
        self._set_preview(current)
        for button in (self.type_radio, self.pattern_radio, self.name_file_radio, self.file_radio, self.manual_radio):
            button.toggled.connect(self._update_method_controls)
        self._update_method_controls()

    def _update_method_controls(self) -> None:
        def show_field(field,visible):
            field.setVisible(visible); field.setEnabled(visible)
            label=self._method_form.labelForField(field)
            if label is not None:label.setVisible(visible)
        show_field(self.type_edit,self.type_radio.isChecked())
        show_field(self.pattern_edit,self.pattern_radio.isChecked())
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
        return {"bpm": "BPM|MON", "hcor": "HCM|HCOR|CH", "vcor": "VCM|VCOR|CV", "quad": "Q", "skew": "SQ|SKQ|SKEW", "tilt": "Q", "cavity": "RFCAV|CAV|RF"}[self.role_kind]

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
            pattern = re.compile(self.pattern_edit.text().strip(), re.I)
            return [i for i, e in self._iter_elements() if pattern.search(self._element_name(e))]
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
        self.selected_ords = values
        self._set_preview(values)
        return True

    def _set_preview(self, values: list[int]) -> None:
        self.table.setRowCount(len(values))
        for row, ordinal in enumerate(values):
            elem = self._lattice[ordinal] if self._lattice is not None and 0 <= ordinal < len(self._lattice) else None
            use = QTableWidgetItem(str(row)); use.setCheckState(Qt.Checked); use.setData(Qt.UserRole, int(ordinal)); self.table.setItem(row, 0, use)
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
        selected=sum(self.table.item(row,0).checkState()==Qt.Checked for row in range(self.table.rowCount()) if self.table.item(row,0))
        self.selection_count.setText(f"{selected} selected / {self.table.rowCount()} in preview")

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
        self.selected_ords = self._validate_array(selected); self.accept()


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
        self._run_started_at = 0.0
        self._last_loco_result = None
        self._waiting_games_dialog: WaitingGamesDialog | None = None
        self._run_cancel_requested = False
        self._orm_comparison_windows = []
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
        group = QGroupBox("Project state")
        group.setLayout(form)
        body.addWidget(group)
        body.addWidget(self.dashboard_summary)
        body.addWidget(QLabel("Recent projects"))
        self.recent_list.setMinimumHeight(100); body.addWidget(self.recent_list)
        body.addStretch(1)
        self.recent_list.itemDoubleClicked.connect(
            lambda item: self.open_project(Path(item.text()))
        )
        return page

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
        self.lattice_path = QLabel("No lattice selected")
        self.lattice_path.setWordWrap(True)
        self.lattice_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lattice_type = QLabel("—")
        self.lattice_elements = QLabel("Unknown")
        choose = QPushButton("Select lattice/model file…")
        choose.setMaximumWidth(260)
        choose.clicked.connect(self.select_lattice)
        form = QFormLayout()
        selection_header=QWidget(); selection_layout=QHBoxLayout(selection_header); selection_layout.setContentsMargins(0,0,0,0)
        selection_title=QLabel("Reference lattice"); selection_title.setStyleSheet("font-weight:700")
        selection_layout.addWidget(selection_title); selection_layout.addStretch(); selection_layout.addWidget(choose)
        form.addRow(selection_header)
        form.addRow("Path", self.lattice_path)
        form.addRow("Type", self.lattice_type)
        form.addRow("Elements", self.lattice_elements)
        self.reference_model_info=QLabel("Not available"); self.reference_model_info.setObjectName("fitReferenceModelCard"); self.reference_model_info.setWordWrap(True); self.reference_model_info.setTextInteractionFlags(Qt.TextSelectableByMouse); self.reference_model_info.setMargin(14)
        form.addRow(self.reference_model_info)
        group = QGroupBox("Lattice selection and metadata")
        group.setLayout(form)
        content_layout.addWidget(group)

        self.element_count_labels = {}
        self.element_preview_tables = {}
        elements_group = QGroupBox("Machine Elements")
        elements_layout = QVBoxLayout(elements_group)
        elements_layout.setContentsMargins(16, 20, 16, 16)
        elements_layout.setSpacing(8)
        elements_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        machine_help = QLabel("Define BPMs, correctors, quadrupoles, and RF cavities after loading the lattice. Bad BPM positions are applied later as positions within the selected BPM list, not as lattice ordinals.")
        machine_help.setWordWrap(True)
        elements_layout.addWidget(machine_help)
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
        content_layout.addWidget(elements_group)
        exclusion_group = QGroupBox("BPM and corrector exclusions")
        exclusion_form = QFormLayout(exclusion_group)
        exclusion_form.setContentsMargins(16, 22, 16, 16)
        exclusion_form.setHorizontalSpacing(18)
        exclusion_form.setVerticalSpacing(12)
        exclusion_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        exclusion_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        exclusion_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        exclusion_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.bad_bpm_positions_edit = QLineEdit(); self.bad_bpm_positions_edit.setPlaceholderText("selected-list positions, e.g. 0, 4, 17")
        self.hcor_exclusions_edit = QLineEdit(); self.vcor_exclusions_edit = QLineEdit()
        self.exclusion_counts = QLabel(); self.exclusion_counts.setWordWrap(True); self.exclusion_counts.setMinimumHeight(36)
        for edit, label in ((self.bad_bpm_positions_edit, "Excluded BPM positions"), (self.hcor_exclusions_edit, "Excluded H-corrector positions"), (self.vcor_exclusions_edit, "Excluded V-corrector positions")):
            edit.setMinimumHeight(34)
            exclusion_form.addRow(label, edit); edit.editingFinished.connect(self._store_exclusions)
        exclusion_form.addRow("Counts", self.exclusion_counts)
        content_layout.addWidget(exclusion_group)
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
        self.measurement_list = QTableWidget(0, 5)
        self.measurement_list.setHorizontalHeaderLabels(["File / Measurement", "Type", "Date", "Time", "Machine / Profile"])
        self.measurement_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5): self.measurement_list.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
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
        self.measurement_metadata_label = QLabel("No measurement metadata loaded."); self.measurement_metadata_label.setWordWrap(True)
        layout.addWidget(self.measurement_hint); layout.addWidget(self.measurement_metadata_label)
        layout.addWidget(self.measurement_list)
        page.layout().addWidget(group)
        self.measurement_role.currentTextChanged.connect(self._update_measurement_hint)
        self._update_measurement_hint(self.measurement_role.currentText())
        return page


    def _fit_page(self) -> QWidget:
        page = self._page("LOCO Configuration")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        workflow_group = QGroupBox("FIT workflow / reusable recipe")
        workflow_layout = QVBoxLayout(workflow_group)
        self.fit_workflow_table = QTableWidget(0, 4)
        self.fit_workflow_table.setHorizontalHeaderLabels(["Stage", "Name", "Iterations", "Starting state"])
        configure_item_view(self.fit_workflow_table)
        self.fit_workflow_table.itemSelectionChanged.connect(self._select_fit_stage)
        self.fit_workflow_table.itemDoubleClicked.connect(self._inspect_fit_stage_result)
        workflow_layout.addWidget(self.fit_workflow_table)
        workflow_buttons = QHBoxLayout()
        for text, slot in (("+ Add stage", self._add_fit_stage), ("Duplicate", self._duplicate_fit_stage),
                           ("Remove", self._remove_fit_stage), ("↑", lambda: self._move_fit_stage(-1)),
                           ("↓", lambda: self._move_fit_stage(1)), ("Save FIT recipe…", self.save_fit_recipe),
                           ("Load FIT recipe…", self.load_fit_recipe),
                           ("Preview full workflow", self.preview_fit_workflow),
                           ("Save FIT run/session…", self.save_fit_run_session),
                           ("Resume FIT run…", self.resume_fit_run)):
            button = QPushButton(text); button.clicked.connect(slot); workflow_buttons.addWidget(button)
        workflow_buttons.addStretch(1); workflow_layout.addLayout(workflow_buttons)
        workflow_help = QLabel("A recipe stores stage strategy only. Measurements and the selected reference lattice remain in the FIT run/session.")
        workflow_help.setWordWrap(True); workflow_layout.addWidget(workflow_help)
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
        self.rm_fixedpath = QCheckBox("fixedpathlength")
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
        self.fixed_delta_skew = self._double_spin(-1.0, 1.0, 1e-3, 9)
        self.fixed_delta_q_tilt = self._double_spin(-1.0, 1.0, 1e-6, 9)
        perturbation_group=QGroupBox("Numerical perturbation steps")
        perturbation_form=QFormLayout(perturbation_group)
        for label, widget in (("Normal quadrupole ΔK", self.fixed_dk), ("Skew quadrupole ΔK", self.fixed_delta_skew), ("Quadrupole tilt Δθ", self.fixed_delta_q_tilt)):
            perturbation_form.addRow(label, widget)
        jacobian_form.addRow(perturbation_group)
        jacobian_group = QGroupBox("Jacobian calculation")
        jacobian_group.setLayout(jacobian_form)
        layout.addWidget(jacobian_group)

        self.outlier_enabled = QCheckBox("Reject outliers")
        self.outlier_sigma = self._double_spin(0.0, 1e6, 10.0, 3)
        self.norm_enabled = QCheckBox("Apply normalization")
        self.norm_mode = QComboBox()
        self.norm_mode.addItems(["component", "global", "none"])
        self.auto_delta = QCheckBox("Automatically adjust numerical perturbations")
        self.loco_fixedpath = QCheckBox("Keep path length fixed")
        self.loco_individuals = QCheckBox("individuals")
        self.loco_remove_coupling = QCheckBox("Remove coupling contribution")
        self.loco_plot_fit_parameters = QCheckBox("Plot fitted parameters")
        rej_form = QFormLayout()
        rej_form.addRow(self.outlier_enabled)
        rej_form.addRow("Sigma cut", self.outlier_sigma)
        rej_form.addRow(self.norm_enabled)
        rej_form.addRow("Normalization mode", self.norm_mode)
        self.loco_individuals.hide()  # compatibility state; superseded by explicit modes
        for widget in (self.auto_delta, self.loco_fixedpath, self.loco_remove_coupling, self.loco_plot_fit_parameters):
            rej_form.addRow(widget)
        rej_group = QGroupBox("Iterations and Outlier Rejection")
        rej_group.setLayout(rej_form)
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
        constraint_form.addRow(self.constraint_enabled)
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
        constraint_group = QGroupBox("Constraints")
        constraint_group.setLayout(constraint_form)
        layout.addWidget(constraint_group)

        self.parameter_checks = {}
        param_group = QGroupBox("Parameter Selection")
        param_layout = QVBoxLayout(param_group)
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
        self.params_init_policy = QLineEdit()
        self.params_init_policy.setPlaceholderText("Uses DEFAULT_INIT_POLICY unless overridden")
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
        self.params_init = QLineEdit()
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
        for label, widget in (("Initialization policy overrides", self.params_init_policy), ("Explicit initial values", self.params_init), ("Normal quadrupole attribute", self.params_quads_attr), ("Normal quadrupole attribute index", self.params_quads_attr_index), ("Skew quadrupole attribute", self.params_skew_attr), ("Skew quadrupole attribute index", self.params_skew_attr_index), ("Tilt R1 attribute", self.params_tilt_attr_r1), ("Tilt R2 attribute", self.params_tilt_attr_r2), ("Tilt update method", self.params_tilt_method)):
            param_form.addRow(label, widget)
        param_layout.addWidget(init_group)
        self._advanced_form_rows = (
            (cm_form, self.params_cmstep_h),
            (cm_form, self.params_cmstep_v),
            (cm_form, self.params_cmstep_file_row),
            (cm_form, self.params_cmstep_resolved),
            (cm_form, self.params_rfstep),
        )
        layout.addWidget(param_group)

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
        self.mcf_source.addItem("Automatic from lattice", "automatic")
        self.mcf_source.addItem("User-defined value", "user")
        self.mcf_user_value = QLineEdit()
        for label, widget in (("RF frequency [Hz]", self.fixed_frequency), ("Harmonic number", self.fixed_harm_number), ("RF frequency attribute", self.rm_rf_attr), ("rfStep", self.fixed_rfstep), ("Momentum compaction source", self.mcf_source), ("Momentum compaction factor", self.mcf_user_value)):
            fixed_form.addRow(label, widget)
        layout.addWidget(self.fixed_group)

        button_row = QHBoxLayout()
        import_button = QPushButton("Import configuration…")
        import_button.clicked.connect(self.import_loco_configuration)
        export_button = QPushButton("Export configuration…")
        export_button.clicked.connect(self.export_loco_configuration)
        button_row.addWidget(import_button)
        button_row.addWidget(export_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.fit_summary = QTextEdit()
        self.fit_summary.setReadOnly(True)
        summary_group = QGroupBox("Configuration summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.fit_summary)
        layout.addWidget(summary_group)
        layout.addStretch(1)
        scroll.setWidget(container)
        page.layout().addWidget(scroll, 1)
        self._load_config_to_widgets()
        self._connect_fit_controls()
        self._add_fit_stage(name="Stage 1", start_from="original_model")
        return page

    def _stage_configuration_snapshot(self) -> dict:
        cfg = self._collect_loco_configuration()
        return {"gui_config": json_safe(__import__("dataclasses").asdict(cfg)),
                "backend_mapping": json_safe(cfg.to_backend_mapping())}

    def _store_active_fit_stage(self) -> None:
        if 0 <= self._active_fit_stage < len(self.fit_recipe.stages) and not self._loading_config:
            self.fit_recipe.stages[self._active_fit_stage].configuration = self._stage_configuration_snapshot()

    def _refresh_fit_workflow_table(self, select: int | None = None) -> None:
        table = self.fit_workflow_table
        table.blockSignals(True); table.setRowCount(len(self.fit_recipe.stages))
        completed = int((self.project.fit_run_session or {}).get("completed_stages", 0))
        for row, stage in enumerate(self.fit_recipe.stages):
            options = stage.configuration.get("backend_mapping", stage.configuration).get("LOCOOptions", {})
            inheritance = {
                "original_model": "Original model — independent start",
                "previous_stage": "Previous fitted state + stage overrides",
                "saved_result": "Saved FIT result + stage overrides",
            }.get(stage.start_from, stage.start_from)
            values = (f"{'✓' if row < completed else '○'} {row + 1}", stage.name,
                      options.get("nIter", 1), inheritance)
            for column, value in enumerate(values): table.setItem(row, column, QTableWidgetItem(str(value)))
        table.blockSignals(False)
        if select is not None and 0 <= select < table.rowCount(): table.selectRow(select)

    def _add_fit_stage(self, checked=False, *, name: str | None = None, start_from: str | None = None) -> None:
        if hasattr(self, "solver_n_iter"):
            configuration = self._stage_configuration_snapshot()
        else:
            configuration = {"backend_mapping": self.project.loco_config.to_backend_mapping()}
        index = len(self.fit_recipe.stages)
        self.fit_recipe.stages.append(FitStage(name or f"Stage {index + 1}", configuration,
                                               start_from or ("original_model" if index == 0 else "previous_stage")))
        self._active_fit_stage = index; self._refresh_fit_workflow_table(index)

    def _select_fit_stage(self) -> None:
        rows = self.fit_workflow_table.selectionModel().selectedRows() if self.fit_workflow_table.selectionModel() else []
        if not rows: return
        selected = rows[0].row()
        if selected == self._active_fit_stage: return
        self._store_active_fit_stage(); self._active_fit_stage = selected
        data = self.fit_recipe.stages[selected].configuration.get("gui_config")
        if data:
            self.project.loco_config = LocoConfiguration.from_dict(data)
            self._load_config_to_widgets()

    def _duplicate_fit_stage(self) -> None:
        self._store_active_fit_stage()
        if not self.fit_recipe.stages: return
        source = deepcopy(self.fit_recipe.stages[max(0, self._active_fit_stage)])
        source.name += " copy"; source.start_from = "previous_stage" if self.fit_recipe.stages else "original_model"
        self.fit_recipe.stages.insert(self._active_fit_stage + 1, source)
        self._active_fit_stage += 1; self._refresh_fit_workflow_table(self._active_fit_stage)

    def _remove_fit_stage(self) -> None:
        if len(self.fit_recipe.stages) <= 1:
            self.statusBar().showMessage("Add at least one FIT stage"); return
        self.fit_recipe.stages.pop(self._active_fit_stage)
        self._active_fit_stage = min(self._active_fit_stage, len(self.fit_recipe.stages) - 1)
        self.fit_recipe.stages[0].start_from = "original_model"
        self._refresh_fit_workflow_table(self._active_fit_stage)

    def _move_fit_stage(self, delta: int) -> None:
        self._store_active_fit_stage(); old = self._active_fit_stage; new = old + delta
        if old < 0 or new < 0 or new >= len(self.fit_recipe.stages): return
        self.fit_recipe.stages[old], self.fit_recipe.stages[new] = self.fit_recipe.stages[new], self.fit_recipe.stages[old]
        self.fit_recipe.stages[0].start_from = "original_model"; self._active_fit_stage = new
        self._refresh_fit_workflow_table(new)

    def save_fit_recipe(self) -> None:
        self._store_active_fit_stage()
        filename = QFileDialog.getSaveFileName(self, "Save FIT recipe", "fit-recipe.json", "FIT recipe (*.json)")[0]
        if filename:
            try: self.fit_recipe.save(filename)
            except Exception as exc: QMessageBox.warning(self, "Cannot save FIT recipe", str(exc))

    def load_fit_recipe(self) -> None:
        filename = QFileDialog.getOpenFileName(self, "Load FIT recipe", "", "FIT recipe (*.json)")[0]
        if not filename: return
        try: self.fit_recipe = FitRecipe.load(filename)
        except Exception as exc: QMessageBox.warning(self, "Cannot load FIT recipe", str(exc)); return
        self._resume_fit_session = None
        self._active_fit_stage = -1; self._refresh_fit_workflow_table(0); self._select_fit_stage()

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
        filename = QFileDialog.getOpenFileName(self, "Resume FIT run", "", "FIT run session (*.json)")[0]
        if not filename: return
        try:
            session = FitRunSession.load(filename)
            current_lattice = str(self.project.resolve_path(self.project.lattice.path)) if self.project.lattice.path else ""
            if current_lattice and session.lattice_checksum != file_sha256(current_lattice):
                raise ValueError("The current reference lattice does not match this FIT run/session.")
            current_measurements = {key: str(self.project.resolve_path(value.path)) for key, value in self.project.measurements.items()}
            if current_measurements and current_measurements != session.measurements:
                raise ValueError("The current measurement binding does not match this FIT run/session.")
            missing = [item for checkpoint in session.checkpoints for item in checkpoint.validate_files()]
            if missing: raise ValueError("Incomplete continuation checkpoint: " + ", ".join(missing))
            recipe_data = deepcopy(session.recipe); recipe_data["stages"] = [FitStage(**item) for item in recipe_data.get("stages", [])]
            self.fit_recipe = FitRecipe(**recipe_data)
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
        self.compare_orms_action = QAction("Compare ORMs", self)
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
        toolbar.addAction(self.compare_orms_action)
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
        self.quad_jacobian_calculator.currentTextChanged.connect(
            self._update_jacobian_option_availability
        )
        self.skew_jacobian_calculator.currentTextChanged.connect(
            self._update_jacobian_option_availability
        )
        self.constraint_quad_exceptions.changed.connect(self._on_fit_config_changed)
        self.constraint_skew_exceptions.changed.connect(self._on_fit_config_changed)
        self.rm_dispersion.toggled.connect(self._update_context_controls)
        self.constraint_enabled.toggled.connect(self._update_context_controls)
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
        for widget in (
            self.constraint_quad_sigma, self.constraint_skew_sigma,
            self.constraint_quad_weights, self.constraint_skew_weights,
            self.constraint_quad_mask, self.constraint_skew_mask,
        ):
            widget.setEnabled(enabled)

    def _set_calculator_value(self, calculator: str) -> None:
        aliases = {"linear": "Linear", "analytical": "Analytical", "numerical": "Numerical", "tracking": "Numerical"}
        backend_value = aliases.get(str(calculator).strip().lower(), calculator)
        index = self.rm_calculator.findData(backend_value)
        if index >= 0:
            self.rm_calculator.setCurrentIndex(index)

    def _update_jacobian_option_availability(self) -> None:
        """Show analytical-only controls without changing their stored values."""
        normal_analytical = self.quad_jacobian_calculator.currentText() == "Analytical"
        skew_analytical = self.skew_jacobian_calculator.currentText() == "Analytical"
        self.normal_analytical_options.setVisible(normal_analytical)
        self.normal_analytical_options.setEnabled(normal_analytical)
        self.skew_analytical_options.setVisible(skew_analytical)
        self.skew_analytical_options.setEnabled(skew_analytical)

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
        enabled = self.resume_previous.isChecked()
        for widget in (self.resume_directory, self.resume_browse, self.resume_ring_file,
                       self.resume_fit_dict_file, self.resume_fit_results_file):
            widget.setEnabled(enabled)
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
        self.project.modified = True
        self._update_fit_summary()
        self._refresh_ui("LOCO configuration updated")

    def _update_fit_summary(self) -> None:
        if not hasattr(self, "fit_summary"):
            return
        cfg = self.project.loco_config
        backend = json.dumps(json_safe(cfg.to_backend_mapping()), indent=2)
        lines=cfg.summary_lines()
        sections=(
            ("Response and solver",lines[:3]),
            ("Data handling and initialization",lines[3:6]),
            ("Fitted parameters",lines[6:]),
        )
        blocks=[]
        for title,values in sections:
            rows=[]
            for value in values:
                label,separator,detail=str(value).partition(":")
                rows.append(
                    f"<tr><td width='190'><b>{escape(label)}</b></td>"
                    f"<td>{escape(detail.strip() if separator else label)}</td></tr>"
                )
            blocks.append(
                f"<h3 style='color:#A98BFF;margin:12px 0 4px 0'>{title}</h3>"
                f"<table cellspacing='0' cellpadding='5' width='100%'>{''.join(rows)}</table>"
            )
        self.fit_summary.setHtml(
            "<h2 style='margin:0 0 6px 0'>Current FIT configuration</h2>"
            "<div>Readable scientific summary of the settings that will be passed to the LOCO backend.</div>"
            +"".join(blocks)+
            "<h3 style='color:#A98BFF;margin:14px 0 4px 0'>Backend mapping (technical)</h3>"
            f"<pre>{escape(backend)}</pre>"
        )

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
            self.project.loco_config = config
            if lattice:
                source = Path(lattice)
                if not source.exists():
                    raise ValueError(f"Configured lattice file does not exist: {source}")
                self.project.lattice = LatticeSelection(path=str(source), file_type=source.suffix.lower().lstrip("."))
                loaded_lattice = self._load_current_lattice()
                if loaded_lattice is None:
                    raise ValueError(f"Unable to load configured lattice: {source}")
                self.project.lattice.element_count = len(loaded_lattice)
                resolved = resolve_example_machine_elements(filename, loaded_lattice)
                if any(getattr(resolved, key) for key in ELEMENT_ROLES):
                    self.project.loco_config.machine_elements = resolved
                    self.project.loco_config._sync_response_matrix_elements()
            for role, value in measurements.items():
                source = Path(value)
                if not source.exists():
                    raise ValueError(f"Configured {role.replace('_', ' ')} file does not exist: {source}")
                self.project.measurements[role] = ImportedDataset(
                    role=role, path=str(source), file_type=source.suffix.lower().lstrip("."),
                    size_bytes=source.stat().st_size,
                    options=measurement_options_from_config(config.source_config).get(role, {}),
                )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        self.project.modified = True
        self._load_config_to_widgets()
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
        self._refresh_ui("New project created")

    @Slot()
    def open_project(self, path: Path | None = None) -> None:
        if not self._confirm_discard_changes():
            return
        existing_recent = list(self.project.recent_projects)
        filename = (
            str(path)
            if path
            else QFileDialog.getOpenFileName(
                self,
                "Open pyLOCO project",
                "",
                "pyLOCO Project (*.pyloco.json);;JSON (*.json)",
            )[0]
        )
        if not filename:
            return
        self.project = ProjectMetadata.load(filename)
        for recent in reversed(existing_recent):
            self.project.add_recent_project(recent)
        self.project.add_recent_project(filename)
        self.dashboard_name.setText(self.project.name)
        self.dashboard_description.setText(self.project.description)
        self._load_config_to_widgets()
        self._load_project_fit_recipe()
        completed = self.project.completed_run
        if completed.results_dir:
            results_dir = self.project.resolve_path(completed.results_dir)
            if results_dir.exists():
                self.results_workspace.load_results(
                    results_dir, runtime=completed.elapsed_seconds
                )
                self._workspace.setCurrentWidget(self.results_page)
        self._refresh_ui(f"Opened {filename}")

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
        self.project.measurements={role:ImportedDataset(role,str(source),source.suffix.lower().lstrip("."),source.stat().st_size,deepcopy(handoff.options[role])) for role,source in handoff.files.items()}
        self.project.measurement_session=deepcopy(handoff.provenance); self.project.measurement_session["manifest"]=str(handoff.manifest)
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
        self.project.modified=True; self._load_config_to_widgets(); self._refresh_ui(f"Imported Measurement Session {handoff.session_id}")
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

    def _load_project_fit_recipe(self) -> None:
        raw = self.project.fit_recipe
        if raw:
            try:
                data = deepcopy(raw); data["stages"] = [FitStage(**item) for item in data.get("stages", [])]
                self.fit_recipe = FitRecipe(**data)
            except Exception:
                self.fit_recipe = FitRecipe()
        else:
            self.fit_recipe = FitRecipe()
        if not self.fit_recipe.stages:
            self._active_fit_stage = -1; self._add_fit_stage(name="Stage 1", start_from="original_model")
        else:
            self._active_fit_stage = -1; self._refresh_fit_workflow_table(0); self._select_fit_stage()

    @Slot()
    def save_project(self) -> None:
        if not self.project.path:
            self.save_project_as()
            return
        try:
            self._snapshot_project_from_widgets()
            self.project.save()
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._refresh_ui(f"Saved {self.project.path}")

    @Slot()
    def save_project_as(self) -> None:
        filename = QFileDialog.getSaveFileName(
            self,
            "Save pyLOCO project",
            self.project.path or f"{self.project.name}.pyloco.json",
            "pyLOCO Project (*.pyloco.json)",
        )[0]
        if filename:
            try:
                self._snapshot_project_from_widgets()
                self.project.save(filename)
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Save failed", str(exc))
                return
            self._refresh_ui(f"Saved {filename}")


    def _load_current_lattice(self):
        """Load the currently selected AT lattice for element detection/preview."""

        if not self.project.lattice.path:
            return None
        try:
            import at
            return at.load_lattice(self.project.resolve_path(self.project.lattice.path))
        except Exception:
            return None

    def _element_preview_rows(self, ords: list[int]) -> list[tuple[int, int, str, str]]:
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
        for key in ELEMENT_ROLES:
            values = list(getattr(elements, key))
            self.element_count_labels[key].setText(f"{len(values)} selected")
            table = self.element_preview_tables[key]
            table.setVisible(advanced)
            rows = self._element_preview_rows(values) if advanced else []
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(str(value)))
        self._update_exclusion_counts()

    def _refresh_reference_model_info(self) -> None:
        if not hasattr(self,"reference_model_info"):return
        path=self.project.lattice.path
        if getattr(self,"_reference_model_info_path",None)==path:return
        self._reference_model_info_path=path
        if not path:self.reference_model_info.setText("Not available");return
        try:
            from pyLOCO.reference_model import load_reference_model
            model=load_reference_model(path,source="FIT reference lattice")
            harmonic=str(model.harmonic_number) if model.harmonic_number is not None else "Not available"
            rf=f"{model.nominal_rf_hz/1e6:.9f} MHz" if model.nominal_rf_hz is not None else "Not available"
            residual=f"{model.rf_harmonic_residual_hz:.6g} Hz" if model.rf_harmonic_residual_hz is not None else "Not available"
            rows=(
                ("Beam energy",f"{model.energy_ev/1e9:.6g} GeV","Reference lattice"),
                ("Circumference",f"{model.circumference_m:.6f} m","AT lattice geometry"),
                ("Revolution frequency f<sub>rev</sub>",f"{model.revolution_frequency_hz:.6f} Hz","c / circumference"),
                ("Harmonic number h",harmonic,model.harmonic_number_source),
                ("RF frequency f<sub>RF</sub>",rf,model.rf_source),
                ("RF consistency residual",residual,"f<sub>RF</sub> − h·f<sub>rev</sub>"),
                ("Fractional tunes Qx / Qy",f"{model.tune[0]:.6f} / {model.tune[1]:.6f}","AT linear optics"),
                ("Chromaticity ξx / ξy",f"{model.chromaticity[0]:.6f} / {model.chromaticity[1]:.6f}","AT linear optics"),
                ("Momentum compaction α<sub>c</sub>",f"{model.momentum_compaction:.9g}","AT model"),
                ("Relativistic term 1/γ²",f"{model.inverse_gamma_squared:.9g}","Lattice energy"),
                ("Slip factor η",f"{model.slip_factor:.9g}","α<sub>c</sub> − 1/γ²"),
            )
            body="".join(f"<tr><td><b>{name}</b></td><td>{value}</td><td style='color:#9CA3AF'>{source}</td></tr>" for name,value,source in rows)
            self.reference_model_info.setText(
                "<h3 style='color:#A98BFF;margin:0 0 8px 0'>Reference model</h3>"
                "<div style='margin-bottom:8px'>Calculated from the loaded lattice with Accelerator Toolbox (AT).</div>"
                f"<table cellspacing='0' cellpadding='5' width='100%'><tr><th align='left'>Quantity</th><th align='left'>Value</th><th align='left'>Source</th></tr>{body}</table>"
                f"<div style='margin-top:10px'><b>File:</b> {escape(model.path.name)} &nbsp; <b>SHA-256:</b> {escape(model.checksum_sha256[:12])}…<br><b>Provenance:</b> {escape(str(model.source))}</div>"
            )
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
        self.project.modified = True; self._update_exclusion_counts()

    @Slot()
    def edit_element_selection(self, role_key: str) -> None:
        elements = self.project.loco_config.machine_elements
        current = list(getattr(elements, role_key))
        dialog = ElementSelectionDialog(self, role_key, current)
        if dialog.exec() != QDialog.Accepted:
            return
        setattr(elements, role_key, dialog.selected_ords)
        group_key={"normal_quadrupole_ords":"normal_quadrupole_groups","skew_quadrupole_ords":"skew_quadrupole_groups","quadrupole_tilt_ords":"quadrupole_tilt_groups"}.get(role_key)
        if group_key:setattr(elements,group_key,[])
        self.project.loco_config._sync_response_matrix_elements()
        self.project.modified = True
        self._refresh_element_selection_ui()
        self._update_fit_summary()
        self._refresh_ui(f"Updated {ELEMENT_ROLES[role_key][0]} selection")

    @Slot()
    def edit_family_groups(self, role_key: str) -> None:
        elements=self.project.loco_config.machine_elements
        group_key={"normal_quadrupole_ords":"normal_quadrupole_groups","skew_quadrupole_ords":"skew_quadrupole_groups","quadrupole_tilt_ords":"quadrupole_tilt_groups"}[role_key]
        dialog=FamilyGroupDialog(self,role_key,list(getattr(elements,role_key)),list(getattr(elements,group_key)))
        if dialog.exec()!=QDialog.Accepted:return
        setattr(elements,group_key,dialog.groups); self.project.modified=True; self._update_fit_summary(); self._refresh_ui(f"Updated {ELEMENT_ROLES[role_key][0]} family groups")

    @Slot()
    def select_lattice(self) -> None:
        filename = QFileDialog.getOpenFileName(
            self,
            "Select lattice/model file",
            "",
            "Model files (*.mat *.h5 *.hdf5 *.npy *.npz *.json *.yaml *.yml);;All files (*)",
        )[0]
        if filename:
            path = Path(filename)
            self.project.lattice = LatticeSelection(
                path=str(path), file_type=path.suffix.lower().lstrip(".")
            )
            lattice = self._load_current_lattice()
            if lattice is not None:
                self.project.lattice.element_count = len(lattice)
            self.project.modified = True
            self._refresh_element_selection_ui()
            self._refresh_ui(f"Selected lattice {path.name}")

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
            self.project.modified = True
            self._refresh_ui(f"Imported {role}: {path.name}")

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
        datasets = metadata.get("datasets") or []
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
        messages = self.project.validation_messages()
        QMessageBox.information(
            self,
            "Project Validation",
            (
                "Project is complete; Run LOCO is enabled."
                if not messages
                else "Missing required inputs:\n\n" + "\n".join(messages)
            ),
        )
        self._refresh_ui("Validation complete")

    def _rename_project(self) -> None:
        self.project.name = (
            self.dashboard_name.text().strip() or "Untitled LOCO Project"
        )
        self.project.modified = True
        self._refresh_ui("Project renamed")

    def _update_project_description(self) -> None:
        self.project.description = self.dashboard_description.text().strip()
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
            self.rm_fixedpath, self.rm_log_info, self.loco_fixedpath,
            self.loco_individuals, self.loco_remove_coupling, self.loco_plot_fit_parameters,
            self.params_init_policy, self.params_cmstep_h, self.params_cmstep_v, self.params_cmstep_file_row, self.params_rfstep,
            self.params_init, self.params_quads_attr, self.params_quads_attr_index, self.params_skew_attr,
            self.params_skew_attr_index, self.params_tilt_attr_r1, self.params_tilt_attr_r2,
            self.params_tilt_method, self.fixed_group,
        )
        for widget in widgets:
            widget.setVisible(advanced)
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

    def _refresh_ui(self, message: str) -> None:
        suffix = " *" if self.project.modified else ""
        self._project_label.setText(f"Project: {self.project.name}{suffix}")
        self.setWindowTitle(f"pyLOCO GUI — {self.project.name}{suffix}")
        self._validation_label.setText(
            "Validation: complete"
            if self.project.is_complete
            else "Validation: incomplete"
        )
        self._validation_label.setObjectName(
            "validationOk" if self.project.is_complete else "validationMissing"
        )
        self.run_loco_action.setEnabled(self.project.is_complete)
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self.lattice_path.setText(self.project.lattice.path or "No lattice selected")
        self.lattice_type.setText(self.project.lattice.file_type or "—")
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
            fields=measurement_display_fields(dataset.path,dataset.options)
            values=(dataset.name,role,fields["date"],fields["time"],fields["machine_profile"])
            for column,value in enumerate(values):
                item=QTableWidgetItem(str(value)); item.setToolTip(fields["tooltip"]); self.measurement_list.setItem(row,column,item)
        self.recent_list.clear()
        self.recent_list.addItems(self.project.recent_projects)
        self.recent_menu.clear()
        for recent in self.project.recent_projects:
            action = self.recent_menu.addAction(recent)
            action.triggered.connect(
                lambda checked=False, value=recent: self.open_project(Path(value))
            )
        missing = self.project.validation_messages()
        self.dashboard_summary.setText(
            "Project complete. Run LOCO is enabled."
            if not missing
            else "Missing inputs:\n" + "\n".join(f"• {m}" for m in missing)
        )
        self._project_explorer.update_project(self.project)
        self.statusBar().showMessage(message, 4000)



    @Slot()
    def run_loco(self) -> None:
        self._store_active_fit_stage()
        messages = self.project.validation_messages()
        if messages:
            QMessageBox.warning(self, "Cannot run LOCO", "Missing required inputs:\n\n" + "\n".join(messages))
            return
        if self._run_thread is not None:
            self.statusBar().showMessage("FIT is already running")
            return
        self.project.loco_config = self._collect_loco_configuration()
        request = LocoRunRequest.from_project(self.project)
        workflow = self.fit_recipe if len(self.fit_recipe.stages) > 1 else None
        if workflow is not None:
            lattice = self._load_current_lattice() or []
            names = [str(getattr(element, "CommonName", None) or getattr(element, "FamName", None) or getattr(element, "Name", "")) for element in lattice]
            report = preflight_recipe(
                workflow, lattice_path=request.lattice_path,
                measurement_identity=request.measurement_session, lattice_element_names=names,
            )
            if not report["compatible"]:
                QMessageBox.warning(self, "FIT workflow preflight failed", "\n".join(report["errors"])); return
        self._run_cancel_requested = False
        self._set_waiting_game_status("running")
        self._run_started_at = __import__("time").monotonic()
        self.results_workspace.begin_run()
        self.run_loco_action.setEnabled(False)
        self._workspace.setCurrentIndex(self._workspace.indexOf(self.results_page))
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
        self._run_thread.started.connect(self._run_worker.run)
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
        self._elapsed_timer.start(500)
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
        self.run_loco_action.setEnabled(self.project.is_complete)
        self.compare_orms_action.setEnabled(self._can_compare_orms())
        self._refresh_ui("LOCO run finished")

    @Slot()
    def _update_elapsed_time(self) -> None:
        if self._run_started_at:
            elapsed = __import__("time").monotonic() - self._run_started_at
            self.run_elapsed_label.setText(self.results_workspace.format_elapsed(elapsed))


    def _can_compare_orms(self) -> bool:
        return "orm" in self.project.measurements and self._latest_model_orm_path() is not None

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
            measured_orm = self._load_measured_orm_for_comparison()
            model_orm = self._load_model_orm_for_comparison()
            window = OrmComparisonWindow(measured_orm, model_orm, self)
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            QMessageBox.warning(self, "ORM Comparison unavailable", str(exc))
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
