"""Shared, read-only GUI control search for Measure, FIT, and Correct."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QToolButton,
    QWidget,
)


SEARCH_ALIASES = {
    "jacobian": ("svd", "singular values", "derivative", "save jacobian", "jacobian calculator"),
    "quadrupole": ("quad", "normal quadrupole", "skew quadrupole", "tilt"),
    "fixedpathlength": ("fixed path length", "fixed circumference", "response matrix"),
    "screenshot": ("screen capture", "camera", "png"),
    "mapping": ("machine mapping", "device mapping", "lattice mapping"),
    "orm": ("orbit response matrix", "response matrix"),
    "dispersion": ("rf response", "rf dispersion"),
}


@dataclass(frozen=True)
class ControlSearchResult:
    label: str
    location: str
    widget: QWidget
    searchable_text: str
    requires_advanced: bool = False


def _clean(text: object) -> str:
    return " ".join(str(text or "").replace("&", "").replace("…", "").split())


def _widget_label(widget: QWidget) -> str:
    for candidate in (
        widget.accessibleName(),
        widget.text() if isinstance(widget, (QAbstractButton, QLabel)) else "",
        widget.title() if isinstance(widget, QGroupBox) else "",
        _form_label(widget),
        widget.currentText() if isinstance(widget, QComboBox) else "",
        widget.placeholderText() if isinstance(widget, QLineEdit) else "",
        widget.objectName(),
    ):
        cleaned = _clean(candidate)
        if cleaned:
            return cleaned
    return widget.metaObject().className()


def _form_label(widget: QWidget) -> str:
    parent = widget.parentWidget()
    while parent is not None:
        layout = parent.layout()
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(widget)
            if isinstance(label, QLabel):
                return label.text()
        parent = parent.parentWidget()
    return ""


def _tab_path(widget: QWidget) -> list[str]:
    path: list[str] = []
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QTabWidget):
            for index in range(parent.count()):
                page = parent.widget(index)
                if page is widget or page.isAncestorOf(widget):
                    path.append(_clean(parent.tabText(index)))
                    break
        parent = parent.parentWidget()
    path.reverse()
    return [part for part in path if part]


def _section_name(widget: QWidget) -> str:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox) and _clean(parent.title()):
            return _clean(parent.title())
        if isinstance(parent, QToolBar):
            return "Toolbar"
        parent = parent.parentWidget()
    return "Window"


def _set_containing_tabs(widget: QWidget) -> None:
    pairs: list[tuple[QTabWidget, int]] = []
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QTabWidget):
            for index in range(parent.count()):
                page = parent.widget(index)
                if page is widget or page.isAncestorOf(widget):
                    pairs.append((parent, index))
                    break
        parent = parent.parentWidget()
    for tabs, index in reversed(pairs):
        tabs.setCurrentIndex(index)


class ControlSearchController(QObject):
    """Index and reveal application controls without activating them."""

    _CONTROL_TYPES = (
        QAbstractButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    )

    def __init__(
        self,
        window: QMainWindow,
        toolbar: QToolBar,
        *,
        advanced_mode_getter: Callable[[], bool] | None = None,
        switch_to_advanced: Callable[[], None] | None = None,
        aliases: dict[str, Iterable[str]] | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.advanced_mode_getter = advanced_mode_getter
        self.switch_to_advanced = switch_to_advanced
        self.aliases = {key: tuple(values) for key, values in SEARCH_ALIASES.items()}
        if aliases:
            self.aliases.update({key: tuple(values) for key, values in aliases.items()})
        self.results: list[ControlSearchResult] = []
        self.compact = compact
        self.search_box = QLineEdit(toolbar)
        self.search_box.setObjectName("globalControlSearch")
        self.search_box.setAccessibleName("Find Setting or Action")
        self.search_box.setPlaceholderText("Find setting or action…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setMaximumWidth(250)
        self.search_box.setMinimumWidth(170)
        self.search_box.setToolTip("Find a setting or action (⌘F). Search never activates a control.")
        self.search_button = None
        if compact:
            self.search_button = QToolButton(toolbar)
            self.search_button.setObjectName("globalControlSearchButton")
            self.search_button.setText("🔍")
            self.search_button.setAccessibleName("Find Setting or Action")
            self.search_button.setToolTip("Find a setting or action (⌘F / Ctrl+F)")
            self.search_button.clicked.connect(self.focus_search)
            self.search_button_action = toolbar.addWidget(self.search_button)
            self.search_editor = QFrame(window)
            self.search_editor.setObjectName("compactControlSearchEditor")
            editor_layout = QHBoxLayout(self.search_editor)
            editor_layout.setContentsMargins(6, 6, 6, 6)
            self.search_box.setParent(self.search_editor)
            editor_layout.addWidget(self.search_box)
            self.search_editor.resize(300, self.search_box.sizeHint().height() + 12)
            self.search_editor.hide()
            self.search_box_action = None
            self.window.installEventFilter(self)
            self.search_box.installEventFilter(self)
        else:
            self.search_button_action = None
            self.search_editor = None
            self.search_box_action = toolbar.addWidget(self.search_box)

        # Compact search lives in a child overlay.  Its results must also stay
        # inside the owning window; a Qt.Popup top-level steals keyboard focus
        # from the editor after the first typed character on macOS.
        self.popup = QFrame(window) if compact else QFrame(window, Qt.Popup)
        self.popup.setObjectName("controlSearchPopup")
        self.popup.setMinimumWidth(440)
        from PySide6.QtWidgets import QVBoxLayout
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(6, 6, 6, 6)
        self.result_list = QListWidget(self.popup)
        self.result_list.setObjectName("controlSearchResults")
        self.result_list.setMinimumHeight(120)
        self.result_list.setMaximumHeight(340)
        popup_layout.addWidget(self.result_list)
        # Child widgets are visible by default when their parent window is
        # shown.  Explicitly hide the empty results frame so it cannot appear
        # at (0, 0) and cover the toolbar before the first search query.
        self.popup.hide()

        self.search_box.textChanged.connect(self._query_changed)
        self.search_box.returnPressed.connect(self._activate_current)
        self.result_list.itemActivated.connect(self._item_activated)
        self.result_list.itemClicked.connect(self._item_activated)
        self.shortcut = QShortcut(QKeySequence.Find, window)
        # Each pyLOCO window owns an independent search index.  Window scope
        # prevents Cmd/Ctrl+F from triggering every open application window.
        self.shortcut.setContext(Qt.WindowShortcut)
        self.shortcut.activated.connect(self.focus_search)

    def focus_search(self) -> None:
        was_visible = self.search_box.isVisible()
        if self.search_editor is not None:
            self._position_compact_editor()
            self.search_editor.show()
            self.search_editor.raise_()
        elif self.search_box_action is not None:
            self.search_box_action.setVisible(True)
        self.search_box.show()
        if not was_visible:
            # QWidgetAction needs one layout pass before its embedded editor
            # can reliably accept focus on Qt/macOS.
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        self.window.raise_()
        self.window.activateWindow()
        self.search_box.setFocus(Qt.ShortcutFocusReason)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self.search_box.setFocus(Qt.ShortcutFocusReason)
        self.search_box.selectAll()
        if self.search_box.text().strip():
            self._query_changed(self.search_box.text())

    def _position_compact_editor(self) -> None:
        """Place compact search below the toolbar, never over its actions."""
        if self.search_editor is None:
            return
        button = self.search_button
        if button is not None and button.isVisibleTo(self.window):
            point = button.mapTo(self.window, QPoint(0, button.height() + 4))
            x = min(point.x(), max(12, self.window.width() - self.search_editor.width() - 12))
        else:
            # Cmd/Ctrl+F remains available when responsive layout moves the
            # search action into More.  A hidden toolbar button has invalid
            # (usually 0,0) anchor geometry, which previously placed the
            # editor across the primary action labels on macOS.
            point = QPoint(0, 0)
            toolbar = self.window.findChild(QToolBar, "mainToolbar")
            if toolbar is not None:
                toolbar_point = toolbar.mapTo(self.window, QPoint(0, toolbar.height() + 4))
                point = QPoint(0, toolbar_point.y())
            x = max(12, self.window.width() - self.search_editor.width() - 12)
        self.search_editor.move(x, max(0, point.y()))

    def eventFilter(self, watched, event):  # type: ignore[override]
        if self.compact and event.type() == QEvent.Resize and watched is self.window:
            if self.search_editor is not None and self.search_editor.isVisible():
                self._position_compact_editor()
        if self.compact and event.type() == QEvent.KeyPress and watched is self.search_box:
            if event.key() == Qt.Key_Escape:
                self.popup.hide()
                if self.search_editor is not None:
                    self.search_editor.hide()
                return True
        return super().eventFilter(watched, event)

    def rebuild_index(self) -> list[ControlSearchResult]:
        results: list[ControlSearchResult] = []
        seen: set[int] = set()
        for widget in self.window.findChildren(QWidget):
            if not isinstance(widget, self._CONTROL_TYPES):
                continue
            if widget is self.search_box or self.popup.isAncestorOf(widget):
                continue
            if bool(widget.property("pyLOCOSearchExclude")):
                continue
            identity = id(widget)
            if identity in seen:
                continue
            seen.add(identity)
            label = _widget_label(widget)
            if not label or label.startswith("qt_"):
                continue
            tabs = _tab_path(widget)
            section = _section_name(widget)
            location_parts = tabs + ([section] if section not in tabs else [])
            location = " › ".join(location_parts) or "Window"
            tooltip = _clean(widget.toolTip())
            form_label = _clean(_form_label(widget))
            object_name = _clean(widget.objectName())
            words = [label, location, tooltip, form_label, object_name]
            lower = " ".join(words).lower()
            for key, related in self.aliases.items():
                if key in lower or any(term.lower() in lower for term in related):
                    words.extend((key, *related))
            results.append(ControlSearchResult(
                label=label,
                location=location,
                widget=widget,
                searchable_text=" ".join(words).lower(),
                requires_advanced=bool(widget.property("pyLOCORequiresAdvanced")),
            ))
        self.results = results
        return results

    def search(self, query: str) -> list[ControlSearchResult]:
        terms = [term.casefold() for term in query.replace(",", " ").split() if term]
        if not terms:
            return []
        matches = [
            result for result in self.rebuild_index()
            if all(term in result.searchable_text for term in terms)
        ]
        phrase = " ".join(terms)
        def relevance(result: ControlSearchResult) -> tuple[int, str, str]:
            label = result.label.casefold()
            location = result.location.casefold()
            score = (100 if phrase in label else 0) + (50 if label.startswith(phrase) else 0)
            score += sum(20 for term in terms if term in label)
            score += sum(5 for term in terms if term in location)
            return (-score, label, location)
        return sorted(matches, key=relevance)

    def _query_changed(self, query: str) -> None:
        matches = self.search(query)[:40]
        self.result_list.clear()
        for index, result in enumerate(matches):
            suffix = " · Advanced" if result.requires_advanced else ""
            item = QListWidgetItem(f"{result.label}  —  {result.location}{suffix}")
            item.setData(Qt.UserRole, index)
            item.setToolTip(result.widget.toolTip() or result.location)
            self.result_list.addItem(item)
        self._visible_matches = matches
        if matches:
            self.result_list.setCurrentRow(0)
            point = self.search_box.mapTo(self.window, QPoint(0, self.search_box.height() + 2))
            popup_width = max(440, self.search_box.width() * 2)
            if self.compact:
                popup_width = min(popup_width, max(260, self.window.width() - 24))
            self.popup.resize(popup_width, self.popup.sizeHint().height())
            if self.compact:
                x = min(max(12, point.x()), max(12, self.window.width() - popup_width - 12))
                self.popup.move(x, point.y())
            else:
                self.popup.move(self.window.mapToGlobal(point))
            self.popup.show()
            self.popup.raise_()
            if self.compact:
                # Showing a child results panel must not interrupt continued
                # typing in the compact editor.
                self.search_box.setFocus(Qt.ShortcutFocusReason)
        else:
            self.popup.hide()

    def _activate_current(self) -> None:
        item = self.result_list.currentItem()
        if item is not None:
            self._item_activated(item)

    def _item_activated(self, item: QListWidgetItem) -> None:
        index = int(item.data(Qt.UserRole))
        if 0 <= index < len(getattr(self, "_visible_matches", [])):
            self.navigate_to(self._visible_matches[index])

    def navigate_to(self, result: ControlSearchResult) -> bool:
        widget = result.widget
        if result.requires_advanced and self.advanced_mode_getter and not self.advanced_mode_getter():
            answer = QMessageBox.question(
                self.window,
                "Advanced setting",
                f"{result.label} is available in Advanced mode. Switch to Advanced mode and show it?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return False
            if self.switch_to_advanced:
                self.switch_to_advanced()
        _set_containing_tabs(widget)
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(widget, 50, 50)
                break
            parent = parent.parentWidget()
        self.popup.hide()
        self._highlight(widget)
        if not widget.isVisibleTo(self.window):
            QMessageBox.information(
                self.window, "Setting currently unavailable",
                f"{result.label} is not shown in the current workflow state. "
                "Complete the preceding workflow step or open the relevant data first.",
            )
        elif not widget.isEnabled():
            reason = _clean(widget.property("pyLOCODisabledReason")) or _clean(widget.toolTip())
            reason = reason or "This control is unavailable in the current application state."
            QMessageBox.information(
                self.window, "Setting currently unavailable",
                f"{result.label} is disabled.\n\n{reason}",
            )
        self.window.statusBar().showMessage(f"Found: {result.label} — {result.location}", 5000)
        return True

    def _highlight(self, widget: QWidget) -> None:
        previous = widget.styleSheet()
        widget.setStyleSheet(
            previous + "; border: 2px solid #FFD54F; background-color: rgba(255,213,79,45);"
        )
        def restore() -> None:
            try:
                widget.setStyleSheet(previous)
            except RuntimeError:
                # The window may have closed before the temporary highlight expires.
                pass
        QTimer.singleShot(1500, restore)


def install_control_search(
    window: QMainWindow,
    toolbar: QToolBar,
    **kwargs,
) -> ControlSearchController:
    return ControlSearchController(window, toolbar, **kwargs)
