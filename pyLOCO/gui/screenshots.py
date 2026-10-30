"""Shared visible-window screenshot support for the pyLOCO applications."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, QTimer, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QToolBar, QToolButton

from .file_dialogs import FileDialog as QFileDialog


_LAST_DIRECTORY_KEY = "screenshots/last_manual_directory"


class WindowScreenshotController(QObject):
    """Capture the currently visible pixels of one Qt top-level window."""

    def __init__(self, window: QMainWindow, *, application_slug: str,
                 directory_provider: Callable[[], str | Path | None]) -> None:
        super().__init__(window)
        self.window = window
        self.application_slug = application_slug
        self.directory_provider = directory_provider
        self.settings = QSettings("pyLOCO", "pyLOCO")
        self.save_action = QAction("📷 Save Screenshot", window)
        self.save_action.setToolTip(
            "Save the visible application window now using the automatic screenshots folder (F8)"
        )
        self.save_action.setShortcut(QKeySequence("F8"))
        self.save_action.setShortcutContext(Qt.ApplicationShortcut)
        self.save_action.triggered.connect(self.save_screenshot)
        # Keep the shortcut available when a modal warning/error dialog owns
        # focus.  Toolbar actions alone cannot be clicked through a modal.
        window.addAction(self.save_action)
        self.save_as_action = QAction("Save Screenshot As…", window)
        self.save_as_action.setToolTip("Choose another PNG filename and folder")
        self.save_as_action.triggered.connect(self.save_screenshot_as)

    def default_directory(self) -> Path:
        value = self.directory_provider()
        if value:
            return Path(value).expanduser().resolve(strict=False) / "screenshots"
        return Path.home() / "Pictures" / "pyLOCO" / "Screenshots"

    def _suggested_path(self, directory: Path | None = None) -> Path:
        folder = directory or self.default_directory()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        return folder / f"pyloco-{self.application_slug}-{stamp}.png"

    @Slot()
    def save_screenshot(self) -> Path | None:
        return self.capture_to(self._suggested_path())

    @Slot()
    def save_screenshot_as(self) -> None:
        remembered = str(self.settings.value(_LAST_DIRECTORY_KEY, "") or "")
        directory = Path(remembered).expanduser() if remembered else self.default_directory()
        filename = QFileDialog.getSaveFileName(
            self.window, "Save Screenshot As", str(self._suggested_path(directory)),
            "PNG image (*.png)",
        )[0]
        if not filename:
            return
        target = Path(filename).expanduser()
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        self.settings.setValue(_LAST_DIRECTORY_KEY, str(target.parent.resolve(strict=False)))
        # Wait until the modal file dialog has closed and the application has
        # repainted, so the dialog can never be included in the image.
        QTimer.singleShot(0, lambda path=target: self.capture_to(path))

    def capture_to(self, target: str | Path) -> Path | None:
        path = Path(target).expanduser().resolve(strict=False)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # When a message box is open it is the current application
            # window. Capture it directly; otherwise capture the main window.
            # QWidget.grab() remains desktop-independent and preserves the
            # visible scroll positions and plot state.
            modal=QApplication.activeModalWidget()
            capture_widget=modal if modal is not None and modal.isVisible() else self.window
            pixmap = capture_widget.grab()
            if pixmap.isNull():
                raise RuntimeError("Qt returned an empty window image")
            if not pixmap.save(str(path), "PNG"):
                raise OSError("Qt could not write the PNG file")
        except Exception as exc:
            self.window.statusBar().showMessage(f"Screenshot failed: {exc}")
            QMessageBox.warning(self.window, "Screenshot failed", str(exc))
            return None
        self.window.statusBar().showMessage(f"Screenshot saved: {path}", 10000)
        return path


def install_window_screenshot_actions(window: QMainWindow, toolbar: QToolBar, *,
                                      application_slug: str,
                                      directory_provider: Callable[[], str | Path | None],
                                      file_menu=None,
                                      compact: bool = False) -> WindowScreenshotController:
    """Install the common camera split-button and optional File-menu actions."""
    controller = WindowScreenshotController(
        window, application_slug=application_slug,
        directory_provider=directory_provider,
    )
    menu = QMenu(window)
    # The main half of the split button already performs the immediate save.
    # Repeating that action in the arrow menu made the two choices look like
    # duplicates, so the menu contains only the alternative destination flow.
    menu.addAction(controller.save_as_action)
    button = QToolButton(toolbar)
    button.setObjectName("screenshotButton")
    if compact:
        # QAction text is deliberately retained for menus and global search;
        # the toolbar uses only the familiar camera glyph.
        button.setText("📷")
        button.clicked.connect(controller.save_screenshot)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setAccessibleName("Save Screenshot")
    else:
        button.setDefaultAction(controller.save_action)
    button.setMenu(menu)
    button.setPopupMode(QToolButton.MenuButtonPopup)
    button.setAccessibleName("Save Screenshot")
    button.setToolTip(
        "Click the camera to save now in the automatic screenshots folder. "
        "Use the arrow to choose another location or filename. Press F8 while "
        "an error or warning dialog is open to capture that dialog."
    )
    toolbar.addWidget(button)
    if file_menu is not None:
        file_menu.addSeparator()
        file_menu.addAction(controller.save_action)
        file_menu.addAction(controller.save_as_action)
    controller.button = button
    return controller
