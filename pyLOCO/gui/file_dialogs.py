"""Shared, theme-aware Qt file and directory dialogs for the pyLOCO GUIs.

The public ``FileDialog`` facade intentionally mirrors the static
``QFileDialog`` API used throughout the applications.  This lets Measure,
FIT, Correct and their result widgets share one non-native Qt experience
without changing the action-specific loading or saving logic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog as _QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

_ORIGINAL_GET_OPEN = _QFileDialog.__dict__.get("getOpenFileName")
_ORIGINAL_GET_SAVE = _QFileDialog.__dict__.get("getSaveFileName")
_ORIGINAL_GET_DIRECTORY = _QFileDialog.__dict__.get("getExistingDirectory")


def _settings_key(caption: str, mode: str) -> str:
    identity = hashlib.sha1(f"{mode}:{caption}".encode("utf-8")).hexdigest()[:16]
    return f"file_dialogs/{identity}/last_directory"


def _expanded(path: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _initial_location(caption: str, mode: str, supplied: str) -> tuple[Path, str]:
    """Return an existing start directory and an optional filename."""
    supplied = str(supplied or "").strip()
    settings = QSettings("pyLOCO", "pyLOCO")
    remembered = str(settings.value(_settings_key(caption, mode), "") or "")
    if supplied:
        candidate = _expanded(supplied)
        if candidate.is_dir():
            return candidate.resolve(), ""
        parent = candidate.parent
        if parent.is_dir():
            return parent.resolve(), candidate.name
    if remembered:
        candidate = _expanded(remembered)
        if candidate.is_dir():
            return candidate.resolve(), ""
    return Path.cwd().resolve(), Path(supplied).name if supplied else ""


def _default_suffix(name_filter: str) -> str:
    match = re.search(r"\*\.([A-Za-z0-9][A-Za-z0-9._-]*)", name_filter or "")
    return match.group(1) if match else ""


def _build_dialog(parent, caption: str, directory: str, name_filter: str, *,
                  mode: str, options=_QFileDialog.Option(0)) -> _QFileDialog:
    start, filename = _initial_location(caption, mode, directory)
    dialog = _QFileDialog(parent, caption, str(start), name_filter)
    dialog.setOption(_QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOptions(dialog.options() | options | _QFileDialog.Option.DontUseNativeDialog)
    if mode == "open":
        dialog.setAcceptMode(_QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(_QFileDialog.FileMode.ExistingFile)
    elif mode == "save":
        dialog.setAcceptMode(_QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(_QFileDialog.FileMode.AnyFile)
        suffix = _default_suffix(name_filter)
        if suffix:
            dialog.setDefaultSuffix(suffix)
    elif mode == "directory":
        dialog.setAcceptMode(_QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(_QFileDialog.FileMode.Directory)
        dialog.setOption(_QFileDialog.Option.ShowDirsOnly, True)
    if filename and mode != "directory":
        dialog.selectFile(filename)
    # The non-native Qt dialog provides an editable filename/location field.
    # Make its direct-path purpose explicit without replacing Qt's browser.
    for edit in dialog.findChildren(QLineEdit):
        if not edit.placeholderText():
            edit.setPlaceholderText("Type or paste an absolute path")
    # QFileDialog's "Look in" control is not selectable on every platform.
    # Expose the current directory as ordinary text and provide an explicit
    # clipboard action so operators can reuse long scientific-data paths.
    folder_row=QWidget(dialog); folder_row.setObjectName("copyFolderPathRow")
    folder_layout=QHBoxLayout(folder_row); folder_layout.setContentsMargins(0,0,0,0)
    folder_layout.addWidget(QLabel("Current folder",folder_row))
    folder_path=QLineEdit(folder_row); folder_path.setObjectName("currentFolderPath"); folder_path.setReadOnly(True); folder_path.setText(dialog.directory().absolutePath()); folder_path.setToolTip("Current folder path. Select the text or use Copy folder path.")
    copy_folder=QPushButton("Copy folder path",folder_row); copy_folder.setObjectName("copyFolderPathButton")
    copy_folder.clicked.connect(lambda: QApplication.clipboard().setText(folder_path.text()))
    dialog.directoryEntered.connect(folder_path.setText)
    folder_layout.addWidget(folder_path,1); folder_layout.addWidget(copy_folder)
    dialog.layout().addWidget(folder_row,dialog.layout().rowCount(),0,1,dialog.layout().columnCount())
    dialog.setSizeGripEnabled(True)
    dialog.resize(900, 620)
    return dialog


def _accepted_path(dialog: _QFileDialog, caption: str, mode: str) -> str:
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return ""
    selected = dialog.selectedFiles()
    if not selected:
        return ""
    candidate = _expanded(selected[0]).resolve(strict=False)
    if mode == "open" and not candidate.is_file():
        return ""
    if mode == "directory" and not candidate.is_dir():
        return ""
    if mode == "save" and not candidate.parent.is_dir():
        return ""
    remembered = candidate if mode == "directory" else candidate.parent
    QSettings("pyLOCO", "pyLOCO").setValue(_settings_key(caption, mode), str(remembered))
    return str(candidate)


class FileDialog:
    """Compatibility facade for the subset of ``QFileDialog`` used by pyLOCO."""

    Option = _QFileDialog.Option
    DontUseNativeDialog = _QFileDialog.Option.DontUseNativeDialog
    ShowDirsOnly = _QFileDialog.Option.ShowDirsOnly

    @staticmethod
    def getOpenFileName(parent=None, caption="", dir="", filter="", selectedFilter="",
                        options=_QFileDialog.Option(0)):
        # Preserve test/integration interception of Qt's established static
        # API while all normal application calls use the styled dialog below.
        if _QFileDialog.__dict__.get("getOpenFileName") is not _ORIGINAL_GET_OPEN:
            return _QFileDialog.getOpenFileName(
                parent, caption, dir, filter, selectedFilter=selectedFilter, options=options,
            )
        dialog = _build_dialog(parent, caption, dir, filter, mode="open", options=options)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        path = _accepted_path(dialog, caption, "open")
        return path, dialog.selectedNameFilter()

    @staticmethod
    def getSaveFileName(parent=None, caption="", dir="", filter="", selectedFilter="",
                        options=_QFileDialog.Option(0)):
        if _QFileDialog.__dict__.get("getSaveFileName") is not _ORIGINAL_GET_SAVE:
            return _QFileDialog.getSaveFileName(
                parent, caption, dir, filter, selectedFilter=selectedFilter, options=options,
            )
        dialog = _build_dialog(parent, caption, dir, filter, mode="save", options=options)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        path = _accepted_path(dialog, caption, "save")
        return path, dialog.selectedNameFilter()

    @staticmethod
    def getExistingDirectory(parent=None, caption="", dir="", options=_QFileDialog.Option.ShowDirsOnly):
        if _QFileDialog.__dict__.get("getExistingDirectory") is not _ORIGINAL_GET_DIRECTORY:
            return _QFileDialog.getExistingDirectory(parent, caption, dir, options=options)
        dialog = _build_dialog(parent, caption, dir, "", mode="directory", options=options)
        return _accepted_path(dialog, caption, "directory")
