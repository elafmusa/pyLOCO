from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLineEdit, QWidget

from pyLOCO.gui.file_dialogs import FileDialog, _accepted_path, _build_dialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    "mode,file_mode,accept_mode",
    [
        ("open", QFileDialog.FileMode.ExistingFile, QFileDialog.AcceptMode.AcceptOpen),
        ("save", QFileDialog.FileMode.AnyFile, QFileDialog.AcceptMode.AcceptSave),
        ("directory", QFileDialog.FileMode.Directory, QFileDialog.AcceptMode.AcceptOpen),
    ],
)
def test_shared_dialog_is_non_native_theme_inheriting_and_path_editable(app, tmp_path, mode, file_mode, accept_mode):
    parent = QWidget(); parent.setStyleSheet("QWidget { color: rgb(12, 34, 56); }")
    initial = tmp_path / "folder with spaces"; initial.mkdir()
    supplied = initial if mode == "directory" else initial / "café result.json"
    dialog = _build_dialog(parent, "Shared dialog", str(supplied), "JSON (*.json)", mode=mode)
    assert dialog.parent() is parent
    assert dialog.testOption(QFileDialog.Option.DontUseNativeDialog)
    assert dialog.fileMode() == file_mode
    assert dialog.acceptMode() == accept_mode
    assert Path(dialog.directory().absolutePath()).resolve() == initial.resolve()
    filename_fields = [field for field in dialog.findChildren(QLineEdit) if field.objectName() == "fileNameEdit"]
    assert filename_fields and "absolute path" in filename_fields[0].placeholderText()
    if mode != "directory":
        assert Path(dialog.selectedFiles()[0]).name == "café result.json"
    dialog.close(); parent.close()


class _Selection:
    def __init__(self, path, accepted=True):
        self.path, self.accepted = str(path), accepted

    def exec(self):
        return QDialog.DialogCode.Accepted if self.accepted else QDialog.DialogCode.Rejected

    def selectedFiles(self):
        return [self.path]


class _Settings:
    def __init__(self, *_args): pass
    def setValue(self, *_args): pass


def test_open_validates_existing_file_and_cancel_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("pyLOCO.gui.file_dialogs.QSettings", _Settings)
    existing = tmp_path / "lattice with spaces.mat"; existing.write_bytes(b"AT")
    assert _accepted_path(_Selection(existing), "Open lattice", "open") == str(existing.resolve())
    assert _accepted_path(_Selection(tmp_path / "missing.mat"), "Open lattice", "open") == ""
    assert _accepted_path(_Selection(existing, accepted=False), "Open lattice", "open") == ""


def test_save_accepts_new_unicode_filename_but_requires_existing_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("pyLOCO.gui.file_dialogs.QSettings", _Settings)
    target = tmp_path / "résultat project.json"
    assert _accepted_path(_Selection(target), "Save project", "save") == str(target.resolve())
    assert _accepted_path(_Selection(tmp_path / "missing" / "result.json"), "Save project", "save") == ""


def test_facade_retains_qfiledialog_options_and_static_api():
    assert FileDialog.DontUseNativeDialog == QFileDialog.Option.DontUseNativeDialog
    assert FileDialog.ShowDirsOnly == QFileDialog.Option.ShowDirsOnly
    assert callable(FileDialog.getOpenFileName)
    assert callable(FileDialog.getSaveFileName)
    assert callable(FileDialog.getExistingDirectory)
