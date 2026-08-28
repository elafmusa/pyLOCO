"""Readable and exportable scientific run summary."""

from __future__ import annotations

import json
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget


class RunSummaryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None
        self.text = QTextEdit(); self.text.setReadOnly(True)
        self.export_json = QPushButton("Export summary JSON…")
        self.export_yaml = QPushButton("Export summary YAML…")
        self.export_json.clicked.connect(lambda: self._export("json")); self.export_yaml.clicked.connect(lambda: self._export("yaml"))
        row = QHBoxLayout(); row.addWidget(self.export_json); row.addWidget(self.export_yaml); row.addStretch(1)
        layout = QVBoxLayout(self); layout.addLayout(row); layout.addWidget(self.text)

    def set_loader(self, loader):
        self.loader = loader
        self.text.setPlainText(json.dumps(loader.run_summary, indent=2))

    def _export(self, kind):
        if self.loader is None: return
        filename = QFileDialog.getSaveFileName(self, "Export run summary", f"run_summary.{kind}", f"{kind.upper()} (*.{kind})")[0]
        if not filename: return
        if kind == "json":
            content = json.dumps(self.loader.run_summary, indent=2)
        else:
            try:
                import yaml
                content = yaml.safe_dump(self.loader.run_summary, sort_keys=False)
            except ImportError:
                content = json.dumps(self.loader.run_summary, indent=2)
        with open(filename, "w", encoding="utf-8") as stream: stream.write(content)
