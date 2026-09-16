"""Readable and exportable scientific run summary."""

from __future__ import annotations

import json
import math
from html import escape
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
        self.text.setHtml(self._summary_html(loader.run_summary))

    @staticmethod
    def _value(value, *, unit=""):
        if value is None:return "Not available"
        if isinstance(value,bool):return "Yes" if value else "No"
        if isinstance(value,float):
            if not math.isfinite(value):return "Not available"
            text=f"{value:.6g}"
        else:text=str(value)
        return escape(text)+(f" {unit}" if unit else "")

    @classmethod
    def _rows(cls, rows):
        return "".join(
            f"<tr><td width='230'><b>{escape(str(label))}</b></td><td>{cls._value(value,unit=unit)}</td></tr>"
            for label,value,unit in rows
        )

    @classmethod
    def _flatten_statistics(cls, title, values):
        if not isinstance(values,dict) or not values:return []
        rows=[]
        for key,value in values.items():
            label=f"{title} — {str(key).replace('_',' ')}"
            if isinstance(value,dict):
                rows.extend(cls._flatten_statistics(label,value))
            else:rows.append((label,value,""))
        return rows

    @classmethod
    def _summary_html(cls, summary):
        project=summary.get("project_name") or "Unnamed project"
        run=summary.get("run_name") or "Unnamed run"
        measurements=summary.get("measurement_files") or {}
        measurement_rows=[(str(role).replace("_"," ").title(),path,"") for role,path in measurements.items()]
        fit_names=[str(name).replace("_"," ") for name in summary.get("fit_list") or []]
        initial=summary.get("initial_chi2"); final=summary.get("final_chi2")
        reduction=None
        if isinstance(initial,(int,float)) and isinstance(final,(int,float)) and initial:
            reduction=100.0*(initial-final)/initial
        result_rows=[
            ("Initial χ²",initial,""),("Final χ²",final,""),("χ² reduction",reduction,"%"),
            ("Initial ORM RMS",summary.get("initial_orm_rms"),"m"),
            ("Fitted ORM RMS",summary.get("fitted_orm_rms"),"m"),
            ("Runtime",summary.get("runtime_seconds"),"s"),
        ]
        result_rows.extend(cls._flatten_statistics("Dispersion",summary.get("dispersion_statistics")))
        result_rows.extend(cls._flatten_statistics("Beta beating",summary.get("beta_beating_statistics_percent")))
        warnings=summary.get("warnings") or []
        warning_html=(
            "<h3 style='color:#F59E0B'>Warnings</h3><ul>"+
            "".join(f"<li>{escape(str(value))}</li>" for value in warnings)+"</ul>"
            if warnings else "<div style='color:#22C55E;font-weight:700;margin-top:12px'>✓ No saved run warnings</div>"
        )
        return (
            f"<h2 style='margin:0;color:#A98BFF'>{escape(str(project))}</h2>"
            f"<div style='font-size:14pt;margin:3px 0 14px 0'>{escape(str(run))}</div>"
            "<h3 style='color:#A98BFF'>Run overview</h3>"
            f"<table cellspacing='0' cellpadding='5' width='100%'>{cls._rows([('Reference lattice',summary.get('lattice_file'),''),('Dispersion included',summary.get('dispersion_enabled'),''),('Horizontal dispersion weight',summary.get('horizontal_dispersion_weight'),''),('Vertical dispersion weight',summary.get('vertical_dispersion_weight'),'')])}</table>"
            "<h3 style='color:#A98BFF'>Measurement inputs</h3>"
            f"<table cellspacing='0' cellpadding='5' width='100%'>{cls._rows(measurement_rows) if measurement_rows else cls._rows([('Files','Not available','')])}</table>"
            "<h3 style='color:#A98BFF'>Fitted parameter classes</h3>"
            f"<div style='line-height:1.6'>{escape(', '.join(fit_names) if fit_names else 'None')}</div>"
            "<h3 style='color:#A98BFF'>Fit results</h3>"
            f"<table cellspacing='0' cellpadding='5' width='100%'>{cls._rows(result_rows)}</table>"
            +warning_html
        )

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
