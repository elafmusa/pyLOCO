"""Presentation helpers for identity-mapping tables.

This module deliberately contains no matching or validation logic.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QAbstractItemView, QHeaderView, QTableWidget


IDENTITY_MATCH_LABELS = {
    "exact_device": "Exact AT Device",
    "exact_fam_name": "Exact AT FamName",
    "legacy_component_device": "Legacy suffix → exact AT Device",
    "legacy_component_fam_name": "Legacy suffix → exact AT FamName",
    "explicit_lattice_ordinal": "Explicit AT index",
}


def identity_match_label(value: object) -> str:
    key = str(value or "")
    return IDENTITY_MATCH_LABELS.get(key, key.replace("_", " ").strip().capitalize() or "Not available")


class CopyableMappingTable(QTableWidget):
    """Detailed table with row selection, scrolling and TSV clipboard copy."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            rows = sorted({index.row() for index in self.selectedIndexes()})
            if rows:
                headers = [self.horizontalHeaderItem(column).text() for column in range(self.columnCount())]
                lines = ["\t".join(headers)]
                for row in rows:
                    lines.append("\t".join(
                        self.item(row, column).text() if self.item(row, column) is not None else ""
                        for column in range(self.columnCount())
                    ))
                QApplication.clipboard().setText("\n".join(lines))
                event.accept()
                return
        super().keyPressEvent(event)


def configure_mapping_table(table: QTableWidget, header_help: tuple[str, ...], *, editable: bool = False) -> None:
    table.setEditTriggers(
        QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        if editable else QAbstractItemView.NoEditTriggers
    )
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.verticalScrollBar().setSingleStep(36)
    table.setMouseTracking(True)
    table.setFocusPolicy(Qt.StrongFocus)
    # Callers must populate the whole table before enabling sorting. Qt can
    # relocate a partially populated row as soon as the value in the active
    # sort column is inserted, which mixes cells from different records.
    table.setSortingEnabled(False)
    # Resize-to-contents makes long identities effectively impossible to
    # inspect on laptop displays.  Start with useful widths, then leave every
    # section under direct operator control.
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table.horizontalHeader().setMinimumSectionSize(80)
    table.horizontalHeader().setStretchLastSection(False)
    table.resizeColumnsToContents()
    for column in range(table.columnCount()):
        table.setColumnWidth(column, min(max(table.columnWidth(column), 105), 360))
    for column, help_text in enumerate(header_help):
        header = table.horizontalHeaderItem(column)
        if header is not None:
            header.setToolTip(help_text)
