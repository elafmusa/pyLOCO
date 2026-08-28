"""Plane-separated beta and dispersion presentation for completed runs."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from .plot_canvas import PlotCanvas


def _table(headers):
    table = QTableWidget(0, len(headers)); table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers); table.setSelectionMode(QAbstractItemView.NoSelection)
    table.verticalHeader().hide(); table.horizontalHeader().setStretchLastSection(True)
    return table


def _finish_table(table):
    table.resizeColumnsToContents()
    table.setFixedHeight(min(190, table.verticalHeader().length() + table.horizontalHeader().height() + 6))


class OpticsView(QWidget):
    """Large independent views for βx, βy, ηx and ηy."""

    def __init__(self, parent=None):
        super().__init__(parent); self.loader = None; self.tabs = QTabWidget()
        self.beta_pages = {}; self.beta_messages = {}; self.beta_tables = {}; self.beta_plots = {}
        self.dispersion_pages = {}; self.dispersion_messages = {}; self.dispersion_tables = {}; self.dispersion_plots = {}
        for plane in ("x", "y"):
            page, message, table, plots = self._make_beta_page(plane)
            self.beta_pages[plane], self.beta_messages[plane], self.beta_tables[plane], self.beta_plots[plane] = page, message, table, plots
            self.tabs.addTab(page, f"β{plane}")
        for plane in ("x", "y"):
            page, message, table, plots = self._make_dispersion_page(plane)
            self.dispersion_pages[plane], self.dispersion_messages[plane], self.dispersion_tables[plane], self.dispersion_plots[plane] = page, message, table, plots
            self.tabs.addTab(page, f"η{plane}")
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(self.tabs, 1)
        self.beta_plot = self.beta_plots["x"]["beating"]
        self.dispersion_plot = self.dispersion_plots["x"]["comparison"]
        self.plot = self.beta_plot

    @staticmethod
    def _make_beta_page(plane):
        page = QWidget(); layout = QVBoxLayout(page)
        message = QLabel(f"No β{plane} results loaded."); message.setWordWrap(True); message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        table = _table(["State", "Min [%]", "Max [%]", "Mean [%]", "RMS [%]", "Max |error| [%]"])
        plot_tabs = QTabWidget(); plots = {
            "curves": PlotCanvas(show_toolbar=True, minimum_height=420),
            "beating": PlotCanvas(show_toolbar=True, minimum_height=420),
        }
        plot_tabs.addTab(plots["curves"], f"β{plane} curves"); plot_tabs.addTab(plots["beating"], f"β{plane} beating")
        layout.addWidget(message); layout.addWidget(table); layout.addWidget(plot_tabs, 1)
        return page, message, table, plots

    @staticmethod
    def _make_dispersion_page(plane):
        page = QWidget(); layout = QVBoxLayout(page)
        message = QLabel(f"No η{plane} results loaded."); message.setWordWrap(True); message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        table = _table(["State", "Min [mm]", "Max [mm]", "Mean [mm]", "RMS [mm]", "Max |error| [mm]"])
        plot_tabs = QTabWidget(); plots = {}
        for key, label in (("comparison", "Comparison"), ("measured", "Measured"), ("initial", "Initial model"), ("fitted", "Fitted model"),
                           ("initial_residual", "Initial residual"), ("fitted_residual", "Fitted residual"), ("residuals", "Residual comparison")):
            plots[key] = PlotCanvas(show_toolbar=True, minimum_height=420); plot_tabs.addTab(plots[key], label)
        layout.addWidget(message); layout.addWidget(table); layout.addWidget(plot_tabs, 1)
        return page, message, table, plots

    def set_loader(self, loader):
        self.loader = loader
        for plane in ("x", "y"):
            self._render_beta(loader, plane); self._render_dispersion(loader, plane)

    def _render_beta(self, loader, plane):
        data = loader.beta_beating_data; message = self.beta_messages[plane]; table = self.beta_tables[plane]; plots = self.beta_plots[plane]
        for plot in plots.values(): plot.clear()
        table.setRowCount(0)
        if not data:
            message.setText(f"β{plane} is not available for this run. The required reference/initial/fitted Twiss arrays were not persisted.")
            table.hide()
            return
        table.show(); reference_kind = data.get("reference_kind", "run_input_lattice")
        message.setText(f"β{plane} uses saved longitudinal position s [m]. Reference: " + ("the resumed fitted lattice before this run's corrections." if reference_kind == "resumed_fitted_lattice" else "the input lattice at the start of this run."))
        axis = np.asarray(data["s"], dtype=float); curve_axis = plots["curves"].figure.add_subplot(111); curve_count = 0
        for state, label, style in (("reference", "Reference", ":"), ("initial", "Initial", "-."), ("fitted", "Fitted", "-")):
            values = data.get(f"beta_{plane}_{state}")
            if values is not None:
                curve_axis.plot(axis, np.asarray(values, dtype=float), style, label=label, linewidth=1.2); curve_count += 1
        curve_axis.set(xlabel="Longitudinal position s [m]", ylabel=f"β{plane} [m]"); curve_axis.grid(True, alpha=.25)
        if curve_count: curve_axis.legend()
        else: curve_axis.text(.5, .5, "β curves were not saved for this run.", ha="center", va="center", transform=curve_axis.transAxes)
        beating_axis = plots["beating"].figure.add_subplot(111); beating_count = 0
        for key, label, style in ((f"beta_beating_{plane}_initial", "Initial β beating", ":"), (f"beta_beating_{plane}", "Fitted β beating", "-")):
            values = data.get(key)
            if values is not None:
                beating_axis.plot(axis, 100.0 * np.asarray(values, dtype=float), style, label=label, linewidth=1.2); beating_count += 1
        beating_axis.axhline(0, color="#8d95a8", linewidth=.8); beating_axis.set(xlabel="Longitudinal position s [m]", ylabel=f"β{plane} beating [%]"); beating_axis.grid(True, alpha=.25)
        if beating_count: beating_axis.legend()
        else: beating_axis.text(.5, .5, "Beta-beating arrays were not saved for this run.", ha="center", va="center", transform=beating_axis.transAxes)
        stats = loader.beta_beating_statistics.get(plane, {})
        for state in ("initial", "fitted"):
            values = stats.get(state)
            if values is None: continue
            row = table.rowCount(); table.insertRow(row); table.setItem(row, 0, QTableWidgetItem(state.title()))
            for column, key in enumerate(("min", "max", "mean", "rms", "max_abs"), 1):
                value = values.get(key); table.setItem(row, column, QTableWidgetItem("—" if value is None else f"{value:.5g}"))
        _finish_table(table)
        for plot in plots.values(): plot.apply_theme(); plot.canvas.draw_idle()

    def _render_dispersion(self, loader, plane):
        data = loader.dispersion_data; message = self.dispersion_messages[plane]; table = self.dispersion_tables[plane]; plots = self.dispersion_plots[plane]
        for plot in plots.values(): plot.clear()
        table.setRowCount(0)
        if not data or plane not in data:
            objective = "included in" if loader.dispersion_included else "not included in"
            message.setText(f"η{plane} was {objective} the LOCO objective. {loader.dispersion_unavailable_reason or 'Dispersion arrays are not available for this run.'}")
            table.hide()
            return
        table.show(); message.setText(("Dispersion was included in the LOCO objective. " if loader.dispersion_included else "Dispersion was not included in the LOCO objective; this is an independent post-fit diagnostic. ") + "Displayed dispersion and model − measurement residuals use mm.")
        values = data[plane]; axis = np.asarray(data.get("axis", np.arange(np.size(values["measured"]))), dtype=float); axis_label = data.get("axis_label", "BPM index in saved ordering")
        measured = np.asarray(values["measured"], dtype=float) * 1000.0; initial = np.asarray(values["initial"], dtype=float) * 1000.0; fitted = np.asarray(values["fitted"], dtype=float) * 1000.0
        comparison = plots["comparison"].figure.add_subplot(111)
        for curve, label, style in ((measured, "Measured", "-"), (initial, "Initial model", ":"), (fitted, "Fitted model", "--")):
            comparison.plot(axis, curve, style, label=label, linewidth=1.2)
        comparison.legend(ncols=3)
        for key, curve, label in (("measured", measured, "Measured"), ("initial", initial, "Initial model"), ("fitted", fitted, "Fitted model")):
            ax = plots[key].figure.add_subplot(111); ax.plot(axis, curve, linewidth=1.2, label=label); ax.legend()
        residual_axis = plots["residuals"].figure.add_subplot(111)
        residual_axis.plot(axis, initial - measured, ":", label="Initial model − measurement", linewidth=1.2)
        residual_axis.plot(axis, fitted - measured, "-", label="Fitted model − measurement", linewidth=1.2)
        residual_axis.axhline(0, color="#8d95a8", linewidth=.8); residual_axis.legend()
        for key, residual, label in (("initial_residual", initial - measured, "Initial model − measurement"),
                                     ("fitted_residual", fitted - measured, "Fitted model − measurement")):
            ax = plots[key].figure.add_subplot(111); ax.plot(axis, residual, linewidth=1.2, label=label)
            ax.axhline(0, color="#8d95a8", linewidth=.8); ax.legend()
        for key, plot in plots.items():
            ax = plot.figure.axes[0]; ax.set(xlabel=axis_label, ylabel=(f"η{plane} residual [mm]" if "residual" in key else f"η{plane} [mm]")); ax.grid(True, alpha=.25)
        stats = loader.dispersion_statistics.get(plane, {})
        for state, suffix in (("Initial − measurement", "before"), ("Fit − measurement", "after")):
            row = table.rowCount(); table.insertRow(row); table.setItem(row, 0, QTableWidgetItem(state))
            for column, key in enumerate(("min", "max", "mean", "rms", "max_abs"), 1):
                value = stats.get(f"{key}_{suffix}"); table.setItem(row, column, QTableWidgetItem("—" if value is None else f"{1000.0 * value:.5g}"))
        _finish_table(table)
        for plot in plots.values(): plot.apply_theme(); plot.canvas.draw_idle()

    def apply_theme(self):
        for collection in (self.beta_plots, self.dispersion_plots):
            for plots in collection.values():
                for plot in plots.values(): plot.apply_theme()
