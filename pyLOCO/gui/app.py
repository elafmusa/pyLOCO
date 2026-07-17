"""Application entry point for the pyLOCO GUI shell.

This module owns QApplication creation and intentionally keeps all GUI
startup code separate from the numerical pyLOCO backend. Milestone 1 is
an offline application shell only; it does not execute LOCO fits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


APPLICATION_NAME = "pyLOCO GUI"
ORGANIZATION_NAME = "pyLOCO"
DEBUG_GUI_FLAG = "--debug-gui-startup"
DEBUG_GUI_ENV = "PYLOCO_GUI_DEBUG"
EXPECTED_THEME_COMMIT = "5fc8c73"


def _debug_requested(argv: Sequence[str]) -> bool:
    """Return whether startup diagnostics should be printed."""

    return DEBUG_GUI_FLAG in argv or os.environ.get(DEBUG_GUI_ENV, "").lower() in {"1", "true", "yes", "on"}


def _qt_argv(argv: Sequence[str]) -> list[str]:
    """Remove pyLOCO-only arguments before creating QApplication."""

    return [arg for arg in argv if arg != DEBUG_GUI_FLAG]


def build_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create and configure the Qt application instance.

    Parameters
    ----------
    argv:
        Optional command-line arguments. If omitted, ``sys.argv`` is used.

    Returns
    -------
    QApplication
        Configured application object ready to show windows.
    """

    raw_argv = list(sys.argv if argv is None else argv)
    app = QApplication(_qt_argv(raw_argv))
    app.setApplicationName(APPLICATION_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setProperty("pyLOCOGuiDebugStartup", _debug_requested(raw_argv))
    return app


def _git_output(args: list[str], cwd: Path) -> str:
    """Run a git command for diagnostics and return a printable value."""

    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"<unavailable: {exc}>"


def _candidate_import_locations() -> list[str]:
    """List pyLOCO package directories visible on sys.path."""

    candidates: list[str] = []
    seen: set[str] = set()
    for entry in sys.path:
        root = Path(entry or os.getcwd())
        candidate = (root / "pyLOCO").resolve()
        if (candidate / "__init__.py").exists():
            text = str(candidate)
            if text not in seen:
                seen.add(text)
                candidates.append(text)
    return candidates


def _runtime_diagnostics(window: MainWindow) -> dict[str, object]:
    """Collect import, git, menu, and toolbar diagnostics for startup debugging."""

    import pyLOCO
    import pyLOCO.gui.main_window as main_window_module

    checkout = Path(main_window_module.__file__).resolve().parents[2]
    package_spec = importlib.util.find_spec("pyLOCO")
    main_window_spec = importlib.util.find_spec("pyLOCO.gui.main_window")
    return {
        "pyLOCO_file": str(Path(pyLOCO.__file__).resolve()),
        "pyLOCO_spec_origin": package_spec.origin if package_spec else None,
        "main_window_module_file": str(Path(main_window_module.__file__).resolve()),
        "main_window_spec_origin": main_window_spec.origin if main_window_spec else None,
        "checkout_root": str(checkout),
        "git_head": _git_output(["rev-parse", "HEAD"], checkout),
        f"commit_{EXPECTED_THEME_COMMIT}_present": _git_output(["cat-file", "-e", f"{EXPECTED_THEME_COMMIT}^{{commit}}"], checkout) == "",
        "sys_path_first_entries": sys.path[:8],
        "pyLOCO_candidates_on_sys_path": _candidate_import_locations(),
        "window": window.startup_diagnostics(),
    }


def _print_runtime_diagnostics(window: MainWindow) -> None:
    """Print startup diagnostics in JSON for users debugging stale installs."""

    print("pyLOCO GUI startup diagnostics:")
    print(json.dumps(_runtime_diagnostics(window), indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pyLOCO GUI shell."""

    app = build_application(argv)
    window = MainWindow()
    if app.property("pyLOCOGuiDebugStartup"):
        _print_runtime_diagnostics(window)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())
