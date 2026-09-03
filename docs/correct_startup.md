# Correct startup and shared appearance — 2026-09-03

Scope: empty application startup and suite shell only. Correction calculations,
mapping/calibration rules, backend writes and safety gates are unchanged. FIT
and Measure application source files were not edited.

Checkpoint packaging note: the isolated commit includes only the Correct launch
button/menu and companion-window handler in the suite shell. Unrelated local
FIT/session-import/dashboard changes are excluded. No FIT calculations or
Measure sources are changed. Shared native-font/theme handling is included.

## Measurement method

`Examples/Correct/profile_startup.py` times fresh processes with `perf_counter`.
`suite_correct` first opens the FIT window, processes its initial events, then
times the same `MainWindow.open_correct_app()` handler used by the Correct action.
The import is timed separately immediately before calling the handler. No result
source is loaded, no machine is contacted, and windows are closed by the probe.
The final timestamp is a zero-delay callback in the native Qt event loop: a
responsiveness proxy, not an OS-level mouse-click or compositor-paint measurement.
Warm filesystem/font caches and normal scheduling introduce run-to-run variation.

Run from the repository root:

```sh
PYTHONPATH=. .venv/bin/python Examples/Correct/profile_startup.py suite_correct
PYTHONPATH=. .venv/bin/python Examples/Correct/profile_startup.py correct
PYTHONPATH=. .venv/bin/python Examples/Correct/profile_startup.py measure
PYTHONPATH=. .venv/bin/python Examples/Correct/profile_startup.py fit
```

Set `QT_QPA_PLATFORM=offscreen` for CI; the numbers below were measured with native
macOS windows, not offscreen rendering. No splash screen was introduced.

## Observed timings

Seconds from handler start (suite) or module import start (fresh applications):

| Startup | Before | After |
|---|---:|---:|
| FIT → empty Correct, first event-loop callback | 1.292 | 0.249 |
| Fresh Correct, first event-loop callback | 0.862 | 0.594 |
| Fresh Measure, comparison (unchanged) | 1.058 | not modified |
| Fresh FIT, comparison (unchanged) | 1.102 | not modified |

The suite-to-Correct path improved about 81%; direct Correct improved about 31%
in these representative runs. This is not a cold-boot benchmark or a guaranteed
latency bound.

Detailed suite-to-Correct trace (milliseconds; the finer post-fix rerun was 259 ms):

| Milestone / duration | Before | After |
|---|---:|---:|
| Import complete | 21.3 | 19.7 |
| QApplication | already running | already running |
| Correct constructor begins | approximately 21 | 20.0 |
| Registry/model state ready, UI begins | 21.7 | 33.3 |
| Registry constructor duration | negligible; no connection | 0.102 |
| UI construction complete | 137.4 | 134.3 |
| Correct constructor complete | included in handler return | 134.4 |
| First show returned | included in handler return | 180.2 |
| Handler returned / window shown | 1210.8 | 181.4 |
| First responsive event-loop callback | 1291.6 | 258.7 |

Before the change, Correct unconditionally applied Light to the shared QApplication
and then reapplied the application stylesheet with its amber overrides. Most of
the delay after UI construction was spent in this construction-time appearance
reset, which also restyled FIT. Correct now leaves an existing suite theme alone
and applies amber accents only to its own window.

Six empty Matplotlib canvases were also built eagerly. The Review-page construction
cost was 238 ms on the fresh native baseline, compared with 9 ms for placeholders
after the fix. Canvases are created by `_ensure_plots()` only once correction data
is supplied. Existing plot calculations/rendering after loading are unchanged.

## Import / initialization audit

- No AT/pyAT, SciPy, or `pyLOCO.pyloco` was imported in either empty startup path.
- Matplotlib was imported by empty Correct before; it is absent in a fresh empty
  Correct process after. In-process FIT already has Matplotlib loaded.
- Lightweight correction model definitions, NumPy, YAML and schema/HDF5 support
  remain imported. No `CorrectionReview` or machine snapshot is constructed.
- `InterfaceRegistry` stores factories/profile key and resolves the repository
  root; it does not enumerate repositories, load a machine lattice, connect, or
  read mapping/calibration files on construction.
- No correction-result discovery or directory scan occurs in empty startup.
  Mapping, calibration and readback remain explicit user actions.
- FIT's existing selected-result handoff is unchanged: if the user opens Correct
  from an active result, the source load is queued after showing the window. This
  report benchmarks empty startup, not potentially expensive requested data loads.

## Shared appearance

`pyLOCO.gui.appearance` accesses the existing FIT namespace:
`QSettings("pyLOCO", "pyLOCO GUI")`, key `appearance/theme`.

- Existing QApplication appearance wins when Correct opens from FIT.
- Direct Correct loads the persisted FIT preference (default Dark if absent).
- Correct does not change an existing QApplication's name or organization.
- Toggle actions persist to that same namespace. When FIT is present, the helper
  uses its existing appearance action to keep menus/current theme/plots in sync.
- Correct watches the shared plot-theme property so FIT-originated toggles update
  its button and existing canvases too. No Correct-specific preference is created.
- Correct's amber accent stylesheet is window-local, never appended to the
  application's stylesheet.

## Shell audit and validation

The shared icon is now explicitly assigned to the Correct window; default size is
capped to available screen geometry. Existing logo/About behavior, status bar,
four Correction pages, workflow, table scrolling and suite launch/handoff are
preserved. No FIT/Measure redesign or new machine functionality was introduced.

`tests/test_correct_startup.py` verifies Dark/Light inheritance through the actual
FIT handler, reciprocal theme toggles, close/reopen, isolated persisted preferences
in fresh direct processes, absence of heavy imports, lazy canvases and empty
backend state. All six cases passed both offscreen and with native macOS Qt.
Existing Correction/read-only/shared-suite/About regression tests are also run.
The old detached-Correct expectation in one suite test was updated to the existing
in-process handler; no application launch behavior was changed to satisfy it.

Final results: **44 passed** across startup, Correct dry-run, PETRA read-only,
suite integration and About-dialog regressions (83.36 s). The six startup/theme
cases also passed separately on native macOS Qt (9.05 s), including two consecutive
fresh processes verifying persisted toggles. Compilation sanity checks passed.
