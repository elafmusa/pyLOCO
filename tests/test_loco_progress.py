from types import SimpleNamespace

import numpy as np
import pytest

from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import _WorkflowProgressReporter, pyloco
from pyLOCO.response_matrix import response_matrix


def test_progress_reporter_is_monotonic_bounded_and_supports_arbitrary_iterations():
    events = []
    reporter = _WorkflowProgressReporter(events.append, 7)
    reporter.emit("initialization", 0.0, "Initializing")
    for iteration in range(1, 8):
        reporter.emit(
            "model_calculation",
            reporter.iteration_fraction(iteration, 0.0),
            "Model",
            iteration=iteration,
        )
        reporter.emit(
            "iteration_complete",
            reporter.iteration_fraction(iteration, 1.0),
            "Complete",
            iteration=iteration,
        )
    reporter.emit("final_results", 0.97, "Final results", iteration=7)
    reporter.emit("completed", 1.0, "LOCO completed successfully", iteration=7)

    fractions = [event["workflow_fraction"] for event in events]
    assert fractions == sorted(fractions)
    assert all(0.0 <= value <= 1.0 for value in fractions)
    assert fractions[-1] == 1.0
    assert [event["iteration"] for event in events if event["phase"] == "iteration_complete"] == list(range(1, 8))
    assert all(event["total_iterations"] == 7 for event in events)


def test_successful_pyloco_reaches_one_only_on_completed_event(tmp_path):
    at = pytest.importorskip("at")
    drift = at.Drift("D", 0.5)
    bend = at.Dipole("B", 1.0, 2 * np.pi / 20)
    cell = at.Lattice([
        drift, bend, at.Monitor("BPM"),
        at.Corrector("HC", 0.0, [0.0, 0.0]),
        at.Corrector("VC", 0.0, [0.0, 0.0]),
        at.Quadrupole("QF", 0.4, 1.0), drift,
        at.Quadrupole("QD", 0.4, -1.0),
    ], energy=1e9)
    ring = cell * 20
    bpm = np.asarray(at.get_refpts(ring, "BPM"), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HC"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VC"), dtype=int)
    steps = [np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5)]
    measured = response_matrix(ring.deepcopy(), config=RMConfig(
        bpm_ords=bpm, cm_ords=[hcor, vcor], dkick=steps,
        calculator="Linear", includeDispersion=False, fixedpathlength=False,
    ))
    events = []

    pyloco(
        ring,
        algorithm="lm",
        nIter=2,
        nLMIter=1,
        used_bpms_ords=bpm,
        used_cor_ords=[hcor, vcor],
        quads_ords=[],
        skew_ords=[],
        CAVords=[],
        quads_tilt_ind=[],
        nHBPM=len(bpm), nVBPM=len(bpm),
        nHorCOR=len(hcor), nVerCOR=len(vcor),
        orm_measured=measured,
        weights=np.ones((2 * len(bpm), 1)),
        measured_eta_x=np.zeros(len(bpm)), measured_eta_y=np.zeros(len(bpm)),
        CMstep=steps, rfStep=-3000.0,
        fit_list=("hbpm_gain", "vbpm_gain"),
        fit_cfg=FitInitConfig(fit_list=("hbpm_gain", "vbpm_gain"), CMstep=steps),
        remove_coupling_=False,
        fixedpathlength=False,
        output_dir=tmp_path,
        progress_callback=events.append,
    )

    fractions = [event["workflow_fraction"] for event in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert [event["phase"] for event in events if event["workflow_fraction"] == 1.0] == ["completed"]
    assert [event["iteration"] for event in events if event["phase"] == "iteration_complete"] == [1, 2]


def test_pyloco_failure_never_reports_success():
    events = []
    with pytest.raises(Exception):
        pyloco(None, nIter=3, progress_callback=events.append)
    assert events
    assert max(event["workflow_fraction"] for event in events) < 1.0
    assert all(event["phase"] != "completed" for event in events)


def test_pyloco_cancellation_never_reports_success():
    events = []
    with pytest.raises(RuntimeError, match="cancelled"):
        pyloco(
            None,
            nIter=5,
            progress_callback=events.append,
            cancel_callback=lambda: True,
        )
    assert events[-1]["workflow_fraction"] < 1.0
    assert all(event["phase"] != "completed" for event in events)


def test_progress_callback_is_optional_for_existing_callers():
    with pytest.raises(Exception) as without_callback:
        pyloco(None, nIter=1)
    with pytest.raises(type(without_callback.value)):
        pyloco(None, nIter=1, progress_callback=None)


def _qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_gui_worker_progress_signal_updates_workspace(monkeypatch):
    from PySide6.QtCore import QEventLoop, QThread
    from pyLOCO.gui import main_window
    from pyLOCO.gui.main_window import LocoRunWorker
    from pyLOCO.gui.results.results_workspace import ResultsWorkspace

    result = SimpleNamespace(results_dir="unused", elapsed_seconds=1.0, output_files=[])

    def fake_run(_request, **kwargs):
        kwargs["progress_callback"]({
            "phase": "jacobian_calculation",
            "iteration": 3,
            "total_iterations": 4,
            "workflow_fraction": 0.72,
            "message": "Computing analytical quadrupole Jacobian.",
        })
        return result

    monkeypatch.setattr(main_window, "run_loco_request", fake_run)
    app = _qt_app()
    workspace = ResultsWorkspace()
    workspace.begin_run()
    worker = LocoRunWorker(SimpleNamespace(backend_mapping={"LOCOOptions": {}}))
    thread = QThread()
    loop = QEventLoop()
    worker.moveToThread(thread)
    worker.progress.connect(workspace.update_progress)
    worker.finished.connect(loop.quit)
    worker.failed.connect(loop.quit)
    thread.started.connect(worker.run)
    thread.start()
    loop.exec()
    thread.quit()
    thread.wait()
    app.processEvents()

    assert workspace.run_progress.value() == 72
    assert workspace.run_iteration_label.text() == "Iteration 3 of 4"
    assert workspace.run_status_label.text() == "Computing analytical quadrupole Jacobian."
    workspace.close()


def test_gui_failure_and_cancellation_retain_last_progress():
    from pyLOCO.gui.results.results_workspace import ResultsWorkspace

    app = _qt_app()
    for cancelled, expected in ((False, "Failed"), (True, "Cancelled")):
        workspace = ResultsWorkspace()
        workspace.begin_run()
        workspace.update_progress({
            "iteration": 2,
            "total_iterations": 6,
            "workflow_fraction": 0.41,
            "message": "Computing numerical Jacobian.",
        })
        workspace.fail_run(cancelled=cancelled)
        assert workspace.run_progress.value() == 41
        assert workspace.run_status_label.text() == expected
        workspace.close()
    app.processEvents()
