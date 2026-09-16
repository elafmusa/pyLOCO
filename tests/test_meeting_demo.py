from __future__ import annotations

from pathlib import Path

from pyLOCO.correct.model import load_review
from pyLOCO.gui.backend import LocoRunRequest
from pyLOCO.gui.models.project import ProjectMetadata
from pyLOCO.gui.results.results_loader import ResultsLoader
from pyLOCO.gui.results.optics_view import OpticsView


DEMO = Path(__file__).resolve().parents[1] / "Examples" / "Demo"


def test_live_demo_is_portable_valid_and_one_iteration():
    path = DEMO / "01_PETRAIII_Fit_1_Iteration_Live.pyloco.json"
    text = path.read_text(encoding="utf-8")
    assert not any(value in text for value in ("/Users/", "/home/", "/private/tmp/", "Documents/Codex"))
    project = ProjectMetadata.load(path)
    assert project.loco_config.solver.nIter == 1
    assert project.validation_messages() == []
    request = LocoRunRequest.from_project(project)
    assert Path(request.lattice_path).is_file()
    assert all(Path(value).is_file() for value in request.measurements.values())


def test_completed_demo_restores_real_eight_iteration_artifacts():
    project_path = DEMO / "02_PETRAIII_Fit_8_Iterations_Completed.pyloco.json"
    project = ProjectMetadata.load(project_path)
    loader = ResultsLoader(project.resolve_path(project.completed_run.results_dir))
    assert len(loader.chi2_history) == 8
    assert [entry["iteration"] for entry in loader.iteration_entries] == list(range(9))
    assert loader.measured_orm.shape == loader.fitted_orm.shape == (470, 413)
    assert loader.initial_orm.shape == (470, 413)
    assert loader.final_chi2 == 5.807435939033083
    assert len(loader.quadrupole_parameter_rows) == 398
    assert all(row["initial"] is not None and row["delta_k"] is not None for row in loader.quadrupole_parameter_rows)
    assert loader.beta_beating_data["beta_x_fitted"].size == 7675
    assert loader.beta_beating_data["beta_y_fitted"].size == 7675
    assert loader.dispersion_data["x"]["fitted"].size == 235
    assert loader.dispersion_data["y"]["fitted"].size == 235
    blocks = loader.fitted_parameter_blocks
    assert blocks["hbpm_coupling"].stop - blocks["hbpm_coupling"].start == 235
    assert blocks["vbpm_coupling"].stop - blocks["vbpm_coupling"].start == 235
    assert blocks["hcor_coupling"].stop - blocks["hcor_coupling"].start == 219
    assert blocks["vcor_coupling"].stop - blocks["vcor_coupling"].start == 194
    assert loader.jacobian_available is False


def test_completed_demo_renders_native_beta_and_dispersion_plots():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    project = ProjectMetadata.load(
        DEMO / "02_PETRAIII_Fit_8_Iterations_Completed.pyloco.json"
    )
    result_dir = project.resolve_path(project.completed_run.results_dir)
    view = OpticsView()
    view.resize(1000, 400)
    view.show()
    view.set_loader(ResultsLoader(result_dir, iteration=8))
    app.processEvents()
    assert view.beta_plots["x"]["curves"].figure.axes[0].lines
    assert view.beta_plots["y"]["beating"].figure.axes[0].lines
    assert view.dispersion_plots["x"]["comparison"].figure.axes[0].lines
    assert view.beta_plots["x"]["curves"].canvas.height() >= 80
    view.close()
    app.processEvents()


def test_individual_correction_changes_only_selected_lattice_element():
    from types import SimpleNamespace

    import numpy as np

    from pyLOCO.set_parameters import set_correction

    ring = [
        SimpleNamespace(CommonName="QF", PolynomB=np.array([0.0, 1.0, 0.0])),
        SimpleNamespace(CommonName="QF", PolynomB=np.array([0.0, 1.0, 0.0])),
    ]
    set_correction(ring, [1.25], [1], individuals=True)
    assert ring[0].PolynomB[1] == 1.0
    assert ring[1].PolynomB[1] == 1.25


def test_correction_demo_contains_real_expanded_final_corrections():
    plan = DEMO / "03_PETRAIII_Correction_Review.pyloco-correction.json"
    text = plan.read_text(encoding="utf-8")
    assert not any(value in text for value in ("/Users/", "/home/", "/private/tmp/", "Documents/Codex"))
    review = load_review(plan)
    assert len(review.items) == 398
    assert all(item.correction_type == "normal_quadrupole" for item in review.items)
    assert all(abs(item.raw_fitted_delta + item.recommended_machine_delta) < 1e-14 for item in review.items)
    assert review.to_plan().metadata["safety"].startswith("OFFLINE DRY RUN")
