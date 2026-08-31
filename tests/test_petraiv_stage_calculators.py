import ast
from pathlib import Path

import numpy as np
import pytest


PETRAIV_DIR = Path(__file__).resolve().parents[1] / "Examples" / "PETRAIV"


@pytest.fixture(scope="module")
def driver():
    source_path = PETRAIV_DIR / "improve_pyLOCO_latest.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {
        "benchmark_stage_response_matrix_calculators",
        "run_latest_pyloco_two_stage",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace = {"np": np, "Path": Path}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.isupper():
            try:
                namespace[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    namespace["FitInitConfig"] = lambda **kwargs: kwargs
    namespace["save_json"] = lambda *_args, **_kwargs: None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("Linear", ("Linear", "Linear")),
        ("Analytical", ("Analytical", "Linear")),
        ("Tracking", ("Tracking", "Tracking")),
    ],
)
def test_benchmark_stage_orm_selection(driver, requested, expected):
    assert driver["benchmark_stage_response_matrix_calculators"](requested) == expected


def test_two_stage_wrapper_passes_independent_orm_calculators(driver, monkeypatch, tmp_path):
    calls = []

    class Ring:
        def deepcopy(self):
            return self

        def disable_6d(self):
            pass

    ring = Ring()

    def fake_pyloco(model_ring, **kwargs):
        calls.append(kwargs)
        return ([], {}, model_ring, np.zeros((2, 2)), np.eye(2), [], [], {})

    monkeypatch.setitem(driver, "pyloco", fake_pyloco)
    monkeypatch.setitem(driver, "make_constraint_config", lambda *_args: object())
    monkeypatch.setitem(driver, "save_stage_returns", lambda *_args, **_kwargs: None)

    driver["run_latest_pyloco_two_stage"](
        model_ring=ring,
        CMstep=[np.asarray([1.0]), np.asarray([1.0])],
        CAVords=np.asarray([9]),
        quad_indices=np.asarray([0]),
        skew_quad_indices=np.asarray([1]),
        used_cor_ords=[[2], [3]],
        used_bpms_ords=np.asarray([4]),
        measured_orm=np.zeros((2, 2)),
        sigma_w=np.ones(2),
        measured_eta_x=np.zeros(1),
        measured_eta_y=np.zeros(1),
        output_dir=tmp_path,
        stage1_response_matrix_calculator="Analytical",
        stage2_response_matrix_calculator="Linear",
    )

    assert [call["response_matrix_calculator"] for call in calls] == [
        "Analytical",
        "Linear",
    ]
    assert calls[0]["nIter"] == 4
    assert calls[1]["nIter"] == 4
    assert all(call["analytical_implementation"] == "vectorized" for call in calls)
    assert all(call["analytical_use_mp"] is True for call in calls)
    assert all(call["analytical_dispersion_calculator"] == "Linear" for call in calls)


def test_linear_numerical_defaults_remain_linear_in_both_stages(driver):
    assert driver["benchmark_stage_response_matrix_calculators"]("Linear") == (
        "Linear",
        "Linear",
    )
    assert driver["QUAD_JACOBIAN_CALCULATOR"] == "Numerical"
    assert driver["SKEW_JACOBIAN_CALCULATOR"] == "Numerical"
