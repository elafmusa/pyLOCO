"""Regression tests for shared analytical phase-ordering helpers."""

from __future__ import annotations

import ast
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
NORMAL = REPOSITORY / "pyLOCO" / "analytic_orm_with_normal_quad_errors.py"
SKEW = REPOSITORY / "pyLOCO" / "analytic_orm_with_skew_quad_errors.py"


def _module_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _analytical_function(tree: ast.Module) -> ast.FunctionDef:
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_analytic_orm_variation_with_")
    ]
    return next(
        (node for node in candidates if node.name.endswith("_legacy")),
        candidates[0],
    )


def _nested_definition(path: Path, name: str) -> tuple[str, ast.FunctionDef]:
    source, tree = _module_tree(path)
    outer = _analytical_function(tree)
    node = next(
        child
        for child in outer.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    )
    return textwrap.dedent(ast.get_source_segment(source, node)), node


def _phase_helpers(path: Path, tune=(0.31, 0.32)):
    namespace = {"math": math, "np": np, "Q": np.asarray(tune)}
    for name in ("dphi", "tau"):
        _, node = _nested_definition(path, name)
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["dphi"], namespace["tau"], namespace["Q"]


def test_normal_and_skew_tau_and_dphi_are_textually_identical():
    for name in ("tau", "dphi"):
        normal_source, _ = _nested_definition(NORMAL, name)
        skew_source, _ = _nested_definition(SKEW, name)
        assert skew_source == normal_source

    for path in (NORMAL, SKEW):
        source = path.read_text(encoding="utf-8")
        assert "__author__='Simone Maria Liuzzo, Andrea Franchi'" in source
        assert "# Edited: E.M 20 Aug 2026" in source
        assert "Edited: E.M. 20 Aug 2026" in source


def test_same_location_phase_ordering_for_both_analytical_modules():
    location = SimpleNamespace(mu=np.asarray([1.25, 2.5]))

    for path in (NORMAL, SKEW):
        dphi, tau, tune = _phase_helpers(path)
        for plane in (0, 1):
            assert dphi(plane, location, location, idx_w=10, idx_j=20) == 0.0
            reverse = dphi(plane, location, location, idx_w=20, idx_j=10)
            assert reverse == 2.0 * math.pi * tune[plane]
            assert dphi(plane, location, location) == 0.0
            assert tau(
                plane, location, location, idx_a=20, idx_b=10
            ) == reverse - math.pi * tune[plane]


def test_normal_pi_uses_indices_for_equal_positions_and_is_normal_only():
    namespace = {}
    _, node = _nested_definition(NORMAL, "PI")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(NORMAL), "exec"), namespace)
    pi = namespace["PI"]
    same = SimpleNamespace(s_pos=4.0)

    assert pi(same, same, idx_a=10, idx_b=20) == 1
    assert pi(same, same, idx_a=20, idx_b=10) == 0
    assert pi(same, same) == 0

    _, skew_tree = _module_tree(SKEW)
    skew_outer = _analytical_function(skew_tree)
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "PI"
        for node in ast.walk(skew_outer)
    )


def test_all_active_phase_ordering_calls_supply_available_indices():
    for path in (NORMAL, SKEW):
        _, tree = _module_tree(path)
        outer = _analytical_function(tree)
        calls = [
            node
            for node in ast.walk(outer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        tau_calls = [node for node in calls if node.func.id == "tau"]
        assert tau_calls
        for call in tau_calls:
            assert {keyword.arg for keyword in call.keywords} == {"idx_a", "idx_b"}

        direct_dphi_calls = [
            node
            for node in calls
            if node.func.id == "dphi" and len(node.args) == 3
        ]
        assert direct_dphi_calls
        for call in direct_dphi_calls:
            assert {keyword.arg for keyword in call.keywords} == {"idx_w", "idx_j"}

        pi_calls = [node for node in calls if node.func.id == "PI"]
        if path == NORMAL:
            assert pi_calls
            for call in pi_calls:
                assert {keyword.arg for keyword in call.keywords} == {
                    "idx_a", "idx_b"
                }
        else:
            assert pi_calls == []
