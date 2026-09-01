import importlib.util
from pathlib import Path

import at
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "Examples/PETRAIV/benchmark_analytical_skew_jacobian.py"


def _module():
    spec = importlib.util.spec_from_file_location("petraiv_skew_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skew_benchmark_defaults_and_cli_overrides():
    module = _module()
    defaults = module._parse_args([])
    assert defaults.model_orm == "Linear"
    assert defaults.dispersion_calculator == "Linear"
    assert defaults.dispersion_worker == "rf_only"
    assert defaults.implementation == "vectorized"
    configured = module._parse_args([
        "--model-orm", "Tracking",
        "--dispersion-calculator", "Tracking",
        "--dispersion-worker", "legacy_full_orm",
        "--implementation", "legacy",
    ])
    assert configured.model_orm == "Tracking"
    assert configured.dispersion_calculator == "Tracking"
    assert configured.dispersion_worker == "legacy_full_orm"
    assert configured.implementation == "legacy"


def test_skew_benchmark_production_selection_and_missing_path():
    module = _module()
    if not module.production.LATTICE_FILE.is_file() or not module.production.CORRECTOR_FILE.is_file():
        pytest.skip("PETRA-IV production lattice/corrector inputs are not tracked")
    ring = at.load_lattice(module.production.LATTICE_FILE)
    ring.disable_6d()
    selection, family_counts = module._selection(
        ring, module.production.CORRECTOR_FILE
    )
    assert {name: len(values) for name, values in selection.items()} == module.EXPECTED_COUNTS
    assert family_counts == module.EXPECTED_SKEW_FAMILY_COUNTS
    assert selection["rf_cavities"].tolist() == [21085]
    with pytest.raises(FileNotFoundError, match="resolved-missing-lattice.mat"):
        module._resolve_input_path(
            "/private/tmp/resolved-missing-lattice.mat", "Lattice"
        )


def test_skew_benchmark_timing_event_lookup():
    module = _module()
    events = [{"derivative_seconds": 1.0}, {"derivative_seconds": 2.0}]
    assert module._latest(events, "derivative_seconds") == 2.0
    assert module._latest(events, "missing", 3.0) == 3.0
