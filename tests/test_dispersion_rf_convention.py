from types import SimpleNamespace

import pytest

from pyLOCO.gui.backend import _validate_configured_rf_against_lattice
from pyLOCO.gui.models.project import LocoConfiguration


class _Cavity:
    Frequency = 352_372_169.3993786
    HarmNumber = 992


class _Ring(list):
    harmonic_number = 992


def test_reference_lattice_rf_frequency_and_harmonic_are_authoritative():
    ring = _Ring([object(), _Cavity(), _Cavity()])
    fixed = SimpleNamespace(Frequency=352_372_169.3993786, HarmNumber=992)
    _validate_configured_rf_against_lattice(ring, fixed)
    with pytest.raises(ValueError, match="499664000.*352372169",):
        _validate_configured_rf_against_lattice(
            ring, SimpleNamespace(Frequency=499_664_000.0, HarmNumber=992)
        )
    with pytest.raises(ValueError, match="harmonic number"):
        _validate_configured_rf_against_lattice(
            ring, SimpleNamespace(Frequency=352_372_169.3993786, HarmNumber=3840)
        )


def test_legacy_bipolar_fit_configuration_migrates_to_positive_total_separation():
    config = LocoConfiguration.from_dict({
        "response_matrix": {"rfStep": -200.0, "bidirectional": True},
        "parameters": {"rfStep": -200.0},
        "fixed_parameters": {"rfstep": -200.0},
    })
    assert config.response_matrix.rfStep == 200.0
    assert config.parameters.rfStep == 200.0
    assert config.fixed_parameters.rfstep == 200.0


def test_one_sided_signed_step_remains_supported():
    config = LocoConfiguration.from_dict({
        "response_matrix": {"rfStep": -100.0, "bidirectional": False},
        "parameters": {"rfStep": -100.0},
        "fixed_parameters": {"rfstep": -100.0},
    })
    assert config.response_matrix.rfStep == -100.0
