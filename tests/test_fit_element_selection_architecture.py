import numpy as np
import pytest

from pyLOCO.gui.backend import _build_pyloco_kwargs
from pyLOCO.gui.models.project import LocoConfiguration, validate_element_groups


def test_legacy_project_migrates_tilt_selection_and_individual_modes():
    cfg=LocoConfiguration.from_dict({
        "machine_elements":{"normal_quadrupole_ords":[3,7],"skew_quadrupole_ords":[9]},
        "parameters":{"individuals":False},
    })
    assert cfg.machine_elements.quadrupole_tilt_ords==[3,7]
    assert not cfg.parameters.individuals
    assert not cfg.rejection.skew_individuals
    assert not cfg.rejection.tilt_individuals


def test_independent_element_selections_round_trip():
    cfg=LocoConfiguration.from_dict({"machine_elements":{
        "normal_quadrupole_ords":[1,2],"skew_quadrupole_ords":[3],
        "quadrupole_tilt_ords":[4,5],"normal_quadrupole_groups":[[1,2]],
        "skew_quadrupole_groups":[[3]],"quadrupole_tilt_groups":[[4],[5]],
    }})
    restored=LocoConfiguration.from_dict({"machine_elements":cfg.to_backend_mapping()["MachineElements"]})
    assert restored.machine_elements.quadrupole_tilt_ords==[4,5]
    assert restored.machine_elements.normal_quadrupole_groups==[[1,2]]


@pytest.mark.parametrize("groups,message", [
    ([[1],[]],"empty"), ([[1],[1,2]],"more than one"), ([[1]],"omit"),
    ([[1,6]],"out-of-range"), ([[1,3]],"unselected"),
])
def test_explicit_group_validation(groups,message):
    with pytest.raises(ValueError,match=message):validate_element_groups(groups,[1,2],5,"quadrupole")


def test_backend_uses_explicit_groups_without_changing_fit_math():
    measured={"orm":np.zeros((2,2)),"eta_x":np.zeros(1),"eta_y":np.zeros(1),"noise_x":np.ones(1),"noise_y":np.ones(1),"dispersion_supplied":False}
    indices={"nHBPM":1,"nVBPM":1,"nHorCOR":1,"nVerCOR":1,"used_bpms_ords":np.array([0]),"used_cor_ords":[np.array([1]),np.array([2])],"quads_ords":np.array([3,4]),"skew_ords":np.array([5]),"quads_tilt_ind":np.array([6,7])}
    class RM:dkick=(1e-5,1e-5);includeDispersion=False;rfStep=-3000.;fixedpathlength=False;bidirectional=True;calculator="Linear"
    class Fit:individuals=False
    class Fixed:rfstep=-3000.;Frequency=5e8
    options={"fit_list":["quads"],"machine_element_groups":{"normal_quadrupole_groups":[[3,4]]}}
    result=_build_pyloco_kwargs(ring=None,options=options,rm_cfg=RM(),fit_cfg=Fit(),constraint_cfg=None,fixed_parameters=Fixed(),measured=measured,indices=indices)
    assert result["quads_ords"]==[[3,4]] and result["quad_individuals"] is False
