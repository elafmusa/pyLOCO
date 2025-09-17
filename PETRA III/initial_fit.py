import numpy as np
from collections import OrderedDict
from pyloco_config import _cfg_get
# at the top of initial_fit.py
from typing import Optional, Tuple


def _normalize_cmstep(CMstep, nHorCOR, nVerCOR):
    if np.isscalar(CMstep):
        return (np.full(nHorCOR, float(CMstep)),
                np.full(nVerCOR, float(CMstep)))
    if isinstance(CMstep, (list, tuple)) and len(CMstep) == 2:
        h = np.atleast_1d(CMstep[0]).astype(float).ravel()
        v = np.atleast_1d(CMstep[1]).astype(float).ravel()
        if h.size != nHorCOR or v.size != nVerCOR:
            raise ValueError(f"CMstep[0]/[1] must be lengths ({nHorCOR},{nVerCOR}), got ({h.size},{v.size})")
        return h, v
    CMstep = np.atleast_1d(np.asarray(CMstep, dtype=float)).ravel()
    if CMstep.size == nHorCOR + nVerCOR:
        return CMstep[:nHorCOR], CMstep[nHorCOR:]
    raise ValueError("Unsupported CMstep format for _normalize_cmstep.")



def build_initial_fit_parameters(
    ring, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
    quads_ords, CMstep, individuals=True, skew_ords=None, rfStep=None, quads_tilt=None,
    config=None,
):
    cfg_fit_list    = _cfg_get(config, "fit_list",    fit_list)
    block_order     = _cfg_get(config, "block_order", None)
    init_policy     = _cfg_get(config, "init_policy", None)
    cfg_CMstep      = _cfg_get(config, "CMstep",      CMstep)
    cfg_rfStep      = _cfg_get(config, "rfStep",      rfStep)
    cfg_individual  = _cfg_get(config, "individuals", individuals)
    init_overrides  = _cfg_get(config, "init",        {}) or {}

    # NEW: which attribute (and optional index) to read for quads policy
    cfg_quads_attr        = _cfg_get(config, "quads_attr",        "PolynomB")
    cfg_quads_attr_index  = _cfg_get(config, "quads_attr_index",  1)

    if block_order is None:
        block_order = (
            "hbpm_gain","hbpm_coupling",
            "vbpm_coupling","vbpm_gain",
            "hcor_cal","vcor_cal",
            "hcor_coupling","vcor_coupling",
            "HCMEnergyShift","VCMEnergyShift",
            "delta_rf","quads","skew_quads","quads_tilt",
        )
    if cfg_fit_list is None:
        cfg_fit_list = list(block_order)

    if init_policy is None:
        init_policy = {
            "hbpm_gain": "ones",
            "hbpm_coupling": "zeros",
            "vbpm_coupling": "zeros",
            "vbpm_gain": "ones",
            "hcor_cal": "cmstep:h",
            "vcor_cal": "cmstep:v",
            "hcor_coupling": "zeros",
            "vcor_coupling": "zeros",
            "HCMEnergyShift": "zeros",
            "VCMEnergyShift": "zeros",
            "delta_rf": "rfstep",
            "quads": "quads:attr",       # <- default is “use quads_attr”
            "skew_quads": "zeros",
            "quads_tilt": "tilts:zeros",
        }

    h_step, v_step = _normalize_cmstep(cfg_CMstep, nHorCOR, nVerCOR)

    def _block_len(name):
        if name in ("hbpm_gain", "hbpm_coupling"):
            return nHBPM
        if name in ("vbpm_gain", "vbpm_coupling"):
            return nVBPM
        if name in ("hcor_cal", "hcor_coupling"):
            return nHorCOR
        if name in ("vcor_cal", "vcor_coupling"):
            return nVerCOR
        if name == "HCMEnergyShift":
            return nHorCOR
        if name == "VCMEnergyShift":
            return nVerCOR
        if name == "delta_rf":
            return 1
        if name == "quads":
            return len(quads_ords)
        if name == "skew_quads":
            return 0 if skew_ords is None else len(skew_ords)
        if name == "quads_tilt":
            return 0 if quads_tilt is None else len(quads_tilt)
        raise KeyError(f"Unknown block '{name}'")

    def _parse_attr_and_index(spec: str) -> Tuple[str, Optional[int]]:
        """Return (attr_name, idx_or_None) for strings like 'PolynomB[1]' or 'K'."""
        if isinstance(spec, str) and "[" in spec and spec.endswith("]"):
            name, idx_s = spec[:-1].split("[", 1)
            return name, int(idx_s)
        return spec, None

    def _get_attr_value(el, attr_name: str, idx: Optional[int]):
        try:
            val = getattr(el, attr_name)
        except AttributeError as e:
            raise AttributeError(f"Element has no attribute '{attr_name}'") from e
        if idx is not None:
            try:
                return val[idx]
            except Exception as e:
                raise IndexError(
                    f"Attribute '{attr_name}' is not indexable or index {idx} is invalid"
                ) from e
        return val

    def _policy_default(name, length):
        pol = init_policy.get(name, "zeros")
        if pol == "ones":      return np.ones(length, dtype=float)
        if pol == "zeros":     return np.zeros(length, dtype=float)
        if pol == "cmstep:h":  return np.asarray(h_step, dtype=float)
        if pol == "cmstep:v":  return np.asarray(v_step, dtype=float)
        if pol == "rfstep":
            if length != 1: raise ValueError("rfstep policy expects length 1")
            return np.array([cfg_rfStep], dtype=float)

        if pol.startswith("quads:"):
            token = pol.split(":", 1)[1]

            if token == "attr":
                attr_name, idx_from_spec = _parse_attr_and_index(cfg_quads_attr)
                idx_final = idx_from_spec if idx_from_spec is not None else cfg_quads_attr_index
            else:
                attr_name, idx_from_spec = _parse_attr_and_index(token)
                idx_final = idx_from_spec

            if not cfg_individual:
                base = [_get_attr_value(ring[g[0]], attr_name, idx_final) for g in quads_ords]
            else:
                base = [_get_attr_value(ring[q],    attr_name, idx_final) for q in quads_ords]
            return np.asarray(base, dtype=float)

        if pol == "tilts:zeros":
            return np.zeros(length, dtype=float)

        raise ValueError(f"Unknown init policy '{pol}' for block '{name}'")

    def _to_len(x, n):
        a = np.asarray(x, dtype=float).ravel()
        if a.size == 1: return np.full(n, float(a[0]))
        if a.size != n: raise ValueError(f"Expected length {n}, got {a.size}")
        return a

    def _initial_for(name, length):
        arr = _policy_default(name, length)
        if name in init_overrides:
            val = init_overrides[name]
            if callable(val):
                val = val(length=length, ring=ring, quads_ords=quads_ords,
                          skew_ords=skew_ords, CMstep=cfg_CMstep, rfStep=cfg_rfStep,
                          individuals=cfg_individual)
            arr = _to_len(val, length)
        return arr

    params, blocks, cursor = [], OrderedDict(), 0
    for name in block_order:
        if name not in cfg_fit_list: continue
        L = _block_len(name)
        if L <= 0: continue
        vals = _initial_for(name, L)
        params.append(vals)
        blocks[name] = slice(cursor, cursor + L)
        cursor += L

    initial_fit_parameters = np.concatenate(params) if params else np.zeros(0, dtype=float)
    return initial_fit_parameters, blocks
