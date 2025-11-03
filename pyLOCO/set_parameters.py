import numpy as np
from collections import defaultdict
from typing import Optional, Sequence, Tuple, Any, Dict
from pyloco_config import _cfg_get  # your helper


# ---------- helpers ----------


def _is_array_like(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))

def _commonname_index_map(ring) -> Dict[str, list]:
    """Map CommonName -> list of element indices in the ring."""
    m: Dict[str, list] = defaultdict(list)
    for i, el in enumerate(ring):
        name = getattr(el, "CommonName", None)
        if name is not None:
            m[name].append(i)
    return m

def _set_attr_value(elem: Any, attr_name: str, value: float, index: Optional[int]):
    """
    Set scalar attribute directly; for array-like attributes, write into the provided index.
    """
    try:
        cur = getattr(elem, attr_name)
    except AttributeError as e:
        fam = getattr(elem, "FamName", getattr(elem, "CommonName", "?"))
        raise AttributeError(f"Element {fam} has no attribute '{attr_name}'") from e

    if _is_array_like(cur):
        if index is None:
            raise ValueError(
                f"Attribute '{attr_name}' is array-like; provide an index in config "
                f"(e.g. quads_attr='PolynomB', quads_attr_index=1 or 'PolynomB[1]')."
            )
        arr = np.array(cur, copy=True)
        if not (0 <= index < arr.size):
            raise IndexError(f"{attr_name}[{index}] out of bounds for size {arr.size}")
        arr[index] = float(value)
        setattr(elem, attr_name, arr)
    else:
        setattr(elem, attr_name, float(value))


# ---------- attribute-based corrections ----------
def _resolve_attr_for_block(block: str, config: Optional[object]) -> Tuple[str, Optional[int]]:
    """Return (attr_name, index_or_None) for a logical block from FitInitConfig."""
    if block == "quads":
        raw = _cfg_get(config, "quads_attr", "PolynomB")
        idx = _cfg_get(config, "quads_attr_index", 1)  # default to PolynomB[1]
    elif block == "skew_quads":
        raw = _cfg_get(config, "skew_attr", "PolynomA")
        idx = _cfg_get(config, "skew_attr_index", 1)
    else:
        raw, idx = "PolynomB", 1

    name, idx_in_raw = _parse_attr_and_index(raw)
    if idx_in_raw is not None:
        idx = idx_in_raw
    return name, idx


def _resolve_attr_for_block(block: str, config: Optional[object]) -> tuple[str, Optional[int]]:
    if block == "quads":
        spec = _cfg_get(config, "quads_attr", "PolynomB[1]")
        idx  = _cfg_get(config, "quads_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomB", 1)
    elif block == "skew_quads":
        spec = _cfg_get(config, "skew_attr", "PolynomA[1]")
        idx  = _cfg_get(config, "skew_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomA", 1)
    else:
        # sensible default
        spec = "PolynomB[1]"
        idx  = None
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomB", 1)

    if idx_from_spec is not None:
        idx = idx_from_spec
    return name, idx


def _resolve_attr_for_block_read(block: str, cfg) -> tuple[str, Optional[int]]:
    if block == "skew_quads":
        spec = _cfg_get(cfg, "skew_attr", "PolynomA[1]")
        idx  = _cfg_get(cfg, "skew_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomA", 1)
    else:
        spec = _cfg_get(cfg, "quads_attr", "PolynomB[1]")
        idx  = _cfg_get(cfg, "quads_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomB", 1)

    return name, (idx_from_spec if idx_from_spec is not None else idx)


def set_correction(
    ring,
    r: Sequence[float],
    elem_ind: Sequence,
    *,
    individuals: bool = True,
    block: str = "quads",
    config: Optional[object] = None
):
    """
    Apply corrections into the attribute defined in config for the given block
    ('quads' or 'skew_quads'). Families expanded by CommonName.
    """
    attr_name, idx = _resolve_attr_for_block(block, config)

    # normalize inputs
    if np.isscalar(elem_ind):
        elem_ind = [int(elem_ind)]
    else:
        elem_ind = list(elem_ind)
    r = np.asarray(r, dtype=float).ravel()

    cn_map = _commonname_index_map(ring)

    if individuals:
        if len(r) != len(elem_ind):
            raise ValueError(f"len(r)={len(r)} != len(elem_ind)={len(elem_ind)} for individuals=True")
        for val, i0 in zip(r, elem_ind):
            cname = getattr(ring[i0], "CommonName", None)
            targets = cn_map.get(cname, [i0]) if cname is not None else [i0]
            for ti in targets:
                _set_attr_value(ring[ti], attr_name, val, idx)
    else:
        if len(r) != len(elem_ind):
            raise ValueError(f"len(r)={len(r)} != n_families={len(elem_ind)} for individuals=False")
        for fam_val, fam in zip(r, elem_ind):
            fam = list(fam)
            if not fam:  # skip empty
                continue
            cname = getattr(ring[fam[0]], "CommonName", None)
            targets = set(fam)
            if cname is not None:
                targets.update(cn_map.get(cname, []))
            for ti in sorted(targets):
                _set_attr_value(ring[ti], attr_name, fam_val, idx)
    return ring


# ---------- tilt corrections via R1/R2 ----------
def _make_R_mats(psi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build entrance/exit 6x6 rotation matrices (R1, R2) for a quadrupole tilt Ïˆ (radians)."""
    c, s = np.cos(psi), np.sin(psi)
    R1 = np.diag([c, c, c, c, 1.0, 1.0])
    R1[0, 2] = s;  R1[1, 3] = s
    R1[2, 0] = -s; R1[3, 1] = -s

    c2, s2 = np.cos(-psi), np.sin(-psi)
    R2 = np.diag([c2, c2, c2, c2, 1.0, 1.0])
    R2[0, 2] = s2;  R2[1, 3] = s2
    R2[2, 0] = -s2; R2[3, 1] = -s2
    return R1, R2

def _extract_psi_from_R1(R1: Optional[np.ndarray]) -> float:
    if not isinstance(R1, np.ndarray) or R1.shape != (6, 6):
        return 0.0
    return float(np.arctan2(R1[0, 2], R1[0, 0]))

def _set_R_mats(elem: Any, R1_attr: str, R2_attr: str, psi: float):
    R1, R2 = _make_R_mats(psi)
    setattr(elem, R1_attr, R1)
    setattr(elem, R2_attr, R2)

def set_correction_tilt(
    ring,
    psi_values: Sequence[float],   # radians
    elem_ind: Sequence,
    *,
    individuals: bool = True,
    config: Optional[object] = None,
):
    """
    Apply quadrupole tilts by writing R1/R2 computed from Ïˆ (radians).
    Config keys:
      - quads_tilt_attr_R1: 'R1' (default)
      - quads_tilt_attr_R2: 'R2' (default)
      - quads_tilt_method:  'set' or 'add' (default 'set')
    """
    R1_attr = _cfg_get(config, "quads_tilt_attr_R1", "R1")
    R2_attr = _cfg_get(config, "quads_tilt_attr_R2", "R2")
    method  = _cfg_get(config, "quads_tilt_method", "set").lower()
    if method not in ("set", "add"):
        raise ValueError("quads_tilt_method must be 'set' or 'add'")

    # normalize inputs
    psi_values = np.asarray(psi_values, dtype=float).ravel()
    if np.isscalar(elem_ind):
        elem_ind = [int(elem_ind)]
    else:
        elem_ind = list(elem_ind)

    cn_map = _commonname_index_map(ring)

    def _apply(el, psi):
        if method == "add":
            psi0 = _extract_psi_from_R1(getattr(el, R1_attr, None))
            _set_R_mats(el, R1_attr, R2_attr, psi0 + float(psi))
        else:
            _set_R_mats(el, R1_attr, R2_attr, float(psi))

    if individuals:
        if len(psi_values) != len(elem_ind):
            raise ValueError(f"len(psi_values)={len(psi_values)} != len(elem_ind)={len(elem_ind)} (individuals=True)")
        for psi, i0 in zip(psi_values, elem_ind):
            cname = getattr(ring[i0], "CommonName", None)
            targets = cn_map.get(cname, [i0]) if cname is not None else [i0]
            for ti in targets:
                _apply(ring[ti], psi)
    else:
        if len(psi_values) != len(elem_ind):
            raise ValueError(f"len(psi_values)={len(psi_values)} != n_families={len(elem_ind)} (individuals=False)")
        for psi, fam in zip(psi_values, elem_ind):
            fam = list(fam)
            if not fam:
                continue
            cname = getattr(ring[fam[0]], "CommonName", None)
            targets = set(fam)
            if cname is not None:
                targets.update(cn_map.get(cname, []))
            for ti in sorted(targets):
                _apply(ring[ti], psi)

    return ring



def _parse_attr_and_index(
    spec: Optional[str],
    default_name: str,
    default_idx: Optional[int],
) -> tuple[str, Optional[int]]:
    """
    Accepts: None, 'K', 'PolynomB', 'PolynomB[1]', 'KickAngle[0]', ...
    Returns: (attr_name, index_or_None)
    """
    if not spec:
        return default_name, default_idx
    s = str(spec)
    if "[" in s and s.endswith("]"):
        name, idx_s = s[:-1].split("[", 1)
        return name, int(idx_s)
    return s, default_idx


def _get_attr_scalar(
    el,
    attr_name: str,
    idx: Optional[int]
) -> float:
    try:
        val = getattr(el, attr_name)
    except AttributeError as e:
        fam = getattr(el, "FamName", getattr(el, "CommonName", "?"))
        raise AttributeError(f"Element '{fam}' has no attribute '{attr_name}'") from e

    if np.isscalar(val):
        return float(val)

    arr = np.asarray(val, dtype=float)
    if idx is None:
        raise ValueError(
            f"Attribute '{attr_name}' is array-like; provide an index "
            f"(e.g. 'PolynomB[1]' or quads_attr_index=1)."
        )
    if not (0 <= idx < arr.size):
        raise IndexError(f"Index {idx} out of range for '{attr_name}' (size {arr.size}).")
    return float(arr[idx])

def _resolve_attr_for_block_read(
    block: str,
    cfg
) -> Tuple[str, Optional[int]]:
    """
    Decide which attribute+index to read for a block using config.
    Defaults: quads -> PolynomB[1]; skew_quads -> PolynomA[1].
    """
    if block == "skew_quads":
        spec = _cfg_get(cfg, "skew_attr", "PolynomB[1]")
        idx  = _cfg_get(cfg, "skew_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomB", 1)
        return name, (idx_from_spec if idx_from_spec is not None else idx)
    else:  # "quads"
        spec = _cfg_get(cfg, "quads_attr", "PolynomB[1]")
        idx  = _cfg_get(cfg, "quads_attr_index", None)
        name, idx_from_spec = _parse_attr_and_index(spec, "PolynomB", 1)
        return name, (idx_from_spec if idx_from_spec is not None else idx)


def _initial_values_for_block(ring, ords, *, block: str, individuals: bool, config=None):
    """
    ords: list[int] (individuals) or list[list[int]] (families)
    block: 'quads' or 'skew_quads'
    config is your FitInitConfig (so we can read quads_attr / skew_attr & indices)
    """
    # decide which attribute to read from config
    if block == "skew_quads":
        attr_spec = getattr(config, "skew_attr", "PolynomB[1]")
        default = ("PolynomB", 1)
    else:
        attr_spec = getattr(config, "quads_attr", "PolynomB[1]")
        default = ("PolynomB", 1)

    attr_name, attr_idx = _parse_attr_and_index(attr_spec, *default)

    vals = []
    if individuals:
        # ords = [i, j, k, ...]
        for i in ords:
            vals.append(_get_attr_scalar(ring[int(i)], attr_name, attr_idx))
    else:
        # ords = [[i1,i2,...], [j1,j2,...], ...]  -> take representative of each family
        for fam in ords:
            i0 = int(fam[0]) if hasattr(fam, "__iter__") else int(fam)
            vals.append(_get_attr_scalar(ring[i0], attr_name, attr_idx))
    return vals