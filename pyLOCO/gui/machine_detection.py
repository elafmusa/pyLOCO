"""Machine-component detection helpers shared by the GUI and tests."""

from __future__ import annotations

import re


ROLE_PATTERNS = {
    "bpm": r"BPM|MON",
    "hcor": r"HCM|HCOR|CH",
    "vcor": r"VCM|VCOR|CV",
    "quad": r"^(?!.*(?:SQ|SKQ|SKEW)).*Q",
    "skew": r"(?:^|[_-])(?:SQ|QS|SKQ|SKEW)(?:[_-]|\d|$)",
    "cavity": r"RFCAV|CAV|RF",
}


def element_name(element) -> str:
    values: list[str] = []
    for attribute in ("CommonName", "FamName", "Name", "name"):
        value = getattr(element, attribute, None)
        if value is not None and str(value) not in values:
            values.append(str(value))
    return " / ".join(values)


def detect_machine_elements(lattice, role: str) -> list[int]:
    """Return lattice ordinals using pyAT types first, then role-specific traits.

    Corrector plane separation uses KickAngle when it is informative and falls
    back to established family/name conventions.  RF cavities always use the
    actual :class:`at.elements.RFCavity` definition.
    """

    import numpy as np
    import at

    type_for_role = {
        "bpm": at.elements.Monitor,
        "hcor": at.elements.Corrector,
        "vcor": at.elements.Corrector,
        "quad": at.elements.Quadrupole,
        "skew": at.elements.Quadrupole,
        "cavity": at.elements.RFCavity,
    }
    refs = np.asarray(at.get_refpts(lattice, type_for_role[role]))
    candidates = [int(value) for value in (np.flatnonzero(refs) if refs.dtype == np.bool_ else refs.ravel())]
    if role in {"bpm", "cavity"}:
        return candidates
    if role in {"quad", "skew"}:
        # PETRA III skew quadrupoles are represented by pyAT multipole-capable
        # elements, not necessarily the concrete Quadrupole subclass.
        candidates = list(range(len(lattice)))
        selected = []
        for ordinal in candidates:
            element = lattice[ordinal]
            skew_strength = np.asarray(getattr(element, "PolynomA", []), dtype=float)
            is_skew = skew_strength.size > 1 and not np.isclose(skew_strength[1], 0.0)
            named_skew = bool(re.search(ROLE_PATTERNS["skew"], element_name(element), re.I))
            normal_strength = np.asarray(getattr(element, "PolynomB", []), dtype=float)
            is_normal = isinstance(element, at.elements.Quadrupole) or (normal_strength.size > 1 and not np.isclose(normal_strength[1], 0.0))
            if role == "skew" and (is_skew or named_skew):
                selected.append(ordinal)
            elif role == "quad" and is_normal and not (is_skew or named_skew):
                selected.append(ordinal)
        return selected
    selected = []
    for ordinal in candidates:
        element = lattice[ordinal]
        kick = np.asarray(getattr(element, "KickAngle", [0.0, 0.0]), dtype=float).ravel()
        plane = None
        if kick.size >= 2 and not np.isclose(kick[0], 0.0) and np.isclose(kick[1], 0.0):
            plane = "hcor"
        elif kick.size >= 2 and not np.isclose(kick[1], 0.0) and np.isclose(kick[0], 0.0):
            plane = "vcor"
        if plane is None:
            name = element_name(element)
            h_match = bool(re.search(ROLE_PATTERNS["hcor"], name, re.I))
            v_match = bool(re.search(ROLE_PATTERNS["vcor"], name, re.I))
            plane = "hcor" if h_match and not v_match else "vcor" if v_match and not h_match else None
        if plane == role:
            selected.append(ordinal)
    return selected
