"""Reliable Accelerator Toolbox lattice loading for GUI projects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MatLatticeCandidate:
    variable: str
    element_count: int


class NoMatLatticeError(ValueError):
    pass


class MultipleMatLatticesError(ValueError):
    def __init__(self, candidates: list[MatLatticeCandidate]):
        self.candidates = candidates
        super().__init__("MAT file contains multiple valid Accelerator Toolbox lattices.")


def _mat_variable_names(path: Path) -> list[str]:
    """Enumerate user variables; pyAT remains authoritative for validity."""
    import scipy.io

    try:
        contents = scipy.io.loadmat(path)
        return [str(name) for name in contents if not str(name).startswith("__")]
    except NotImplementedError:
        # MATLAB 7.3 files are HDF5 containers. Enumerating their top-level
        # variables is necessary because pyAT itself requires ``use=``.
        import h5py
        with h5py.File(path, "r") as handle:
            return [str(name) for name in handle if not str(name).startswith("#")]


def mat_lattice_candidates(path: str | Path) -> list[MatLatticeCandidate]:
    """Return variables that pyAT can load as coherent AT lattices."""
    import at

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".mat":
        return []
    candidates: list[MatLatticeCandidate] = []
    for variable in _mat_variable_names(source):
        try:
            ring = at.load_mat(str(source), use=variable)
            count = len(ring)
            if count <= 0 or any(not hasattr(element, "FamName") for element in ring):
                continue
        except Exception:
            continue
        candidates.append(MatLatticeCandidate(variable, count))
    return candidates


def select_mat_lattice(path: str | Path, variable: str | None = None):
    """Load a selected MAT lattice or resolve a unique candidate."""
    import at

    source = Path(path).expanduser().resolve()
    candidates = mat_lattice_candidates(source)
    if variable:
        match = next((candidate for candidate in candidates if candidate.variable == variable), None)
        if match is None:
            available = ", ".join(candidate.variable for candidate in candidates) or "none"
            raise NoMatLatticeError(
                f"Saved MAT variable {variable!r} is not a valid AT lattice in {source.name}. "
                f"Valid lattice variables: {available}."
            )
        return at.load_mat(str(source), use=match.variable), match
    if not candidates:
        raise NoMatLatticeError(
            f"{source.name} contains no variable that Accelerator Toolbox can load as a valid lattice."
        )
    if len(candidates) > 1:
        raise MultipleMatLatticesError(candidates)
    candidate = candidates[0]
    return at.load_mat(str(source), use=candidate.variable), candidate


def load_project_lattice(path: str | Path, mat_variable: str | None = None):
    """Load any supported lattice, honoring a persisted MAT variable."""
    import at

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".mat":
        return select_mat_lattice(source, mat_variable)[0]
    return at.load_lattice(str(source))
