"""Portable Measurement Session manifest schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ._json import read_json, write_json
from .measurement import SCHEMA_VERSION, validate_measurement_file

SESSION_FILE_TYPE = "pyloco.measurement_session"
SESSION_ROLES = frozenset({"orm", "bpm_noise", "dispersion"})


@dataclass(frozen=True)
class SessionFile:
    role: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in SESSION_ROLES:
            raise ValueError(f"Unsupported Measurement Session role: {self.role}")
        candidate = Path(self.path)
        if candidate.is_absolute():
            raise ValueError(f"Measurement Session paths must be portable and relative: {self.path}")
        if ".." in candidate.parts:
            raise ValueError(f"Measurement Session paths may not escape the session directory: {self.path}")


@dataclass(frozen=True)
class MeasurementSession:
    session_id: str
    files: tuple[SessionFile, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    file_type: str = SESSION_FILE_TYPE

    def validate(self, *, base_directory: str | Path | None = None, validate_files: bool = False) -> None:
        if self.file_type != SESSION_FILE_TYPE:
            raise ValueError("Invalid Measurement Session file_type")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported Measurement Session schema version: {self.schema_version}")
        if not self.session_id.strip():
            raise ValueError("Measurement Session requires a non-empty session_id")
        roles = [entry.role for entry in self.files]
        if len(roles) != len(set(roles)):
            raise ValueError("Measurement Session contains a duplicate measurement role")
        for entry in self.files:
            entry.validate()
            if validate_files:
                if base_directory is None:
                    raise ValueError("base_directory is required when validate_files=True")
                info = validate_measurement_file(Path(base_directory) / entry.path)
                if info["kind"] != entry.role:
                    raise ValueError(
                        f"Session role {entry.role!r} does not match measurement kind {info['kind']!r}"
                    )

    def resolve(self, manifest_path: str | Path) -> dict[str, Path]:
        base = Path(manifest_path).resolve().parent
        return {entry.role: (base / entry.path).resolve() for entry in self.files}

    @property
    def missing_roles(self) -> tuple[str, ...]:
        present = {entry.role for entry in self.files}
        return tuple(role for role in ("orm", "bpm_noise", "dispersion") if role not in present)

    @property
    def is_complete_for_loco(self) -> bool:
        return not self.missing_roles

    def to_gui_measurements(self, manifest_path: str | Path) -> dict[str, dict[str, Any]]:
        """Future main-GUI bridge: role -> path/options without schema guessing."""
        resolved = self.resolve(manifest_path)
        return {
            entry.role: {"path": str(resolved[entry.role]), "options": dict(entry.metadata)}
            for entry in self.files
        }


def save_session(path: str | Path, session: MeasurementSession, *, validate_files: bool = True) -> Path:
    destination = Path(path)
    session.validate(base_directory=destination.parent, validate_files=validate_files)
    return write_json(destination, asdict(session))


def load_session(path: str | Path, *, validate_files: bool = True) -> MeasurementSession:
    source = Path(path)
    data = read_json(source)
    files = tuple(SessionFile(**entry) for entry in data.get("files", ()))
    session = MeasurementSession(
        session_id=str(data.get("session_id", "")),
        files=files,
        metadata=dict(data.get("metadata", {})),
        schema_version=str(data.get("schema_version", "")),
        file_type=str(data.get("file_type", "")),
    )
    session.validate(base_directory=source.parent, validate_files=validate_files)
    return session
