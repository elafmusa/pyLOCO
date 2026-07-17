"""UI-only project state for the pyLOCO GUI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_FILE_SUFFIX = ".pyloco.json"
REQUIRED_MEASUREMENTS = ("orm", "dispersion", "bpm_noise")


@dataclass(slots=True)
class ImportedDataset:
    """Metadata for a measurement file imported into the GUI project."""

    role: str
    path: str
    file_type: str
    size_bytes: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(slots=True)
class LatticeSelection:
    """Metadata for a selected lattice/model file."""

    path: str = ""
    file_type: str = ""
    element_count: int | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name if self.path else "No lattice selected"


@dataclass(slots=True)
class ProjectMetadata:
    """Serializable GUI project state that does not touch numerical pyLOCO code."""

    name: str = "Untitled LOCO Project"
    mode: str = "Basic"
    path: str = ""
    modified: bool = False
    lattice: LatticeSelection = field(default_factory=LatticeSelection)
    measurements: dict[str, ImportedDataset] = field(default_factory=dict)
    recent_projects: list[str] = field(default_factory=list)

    @property
    def is_saved(self) -> bool:
        return bool(self.path) and not self.modified

    def validation_messages(self) -> list[str]:
        """Return missing required inputs for enabling LOCO execution."""

        messages: list[str] = []
        if not self.name.strip():
            messages.append("Project name is required.")
        if not self.lattice.path:
            messages.append("A lattice/model file is required.")
        for role in REQUIRED_MEASUREMENTS:
            if role not in self.measurements:
                messages.append(f"{role.replace('_', ' ').title()} data is required.")
        return messages

    @property
    def is_complete(self) -> bool:
        return not self.validation_messages()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modified"] = False
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMetadata":
        lattice_data = data.get("lattice") or {}
        measurement_data = data.get("measurements") or {}
        project = cls(
            name=data.get("name", "Untitled LOCO Project"),
            mode=data.get("mode", "Basic"),
            path=data.get("path", ""),
            modified=False,
            lattice=LatticeSelection(**lattice_data),
            measurements={
                key: ImportedDataset(**value) for key, value in measurement_data.items()
            },
            recent_projects=list(data.get("recent_projects", [])),
        )
        return project

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path).expanduser()
        if target.suffix != PROJECT_FILE_SUFFIX:
            target = target.with_suffix(PROJECT_FILE_SUFFIX)
        self.path = str(target)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.modified = False
        self.add_recent_project(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ProjectMetadata":
        source = Path(path).expanduser()
        project = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        project.path = str(source)
        project.modified = False
        project.add_recent_project(source)
        return project

    def add_recent_project(self, path: str | Path) -> None:
        normalized = str(Path(path).expanduser())
        self.recent_projects = [p for p in self.recent_projects if p != normalized]
        self.recent_projects.insert(0, normalized)
        del self.recent_projects[5:]
