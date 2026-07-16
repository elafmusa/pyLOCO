"""Minimal project metadata model for the Milestone 1 GUI shell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProjectMetadata:
    """Small placeholder for project identity shown in future UI state."""

    name: str = "Untitled LOCO Project"
    mode: str = "Basic"
