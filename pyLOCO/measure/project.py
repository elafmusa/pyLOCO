"""Portable pyLOCO Measure project configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyLOCO.data_schema._json import read_json, write_json

FILE_TYPE = "pyloco.measure_project"
SCHEMA_VERSION = "1.0"


@dataclass
class MeasureProject:
    measurement_type: str = "bpm_noise"
    measurement_name: str = "bpm-noise"
    measurement_label: str = "Mock BPM noise"
    operator_comments: str = ""
    adapter: str = "Mock"
    pysc_profile: str = "ebs"
    bpm_selection_method: str = "all"
    bpm_manual: str = ""
    bpm_names_file: str = ""
    excluded_bpm_positions: str = ""
    hcor_selection_method: str = "all"
    hcor_manual: str = ""
    hcor_names_file: str = ""
    vcor_selection_method: str = "all"
    vcor_manual: str = ""
    vcor_names_file: str = ""
    excluded_hcor_positions: str = ""
    excluded_vcor_positions: str = ""
    readings: int = 20
    delay_seconds: float = 0.1
    settling_delay_seconds: float = 0.0
    verify_restored_orbit: bool = True
    rf_control_mode: str = "manual"
    nominal_rf_hz: float | None = None
    nominal_rf_source: str = "manual"
    rf_step_hz: float = 200.0
    dispersion_direction: str = "bipolar"
    orm_direction: str = "bipolar"
    orm_kick_mode: str = "common"
    orm_horizontal_kick_rad: float = 100e-6
    orm_vertical_kick_rad: float = 100e-6
    orm_kick_file: str = ""
    orm_scaled: bool = False
    output_directory: str = "measurements"
    theme: str = "dark"
    metadata: dict[str, Any] = field(default_factory=dict)
    file_type: str = FILE_TYPE
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.file_type != FILE_TYPE or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Invalid or unsupported pyLOCO Measure project")
        if self.adapter not in {"Mock","pySC Server","PETRA / DOOCS","PETRA III DOOCS"}:
            raise ValueError("Unsupported Measure adapter")
        if self.pysc_profile not in {"ebs", "petra3", "petra3_realistic"}:
            raise ValueError("Unsupported pySC machine profile")
        if self.measurement_type not in {"bpm_noise", "dispersion", "orm"}:
            raise ValueError("Unsupported measurement type")
        if self.adapter in {"PETRA / DOOCS","PETRA III DOOCS"} and self.measurement_type=="orm":
            raise ValueError("ORM acquisition is unavailable in PETRA read-only mode")
        if self.rf_control_mode not in {"manual","automatic"}:
            raise ValueError("Unsupported RF control mode")
        if self.dispersion_direction not in {"bipolar", "positive", "negative"}:
            raise ValueError("Unsupported dispersion direction")
        if self.measurement_type == "dispersion":
            if self.nominal_rf_hz is None or self.nominal_rf_hz <= 0:
                raise ValueError("Dispersion requires an explicitly configured nominal RF frequency")
            if self.rf_step_hz <= 0 or self.settling_delay_seconds < 0:
                raise ValueError("RF step must be positive and settling delay non-negative")
        if self.bpm_selection_method not in {"all", "names_file", "manual"}:
            raise ValueError("Unsupported BPM selection method")
        if self.hcor_selection_method not in {"all", "names_file", "manual"} or self.vcor_selection_method not in {"all", "names_file", "manual"}:
            raise ValueError("Unsupported corrector selection method")
        if self.orm_direction not in {"bipolar","positive","negative"} or self.orm_kick_mode not in {"common","file"}:
            raise ValueError("Unsupported ORM acquisition configuration")
        if self.orm_horizontal_kick_rad<=0 or self.orm_vertical_kick_rad<=0:
            raise ValueError("ORM kick values must be positive")
        if self.readings < 2 or self.delay_seconds < 0:
            raise ValueError("Readings must be at least 2 and delay must be non-negative")
        if not self.measurement_name.strip() or not self.output_directory.strip():
            raise ValueError("Measurement name and output directory are required")
        if Path(self.output_directory).is_absolute():
            raise ValueError("Portable Measure projects require a relative output directory")


def save_measure_project(path: str | Path, project: MeasureProject) -> Path:
    project.validate()
    return write_json(path, asdict(project))


def load_measure_project(path: str | Path) -> MeasureProject:
    project = MeasureProject(**read_json(path))
    project.validate()
    return project
