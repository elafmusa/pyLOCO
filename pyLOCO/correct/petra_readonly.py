"""Explicit PETRA mapping and hard read-only correction diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from pyLOCO.control_system.petra import PETRAReadOnlyAdapter, READ_ONLY_ERROR
from pyLOCO.data_schema._json import write_json

from .model import CorrectionReview

SNAPSHOT_FILE_TYPE="pyloco.petra_readonly_snapshot"
SNAPSHOT_SCHEMA_VERSION="1.0"


@dataclass(frozen=True)
class MagnetMapping:
    lattice_name: str
    control_name: str
    lattice_ordinal: int | None = None


@dataclass(frozen=True)
class PETRAReadOnlySnapshot:
    source_result: str
    timestamp_utc: str
    adapter: str
    magnets: tuple[dict[str,Any],...]
    fraction_comparison: tuple[dict[str,Any],...]
    mapping_file: str | None = None
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    file_type: str = SNAPSHOT_FILE_TYPE

    def validate(self) -> None:
        if self.file_type!=SNAPSHOT_FILE_TYPE or self.schema_version!=SNAPSHOT_SCHEMA_VERSION:raise ValueError("Unsupported PETRA read-only snapshot schema")
        if not self.source_result or not self.timestamp_utc:raise ValueError("PETRA snapshot requires source and timestamp")
        if any(not row.get("lattice_name") for row in self.magnets):raise ValueError("Every PETRA snapshot row requires a lattice name")


def load_mapping(path: str|Path) -> tuple[MagnetMapping,...]:
    source=Path(path).expanduser().resolve(); data=yaml.safe_load(source.read_text(encoding="utf-8")) if source.suffix.lower() in {".yaml",".yml"} else json.loads(source.read_text(encoding="utf-8"))
    entries=data.get("mappings",data) if isinstance(data,dict) else data
    if isinstance(entries,dict):entries=[{"lattice_name":name,"control_name":control} for name,control in entries.items()]
    if not isinstance(entries,list):raise ValueError("PETRA mapping must contain a 'mappings' list or lattice-name mapping")
    result=[]
    for index,row in enumerate(entries):
        if not isinstance(row,dict):raise ValueError(f"Mapping entry {index} is not an object")
        lattice=str(row.get("lattice_name",row.get("element_name",row.get("name","")))).strip(); control=str(row.get("control_name",row.get("power_supply_name",""))).strip(); ordinal=row.get("lattice_ordinal",row.get("ordinal"))
        if not lattice or not control:raise ValueError(f"Mapping entry {index} requires lattice_name and control_name")
        result.append(MagnetMapping(lattice,control,None if ordinal is None else int(ordinal)))
    return tuple(result)


def load_name_set(path: str|Path) -> frozenset[str]:
    source=Path(path).expanduser().resolve()
    if source.suffix.lower() in {".json",".yaml",".yml"}:
        data=yaml.safe_load(source.read_text(encoding="utf-8")); data=data.get("magnets",data.get("names",data)) if isinstance(data,dict) else data
        if not isinstance(data,list):raise ValueError("Calibration warning list must be a list or contain 'magnets'/'names'")
        return frozenset(str(value).strip() for value in data if str(value).strip())
    return frozenset(line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def apply_explicit_mapping(review: CorrectionReview, mappings: Iterable[MagnetMapping]) -> dict[str,int]:
    entries=tuple(mappings); controls={entry.control_name for entry in entries}; duplicate_controls={name for name in controls if sum(entry.control_name==name for entry in entries)>1}
    counts={"mapped":0,"unmapped":0,"ambiguous":0,"duplicate":0}
    for item in review.items:
        candidates=[entry for entry in entries if entry.lattice_name==item.name and (entry.lattice_ordinal is None or entry.lattice_ordinal==item.lattice_ordinal)]
        if not candidates:
            item.control_name=None; status="unmapped"
        elif len(candidates)>1:
            item.control_name=None; status="ambiguous"
        else:
            item.control_name=candidates[0].control_name; status="duplicate" if item.control_name in duplicate_controls else "mapped"
        item.metadata["mapping_status"]=status; counts[status]+=1
    return counts


class PETRACorrectReadOnlyService:
    """Reads PETRA diagnostics and calculations; it contains no write path."""
    def __init__(self, adapter: PETRAReadOnlyAdapter, *, sign_difference_names: Iterable[str]=(), large_difference_names: Iterable[str]=()) -> None:
        self.adapter=adapter; self.sign_difference_names=frozenset(sign_difference_names); self.large_difference_names=frozenset(large_difference_names)

    def write(self,*_args,**_kwargs):
        raise PermissionError(READ_ONLY_ERROR)

    def _fraction_comparison(self,review: CorrectionReview,fractions=(.1,.25,.5,1.0)):
        rows=[]
        for fraction in fractions:
            relative=[]; delta_i=[]; violations=0; warnings=0; unmapped=0
            for item in review.items:
                if not item.included:continue
                if item.metadata.get("mapping_status")!="mapped":unmapped+=1; continue
                if item.machine_value is None or item.current_ampere is None:continue
                final=item.recommended_machine_delta*float(fraction)*item.individual_scale
                if item.machine_value:relative.append(abs(100*final/item.machine_value))
                try:target_current=self.adapter.strength_to_current(item.control_name,item.machine_value+final)
                except Exception:warnings+=1; continue
                delta_i.append(abs(target_current-item.current_ampere))
                if item.min_current_ampere is not None and item.max_current_ampere is not None and not item.min_current_ampere<=target_current<=item.max_current_ampere:violations+=1
                if item.sign_difference or item.calibration_difference_percent is not None:warnings+=1
            rows.append({"fraction":float(fraction),"max_abs_delta_k_over_k_percent":max(relative,default=None),"max_abs_delta_i_ampere":max(delta_i,default=None),"current_limit_violations":violations,"calibration_warnings":warnings,"unmapped_magnets":unmapped})
        return tuple(rows)

    def read_snapshot(self,review: CorrectionReview, *, mapping_file: str|None=None) -> PETRAReadOnlySnapshot:
        stamp=datetime.now(timezone.utc).isoformat(); rows=[]
        for item in review.items:
            status=item.metadata.get("mapping_status","unmapped")
            if item.correction_type!="normal_quadrupole" or status!="mapped":
                rows.append(self._snapshot_row(item)); continue
            try:
                state=self.adapter.read_quadrupole_diagnostics(item.control_name); item.machine_value=float(state["strength_sp"]); item.current_ampere=state.get("current_ampere"); item.metadata["current_source"]=state.get("current_source")
                item.min_current_ampere,item.max_current_ampere=self.adapter.current_limits(item.control_name)
                item.calibrated_target_current_ampere=self.adapter.strength_to_current(item.control_name,item.target_value)
                item.sign_difference=item.control_name in self.sign_difference_names or item.name in self.sign_difference_names
                if item.control_name in self.large_difference_names or item.name in self.large_difference_names:item.calibration_difference_percent=100.0
                if item.sign_difference:item.metadata["calibration_status"]="Sign convention warning"
                elif item.calibration_difference_percent is not None:item.metadata["calibration_status"]="Large calibration discrepancy"
                else:item.metadata["calibration_status"]="Calibration OK"
                item.metadata["diagnostic_source"]="PETRA read-only"
            except Exception as exc:
                item.metadata["calibration_status"]="Calibration unavailable"; item.metadata["diagnostic_error"]=str(exc); item.calibrated_target_current_ampere=None
            rows.append(self._snapshot_row(item))
        comparison=self._fraction_comparison(review); review.real_fraction_comparison=comparison
        snapshot=PETRAReadOnlySnapshot(review.source_result,stamp,type(self.adapter).__name__,tuple(rows),comparison,mapping_file); snapshot.validate(); return snapshot

    @staticmethod
    def _snapshot_row(item):
        return {"index":item.index,"correction_type":item.correction_type,"lattice_name":item.name,"lattice_ordinal":item.lattice_ordinal,"control_name":item.control_name,"mapping_status":item.metadata.get("mapping_status","unmapped"),"machine_k":item.machine_value,"current_ampere":item.current_ampere,"current_source":item.metadata.get("current_source"),"raw_loco_delta_k":item.raw_fitted_delta,"recommended_machine_delta_k":item.recommended_machine_delta,"global_fraction":item.global_scale,"individual_fraction":item.individual_scale,"final_delta_k":item.final_delta,"target_k":item.target_value if item.machine_value is not None else None,"target_current_ampere":item.target_current_ampere,"delta_i_ampere":item.delta_i_ampere,"min_current_ampere":item.min_current_ampere,"max_current_ampere":item.max_current_ampere,"current_limit_margin_ampere":item.current_limit_margin_ampere,"current_limit_status":item.current_limit_status,"calibration_status":item.calibration_status,"sign_difference":item.sign_difference,"calibration_difference_percent":item.calibration_difference_percent,"diagnostic_error":item.metadata.get("diagnostic_error")}


def save_snapshot(path: str|Path,snapshot: PETRAReadOnlySnapshot) -> Path:
    snapshot.validate(); return write_json(path,asdict(snapshot))


def load_snapshot(path: str|Path) -> PETRAReadOnlySnapshot:
    data=json.loads(Path(path).read_text(encoding="utf-8")); snapshot=PETRAReadOnlySnapshot(source_result=str(data.get("source_result","")),timestamp_utc=str(data.get("timestamp_utc","")),adapter=str(data.get("adapter","")),magnets=tuple(data.get("magnets",())),fraction_comparison=tuple(data.get("fraction_comparison",())),mapping_file=data.get("mapping_file"),schema_version=str(data.get("schema_version","")),file_type=str(data.get("file_type",""))); snapshot.validate(); return snapshot
