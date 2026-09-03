"""Qt-free correction review model. This module contains no machine write path."""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from pyLOCO.data_schema import CorrectionPlan, CorrectionRecord, load_correction_plan, save_correction_plan
from pyLOCO.gui.results.results_loader import ResultsLoader


@dataclass(frozen=True)
class WarningThresholds:
    noteworthy_relative_percent: float = 2.0
    serious_relative_percent: float = 5.0
    calibration_difference_percent: float = 20.0


@dataclass
class CorrectItem:
    index: int
    correction_type: str
    name: str
    lattice_ordinal: int | None
    unit: str
    initial_value: float | None
    fitted_value: float | None
    raw_fitted_delta: float
    recommended_machine_delta: float
    sign_convention: str
    family: str | None = None
    control_name: str | None = None
    global_scale: float = 0.1
    individual_scale: float = 1.0
    included: bool = True
    exclusion_reason: str = ""
    current_ampere: float | None = None
    ampere_per_unit: float | None = None
    min_current_ampere: float | None = None
    max_current_ampere: float | None = None
    calibration_difference_percent: float | None = None
    sign_difference: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    machine_value: float | None = None
    calibrated_target_current_ampere: float | None = None

    @property
    def final_delta(self) -> float:
        return self.recommended_machine_delta * self.global_scale * self.individual_scale if self.included else 0.0

    @property
    def target_value(self) -> float | None:
        baseline=self.machine_value if self.machine_value is not None else self.initial_value
        return None if baseline is None else baseline + self.final_delta

    @property
    def relative_percent(self) -> float | None:
        if self.initial_value in (None, 0): return None
        return 100.0 * self.final_delta / self.initial_value

    @property
    def delta_i_ampere(self) -> float | None:
        if self.calibrated_target_current_ampere is not None and self.current_ampere is not None:
            return self.calibrated_target_current_ampere-self.current_ampere
        return None if self.ampere_per_unit is None else self.final_delta * self.ampere_per_unit

    @property
    def target_current_ampere(self) -> float | None:
        if self.calibrated_target_current_ampere is not None:return self.calibrated_target_current_ampere
        delta = self.delta_i_ampere
        return None if self.current_ampere is None or delta is None else self.current_ampere + delta

    @property
    def current_limit_margin_ampere(self) -> float | None:
        target=self.target_current_ampere
        if target is None or self.min_current_ampere is None or self.max_current_ampere is None:return None
        return min(target-self.min_current_ampere,self.max_current_ampere-target)

    @property
    def current_limit_status(self) -> str:
        target = self.target_current_ampere
        if target is None or self.min_current_ampere is None or self.max_current_ampere is None: return "Not available"
        return "VIOLATION" if not self.min_current_ampere <= target <= self.max_current_ampere else "Within limits"

    @property
    def calibration_status(self) -> str:
        explicit=self.metadata.get("calibration_status")
        if explicit:return str(explicit)
        if self.ampere_per_unit is None:return "Not available"
        return "Available (Mock)" if "Mock" in str(self.metadata.get("diagnostic_source","")) else "Available (source)"

    def warnings(self, thresholds: WarningThresholds) -> tuple[str, ...]:
        result=[]; relative=abs(self.relative_percent or 0.0)
        if self.metadata.get("mapping_status") in {"unmapped","ambiguous","duplicate"}:result.append(f"mapping_{self.metadata['mapping_status']}")
        if relative >= thresholds.serious_relative_percent: result.append("serious_relative_correction")
        elif relative >= thresholds.noteworthy_relative_percent: result.append("noteworthy_relative_correction")
        if self.sign_difference: result.append("sign_difference")
        if self.calibration_difference_percent is not None and abs(self.calibration_difference_percent) >= thresholds.calibration_difference_percent: result.append("large_calibration_difference")
        if self.metadata.get("calibration_status") == "Calibration unavailable":result.append("calibration_unavailable")
        if self.current_limit_status == "VIOLATION": result.append("current_limit_violation")
        return tuple(result)


class CorrectionReview:
    """Mutable dry-run review state; raw corrections are never modified."""

    def __init__(self, items: Iterable[CorrectItem], source_result: str, *, global_scale: float = .1,
                 thresholds: WarningThresholds | None = None, comments: str = "") -> None:
        self.items=list(items); self.source_result=str(source_result); self.global_scale=float(global_scale)
        self.thresholds=thresholds or WarningThresholds(); self.comments=comments
        for item in self.items: item.global_scale=self.global_scale
        self._raw_snapshot=tuple(item.raw_fitted_delta for item in self.items)

    def set_global_scale(self, value: float) -> None:
        if not math.isfinite(value) or value < 0: raise ValueError("Global correction fraction must be finite and non-negative")
        self.global_scale=float(value)
        for item in self.items: item.global_scale=self.global_scale
        self.assert_raw_immutable()

    def assert_raw_immutable(self) -> None:
        if tuple(item.raw_fitted_delta for item in self.items) != self._raw_snapshot: raise RuntimeError("Raw fitted corrections were modified")

    def comparison(self, fractions=(.1,.25,.5,1.0)) -> tuple[dict[str, Any], ...]:
        result=[]
        for fraction in fractions:
            rel=[]; delta_i=[]; violations=0
            for item in self.items:
                if not item.included: continue
                final=item.recommended_machine_delta*float(fraction)*item.individual_scale
                if item.initial_value not in (None,0): rel.append(abs(100*final/item.initial_value))
                if item.ampere_per_unit is not None:
                    change=final*item.ampere_per_unit; delta_i.append(abs(change))
                    if item.current_ampere is not None and item.min_current_ampere is not None and item.max_current_ampere is not None and not item.min_current_ampere <= item.current_ampere+change <= item.max_current_ampere: violations+=1
            calibration_warnings=sum(any(code in {"sign_difference","large_calibration_difference","calibration_unavailable"} for code in item.warnings(self.thresholds)) for item in self.items if item.included)
            unmapped=sum(item.included and item.metadata.get("mapping_status")!="mapped" for item in self.items)
            result.append({"fraction":float(fraction),"max_abs_delta_k_over_k_percent":max(rel,default=None),"max_abs_delta_i_ampere":max(delta_i,default=None),"current_limit_violations":violations,"calibration_warnings":calibration_warnings,"unmapped_magnets":unmapped})
        return tuple(result)

    def to_plan(self, plan_id: str = "correction-plan") -> CorrectionPlan:
        records=[]; states={}
        for item in self.items:
            if item.initial_value is None:
                raise ValueError(f"Cannot save correction record {item.name!r}: initial value is unavailable; it will not be fabricated")
            metadata={**item.metadata,"fitted_value":item.fitted_value,"control_name":item.control_name,"included":item.included,"exclusion_reason":item.exclusion_reason,"evaluated_final_delta":item.final_delta,"machine_value":item.machine_value,"current_ampere":item.current_ampere,"target_current_ampere":item.target_current_ampere,"delta_i_ampere":item.delta_i_ampere,"ampere_per_unit":item.ampere_per_unit,"min_current_ampere":item.min_current_ampere,"max_current_ampere":item.max_current_ampere,"calibration_difference_percent":item.calibration_difference_percent,"sign_difference":item.sign_difference,"warnings":list(item.warnings(self.thresholds)),"sign_convention":item.sign_convention}
            records.append(CorrectionRecord(item.correction_type,item.name,item.lattice_ordinal,item.unit,float(item.initial_value),item.raw_fitted_delta,item.recommended_machine_delta,item.individual_scale,None,item.family,metadata))
            states[str(item.index)]={"included":item.included,"exclusion_reason":item.exclusion_reason}
        provenance={key:self.items[0].metadata.get(key) for key in ("source_results_directory","source_iteration","source_state","fit_timestamp","measurement_session") if self.items and self.items[0].metadata.get(key) is not None}
        return CorrectionPlan(plan_id,self.source_result,tuple(records),self.global_scale,"dry_run",self.comparison(),{"created_utc":datetime.now(timezone.utc).isoformat(),"comments":self.comments,"warning_thresholds":asdict(self.thresholds),"record_states":states,"source_provenance":provenance,"safety":"OFFLINE DRY RUN — no machine setpoints changed"})

    @classmethod
    def from_plan(cls, plan: CorrectionPlan) -> "CorrectionReview":
        items=[]; source_provenance=dict(plan.metadata.get("source_provenance",{}))
        for index,record in enumerate(plan.records):
            meta=dict(record.metadata); included=bool(meta.pop("included",True)); reason=str(meta.pop("exclusion_reason",""))
            meta={**source_provenance,**meta}; target_current=meta.pop("target_current_ampere",None); machine_value=meta.pop("machine_value",None)
            items.append(CorrectItem(index,record.correction_type,record.name,record.lattice_ordinal,record.unit,record.initial_value,meta.pop("fitted_value",None),record.raw_fitted_delta,record.recommended_machine_delta,str(meta.pop("sign_convention","Loaded correction-plan convention")),record.family,meta.pop("control_name",None),plan.global_scale,record.individual_scale,included,reason,meta.pop("current_ampere",None),meta.pop("ampere_per_unit",None),meta.pop("min_current_ampere",None),meta.pop("max_current_ampere",None),meta.pop("calibration_difference_percent",None),bool(meta.pop("sign_difference",False)),meta,machine_value,target_current))
        return cls(items,plan.source_result,global_scale=plan.global_scale,comments=str(plan.metadata.get("comments","")))


def _result_items(path: Path,iteration: int|None=None) -> list[CorrectItem]:
    loader=ResultsLoader(path,iteration=iteration); result=[]; correction=loader.quadrupole_corrections; provenance={"source_results_directory":str(loader.result_dir),"source_iteration":iteration,"source_state":"Final" if iteration is None else f"Iteration {iteration}","fit_timestamp":loader.summary.get("completed_utc",loader.summary.get("timestamp")),"measurement_session":loader.request.get("measurement_session",{})}
    if correction is not None:
        for i,(name,ordinal,initial,fitted,recommended) in enumerate(zip(correction["names"],correction["ordinals"],correction["initial"],correction["fitted"],correction["delta_k_apply"])):
            result.append(CorrectItem(i,"normal_quadrupole",str(name),int(ordinal),"m⁻²",float(initial),float(fitted),float(fitted-initial),float(recommended),str(correction["sign_convention"]),metadata=dict(provenance)))
    offset=len(result); type_map={"skew_quads":"skew_quadrupole","quads_tilt":"quadrupole_tilt"}
    for block in loader.parameter_blocks:
        if block.key not in type_map: continue
        for i,value in enumerate(block.values):
            identity=loader.parameter_identity(block.key,i); initial=None if block.baseline is None else float(block.baseline[i]); raw=float(value)-(initial or 0.0)
            result.append(CorrectItem(offset+i,type_map[block.key],str(identity.get("element_name") or f"{block.label} {i}"),identity.get("lattice_ordinal"),block.unit,initial,float(value),raw,-raw,"Recommended machine Δ = initial model value − fitted model value",metadata=dict(provenance)))
        offset=len(result)
    if not result: raise ValueError("No normal quadrupole, skew quadrupole, or quadrupole-tilt corrections were found in this Results directory")
    return result


def _legacy_items(data: Any) -> list[CorrectItem]:
    entries=data if isinstance(data,list) else data.get("corrections",data.get("records",data.get("quadrupoles",[])))
    if not isinstance(entries,list): raise ValueError("Legacy correction JSON must contain a corrections/records/quadrupoles list")
    result=[]
    for i,row in enumerate(entries):
        if not isinstance(row,dict): raise ValueError(f"Legacy correction entry {i} is not an object")
        name=row.get("name") or row.get("element_name") or row.get("family")
        if not name: raise ValueError(f"Legacy correction entry {i} has no element/family name")
        initial=row.get("initial",row.get("initial_k")); fitted=row.get("fitted",row.get("fitted_k")); raw=row.get("raw_fitted_delta")
        if initial is None or fitted is None: raise ValueError(f"Legacy correction entry {i} must provide explicit initial and fitted values")
        if raw is None and initial is not None and fitted is not None: raw=float(fitted)-float(initial)
        recommended=row.get("recommended_machine_delta",row.get("delta_k_apply"))
        if raw is None or recommended is None: raise ValueError(f"Legacy correction entry {i} must provide raw fitted and recommended machine deltas (or explicit initial/fitted and delta_k_apply)")
        diagnostic_source=row.get("diagnostic_source") or ("Mock — explicit offline values" if str(row.get("control_name","")).startswith("MOCK") else "Source file")
        result.append(CorrectItem(i,str(row.get("correction_type","normal_quadrupole")),str(name),row.get("lattice_ordinal",row.get("ordinal")),str(row.get("unit","m⁻²")),float(initial),float(fitted),float(raw),float(recommended),str(row.get("sign_convention","Explicit legacy values")),family=row.get("family"),control_name=row.get("control_name"),individual_scale=float(row.get("individual_scale",1.0)),included=bool(row.get("included",True)),exclusion_reason=str(row.get("exclusion_reason","")),current_ampere=row.get("current_ampere"),ampere_per_unit=row.get("ampere_per_unit"),min_current_ampere=row.get("min_current_ampere"),max_current_ampere=row.get("max_current_ampere"),calibration_difference_percent=row.get("calibration_difference_percent"),sign_difference=bool(row.get("sign_difference",False)),metadata={"diagnostic_source":diagnostic_source}))
    return result


def load_review(path: str | Path, *, iteration: int|None=None) -> CorrectionReview:
    source=Path(path).expanduser().resolve()
    if source.is_dir(): return CorrectionReview(_result_items(source,iteration),str(source),global_scale=.1)
    data=yaml.safe_load(source.read_text(encoding="utf-8")) if source.suffix.lower() in {".yaml",".yml"} else json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data,dict) and data.get("file_type")=="pyloco.correction_plan":
        if source.suffix.lower()==".json": plan=load_correction_plan(source)
        else: plan=CorrectionPlan(plan_id=str(data.get("plan_id","")),source_result=str(data.get("source_result","")),records=tuple(CorrectionRecord(**entry) for entry in data.get("records",())),global_scale=float(data.get("global_scale",1)),application_state=str(data.get("application_state","dry_run")),fraction_comparison=tuple(data.get("fraction_comparison",())),metadata=dict(data.get("metadata",{})),schema_version=str(data.get("schema_version","")),file_type=str(data.get("file_type",""))); plan.validate()
        review=CorrectionReview.from_plan(plan); stored=Path(review.source_result)
        if not stored.is_absolute(): review.source_result=str((source.parent/stored).resolve())
        return review
    return CorrectionReview(_legacy_items(data),str(source),global_scale=float(data.get("global_scale",.1)) if isinstance(data,dict) else .1)


def apply_mock_diagnostics(review: CorrectionReview) -> None:
    """Attach deterministic, visibly Mock-only current diagnostics; never writes."""
    for item in review.items:
        item.control_name=item.control_name or f"MOCK-PS-{item.index+1:03d}"
        if item.current_ampere is None:item.current_ampere=10.0+(item.index%11)
        if item.ampere_per_unit is None:item.ampere_per_unit=80.0+(item.index%7)*5.0
        if item.min_current_ampere is None:item.min_current_ampere=-100.0
        if item.max_current_ampere is None:item.max_current_ampere=100.0
        item.metadata["diagnostic_source"]="Mock — deterministic offline values"; item.metadata["mapping_status"]="mapped"


def save_review(path: str | Path, review: CorrectionReview) -> Path:
    destination=Path(path).expanduser().resolve(); plan=review.to_plan(destination.stem); source=Path(review.source_result)
    if source.is_absolute(): plan=replace(plan,source_result=os.path.relpath(source,destination.parent))
    if destination.suffix.lower() in {".yaml",".yml"}:
        plan.validate(); destination.write_text(yaml.safe_dump(asdict(plan),sort_keys=False),encoding="utf-8"); return destination
    return save_correction_plan(destination,plan)


def save_review_csv(path: str | Path, review: CorrectionReview) -> Path:
    destination=Path(path); fields=("index","apply","lattice_ordinal","name","control_name","correction_type","initial_value","fitted_value","raw_fitted_delta","recommended_machine_delta","relative_percent","global_scale","individual_scale","final_delta","target_value","current_ampere","target_current_ampere","delta_i_ampere","min_current_ampere","max_current_ampere","calibration_status","current_limit_status","exclusion_reason","warnings","unit","sign_convention")
    with destination.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for item in review.items: writer.writerow({"index":item.index,"apply":item.included,"lattice_ordinal":item.lattice_ordinal,"name":item.name,"control_name":item.control_name,"correction_type":item.correction_type,"initial_value":item.initial_value,"fitted_value":item.fitted_value,"raw_fitted_delta":item.raw_fitted_delta,"recommended_machine_delta":item.recommended_machine_delta,"relative_percent":item.relative_percent,"global_scale":review.global_scale,"individual_scale":item.individual_scale,"final_delta":item.final_delta,"target_value":item.target_value,"current_ampere":item.current_ampere,"target_current_ampere":item.target_current_ampere,"delta_i_ampere":item.delta_i_ampere,"min_current_ampere":item.min_current_ampere,"max_current_ampere":item.max_current_ampere,"calibration_status":item.calibration_status,"current_limit_status":item.current_limit_status,"exclusion_reason":item.exclusion_reason,"warnings":";".join(item.warnings(review.thresholds)),"unit":item.unit,"sign_convention":item.sign_convention})
    return destination
