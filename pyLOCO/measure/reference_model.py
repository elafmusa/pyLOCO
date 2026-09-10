"""Optional reference-lattice metadata and comparison helpers for Measure.

This module is deliberately outside the GUI: it performs no acquisition and
never communicates with a control system.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import yaml

from pyLOCO.control_system.pysc_profiles import load_pysc_profile


@dataclass(frozen=True)
class ReferenceModel:
    path: Path
    checksum_sha256: str
    source: str
    ring: object
    tune: tuple[float, float]
    momentum_compaction: float
    inverse_gamma_squared: float
    slip_factor: float
    nominal_rf_hz: float | None
    energy_ev: float = float("nan")
    circumference_m: float = float("nan")
    revolution_frequency_hz: float = float("nan")
    harmonic_number: int | None = None
    harmonic_number_source: str = "Not available"
    rf_source: str = "Not available"
    rf_harmonic_residual_hz: float | None = None
    chromaticity: tuple[float, float] = (float("nan"), float("nan"))

    def provenance(self) -> dict[str, object]:
        return {"reference_model_path":str(self.path),"reference_model_sha256":self.checksum_sha256,
                "reference_model_source":self.source,"model_tune_qx":self.tune[0],"model_tune_qy":self.tune[1],
                "model_momentum_compaction":self.momentum_compaction,"model_inverse_gamma_squared":self.inverse_gamma_squared,
                "model_slip_factor":self.slip_factor,"model_nominal_rf_hz":self.nominal_rf_hz,
                "model_energy_ev":self.energy_ev,"model_circumference_m":self.circumference_m,
                "model_revolution_frequency_hz":self.revolution_frequency_hz,
                "model_harmonic_number":self.harmonic_number,"model_harmonic_number_source":self.harmonic_number_source,
                "model_rf_source":self.rf_source,"model_rf_harmonic_residual_hz":self.rf_harmonic_residual_hz,
                "model_chromaticity_x":self.chromaticity[0],"model_chromaticity_y":self.chromaticity[1]}


def _mat_key(profile) -> str | None:
    if profile.configuration.get("state_file"):
        data=json.loads(profile.resolve("state_file").read_text()); return data.get("lattice",{}).get("use")
    if profile.configuration.get("configuration_file"):
        data=yaml.safe_load(profile.resolve("configuration_file").read_text()); return data.get("lattice",{}).get("use")
    return None


def load_reference_model(path: str | Path, *, source: str="User-selected reference model", mat_key: str | None=None) -> ReferenceModel:
    import at
    source_path=Path(path).expanduser().resolve()
    ring=at.load_mat(str(source_path),mat_key=mat_key) if mat_key else at.load_lattice(str(source_path))
    ring=ring.disable_6d(copy=True)
    _,ringdata,_=ring.get_optics(get_chrom=True)
    tune=tuple(float(v) for v in np.asarray(ringdata.tune)[:2])
    alpha=float(ring.get_mcf())
    # AT reports 1/gamma^2-alpha; derive both displayed terms from the same
    # lattice API and avoid maintaining a second rest-energy constant.
    at_slip=float(ring.get_slip_factor()); inverse=float(alpha+at_slip)
    slip=float(-at_slip)
    frequencies=[float(getattr(elem,"Frequency")) for elem in ring if hasattr(elem,"Frequency") and float(getattr(elem,"Frequency"))>0]
    rf=float(np.median(frequencies)) if frequencies else None
    energy=float(ring.energy); circumference=float(ring.circumference)
    frev=float(at.constants.clight/circumference)
    cavities=[elem for elem in ring if hasattr(elem,"Frequency")]
    explicit=getattr(ring,"harmonic_number",None)
    cavity_harmonics=[int(elem.HarmNumber) for elem in cavities if getattr(elem,"HarmNumber",None)]
    harmonic=int(explicit) if explicit else (cavity_harmonics[0] if cavity_harmonics and len(set(cavity_harmonics))==1 else None)
    harmonic_source="Reference lattice / AT" if explicit else ("Reference lattice cavities" if harmonic else "Not available")
    residual=float(rf-harmonic*frev) if rf is not None and harmonic is not None else None
    chrom=tuple(float(v) for v in np.asarray(ringdata.chromaticity)[:2])
    return ReferenceModel(source_path,sha256(source_path.read_bytes()).hexdigest(),source,ring,tune,alpha,inverse,slip,rf,
                          energy,circumference,frev,harmonic,harmonic_source,
                          "Reference lattice cavities" if rf is not None else "Not available",residual,chrom)


def reference_model_for_pysc(profile_key: str) -> ReferenceModel:
    profile=load_pysc_profile(profile_key)
    return load_reference_model(profile.resolve("lattice_file"),source=f"pySC profile manifest — {profile.label} / {profile.scenario}",mat_key=_mat_key(profile))


def comparison_metrics(measured, model) -> dict[str,float]:
    measured=np.asarray(measured,float); model=np.asarray(model,float); difference=measured-model
    measured_norm=float(np.linalg.norm(measured)); model_norm=float(np.linalg.norm(model))
    denom=measured_norm*model_norm
    return {"rms_measured":float(np.sqrt(np.mean(measured**2))),"rms_model":float(np.sqrt(np.mean(model**2))),
            "rms_difference":float(np.sqrt(np.mean(difference**2))),
            "max_abs_difference":float(np.max(np.abs(difference))) if difference.size else float("nan"),
            "relative_norm_difference":float(np.linalg.norm(difference)/model_norm) if model_norm else float("nan"),
            "cosine_similarity":float(np.vdot(measured,model).real/denom) if denom else float("nan"),
            "fitted_gain":float(np.vdot(model,measured).real/(model_norm**2)) if model_norm else float("nan")}


def resolve_device_ordinals(model: ReferenceModel, devices) -> tuple[int, ...]:
    """Resolve adapter devices by explicit numeric identifier or stable lattice name."""
    names={}
    for index,element in enumerate(model.ring):
        for attr in ("CommonName","FamName","name"):
            value=getattr(element,attr,None)
            if value: names.setdefault(str(value),[]).append(index)
    result=[]
    for device in devices:
        identifier=str(getattr(device,"identifier",device))
        candidates=[]
        for value in (str(getattr(device,"name",identifier)),identifier):
            for part in value.split("|"):
                token=part.strip()
                if token.startswith(("MAGNET:","BPM:")):token=token.split(":",1)[1]
                if token.endswith((":X",":Y")):token=token[:-2]
                candidates.extend((token,token.rsplit("/",1)[0]))
        numeric=next((item for item in candidates if item.isdigit()),None)
        if numeric is not None:ordinal=int(numeric)
        else:
            matches=[]
            for candidate in dict.fromkeys(candidates):
                candidate_matches=names.get(candidate,[])
                if len(candidate_matches)==1:
                    matches=candidate_matches; break
            if len(matches)!=1: raise ValueError(f"Reference-model device mapping is not unique: {identifier}")
            ordinal=matches[0]
        result.append(ordinal)
    return tuple(result)


def model_dispersion(model: ReferenceModel, bpm_devices) -> tuple[np.ndarray,np.ndarray]:
    ordinals=resolve_device_ordinals(model,bpm_devices)
    order=np.argsort(ordinals); sorted_ordinals=np.asarray(ordinals)[order]
    _,_,data=model.ring.get_optics(refpts=sorted_ordinals)
    dispersion=np.asarray(data.dispersion)
    inverse=np.argsort(order)
    return dispersion[inverse,0].copy(),dispersion[inverse,2].copy()


def model_orm(model: ReferenceModel, bpm_devices, horizontal_correctors, vertical_correctors,
              kick_h_rad, kick_v_rad, *, scaled: bool) -> np.ndarray:
    """Calculate selected-device ORM through pyLOCO's validated implementation."""
    from pyLOCO.config import RMConfig
    from pyLOCO.response_matrix import response_matrix
    bpm=resolve_device_ordinals(model,bpm_devices); h=resolve_device_ordinals(model,horizontal_correctors); v=resolve_device_ordinals(model,vertical_correctors)
    bo=np.argsort(bpm); ho=np.argsort(h); vo=np.argsort(v)
    kh=np.asarray(kick_h_rad,float); kv=np.asarray(kick_v_rad,float)
    matrix=np.asarray(response_matrix(model.ring,config=RMConfig(bpm_ords=np.asarray(bpm)[bo],cm_ords=(np.asarray(h)[ho],np.asarray(v)[vo]),cav_ords=[],dkick=(kh[ho],kv[vo]),bidirectional=True,includeDispersion=False,calculator="Linear")))
    nb=len(bpm); matrix=np.vstack((matrix[:nb][np.argsort(bo)],matrix[nb:][np.argsort(bo)]))
    columns=np.concatenate((np.argsort(ho),len(h)+np.argsort(vo))); matrix=matrix[:,columns]
    if scaled:
        separation=np.concatenate((kh,kv)); matrix=matrix/separation[np.newaxis,:]
    return matrix


def store_reference_model_arrays(path: str | Path, arrays: dict[str, np.ndarray], *, units: dict[str,str], provenance: dict[str,object]) -> None:
    """Append diagnostic MODEL arrays without touching acquired datasets."""
    import h5py
    with h5py.File(path,"a") as handle:
        group=handle.require_group("reference_model")
        for name,value in arrays.items():
            if name in group:del group[name]
            dataset=group.create_dataset(name,data=np.asarray(value)); dataset.attrs["unit"]=units[name]
        for key,value in provenance.items():
            if value is not None and isinstance(value,(str,int,float,bool,np.integer,np.floating)):group.attrs[key]=value
