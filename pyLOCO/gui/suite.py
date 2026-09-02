"""Shared pyLOCO Suite handoff, launch, and Measurement Session inspection."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

from pyLOCO.data_schema import load_session


_DETACHED_SUITE_PIDS: dict[str,int] = {}


@dataclass(frozen=True)
class SessionImport:
    manifest: Path
    session_id: str
    files: dict[str,Path]
    options: dict[str,dict[str,Any]]
    available_roles: tuple[str,...]
    missing_roles: tuple[str,...]
    provenance: dict[str,Any]


def _strings(dataset) -> list[str]:
    return [value.decode() if isinstance(value,bytes) else str(value) for value in np.asarray(dataset)]


def _metadata_json(handle) -> dict[str,Any]:
    if "metadata/json" not in handle:return {}
    value=handle["metadata/json"][()]; value=value.decode() if isinstance(value,bytes) else str(value)
    try:data=json.loads(value)
    except (TypeError,ValueError):return {}
    return data if isinstance(data,dict) else {}


def inspect_measurement_session(path: str|Path) -> SessionImport:
    manifest=Path(path).expanduser().resolve(); session=load_session(manifest); files=session.resolve(manifest); options={}; reference_bpms=None
    for role,source in files.items():
        with h5py.File(source,"r") as handle:
            metadata=_metadata_json(handle); current={"session_role":role,"acquisition_metadata":metadata,"units_convention":str(handle.attrs.get("units_convention","SI"))}
            if "names/bpms" in handle:
                bpms=_strings(handle["names/bpms"]); current["bpm_names"]=bpms
                if reference_bpms is None:reference_bpms=bpms
                elif bpms!=reference_bpms:raise ValueError(f"Measurement Session BPM ordering differs in {role}; no realignment was performed")
            if role=="orm":
                matrix=np.asarray(handle["response_matrix"]); hnames=_strings(handle["names/horizontal_correctors"]); vnames=_strings(handle["names/vertical_correctors"])
                if matrix.shape!=(2*len(current.get("bpm_names",())),len(hnames)+len(vnames)):raise ValueError("ORM shape is inconsistent with stored BPM/corrector ordering")
                row_order=str(handle.attrs.get("row_order","")); column_order=str(handle.attrs.get("column_order",""))
                if row_order!="horizontal_bpms,vertical_bpms" or column_order!="horizontal_correctors,vertical_correctors":raise ValueError("Unsupported ORM ordering; pyLOCO requires H BPM then V BPM rows and H then V corrector columns")
                current.update(dataset="response_matrix",transpose=False,scale=1.0,horizontal_corrector_names=hnames,vertical_corrector_names=vnames,requested_kick_h_rad=np.asarray(handle["kicks/horizontal/requested"]).astype(float).tolist(),requested_kick_v_rad=np.asarray(handle["kicks/vertical/requested"]).astype(float).tolist(),actual_kick_h_rad=np.asarray(handle["kicks/horizontal/actual"]).astype(float).tolist(),actual_kick_v_rad=np.asarray(handle["kicks/vertical/actual"]).astype(float).tolist(),scaled=bool(handle.attrs.get("scaled",False)),direction=str(handle.attrs.get("direction","bipolar")),bidirectional=str(handle.attrs.get("direction","bipolar"))=="bipolar",row_order=row_order,column_order=column_order,response_matrix_unit=str(handle.attrs.get("response_matrix_unit","m")))
            elif role=="dispersion":
                current.update(datasets={"horizontal":"measured_eta_x","vertical":"measured_eta_y"},horizontal_scale=1.0,vertical_scale=1.0,rf_step_hz=float(handle.attrs["rf_step_hz"]),bidirectional=bool(handle.attrs.get("bidirectional",False)),measured_eta_definition=str(handle.attrs.get("measured_eta_definition","orbit difference for rf_step_hz")),restoration_status=str(handle.attrs.get("restoration_status","not_verified")))
            elif role=="bpm_noise":current.update(datasets={"horizontal":"Noise_BPMx","vertical":"Noise_BPMy"},horizontal_scale=1.0,vertical_scale=1.0)
            options[role]=current
    timestamps={role:options[role].get("acquisition_metadata",{}).get("timestamp_utc") for role in options}
    provenance={"session_id":session.session_id,"manifest":str(manifest),"measurement_files":{role:str(value) for role,value in files.items()},"acquisition_timestamps":timestamps,"session_metadata":session.metadata}
    return SessionImport(manifest,session.session_id,files,options,tuple(role for role in ("orm","bpm_noise","dispersion") if role in files),session.missing_roles,provenance)


def launch_suite_application(application: str, *arguments: str) -> tuple[bool,str]:
    modules={"fit":"pyLOCO.gui.app","measure":"pyLOCO.measure.app","correct":"pyLOCO.correct.app"}
    if application not in modules:raise ValueError(f"Unknown pyLOCO Suite application: {application}")
    previous=_DETACHED_SUITE_PIDS.get(application)
    if previous is not None:
        try:os.kill(previous,0)
        except ProcessLookupError:_DETACHED_SUITE_PIDS.pop(application,None)
        except PermissionError:return True,f"already running (process {previous})"
        else:return True,f"already running (process {previous})"
    from PySide6.QtCore import QProcess
    ok,pid=QProcess.startDetached(sys.executable,["-m",modules[application],*[str(value) for value in arguments]])
    if ok:_DETACHED_SUITE_PIDS[application]=int(pid)
    return bool(ok),f"process {pid}" if ok else f"Could not launch {application}"


def present_single_about_dialog(owner, factory, *, attribute: str = "_about_dialog"):
    """Open one reusable About dialog as a non-blocking child of ``owner``.

    ``QDialog.open`` with window modality gives macOS a genuine parent-attached
    sheet instead of a second independent application window.  It does not
    enter a nested event loop, replace central content, or hide the owner.
    """
    from PySide6.QtCore import Qt

    dialog=getattr(owner,attribute,None)
    if dialog is None:
        dialog=factory(); dialog.setParent(owner,Qt.Dialog); dialog.setWindowModality(Qt.WindowModal); setattr(owner,attribute,dialog)
    maximum_width=max(440,min(720,owner.width()-80)); maximum_height=max(480,min(640,owner.height()-60))
    dialog.setMaximumSize(maximum_width,maximum_height)
    dialog.resize(min(dialog.width(),maximum_width),min(dialog.height(),maximum_height))
    if not dialog.isVisible(): dialog.open()
    dialog.raise_(); dialog.activateWindow()
    # Non-sheet platforms use this centered position; native macOS sheets
    # ignore the move and remain attached to the parent title bar.
    dialog.move(owner.mapToGlobal(owner.rect().center())-dialog.rect().center())
    return dialog
