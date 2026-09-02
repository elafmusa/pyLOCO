"""Lazy, hard read-only PETRA/DOOCS adapter.

No DOOCS package is imported until a PETRA operation is explicitly requested.
"""
from __future__ import annotations

from importlib import import_module
from time import time
from typing import Any, Iterable, Sequence

import numpy as np

from .adapters import AdapterCapability, ChannelSample, ReadOnlyAdapter

BPM_ADDRESS_X = "PETRA/REFORBIT/*/SA_X_RAW"
BPM_ADDRESS_Y = "PETRA/REFORBIT/*/SA_Y_RAW"
MAGNET_BASE = "PETRA/MAGNET.ML"
CALIBRATION_BASE = "PETRA.MAGNETS/MAGNET.ML"
NM_TO_M = 1e-9
READ_ONLY_ERROR = "Machine writes are disabled in PETRA read-only mode."


class OptionalDependencyUnavailable(RuntimeError):
    pass


class PETRAReadOnlyAdapter(ReadOnlyAdapter):
    """Read-only access matching the established PETRA measurement scripts."""

    def __init__(self, bpm_names: Sequence[str] = (), horizontal_corrector_names: Sequence[str] = (),
                 vertical_corrector_names: Sequence[str] = (), *, pydoocs_module=None,
                 doocs4py_module=None) -> None:
        self.bpm_names=tuple(str(name).strip() for name in bpm_names if str(name).strip())
        self.horizontal_corrector_names=tuple(str(name).strip() for name in horizontal_corrector_names if str(name).strip())
        self.vertical_corrector_names=tuple(str(name).strip() for name in vertical_corrector_names if str(name).strip())
        self._pydoocs_module=pydoocs_module; self._doocs4py_module=doocs4py_module
        self.history: list[tuple[str,str,Any]]=[]

    @property
    def capabilities(self):
        return frozenset({AdapterCapability.READ,AdapterCapability.BATCH_READ})

    def _pydoocs(self):
        if self._pydoocs_module is None:
            try:self._pydoocs_module=import_module("pydoocs")
            except ImportError as exc:raise OptionalDependencyUnavailable("PETRA / DOOCS requires the optional 'pydoocs' package.") from exc
        return self._pydoocs_module

    def _doocs4py(self):
        if self._doocs4py_module is None:
            try:self._doocs4py_module=import_module("doocs4py")
            except ImportError as exc:raise OptionalDependencyUnavailable("PETRA calibration helpers require the optional 'doocs4py' package.") from exc
        return self._doocs4py_module

    @staticmethod
    def _wildcard_values(response, address):
        data=response.get("data") if isinstance(response,dict) else None
        if data is None:raise ValueError(f"DOOCS response for {address} has no data field")
        rows=list(data)
        if len(rows)<2:raise ValueError(f"DOOCS response for {address} does not contain the expected two trailing metadata rows")
        rows=rows[:-2]
        try:values=np.asarray([row[1] for row in rows],dtype=float)
        except Exception as exc:raise ValueError(f"Unexpected DOOCS wildcard data format for {address}") from exc
        labels=tuple(str(row[0]) for row in rows) if all(isinstance(row[0],str) for row in rows) else ()
        return values,labels

    def read_orbit(self):
        api=self._pydoocs(); response_x=api.read(BPM_ADDRESS_X); response_y=api.read(BPM_ADDRESS_Y)
        x,labels_x=self._wildcard_values(response_x,BPM_ADDRESS_X); y,labels_y=self._wildcard_values(response_y,BPM_ADDRESS_Y); x=x*NM_TO_M; y=y*NM_TO_M
        if x.size!=y.size:raise ValueError(f"PETRA BPM X/Y lengths differ: {x.size} != {y.size}")
        if labels_x and labels_y and labels_x!=labels_y:raise ValueError("PETRA BPM X/Y wildcard device ordering differs")
        if self.bpm_names and x.size!=len(self.bpm_names):raise ValueError(f"PETRA orbit contains {x.size} BPMs but the configured BPM list contains {len(self.bpm_names)}; no truncation or reordering was performed")
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("PETRA BPM orbit contains non-finite values")
        raw_labels=labels_x or labels_y or tuple(str(index) for index in range(x.size)); configured=self.bpm_names or raw_labels; self.last_orbit_mapping=tuple(zip(configured,raw_labels))
        self.history.extend((("read",BPM_ADDRESS_X,x.copy()),("read",BPM_ADDRESS_Y,y.copy())))
        return x,y

    @staticmethod
    def bpm_channel(name: str, plane: str) -> str:
        return f"PETRA:BPM:{name}:{plane.upper()}"

    def read_many(self, channels: Iterable[str]):
        requested=tuple(channels); x,y=self.read_orbit(); names=self.bpm_names or tuple(str(i) for i in range(x.size)); lookup={}
        stamp=time()
        for index,name in enumerate(names):
            lookup[self.bpm_channel(name,"X")]=ChannelSample(self.bpm_channel(name,"X"),float(x[index]),stamp)
            lookup[self.bpm_channel(name,"Y")]=ChannelSample(self.bpm_channel(name,"Y"),float(y[index]),stamp)
        missing=[channel for channel in requested if channel not in lookup]
        if missing:raise KeyError("Unknown PETRA BPM channel(s): "+", ".join(missing))
        return {channel:lookup[channel] for channel in requested}

    def read(self, channel: str):
        if channel.startswith("PETRA:BPM:"):return self.read_many((channel,))[channel]
        response=self._pydoocs().read(channel); value=response.get("data") if isinstance(response,dict) else response
        result=ChannelSample(channel,float(np.asarray(value)),time()); self.history.append(("read",channel,result.value)); return result

    def write(self,*_args,**_kwargs):
        raise PermissionError(READ_ONLY_ERROR)

    @staticmethod
    def magnet_address(name: str, field: str) -> str:
        if field not in {"KICK.SP","KICK.RBV","STRENGTH.SP","CURRENT.SP","CURRENT.RBV"}:raise ValueError(f"Unsupported PETRA magnet diagnostic field: {field}")
        return f"{MAGNET_BASE}/{name}/{field}"

    def read_corrector_diagnostics(self,name: str):
        return {field:self.read(self.magnet_address(name,field)).value for field in ("KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV")}

    def read_quadrupole_diagnostics(self,name: str):
        """Read verified PETRA quadrupole state without guessing unavailable channels.

        ``STRENGTH.SP`` is authoritative for the current machine strength.  Current
        readback is preferred and the current setpoint is used only when readback is
        unavailable.  Missing current channels remain explicit rather than being
        reconstructed through an unverified CURRENT2STRENGTH conversion.
        """
        strength=self.read(self.magnet_address(name,"STRENGTH.SP")).value
        current=None; current_source="unavailable"
        for field in ("CURRENT.RBV","CURRENT.SP"):
            try:
                current=self.read(self.magnet_address(name,field)).value; current_source=field; break
            except Exception:
                continue
        return {"strength_sp":float(strength),"current_ampere":None if current is None else float(current),"current_source":current_source}

    def strength_to_current(self,name: str,strength: float) -> float:
        result=self._doocs4py().get(f"{CALIBRATION_BASE}/{name}/STRENGTH2CURRENT",float(strength)); return float(result.value)

    def current_to_strength(self,name: str,current: float) -> float:
        raise NotImplementedError("No verified PETRA CURRENT2STRENGTH mapping exists in the repository scripts.")

    def current_limits(self,name: str):
        api=self._doocs4py(); minimum=api.get(f"{CALIBRATION_BASE}/{name}/MIN_CURRENT_1").value; maximum=api.get(f"{CALIBRATION_BASE}/{name}/MAX_CURRENT_1").value; return float(minimum),float(maximum)

    def list_devices(self,kind: str):
        if kind=="bpm":return tuple({"name":name,"x_channel":self.bpm_channel(name,"X"),"y_channel":self.bpm_channel(name,"Y")} for name in self.bpm_names)
        names=self.horizontal_corrector_names if kind=="hcor" else self.vertical_corrector_names if kind=="vcor" else ()
        plane="Horizontal" if kind=="hcor" else "Vertical"
        return tuple({"name":name,"setpoint_channel":self.magnet_address(name,"KICK.SP"),"readback_channel":self.magnet_address(name,"KICK.RBV"),"plane":plane} for name in names)

    def test_connection(self, corrector_limit: int = 3):
        x,y=self.read_orbit(); diagnostics={}
        for name in (self.horizontal_corrector_names+self.vertical_corrector_names)[:max(0,corrector_limit)]:diagnostics[name]=self.read_corrector_diagnostics(name)
        calibration="unavailable"
        if diagnostics:
            try:self.current_limits(next(iter(diagnostics))); calibration="available"
            except Exception:calibration="unavailable"
        return {"bpms":int(x.size),"bpm_orbit":"available","corrector_readback":"available" if diagnostics else "unavailable","calibration":calibration,"rf_readback":"unavailable","correctors":diagnostics}
