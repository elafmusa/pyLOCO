"""pySC control-system server interface used by the DEMO backend."""
from __future__ import annotations

from typing import Any

import numpy as np
from pySC.apps.interface import AbstractInterface
from pySC.control_system.client import read, write


class pySCServerOrbitInterface(AbstractInterface):
    """Connect pyLOCO to ``pySC.control_system.server`` over TCP."""

    host: str = "127.0.0.1"
    port: int = 13131
    rf_system: str = "main"
    use_design: bool = False
    bba: bool = True
    subtract_reference: bool = True

    model_config = {"arbitrary_types_allowed": True}

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x = self._read_array("ORBIT/RAW/X")
        y = self._read_array("ORBIT/RAW/Y")
        if x.shape != y.shape:
            raise RuntimeError(f"pySC server orbit shape mismatch: X{x.shape}, Y{y.shape}")
        return x, y

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.get_orbit()
        return np.zeros_like(x), np.zeros_like(y)

    def get(self, name: str) -> float:
        return float(self._required_read(f"MAGNET/{name}"))

    def set(self, name: str, value: float) -> None:
        self._write(f"MAGNET/{name}", value)

    def get_many(self, names: list) -> dict[str, float]:
        return {str(name): self.get(str(name)) for name in names}

    def set_many(self, data: dict[str, float]) -> None:
        for name, value in data.items():
            self.set(name, value)

    def get_rf_main_frequency(self) -> float:
        return float(self._required_read(f"RF/{self.rf_system}/FREQUENCY"))

    def set_rf_main_frequency(self, frequency: float) -> None:
        self._write(f"RF/{self.rf_system}/FREQUENCY", frequency)

    def _read_array(self, variable: str) -> np.ndarray:
        array = np.asarray(self._required_read(variable), dtype=float)
        if array.ndim != 1:
            raise RuntimeError(f"pySC server returned non-1D array for {variable}: {array.shape}")
        return array

    def _required_read(self, variable: str) -> Any:
        value = read(self._address(variable))
        if value is None:
            raise RuntimeError(f"pySC server returned no value for {variable}")
        return value

    def _write(self, variable: str, value: float) -> None:
        write(self._address(variable), float(value))

    def _address(self, variable: str) -> str:
        return f"{self.host}:{self.port}/{variable}"
