from __future__ import annotations

from typing import Any, Tuple

import doocs4py
import time
import numpy as np
from pySC.apps.interface import AbstractInterface
import pydoocs

BPM_ADDRESS_X = "PETRA/REFORBIT/*/SA_X_RAW"
BPM_ADDRESS_Y = "PETRA/REFORBIT/*/SA_Y_RAW"
MAGNETML = "PETRA/MAGNET.ML"
NM_TO_M = 1e-9

CONVERSION_WAIT = 0.1 #s

PKDK_mags = pydoocs.names('PETRA/MAGNET.ML/PKDK*')
PKDK_ratio = 5
PKDK_ref_mag = 'DKA_SWL_27'
PKPDA_mags = pydoocs.names('PETRA/MAGNET.ML/PKPDA*')
PKPDA_ratio = 2.2
PKPDA_ref_mag = 'PDA_OR_99B'
PKPDAX_mags = ['PKPDA_NR_66', 'PKPDA_NR_77', 'PKPDA_NR_99', 'PKPDA_OR_66', 'PKPDA_OR_77', 'PKPDA_OR_99']
PKPDAX_ratio = 2.2
PKPDAX_ref_mag = 'PDAA_NOR_37'
PKPDD_mags = pydoocs.names('PETRA/MAGNET.ML/PKPDD*')
PKPDD_ratio = 2.2
PKPDD_ref_mag = 'PDD_NR_87A'
PKPDE_map = {'PKPDE_NR_72': 'PDE_NR_72',
             'PKPDE_NR_93': 'PDE_NR_93',
             'PKPDE_OR_72': 'PDE_OR_72',
             'PKPDE_OR_93': 'PDE_OR_93'}
PKPDC_map = {'PKPDC_NOR_53': 'PDC_NOR_53',
             'PKPDC_NOR_99': 'PDC_NOR_99',
             'PKPDC_OL_143': 'PDC_OL_143',
             'PKPDC_OL_97': 'PDC_OL_97',
             'PKPDC_OL_74': 'PDC_OL_74'
             }

ALL_PD_mags = PKPDA_mags + PKPDAX_mags + PKPDD_mags + list(PKPDE_map.keys()) + list(PKPDC_map.keys())

class petra3Interface(AbstractInterface):
    """pySC correction interface backed by a running pySC control-system server."""

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        data_x = doocs4py.get(BPM_ADDRESS_X)
        data_y = doocs4py.get(BPM_ADDRESS_Y)
        orbit_x = np.array([dd[1] for dd in data_x['data'][:-2]])
        orbit_y = np.array([dd[1] for dd in data_y['data'][:-2]])
        orbit_x *= NM_TO_M
        orbit_y *= NM_TO_M
        if orbit_x.shape != orbit_y.shape:
            raise RuntimeError(f"pySC server orbit shape mismatch: X{orbit_x.shape}, Y{orbit_y.shape}")
        return orbit_x, orbit_y

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.get_orbit()
        return np.zeros_like(x), np.zeros_like(y)

    def get(self, name_: str) -> float:
        name = self._check_name(name_)
        backleg, ref_mag, ratio = self._get_backleg_info(name)
        if backleg:
            current_ref = doocs4py.get(f"{MAGNETML}/{ref_mag}/CURRENT.SP")["data"]
            current_backleg_0 = doocs4py.get(f"{MAGNETML}/{name}/CURRENT.SP")["data"]
            total_current = current_ref + ratio*current_backleg_0
            doocs4py.set(f"{MAGNETML}/{ref_mag}/CURRENT2KICK", total_current)
            time.sleep(CONVERSION_WAIT)
            total_kick = doocs4py.set(f"{MAGNETML}/{ref_mag}/CURRENT2KICK_RESULT")["data"]
            ref_kick = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK.SP")["data"]
            setpoint = total_kick - ref_kick
        else:
            setpoint = doocs4py.get(f"{MAGNETML}/{name}/KICK.SP")["data"]

        return float(setpoint)

    def set(self, name_: str, value: float) -> None:
        name = self._check_name(name_)
        backleg, ref_mag, ratio = self._get_backleg_info(name)
        if backleg:
            ref_kick = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK.SP")["data"]
            total_kick = ref_kick + value
            doocs4py.set(f"{MAGNETML}/{ref_mag}/KICK2CURRENT", total_kick)
            time.sleep(CONVERSION_WAIT)
            total_current = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK2CURRENT_RESULT")["data"]
            ref_current = doocs4py.get(f"{MAGNETML}/{ref_mag}/CURRENT.SP")["data"]
            current_backleg = (total_current - ref_current) / ratio
            doocs4py.set(f"{MAGNETML}/{name}/CURRENT.SP", current_backleg)
        else:
            doocs4py.set(f"{MAGNETML}/{name}/KICK.SP", value)

    def get_many(self, names_: list) -> dict[str, float]:
        backlegs = {}
        ref_mags = {}
        ratios = {}
        names = [self._check_name(nn) for nn in names_]
        for name in names:
            backlegs[name], ref_mags[name], ratios[name] = self._get_backleg_info(name)

        setpoints = {}
        for name in names:
            if backlegs[name]:
                ref_mag = ref_mags[name]
                current_ref = doocs4py.get(f"{MAGNETML}/{ref_mag}/CURRENT.SP")["data"]
                current_backleg_0 = doocs4py.get(f"{MAGNETML}/{name}/CURRENT.SP")["data"]
                total_current = current_ref + ratios[name]*current_backleg_0
                doocs4py.set(f"{MAGNETML}/{name}/CURRENT2KICK", total_current)

                time.sleep(CONVERSION_WAIT)
        # for name in names:
        #     if backlegs[name]:
                total_kick = doocs4py.set(f"{MAGNETML}/{ref_mag}/CURRENT2KICK_RESULT")["data"]
                ref_kick = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK.SP")["data"]
                setpoint = total_kick - ref_kick
            else:
                setpoint = doocs4py.get(f"{MAGNETML}/{name}/KICK.SP")["data"]
            setpoints[name] = setpoint
        return setpoints

    def set_many(self, data: dict[str, float]) -> None:
        backlegs = {}
        ref_mags = {}
        ratios = {}
        names = [self._check_name(nn) for nn in list(data.keys())]
        for name in names:
            backlegs[name], ref_mags[name], ratios[name] = self._get_backleg_info(name)

        currents_to_send = {}
        kicks_to_send = {}
        for name in names:
            value = data[name]
            if backlegs[name]:
                ref_mag = ref_mags[name]
                ref_kick = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK.SP")["data"]
                total_kick = ref_kick + value
                doocs4py.set(f"{MAGNETML}/{ref_mag}/KICK2CURRENT", total_kick)

        # time.sleep(CONVERSION_WAIT)
        # for name in names:
        #     value = data[name]
        #     if backlegs[name]:
                ref_mag = ref_mags[name]
                total_current = doocs4py.get(f"{MAGNETML}/{ref_mag}/KICK2CURRENT_RESULT")["data"]
                ref_current = doocs4py.get(f"{MAGNETML}/{ref_mag}/CURRENT.SP")["data"]
                current_backleg = (total_current - ref_current) / ratios[name]
                currents_to_send[name] = current_backleg
            else:
                kicks_to_send[name] = value

        for name in currents_to_send.keys():
            if name in ALL_PD_mags:
                continue
            doocs4py.set(f"{MAGNETML}/{name}/CURRENT.SP", currents_to_send[name])
        for name in kicks_to_send.keys():
            doocs4py.set(f"{MAGNETML}/{name}/KICK.SP", kicks_to_send[name])

    def get_rf_main_frequency(self) -> float:
        return float(0)

    def set_rf_main_frequency(self, frequency: float) -> None:
        pass

    def _check_name(self, name: str):
        if name in PKPDE_map.keys():
            new_name = PKPDE_map[name]
        elif name in PKPDC_map.keys():
            new_name = PKPDC_map[name]
        else:
            new_name = name
        return new_name

    def _get_backleg_info(self, name: str) -> Tuple[bool, str, float]:
        if name in PKDK_mags:
            backleg = True
            ref_mag = PKDK_ref_mag
            ratio = PKDK_ratio
        elif name in PKPDA_mags and name not in PKPDAX_mags:
            backleg = True
            ref_mag = PKPDA_ref_mag
            ratio = PKPDA_ratio
        elif name in PKPDAX_mags:
            backleg = True
            ref_mag = PKPDAX_ref_mag
            ratio = PKPDAX_ratio
        elif name in PKPDD_mags:
            backleg = True
            ref_mag = PKPDD_ref_mag
            ratio = PKPDD_ratio
        else:
            ref_mag = ""
            ratio = 0
            backleg = False

        return backleg, ref_mag, ratio
