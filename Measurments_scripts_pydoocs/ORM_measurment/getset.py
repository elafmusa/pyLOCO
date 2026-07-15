import numpy as np
import datetime
import time
import logging
import pydoocs
from p3_interface import petra3Interface

interface = petra3Interface()
BPM_ADDRESS_X = 'PETRA/REFORBIT/*/SA_X_BBAGO'
BPM_ADDRESS_Y = 'PETRA/REFORBIT/*/SA_Y_BBAGO'
NM_TO_M = 1e-9


def get_pydoocs_orbit():
    data_x = pydoocs.read(BPM_ADDRESS_X)
    data_y = pydoocs.read(BPM_ADDRESS_Y)
    orbit_x = np.array([dd[1] for dd in data_x['data'][:-2]])
    orbit_y = np.array([dd[1] for dd in data_y['data'][:-2]])
    orbit_x *= NM_TO_M
    orbit_y *= NM_TO_M
    return orbit_x, orbit_y

def get_average_orbit(n_orbits=10, dt=0.1):
    orbit_x, orbit_y = get_pydoocs_orbit()
    all_orbit_x = np.zeros((len(orbit_x), n_orbits))
    all_orbit_y = np.zeros((len(orbit_y), n_orbits))
    
    all_orbit_x[:,0] = orbit_x
    all_orbit_y[:,0] = orbit_y
    for ii in range(1, n_orbits):
        time.sleep(dt)
        all_orbit_x[:, ii], all_orbit_y[:, ii] = get_pydoocs_orbit()
    
    mean_orbit_x = np.mean(all_orbit_x, axis=1)
    mean_orbit_y = np.mean(all_orbit_y, axis=1)
    std_orbit_x = np.std(all_orbit_x, axis=1)
    std_orbit_y = np.std(all_orbit_y, axis=1)
    return mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y

def get_setpoint(magnet: str, kick: bool = True) -> float:
	return interface.get(magnet)
        
def set_setpoint(magnet: str, value: float, kick: bool = True):
    interface.set(magnet, value, wait=True)
    return