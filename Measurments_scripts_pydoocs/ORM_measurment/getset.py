import numpy as np
import datetime
import time
import logging
import pydoocs

BPM_ADDRESS_X = 'PETRA/REFORBIT/*/SA_X_BBAGO'
BPM_ADDRESS_Y = 'PETRA/REFORBIT/*/SA_Y_BBAGO'
NM_TO_M = 1e-9
RBV_TOL = 5e-1
MAX_SLEEP = 120
RBV_WAIT = 0.1


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
def get_setpoint(magnet: str, kick: bool = True) -> float:

    if magnet in PKDK_mags:
        backleg = True
        ref_mag = PKDK_ref_mag
        ratio = PKDK_ratio
    elif magnet in PKPDA_mags and not magnet in PKPDAX_mags:
        backleg = True
        ref_mag = PKPDA_ref_mag
        ratio = PKPDA_ratio
    elif magnet in PKPDAX_mags:
        backleg = True
        ref_mag = PKPDAX_ref_mag
        ratio = PKPDAX_ratio
    elif magnet in PKPDD_mags:
        backleg = True
        ref_mag = PKPDD_ref_mag
        ratio = PKPDD_ratio
    else:
        backleg = False
        
        
    if kick:
        address = f'PETRA/MAGNET.ML/{magnet}/KICK.SP'
        if backleg:
            current_ref = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT.SP')['data']
            current_backleg_0 = pydoocs.read(f'PETRA/MAGNET.ML/{magnet}/CURRENT.SP')['data']
            total_current = current_ref + ratio*current_backleg_0
            pydoocs.write(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT2KICK', total_current)
            time.sleep(0.1)
            total_kick = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT2KICK_RESULT')['data']
            ref_kick = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/KICK.SP')['data']
            return total_kick - ref_kick

            
    else:
        address = f'PETRA/MAGNET.ML/{magnet}/STRENGTH.SP'
        if backleg:
            current_ref = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT.SP')['data']
            current_backleg_0 = pydoocs.read(f'PETRA/MAGNET.ML/{magnet}/CURRENT.SP')['data']
            total_current = current_ref + ratio*current_backleg_0
            pydoocs.write(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT2STRENGTH', total_current)
            time.sleep(0.1)
            total_strength = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT2STRENGTH_RESULT')['data']
            ref_strength = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/STRENGTH.SP')['data']
            return total_strength - ref_strength
        
    return pydoocs.read(address)['data']


def wait_on_current(magnet: str):
    rbv = pydoocs.read(f'PETRA/MAGNET.ML/{magnet}/CURRENT.RBV')['data']
    sp = pydoocs.read(f'PETRA/MAGNET.ML/{magnet}/CURRENT.SP')['data']
    time_slept = 0
    while abs(rbv - sp) > RBV_TOL:
        time.sleep(RBV_WAIT)
        time_slept += RBV_WAIT
        if time_slept > MAX_SLEEP:
            raise Exception('Have been waiting for far too long.')
        rbv = pydoocs.read(f'PETRA/MAGNET.ML/{magnet}/CURRENT.RBV')['data']
    return


def set_setpoint(magnet: str, value: float, kick: bool = True):
        
    if magnet in PKDK_mags:
        backleg = True
        ref_mag = PKDK_ref_mag
        ratio = PKDK_ratio
    else:
        backleg = False

    if kick:
        address = f'PETRA/MAGNET.ML/{magnet}/KICK.SP'
        if backleg:
            ref_kick = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/KICK.SP')['data']
            total_kick = ref_kick + value
            pydoocs.write(f'PETRA/MAGNET.ML/{ref_mag}/KICK2CURRENT', total_kick)
            time.sleep(0.1)
            total_current = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/KICK2CURRENT_RESULT')['data']
            ref_current = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT.SP')['data']
            current_backleg = (total_current - ref_current)/ratio
            pydoocs.write(f'PETRA/MAGNET.ML/{magnet}/CURRENT.SP', current_backleg)
            wait_on_current(magnet)
            return
    else:
        address = f'PETRA/MAGNET.ML/{magnet}/STRENGTH.SP'
        if backleg:
            ref_strength = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/STRENGTH.SP')['data']
            total_strength = ref_strength + value
            pydoocs.write(f'PETRA/MAGNET.ML/{ref_mag}/STRENGTH2CURRENT', total_strength)
            time.sleep(0.1)
            total_current = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/STRENGTH2CURRENT_RESULT')['data']
            ref_current = pydoocs.read(f'PETRA/MAGNET.ML/{ref_mag}/CURRENT.SP')['data']
            current_backleg = (total_current - ref_current)/ratio
            pydoocs.write(f'PETRA/MAGNET.ML/{magnet}/CURRENT.SP', current_backleg)
            wait_on_current(magnet)
            return

    pydoocs.write(address, value)
    wait_on_current(magnet)
    return