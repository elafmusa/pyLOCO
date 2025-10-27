import numpy as np
import pydoocs
import time
from getset import get_setpoint, set_setpoint


#BPM_ADDRESS_X = 'PETRA/REFORBIT/*/SA_X_BBAGO'
#BPM_ADDRESS_Y = 'PETRA/REFORBIT/*/SA_Y_BBAGO'

NM_TO_M = 1e-9
BPM_ADDRESS_X = 'PETRA/REFORBIT/*/SA_X_RAW'
BPM_ADDRESS_Y = 'PETRA/REFORBIT/*/SA_Y_RAW'


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

    all_orbit_x[:, 0] = orbit_x
    all_orbit_y[:, 0] = orbit_y
    for ii in range(1, n_orbits):
        time.sleep(dt)
        all_orbit_x[:, ii], all_orbit_y[:, ii] = get_pydoocs_orbit()

    print(f"Reading orbit_x[:3] = {orbit_x[:3]}")
    print(f"Reading orbit_y[:3] = {orbit_y[:3]}")

    mean_orbit_x = np.mean(all_orbit_x, axis=1)
    mean_orbit_y = np.mean(all_orbit_y, axis=1)
    std_orbit_x = np.std(all_orbit_x, axis=1)
    std_orbit_y = np.std(all_orbit_y, axis=1)
    return mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y





def read_all_non_strength(magnet: str):
    """Return dict of all magnet channels except STRENGTH."""
    base = f'PETRA/MAGNET.ML/{magnet}'
    def rd(suffix):
        return float(pydoocs.read(f'{base}/{suffix}')['data'])
    return {
        "KICK_SP": rd('KICK.SP'),
        "KICK_RBV": rd('KICK.RBV'),
        "CURRENT_SP": rd('CURRENT.SP'),
        "CURRENT_RBV": rd('CURRENT.RBV'),
    }


def response_matrix(
    bpm_names,
    cm_names,
    dkick=100e-6,
    bidirectional=True,
    includeDispersion=False,
    hor_dispersion_weight=1,
    ver_dispersion_weight=1,
    scaled=False
):
    n_bpm = len(bpm_names)
    n_hcor, n_vcor = len(cm_names[0]), len(cm_names[1])
    n_cm = n_hcor + n_vcor
    RM = np.full((2 * n_bpm, n_cm), np.nan)


    orbit_x0, orbit_y0, std_orbit_x0, std_orbit_y0 = get_average_orbit(n_orbits=10, dt=0.1)
    print(f"ref orbit x = {orbit_x0[:3]}")
    print(f"ref orbit y = {orbit_y0[:3]}")

   

    def _get_dkick_for(n_dim, j):
        if isinstance(dkick, (list, tuple, np.ndarray)):

            try:
                return float(dkick[n_dim][j])
            except Exception:
                return float(dkick[j])
        else:
            return float(dkick)

    cnt = 0
    for n_dim in [0]: 
        for j, cm_name in enumerate(cm_names[n_dim]):
            this_dkick = _get_dkick_for(n_dim, j)
            kick0 = get_setpoint(cm_name, kick=True)
            print(f"this_dkick = {this_dkick}")
            print(f"[{cnt}] {cm_name}: kick_s = {kick0}")

           
            r0 = read_all_non_strength(cm_name)
          

            if bidirectional:
        
                kick_value_p = kick0 + this_dkick / 2.0
                set_setpoint(magnet=cm_name, value=kick_value_p, kick=True)
                time.sleep(1)
                kick_read_p = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_p = {kick_read_p}")
                rp = read_all_non_strength(cm_name)
             
                orbit_plus_x, orbit_plus_y, std_plus_x, std_plus_y = get_average_orbit(n_orbits=10, dt=0.1)
               
      
                kick_value_n = kick0 - this_dkick / 2.0
                set_setpoint(magnet=cm_name, value=kick_value_n, kick=True)
                time.sleep(1)
                kick_read_m = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_m = {kick_read_m}")

                rmv = read_all_non_strength(cm_name)
               


                orbit_minus_x, orbit_minus_y, std_minus_x, std_minus_y = get_average_orbit(n_orbits=10, dt=0.1)
                
                set_setpoint(magnet=cm_name, value=kick0, kick=True)
                kick_f = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_f = {kick_f}")
                rf = read_all_non_strength(cm_name)
                

                dx = (orbit_plus_x - orbit_minus_x)  #- orbit_x0
                dy = (orbit_plus_y - orbit_minus_y) #- orbit_y0

            else:
                orbit_x0, orbit_y0, std_orbit_x0, std_orbit_y0 = get_average_orbit(n_orbits=10, dt=0.1)
                kick_value = kick0 + this_dkick
                set_setpoint(magnet=cm_name, value=kick_value, kick=True)
                kick_read_p = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick (single) = {kick_read_p}")
                rp = read_all_non_strength(cm_name)
              
                orbit_x, orbit_y, std_x, std_y = get_average_orbit(n_orbits=10, dt=0.1)
                
                # Reset
                set_setpoint(magnet=cm_name, value=kick0, kick=True)
                kick_f = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_f = {kick_f}")
                rf = read_all_non_strength(cm_name)
                

                dx = orbit_x - orbit_x0
                dy = orbit_y - orbit_y0

            RM[:, cnt] = np.concatenate((dx, dy))
            cnt += 1

    if scaled:
        if isinstance(dkick, (list, tuple, np.ndarray)):
            cor_kicks = np.concatenate((np.array(dkick[0]).ravel(), np.array(dkick[1]).ravel()))
        else:
            cor_kicks = np.full(n_cm, float(dkick))
        RM = RM / cor_kicks[np.newaxis, :]

    return RM, logs



def load_names(filename):
    with open(filename, 'r') as file:
        names = [line.strip() for line in file.readlines()]
    return names

HCM_names = load_names("HCM_names2.txt")
VCM_names = load_names("VCM_names2.txt")
BPM_names = load_names("BPM_names.txt")
cm_names = [HCM_names, VCM_names]
cor_kicks = 100e-6


start_time = time.time()

RM, logs = response_matrix(
    BPM_names,
    cm_names,
    dkick=cor_kicks,
    bidirectional=True,
    includeDispersion=False,
    hor_dispersion_weight=1,
    ver_dispersion_weight=1,
    scaled=True
)

end_time = time.time()
executing_time = end_time - start_time
print(f"\ntime: {executing_time:.3f} seconds")

import h5py
with h5py.File("measured_orm_loco.h5", "w") as f:
    # Main dataset
    f.create_dataset("response_matrix", data=RM)
    f.create_dataset("dkick", data=cor_kicks)
    f.attrs["n_orbits"] = 10
    f.attrs["dt"] = 0.1
    f.attrs["bidirectional"] = True
    f.attrs["scaled"] = True
    f.attrs["execution_time_sec"] = executing_time
