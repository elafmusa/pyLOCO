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

    logs = {
        "bpm_names": bpm_names,
        "hcor_names": cm_names[0],
        "vcor_names": cm_names[1],

        "orbit0_x": orbit_x0,
        "orbit0_y": orbit_y0,
        "std_orbit0_x": std_orbit_x0,
        "std_orbit0_y": std_orbit_y0,

        "kick0": np.full(n_cm, np.nan),
        "kick_p": np.full(n_cm, np.nan),
        "kick_m": np.full(n_cm, np.nan),
        "kick_f": np.full(n_cm, np.nan),
        "dkick_used": np.full(n_cm, np.nan),

  
        "KICK_SP_0": np.full(n_cm, np.nan),
        "KICK_RBV_0": np.full(n_cm, np.nan),
        "CURRENT_SP_0": np.full(n_cm, np.nan),
        "CURRENT_RBV_0": np.full(n_cm, np.nan),

        "KICK_SP_p": np.full(n_cm, np.nan),
        "KICK_RBV_p": np.full(n_cm, np.nan),
        "CURRENT_SP_p": np.full(n_cm, np.nan),
        "CURRENT_RBV_p": np.full(n_cm, np.nan),

        "KICK_SP_m": np.full(n_cm, np.nan),
        "KICK_RBV_m": np.full(n_cm, np.nan),
        "CURRENT_SP_m": np.full(n_cm, np.nan),
        "CURRENT_RBV_m": np.full(n_cm, np.nan),

        "KICK_SP_f": np.full(n_cm, np.nan),
        "KICK_RBV_f": np.full(n_cm, np.nan),
        "CURRENT_SP_f": np.full(n_cm, np.nan),
        "CURRENT_RBV_f": np.full(n_cm, np.nan),

        "orbit_plus_x": np.full((n_bpm, n_cm), np.nan),
        "orbit_plus_y": np.full((n_bpm, n_cm), np.nan),
        "std_orbit_plus_x": np.full((n_bpm, n_cm), np.nan),
        "std_orbit_plus_y": np.full((n_bpm, n_cm), np.nan),

        "orbit_minus_x": np.full((n_bpm, n_cm), np.nan),
        "orbit_minus_y": np.full((n_bpm, n_cm), np.nan),
        "std_orbit_minus_x": np.full((n_bpm, n_cm), np.nan),
        "std_orbit_minus_y": np.full((n_bpm, n_cm), np.nan),
    }

    def _get_dkick_for(n_dim, j):
        if isinstance(dkick, (list, tuple, np.ndarray)):

            try:
                return float(dkick[n_dim][j])
            except Exception:
                return float(dkick[j])
        else:
            return float(dkick)

    cnt = 0
    for n_dim in [0,1]: 
        for j, cm_name in enumerate(cm_names[n_dim]):
            this_dkick = _get_dkick_for(n_dim, j)
            kick0 = get_setpoint(cm_name, kick=True)
            print(f"this_dkick = {this_dkick}")
            print(f"[{cnt}] {cm_name}: kick_s = {kick0}")

            logs["kick0"][cnt] = kick0
            logs["dkick_used"][cnt] = this_dkick

            r0 = read_all_non_strength(cm_name)
            logs["KICK_SP_0"][cnt] = r0["KICK_SP"]
            logs["KICK_RBV_0"][cnt] = r0["KICK_RBV"]
            logs["CURRENT_SP_0"][cnt] = r0["CURRENT_SP"]
            logs["CURRENT_RBV_0"][cnt] = r0["CURRENT_RBV"]

            if bidirectional:
        
                kick_value_p = kick0 + this_dkick / 2.0
                set_setpoint(magnet=cm_name, value=kick_value_p, kick=True)
                time.sleep(1)
                kick_read_p = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_p = {kick_read_p}")
                logs["kick_p"][cnt] = kick_read_p
                rp = read_all_non_strength(cm_name)
                logs["KICK_SP_p"][cnt] = rp["KICK_SP"]
                logs["KICK_RBV_p"][cnt] = rp["KICK_RBV"]
                logs["CURRENT_SP_p"][cnt] = rp["CURRENT_SP"]
                logs["CURRENT_RBV_p"][cnt] = rp["CURRENT_RBV"]


                orbit_plus_x, orbit_plus_y, std_plus_x, std_plus_y = get_average_orbit(n_orbits=10, dt=0.1)
                logs["orbit_plus_x"][:, cnt] = orbit_plus_x
                logs["orbit_plus_y"][:, cnt] = orbit_plus_y
                logs["std_orbit_plus_x"][:, cnt] = std_plus_x
                logs["std_orbit_plus_y"][:, cnt] = std_plus_y

      
                kick_value_n = kick0 - this_dkick / 2.0
                set_setpoint(magnet=cm_name, value=kick_value_n, kick=True)
                time.sleep(1)
                kick_read_m = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_m = {kick_read_m}")
                logs["kick_m"][cnt] = kick_read_m

                rmv = read_all_non_strength(cm_name)
                logs["KICK_SP_m"][cnt] = rmv["KICK_SP"]
                logs["KICK_RBV_m"][cnt] = rmv["KICK_RBV"]
                logs["CURRENT_SP_m"][cnt] = rmv["CURRENT_SP"]
                logs["CURRENT_RBV_m"][cnt] = rmv["CURRENT_RBV"]



                orbit_minus_x, orbit_minus_y, std_minus_x, std_minus_y = get_average_orbit(n_orbits=10, dt=0.1)
                logs["orbit_minus_x"][:, cnt] = orbit_minus_x
                logs["orbit_minus_y"][:, cnt] = orbit_minus_y
                logs["std_orbit_minus_x"][:, cnt] = std_minus_x
                logs["std_orbit_minus_y"][:, cnt] = std_minus_y

                set_setpoint(magnet=cm_name, value=kick0, kick=True)
                kick_f = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_f = {kick_f}")
                logs["kick_f"][cnt] = kick_f
                rf = read_all_non_strength(cm_name)
                logs["KICK_SP_f"][cnt] = rf["KICK_SP"]
                logs["KICK_RBV_f"][cnt] = rf["KICK_RBV"]
                logs["CURRENT_SP_f"][cnt] = rf["CURRENT_SP"]
                logs["CURRENT_RBV_f"][cnt] = rf["CURRENT_RBV"]


                dx = (orbit_plus_x - orbit_minus_x)  #- orbit_x0
                dy = (orbit_plus_y - orbit_minus_y) #- orbit_y0

            else:
                orbit_x0, orbit_y0, std_orbit_x0, std_orbit_y0 = get_average_orbit(n_orbits=10, dt=0.1)
                kick_value = kick0 + this_dkick
                set_setpoint(magnet=cm_name, value=kick_value, kick=True)
                kick_read_p = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick (single) = {kick_read_p}")
                logs["kick_p"][cnt] = kick_read_p
                rp = read_all_non_strength(cm_name)
                logs["KICK_SP_p"][cnt] = rp["KICK_SP"]
                logs["KICK_RBV_p"][cnt] = rp["KICK_RBV"]
                logs["CURRENT_SP_p"][cnt] = rp["CURRENT_SP"]
                logs["CURRENT_RBV_p"][cnt] = rp["CURRENT_RBV"]

                orbit_x, orbit_y, std_x, std_y = get_average_orbit(n_orbits=10, dt=0.1)
                logs["orbit_plus_x"][:, cnt] = orbit_x
                logs["orbit_plus_y"][:, cnt] = orbit_y
                logs["std_orbit_plus_x"][:, cnt] = std_x
                logs["std_orbit_plus_y"][:, cnt] = std_y

                # Reset
                set_setpoint(magnet=cm_name, value=kick0, kick=True)
                kick_f = get_setpoint(cm_name, kick=True)
                print(f"[{cnt}] kick_f = {kick_f}")
                logs["kick_f"][cnt] = kick_f
                rf = read_all_non_strength(cm_name)
                logs["KICK_SP_f"][cnt] = rf["KICK_SP"]
                logs["KICK_RBV_f"][cnt] = rf["KICK_RBV"]
                logs["CURRENT_SP_f"][cnt] = rf["CURRENT_SP"]
                logs["CURRENT_RBV_f"][cnt] = rf["CURRENT_RBV"]

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

HCM_names_10 = [HCM_names[ii] for ii in np.linspace(0, len(HCM_names), 50, dtype=int, endpoint=False)]
VCM_names_10 = [VCM_names[ii] for ii in np.linspace(0, len(VCM_names), 50, dtype=int, endpoint=False)]


cm_names = [HCM_names_10, VCM_names_10]



start_time = time.time()

RM, logs = response_matrix(
    BPM_names,
    cm_names,
    dkick=100e-6,
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
#with h5py.File("orm_8july26_50cor_quads_error.h5", "w") as f:
    # Main dataset
#    f.create_dataset("response_matrix", data=RM)
#    f.create_dataset("dkick", data=100e-6)
#    f.attrs["n_orbits"] = 10
#    f.attrs["dt"] = 0.1
#    f.attrs["bidirectional"] = True
#    f.attrs["scaled"] = True
#    f.attrs["execution_time_sec"] = executing_time


with h5py.File("orm_8july26_50cor_quads_error.h5", "w") as f:
    f.create_dataset("response_matrix", data=RM)

    log_group = f.create_group("logs")
    for key, value in logs.items():
        if isinstance(value, list):
            value = np.array(value, dtype="S")
        elif isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
            value = value.astype("S")

        log_group.create_dataset(key, data=value)

    f.attrs["n_orbits"] = 10
    f.attrs["dt"] = 0.1
    f.attrs["bidirectional"] = True
    f.attrs["scaled"] = True
    f.attrs["execution_time_sec"] = executing_time    
