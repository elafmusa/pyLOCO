import os

import numpy as np
import pydoocs
import time
import h5py
import matplotlib.pyplot as plt
from getset import get_setpoint, set_setpoint

from pathlib import Path
from datetime import datetime
measurement_name = "after_loco"
measurement_label = "FullORM_100urad"

measurement_dir = Path("measurements") / measurement_name
measurement_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving data to:\n{measurement_dir.resolve()}")

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

def save_checkpoint(
    filename,
    RM,
    logs,
    bpm_names,
    cm_names,
    dkick,
    measurement_name,
    measurement_label,
    start_timestamp,
    n_orbits,
    dt,
    bidirectional,
    scaled,
    executing_time,
):

    tmp_filename = str(filename) + ".tmp"
    try:
        with h5py.File(tmp_filename, "w") as f:
            f.create_dataset("response_matrix", data=RM)
            f.attrs["dkick_rad"] = dkick
            f.attrs["dkick_urad"] = dkick * 1e6

            f.attrs["measurement_name"] = measurement_name
            f.attrs["measurement_label"] = measurement_label
            f.attrs["timestamp"] = start_timestamp

            f.attrs["n_orbits"] = n_orbits
            f.attrs["dt"] = dt

            f.attrs["bidirectional"] = bidirectional
            f.attrs["scaled"] = scaled
            f.attrs["response_matrix_unit"] = "m" if not scaled else "m/rad"

            f.attrs["execution_time_sec"] = executing_time

            logs_group = f.create_group("logs")
            for key, value in logs.items():
                logs_group.create_dataset(key, data=value)

            f.create_dataset(
            "bpm_names",
            data=np.array(bpm_names, dtype=h5py.string_dtype())
        )

            f.create_dataset(
                "hcor_names",
                data=np.array(cm_names[0], dtype=h5py.string_dtype())
            )

            f.create_dataset(
                "vcor_names",
                data=np.array(cm_names[1], dtype=h5py.string_dtype())
            ) 
        os.replace(tmp_filename, filename)    

    except Exception as e:
        print(f"Failed to save checkpoint to {tmp_filename}: {e}")
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        raise  



def response_matrix(
        orm_file,
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
    n_orbits=10
    dt=0.1

    orbit_x0, orbit_y0, std_orbit_x0, std_orbit_y0 = get_average_orbit(n_orbits=n_orbits, dt=dt)
    print(f"ref orbit x = {orbit_x0[:3]}")
    print(f"ref orbit y = {orbit_y0[:3]}")

    logs = {

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


                orbit_plus_x, orbit_plus_y, std_plus_x, std_plus_y = get_average_orbit(n_orbits=n_orbits, dt=dt)
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



                orbit_minus_x, orbit_minus_y, std_minus_x, std_minus_y = get_average_orbit(n_orbits=n_orbits, dt=dt)
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
                orbit_x0, orbit_y0, std_orbit_x0, std_orbit_y0 = get_average_orbit(n_orbits=n_orbits, dt=dt)
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

                orbit_x, orbit_y, std_x, std_y = get_average_orbit(n_orbits=n_orbits, dt=dt)
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
            try:
                save_checkpoint(
                    orm_file,
                    RM,
                    logs,
                    bpm_names,
                    cm_names,
                    dkick,
                    measurement_name,
                    measurement_label,
                    start_timestamp,
                    n_orbits,
                    dt,
                    bidirectional,
                    scaled,
                    time.time() - start_time,
                )
            except Exception as e:
                print(f"Checkpoint failed: {e}")
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

HCM_names = load_names("HCM_names_control.txt")
VCM_names = load_names("VCM_names_control.txt")
BPM_names = load_names("BPM_names.txt")

HCM_names_10 = [HCM_names[ii] for ii in np.linspace(0, len(HCM_names), 10, dtype=int, endpoint=False)]
VCM_names_10 = [VCM_names[ii] for ii in np.linspace(0, len(VCM_names), 10, dtype=int, endpoint=False)]

cm_names = [HCM_names_10, VCM_names_10]
#cm_names = [HCM_names, VCM_names]


start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
start_time = time.time()

dkick = 100e-6  # 100 urad
n_orbits=10
dt=0.1
bidirectional = True
includeDispersion = False
scaled = False
hor_dispersion_weight=1,
ver_dispersion_weight=1,
orm_file = measurement_dir / f"ORM_{measurement_label}_{start_timestamp}.h5"

RM, logs = response_matrix(
    orm_file=orm_file,
    bpm_names=BPM_names,
    cm_names=cm_names,
    dkick=dkick,
    bidirectional=bidirectional,
    includeDispersion=includeDispersion,
    hor_dispersion_weight=hor_dispersion_weight,
    ver_dispersion_weight=ver_dispersion_weight,
    scaled=scaled
)



end_time = time.time()
executing_time = end_time - start_time

print(f"\ntime: {executing_time:.3f} seconds")



try:
    save_checkpoint(
        orm_file,
        RM,
        logs,
        BPM_names,
        cm_names,
        dkick,
        measurement_name,
        measurement_label,
        start_timestamp,
        n_orbits,
        dt,
        bidirectional,
        scaled,
        time.time() - start_time,
    )
except Exception as e:
    print(f"Checkpoint failed: {e}")



print(f"\nMeasurement completed in {executing_time:.2f} s")

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

im = ax.imshow(
    RM,
    aspect="auto",
    origin="lower",
    cmap="RdBu_r"
)

ax.set_title(
    f"{measurement_name} | {measurement_label}\n"
    f"{start_timestamp}\n"
    f"Kick = {dkick*1e6:.0f} µrad"
)

ax.set_xlabel("Corrector")
ax.set_ylabel("BPM")

plt.colorbar(im, ax=ax, label="Response [m]")

plot_file = measurement_dir / f"ORM_{measurement_label}_{start_timestamp}.png"

try:
    plt.savefig(plot_file, dpi=300)
except Exception as e:
    print(f"Failed to save plot: {e}")
plt.show()

print(f"Saved ORM : {orm_file.name}")
print(f"Saved plot: {plot_file.name}")