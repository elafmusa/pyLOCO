import numpy as np
import pydoocs
import time
import h5py
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
measurement_name = "after_loco"

measurement_dir = Path("measurements") / measurement_name
measurement_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving data to:\n{measurement_dir.resolve()}")
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

    #print(f"Reading {0}: orbit_x[:5] = {all_orbit_x[:5, 0]}")
    #print(f"Reading {0}: orbit_y[:5] = {all_orbit_y[:5, 0]}")
    
    #print(f"Reading {1}: orbit_x[:5] = {all_orbit_x[:5, 1]}")
    #print(f"Reading {1}: orbit_y[:5] = {all_orbit_y[:5, 1]}")


    mean_orbit_x = np.mean(all_orbit_x, axis=1)
    mean_orbit_y = np.mean(all_orbit_y, axis=1)
    std_orbit_x = np.std(all_orbit_x, axis=1)
    std_orbit_y = np.std(all_orbit_y, axis=1)
    return mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y



start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

start_time = time.time()
n_orbits = 1620
dt = 0.1
mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y = get_average_orbit(n_orbits=n_orbits, dt=dt)

#mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y = get_average_orbit(n_orbits=1800, dt=0.1) # measuring for 3 min 180/0.1 = 1800 readings

end_time = time.time()
executing_time = end_time - start_time

print(f"\ntime: {executing_time:.3f} seconds")



bpm_file = measurement_dir / f"BPM_noise_{start_timestamp}.h5"

with h5py.File(bpm_file, "w") as f:
    f.create_dataset("mean_orbit_x", data=mean_orbit_x)
    f.create_dataset("mean_orbit_y", data=mean_orbit_y)
    f.create_dataset("std_orbit_x", data=std_orbit_x)
    f.create_dataset("std_orbit_y", data=std_orbit_y)
    f.attrs["measurement_name"] = measurement_name
    f.attrs["timestamp"] = start_timestamp
    f.attrs["n_orbits"] = n_orbits
    f.attrs["dt"] = dt
    f.attrs["execution_time_sec"] = executing_time
    f.attrs["bpm_address_x"] = BPM_ADDRESS_X
    f.attrs["bpm_address_y"] = BPM_ADDRESS_Y
    f.attrs["orbit_unit"] = "m"    

print(f"Saved: {bpm_file.resolve()}")


fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Horizontal BPM noise
axs[0].plot(std_orbit_x * 1e6, ".-", ms=4)
axs[0].set_ylabel("Noise X [µm]")
axs[0].set_title(
    f"{measurement_name} | {start_timestamp}\n"
    f"{n_orbits} orbits, dt={dt:.1f} s"
)
axs[0].grid(True)

# Vertical BPM noise
axs[1].plot(std_orbit_y * 1e6, ".-", ms=4)
axs[1].set_ylabel("Noise Y [µm]")
axs[1].set_xlabel("BPM Index")
axs[1].grid(True)

plt.tight_layout()
plot_file = measurement_dir / f"BPM_noise_{start_timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved plot: {plot_file.resolve()}")