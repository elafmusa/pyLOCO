import numpy as np
import pydoocs
import time
import h5py
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
measurement_name = "after_loco_correction"

#measurement_label = "RF_reference"
#measurement_label = "RF_plus1500Hz"
measurement_label = "RF_minus1500Hz"

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

 


    mean_orbit_x = np.mean(all_orbit_x, axis=1)
    mean_orbit_y = np.mean(all_orbit_y, axis=1)
    
    

    print(f"Reading {0}: orbit_x[:5] = {mean_orbit_x[:5]}")
    print(f"Reading {0}: orbit_y[:5] = {mean_orbit_y[:5]}")
    
    
    std_orbit_x = np.std(all_orbit_x, axis=1)
    std_orbit_y = np.std(all_orbit_y, axis=1)
    return mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y


start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

start_time = time.time()

n_orbits =50
dt = 0.1
mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y = get_average_orbit(n_orbits=n_orbits, dt=dt)

end_time = time.time()
executing_time = end_time - start_time

print(f"\ntime: {executing_time:.3f} seconds")




# ============================================================
# Save current dispersion measurement
# ============================================================

dispersion_file = (
    measurement_dir
    / f"Dispersion_{measurement_label}_{start_timestamp}.h5"
)

with h5py.File(dispersion_file, "w") as f:

    # Current measurement
    f.create_dataset("mean_orbit_x", data=mean_orbit_x)
    f.create_dataset("mean_orbit_y", data=mean_orbit_y)

    f.create_dataset("std_orbit_x", data=std_orbit_x)
    f.create_dataset("std_orbit_y", data=std_orbit_y)

    # Metadata
    f.attrs["measurement_name"] = measurement_name
    f.attrs["measurement_label"] = measurement_label
    f.attrs["timestamp"] = start_timestamp
    f.attrs["n_orbits"] = n_orbits
    f.attrs["dt"] = dt
    f.attrs["execution_time_sec"] = executing_time
    f.attrs["bpm_address_x"] = BPM_ADDRESS_X
    f.attrs["bpm_address_y"] = BPM_ADDRESS_Y
    f.attrs["orbit_unit"] = "m"

print(f"Saved: {dispersion_file.resolve()}")


# ============================================================
# If this is the -1500 Hz measurement:
# automatically find the latest +1500 Hz measurement
# and save the combined result in the SAME file
# ============================================================

if measurement_label == "RF_minus1500Hz":

    plus_files = sorted(
        measurement_dir.glob("Dispersion_RF_plus1500Hz_*.h5"),
        key=lambda p: p.stat().st_mtime
    )

    if len(plus_files) == 0:
        print("WARNING: No RF_plus1500Hz file found.")
        print("Only the current RF_minus1500Hz measurement was saved.")

    else:

        dispersion_file_plus = plus_files[-1]

        print(
            "\nUsing +1500 Hz measurement:\n"
            f"{dispersion_file_plus}"
        )

        # ----------------------------------------------------
        # Read +1500 Hz measurement
        # ----------------------------------------------------

        with h5py.File(dispersion_file_plus, "r") as f:

            measured_eta_x_plus = f["mean_orbit_x"][:]
            measured_eta_y_plus = f["mean_orbit_y"][:]

            std_eta_x_plus = f["std_orbit_x"][:]
            std_eta_y_plus = f["std_orbit_y"][:]

        # ----------------------------------------------------
        # Current measurement is the -1500 Hz measurement
        # ----------------------------------------------------

        measured_eta_x_minus = mean_orbit_x
        measured_eta_y_minus = mean_orbit_y

        std_eta_x_minus = std_orbit_x
        std_eta_y_minus = std_orbit_y

        # ----------------------------------------------------
        # Difference:
        #
        #     orbit(-1500 Hz) - orbit(+1500 Hz)
        #
        # This corresponds to what you currently calculate as:
        #
        # measured_eta_x_ = measured_eta_x_1 - measured_eta_x_2
        # measured_eta_y_ = measured_eta_y_1 - measured_eta_y_2
        # ----------------------------------------------------

        measured_eta_x = (
            measured_eta_x_minus - measured_eta_x_plus
        )

        measured_eta_y = (
            measured_eta_y_minus - measured_eta_y_plus
        )

        # ----------------------------------------------------
        # Save everything into the current -1500 Hz HDF5 file
        # ----------------------------------------------------

        with h5py.File(dispersion_file, "a") as f:

            # Explicitly save both measurements
            f.create_dataset(
                "mean_orbit_x_minus1500Hz",
                data=measured_eta_x_minus
            )

            f.create_dataset(
                "mean_orbit_y_minus1500Hz",
                data=measured_eta_y_minus
            )

            f.create_dataset(
                "mean_orbit_x_plus1500Hz",
                data=measured_eta_x_plus
            )

            f.create_dataset(
                "mean_orbit_y_plus1500Hz",
                data=measured_eta_y_plus
            )

            # Save the difference directly
            f.create_dataset(
                "measured_eta_x",
                data=measured_eta_x
            )

            f.create_dataset(
                "measured_eta_y",
                data=measured_eta_y
            )

            # Useful metadata
            f.attrs["plus1500Hz_source_file"] = (
                dispersion_file_plus.name
            )

            f.attrs["dispersion_difference"] = (
                "orbit_minus1500Hz - orbit_plus1500Hz"
            )

        print("\nCombined dispersion data saved.")
        print(f"File: {dispersion_file}")
        print(
            "Saved datasets:"
            "\n  measured_eta_x"
            "\n  measured_eta_y"
            "\n  mean_orbit_x_minus1500Hz"
            "\n  mean_orbit_y_minus1500Hz"
            "\n  mean_orbit_x_plus1500Hz"
            "\n  mean_orbit_y_plus1500Hz"
        )

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs[0].plot(mean_orbit_x * 1e3, ".-", ms=4, label="Mean orbit")
axs[0].set_ylabel("Dispersion Orbit X [mm]")
axs[0].set_title(
    f"{measurement_name} | {measurement_label} | {start_timestamp}\n"
    f"{n_orbits} orbits, dt={dt:.1f} s"
)
axs[0].grid(True)
axs[0].legend()

axs[1].plot(mean_orbit_y * 1e3, ".-", ms=4, label="Mean orbit")
axs[1].set_ylabel("Dispersion Orbit Y [mm]")
axs[1].set_xlabel("BPM Index")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()

plot_file = measurement_dir / f"Dispersion_{measurement_label}_{start_timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot: {plot_file.resolve()}")
print(f"\nMeasurement completed in {executing_time:.2f} s")
print(f"Saved HDF5: {dispersion_file.name}")
print(f"Saved plot : {plot_file.name}")