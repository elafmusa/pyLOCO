import time
import h5py
import numpy as np
import pydoocs
import json


loco_name = "constraint_tune"  

def load_names(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def get_setpoint(magnet: str, kick: bool = True) -> float:
    addr = f"PETRA/MAGNET.ML/{magnet}/{'KICK.SP' if kick else 'STRENGTH.SP'}"
    data = pydoocs.read(addr)["data"]
    return float(np.asarray(data))  # robust coercion

def set_setpoint(magnet: str, value: float, kick: bool = True):
    addr = f"PETRA/MAGNET.ML/{magnet}/{'KICK.SP' if kick else 'STRENGTH.SP'}"
    pydoocs.write(addr, float(value))
    #time.sleep(3)


print("Apply correction")
start_time = time.time()

quads_names = load_names("quads_names_control.txt")
#quads_names = load_names("skew_names.txt")

from pathlib import Path

correction_file = "correction_constraint_tune.json" #correction_c3
with open(correction_file, "r") as f:
    delta_q_s = json.load(f)

print(f"Loaded corrections from {correction_file}")

# Normal quads
normal_deltas = delta_q_s["normal_quads"]["normal_quads_family"]["delta"]
normal_lengths = delta_q_s["normal_quads"]["normal_quads_family"]["length"]

# Skew quads
skew_deltas = delta_q_s["skew_quads"]["delta"]
skew_lengths = delta_q_s["skew_quads"]["length"]


delta_correction = normal_deltas  ##### for skew also #skew_deltas
#delta_correction = np.atleast_1d(delta_correction).astype(float)

if len(normal_deltas) != len(quads_names):
    raise ValueError(
    f"JSON contains {len(normal_deltas)} corrections "
    f"but quads_names contains {len(quads_names)} magnets."
)

original_setpoints = []
k0_correction_ = []
perturbed_setpoints = []


for i, quad_name in enumerate(quads_names):
    
    #if i <=78:
    #    continue
    k0 = get_setpoint(quad_name, kick=False)

    correction = float(delta_correction[i])

    new_value = k0 + 0.10*correction

    set_setpoint(quad_name, new_value, kick=False)

    k_corr = get_setpoint(quad_name, kick=False)

    original_setpoints.append(k0)
    k0_correction_.append(correction)
    perturbed_setpoints.append(k_corr)

    print(f"{i}: {quad_name}")
    print(f"  Original setpoint     = {k0:.6f}")
    print(f"  Applied correction    = {correction:+.6e}")
    print(f"  New setpoint          = {new_value:.6f}")
    print(f"  Actual setpoint read  = {k_corr:.6f} "
          f"(delta={k_corr-new_value:+.3e})\n")

exec_time = time.time() - start_time
print(f"\nTime: {exec_time:.3f} s for full script")

#log_file = Path("./correction") / loco_name / "quads_correction.h5"

#with h5py.File(log_file, "w") as f:
#    f.attrs["execution_time_full"] = exec_time
#    f.attrs["Kick"] = False
#    f.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

#    f.create_dataset("quad_names", data=[n.encode() for n in quads_names])
#    f.create_dataset("original_setpoints", data=np.asarray(original_setpoints))
#    f.create_dataset("perturbed_setpoints", data=np.asarray(perturbed_setpoints))
#    f.create_dataset("k0_correction", data=np.asarray(k0_correction_))
