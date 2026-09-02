# pySC machine profiles

The generic pySC Server backend has machine-specific profiles outside the
Measure GUI and acquisition services.

Start either simulation from the repository root:

```bash
source .venv/bin/activate
python Examples/Demo/start_pysc_server.py --profile ebs
python Examples/Demo/start_pysc_server.py --profile petra3
python Examples/Demo/start_pysc_server.py --profile petra3_realistic
```

Only one process can use the default `127.0.0.1:13131` endpoint at a time.
The launcher generates the selected profile's device catalog directly from
the served `SimulatedCommissioning` object and installs the local pySC 1.5.2
compatibility handlers. Site-packages are not modified.

## Profiles

- `ebs/validated_demo`: references the existing validated saved EBS state and
  lattice. Its manifest contains runtime-only demo-noise defaults and can be
  extended later with named error scenarios without adding errors to the GUI.
- `petra_iii/official`: immutable colleague-supplied nominal baseline. The
  original configuration, lattice, mappings and name files are preserved in
  the directory structure expected by `petra3_conf.yaml`.
- `petra_iii/realistic_errors`: reproducible, uncorrected experimental machine
  built from byte-identical official lattice/mapping inputs with seed 20260907
  and the complete requested/truncated-Gaussian error budget in `profile.yaml`.

Alternative fitted/error lattices should be added as new scenario manifests;
they must never replace the official baseline. Before activation they must
pass element-order, `CommonName`, BPM/HCM/VCM mapping, RF, energy and dimensional
compatibility checks.
