# Publication and local-artifact manifest

## Suitable for repository publication

- `README.md`, `CHECKLIST.md`, `PACKAGE_MANIFEST.md`
- `build_modern_ebs_package.py`
- `results/README.md`

Before publishing `VALIDATION.json`, remove its absolute local project path.
Its hashes and scientific counts are suitable for publication.

## Local-only; do not commit by default

- `configs/EBS_pyaml_MDT_smoke.pyloco.json`
- `configs/EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json`
- `configs/betamodel_conf_pyaml_test.local.yaml`
- `lattice/betamodel.mat`
- `measurements/*`
- `mappings/EBS_measurement_devices.validated.local.json`

These contain binary measurements, control-system identities, acquisition
provenance, machine configuration, or personal paths. Share only through the
approved ESRF data-transfer route.

## Review before publication

- `configs/EBS_modern_selection_provenance.json` is checksum-bound scientific
  provenance with AT names and ordinals; it has no personal absolute path.
- `mappings/EBS_three_quadrupoles.correct.validated.json` contains real device
  identities, although its lattice path is package-relative.
- Older `output/`, `run_fit_smoke.py`, and `ebs_small_orm_fit.yaml` are retained
  historical smoke artifacts and are not part of the modern bundle.
