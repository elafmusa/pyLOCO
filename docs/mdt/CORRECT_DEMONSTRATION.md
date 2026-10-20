# Correct read-only demonstration from FIT results

## Required inputs

- a complete pyLOCO FIT results directory;
- the same 4434-element reference lattice retained in the result provenance;
- the approved simulator pyAML YAML;
- an explicit Correct mapping bound to that lattice, YAML checksum and device
  inventory.

For the restricted example the fitted identities are:

| Lattice element | Zero-based AT ordinal | pyAML component | Control unit |
|---|---:|---|---|
| `QD3J` | 27 | `B2L` | `m^-1` |
| `QF4J` | 36 | `B2L` | `m^-1` |
| `QD5B` | 51 | `B2L` | `m^-1` |

FIT stores normal-quadrupole `K` and `ΔK` in `m^-2`. Correct uses the magnetic
length saved from the lattice and calculates `Δ(KL)=ΔK·L` in `m^-1`. It does
not reinterpret electrical current as integrated strength.

## Procedure

1. Launch:

   ```bash
   pyloco-correct --results <FIT_RESULTS_DIRECTORY> --iteration 2
   ```

   Alternatively click **Load FIT results…** and choose the directory.

2. On **Correction Source**, confirm the selected stage/iteration and the three
   fitted magnet name+ordinal identities.
3. Select the pyAML backend. Load the approved YAML and connect to the intended
   simulator with writes disabled for this demonstration.
4. On **Machine / Mapping**, click **Load completed mapping…** and select the
   exact mapping bundle. Correct revalidates:
   - lattice SHA-256 and element count;
   - YAML SHA-256, mode and configured arrays;
   - inventory fingerprint and count;
   - unique source name+ordinal and unique target identity;
   - normal-quadrupole family, `B2L` component and `m^-1` unit.
5. Require `3 of 3 magnets mapped`. Missing, changed, duplicated or wrong-family
   mappings block preview.
6. Click **Read current integrated strengths**. This is a read-only public pyAML
   `get()` operation. Review any calibration/current-limit warning; a numeric
   value alone does not certify real-machine calibration.
7. Click **Preview machine changes**. Confirm current KL, target KL, fitted
   `ΔK`, magnetic length and final `Δ(KL)` for every magnet.
8. Do not click **Apply…**. Preserve the results directory and exported preview
   evidence.

## Supported result inputs

- A current pyLOCO Results directory is the authoritative path and can be
  opened directly; no new FIT is required.
- A saved pyLOCO correction plan can be reopened.
- The GUI also accepts explicit legacy JSON/YAML correction records only when
  they include the required initial, fitted and recommended-delta fields.
- There is no generic import that turns an arbitrary list of correction values
  into an authoritative FIT result. Do not invent ordinals, signs or units.

## Fail closed

Do not preview if the result iteration is unclear, the reference lattice is
unavailable, the mapping bundle is incompatible, a magnetic length is missing,
or an integrated-strength read cannot be obtained. Do not apply any correction
until ESRF confirms B2L/A2L units, sign, calibration, limits and rollback on the
actual target.
