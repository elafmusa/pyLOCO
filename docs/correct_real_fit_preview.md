# Real FIT → Correct: normal-B2 preview only

## Selected result and provenance

Source: `/Users/musa/Desktop/LOCO_test_2023/ring_pyLOCO_Noerr_16iter.mat`.
The executed workflow is recorded in `example_2026_Noerr.ipynb`: code cell 0
loads `p3_v24.mat`, disables 6D and loads `quad_ind_2024.npy`; code cell 7
fits with the individual-quadrupole option; cells 8–10 save the final lattice,
16-row parameter history and per-iteration parameter dictionary.
Cell numbers here count code cells only, from zero.

Independent audit of the saved final dictionary's 398 `quads` values against
the fitted lattice's selected `PolynomB[1]` coefficients: maximum difference
**0 m^-2**. These are individual K parameters, not family multipliers.
Other fitted blocks (skew, tilt, gains, coupling, corrector calibration, etc.)
are deliberately excluded from this milestone.

Initial lattice SHA256:
`e1dbab6e5a293adf73c8987c9bdc96f860adb6fb8becbad2d8e3b6f2a7952512`.
It is byte-identical to the official profile lattice: 3693 elements,
6.0798 GeV, identical element identities and initial quadrupole K values.
The recent 7675-element/6.0 GeV native GUI fit was rejected as the first source:
different baseline optics and unresolved identities; no ordinal transfer used.

This historical result is **not a fit of the current fixed-seed realistic_errors
machine**. Structural compatibility does not establish that its correction
will improve that machine's optics. No such improvement is claimed.

## Mapping and calculations

Each selected source ordinal indexes only its own initial/fitted result pair.
The target is looked up uniquely using **CommonName AND FamName**, then the
active SC diagnostics supply the official element ordinal, B2 control and
simulation calibration. There is no guessed name, family expansion or source
ordinal used as a target address. A missing/ambiguous mapping rejects the whole
preview. The active profile must be `petra3_realistic` with the official hash.

- Fitted normal parameters: **398**.
- Uniquely mapped B2 controls: **398 of the 417 registered controls**.
- Unmapped: **0**. Ambiguous: **0**. Family expansion: **none**.
- Full recommended physical ΔK = **initial K − fitted K**, unit **m^-2**.
- Full ΔK RMS: **0.0024697633774331943 m^-2**.
- Full max |ΔK|: **0.006845590566284643 m^-2**.
- Fixed 10% ΔK RMS: **0.00024697633774331943 m^-2**.
- Fixed 10% max |ΔK|: **0.0006845590566284643 m^-2**.

Preview conversion: `physical K = factor × control + offset`;
`control increment = 0.1 × (initial − fitted) / factor`;
`proposed control = current control + control increment`.
The factor is dimensionless **pySC simulation calibration**, not PETRA hardware
calibration. All K and control values in this B2 workflow have unit m^-2.

Example: `Q1K_SWR_8` / `Q1K_1_2` → `Q1K_1_2/B2`:

| Quantity | Value |
|---|---:|
| FIT initial K | 0.119649 |
| FIT fitted K | 0.11601180652951798 |
| Full recommended physical ΔK | 0.0036371934704820252 |
| 10% physical ΔK | 0.00036371934704820255 |
| Current simulation physical K | 0.11964905950252863 |
| Simulation calibration factor | 1.0000004973090342 |
| Required control increment | 0.0003637191661673753 |
| Proposed control | 0.12001271916616738 |
| Expected physical K | 0.12001277884957684 |

The last two are **proposals, not applied values**.

## Click sequence

1. Open Correct; choose **pySC Server**, then **B2 simulation**.
2. Select **PETRA III / realistic_errors** and the running server's diagnostic
   port; click **Connect / discover B2 controls**.
3. Click **Load real FIT bundle… (preview only, 10%)** and select
   `Examples/Correct/petra_noerr16_fit_preview.json`.
4. Click **Preview — zero writes**. Inspect the 398-row table and mapping summary.

The JSON is a provenance descriptor, **not a synthetic correction request**:
the loader reads actual initial/fitted `.mat` files and a numeric ordinal `.npy`
directly. All input hashes are checked; no pickled fit dictionaries are loaded
by the GUI. Paths can be relocated while preserving hashes. This first loader
is limited to explicitly audited individual-K bundles, not arbitrary family
results or cross-structure transfer. The existing toolbar Results importer is
unchanged; use the new B2-workspace button for this verified simulation preview.

Apply is disabled and programmatically blocked for this FIT path. The existing
single/3–5 synthetic transaction limits and safety machinery are unchanged.
No journal is generated because there is no write transaction.

## Validation (2026-09-03)

GET-only native GUI validator against isolated PETRA realistic ports 13331/13332:
**398 control GETs, zero SETs**, identical full diagnostics before/after,
398 mapped preview rows, Apply disabled. Header identifies DEMO and
PETRA III / realistic_errors, seed 20260907.

Run from repository root with an already running isolated server:

```bash
PYTHONPATH=. .venv/bin/python Examples/Correct/validate_fit_preview_gui.py --diagnostics-port 13332
```

Screenshot is written under `/private/tmp`, not committed. Regression checks:
65 Correction tests and 16 suite integration/about tests passed. FIT and Measure
mathematics, simulation errors, server protocol, skew and LIVE support unchanged.
