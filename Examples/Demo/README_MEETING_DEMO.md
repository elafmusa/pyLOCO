# pyLOCO Suite meeting demo

Run from the repository root with the pyLOCO virtual environment active.

## 1 — Live one-iteration PETRA III fit

```bash
./Examples/Demo/01_run_live_fit.sh
```

Opens a validated PETRA III coupling-fit project with one outer iteration. Review Machine Components, Measurements and Fit, click **Validate**, then **Run LOCO**. Existing timing indicates approximately **5 minutes** on this Mac (the matching validated one-iteration run took 287.9 s).

## 2 — Completed eight-iteration fit

```bash
./Examples/Demo/02_open_completed_fit.sh
```

Opens the Results workspace immediately with an **actual current pyLOCO numerical run**. It reaches **χ² = 5.807** after eight iterations and includes all four coupling blocks, 235 retained BPMs, 219 horizontal correctors, 194 vertical correctors, 398 individual quadrupoles and 16 skew quadrupoles. Every iteration contains native ORM, βx/βy, ηx/ηy, residual, fitted-lattice and parameter artifacts. Runtime was 1,672 seconds. Large Jacobian files are intentionally omitted.

## 3 — Correction review

```bash
./Examples/Demo/03_open_correction_review.sh
```

Opens **OFFLINE • DRY RUN** pyLOCO Correct with the same run's 398 individual PETRA III quadrupole corrections. No machine write is possible. Names, lattice ordinals, initial/fitted K, raw fitted ΔK, recommended correction, scaling, filtering and fraction comparison are immediately available. PETRA control-system mapping/current calibration are intentionally not loaded.

Meeting sequence: **measured machine → one-iteration live fit → eight-iteration convergence/results → final correction review**.

## Monday pySC Server setup

Install the exact tested control-system package once:

```bash
cd /Users/musa/Desktop/pyLOCO
.venv/bin/pip install -e '.[gui,pysc-demo]'
```

Terminal 1 — server:

```bash
cd /Users/musa/Desktop/pyLOCO
MPLCONFIGDIR=/tmp/pyloco-mpl-pysc .venv/bin/python Examples/Demo/start_pysc_demo_server.py
```

Wait for `Listening on 127.0.0.1:13131` and `Accepting commands...`.
The launcher loads the existing EBS pySC state and regenerates
`pysc_demo_catalog.json` from that same SC object.
It applies a demo-only BPM-noise override of σx = σy = 1.5 µm without changing
the saved EBS state. Override it when needed with `--bpm-noise-x-um VALUE` and
`--bpm-noise-y-um VALUE`.

### pySC corrector units

The demo's `B1L` and `A1L` controls are integrated normalized first-order
dipole strengths, `K₀L`. They are dimensionless in SI and numerically represent
the paraxial steering angle in radians, so ORM setpoints and changes are shown
in rad or µrad. `B1L` is the horizontal steering control and `A1L` is vertical.
The EBS configuration lists `B1L` under `invert`, so its control/setpoint sign
is inverted when mapped to the underlying AT normal-dipole component; `A1L`
uses the direct sign. Machine inventory pages show corrector counts, not an
arbitrary first-device value. Device names, setpoints, Δkick and restored
readbacks remain visible in ORM and Correction, where they have context.

Terminal 2 — Measure:

```bash
cd /Users/musa/Desktop/pyLOCO
MPLCONFIGDIR=/tmp/pyloco-mpl-measure .venv/bin/python -m pyLOCO.measure.app
```

## Monday Measure / Correct click script

1. In **Machine**, select **pySC Server**. Confirm the persistent `DEMO • pySC SERVER` badge, then click **Test connection**. It should report 320 BPMs plus RF and H/V corrector readback.
2. In **BPMs**, choose **Manual names / positions**, enter `0, 1, 2`, and click **Preview selection**. These are zero-based catalog positions and display the real EBS SC BPM names.
3. Open **Measurement**. Configuration is on the left and the complete acquisition workspace is visible on the right—no Review-tab navigation or resizing is required. Choose **BPM Noise**, 5 readings, 0.05 s delay, then click **Start BPM-noise measurement**; show live orbit, four final plots, and the highlighted saved HDF5/session path.
4. Choose **Dispersion**, **Automatic**, a 200 Hz total bipolar RF separation (±100 Hz), 3 readings, 0.05 s delay and 0.05 s settling. The plan shows the calculated negative/positive RF frequencies. Start once. Show the pyLOCO-compatible `negative − positive` RF orbit difference, physical Dₓ/Dᵧ using the served EBS lattice slip factor, and the `restored` RF status.
5. Choose **ORM**. On **BPMs**, click **Demo: select first only** once under Horizontal and once under Vertical correctors. Use 1 µrad bipolar kicks and 3 readings. Start; show the named selected column, heatmap, and both `restored` statuses.
6. Launch Correct in another Terminal 2 with `.venv/bin/pyloco-correct`. Load `Examples/Demo/pysc_demo_correction.json`, select **pySC Server**, and click **Preview machine changes**. Verify current/proposed values.
7. Click **Apply…** only after reviewing the confirmation naming `DEMO • pySC SERVER`. Confirm and show the matching readback. This demo correction intentionally changes the simulated corrector by only 0.1 µrad.

If the server is absent or a dependency/catalog is missing, the GUI reports the
error and blocks acquisition. It does not substitute Mock.

## Validated screenshots

The complete sequence was exercised against the actual local pySC server. The
screenshots are in `Examples/Demo/screenshots/monday-controls-demo/`:

1. `01-measure-pysc-connected.png` — DEMO badge and connected readbacks
2. `02-measure-bpm-selection.png` — searchable three-BPM selection
3. `03-measure-bpm-noise-live.png` — running/progress state
4. `04-measure-bpm-noise-result.png` — noise plot and saved output
5. `05-measure-dispersion-result.png` — measured Dₓ and RF restoration
6. `06-measure-fast-orm-selection.png` — one-click single-corrector setup
7. `07-measure-orm-result.png` — named ORM column and heatmap
8. `08-correct-preview.png` — current/change/proposed preview
9. `09-correct-readback.png` — matching readback and success state
10. `10-native-measurement-start-visible.png` — native 1200×800 Mac window with connected pySC backend and visible/enabled Start button
11. `11-bpm-noise-acquisition-running.png` — 20-reading acquisition remains on Measurement with progress and live orbit
12. `12-bpm-noise-completed-horizontal.png` — horizontal point plot with the backend-supplied demo-noise reference
13. `13-bpm-noise-completed-vertical.png` — vertical point plot with the backend-supplied demo-noise reference

The reproducible capture runner is `Examples/Demo/capture_monday_gui_workflow.py`.
`Examples/Demo/verify_interactive_measure.py` runs the native Qt click-path check,
including real Test connection and Start-button clicks against the server.
