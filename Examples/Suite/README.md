# pyLOCO Suite portable Mock handoff

This small example demonstrates the application boundary between pyLOCO
Measure and pyLOCO Fit without connecting to a control system or running a
large accelerator model.

1. Start `pyloco-gui`.
2. Choose **Open Measurement Session…**.
3. Open `mock_session/mock_complete.pyloco-session.json`.

The Fit project receives the ORM, BPM-noise and dispersion file references,
their exact BPM/corrector order, per-corrector requested and actual kicks, RF
step, acquisition metadata and session provenance. The session intentionally
contains no lattice, so it cannot be run as a fit until a compatible Mock
lattice/model is selected.

For the second handoff boundary, start `pyloco-correct` and open
`mock_results/`. This tiny precomputed Mock Results directory contains two
normal-quadrupole corrections plus the originating Measurement Session
provenance. It is an integration fixture, not a claim that the included Mock
session and lattice form a scientifically representative LOCO fit.

All paths in the session manifest are relative. The HDF5 files are small,
deterministic schema examples and contain no machine writes or PETRA data.
