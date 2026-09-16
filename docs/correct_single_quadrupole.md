# Single PETRA pySC B2 transaction

Only PETRA III / realistic_errors normal B2 is enabled for application. No
skew, bulk FIT correction, or LIVE DOOCS application is enabled. FIT and Measure
are unchanged. The test input is synthetic, not a claimed LOCO fit result.

## Run

From the repository, with unused local ports:

```sh
.venv/bin/python Examples/Demo/start_pysc_server.py --profile petra3_realistic --port 13331 --diagnostics-port 13332 --catalog /tmp/correct-profile-catalog.json
```

In another terminal:

```sh
.venv/bin/pyloco-correct
```

Choose **pySC Server → Single B2 simulation → PETRA III / realistic_errors**.
Set diagnostic port **13332**, then Connect. Expect **417 B2 controls**.
Load `Examples/Correct/petra_single_b2.json`, Preview, Confirm and Apply,
then **Restore original**. The mapped file determines the target; the inventory
is informational, not an implicit mapping override.

The optional diagnostic HTTP endpoint is read-only and bound to localhost. It
reads `PolynomB[1]` directly from the same SC object used by the normal pySC
server. It does not calculate the reported physical readback from the command.
Existing pySC magnet commands are unchanged. Do not operate another client
against this test server during a transaction; the protocol has no atomic
compare-and-set or exclusive ownership. Stale preview and mismatched snapshots
are rejected, but cannot eliminate concurrent external-writer races.

Mapping validates the lattice SHA256, official CommonName/index, exact B2
control and units. This milestone requires the same official lattice identity
on the source side: transfer from a differently ordered fitted lattice remains
disabled. Calibration factors are simulation truth, not hardware calibration.
The deliberately small physical change is capped at 1e-5 m^-2.

Journals are fsynced before SET under `correction-transactions/`. Failed Apply
attempts restore the current item, including a SET followed by a failed GET.
Restoration failures are surfaced and retained. Recover journal supports the
same running server instance; a restarted/different server is deliberately
rejected. Restore writes the recorded original, never zero. One item only means
there are no previously completed bulk items in this milestone.

## Validated result

Native GUI Preview → confirmation → Apply → Restore was exercised against the
actual isolated pySC server. Target `Q0K2_7_1/B2`, official CommonName
`Q0K2_SWR_0`, index 1, seed 20260907:

| Quantity | Value (m^-2 except dimensionless factor) |
|---|---:|
| Original control | -0.0794865 |
| Original physical K | -0.07944973020907886 |
| Requested physical change | 1e-6 |
| Calibration factor | 0.9995374083533538 |
| Control increment | 1.000462805736714e-6 |
| Applied control readback | -0.07948549953719426 |
| Actual physical K | -0.07944873020907886 |
| Actual physical change | 1.000000000001e-6 |
| Restored control | -0.0794865 |
| Restored physical K | -0.07944973020907886 |

The GUI showed DEMO • pySC SERVER, PETRA III / realistic_errors, seed, inventory,
and separate command/physical readbacks. Both original values were restored.
