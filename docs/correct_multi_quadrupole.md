# Small synthetic PETRA III B2 transaction

The B2 simulation workspace retains the single-item transaction and additionally
accepts 3–5 explicitly mapped synthetic normal-quadrupole items. It does not
consume real FIT corrections or enable skew or LIVE writes. No FIT, Measure,
machine-error configuration, or server protocol changes are involved.

Use the server launch command in `correct_single_quadrupole.md`, then select
pySC Server → B2 simulation → PETRA III / realistic_errors → diagnostic port
13332 → Connect. Load `Examples/Correct/petra_four_b2.json`, Preview, confirm
Apply, then Restore original. The calibration factors are simulation truth,
not real PETRA hardware calibration. Do not use another writing client on the
test server during this workflow.

The table has one row per magnet with physical K, requested physical change,
simulation calibration factor, proposed control, applied control readback,
achieved physical change and restoration status. Exact values and mappings
remain in the journal. The current-K column is the pre-Apply physical K; the
applied readback column preserves the Apply measurement after restoration.

## Transaction safety

- Preview verifies every explicit source/official lattice identity, B2 control,
  unit, calibration and command/physical consistency; no SET occurs.
- All originals are checked before the first write, and each item is checked
  again immediately before its SET.
- The complete journal is persisted before any write. Each current item is
  marked attempted and persisted BEFORE its SET, covering lost replies.
- Apply verifies command and independent lattice readback per item and again
  for the whole final state.
- Any failure restores attempted items in reverse order, including the current
  item. Untouched items are verified without writing them.
- A failed restoration does not prevent attempts on the remaining items.
  Failures remain explicit in the journal and can be retried.
- Restore requires exact equality with both original control and physical K.
- Restored counts include restored attempted items. Untouched originals are
  counted separately. Applied counts mean acknowledged SETs: a lost reply is
  not falsely counted as success even though its attempted item is restored.

## Real-server validation

Profile: PETRA III / realistic_errors; seed 20260907. Four controls distributed
around the official 3693-element lattice:

| Control | Index | Requested ΔK [m^-2] | Achieved physical ΔK [m^-2] |
|---|---:|---:|---:|
| Q0K2_7_1/B2 | 1 | +1e-6 | +1.000000000001e-6 |
| QF_95_105/B2 | 870 | -1.5e-6 | -1.4999999999876223e-6 |
| QB4_NOR_89_288_209/B2 | 1805 | +8e-7 | +7.999999999119822e-7 |
| QF_105_313/B2 | 2821 | -1.2e-6 | -1.2000000000067512e-6 |

Success: requested 4, acknowledged applied 4, verified 4, restored 4,
all restored YES. Native GUI confirmation/Apply/Restore was also exercised.

Failure test: a test-only client wrapper issued the actual third SET and then
raised a deliberate error. Items 1–3 were restored and both domains verified
exactly; item 4 was not written and its original values verified. Summary:
requested 4, acknowledged applied 2, verified before failure 2, restored 3,
untouched verified 1, all restored YES. No fault-injection mechanism is enabled
in the GUI or production server.

Reproduce both scientific tests on the isolated server:

```sh
PYTHONPATH=. .venv/bin/python Examples/Correct/validate_multi_b2.py --diagnostics-port 13332
```

Journals are runtime outputs, not source artifacts. Recovery is tied to the
same server instance; restarted or mismatched server identities are rejected.
