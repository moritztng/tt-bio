# Re-gate attempt 5 — VERDICT: HOLD. Lever green everywhere it was measured, gate never finished.

`TT_BIO_SDPA_FUSED_LARGE_S` stays **off on main**. The flip lives on
`wk/ttx-a3-sdpa-ship-remerge`, pushed, unmerged. Nothing measured came back against the lever.
What stops the merge is that four of the gate's arms have now failed to record across three
consecutive passes, and this pass they failed because **no Blackhole card in the fleet was
available to run them**.

## What is green

| arm | result | where |
|---|---|---|
| below-cap neutrality, 768 aa | `38aabd4058facb3f` off/on/off, pLDDT 0.823946 all three | gate4, qb1 |
| below-cap neutrality, 1024 aa (the cap boundary) | `649aad7b46727c7e` off/on/off, pLDDT 0.821445 all three | gate4, qb1 |
| below-cap neutrality, 298 + 512 aa | one digest per size off/on/off | gate, gate2 |
| pytest, flag default-ON | 7 failed, 3670 passed — **0 attributable to the flip** | gate5, qb2 |
| UX surfaces, flag default-ON | GATE PASS, every surface cleared progress + parse + results | gate5, qb2 |
| 1536 aa fold, 200 steps | 1.1856x against a 1.21 % same-session A/A floor, 1.007 Å | banked, not re-measured |

Below the cap the flip is a no-op **by construction**, not only by measurement: the route is
gated on `q_len > _triatt_sdpa._Q_SPLIT_MAX_S` (`tt_bio/tenstorrent.py:1844`), so at and below
1024 tokens the branch is unreachable. The six neutrality folds confirm the construction holds
with `TT_BIO_TRANSITION_L1_ROWS` on, which is the one thing the 66-commit re-merge put at risk.

## The 7 pytest failures, all attributed, none the flip's

Two different controls, because the failures split into two kinds.

**Four are same-tree reds that the flag-off control reproduces.** `pytestoff-2` on the same tree
and card gives 5 failed / 559 passed against `pytest-2`'s 4 failed / 560 passed. The extra
failure in the control is `test_the_above_cap_route_is_strictly_above_the_cap`, which asserts the
flag reads `True` — it fails only in the off arm, which is what makes the control a control and
not a no-op. The four shared reds are three `test_capacity_gate` cells plus
`test_size_ladder_gate`, both recorded-cell debt.

**Three are tree-state tests, verified directly against `origin/main` rather than by control** —
an env var cannot move a tree-state assertion in either direction, so the control says nothing
about them and the tree does:

| failure | checked against `origin/main` |
|---|---|
| `test_repo_root_has_no_stray_directories` | `artifacts` and `patches` are both tracked directories in main's root (`git ls-tree origin/main`) |
| `test_cited_perf_artifact_exists` x2 | main's own `tt_bio/tenstorrent.py` carries both citations, and neither `perf/b2z2_adaln_sdpa/chunks_wh_c12.json` nor `perf/b2z2_layout/PER-SITE-TABLE.md` exists in main's tree |

The branch's entire code diff against main is five files, +43/-15: the flag default, the
`SDPA_FUSED_LARGE_S_STATS` counter, one README row, one docs section, one test. It touches none
of the files any of the seven failures reads.

## Why the gate stopped: three hosts, three different disqualifications

Measured this pass, not remembered.

- **qb1 is power-cycling at the chassis level**, and the netmon record is unambiguous about which
  kind of fault that is. `qb1-lan4`, `qb1-bmc`, `qb1-ts`, `qb1-ts22` and `qb1-tsp` all go `1->0`
  together, all come back together, and repeat: dark 20:24Z-21:46Z (82 min), **up 21:46Z-21:56Z
  (10 min), dark again from 21:56Z through the end of this pass** (1 h 43 m and counting). The BMC
  NIC is independent of the host OS, so a BMC that dies *with* the host is a power fault and not a
  host hang, and qb2 answered on the same LAN throughout, which rules out a switch flap. Redfish at
  `192.168.178.25` answers neither ping nor TLS while it is dark, so the documented remote power
  path is gone exactly when it would be needed.

  The 10-minute up window is the diagnostic, not a hopeful sign: qb1's BMC power-restore policy was
  set to `always-on` at 17:31Z, so the box boots itself the moment mains returns. **It recovering on
  its own is the policy working, and it dying again ten minutes later is the fault not clearing.**
  A "turn it back on" would not have helped — it already turned itself on. This is a supply or PSU
  fault that wants a human at the box, and it is already the subject of open ask 8537 from
  `qbgpt6-rootcause`, so this pass added evidence to that ask rather than a second one. Either way
  a box with a 10-minute duty cycle cannot carry a 44-leg parity arm. qb1 was the host running
  gate4, and it is the only 4-card Blackhole box.
- **qb2** is under an authorized exclusive reservation by `qbgpt6-rootcause`, which at 23:13Z
  stopped this gate's driver (PID 12054) mid-`pytestoff-3` and parked both of its crontab entries.
  That is correct behaviour on their side and not something to contest: Moritz put the QuietBox
  root cause ahead of everything else. qb2 also reset twice in the 80 minutes before this pass
  (22:20Z, 23:01Z), so it could not have carried a multi-hour arm anyway.
- **pc** is disqualified on both halves of the gate. Its single p150a is the card root-caused in
  `pc-card0-512aa-fold-nondeterminism` as silently miscomputing matmuls at a low location-keyed
  rate at every size, so no bit-exact or parity result measured there means anything; and it runs
  a custom 130-core firmware, so its timings are not the p150a the baselines were recorded on.

## The arms that never recorded, and what they would cost

`capacity` (15 cells), `size-ladder` (9 models), `perf` (20 models), `parity` (44 legs). On a
healthy box the driver runs them unattended and resumably — `perf/ttx_a3/gate_drive.sh` skips any
arm with an `rc=` row and reruns any arm with a bare `START`, so a reset costs one arm, not the run.

Worth knowing before spending that time: **only the arms that reach past 1024 tokens can see this
lever at all.** Everything at or below the cap is unreachable code, which the six neutrality folds
have now shown twice. That makes the capacity roster at 1536 tokens and the above-cap perf cells
the decisive arms, and the sub-1024 ladder rungs and parity legs a formality the gate requires
rather than evidence about the flip.

## The blocker that a healthy box would not clear

`docs/size_ladder_baseline.d/{boltz2,esmfold2}.json` is stale against nine already-merged
default-on levers, so the size-ladder arm is red for **every** branch including main with this
flag off. `tt-bio-sizeladder-p300c-refresh` owns it and is still in flight — no
`VERDICT-SIZELADDER-P300C` line in its state as of this pass. Until it lands, "fully green" is
not reachable for anyone, and a re-record done from here would bless nine sibling levers this
task never measured. That refusal is the gate working, not the gate broken.

## What the next launch does

1. Check qb1 has been up for more than an hour, not merely that it answers: it self-boots on mains
   return and has died again ten minutes later. Do not spend a pass on its BMC either way.
2. Check `qbgpt6-rootcause` has released qb2 (`crontab -l` on qb2: this gate's two entries
   un-commented).
3. Check `tt-bio-sizeladder-p300c-refresh` has written its verdict line.
4. With any one of those: `bash perf/ttx_a3/resume_after_boot.sh` with `GATE_OUT`, `GATE_CARD`,
   `GATE_WORKER` set, `GATE_SKIP_NEUT=1` (all six neutrality folds are banked, twice). Run the
   capacity and perf arms first; they are the ones that can see the lever.

## Reused, not rebuilt

`perf/ttx_a3/gate_drive.sh` and `resume_after_boot.sh` (the resumable driver and its cron relaunch),
`perf/ttx_a3/fold_parity_a3.py` for the neutrality folds and their digests, the release gate's own
`capacity_gate.py` / `full_parity_gate.py` / size-ladder scripts as RELEASING.md writes them, and
the banked 1536 aa ratio and Ångström reading, which this pass did not re-measure.
