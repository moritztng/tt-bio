# b2z2-bh-compose-v2 — the atom key gather elision, measured on a Blackhole fold

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: VERIFY/BENCHMARK + ACCELERATE + ALWAYS-ON
memories read: `b2z2-radical-2x-wave2`, `b2z-radical-2x-campaign`,
`precision-change-298aa-control-blind-to-512aa-failure`, `merged-lever-defaults-off-is-not-a-landed-win`,
`perf-page-cell-is-historical-not-live-baseline`, `no-speedup-by-skipping-the-models-own-work`,
`benchlock-one-shot-check-blind-to-mid-run-contention`, `git-merge-no-conflict-markers-can-still-be-semantically-broken`,
`unified-solution-not-per-model-patches`, `perf-gate-single-shot-legs-recurring-false-alarm`

STATUS: RUN IN FLIGHT — numbers land below when the benchlocked session finishes.

## 0. The premise correction this pass starts from

`b2z2-bh-compose-landed` §1 records: *"`b2z2-layout-op-elision` has an empty diff against main."*
**That is wrong.** `git diff --stat origin/main...origin/wk/b2z2-layout-op-elision -- tt_bio/`
reads **122 insertions, 15 deletions in `tt_bio/tenstorrent.py`** across four engine commits
(`e902b5cf`, `ccdc36ef`, `09964e25`, `9ca6c330`). The branch's first commit, `409b535f`, is the
pre-registered prediction and touches only `perf/` and `state/` — a check run at that commit sees
an empty engine diff, and the four engine commits landed after it. The predecessor read the
branch before it had concluded and recorded the reading as a property of the branch.

## 1. The merge, checked rather than assumed

`layout-op-elision` is a sixth engine-code branch. Checked pairwise with
`git merge-tree --write-tree`, all exit 0, no conflicts:

| pair | result |
|---|---|
| elision x algebraic-reformulation (K2) | clean, tree `d1b5bc4a` |
| elision x dst-resident-fusion (DST) | clean, tree `b811704d` |
| K2 x DST | clean, tree `c7f0dabae` |

Neither of the two conflicts the brief warned about is in this set: `host-residual-zero` x
`msa-track-attack` and `cb-depth-prefetch` x `dst-resident-fusion` both involve branches this pass
does not compose. (The sibling relaunch of `b2z2-bh-compose-landed` was running exactly those two,
HOST and CBD, on card 1 while this ran — the two passes are disjoint and both correctly serialised
on `benchlock`.)

**A clean merge is not a correct merge** (`git-merge-no-conflict-markers-can-still-be-semantically-
broken`), and two of the three branches edit the same file. So the hunks were compared directly:

* **K2** occupies `tt_bio/tenstorrent.py` lines **6014-6690**, all of it inside `TriangleAttention`
  — the trunk.
* **ELI** occupies lines **686-1172** (module-level helpers and the env flag) and **7052-10197**,
  inside `AttentionPairBias` / `Diffusion` / `DiffusionModule` — the sampler.
* **DST** does not touch that file at all: `tt_bio/reblock_permute.py` plus its own compute kernel.

Disjoint line ranges, disjoint classes, disjoint phases of the fold. That is also what makes the
additivity product meaningful: these three do not contend for the same term.

## 2. Run status (this pass)

The measurement is launched detached on qb2 card 3 and holds `benchlock` (pid 645829, acquired
16:10:50Z). It has NOT started folding, because benchlock is correctly refusing to start the clock
while a foreign fold burns CPU: another worker's `scripts/full_parity_gate.py` has been running
27+ min and is folding OpenFold3 at 27 % of a core.

**The quiet-wait was deliberately raised to `BENCHLOCK_LOAD_WAIT_S=3000`.** The default is 900 s,
and when it expires benchlock does **not** abort — it prints
`WARNING after ${LOADWAIT}s ... Proceeding, RECORD THIS` and measures anyway
(`benchlock.sh:93-96`). That is precisely how `b2z2-bh-compose-landed` lost its K2DST timing leg:
it proceeded under loadavg 4.6-9.2 and read the lever as slower than its own base. A first launch
of this run at the default would have started measuring at 16:24:38Z into a live parity gate, so
it was killed by explicit pid (626669, 626666) and relaunched with the longer wait rather than
allowed to produce a contaminated number.

Arms: `base, ELI, base, UNION` x 10 reps, plus 3 discarded warm folds. n=20 base, n=10 ELI,
n=10 UNION, 10 A/A pairs. `--skip-298`: all three levers are bit-exact, so the 298 aa control has
nothing to certify here; the CIF sha256 across arms is the parity evidence.

Output lands in `perf/b2z2_compose/out/eli_512_qb2c3.json` (written incrementally, one dump per
fold, so a partial run is still readable).

## 3. How the next pass resumes

Nothing needs rebuilding. The composed tree is committed and pushed
(`wk/b2z2-bh-compose-v2` @ `d8342a43`, on origin via the pc/laptop relay — qb2 has no GitHub
credential, `git push` there dies with `could not read Username for 'https://github.com'`).

1. **Check the queued run first.** `perf/b2z2_compose/out/eli_run.log` on qb2.
   * If it contains `benchlock: ... acquired after Ns` and then `rep0 base ...` lines, it ran.
     `perf/b2z2_compose/out/eli_512_qb2c3.json` is written one dump per fold, so even a partial
     run is readable and n>=5 per arm is already a usable answer.
   * **If it contains `benchlock: WARNING after 3000s ... Proceeding, RECORD THIS`, the timing
     leg is contaminated and must not be quoted.** Re-run it. The parity leg (CIF sha256 across
     arms) survives contamination and is still usable.
   * If the process is gone with neither, relaunch the command in §4.
2. **Score it:**
   `python3 perf/b2z2_compose/score.py --run perf/b2z2_compose/out/eli_512_qb2c3.json
   --cifdir perf/b2z2_compose/out/cif --out perf/b2z2_compose/out/scored_eli_qb2c3.json`
3. **Read the prediction before the numbers**, `perf/b2z2_compose/PREDICTION_v2.md`, committed at
   `02434441` before the first fold existed. Six falsifiers, each able to fire.

The arms cannot silently run their base: every fold asserts its own lever counters
(`ATOM_SHIFT_GATHER_STATS` must read 1200/0 with ELI on and 0/1200 with it off, `FUSED_STATS`
likewise for K2) and aborts the run if they contradict the arm. That is brief bar 5 enforced at
the fold rather than inspected afterwards.

`STATS_GATED` still cannot distinguish DST's two variants — it counts served calls in both arms,
because DST is a compile-time switch inside the same op. `_cache_key_gated` carries
`GATE_DST_RESIDENT`, so the arms are genuinely different compiled programs; the counter just
cannot witness it. Unchanged from the predecessor's pass and unchanged in its consequence: DST
measured 0.99748x on this cell and nothing turns on it.

## 4. Reproduce

    cd /home/ttuser/.coworker/wt/b2z2-bh-compose-v2
    setsid nohup env BENCHLOCK_WAIT_S=2400 BENCHLOCK_LOAD_WAIT_S=3000 \
      ~/.coworker/scripts/benchlock.sh worker:b2z2-bh-compose-v2 -- \
      env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
          TT_BIO_LEASE_HOLDER=worker:b2z2-bh-compose-v2 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2z2_compose/ab_compose.py \
        --out perf/b2z2_compose/out/eli_512_qb2c3.json \
        --cifdir perf/b2z2_compose/out/cif --reps 10 --skip-298 \
      > perf/b2z2_compose/out/eli_run.log 2>&1 < /dev/null &

## 5. Deliverable lines — NOT YET EARNED

FOLD-SECONDS-BH: pending — the run is queued behind another worker's parity gate.
FOLD-RATIO-BH: pending.
UNION-DISCOUNT: pending.
ARMS-COMPOSED: 3 merged and verified disjoint (ELI, K2, DST); 0 measured so far.
PARITY: pending (CIF sha256 across base/ELI/UNION).

No number is quoted in this pass. The prediction is registered, the tree is composed and
checked, the harness is instrumented, and the run is queued — but a fold ratio that was never
measured is not a result, and a fold ratio measured under a live parity gate is a wrong one.
