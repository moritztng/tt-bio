# qb2 768/1024 rows — run notes

Not the deliverable. The deliverable is a dated section in `~/.coworker/state/protenix-v2-sizes-perf.md`,
and it is not written until the folds have produced numbers. These notes exist so a restart does not
re-derive the two things this pass established before the device was ever reached.

## 1. Why the previously registered run produced zero JSON

`0d9c7da1` registered the run and launched it. Every one of its six processes died in **4 seconds**,
before opening the device:

```
File "perf/size512/fold_ab512.py", line 281, in set_arm
    T._TRANSITION_CHUNK_SEEN.clear()
AttributeError: module 'tt_bio.tenstorrent' has no attribute '_TRANSITION_CHUNK_SEEN'
```

`fold_ab512.py` was lifted from `wk/protenix-v2-sizes-perf`, where that dict is a Transition chunk
census added to the engine. It never left that branch. The harness therefore only ran against the
branch it was written on, and moving it anywhere else was an instant crash at `set_arm` — which runs
*before* `build_fold`, so the failure is free and loud rather than expensive and silent. Evidence kept
in `perf/pxsizes/q1_qb2.log`.

Fixed in `7a2bc405` by reading the dict through `getattr` and recording absent as absent. The census
is not worth porting into the engine: it costs two device syncs per Transition call.

The four counters this task actually needs are all on main, checked by `hasattr` before relaunching:
`reblock_permute.STATS_GATED` (E6), `triatt_sdpa.STATS` (K2), `triatt_qkv.STATS` / `TAIL_STATS`
(K1 / K1b). `_TRANSITION_CHUNK_SEEN` was the only missing name of the 50 the harness touches.

**Transferable:** a perf harness that pokes engine internals is only portable to branches carrying
those internals. Check the attribute set, not just the file.

## 2. Why each arm runs twice

The brief asks for one clean pair per size. One pair is not clean here, for a reason specific to qb2:

§12.2 of the state doc records that **no `base4` arm has ever completed at 768 or 1024 on qb2** — both
multi-arm legs died in fold 2, before reaching it. So the ttnn fallback kernels that K1/K1b displace
have never JIT-compiled at these shapes on this box, while the `on` arm's kernels have (qb2 folded
both sizes `on` in the first pass). §11.3 measures first-ever compile at **71.70 s at 768** and
**80.99 s at 1024**, against a cache-warm per-process cold cost of 7.55 s / 5.52 s.

The qb1 gaps this run corroborates are **8.396 s at 768** and **15.103 s at 1024**. An uncontrolled
first-compile inside `base4` is 5-10x that and lands on the slow side of the comparison, so it would
**manufacture a win of exactly the size being claimed**. Two processes per arm, order
`on, base4, on, base4`, ratio from the second of each, `|p1 - p2|` per arm reported as both the
compile magnitude and that arm's own cross-process floor.

## 3. Instrument

`perf/pxsizes/run_q1b_qb2.sh`, predictions Q1-Q8 and both stop rules in its header, committed at
`28527343` before launch. One arm per process with `--skip-cold` (§12 makes that permanent, not a qb2
workaround). One benchlock hold per size. Card 2, ttnn 0.68.0 verified at run start, fixture sha256s
printed into the log and byte-identical to the files the qb1 rows used.

`perf/pxsizes/q1b_report.py` turns the JSONs into the row, the spreads, the stop rule, the component
decomposition and the counter table. It computes ratios and shares and nothing else.

## 4. Contention

The run was launched 14:57Z and queued: `openfold3-to-3x-perdollar` has held benchlock since 14:22Z
for a 1024 aa two-arm A/B, 40 min in with 2435 s of CPU and no output written. Not diagnosed further
on purpose — `py-spy` pauses its target, and pausing another worker's benchlocked timed fold would
corrupt its numbers.

## 5. Pass 2 (2026-08-13 16:35-17:20Z): 768 landed, 1024 queued

768 is **done and written to the state doc, §14**: `base4/on = 155.350 / 134.732 = 1.1530x`, gap
20.618 s, worst cross-process spread 4.954 s, stop rule satisfied, byte-exact parity across all four
arms (CIF `6697ecc2892d2993`, plDDT 0.787723). Artifacts `perf/pxsizes/q1b_768_*.json`, committed.

**The headline is that it contradicts qb1's 1.0540x, and the counters say why:** K2
(`persistent_mask`) serves 1208 calls on qb2's 11x10 grid and 0 on qb1's 13x10. E6 serves 0 on both.
`base4` TriAtt agrees across hosts to 0.6 % while `on` TriAtt is 11.9 s faster on qb2, so the extra
win is K2 alone. See §14.4.

### For the next launch — do NOT re-run 768, and do NOT relaunch the script blind

`run_q1b_qb2.sh` was launched detached at 16:42Z and **is still live**, queued on benchlock behind
`openfold3-1024aa-confirm-and-merge` (which took the lock at 16:53:14Z, the second the 768 leg
released it). The 1024 leg writes `q1b_1024_{on,base4}_{p1,p2}.json` one file per arm as it goes.

1. `ps -eo pid,etimes,args | grep run_q1b_qb2` first. If it is alive, **do not launch a second
   copy** — two of them would interleave four processes each on one card.
2. If the JSONs are there, `perf/pxsizes/q1b_report.py` prints the row; write it into state §14.6
   and replace the "owed" line in §14.7's table.
3. If the run died (host hang, timeout), relaunch only the 1024 leg:
   `bash ~/.coworker/scripts/benchlock.sh protenix-v2-qb2-768-1024-rows -- bash perf/pxsizes/run_q1b_qb2.sh --leg 1024 1300`
4. Q3's 1024 half is untested. Given 768, expect the qb2 1024 ratio **above** qb1's 1.0524x if K2
   still serves at 1024 on this grid, and near it if the `pm_over_l1` gate closes at the larger
   shape. The counter in the JSON answers it directly — read `persistent_mask.served`.

### Worth its own leg, not run here

K2's contribution on qb2 is inferred from a cross-host subtraction, which is the weakest step in
§14.4. A `nok2` arm at 768 on qb2 (the harness already has that arm) would measure it directly in
one process, ~155 s. It was not run because the script goes straight from the 768 leg into the 1024
leg under a fresh benchlock hold, and a co-tenanted arm would not be comparable to the four above.
