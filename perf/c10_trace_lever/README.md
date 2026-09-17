# c10-trace-lever — pricing the ttnn diffusion trace on the Boltz-2 fold

**VERDICT: NO-GO.** Trace replay of the per-step DiT device stream removes nothing from the
Boltz-2 fold. Predicted 3.04 s at 512 aa; measured **-0.021 s**, that is 21 ms *slower*, against
an A/A floor of 0.055 s. At 298 aa, predicted 1.48 s, measured -0.016 s against a 0.043 s floor.
The lever engages, runs the identical 200-step program sequence and produces a byte-identical
structure. It just does not make the fold faster.

Nothing shipped, nothing merged, and **no production source changed**: `git diff e6004abba --
tt_bio scripts/gpu_vs_tt/tt_baseline.py perf/other512/fold_ab_multi.py` is empty in both captures
and the harness refuses to run if it is not. The arm is an attribute flipped on the live model
inside the harness process.

## CLOCK

Two pinned arms, 1350 and 1000 MHz, requested through ARC FORCE_AICLK before every label and
sampled at about 1 kHz *during* every fold. In all 42 accepted folds the during-fold minimum
equalled the maximum equalled that fold's own requested target, with zero read errors and every
sample gap inside 10 ms. Board power tracks the clock, 79 W at 1350 MHz against 53 W at 1000 MHz
on the 512 aa arms, so each arm is witnessed by physics and not only by a telemetry register.
The 1000 MHz block trips `scripts/aiclk_watch.sh`'s CLAMPED-UNDER-LOAD guard on purpose; the raw
samples say it was requested, not suffered. Card 0 only, p300c, FORCE_AICLK released with a
checked return at both sizes.

## PREDICTED, recorded before any fold

`prediction.json`, digested into every `result.json`. Predicted delta at 512 aa: **3.04 s**
central, range 1.5 to 3.6 s, landing the fold near 11.84 s. At 298 aa: 1.48 s, range 0.7 to 1.8 s.
Predicted model cycles deleted: exactly zero, because both arms run the identical 200-step,
3-recycle, 1-sample, seed-0 configuration and the identical device program. Predicted parity:
byte-identical CIF, because the code documents the replayed graph as the exact captured program
with new input buffer contents.

The 3.04 s came from the parent's reach note (`perf/c10_orchestrator/dispatch_hypothesis/`,
`07c9db537`), rescaled onto `c10-fixed-cost`'s measured F of 3.9830 s on this exact tree: 219,200
of the fold's ~287,000 program-launching calls are the diffusion loop, and at 13.878 us of
clock-immune cost per launching call those carry 3.04 s.

Pre-registered kill criterion: a median gain at 512 aa inside the 0.055 s A/A floor retires the
lever in one pass. **It fired.**

## MEASURED

20 accepted warm folds per size after one discarded cold fold, arms interleaved inside one
process and one device context with the order reversed on odd repetitions. The 1 GiB trace region
is reserved in *both* arms, so the region is held constant and the only thing that varies between
the two labels of a pair is which callable the sampler invokes per step.

| size | clock | untraced median | traced median | delta | paired median delta | A/A floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 aa | 1350 MHz | 14.8746 s | 14.8960 s | **-0.0214 s** | -0.0052 s | 0.055 s |
| 512 aa | 1000 MHz | 18.5087 s | 18.5470 s | -0.0383 s | -0.0507 s | 0.055 s |
| 298 aa | 1350 MHz | 9.6580 s | 9.6741 s | **-0.0161 s** | -0.0244 s | 0.043 s |
| 298 aa | 1000 MHz | 12.3279 s | 12.3570 s | -0.0292 s | -0.0293 s | 0.043 s |

Six folds per arm at 1350 MHz, four at 1000 MHz. Every cell's delta is negative and every one is
inside that size's A/A floor. Speedup 0.9986x at 512 aa and 0.9983x at 298 aa. In Mcycles at the
measured clock: **-28.9 Mcycles at 512 aa and -21.8 Mcycles at 298 aa**, against a predicted
+4104 and +1998.

The consistent negative sign across all four cells is what the one-off trace capture costs: two
warmup steps and one recorded step of untraced diffusion, which is about 3 steps out of 200,
roughly 40 ms at 512 aa. That matches the observed 21 to 38 ms deficit. The lever's per-step
saving is not merely small, it is smaller than its own setup cost.

## The lever engaged. It is not a no-op.

Four independent witnesses, all recorded per fold:

- Both score-model entry points carry an identical counting shim, installed in both arms. Every
  traced fold ran exactly 200 calls into `forward_traced` and zero into `forward`; every untraced
  fold the reverse. A fold is rejected if that does not hold, so no fewer steps and no skipped
  work.
- Every traced fold left a live `_diff_trace` behind and every untraced fold left none. Capture
  happened, once per fold, and it was released between folds by the model's own
  `reset_static_cache`.
- The call census (`callcount.py`, `runs/census512/`) counts `ttnn.deallocate` per fold with the
  same wrapper in both arms, at 512 aa, 200 steps, 3 recycles. The untraced fold issues
  **101,559** of them; the traced fold issues **24,413**. Trace replay deleted **77,146 calls,
  76 % of the highest-frequency ttnn call in the fold**, and the fold got no faster. The 76 %
  matches the parent's predicted 76.4 % diffusion share of program-launching calls almost
  exactly, so the campaign's call accounting was right and only its cost attribution was wrong.
- The output is byte-identical, which a stale or bypassed trace would not reliably produce.

## PARITY

**Byte-identical.** One distinct CIF SHA256 across all 20 accepted folds at 512 aa and one across
all 20 at 298 aa, the same digest in both arms:
`bb7f0e1af7e271ee...` at 512 aa and `24c4b534d763788c...` at 298 aa. pLDDT is one value per size,
0.845919 and 0.908011, identical in both arms.

The float64 Kabsch scorer confirms it independently: maximum independently superposed domain
all-atom RMSD between a traced and an untraced structure is **3.7e-15 A** at 512 aa against the
0.60 A bar whose user seed floor is 1.84 A, and **7.1e-15 A** at 298 aa against that fixture's own
0.35 A bar whose archived upstream seed pairs span 0.741 to 0.847 A. That is float64 rounding, not
a deviation.

So the code's documented claim holds: the replayed graph is the exact captured program with new
input buffer contents. The lever costs nothing in accuracy. It also buys nothing in time.

## What this says about the campaign

The finding is not "trace replay is broken". It is that **the per-step host dispatch of the
diffusion loop is not on the critical path**. ttnn dispatch is asynchronous: the host queues the
1,096 programs of a step while the device executes the previous one, and at 512 aa the host stays
ahead. Removing 219,200 python-side program launches per fold therefore removes 219,200 launches
from a thread that was not the bottleneck.

That refutes the per-call reading of the clock-immune term. `c10-fixed-cost` measured F = 3.9830 s
at 512 aa and 1.9500 s at 298 aa; the campaign's working hypothesis put about 3.02 s of the 512 aa
figure inside the diffusion loop as reachable per-call dispatch. This row deletes essentially all
of that dispatch and the fold does not move, so **F is not mostly per-call host dispatch in the
diffusion loop**. F has to be somewhere else: featurization, MSA handling, output writing, or
device time in a clock domain AICLK does not drive. F being size dependent, roughly doubling from
298 to 512 aa, already pointed the same way.

The obvious follow-up, a trunk trace over the Pairformer stack, is **not supported by this
evidence**. A Pairformer layer is 137 programs against a diffusion step's 1,096, so the trunk has
strictly less per-call dispatch to remove per unit of device work than the loop that just returned
zero. Someone should still confirm the trunk is device-bound rather than assume it, but building a
trunk trace on the strength of the dispatch hypothesis would be building on a hypothesis this row
refuted.

## The 1 GiB region does not push a served size into OOM

| fixture | 1 GiB region | no region |
| --- | --- | --- |
| 1024 aa | completes, rc 0, pLDDT 0.822942 | completes, rc 0 |
| 1568 aa | `Buffer is not allocated` in `ttnn.reallocate`, rc 2 | same error, rc 2 |

1568 aa is the largest committed fixture and it fails identically with and without the region, so
the region is not what breaks it. That failure is reported here, not diagnosed; it belongs to
whoever owns large-target support, not to this row. The repeated L1 circular-buffer `TT_THROW`
lines at 1024 aa appear in both configurations and are the normal shape-routing fallback.
`runs/oom1024_region`, `runs/oom1024_noregion`, `runs/oom1568`, `runs/oom1568_noregion`.

## CONTROL

Every c10-bare-baseline and c10-fixed-cost control this row inherits is closed and was not redone;
`control.py`, `clockarm.py`, `force_aiclk.py`, `astsame.py`, `ambient.py`, `audit_accuracy.py` and
`audit_prerequisites.py` are byte-identical copies of c10-fixed-cost's. What this row adds is the
arm, and the arm is checked in both directions on every fold: the flag, the two step counters, and
the presence or absence of a live trace all have to agree with the label or the fold is rejected.
`reduce.py` re-derives every number from the committed capture on CPU and re-applies all of it.

A smoke capture at 20 steps (`runs/smoke2/`) proved capture, replay and digest equality before any
measurement fold was spent. It is marked `smoke: true` and `reduce.py` refuses to score it.

## COVERAGE

Every fold carries its own during-fold clock, board power, device-holder and host CPU coverage.
Device holders were sampled every 100 ms across all four chips by an independent thread; no
foreign holder at any time and no node other than 0 ever opened. Host CPU was sampled every 250 ms
over the whole locked window with per-session attribution. Containment stayed active, driver
srcversion A10759A24565BC5BBE903C5 and the boot ID were re-read before and after every fold, and
benchlock ran unchanged with its 60 s waits.

## UNCOUNTED

No device-cycle census, no per-op attribution, and no direct measurement of how much host wall
clock the diffusion loop's dispatch actually costs; this row shows that removing it does not
change the fold, not what it would cost if the device were faster. The mechanism split the second
clock was there to make is **not answered and cannot be**: with both deltas inside the A/A floor
there is no gain to attribute to a clock domain, so `analysis.json`'s `mechanism.verdict` field is
noise and must not be quoted. The 1000 MHz block's real value here is that it reproduces the null
at a second operating point rather than leaving it a single-clock result.

The trace region's own cost is bounded but not isolated: this row's untraced 1350 MHz cell reads
14.8746 s with 1 GiB reserved, against c10-fixed-cost's 14.9306 s and c10-bare-baseline's
14.8813 s without it. All three sit inside 0.056 s, so reserving the region costs nothing
measurable at 512 aa, but that is a cross-session comparison, not an in-process control.

## ARTIFACT

`perf/c10_trace_lever/` on branch `wk/c10-trace-lever`.

- `criterion.json`, `prediction.json` — recorded before any fold, digested into every capture.
- `runs/ab1/` — `launch.log`, per-size `result.json` with every fold row, `analysis.json`, and
  lossless gzip JSONL of every clock, power, holder and host CPU sample.
- `runs/smoke2/` — the 20-step proof that capture works. Not a measurement.
- `runs/census512/` — the `ttnn.deallocate` call census. Instrumented, so its seconds are not a
  measurement either.

Reproduce:

```
bash perf/c10_trace_lever/run.sh <name> 512 298
python3 perf/c10_trace_lever/reduce.py perf/c10_trace_lever/runs/<name>
```

`reduce.py` exits non-zero unless the capture completed and every prerequisite holds.
