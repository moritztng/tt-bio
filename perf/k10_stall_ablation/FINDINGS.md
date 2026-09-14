# k10-p1-stall-ablation — what does the compute-side input stall respond to?

Owner: `k10-p1-stall-ablation`, whglx card 2. **whglx is Wormhole. The number of record (17.989 s,
512 aa Boltz-2) is Blackhole.** Every measured number below is Wormhole and says so.
Rig, captures and response tables: `perf/k10_stall_ablation/` on `wk/k10-p1-stall-ablation`
(`ablate.py`, `split.py`, `run_whglx.sh`, `sweep{1,2,3}.json`, `sweep{1,2,3}_response.json`,
`sweep{1,2,3}_ops_perf.csv.gz`). No model code changed: `git diff origin/main...HEAD` is confined
to `perf/`.

Build: the stock profiler tt-metal at `/home/mthuening/work/b2z2-profiler/tt-metal`, v0.68.0
`1452925b`, Tracy on, `--enable-sum-profiling`. Every number here comes from that build. The
qb2 reader-side DM counters were not needed: this task asks a compute-side question.

Host thread cap `runtime.host_thread_cap_env(8)` on a 64-core box -> OMP/MKL/OPENBLAS/NUMEXPR
`NUM_THREADS=8`, in effect for all three sweeps, loadavg 5.3-18.9 at start. Per-arm wall spread
inside a sweep is 1.002x-1.06x, well under the 1.10x sign that the cap is not applied.

Threads, said correctly: `CB-COMPUTE-WAIT-FRONT` is **TRISC0** (unpack, blocked on input tiles),
`CB-COMPUTE-RESERVE-BACK` is **TRISC2** (pack, blocked on output room). Neither is TRISC1 and
neither says anything about the reader. Each is divided by that op's OWN `CORE COUNT` and
compared to its own thread's residency. **0 of 185 ops normalise inconsistently** on the right
divisor, against `cb_split.py`'s 51-of-272 figure, which was an artifact of dividing by TRISC1.
No "useful math" or "recoverable seconds" figure appears below: no instrument in this campaign
measures TRISC1's own stall.

## WHICH CB-DEPTH CASE EACH OP IS IN — the classification, first

* **GenericOp / `minimal_matmul`** ships `_MM_BLOCK[(4, 16)] = (4, 4, 1, 4, 1)`: `K_block == kt == 4
  == the whole contraction`. The producer emits **ONE** K block per output block, so **CB depth is
  worthless on this op as shipped**. Now measured rather than argued: 2x -> 3x is 0.9968x and
  2x -> 4x is 1.0022x against a 1.0021x A/A wall floor, with the PROGRAM HASH different in every
  arm so the knob provably engaged.
* **Matmul / `bmm_large_block`** emits `K / in0_block_w` blocks, i.e. **more than one**. Depth is
  live in principle there but `ttnn.matmul` exposes no knob for it, so it was probed through
  `K_block` on the generic transcription. At `K_block = 2` (two blocks) a 2x -> 4x deeper CB is
  **0.9528x, 4.7 % SLOWER** -- deeper costs L1 and buys nothing.

INSTRUMENT-CONTROL: **passed, prediction written first.** Two stock ops whose binding limit is
decidable from arithmetic intensity, interleaved in the same process as everything else. Predicted:
the DRAM-bound `ttnn.add` (4096x4096 bf16, 96 MB, 0.17 flop/byte) above 80 % per-core
CB-WAIT-FRONT / TRISC0, the compute-bound HiFi4 `ttnn.matmul` (2048^3, 17.2 Gflop) under 10 %.
Measured, WH: **89.9 %** and **6.8 %**, a 13.2x separation, stable to 0.1 pp across all three
sweeps. Second control: the rig reproduces the model's own capture -- `ttnn.matmul` on a
DRAM-interleaved pair activation reads 82.8 % in the rig against 85.1 % for the same population in
the committed WH Pairformer block.

A/A floor, interleaved, same session: **generic wall 1.0021x / in-frac 0.00 pp**, **matmul wall
1.0003x / 0.01 pp**. The reduced-M arms carry their own, wider floor -- `gen.smallM.dram` against
`gen.smallM.dram2` are the same program hash and read 1.0020x, 1.0045x and **1.0215x** across the
three sessions -- so a reduced-M row is scored against 1.0215x, not against 1.0021x. That is what
demotes generic in0 -> L1 at M/8 from a 1.9 % result to a row sitting on its own floor: it reads
1.0148x, 1.0198x and 1.0191x across the three sessions, always in the same direction but never
clear of the worst-case floor. Generic out -> L1 at the same M is 1.2220x, 1.2189x, 1.2309x and is
not in doubt. Across the three separate sessions the same arm reads 2347.99 / 2367.73 /
2354.55 us (generic, **1.0084x**) and 1736.69 / 1738.84 / 1739.14 us (matmul, **1.0014x**). The
within-session floor is the bar for every row below, because every row is an interleaved arm.

## RESPONSE

RESPONSE: eleven knobs, one variable at a time, all arms interleaved in one process and one
device open, each against its own A/A floor (generic 1.0021x wall / 0.00 pp, matmul 1.0003x /
0.01 pp). Full table below. Headline: `d(wall)/d(stall)` is **1.17 = 1/0.847 almost everywhere**,
which is the trivial value meaning the op changed size and not composition. Only three rows
escape it -- `M_block` 4 -> 8 at **0.291** (-8.18 pp, 1.0315x) and math fidelity on `ttnn.matmul`
at **-0.517 / -0.474** (stall UP 3.13-4.41 pp, wall flat inside 1.2 %). The one knob that moves
the stall's share by more than 3 pp is operand residency: in0 -> L1 on `ttnn.matmul`,
**-27.19 pp, 1.3976x**.

sweep3, whglx card 2, WH, 5 interleaved reps of 37 arms in one process and one device open,
medians. `d(in/T0)` in percentage points; `d(wall)` is control/arm on `DEVICE KERNEL DURATION`;
`d(wall)/d(stall)` is microseconds of op wall bought per microsecond of input stall removed.
Op A is the shipped generic `minimal_matmul` at the boltz2 c_z=128 qkv+gate shape
(262144x128 @ 128x512, 72 cores, DRAM, HiFi4, 2354.55 us, in/T0 **84.7 %**); op B is `ttnn.matmul`
at the same activation against a 128x128 weight (1739.14 us, in/T0 **82.8 %**).

| knob | op | d(in/T0) | d(wall) | d(w)/d(s) | A/A floor | verdict |
|---|---|---|---|---|---|---|
| `K_block` 4 -> 2 | gen | +0.02 pp | **0.9194x** | 1.175 | 1.0021x / 0.00 pp | 8.8 % SLOWER |
| `K_block` 4 -> 1 | gen | -3.01 pp | **0.8266x** | 1.487 | 1.0021x / 0.00 pp | 21 % SLOWER |
| `M_block` 4 -> 8 | gen | **-8.18 pp** | **1.0315x** | **0.291** | 1.0021x / 0.00 pp | the one compositional win at full size |
| grid 8x9 -> 8x8 | gen | -0.04 pp | 0.9960x | 1.311 | 1.0021x / 0.00 pp | NULL |
| the recorded `(8,2,1,4,1)` on 8x8 | gen | +2.10 pp | **0.9078x** | 0.932 | 1.0021x / 0.00 pp | 9.2 % SLOWER |
| in1 -> L1 | gen | -0.40 pp | 1.0235x | 0.998 | 1.0021x / 0.00 pp | 2.3 %, at the floor's edge |
| in1 -> L1 | mm | +0.02 pp | 0.9998x | n/a | 1.0003x / 0.01 pp | NULL |
| in0 -> L1, M/8 | gen | -0.60 pp | 1.0191x | 0.881 | **1.0215x** / 0.13 pp | at its own floor |
| in0 -> L1, M/8 | mm | **-27.19 pp** | **1.3976x** | 1.065 | 1.0003x / 0.01 pp | **the lever** |
| out -> L1, M/8 | gen | -3.51 pp | **1.2309x** | 1.047 | **1.0215x** / 0.13 pp | real |
| CB depth 2x -> 3x, `K_block`=4 | gen | -0.08 pp | 0.9968x | n/a | 1.0021x / 0.00 pp | NULL, program engaged |
| CB depth 2x -> 4x, `K_block`=4 | gen | -0.37 pp | 1.0022x | n/a | 1.0021x / 0.00 pp | NULL, program engaged |
| CB depth 2x -> 4x, `K_block`=2 | gen | +0.75 pp | 0.9528x | 1.001 | 1.0021x / 0.00 pp | 4.7 % SLOWER |
| HiFi4 -> HiFi2 | gen | -0.10 pp | 1.0138x | 1.087 | 1.0021x / 0.00 pp | 1.4 % |
| HiFi4 -> LoFi | gen | -0.14 pp | 1.0130x | 1.085 | 1.0021x / 0.00 pp | 1.3 % |
| HiFi4 -> HiFi2 | mm | **+3.13 pp** | 1.0093x | **-0.517** | 1.0003x / 0.01 pp | stall UP, wall flat |
| HiFi4 -> LoFi | mm | **+4.41 pp** | 1.0119x | **-0.474** | 1.0003x / 0.01 pp | stall UP, wall flat |
| bfp8 on in1 | mm | +0.02 pp | 1.0011x | n/a | 1.0003x / 0.01 pp | NULL |
| bfp8 on both | mm | -0.56 pp | 1.0996x | 0.812 | 1.0003x / 0.01 pp | wall only, probe not lever |
| quantum M/2 | gen | -0.56 pp | 1.9803x | 1.173 | 1.0021x / 0.00 pp | scaling probe |
| quantum M/4 | gen | -1.79 pp | 3.9015x | 1.173 | 1.0021x / 0.00 pp | scaling probe |
| quantum M/2 | mm | -1.43 pp | 1.4395x | 1.964 | 1.0003x / 0.01 pp | scaling probe |
| quantum M/4 | mm | -10.51 pp | 3.1149x | 1.192 | 1.0003x / 0.01 pp | scaling probe |
| grid 4x9 at the SAME per-core quantum | gen | -4.31 pp | **1.2892x** | 1.004 | 1.0021x / 0.00 pp | **not a null** |

Three things in that table are worth saying out loud.

**`d(wall)/d(stall)` is 1.17 almost everywhere, and 1.17 is the trivial value.** `1 / 0.847` is
exactly the reciprocal of the stall's share of TRISC0. A knob that reads 1.17 did not change the
op's composition; it made the op bigger or smaller and the stall came along in proportion. Only
three rows escape it: `M_block` 4 -> 8 at **0.291**, and the two fidelity rows on `ttnn.matmul` at
**-0.517** and **-0.474**. Cutting the MAC passes 4x on `ttnn.matmul` *raises* the input stall by
4.41 pp and leaves the wall inside 1.2 %. That is the brief's own most-valuable outcome, from the
op side: **on this op the math is not the constraint and neither is the part of the stall that
tracks it.**

**The recorded "`K_block = 2` is worth 1.159x" does not hold here, and it was never a single
variable.** Its source (`b2z2-genericop-matmul-gap`) compares `(8,2,1,4,1)` on 8x8 against the
shipped `(4,4,1,4,1)` on 8x9, so `M_block`, `K_block` and the grid all move at once, at a much
smaller shape (M = 256 tiles against 8192 here). Decomposed at the Pairformer pair-tensor shape:
grid alone **0.9960x** (null), `K_block` alone **0.9194x** (slower), `M_block` alone **1.0315x**
(the entire win), and the recorded config as a whole **0.9078x** -- 9.2 % slower than what ships.
**The credit belongs to `M_block`, not to `K_block`, and the combination is shape-dependent enough
to invert.**

**Grid shape at a fixed per-core quantum is NOT a null on this part, and it is not the
`util-grid-coverage` experiment.** 72 has no second factorisation inside an 8x9 device, so shape
cannot move with the core count held fixed; what can be held fixed is the per-core quantum
(`transpose` is true, so `in0_axis_cores == gx`, and 8192/8 == 4096/4 == 1024 tiles per core with
`N_tiles_per_core` untouched at 2). Half the grid doing exactly the same work per core runs
**1.2892x faster**, so the 72-core arm is contending on something shared. Fitting
`t = a*(total work) + b*(per-core work)` to 2354.55 / 1186.03 / 1826.42 us gives **a = 1069 us
(45 %) shared, b = 1286 us (55 %) per-core**. `util-grid-coverage` held the cores engaged constant
and is untouched by this; the two measure different things.

## MODEL

MODEL: **the per-core TRISC0 input stall is a function of exactly one variable -- whether the bulk
operand reaches the unpacker without a DRAM round trip (by placement or by reuse) -- and of nothing
else in the program configuration.** Falsifier: any config knob that moves in/T0 by more than 3 pp
on either op without changing where the bulk operand lives or how often each fetched block is
reused. None of the eleven swept did.

**The per-core TRISC0 input stall is a function of one variable -- whether the bulk operand reaches
the unpacker without a DRAM round trip -- and of nothing else in the program configuration.**

Everything else in the sweep (`K_block`, CB depth, math fidelity, input dtype, per-core quantum,
grid shape at a fixed core count) leaves the stall's share of TRISC0 inside 3 pp of 84.7 %
(generic) and 82.8 % (`ttnn.matmul`), which is why `d(wall)/d(stall)` sits at the trivial 1.17.
Two knobs moved the share by more than 3 pp and both are the same variable wearing different
clothes: in0 -> L1 on `ttnn.matmul` (**-27.19 pp, 1.3976x**) is residency by placement, and
`M_block` 4 -> 8 (**-8.18 pp, 1.0315x**) is residency by reuse -- twice as many output tiles per
in0 block, so each DRAM-read block is consumed twice as many times.

**Falsifier**: any program-config knob that moves in/T0 by more than 3 pp on either op without
changing where the bulk operand lives or how many times each fetched block is reused. None of the
eleven knobs swept did.

**The model's own capture already contains the natural experiment, and it agrees.** Across the 339
`Matmul` calls in one fenced WH Pairformer block (`sweep`-independent, from the committed census
capture):

| `Matmul` rows, one WH Pairformer block x3 reps | n | device | in/T0 |
|---|---|---|---|
| in0 in **DRAM** | 42 | 37.6184 ms | **85.1 %** |
| in0 in **L1** | 297 | 45.4336 ms | **16.4 %** |

A 5.2x difference in stall share across exactly the variable the ablation names, observed in the
shipping model rather than in a rig. It is observational (the two populations differ in shape too),
which is why the controlled single-variable arm matters: same shape, same grid, same fidelity, only
the memory config moves, -27.19 pp.

## PREDICTS — per-lever predicted value for Phase 2

PREDICTS: L1 row-blocking of the DRAM-reading matmuls **1.0437x on the WH Pairformer block ->
1.0538x predicted BH** (upper bound; layout k = 1.23 +/- 15 %); `_MM_BLOCK` `M_block` 4 -> 8
**1.0190x WH -> 1.0234x BH**; generic output residency 1.2309x on the op at M/8, WH;
`K_block` **NEGATIVE (0.9194x)**; CB depth **ZERO**; math fidelity **ZERO on the wall**; bfp8 a
probe at 1.0996x on the op wall and still an end-to-end loss. Detail per lever:

Measured on Wormhole. Transfer to Blackhole uses `k10-transfer-function`'s per-kind rule, not a
single scalar: layout k = 1.23 +/- 15 %, arithmetic k = 0.59 +/- 20 %.

1. **Operand residency on the DRAM-reading matmuls (layout).** The 42 DRAM-in0 `Matmul` calls are
   37.6184 ms of the 255.3727 ms three-rep block, i.e. **14.73 %**, at 85.1 % input stall. The
   controlled arm is **1.3976x** on that op. If it carried, 37.6184 -> 26.92 ms, saving 3.57 ms of
   the 85.1242 ms block: **1.0437x on the WH Pairformer block**, predicted BH margin
   0.0437 x 1.23 = **1.0538x on the block, +/- 15 % on the margin**. This does NOT convert to fold
   seconds here -- the block's share of the 17.989 s fold is the orchestrator's number, not mine.
   **It is not a `memory_config` flip.** At production shapes in0 is 64 MiB and the output 256 MiB;
   neither fits this part's L1, which is why the arm had to run at M/8. The lever is row-blocking,
   and `tt-bio-l1-residency-row-blocking-pincer` plus
   `l1-budget-derived-from-live-grid-makes-output-host-dependent` are already on record against it.
2. **`M_block` 4 -> 8 on the generic `minimal_matmul` entries (layout/reuse).** Measured
   **1.0315x** on the op, WH, `d(wall)/d(stall)` 0.291, PCC untested by this task. The GenericOp
   population is 87.0197 ms / 34.08 % of the block; at 1.0315x on the `minimal_matmul` subset only
   (52.3 ms of that 87.0 ms), 52.3 -> 50.7 ms, **1.0190x on the WH block**, predicted BH
   **1.0234x**. Cheapest lever in the sweep: it is one tuple in `_MM_BLOCK`.
   Guard: changing `M_block` changes neither the contraction order nor the accumulation order, so
   unlike `K_block` it should stay bit-exact -- unverified here, verify before shipping.
3. **Output residency on the generic op**, 1.2220-1.2309x at M/8 over three sessions (WH). Same L1-capacity problem as (1),
   and the output is the bigger tensor, so it is strictly harder. Listed because it is real, not
   because it is near.
4. **`K_block`: predicted value NEGATIVE. Do not build it.** 0.9194x at 2, 0.8266x at 1, and the
   recorded `(8,2,1,4,1)` config as a whole 0.9078x at this shape.
5. **CB depth: predicted value ZERO**, proven with the program hash differing so the knob engaged;
   negative (0.9528x) where the producer does emit two blocks.
6. **Math fidelity: predicted value ZERO on the wall**, 1.0093x-1.0138x against a 1.0021x floor.
   Consistent with the recorded HiFi3 end-to-end null at 0.997-1.009x, and now with a mechanism:
   on `ttnn.matmul` it *raises* the stall 4.41 pp while leaving the wall flat.
7. **bfp8: stays a probe.** 1.0996x on the op wall with both operands narrowed, which confirms the
   stall responds to bytes -- and it is still the recorded end-to-end loss (0.949x, +18.9 % bytes,
   1.496 A) because `reblock_permute.eligible_back` flips 24/24 served to 24/24 declined.

## What this pass did not do

No Blackhole arm: this task's grant is whglx card 2 and every ratio above is Wormhole. The
residency lever is measured at M/8 only, because production operands do not fit L1 -- so its
**1.3976x is an upper bound on what a row-blocked version can return**, not a prediction of it.
`M_block` 4 -> 8 has no PCC or bit-exactness check. The rig dispatches each arm solo with a device
sync, so its absolute stall shares (84.7 % / 82.8 %) run higher than the same ops inside the block
(57.9 % / 85.1 %); the ratios between arms are what this task claims, not the absolute level.

VERDICT: PARTIAL — the response curve is complete on Wormhole against a calibrated instrument
(89.9 % against 6.8 % on the two known-answer controls) and it names exactly one live variable:
operand residency, by placement or by reuse. Five of the eleven knobs are proven dead
(`K_block`, CB depth, math fidelity, bfp8 on the weight, grid shape at a fixed core count), and
the campaign's recorded 1.159x `K_block` win is a three-variable change whose credit belongs to
`M_block`. Phase 2 gets two candidates with numbers attached — L1 row-blocking of the DRAM-reading
matmuls at a predicted **1.0538x on the BH Pairformer block** (upper bound, layout k = 1.23 +/- 15 %)
and the one-tuple `M_block` 4 -> 8 change at **1.0234x** — and a standing instruction not to spend
a day on CB depth or `K_block`.
