# Boltz-2 512 aa on Blackhole: the baseline re-measured, and where the fold's bytes go

P0 of the `boltz2-2x` campaign. Card 1 of qb2 (p300c, one Blackhole processor), ttnn 0.68.0,
commit `13604183`, protocol `perf/size512/fixtures/cdk2x2_512.yaml` + its fixed 35-row a3m,
3 recycles, 200 sampling steps, 1 diffusion sample, seed 0, templates off. Timed region is
`_WorkerState.predict_one`: host featurisation, fold, CIF write. Both timed runs held
`benchlock` (`acquired after 2s, loadavg 1.93 3.38 4.44` for the `--fast` run; the default run
acquired first and released at `loadavg 5.91 4.93 5.13`).

## 1. The baseline has not moved

**23.841 s**, median of n=3 warm (23.841 / 23.795 / 24.200), cold fold discarded. Host 0.323 s,
device 23.465 s. The published cell is 23.504 s from 2026-08-26. The gap is 0.337 s, 1.4 %, and
this session's own A/A floor -- the spread between the plain and instrumented arms of the same
fold -- is **0.405 s (1.70 %)**. The difference is smaller than the floor, so nothing regressed
and nothing improved: current main folds the published cell at the published speed.

Use 23.841 s as the campaign's denominator, not 23.504 s, because every child's arm is measured
against this session's instrument on this card.

## 2. The published CIF digest is dead, and that is not a regression

The published cell's `fca25e32ea181ae2` is **not** what current main writes. Main writes
`4f3995a69be5d610`, plDDT 0.849627, bit-identical across all three warm folds and across the
attrib and census folds in the same process.

The structure did move, and the size of the move depends entirely on which fixture you read it on:

| comparison | all-atom RMSD | CA RMSD | mean plDDT then -> now |
|---|---|---|---|
| `cdk2x2_512`, whole structure | **8.597 A** | 8.444 A | 0.8694 -> 0.8506 |
| `cdk2x2_512`, residues 1-290 superposed alone | 0.865 A | | |
| `cdk2x2_512`, residues 301-512 superposed alone | 0.980 A | | |
| **`cdk2x2_298` monomeric control** | **0.218 A** | 0.092 A | 0.9114 -> 0.9102 |

The 8.6 A is the chimera's hinge, exactly as memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`
says: superpose each domain on itself and both land under 1 A, so the two halves are intact and
only the angle between them changed. On the monomeric control, which is the fixture the campaign's
parity bar is actually defined on, main's total drift since 2026-08-26 is **0.218 A -- inside the
0.35 A pass bar** -- with plDDT flat to 0.0012.

References for both sides are committed under `ref/` so no later child has to re-fold to compare:
`published_fca25e32_cdk2x2_512.cif` and `published_388bd6db_cdk2x2_298.cif` are the 2026-08-26
structures; `main20260911_4f3995a6_cdk2x2_512.cif` and `main20260911_71653ff7_cdk2x2_298.cif` are
what current main writes.

**What this costs the campaign:** any child that scores its arm against `fca25e32ea181ae2` will
read a bit-exactness failure that its own change did not cause. The reference is
`4f3995a69be5d610` / control `71653ff7...` until main moves again.

## 3. The 1.400 TB remainder does not exist

Measured directly, the fold moves **5.386 TB**, not 6.45 TB, and **99.9 %** of it is inside a
named module. The four signatures the published budget instrumented carry 5051 GB; everything
outside them is **334 GB**, of which 328 GB now has a name.

**REMAINDER-ATTRIBUTED: 98.1 %.**

The 1.400 TB was never measured. It was inferred: the roofline pass took the 5.42 s of fold time
its four brackets did not cover and charged it at the same GB/s the brackets achieved. That
assumption is what fails. The unbracketed time does not run at the phase rate; it moves 334 GB,
so it runs at roughly 125 GB/s, and about 1.07 TB of assumed traffic is not there.

Where the 328 GB went: the published "MSA block" row (16 x 11.809 GB) is not the MSA layer. It is
the pair-only `PairformerLayer|1x512x512x128` that runs inside it. The real `MSALayer` moves
**32.244 GB/call**, and the difference -- `OuterProductMean` at 11.778 GB/call and
`PairWeightedAveraging` at 7.651 GB/call -- is the whole remainder, 0.516 TB against the 0.189 TB
that was counted.

The fold now closes by construction:

| phase | calls | GB/call | TB/fold | share |
|---|---|---|---|---|
| `TrunkModule/Pairformer/PairformerLayer` | 256 | 12.169 | 3.115 | 57.8 % |
| `DiffusionModule` (token layers + atom transformer) | 200 | 8.254 | 1.651 | 30.7 % |
| `TrunkModule/MSA/MSALayer` | 16 | 32.244 | 0.516 | 9.6 % |
| `PairformerModule/Pairformer/PairformerLayer` (confidence) | 8 | 12.169 | 0.097 | 1.8 % |
| glue, belonging to no module | | | 0.006 | 0.1 % |

The four published per-call figures reproduce to six digits on this instrument (12.169135,
11.808086, 0.517064, 0.214449 GB/call), so the byte counting is the same counting; only the
remainder term changed.

## 4. The model the campaign is priced off, corrected

    t_device = B / R      B = 5.386 TB, t = 23.465 s  =>  R = 229.5 GB/s

That is **53.4 % of the 429.9 GB/s streaming roof**, not 65 %. Both terms move against the plan:
there is 1.07 TB less traffic to delete than the portfolio assumed, and more headroom in R --
a perfect-rate fold would be 12.5 s, so R alone has a 1.87x ceiling rather than 1.54x. 88.8 % of
the fold's wall is inside the bracketed phases; the other 2.7 s is not bandwidth-bound and will
not respond to byte deletion at all.

## 5. Per-op and per-dtype attribution

`ATTRIBUTION.md` (generated, with `attribution_512_qb2c1.json` behind it) gives every captured
unit: sub-unit split, per-op DRAM read/write, and the dtype census by bytes read. Three things it
settles.

**The dtype census is closed.** Every unit in the structure path reads **100.00 % BFLOAT16** by
DRAM bytes, except the two diffusion transformer stacks, which read 0.01 % and 0.07 % `BFLOAT4_B`
(0.8 MB/call, the compressed conditioning). No fp32 anywhere. This is now a runtime census over
the tensors the ops actually touched, not a reachability argument over gate defaults
(memory `rf3-runs-bf16-on-gpu-kernel-counter-is-not-a-dtype`).

**Inside the 12.169 GB pairformer block**, by sub-unit:

| sub-unit | GB read | GB written | ttnn ops |
|---|---|---|---|
| TriangleMultiplication | 5.507 | 2.047 | 66 |
| TriangleAttention | 2.718 | 0.982 | 56 |
| Transition | 0.543 | 0.202 | 274 |
| AttentionPairBias | 0.146 | 0.024 | 32 |

The two triangle ops are 92 % of the block, and inside them the traffic is not in the projections:
`ttnn.generic_op` (the fused trimul and triangle-attention kernels) reads 6.256 GB and writes
1.749 GB, 66 % of the block. `ttnn.allocate_tensor_on_device` accounts for another 0.940/0.940 GB.
A fusion boundary for this block has to be drawn around the generic ops, not around the linears --
the whole `Transition` costs 0.745 GB in 274 ops, 6 % of the block.

**The diffusion token layer's 214.45 MB/call**, itemised at last:

| sub-unit | MB/call | share |
|---|---|---|
| AttentionPairBias | 142.9 | 66.6 % |
| ConditionedTransitionBlock | 33.4 | 15.6 % |
| AdaLN | 27.5 | 12.8 % |
| layer glue | 10.6 | 4.9 % |

and the top ops inside it are `softmax` 33.6 MB, `matmul` 30.4 MB, the transition's `linear`
25.6 MB, attention's `linear` 18.1 MB, `add_` 16.8 MB. At 512 tokens with 16 heads the attention
logit matrix is 8.4 MB in bf16, so softmax alone round-trips it four times and the attention
matmuls three more. The ~180 MB that `b2x-diffusion-layer-bytes` could not explain is the logit
matrix, moved a dozen times, not the token tensor.

## 6. `--fast` at 512 aa: accurate, and not faster

Same card, same protocol, same process shape, its own benchlock:

| arm | fold (median n=3) | plDDT | `cdk2x2_512` sha256 | `cdk2x2_298` control vs default |
|---|---|---|---|---|
| shipped default | **23.841 s** | 0.849627 | `4f3995a69be5d610` | reference |
| `--fast` | **25.201 s** | 0.865215 | `6ac64707dc71dedc` | **0.294 A** all-atom, 0.139 A CA |

`--fast` is **0.95x: 1.360 s slower**, against a predicted 1.15-1.35x speedup. The two arms ran in
separate processes, so the honest comparison is cross-process, but the offset that costs is ~0.3 %
in this lineage and the two arms' own A/A floors were 1.70 % and 0.44 % -- a 5.7 % gap is not a
process boundary.

The accuracy half is the surprise worth flagging: `--fast` at 512 aa lands at **0.294 A all-atom /
0.139 A CA on the monomeric control, inside the 0.35 A pass bar**, with plDDT 0.9106 against the
default's 0.9095. `_FAST_MODE` at this size is accurate enough to ship and buys nothing, so what
it bounds is not the precision axis in general -- it moves weights to `bfloat8_b`, and weights are
110 GB of a 5386 GB fold, 2 % of the traffic. The activation-side bet (`b2x-bfp8-pair-track`) is
untouched by this result. What this does say is that the *other* half of `_FAST_MODE`, the wider
trimul L1 path, is a loss at 512 aa on a p300c, and that a knob whose accuracy cost is inside the
bar can still be the wrong knob.

## Reproducing

    # one process, one device open: baseline + control + attribution + census
    benchlock.sh b2x-baseline-attrib -- env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:b2x-baseline-attrib python3 -u \
      perf/b2x-baseline-attrib/baseline_attrib.py --phases baseline,control,attrib,census \
      --reps 3 --out perf/b2x-baseline-attrib/attrib_512_qb2c1.json \
      --cifdir perf/b2x-baseline-attrib/cifs
    # the table
    python3 perf/b2x-baseline-attrib/attrib_table.py --run .../attrib_512_qb2c1.json \
      --out-json .../attribution_512_qb2c1.json --out-md .../ATTRIBUTION.md
    # parity against any reference CIF
    python3 perf/b2x-baseline-attrib/control_rmsd.py --ref ref/<a>.cif --arm <b>.cif --out <x>.json
    python3 perf/b2x-baseline-attrib/domain_rmsd.py --ref ref/<a>.cif --arm <b>.cif   # per-domain

---

# Update, 2026-09-11 18:0x CEST: the byte counter was wrong, and the fold is dispatch-bound

Two corrections and one new result. Everything above section 6 that quotes a per-call byte figure
is on the old counter and is superseded by `ROOF_DEFICIT.md` and by this section.

## 7. Every byte figure in section 3 and section 5 is 1.1-3.1x high

`b2x-diffusion-layer-bytes` found that the published counter dedupes DRAM reads by **tensor id**,
and ttnn hands a reshape or an unsqueeze a fresh tensor id over the same buffer, so a free
metadata view was charged a full DRAM read. My instrument reproduced the published per-call
figures to six digits, which is exactly what a copy of the same rule does.

Recounted from **my own 26 captures** with the corrected rule (`real_traffic.py`, dedupe on buffer
address, views free, in-place charged a read and a write):

| phase | this task, old rule | corrected | sibling's independent capture |
|---|---|---|---|
| pairformer block | 12 169 MB | **6 650.7 MB** | 6 651 MB |
| MSA layer | 32 244 MB | **20 550.7 MB** | (not captured) |
| token DiT layer | 214.4 MB | **198.9 MB** | 198.94 MB |
| atom transformer layer | 517.1 MB | **303.9 MB** | 303.91 MB |

Two harnesses, two cards, agreement to 0.005 % on all three shared phases. **The fold's
instrumented traffic is 3.405 TB, not 5.386 TB and not 6.45 TB.**

The section 3 conclusion survives the recount in structure: the remainder is still not 1.400 TB,
the MSA layer is still 3.1x what the published "MSA block" row counted, and the fold still closes
inside named modules. Only the magnitudes move.

## 8. Nothing is bandwidth-bound. Nothing is compute-bound. The host is the fold.

`ROOF_DEFICIT.md` has the full table. Every named phase, against both roofs measured on this part:

| phase | ms/call | % of 429.9 GB/s | % of 85.96 TFLOP/s | ops/call | us/op | deficit s |
|---|---|---|---|---|---|---|
| pairformer block | 42.199 | 36.7 % | 13.9 % | 428 | 98.6 | **6.035** |
| diffusion step | 40.263 | 38.1 % | 9.3 % | 1816 | 22.2 | **4.214** |
| — token DiT layer | 1.078 | 42.9 % | 13.3 % | 60 | 18.0 | 2.398 |
| — atom transformer layer | 1.957 | 36.1 % | 2.7 % | 66 | 29.7 | 1.288 |
| MSA block | 122.781 | 38.9 % | 5.6 % | 1104 | 111.2 | **1.008** |

(The two indented rows are inside the diffusion step, not additional to it.) Deficit is what the
phase gives back if it ran at 80 % of the streaming roof, which is the ranking the campaign should
be steered by. Of 21.159 s of phase time, **7.920 s is explained by moving those bytes at the
streaming roof and 13.239 s is not**. The fold's whole 206.71 TFLOP is 2.405 s at the compute
roof. **Delete every byte of every named phase and the fold is still 15.921 s = 1.497x.**

So the diagnostic's third row fires everywhere, and the missing time has a name:

**One fold issues 487 202 ttnn calls, and 21.846 s of a 26.037 s fold is main-thread CPU — 83.9 %.**

Measured with `time.thread_time()` on the calling thread, which counts only the CPU that thread
burns, in `host_dispatch_probe.py`. 19.747 s of the wall is spent *inside* ttnn entry points, at a
mean 40.5 us per call. The probe ran co-tenanted at loadavg 6.5 so its wall is ~9 % above the
benchlocked 23.841 s; the ratio is the result, not the wall.

`process_time()` is 57.5 s on the same folds and is **not** the discriminator — tt-metal's
completion thread busy-waits, so process CPU is near 2x the wall whatever the answer is. The
calling thread is the one issuing work, and it is busy 84 % of the fold.

The obvious alternative reading is that the calling thread spins on a full command queue, i.e.
that this is the device's backpressure wearing a host costume. Three things argue against it.
The device is at 37 % of one roof and 10 % of the other, so there is nothing to back up against.
`ttnn.deallocate`, a host-only call with no device work, costs 0.6 us across 138 213 calls, so
cheap calls are not stalling. And the expensive calls are the ones that build a program or a
config: `ttnn.linear` is 7.025 s over 109 887 calls, 64 us each.

**This also explains the co-tenancy sensitivity.** The published cell reads +10.7 % when the box is
busy. A device-bound fold would not care what the host is doing. This one does, because the host
is on the critical path.

## 9. The 0.382 s vs 2.74 s question, answered

They measure different things and neither describes the fold.

* **0.382 s** (published cell; I measure 0.323 s this session) is host time *outside*
  `model.predict_step`: featurisation 0.197 s, CIF write 0.058 s. It is correct, and it says
  nothing whatever about what the host does during the fold.
* **2.74 s** (`b2x-fusion-boundary`, derived from an op count) is an estimate of host issue cost
  *inside* the fold. The right quantity, and low by 8x.
* **21.8 s** is the measurement. Host issue is overlapped with device execution, so it is not a
  separate 21.8 s of fold time, but it is on the critical path for most of the fold.

## 10. What this means for the portfolio

The byte-deleting bets are priced against 7.920 s of byte-explained phase time, not against the
23.841 s fold. That is the ceiling the campaign has been hitting, and it is why every lever has
landed at 1.05-1.15x.

By bytes the pairformer is the target: 1.756 TB of 3.405 TB. **By op count the diffusion path is:
363 200 of ~487 000 calls per fold, 74 %, for 1.320 TB, 39 %.** A lever that removes ops from the
sampler is aimed at three quarters of the dispatch and nobody in the portfolio is holding one.
The 22.2 us/op there is not a fixed cost yet — `b2x-op-cost-curve` is queued to establish whether
it is — but the op *count* is measured and it is where the fold's calls are.
