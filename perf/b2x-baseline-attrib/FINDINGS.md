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
