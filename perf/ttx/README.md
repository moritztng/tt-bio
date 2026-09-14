# TTX phase 0 — is the newer stack usable on Blackhole, and whose bug is the 18 A fold?

`b2z-ttnn-upgrade` found that ttnn 0.78.0 folds the 298 aa control 18.3 A away from the 0.68.0 arm
at half the plDDT, on Wormhole, through a binary-patched UMD. Nobody had said which side was wrong,
and "the new build disagrees with the old build" names no culprit. This directory answers both on
Blackhole, the part the published number runs on.

## Running the newer stack on a p300c

`run_arm.sh new <card> 298 <outdir>` folds on ttnn 0.78.0; `old` folds on the pin. Two things 0.78.0
needs on qb2 that 0.68.0 did not:

* `TT_MESH_GRAPH_DESC_PATH=perf/ttx/mgd/bh_1x1.textproto`. Without it `ttnn.open_device` dies at
  `tt_metal/llrt/tt_cluster.cpp:275`, "Custom fabric mesh graph descriptor path must be specified for
  CUSTOM cluster type". The box lands in CUSTOM because upstream's `mobo_to_bus_ids` table has no
  entry for its motherboard (`Unknown motherboard 'B850M-C'`), and upstream's own p300 descriptors
  describe 2 and 4 chips, which cannot map onto one pinned card. The committed descriptor is 1x1.
* SFPI 7.72.0, which 0.78.0 pins in `tt_metal/sfpi-version`. The system toolchain in
  `/opt/tenstorrent/sfpi` is 7.35.3 and rejects `-ftt-nttp`, `-ftt-constinit`, `-ftt-consteval`,
  `-ftt-no-dyninit`. Unpack it into the venv's own `runtime/sfpi`, which tt-metal resolves first, so
  the shared system toolchain stays where every other task on the box expects it.

No UMD patch and no source build: the released cp312 wheel opens a Blackhole chip in 2.6 s.

## The answer, scored against something that is neither stack

Both arms fold `cdk2x2_298` on qb2 card 2 with `TT_BIO_SHARED_DRAW_SEED=0`, so the diffusion noise
is the same stream upstream's own reference consumed (first draw `(1, 2400, 3)`,
`84c748a64d2a8963`, checked in both run records). Every hand-written fused kernel is off on both
arms, since they do not compile against 0.78's LLK surface. Scored with
`perf/b2z2_fusebias/score.py` unmodified against the committed `gpurefshared-s0` CIF, which is
upstream `boltz==2.2.1` at fp32.

| arm | all-atom RMSD vs upstream fp32 | CA-lDDT vs upstream | CA RMSD vs 1HCL | plDDT |
|---|---|---|---|---|
| ttnn 0.68.0, the pin | **0.4241 A** | 0.99636 | 0.7457 A | 0.9132 |
| ttnn 0.78.0 | **19.5562 A** | 0.31453 | 19.1921 A | 0.3597 |
| upstream fp32, 4 seeds | (reference) | | 0.5974 - 0.7398 A | 0.9092 - 0.9115 |

The third column is the one that closes the attribution. 1HCL is the experimental crystal structure
and no sampler argument reaches it. The pin lands at 0.7457 A, inside the reference's own four-seed
spread. 0.78.0 lands at 19.1921 A with CA-lDDT 0.31. It is not a drift and it is not fixture
sensitivity: on 0.78.0 the model does not fold the protein.

So the 18.3 A is **not ours in the sense of a fixture or a harness artifact**, and it is not the UMD
patch either, which this Blackhole run does not use. What remains open, and is the next step, is
whether 0.78.0 mis-serves a *configured* call tt-bio makes or is broken in an op outright: 20 ops at
their default configurations agree between the stacks, 16 of them bit-identically
(`perf/b2z_ttnn`), so the failure lives somewhere those defaults do not reach.

## The break is one release: ttnn 0.70.1

Same fold, same protocol, `TT_BIO_SHARED_DRAW_SEED=0`, qb2 card 2, plDDT and the CIF digest:

| ttnn | plDDT | CIF sha256[:16] |
|---|---|---|
| 0.68.0, the pin | 0.9132 | `aacd32c44c59ba85` |
| 0.69.0 | 0.9132 | `aacd32c44c59ba85` — byte-identical to the pin |
| **0.70.1** | **0.3642** | `e22acff2a06c6681` |
| 0.71.2 | 0.3642 | `e22acff2a06c6681` — byte-identical to 0.70.1 |
| 0.73.1 | 0.5764 | `01a6109c5aa0b069` |
| 0.78.0 | 0.3597 | `f5a4b4a1b9b098f2` |

0.69.0 reproduces the pin byte for byte, so ten minor versions of API and library change cost
nothing up to that point. 0.70.1 breaks it, and 0.71.2 breaks it identically, which makes this one
deterministic change rather than drift. The release window is 2026-05-05 to 2026-05-15.

## Three alternatives, all closed

**Seed-basin luck.** One seed in four can sit on a sampler decision boundary, where any bf16
perturbation flips the fold. Not this: seeds 1, 2, 3 read plDDT 0.9110 / 0.9133 / 0.9143 on 0.69.0
and 0.3638 / 0.3557 / 0.3561 on 0.70.1. Four for four, both ways.

**The UMD heartbeat patch** the Wormhole pass had to apply. Blackhole needs no patch, and the
failure reproduces here anyway.

**Our own sensitivity at the first diverging call.** The op trace puts the first disagreement at the
MSA pair-weighted-averaging softmax (`tt_bio/tenstorrent.py:9591`), and the divergence grows from
there with no step: 3.3e-4 at call 16, 1.7e-2 by call 40, 0.33 by call 253, 1.0 by call 557. So the
obvious reading is that the stacks differ a little there and the model amplifies it. Measured, that
reading is wrong. The real difference entering at that call is **0.08 bf16 ULP averaged over the
5.74% of elements that differ, max 2 ULP, 99.8% coherent**, and perturbing every element of that
same output on the good stack by a full ULP, coherently, costs:

| arm | vs upstream fp32, all-atom | CA-lDDT | vs 1HCL crystal, CA |
|---|---|---|---|
| unperturbed control | 0.42410 A | 0.99636 | 0.7457 A |
| +-1 ULP, random sign | 0.70283 A | 0.97907 | 0.8743 A |
| +1 ULP, coherent | 0.51327 A | 0.99124 | 0.7808 A |

A perturbation strictly larger than the observed one costs 0.09 A and keeps CA-lDDT at 0.991. The
model is not knife-edge there, so that call is not the cause.

## Where it is not, which is most of the way there

Every op the trace covers is individually accurate on 0.70.1, scored in situ against a float64
reference built from the operands the device actually got, with the fold's own program config and
memory config:

| call | op | shapes | 0.69.0 rel L2 vs float64 | 0.70.1 rel L2 vs float64 |
|---|---|---|---|---|
| 16 | `softmax` | [8,320,320] | 1.8e-3 | 1.9e-3 |
| 246 | `linear` | [1,320,320,128] @ [128,128] | 1.636e-3 | 1.636e-3 |
| 247 | `linear` | same, larger operands | 1.730e-3 | 1.730e-3 |
| 250 | `linear` | [320,320,128] @ [128,4] | 1.650e-3 | 1.656e-3 |
| 253 | `linear` | [320,320,128] @ [128,128] | 1.630e-3 | 1.768e-3 |

PCC is 0.9999984 or better on both sides everywhere. Call 253 is where the relative L2 between the
two stacks' outputs steps from 2.3e-3 to 0.559 in a single call, and it is still computing its own
inputs correctly: what changed is that its **inputs** arrived 22% apart (float64 reference absmax
21.786 against 26.583). Those inputs come from a call the wrapper does not cover.

So the traced surface is clean and the damage enters through the untraced one. `op_trace.py`'s `OPS`
list covers the out-of-place ops only. It misses the in-place variants tt-bio leans on (`add_`,
`mul_`, `softmax_in_place`), `ttnn.experimental.*` including `minimal_matmul` and
`nlp_create_qkv_heads`, and `ttnn.transformer.scaled_dot_product_attention`. Widening the wrapper
to those and re-running the 236-257 window is the next step, and it is short: the window is already
known and so is the instrument.
