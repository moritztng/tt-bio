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

So the traced surface is clean and the damage enters through the untraced one: the first `OPS`
list in `op_trace.py` covered the out-of-place ops only. Widening it found the call in one pass.

## Root cause: ttnn 0.70.1 changed what `attn_mask` means, and tt-bio compensates for the old one

`ttnn.transformer.scaled_dot_product_attention` applies `attn_mask` differently on the two sides
of 0.70.1:

| ttnn | what the op computes |
|---|---|
| 0.69.0 and earlier | `softmax((QK^T + mask) * scale) @ V` |
| 0.70.1 and later | `softmax(QK^T * scale + mask) @ V`, which is torch's convention |

Measured, not read. `op_trace.py --dump-args 346` saves the operands and the output of the first
call whose cross-stack difference is not rounding, and `sdpa_mask_convention.py` scores each arm's
real output against both float64 references built from that arm's own operands:

| ttnn | vs mask-after-scale | vs mask-inside-scale |
|---|---|---|
| 0.69.0 | rel L2 **3.6437e-1**, PCC 0.9309 | rel L2 **1.3888e-2**, PCC 0.99994 |
| 0.70.1 | rel L2 **2.0057e-2**, PCC 0.99989 | rel L2 **3.7956e-1**, PCC 0.9319 |

Each arm sits at bf16 distance from one convention and 18 to 27 times further from the other. The
operands agree across the arms to 1e-5 relative going in (`in_l2` in the scored record), so this
is the op, not something it was handed.

tt-bio pre-multiplies the pair bias by `_bias_scale = head_dim**0.5` (`tenstorrent.py:7578`) and
passes `scale = head_dim**-0.5`, so under the old convention the bias arrives at 1x. Under the new
one nothing cancels it and the bias arrives `sqrt(32) = 5.657x` too large. That is the whole 18 A.

**Whose bug: ours.** Both versions' docstrings say the same thing, "This API mimics the PyTorch
API of the same name", and torch adds `attn_mask` after scaling. 0.69.0 did not do what it
documented; 0.70.1 does. Our `_bias_scale` pre-scaling is a compensation for undocumented
behaviour, so the call was always wrong against the contract and only worked because the
implementation was wrong the matching way. Upstream shipped the correction silently, with no
docstring change and nothing in the release notes, which is why nobody caught it — but the latent
dependency is in our code and we own it.

Nothing is wrong on the pin today: `_fp32_softmax_attention` and the fused `_triatt_sdpa` kernel
both add the bias before applying scale (`tenstorrent.py:7166`), so every path agrees with itself
on 0.68.0. The break is confined to upgrading.

### The fold recovers, on both broken versions

`fold_maskfix.py` scales the mask by `scale` before the call, which restores the old identity
`softmax(QK^T*scale + mask*scale) == softmax((QK^T + mask)*scale)` without touching model code.
Same fixture, same shared draws, qb2 card 2, scored with `b2z2_fusebias/score.py` unmodified:

| arm | vs upstream boltz 2.2.1 fp32, all-atom | CA-lDDT | vs 1HCL crystal, CA | plDDT |
|---|---|---|---|---|
| 0.68.0, the pin | 0.4241 A | 0.99636 | 0.7457 A | 0.9132 |
| 0.69.0 | 0.4241 A | 0.99636 | 0.7457 A | 0.9132 |
| 0.70.1 | 21.0793 A | 0.31142 | 20.6619 A | 0.3642 |
| **0.70.1 + mask fix** | **0.5140 A** | **0.99174** | **0.7749 A** | **0.9128** |
| 0.78.0 | 19.5562 A | 0.31453 | 19.1921 A | 0.3597 |
| **0.78.0 + mask fix** | **0.4677 A** | **0.99167** | **0.7070 A** | **0.9125** |
| upstream fp32, seeds 0-3 | (reference) | | 0.5974 - 0.7398 A | 0.9092 - 0.9115 |

0.78.0 with the fix lands 0.4677 A from upstream fp32 and 0.7070 A from the crystal, inside the
reference's own four-seed spread and closer to the crystal than the pin is. The porting bill for
the accuracy half of this upgrade is one op-convention fix.

`fold_maskfix.py` is a blanket wrapper, correct only where every SDPA site pre-scales its bias.
Boltz-2 does (`scale_pair_bias=True`). An OpenFold3 site constructs with `scale_pair_bias=False`
and `_bias_scale = 1.0`, and those sites are already right under the new convention and would be
broken by a blanket wrapper. The shipping fix is per-site: on ttnn >= 0.70.1, divide the bias by
`_bias_scale` at the `ttnn.transformer.scaled_dot_product_attention` call sites only, leaving the
fp32-softmax and fused-kernel paths alone. Not applied here: the pin does not move in this task.

### How it was found, and the instrument that hid it

`op_trace.py`'s first version wrapped out-of-place `ttnn.<op>` only, and reported the first call
whose output hash differs. That is call 18, the MSA pair-weighted-averaging softmax, and it is a
0.08 bf16 ULP rounding difference which a deliberate full-ULP perturbation showed costs 0.09 A.
Two things fixed the instrument:

* **Scan the L2 of each output, not the hash** (`scan_l2.py`). Hash difference finds the first
  bit that moves; relative L2 finds the first call that moves the answer. On the widened trace
  everything before call 346 is under 0.2% and call 346 is 5.7%.
* **Wrap the in-place and submodule ops** (`UNTRACED` in `op_trace.py`): `add_`, `multiply_`,
  `softmax_in_place`, `ttnn.experimental.*` and `ttnn.transformer.*`. The 600-call census is 109
  `multiply_`, 26 `add_`, 12 `minimal_matmul`, 2 `nlp_create_qkv_heads`, 2 `nlp_concat_heads` and
  2 SDPA, none of which the first wrapper could see. The culprit was in that set.

The reference discipline matters as much: every score here is against float64 built from the
operands the device actually got, with the fold's own program config and memory config. Comparing
the two builds to each other names no culprit, and it would have pointed at 0.70.1 as the broken
side when 0.70.1 is the side that is right.

### Reproducing

    perf/ttx/trace_arm.sh 0.69.0 2 600 /tmp/w69.jsonl          # widened trace, both arms
    perf/ttx/trace_arm.sh 0.70.1 2 600 /tmp/w70.jsonl
    perf/ttx/scan_l2.py /tmp/w69.jsonl /tmp/w70.jsonl          # -> call 346, SDPA, 5.7%
    perf/ttx/trace_arm.sh 0.70.1 2 370 /tmp/s70.jsonl --score 346,349
    perf/ttx/report_scored.py /tmp/s69.jsonl /tmp/s70.jsonl    # operands agree, outputs do not
    perf/ttx/trace_arm.sh 0.70.1 2 350 /tmp/d/trace.jsonl --dump-args 346 --dump-dir /tmp/d/0.70.1
    perf/ttx/sdpa_mask_convention.py /tmp/d                    # which convention each arm obeys
    TTX_FOLD_ENTRY=$PWD/perf/ttx/fold_maskfix.py perf/ttx/run_arm.sh new 2 298 /tmp/fix

One wrinkle worth knowing: 0.70.1 and 0.78.0 both dump a stack trace at interpreter teardown after
the fold has written its output, and 0.78.0 segfaults there. It is after `folds.json` and the CIF
are on disk, so it costs nothing, but a runner that checks the exit status will call a good fold a
failure.
