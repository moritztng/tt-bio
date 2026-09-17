# Gradient-based hallucination on Protenix v2: feasibility, cost, and what to ask the customer

Decision memo, 2026-09-17. Feasibility study only — nothing here is implemented or merged, and
no model code changed. Full evidence and the per-op inventory are in
`~/.coworker/state/hall-grad-feasibility.md`; this is the short version.

## Answer in three lines

It is feasible. The kernels are already at our pin, the binding constraint is activation memory
at 800 aa, and the recommended path costs about 40 engineer-days. The reason not to start
building it this week is that a forward-only route runs today and is only 1.36-1.38x slower.

## The kernels are not the blocker

At ttnn 0.68.0 all 14 `moreh_*_backward` ops are present, plus 103 eltwise `*_bw` ops, and
between them they cover every op class Protenix dispatches. Signatures are captured verbatim in
`op_inventory_out.txt` — run `op_inventory.py` to regenerate. Two things worth knowing:

- **Do not build the backward on `moreh`.** `grep -rn -i sharded ttnn/cpp/ttnn/operations/moreh/`
  returns zero hits. The whole family is interleaved-only, so it forfeits every L1-residency
  lever tt-bio ships. `moreh_matmul_backward` is anyway just two `moreh_matmul` calls plus a
  broadcast sum. A matmul's backward is a matmul whatever kernel ran the forward, so use the
  production matmul path.
- **One item needs a hand-written kernel:** the triangle-attention backward. Protenix runs 100%
  on the fused SDPA (`fp32_softmax_stats.calls` is 0 in the 512 aa census), and that kernel never
  materialises its score tensor. The backward has to recompute scores in row chunks, flash-style.
  The forward's own row blocking gives the loop structure.

The prior that our fused kernels are the problem does not survive measurement. Unfusing the
trimul tail costs 0.335 s of a 32.3 s ESMFold2 512 aa fold (32.329 -> 31.994, median of 3,
`perf/trimul_f1/page_esmfold2_f1_qb2c2.json`) — 1.0105x. Cheap to give up.

## Memory is what binds

At 800 aa with c_z=256 in bf16, the triangle-attention score tensor `[N, 8, N, N]` is **8.19 GB**,
24% of a Blackhole chip by itself. One pairformer block's minimum retained set is 11.82 GB, so
the 48-block stack unchecked is 567 GB. Checkpointing is mandatory and the granularity has to be
at most one block. With per-block checkpointing: 15.76 GB of tape plus 11.82 GB live = 27.58 GB.

| | Blackhole (34.23 GB/chip) | Wormhole Galaxy (12 GiB/chip) |
|---|---|---|
| 512 aa | fits, 22.9 GB margin | fits, under 1 GB spare |
| 800 aa | fits, 6.65 GB margin | **does not fit** |

DRAM figures come from the soc descriptors (`blackhole_140_arch.yaml:112`,
`wormhole_b0_80_arch.yaml:135`), not from recall. Gradient hallucination at 800 aa is
Blackhole-only.

## What it costs to run

Anchored on `perf/pxv1/v2_512_r{10,2}_pc0_a.json`: same tt-bio, ttnn 0.68.0, pc card 0 p150a,
**aiclk 1350 MHz recorded in the artifact**, only recycles differ. 50.253 s at r10 and 20.387 s
at r2 give 3.733 s per trunk recycle and a 12.921 s residual, and the split refits the r2 point
to the millisecond.

Per gradient step at 512 aa / 3 recycles, backward multiplier 2.293x (input grads only, so a
weight matmul's backward is 1x forward; the trimul einsum and the attention products need both
operands and cost 2x; per-block recompute adds the 1x):

| subgraph | s/step @512 | s/step @800 | 410 steps @512 |
|---|---|---|---|
| confidence head only | 0.71 | 1.71 | — answers the wrong question |
| **distogram -> last recycle** | **19.76** | **46.17** | **2.25 h** |
| + one denoising step | 32.83 | 76.16 | 3.74 h |
| full 200-step trajectory | 115.21 | 275.77 | 13.12 h, and a NO-GO |

The confidence head alone gives `d(loss)/d(z_trunk, s_trunk, coords)`. The sequence enters 48
blocks upstream, so nothing there can update a sequence. It should not be sold as gradient
hallucination.

The full trajectory is a NO-GO on numerics, not on cost: 200 chained bf16 backward steps is the
failure mode our own AF2-IG work root-caused, where chained error accumulates coherently per op
instead of cancelling.

The cheapest real gradient is a distogram loss backpropagated through the last trunk recycle,
with stop-gradient on the earlier nine — what AlphaFold does in training and what ColabDesign
does. It needs no diffusion at all. Protenix v2's `distogram_head` is one `(64, 256)` linear
already sitting in the checkpoint, and tt-bio does not currently implement it.

## The tape

No autograd engine exists, and `moreh_*_backward` are raw kernels, so something has to hold the
tape. Use **torch.autograd, one `autograd.Function` per pairformer block, ttnn tensors resident
on device** so autograd holds only handles. Protenix's forward is a static Python call sequence
with shapes fixed per fold by the token-axis bucketing, which is the easy case.

This also retires the worry that transfers would dominate: with the tape resident there are no
per-step transfers, and even staging block boundaries to host is 15.76 GB each way at a realistic
25 GB/s over PCIe Gen5 x16, about 1.3 s against a 20-46 s step. Cost 30-52 engineer-days. A
hand-written tape and reverse pass adds 5-8 days to rebuild what torch ships and leaves us
owning it: 38-60 days. Both write the same per-module maths, so the cheaper tape wins outright.

## Effort

**40 engineer-days, +/-12.** Triangle-attention backward 3-6 d (the one escalation risk: +5-10 d
if it needs a new tt-metal kernel). Nine per-module backwards with float64 gradchecks 13.5-22.5 d.
autograd wiring and checkpointing harness 3-5 d. Sequence-logit interface 2-4 d. Objectives,
optimiser loop and CLI 4-6 d. float64 parity harness and whole-trunk gradcheck at 128 aa 4-8 d.

The float64 reference is not optional. A transform that is wrong rather than imprecise is a hard
stop, and the nightly bar on these ops is a 0.1 allclose (`test_moreh_layer_norm.py` uses
`rtol = atol = 0.1`), nowhere near what a gradient needs. `moreh_layer_norm_backward`'s
`validate_inputs` is an empty function body, so a bad dtype is not rejected at the API either.
It also needs `mean` and `rstd`, which stock `ttnn.layer_norm` does not return, and it does not
support bfloat8_b ("bfloat8_b is not supported in the kernel", `test_moreh_layer_norm.py:476`),
so the `--fast` bf8 trunk is off-limits on the norm path.

## Why I am not recommending we start building

ColabDesign's own source defaults
(`raw.githubusercontent.com/sokrypton/ColabDesign/main/colabdesign/af/design.py`): the gradient
route `design_3stage` is `soft_iters=300, temp_iters=100, hard_iters=10` = **410 gradient steps**.
The forward-only routes are `_design_mcmc steps=1000` and `design_semigreedy iters=100, tries=10`
= **1000 forward passes**. That is **2.44 forward passes per gradient step**.

In our measured wall-clock at 512 aa / 3 recycles with a distogram objective:

- MCMC, 1000 x 11.20 s = **3.11 h/design**
- gradient, 410 x 19.76 s = **2.25 h/design**
- the gradient buys **1.38x** (1.36x at 800 aa)

Forty engineer-days cannot be justified on 1.4x. Two qualifiers in the other direction, both
real. The zeroth-order loop gets no batch speedup — `fold_many`'s docstring records the trunk at
0.93x (pair transition) and 1.07x (tri-attention) per prediction at B=8, so 1000 forwards is 1000
serial trunk runs and parallelism has to come from more chips, which is the thing TT scales well.
And gradient and MCMC hallucination do not converge to the same designs, so "use MCMC instead" is
only an answer if it converges on the customer's objective. That is theirs to judge, not ours.

## Recommended path, soonest-first

1. **Ship the forward-only hallucination loop on Protenix now.** Days, not months. 1000 forwards
   at 11.20 s is 3.11 h/design at 512 aa. This is the field's own default algorithm at its own
   default step count, not a downgrade we invented.
2. **Add the distogram head** — one `(64, 256)` linear. Both routes want the cheap trunk-only
   objective, so it pays for itself either way.
3. **Send the questions below, and build the gradient path only if the answer to Q3 is quality.**

Moritz's stated position to the customer ("currently forward only, so MCMC-style hallucination
should run today, but not yet gradient based") is accurate and needs no walking back.

## Questions for Siddhant, Samarth and David

1. Which **objective**? A distogram or contact loss is trunk-only and needs no diffusion — the
   cheapest real gradient we can give you. A pLDDT/pTM/coords objective needs the diffusion and
   costs ~1.7x per step.
2. Which **subgraph**? Last recycle only with stop-gradient on the rest, as AlphaFold training and
   ColabDesign both do? Or all 10? We are calling the full 200-step trajectory a NO-GO on chained
   bf16 numerics and would like to know whether that matches your experience.
3. Do you need the gradient for **design quality or for throughput**? At ColabDesign's own default
   step counts, MCMC/semigreedy is only 1.36-1.38x slower here, not 50x. If it is throughput, the
   forward-only path ships in days. If it is quality on your objective, that is a real reason and
   the one we would build against.
4. What **target length**? 800 aa is Blackhole-only; a 12 GiB Wormhole Galaxy chip cannot hold the
   15.76 GB tape. At or below 512 aa there is far more room.
5. What **convergence** bar — is 410 gradient steps enough on Protenix, or do you need more?
6. Is **bf16** gradient precision acceptable, or do you need fp32 accumulation on the backward?
   That roughly doubles activation memory and moves the estimate.
7. How is the **sequence parametrised** — soft one-hot / PSSM logits, or straight-through discrete?

## Reproducing

```
python3 perf/hall_grad/cost_model.py --n 512 --cycles 3
python3 perf/hall_grad/cost_model.py --n 800 --cycles 3
/home/moritz/tt-bio/env/bin/python perf/hall_grad/op_inventory.py
```

`cost_model.py` opens no device and hardcodes no architecture constant that is not read from the
checkpoint. Note that the `perf/px4pd` size-ladder artifacts (256/640/768 aa) carry no `aiclk`
field, so they cannot anchor a perf claim on their own; the 512 aa pair used for every timing
above does record 1350 MHz. The 800 aa fold time itself belongs to row `hall-capacity-800aa` —
the two-point exponents here disagree (1.86 on 256->640, 1.46 on 640->768), so it needs measuring
rather than extrapolating.
