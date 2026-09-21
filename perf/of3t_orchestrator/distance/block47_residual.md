# Block 47's residual: not the denominator, not the pair track, and five times the mass that passes

Pass 112, orchestrator, from `of3t-rebase`'s own crop-64 arms at 0.4.3 (copies in
`perf/of3t_orchestrator/verify_trunk043/`). All figures recomputed here, A14 applied, 52 tensors
per block on both sides.

## The residual is uniform across the block, not graded by attention

| sub-module | n | block 0 | block 23 | block 47 | 23 → 47 |
|---|---|---|---|---|---|
| `attn_pair_bias` | 6 | 1.2905e-01 | 1.3415e+00 | 8.7819e-01 | **0.65x — falls** |
| `single_transition` | 5 | 2.1154e-02 | 2.7792e-02 | 3.8253e-01 | **13.8x** |
| `pair_stack` | 41 | 1.0021e-02 | 1.6658e-02 | 1.9691e-01 | **11.8x** |
| whole block (median) | 52 | 1.2136e-02 | 1.9191e-02 | 2.0129e-01 | 10.5x |

**`single_transition` has no attention and no pair coupling, and it jumps 13.8x — harder than
anything else.** The campaign's standing lead, D8's *"graded by attention and pair-track
involvement, `single_transition` the only passer"*, describes blocks 0 and 23. At block 47 that
grading **disappears**: every sub-module degrades together, and the one sub-module the grading
was built on degrades hardest. `attn_pair_bias` actually *improves* from block 23.

So whatever block 47 has is **not** a pair-track or attention mechanism.

## It is not a shrinking denominator either

The A15/D17 failure mode — a relative error inflated by a small reference — is refuted, and in the
strongest direction:

| block | median `ref_norm` | total squared norm | median `rel_l2` |
|---|---|---|---|
| 0 | 4.047e-03 | 1.298e-02 | 1.2136e-02 |
| 23 | 3.537e-03 | 7.724e-03 | 1.9191e-02 |
| 47 | **8.467e-03** | **1.086e-01** | 2.0129e-01 |

Block 47's reference gradients are **2.394x LARGER** than block 23's, not smaller, and its total
squared norm is **14x** larger. A bigger denominator should make relative error *smaller*. So
block 47's **absolute** error is roughly **25x** block 23's (2.394 × 10.489). The residual is real
and large.

## And it sits on five times the mass that passes

Per-block squared gradient norm over the 52 compared tensors, against the trunk's
0.5991 (`reach_by_norm_043.json`) and the trunk's 5.828 % share of the model:

| block | % of trunk | % of model |
|---|---|---|
| 0 | 2.17 % | 0.126 % |
| 23 | 1.29 % | 0.075 % |
| **47** | **18.13 %** | **1.057 %** |

- **Passing** (blocks 0 and 23): **0.201 %** of the model's squared norm.
- **Failing** (block 47): **1.057 %** — **5.2x the passing pair.**

**This corrects pass 111.** That estimate put blocks 0 and 23 at ~0.32 % by assuming the 48 blocks
carry equal mass. They do not — three of 48 blocks hold **21.6 %** of the trunk's gradient mass,
and the deep block holds most of it. The measured figure is **0.201 %**, and the campaign's
passing-gradient coverage is smaller than pass 111 claimed while the failing block is larger.

## What this leaves for whoever takes it

A residual that is uniform across sub-modules, grows sharply with depth, carries disproportionate
gradient mass, and is not explained by D23. The block ladder is the instrument — a fourth and
fifth rung between 23 and 47 would say whether the jump is smooth or a step, which the three
points here cannot distinguish.
