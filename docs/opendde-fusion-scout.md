# OpenDDE missed-fusion audit (full forward path)

## Result

No production change is justified. OpenDDE's whole `predict` forward path was
audited op-by-op for missed fusions (direct ttnn fused ops and theoretical
single-kernel collapses), then profiled end-to-end on a real production fold.
Every OpenDDE-specific surface is either host/upload-bound (no device-kernel
fusion applies) or a shared Protenix-v2 primitive already proven at its
dispatch/compute ceiling by a prior closed scout. The maximum possible win from
making **all** OpenDDE-specific novel compute free is **1.031×** wall-clock, and
no device-kernel fusion captures any meaningful fraction of that.

Runtime code is unchanged. No release gate was needed.

This is the documented ceiling for the full forward path; the earlier
`docs/opendde-kernel-scout.md` covered only `StructuralTokenExpander`.

## What is and isn't OpenDDE-specific

OpenDDE reuses the Protenix-v2 trunk / diffusion / confidence stack verbatim
(`docs/opendde-port.md`). The shared primitives are closed and were **not**
re-measured (per the standing rule — re-measuring a byte-identical closed
primitive wastes a turn):

| Component | Where | Verdict | Source |
|---|---|---|---|
| Pairformer trunk (TriangleMultiplication, TriangleAttention, OuterProductMean, Transition) | residue trunk, 10 cycles | closed | `docs/trimul-*`, `docs/boltz2-protenix-kernel-scout.md` |
| token DiT AttentionPairBias | diffusion, 24 blocks × 200 steps | closed (QKV packed, pair-bias precomputed, gate-pack regresses + breaks parity) | `docs/boltz2-dit-attention-kernel-scout.md` |
| AdaLN conditioning + `addcmul` | diffusion DiT | closed (bit-identical `addcmul`, 0.999× — compute-bound) | `docs/protenix-conditioning-scout.md` |
| AtomAttention encoder/decoder | diffusion, 3+3 blocks × 200 steps | closed (already ships packed-KV; KV-pack regresses 0.928×) | `docs/atomattention-kernel-scout.md` |

The OpenDDE-specific novel compute is exactly:

1. `StructuralTokenExpander` — the residue→structural-token expander (closed in
   `docs/opendde-kernel-scout.md`: host/upload-bound, 2.21% of a production
   fold, device matmul 0.6% of the block).
2. The 4-block `Pairformer` refiner at the expander seam — a reused Protenix
   Pairformer (closed above), on the structural-token axis.
3. The structural-token **conditioning glue** — the only previously-unprofiled
   surface. Runs **once per fold**, before the sampler loop:
   - `relp_struct` — `_generate_relp` on the structural-token axis (host
     `index_select` over `parent_residue_idx`).
   - `_diffusion_pair_cond(z_st, relp_struct)` — the shared conditioning method;
     OpenDDE's only addition is the `linear_no_bias_z_trunk` branch that
     LN+projects `z_trunk` 384→128 before concat (gated on key presence).
   - `_plm_z_term(pair_z, a2s, …)` — host torch `layer_norm` + `linear` +
     windowed `index_select` gather that scatters the structural-token pair onto
     the atom-pair axis (replaces Protenix's residue `a2t`).
   - `_dit_block_biases(dit_z, structural_pair_attn_bias)` — injects the
     expander's additive pair-attention bias into the 24 DiT block biases: one
     upload of `(1,1,Ns,Ns)` scaled by `sqrt(head_dim)`, then 24 `ttnn.add`s.
     Cached in `cond["dit_block_biases"]` and replayed every step, so this is a
     per-fold cost, not per-step.

## Measured production profile (qb2, card 2)

Real `opendde.pt` weights, 7ROA (117 res → Ns=229 structural tokens, 900 atoms),
bf16, HiFi4 / fp32 dest-acc, 10 cycles / 200 steps / 1 sample, warm program
cache, device-synchronized. Reproduce:

```bash
TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=<env>/.../p150_mesh_graph_descriptor.textproto \
    PYTHONPATH=<worktree> /home/ttuser/tt-bio-dev/env/bin/python3 \
    scripts/opendde_fusion_scout.py
```

| phase | time | share of fold |
|---|---:|---:|
| trunk (residue, shared, closed) | 7.712 s | 65.7% |
| diffusion sampler (shared DiT on Ns axis, closed) | 3.408 s | 29.0% |
| `expand_and_refine` (expander + refiner seam) | 0.461 s | 3.9% |
|   of which expander only (closed, host-bound) | 0.288 s | 2.46% |
|   of which refiner seam (shared Pairformer, closed) | 0.172 s | 1.47% |
| **OpenDDE conditioning glue (this pass)** | **0.068 s** | **0.58%** |
|   of which `_diffusion_pair_cond` (incl. z_trunk compress branch) | 0.039 s | 0.33% |
|   of which `relp_struct` + residue `relp` (host, 2 calls) | 0.016 s | 0.13% |
|   of which `_plm_z_term` (host scatter) | 0.007 s | 0.06% |
|   of which `_dit_block_biases` (24-add bias injection) | 0.007 s | 0.06% |
| residual gap (input embedder + `S_struct` + reshapes) | 0.083 s | 0.71% |
| **total fold** | **11.732 s** | |

The on-device ttnn op count issued **inside** the OpenDDE-specific glue (counted
only during `_diffusion_pair_cond` + `_dit_block_biases`, both once per fold) is
117 `linear` / 56 `layer_norm` / 26 `add` / 3 `concat` — i.e. a few hundred
dispatches **per fold**, none per step. For context the shared DiT issues ~960
ops per *step* × 200 steps.

## The fusion question

Amdahl ceiling on **all** OpenDDE-specific novel compute (expander + refiner
seam + conditioning glue) = `1 / (1 - 0.0440) = 1.031×`. The expander is
host/upload-bound (closed). The refiner is a shared Pairformer (closed). The
conditioning glue is 0.58% of the fold, runs once, and its only on-device
additions are:

- **`z_trunk` 384→128 compress**: `layer_norm(z_trunk) → linear`. No fused
  ttnn `layer_norm+linear` exists, and the slice is 0.039 s per fold — fusing
  its epilogue would save sub-millisecond.
- **`structural_pair_attn_bias` injection**: 24 × `ttnn.add(compute_bias(z_dev), extra)`.
  The `extra` bias is per-position `(1,1,Ns,Ns)` broadcast over heads; a linear
  bias operand is 1D over the output channel dim, so it cannot fold into
  `compute_bias`'s internal linear (the same per-position-bias reason that
  closed the expander's `pair_init_bias` in `docs/opendde-kernel-scout.md`).
  24 adds once per fold = 0.007 s.
- **`_plm_z_term`**: pure host torch gather. No device op to fuse.

No direct ttnn fusion applies to any of these, and no theoretical
single-kernel collapse exists for a once-per-fold host/upload + additive-bias
sequence. The shared 94.7% (trunk + diffusion) is closed.

## Resolution

OpenDDE's forward path is at its dispatch/compute ceiling. The OpenDDE-specific
novel compute is 4.4% of a production fold with a 1.031× Amdahl ceiling, is
host/upload-bound or shared-closed, and contains no device-kernel fusion
candidate. The diffusion (29.0%) runs on the structural-token axis (Ns=229 vs
117 residue tokens) but is the byte-identical shared DiT / atom-encoder /
AdaLN-conditioning block already proven device-compute-bound. No fusion is
prototyped; no code changes. This is a documented ceiling, not a missed lever.
