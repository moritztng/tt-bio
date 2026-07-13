# ESMFold2 missed-fusion audit (full forward path)

## Result

No production change is justified. ESMFold2's whole `predict` forward path was
audited op-by-op for missed fusions (direct ttnn fused ops and theoretical
single-kernel collapses), then profiled end-to-end on a real production fold on
qb2. Every ESMFold2-specific surface is either a shared primitive already proven
at its dispatch/compute ceiling by a prior closed scout, or a dispatch-bound pile
whose existing fused ttnn op cannot apply (a hard kernel constraint, not a
missing call site). The maximum possible win from making **all** ESMFold2-specific
novel compute free is **~1.06×** wall-clock (best case, short proteins; ~1.04× at
production length), and no device-kernel fusion captures any meaningful fraction
of it.

Runtime code is unchanged. No release gate was needed.

This is the documented full-forward-path audit; the earlier
`docs/esmfold2-bf8-trunk-scout.md` covered only bf8 trunk-weight precision (a
different lever, not a fusion audit).

## What is and isn't ESMFold2-specific

ESMFold2 reuses tt-bio's shared primitives. The closed ones were **not**
re-measured (re-measuring a byte-identical closed primitive wastes a turn):

| Component | Where | Verdict | Source |
|---|---|---|---|
| Folding-trunk `TriangleMultiplication` (tri_out + tri_in) | 48 PairUpdateBlocks × 3 trunk loops | closed (DRAM-bandwidth-bound on the pair tensor; weights tiny vs `[N,N,128]` pair tensor) | `docs/esmfold2-bf8-trunk-scout.md`, `docs/boltz2-protenix-kernel-scout.md` |
| Pair-transition `SwiGLUFFN` | 48 trunk blocks + diffusion conditioning | closed | `docs/esmc-swiglu-fusion-scout.md` |
| Trunk residual `add`s (3 per block) | 48 × 3 loops | closed ("making every add free stays below 1.1%") | `docs/boltz2-fusion-scout.md` |
| Token DiT `AttentionPairBias` + `ConditionedTransitionBlock` | diffusion, 12 blocks × ~14 steps | closed (QKV packed, pair-bias precomputed/cached) | `docs/boltz2-dit-attention-kernel-scout.md` |
| AdaLN conditioning (`addcmul` modulation) | token DiT + SWA atom blocks | closed (bit-identical `addcmul`, 0.999× — compute-bound) | `docs/protenix-conditioning-scout.md` |
| ESMC-6B LM + rotary embedding | input embedder | closed (fused `rotary_embedding` shipped behind a tile/head-dim gate) | `docs/esmc-attention-kernel-scout.md` |

The ESMFold2-specific novel compute is exactly:

1. **`SWAAtomTransformer`** — the token-free sliding-window atom transformer with
   3D-RoPE (3 encoder + 3 decoder blocks × ~14 diffusion steps). This is NOT the
   Protenix `AtomAttention` (full attention, closed in
   `docs/atomattention-kernel-scout.md`); ESMFold2's is windowed (window 128)
   with 3D spatial+uid RoPE, `head_dim=32`, `n_heads=4`, `d_atom=128`.
2. **Atom↔token scatter/gather aggregation** — the `scatter_m @ q_to_a` (encoder)
   and `gather_g @ a_to_q` (decoder) matmuls that fold per-atom features into
   per-token repr and back. One matmul each per diffusion step.
3. **3D-RoPE table + `band_mask` construction** — host-side torch, per-fold (not
   per-step), negligible.

## Measured production profile (qb2, card 0)

Real `biohub/ESMFold2` weights, `examples/prot.yaml` sequence (L=117, ~900 atoms
— the 7ROA-gate size), bf16, 3 trunk loops / 20 diffusion steps / 1 sample, warm
program cache, device-synchronized. Reproduce:

```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=<env>/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
PYTHONPATH=<worktree> /home/ttuser/tt-bio-dev/env/bin/python3 \
    scripts/esmfold2_op_profile.py --protein prot --loops 3 --steps 20
# SWA sub-op breakdown:
    scripts/esmfold2_swa_profile.py --protein prot --loops 3 --steps 20
```

### Forward-path breakdown (L=117)

| phase | time | share of fold |
|---|---:|---:|
| ESMC-6B LM + inputs embedder (shared, closed) | ~0.50 s | ~23% |
| folding trunk — 48 PairUpdateBlocks (shared trimul+SwiGLU, closed) | ~1.40 s | ~65% |
| **diffusion sampling (structure head)** | **0.219 s** | **10.2%** |
|   token DiT — 12 blocks (shared, closed) | 0.099 s | 4.6% |
|   atom encoder — SWA, 3 blocks (**ESMFold2-specific**) | 0.045 s | 2.1% |
|   atom decoder — SWA, 3 blocks (**ESMFold2-specific**) | 0.040 s | 1.9% |
|   conditioning `cond_single` (shared SwiGLU, closed) | 0.005 s | 0.23% |
|   conditioning `cond_pair` (cached, step-invariant) | 0.002 s | 0.10% |
| confidence head | 0.067 s | 3.1% |
| **total** | **2.15 s** | |

### SWA atom transformer sub-op breakdown (the only ESMFold2-specific surface)

Aggregated over 87 SWA block calls (6 blocks × ~14 steps), device-synchronized:

| sub-op | time | share of fold | dispatches/call |
|---|---:|---:|---:|
| SDPA (sliding-window, **already fused**) | 0.042 s | 1.93% | 1 |
| **RoPE (`apply_rotary` ×2, 6-op rotate-half pile, unfused)** | **0.033 s** | **1.52%** | 12 eltwise |
| projections (qkv / gate / out matmuls, compute-bound) | ~0.020 s | ~0.9% | 3 matmul |
| `_modulate` adaLN (closed: `addcmul` 0.999×) | 0.015 s | 0.69% | 4 |
| `rms_norm` (q/k norm + block norms) | 0.009 s | 0.41% | 1 each |
| SwiGLU FFN (shared, closed) | included above | — | — |

The only dispatch-bound elementwise pile is RoPE (1.52% of fold). Every other
sub-op is either already fused (SDPA), a matmul (compute-bound), or the closed
adaLN modulation.

## Candidate audit

### 1. Fused RoPE for the SWA atom transformer — BLOCKED (hard kernel constraint)

This is the one real direct-fusion miss: ESMC ships the fused
`ttnn.experimental.rotary_embedding` kernel (one dispatch per tensor, replacing
`apply_rotary`'s six-op rotate-half pile — the single largest share of ESMC
attention, `docs/esmc-attention-kernel-scout.md`), but the SWA atom transformer
still calls the unfused `apply_rotary`. Extending that fusion to the SWA path is
the obvious candidate.

It **cannot** be applied. The fused kernel requires the input's last dimension
(head_dim) divisible by 64:

```
TT_FATAL: Input X dimension (32) must be divisible by 64 for tiling.
  rotary_embedding.cpp:27
```

ESMC uses `head_dim=128` (passes); ESMFold2's SWA atom transformer uses
`head_dim=32` (`d_atom=128 / n_heads=4`, fixed by the checkpoint). Padding
`head_dim` 32→64 changes the `rotate_half` pairing (it pairs the real 16+16
halves against the padded half), so the rotated result is numerically wrong
without a scatter/permute reshape of both the activations and the cos/sin tables
— far more invasive than the ~1.2% dispatch win it could buy, and not bit-exact.

A custom fused-RoPE kernel for `head_dim=32` is a theoretical-fusion
(kernel-work) request, not a direct ttnn fusion. The realistic ceiling is the
1.52% RoPE share minus the fused kernel's own compute — under 1.2% of fold,
below the bar set by the prior closed scouts (OpenDDE 1.031×, Boltz-2 1.01× were
both declared dead ends).

### 2. Fused `_modulate` (adaLN scale/shift into the rms_norm) — CLOSED

`SWAAtomBlock._modulate(x, scale, shift) = rms_norm(x)·(1+scale) + shift` looks
foldable into a single `ttnn.rms_norm(x, weight=1+scale, bias=shift)`. It is not:
`scale`/`shift` are **per-token** (`[B,N,d_atom]`, from the conditioning `c_l`),
while `ttnn.rms_norm`'s weight/bias are **per-channel** (`[d_atom]`). The
per-token affine cannot be folded into a per-channel norm weight. The
applicable fusion is `ttnn.addcmul(shift, rms_norm, 1+scale)` (collapsing the
`mul`+`add`), which is exactly the shared adaLN modulation already evaluated in
`docs/protenix-conditioning-scout.md`: **bit-identical, 0.999× e2e —
compute-bound, no headroom.** Not re-measured.

### 3. SDPA, projections, scatter/gather, FFN — no fusion applies

SDPA already ships the fused `ttnn.transformer.scaled_dot_product_attention`.
The qkv/gate/out projections and the atom↔token scatter/gather matmuls are
dense matmuls (compute- or DRAM-bound, not dispatch-bound). The SwiGLU FFN is the
shared closed primitive. The scatter/gather matrices are step-invariant
(resident, built once) so a segment-sum kernel would only avoid re-reading the
cached sparse matrix — a <0.5% custom-kernel lever, not a fusion.

## Amdahl ceiling

Making **all** ESMFold2-specific novel compute (SWA atom transformer +
scatter/gather + `cond_single`) free at once: `0.085 + 0.002 + 0.005 ≈ 0.092 s`
= **4.3% of fold** at L=117 → **1.045×**; at the shorter gb1 (L=56, diffusion
17% of fold) the ceiling is **~1.07×**. No single realizable fusion captures a
meaningful fraction of that (the one dispatch-bound pile is blocked by the
head_dim=64 kernel constraint; the modulation is closed at 0.999×). The fold is
dominated by the shared, closed folding trunk (65% — DRAM-bandwidth-bound on the
pair tensor) and the ESMC-6B LM (23%).

## `esmfold2-fast`

The `--model esmfold2-fast` variant is a 24-block trunk checkpoint with no MSA
encoder — same architecture, fewer trunk blocks. The forward-path surfaces are
identical (the same SWA atom transformer, the same shared trimul/SwiGLU/DiT
primitives), so the fusion audit is the same: no new surface, no new win. The
trunk share is higher relative to the (unchanged) diffusion stage, so the
ESMFold2-specific Amdahl ceiling is even lower than the 48-block model.

## Outcome

ESMFold2's full forward path is at its fusion floor on the current ttnn stack.
The single ESMFold2-specific dispatch-bound pile (SWA RoPE) is blocked from the
existing fused kernel by a `head_dim=64` constraint (SWA is `head_dim=32`); a
custom `head_dim=32` RoPE kernel is a kernel-feature request, not a fusion, and
its realistic ceiling is under 1.2% of fold. Everything else is a shared closed
primitive or an already-fused op. This is an honest dead end, matching the shape
of the Boltz-2 / Protenix-v2 / OpenDDE fusion scouts.

Artifacts: `scripts/esmfold2_op_profile.py` (forward-path + diffusion breakdown),
`scripts/esmfold2_swa_profile.py` (SWA sub-op breakdown). Both read-only w.r.t.
the model.
