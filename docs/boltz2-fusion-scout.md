# Boltz-2 missed-fusion audit

## Result

No production change is justified. Boltz-2's ttnn dispatch graph was audited
op-by-op for missed fusions (direct ttnn fused ops and theoretical
single-kernel collapses). Every major hot component is either already shipping
its fused form, already proven dispatch-bound with no headroom by a prior
scout, or — for the one previously-unprofiled primitive (the MSA
`PairWeightedAveraging` per-head loop) — measured here to have a trace-replay
ceiling of **1.000×**. There is no real op-fusion / dispatch-collapse win to
capture on the current fleet ttnn build.

Runtime code is unchanged. No release gate was needed.

## Method

Hardware: qb2 physical card 1, one Blackhole P150. Real Boltz-2 checkpoint
weights (`~/.boltz/boltz2_conf.ckpt`). `TT_VISIBLE_DEVICES=1`,
`PYTHONPATH=<worktree>`, `~/tt-bio-dev/env/bin/python` (has omegaconf for the
Boltz-2 lightning checkpoint + system ttnn), and
`TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto` (P300
detection). Every timed number is the median of 3 warm same-shape calls in one
process, bracketed by a device synchronize. Parity is bit-exact PCC / max-abs
against the untimed baseline.

The dispatch-collapse ceiling for each candidate is **ttnn trace replay**:
capturing the whole primitive's op stream into one trace and replaying it
removes all host enqueue / dispatch overhead. If trace replay is no faster than
eager, the op is device-bound (compute or bandwidth), not dispatch-bound, and
no fusion that only collapses dispatches can help. This is the same floor used
by `docs/permodel-kernel-scout.md` and `docs/difftransformer-swiglu-scout.md`.

## Component-by-component

Boltz-2's device graph (`tt_bio/tenstorrent.py`) is the AF3 building set. The
table covers every block that runs on a `predict`, with the fusion verdict and
where it was settled.

| Component | Where | Prior fusion verdict | Source |
|---|---|---|---|
| TriangleMultiplication | trunk pair stack | channel-move decomposition already in main; whole-op megakernel refuted (L1 overflow / 3× worse where it fits) | `docs/trimul-megakernel-retry.md`, `docs/trimul-largeseq-ceiling.md` |
| TriangleAttention | trunk pair stack | 26.5-29.6% of the trunk cycle but no avoidable move; permute→transpose 1.000×; QKV+gate pack 0.58-0.61× | `docs/boltz2-protenix-kernel-scout.md` |
| OuterProductMean | MSA module | transpose decomp 1.07× isolated, ≤1.0016× of trunk, not bit-exact at N=1024 | `docs/boltz2-protenix-kernel-scout.md` |
| Transition (pair) | trunk + MSA | SwiGLU (`linear silu` × `linear` → mul → `linear`). The ESMFold2 `fuse_swiglu` win does not apply: that epilogue kernel is absent from the fleet ttnn build (see below) | `docs/kernel-scout-next.md`, `docs/difftransformer-swiglu-scout.md` |
| AttentionPairBias (trunk + DiT) | trunk s-track, DiT attention | QKV already one packed linear; pair-bias precomputed once and replayed as an SDPA mask; gate-pack regresses 0.97× and breaks parity (5.2 Å @200 steps) | `docs/boltz2-dit-attention-kernel-scout.md`, `docs/atomattention-kernel-scout.md` |
| DiffusionTransformer attention half | diffusion (24 blocks × 200 steps) | dispatch-bound, 17 launches/call, no headroom; QKV/gate packing regresses + breaks parity | `docs/boltz2-dit-attention-kernel-scout.md` |
| ConditionedTransitionBlock (DiT FFN) | diffusion (24+3+3 blocks × 200 steps) | SwiGLU `swish`/`gates` pack nets ~1.6% with an instability spike; `fuse_swiglu` kernel absent from fleet build; linear-width (ESMC-like) so the wide-intermediate DRAM win does not apply | `docs/difftransformer-swiglu-scout.md` |
| AtomAttention encoder/decoder | diffusion (3+3 blocks × 200 steps) | Boltz-2 already ships the fused packed-kv + single `keys_indexing` matmul gather form; KV-pack regresses 0.928× | `docs/atomattention-kernel-scout.md` |
| AdaLN modulation | DiT attention + FFN | sigmoid-fused multiply already uses `input_tensor_b_activations=[SIGMOID]`; the two `s`-projections pack into a wider matmul that regresses (same family as DiT QKV-pack) | this pass, by inspection |
| PairWeightedAveraging (MSA) | trunk MSA module (4 blocks) | **profiled here, see below** — trace-replay ceiling 1.000× | this pass |

## The one previously-unprofiled primitive: PairWeightedAveraging

`PairWeightedAveraging` (`tt_bio/tenstorrent.py`) is the AF3 MSA weighted
average. It runs a **Python `for i in range(n_heads)` loop** with sliced
per-head weights: per head it issues `linear(z, z_weight[:, i:i+1])`,
`linear(m, m_weight[:, i*hd:(i+1)*hd])`, `softmax`, `matmul(v, w^T)`,
`linear(m, g_weight[...])`, sigmoid-gated `multiply`, `linear(o_weight[...])`,
and an accumulator `add`. At Boltz-2's config (8 heads, 4 MSA blocks) that is
~320 ttnn launches per trunk recycling iteration from one Python loop — the
textbook "dispatch pile" shape the Accelerate workstream targets.

It had never been profiled (the `pwa` mode in
`scripts/boltz2_protenix_kernel_scout.py` existed but was never run or
documented). This pass ran it with real Boltz-2 weights and Boltz-2 shapes.

Boltz-2 PWA config (from `~/.boltz/boltz2_conf.ckpt`):
4 blocks, `n_heads=8`, `head_dim=32`, `c_m=64`, `c_z=128`. Boltz-2 pads the MSA
dim to `MSA_PAD_MULTIPLE=1024` even for a single-sequence (no-MSA) predict, so
the contraction dim is 1024, not 1.

Trace-replay ceiling (the dispatch-collapse floor), real Boltz-2 weights,
MSA=1024, bit-exact parity on every record:

| N (padded seq) | eager baseline | trace replay | trace speedup | parity PCC / max-abs |
|---:|---:|---:|---:|---:|
| 128 | 0.02253 s | 0.02253 s | **1.0000×** | 1.0 / 0.0 |
| 256 | 0.04709 s | 0.04709 s | **1.0001×** | 1.0 / 0.0 |
| 512 | 0.12522 s | 0.12507 s | **1.0012×** | 1.0 / 0.0 |

Reproduce:

```bash
WT=<worktree>
env TT_VISIBLE_DEVICES=1 \
    TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    PYTHONPATH=$WT \
    ~/tt-bio-dev/env/bin/python $WT/scripts/boltz2_pwa_fusion_scout.py \
        --sizes 128 256 512 --msa-depth 1024 --repeats 3 \
        --checkpoint /home/ttuser/.boltz/boltz2_conf.ckpt
```

### Why no headroom

Trace replay collapses the entire 4-block PWA (Python loop and all ~320
launches) into a single captured device program and replays it with zero host
enqueue. The measured ceiling is 1.000× at every size, so the device is busy
for the full eager wall-clock — host dispatch fully overlaps device compute.
The per-head Python loop *looks* like a dispatch pile but is not
dispatch-bound.

Per-op synchronized accounting on the same op (Protenix-v2 weights, same
`PairWeightedAveraging` class, `scripts/boltz2_protenix_kernel_scout.py pwa`,
documented here for the breakdown) shows the cost is the matmuls: `linear` is
69% of op-sync time at N=512 (96 per-head sliced-weight linears across the
blocks), `matmul` and `permute` most of the rest. Each per-head `linear` has a
1-4-tile contraction (`c_m=64`, `c_z=128`) and a 1-tile-per-head output, i.e.
the linears are already at the bandwidth floor for reading their input;
collapsing the 8 per-head linears into one full-width matmul (the residual
theoretical lever) keeps total compute and input traffic constant and only
changes tile efficiency — the same packed-matmul family that regressed for the
DiT attention (0.58-0.61×) and was marginal-with-instability for the
ConditionedTransitionBlock (~1.6%, `docs/difftransformer-swiglu-scout.md`).
Not pursued: the 1.000× trace floor says the device is the bottleneck, so a
packed rewrite can at best shave tile overhead on already-tiny matmuls.

### Share of a real predict

PWA runs once per trunk recycling iteration (default recycles), inside the MSA
module's 4 blocks, alongside `msa_transition`, `outer_product_mean`, and a full
`PairformerLayer`. At `prot.yaml` (117 residues) the trunk is the minority of
wall-clock (diffusion dominates at small L per `docs/boltz2-throughput-loop` /
`docs/boltz2-fast-perf-2026-06`); at large L the trunk grows but PWA is one
sub-component of one module of it. The 1.000× trace floor holds at N=512
(large-L trunk-regime), so even the regime where PWA is largest offers no
dispatch-collapse win.

## The `fuse_swiglu` build gap

Two separate scouts (this one and `docs/difftransformer-swiglu-scout.md`)
confirm the ESMFold2 pair-transition win's epilogue kernel
(`ttnn.experimental.minimal_matmul(..., fuse_swiglu=True)`,
`docs/kernel-scout-next.md`) is **not in the qb2 fleet ttnn build**: the kwarg
raises `TypeError`, and `minimal_matmul.__doc__` exposes only
`bias_tensor / fused_activation / config / memory_config / dtype /
compute_kernel_config`. Boltz-2's `Transition` (pair) and
`ConditionedTransitionBlock` (DiT) both contain SwiGLU motifs that this kernel
would fuse. On a future ttnn build that ships `fuse_swiglu`, the pair
`Transition` (quadratic, O(L²) intermediate — the ESMFold2-winnable shape, unlike
the linear-width DiT CTB) becomes the candidate worth re-measuring; until then
it is inert and this audit records the gap rather than guessing at a number.

## Verdict

Boltz-2 has no real op-fusion / dispatch-collapse win available on the current
fleet ttnn build. The shared Pairformer/triangle primitives were already
exhausted by the sibling-model scouts; the DiT attention and FFN halves are
dispatch-bound with no headroom (packing regresses and/or breaks parity); and
the one previously-unprofiled primitive, the MSA `PairWeightedAveraging`
per-head loop, has a measured trace-replay ceiling of 1.000× with real
Boltz-2 weights. The only build-gated future lever is `fuse_swiglu` for the
quadratic pair `Transition`, which needs a ttnn build bump (release-gated) and
is recorded here, not shipped.
