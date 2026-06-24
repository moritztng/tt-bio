# 2026-06-24 — Multi-device tensor-parallel trunk (Boltz-2 --fast): measured, op-win real, e2e dead end

**Branch:** `exp/perf-20260624-tp-trimul` (off `main` 8a40c42). **NOT merged/pushed.**
Code reverted to main after measurement; only `.perf-runs/20260624/` artifacts remain.

## Goal
Every prior journal names "multi-device / tensor-parallel (TP) trunk across the 4
cards" as the last big un-tried lever but none ever measured it. This run measures
it end-to-end on the tt-quietbox 4×Blackhole.

## Hardware go-signal (new)
The 4 Blackhole chips form a **degree-2 ring** (fabric topology auto-discovery:
"Internal Graph ... Degree histogram {2:4}"). `ttnn.open_mesh_device(MeshShape(1,4))`
works; CCL (`all_gather`/`reduce_scatter`) is available **once fabric is enabled**
(`ttnn.set_fabric_config(FabricConfig.FABRIC_1D_RING)` — without it `all_gather`
TT_FATALs "un-initialized fabric context"). `ttnn.from_torch(device=mesh)` **auto-
replicates** (no per-weight mesh_mapper needed) — so a whole-model mesh port is small.

## The shardable work
Per-op-type synced device profile (whole run, `TT_OPPROF`, `optprof/sitecustomize.py`):
- **TriangleMultiplication 31.5%**, **TriangleAttention 25.0%** (the O(L³) ops),
  AttentionPairBias 18.3%, Transition 11.8%, PairWeightedAveraging 7.5%, OuterProductMean 5.7%.
So 56.5% of trunk device-compute is the two O(L³) triangle ops — the TP target.

Trimul's `for i in range(n_pairs)` loop chunks over the **hidden channel dim**
(C=32, n_pairs=4 with hidden=128); each chunk's whole pipeline (proj + gate +
permute + O(L³) matmul) is independent, concatenated only at the end → a perfect
fit for a 1×4 mesh: **one channel chunk per chip, one all_gather of the hidden**.

## Op-level result — REAL and bit-identical (`mesh_trimul.py`)
Validated full-TriangleMultiplication channel-TP vs the real single-device op:

| L | single | mesh+all_gather (ring) | speedup | PCC / maxdiff |
|---|--------|------------------------|---------|---------------|
| 256 | 2.62 ms | 1.82 ms | **1.44×** | 1.0 / 0.0 |
| 512 | 9.16 ms | 5.23 ms | **1.75×** | 1.0 / 0.0 |
| 686 | 39.97 ms | 16.55 ms | **2.42×** | 1.0 / 0.0 |

Bit-identical (maxdiff=0) — sharding splits the identical kernel stream. Scales
with L (compute ÷4 grows O(L³); the all_gather grows O(L²)). The **isolated cubic
matmul alone** does NOT win (`tp_trimul_spike.py`: 4× compute but the all_gather of
the pair tensor ≈ the matmul cost → 0.62–1.06×, no crossover). The full-op wins
because ~85% of a trimul call is *other* per-chunk work that also shards, against
just one gather.

## Integration — measured REGRESS at every size → reverted
Wired an env-gated (`TT_BIO_TP=1`) whole-trunk-on-mesh path: `get_device()` opens a
1×N ring mesh, `_to_torch` takes the replicated device-0 copy, TriangleMultiplication
shards its in-weights (`ShardTensorToMesh`) + `all_gather`s the hidden, everything
else auto-replicates (lossless by construction). One mesh worker via `_local_workers`.

| | single-device | TP 4-card | Δ e2e |
|---|---|---|---|
| **L=256** trunk / diff / e2e | 8.98 / 7.47 / **16.84s** | 9.25 / 15.16 / **24.91s** | **+48%** |
| **L=512** e2e (warm) | 35.71s | **48.6s** | **+36%** |

Root causes (both fundamental to the *naive whole-model* port, not bugs):
1. **Diffusion doubles** (7.47→15.16s @256). The 200-step host sampling loop does a
   device→host→device round-trip per step; on a mesh `_to_torch` now `ConcatMeshTo
   Tensor`-gathers 4 replicated copies and `from_torch` re-replicates to 4 → host
   interop, which already bounds diffusion, inflates ~2×. Diffusion gains nothing
   from TP (it's replicated) yet pays all the mesh tax.
2. **Even the trunk regresses at L=256** (+3%): the per-call `all_gather` (192+
   trimul calls) + the 44–69% of trunk that is replicated non-trimul ops paying
   mesh dispatch overhead exceed the modest 1.44× trimul gain. Only at L≥512 (1.75–
   2.42× trimul) could the trunk net-positive — but e2e still regresses via (1).
3. **L=512 L1 clash**: the L1-resident trimul config (≤640 fast) clashes with the
   all_gather L1 buffers ("static circular buffers clash with L1 buffers"); worked
   around by forcing the TP trimul path to DRAM (each chip holds ¼ the channels so
   L1-residency is unnecessary) — runs, but does not change the e2e verdict.

## Verdict / recommendation
**Not merge-worthy.** The op-level TP win is real, bit-identical and L-scaling, but a
naive whole-model mesh port regresses e2e at every size. An e2e win would require
**trunk-only mesh execution** (run only the trunk replicated-on-mesh, migrate s/z to
a single device before diffusion so the 200-step host loop stays single-device) AND
amortizing the per-trimul all_gather (e.g. keep z channel-sharded across consecutive
ops / batch gathers). That is a substantially larger, higher-risk change than can be
validated across all inputs in one night. Code reverted; spike scripts + this report
preserved so the path is resumable.

## Artifacts (kept on branch)
- `tp_trimul_spike.py` — isolated cubic-matmul TP (no crossover; comms-bound).
- `mesh_trimul.py` — **validated bit-identical full-trimul TP, 1.44–2.42×**.
- `trimul_realcost.py` — real single-device trimul call cost (8.85 ms @512).
- `~/.tt-bio-perf/optprof/sitecustomize.py` — per-op-type device profiler (`TT_OPPROF=1`).
