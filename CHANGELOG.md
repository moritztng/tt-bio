# Changelog

All notable changes to TT-Bio are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut from a commit that has passed the on-hardware test suite (see `RELEASING.md`).

## [Unreleased]

## [0.3.0] - 2026-07-17

First release shipping **OpenDDE** antibody-antigen co-folding (`--model opendde` / `opendde-abag`, built on the Protenix-v2 stack plus a structural-token expander), the **ESMC fused-RoPE** attention kernel (an accuracy-neutral speedup for the embed path), and opt-in **diffusion trace replay** for the Boltz-2, BoltzGen, and OpenDDE CLIs plus the Protenix-v2 Python API. Also lands the standing **perf-regression** and **UX-regression** harnesses as release-gate legs, plus the per-card performance baseline fix.

OpenDDE's antibody-antigen accuracy is weak on `9dsg`, a confirmed reference-level ceiling rather than a port bug; the device-vs-reference results for `9dsg` and `1ahw` are in `docs/pharma-benchmark.md`.

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0, Blackhole P150a):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.863 Å | 0.891 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 1.774 Å | 0.915 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.725 Å | 0.909 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 1.417 Å | 0.936 | ≤6.0 Å / ≥0.50 | PASS |

**BoltzGen designability** — n=4, `examples/binder.yaml`: scRMSD median 0.820 Å, 4/4 designs (100%) ≤2 Å (floor ≤2.0 Å / ≥50%) — PASS.

**ESMC embedding parity** (fused-RoPE shipped path vs reference esm, 76-residue sequence, PCC floor 0.99):

| model | per-res PCC | pooled | logits | argmax | result |
|---|---|---|---|---|---|
| esmc-300m | 0.99961 | 0.99993 | 0.99990 | 1.0000 | PASS |
| esmc-600m | 0.99964 | 0.99989 | 0.99996 | 1.0000 | PASS |

**UX gate** (`scripts/ux_regression.py`, `examples/trpcage.yaml`): every shipped surface (Boltz-2, ESMFold2, ESMFold2-fast, Protenix-v2, OpenDDE, ESMC-600m embed) cleared live-progress advancement, strict mmCIF/npz parse, and results/manifest shape — PASS.

**Perf gate** (`scripts/perf_regression.py`, Blackhole P150a, trpcage 20 aa single-sequence, 1 recycle / 10 steps / 1 sample, warm 2+5, ±15% threshold):

| model | metric | baseline | current | delta | result |
|---|---|---|---|---|---|
| boltz2 | structures/s | 1.186 | 1.190 | +0.3% | PASS |
| esmfold2 | structures/s | 1.665 | 1.705 | +2.4% | PASS |
| esmfold2-fast | structures/s | 2.271 | 2.290 | +0.8% | PASS |
| protenix-v2 | structures/s | 2.406 | 2.383 | -1.0% | PASS |
| opendde | structures/s | 1.920 | 1.922 | +0.1% | PASS |
| esmc-600m | seq/s | 21.09 | 20.92 | -0.8% | PASS |

No perf regression. No OOM observed through the gate targets.

### Added
- **OpenDDE** antibody-antigen co-folding (`opendde` / `opendde-abag`).
- **ESMC fused-RoPE** attention kernel for the embed path (accuracy-neutral speedup).
- Opt-in **diffusion trace replay** for the Boltz-2, BoltzGen, and OpenDDE CLIs and the Protenix-v2 Python API.
- **perf-regression** and **UX-regression** harnesses as standing release-gate legs.

### Fixed
- Perf gate compares against the correct per-card-type baseline (P300c vs P150a mismatch no longer reads as a false regression).

## [0.2.5] - 2026-07-11

Protenix-v2 accuracy fixes — the template embedder never ran in any real `predict` call
(`nt` always 0), and the trunk ran at 3 recycles instead of its spec 10; fixing both closes
a real delivered-RMSD gap that every PyPI install of 0.2.4 and earlier ships with. Also
includes the `embed --controller` persistent-worker dispatch and ESMC-6B multicard fanout
fix below (already hardware-gated at merge time, re-confirmed on this combined HEAD).

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.77 Å | 0.917 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.73 Å | 0.797 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.72 Å | 0.909 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 1.42 Å | 0.935 | ≤6.0 Å / ≥0.50 | PASS |

**Protenix-v2: 3.87 Å → 1.42 Å** — the template-embedder + recycling fixes below close the
gap to the other models; it's no longer the accuracy outlier. Boltz-2/ESMFold2/ESMFold2-fast
unchanged within seed-to-seed noise vs 0.2.4.

**BoltzGen designability** — n=4, `examples/binder.yaml`: scRMSD median 0.67 Å, 4/4 designs
(100%) ≤2 Å — no regression vs 0.2.4's n=8 measurement (0.84 Å median, 7/8 ≤2 Å).

No OOM: `examples/615.yaml` and `examples/1303.yaml` (Boltz-2 `--fast`) completed cleanly;
the full supported range to `examples/3233.yaml` (4-chain multimer + ligand) was already
verified OOM-free on this unchanged Boltz-2 code (`docs/boltz2-tt-vs-nvidia.md`). No perf
regression: Boltz-2 `--fast` warm e2e at L=615 is **46.5 s**, vs the 43.4 s 0.2.4-era
baseline — within run-to-run/environment noise on the same unchanged code path.

### Fixed
- **Protenix-v2: template embedder never ran** — `nt` (template count) was always 0 in
  every real `predict` call, so the template-embedder pass was silently skipped.
- **Protenix-v2: `recycling_steps` default 3 → 10** — the trunk now runs at its spec
  recycle count (previously reused Boltz-2/ESMFold2's default of 3); the correct
  default once the template-embedder fix above made recycling actually informative.
  This makes Protenix-v2 slower per-fold than 0.2.4 (more recycles) — expected,
  not a regression; see the gate wall-clock above.
- ESMC-6B `--devices` fanout regression past 2 cards, root-caused to two independent
  host-side bottlenecks (both fixed, verified bit-exact, end-to-end scaling now
  monotonic to 4 cards — see `docs/esmc-multicard-scaling.md`):
  - **Redundant weight loading**: the N data-parallel workers now share one host-tiled
    copy of the 24 GB checkpoint via a `/dev/shm` cache (`esmc.load_esmc6b_shared` +
    `tenstorrent.weight_cache`) instead of each independently reading+tiling it.
    Per-worker load drops from ~10–16 s (∝N, bandwidth-contended) to ~2.2 s.
  - **Host CPU thread-pool oversubscription**: each shard subprocess's torch/OMP/BLAS
    pools defaulted to *all* host cores, so N co-resident shards oversubscribed the
    host (~21 loadavg on a 16-core host at N=4). `esmc._thread_cap_env` caps them to
    `cores // n_workers`, mirroring the existing `main._cap_worker_threads` fix for
    the fleet worker pool.
  - Net: esmc-6b/N=256 on qb2 goes from 0.66x@4-cards (regression) to **1.49x@4-cards**
    (monotonic 1.00x → 1.33x → 1.43x → 1.49x). Bit-exact vs single-card
    (`scripts/esmc6b_shared_cache_parity.py`, `scripts/esmc_multicard_parity.py`,
    max|Δ|=0); all other models and the single-card path are unchanged.

### Added

- `tt-bio embed --controller URL`: dispatch to a persistent `tt-bio controller`/`worker`
  pool instead of spawning per-call subprocesses. A worker's ESMC model stays resident
  across calls, so the weight reload that dominates `--devices` wall-clock for
  `esmc-6b` (see `docs/esmc-multicard-scaling.md`) becomes a one-time cost per worker
  lifetime instead of a per-invocation tax (measured: esmc-6b N=48 50.0s cold -> 9.1s
  warm on 1 card, 261s cold -> 13.4s warm on 2 cards; bit-exact vs single-shot). Reuses
  the existing predict/design scheduler/lease machinery (`tt_bio/distributed.py`,
  `tt_bio/worker.py`) — no new dispatch mechanism. `--devices` (per-call subprocess
  fanout) is unchanged and still the right choice for one-off invocations with no
  standing controller.

### Measured
- Re-measured `esmc-300m`/`esmc-600m` `--devices` wall-clock scaling on qb2 post
  thread-cap fix (N=48/256/4096, see `docs/esmc-multicard-scaling.md`): the original
  table's `esmc-600m/N=256` 3-card 0.62x cliff does not reproduce (now a 0.87x dip,
  within run-to-run noise) — no regression for either model at any previously-fine
  config. New finding: both models scale far more modestly on qb2 (~1.1x@4cards for
  N=4096) than the original table's qb1 numbers (~2x), most likely because `embed
  --devices` pays an extra per-shard mesh-topology setup cost on qb2 that `esmc-6b`'s
  large weight load absorbs but these smaller models don't — also surfaced that
  `embed --devices` with >1 device currently TT_FATALs out-of-the-box on qb2 unless
  `TT_MESH_GRAPH_DESC_PATH` is set manually (the `predict` path already handles this
  P300-board-misdetection quirk automatically; `embed`'s fanout path doesn't yet).
  Parity re-verified bit-exact for both models.

## [0.2.4] - 2026-07-10

Device-resident trunk for `tt-bio gen` (BoltzGen) — no structure-model code changed for
Boltz-2/ESMFold2/Protenix-v2 (the new `TokenDistanceRecycle`/`TrunkModule` params default to
off/`None`, purely additive).

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.43 Å | 0.944 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.76 Å | 0.798 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.74 Å | 0.907 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 3.87 Å | 0.706 | ≤6.0 Å / ≥0.50 | PASS |

No regression vs 0.2.3 (within TT diffusion's seed-to-seed variance band).

**BoltzGen designability** — n=8 fixed-length-100 designs, `examples/binder.yaml`: scRMSD
median 0.84 Å (resident) vs 0.91 Å (host), 7/8 designs ≤2 Å strict pass (comparable to host's
8/8) — no regression. Wall-clock (design + refold + confidence + analysis + filtering) **697 s
→ 479 s, ~31% faster**. See `docs/boltzgen-resident-trunk.md`.

### Added
- **BoltzGen device-resident trunk** — `TokenDistanceRecycle` (mirrors `TemplateRecycle`) keeps
  the per-iteration token-distance injection fully on-device, collapsing 4 host↔device
  crossings/iteration to 2 (only the template sub-module still round-trips). `Boltz.__init__`
  takes `use_resident_trunk: bool = True`; set `false` to fall back to the original host path.

### Changed
- Promoted Protenix-v2's diffusion denoiser-unit and `AttentionPairBias(has_s=True)` ad-hoc
  checks to proper pytest cases (test-coverage only, no functional change).

## [0.2.3] - 2026-07-09

Multi-card fanout parity for `predict`, a designability (scRMSD) verify script for `tt-bio gen`,
and `tt-bio embed` input/UX polish. No structure-model code changed vs 0.2.2 (`tt_bio/boltz2.py`,
`protenix.py`, `esmfold2.py`, `tenstorrent.py` are byte-identical) — only `esmc.py` and the CLI
(`main.py`) changed, so the release gate below is a confirmation run, not a re-verification.

**Release gate** (`scripts/release_gate.py`, `examples/prot.yaml`, 200 steps / 5 samples, seed 0):

| model | CA-RMSD | TM | floor | result |
|---|---|---|---|---|
| Boltz-2 | 1.60 Å | 0.931 | ≤3.0 Å / ≥0.75 | PASS |
| ESMFold2 | 2.28 Å | 0.832 | ≤4.0 Å / ≥0.65 | PASS |
| ESMFold2-fast | 1.74 Å | 0.907 | ≤4.5 Å / ≥0.60 | PASS |
| Protenix-v2 | 3.87 Å | 0.706 | ≤6.0 Å / ≥0.50 | PASS |

Full test suite: 71 passed, 46 skipped (missing optional reference checkpoints/packages, same
gap as prior releases), 0 failed. No OOM: `examples/615.yaml` and `examples/1303.yaml`
(Boltz-2 `--fast`) completed cleanly; the full supported range up to `examples/3233.yaml`
(4-chain multimer + ligand) was already verified OOM-free on this same unchanged model code
(`docs/boltz2-tt-vs-nvidia.md`). No perf regression: Boltz-2 `--fast` warm e2e at L=615 is
**43.4 s**, matching the 0.2.2-era baseline exactly (same code path since before 0.2.2).

### Added
- **`tt-bio predict --devices`** — alias for `--device_ids` (comma-separated card ids), matching `tt-bio embed`'s flag name; `--device_ids` still works for back-compat.
- **BoltzGen designability (scRMSD) verify script** — `scripts/boltzgen_designability.py` harvests the self-consistency RMSD `tt-bio gen` already computes and summarizes/gates on it; see `docs/boltzgen-designability.md`.
- **`tt-bio embed --devices` wall-clock scaling measured** (`docs/esmc-multicard-scaling.md`) — real ~2x @ 4 cards for `esmc-600m` on large batches, but flat/worse for small batches and for `esmc-6b` beyond 2 cards (concurrent weight-load contention); README softened to match. Performance-only finding, no change to the (already bit-exact) sharding correctness.

### Changed
- **`tt-bio embed` input handling** — `DATA` now also accepts a YAML `{id: sequence}` mapping or a bare sequence string (previously FASTA file/directory only), writes a `manifest.json` (model/pool/shapes/dtype + which output file holds each sequence) alongside the embeddings, and reports bad input as a one-line error instead of a raw traceback.

## [0.2.2] - 2026-07-09

Turns MSA on by default for Boltz-2 / Protenix-v2 (the fix for the misleading no-MSA
accuracy result) and ships the ESMC multi-card embedding fanout. No model numerics changed
vs 0.2.1 — the MSA compute path was already hardware-gated; this only flips its default and
adds a local-DB→online fallback with a privacy notice, plus a `--single_sequence` opt-out.
Ground-truth gate on the default path (`examples/prot.yaml`): Boltz-2 CA-RMSD 2.49 Å / TM
0.78, Protenix-v2 3.47 Å / TM 0.75.

### Added
- **Multi-card fanout for `tt-bio embed`** — `--devices 0,1,2,3` (CLI) / `devices=[...]` (`tt_bio.esmc.embed`) shards a sequence set across several TT cards, one pinned worker per card, and reassembles the embeddings in input order. Data-parallel and lossless: each shard's output is bit-exact to the single-card path (verified on-hardware, Δ=0 per-residue/pooled/logits).
- **`--single_sequence` flag** for `predict` — deliberately fold Boltz-2/Protenix-v2 without an MSA (skips both the local-DB lookup and the online fallback), for batch-screening orphan sequences.

### Changed
- **Boltz-2 and Protenix-v2 use an MSA by default** — these MSA-dependent models no longer silently fold single-sequence. With no MSA flags, `predict` uses a local ColabFold DB (`~/.boltz/msa_db`) if present, else falls back to the online ColabFold server and prints a one-line notice naming the server the sequences are sent to (they leave the machine). Pass `--msa_db_path` for a private offline DB, or `--single_sequence` to skip the MSA. ESMFold2 / ESMFold2-Fast are unchanged (single-sequence by design). Ground-truth gate on the default path (`examples/prot.yaml`): Boltz-2 CA-RMSD 2.49 Å / TM 0.78, Protenix-v2 3.47 Å / TM 0.75.

## [0.2.1] - 2026-07-09

Adds the ESMC embeddings capability merged since 0.2.0 (already hardware-gated at merge
time) and fixes packaging/docs metadata that was stale since 0.2.0. No model code changed
for existing capabilities — the 0.2.0 accuracy/perf/OOM gate still holds.

### Added
- **ESMC protein-language-model embeddings** — `tt-bio embed` CLI + Python API
  (`tt_bio.esmc.embed`): per-residue and pooled embeddings from ESMC-300M/600M/6B, no
  folding head or MSA required. Parity vs reference ESMC: per-residue/pooled PCC
  0.9995-0.9999 across variants (normal and `--fast`).
- Automatic batching + length-bucketing for `tt-bio embed` on ESMC-300M/600M (~18.5x warm
  throughput vs unbatched); exact row-independence (masked batched output bit-identical to
  running each sequence alone), PCC 0.9996+.

### Fixed
- `pyproject.toml` `description` was still "Boltz-2 implementation..." — now lists every
  shipped capability (Boltz-2, ESMFold2, Protenix-v2, BoltzGen, ESMC).
- `pyproject.toml` had no `readme` field, so the PyPI project page rendered with an empty
  long description — now points at `README.md`.
- README: `pip install tt-bio` (PyPI) is now the primary install path (the wheel has been
  on PyPI since 0.2.0); git/source moved to a secondary section. Intro paragraph now
  mentions ESMC embeddings. The dense Boltz-2/ESMFold2/Protenix-v2 feature-support
  paragraph is now a compact table.

## [0.2.0] - 2026-07-09

Release gate verified on Blackhole (p150a): Protenix-v2 e2e real-weight parity (seed0-vs-reference
Kabsch RMSD 8.7 Å, within the sampler's own seed-to-seed variance band); Protenix component parity
14/14, Boltz-2 13/13, ESMFold2 plddt/distogram parity, host suite green; no OOM across the supported
size range.

### Added
- **Protenix-v2 denoise ttnn trace** — opt-in `fold(trace=True)` (with
  `get_device(trace_region_size=1 << 30)`): captures and replays the dispatch-bound
  denoise stream. Lossless (bit-exact vs untraced) and ~22% faster warm diffusion at L256,
  a larger end-to-end win as `diffusion_samples` grows.

### Changed
- Trace/device toggles are now normal function arguments (`fold(trace=...)`,
  `get_device(trace_region_size=...)`) instead of environment variables.

### Fixed
- Input validation hardening: unique chain ids past 26 chains, reject inputs that share a
  name stem, keep blank-id FASTA chains, reject empty polymer sequences, and validate
  explicit `--device_ids` against the cards actually present.
- `tt_bio.__version__` now reports the installed `tt-bio` version (previously read the wrong
  package and could be undefined).
- README/docs consistency pass (flags, examples, model list).

## [0.1] - initial
- Boltz-2, ESMFold2, Protenix-v2 structure prediction and BoltzGen binder design on
  Tenstorrent Blackhole / Wormhole, single- and multi-card. Installed from source.
