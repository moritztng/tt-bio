# Changelog

All notable changes to TT-Bio are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut from a commit that has passed the on-hardware test suite (see `RELEASING.md`).

## [Unreleased]

### Added
- **Multi-card fanout for `tt-bio embed`** — `--devices 0,1,2,3` (CLI) / `devices=[...]` (`tt_bio.esmc.embed`) shards a sequence set across several TT cards, one pinned worker per card, and reassembles the embeddings in input order. Data-parallel and lossless: each shard's output is bit-exact to the single-card path (verified on-hardware, Δ=0 per-residue/pooled/logits).
- **`--single_sequence` flag** for `predict` — deliberately fold Boltz-2/Protenix-v2 without an MSA (skips both the local-DB lookup and the online fallback), for batch-screening orphan sequences.

### Changed
- **Boltz-2 and Protenix-v2 use an MSA by default** — these MSA-dependent models no longer silently fold single-sequence (the cause of the alarming "~10 Å Protenix-v2" result; see `docs/protenix-accuracy-investigation.md`). With no MSA flags, `predict` uses a local ColabFold DB (`~/.boltz/msa_db`) if present, else falls back to the online ColabFold server and prints a one-line notice naming the server the sequences are sent to (they leave the machine). Pass `--msa_db_path` for a private offline DB, or `--single_sequence` to skip the MSA. ESMFold2 / ESMFold2-Fast are unchanged (single-sequence by design). Ground-truth gate on the default path (`examples/prot.yaml`): Boltz-2 CA-RMSD 2.49 Å / TM 0.78, Protenix-v2 3.47 Å / TM 0.75.

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
