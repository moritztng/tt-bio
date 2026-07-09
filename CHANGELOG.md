# Changelog

All notable changes to TT-Bio are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut from a commit that has passed the on-hardware test suite (see `RELEASING.md`).

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
