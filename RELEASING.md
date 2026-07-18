# Releasing TT-Bio

A release is cut only from the exact commit that passes the host suite and every
on-device gate below. Run device checks serially on an otherwise idle card.

## Prerequisites

- Python 3.10 or 3.12 with the project and test dependencies installed
- A supported Tenstorrent card with the matching driver and TT-NN runtime
- Model checkpoints already cached or reachable
- `ESM_ROOT` pointing to an upstream `evolutionaryscale/esm` checkout
- ColabFold access or a cached A3M for `examples/prot.yaml`

## Required gates

Run from the repository root:

```bash
python3 -m pytest -v --tb=short

TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm PYTHONPATH="$PWD" \
  python3 scripts/release_gate.py

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/perf_regression.py

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/ux_regression.py
```

The accuracy gate covers Boltz-2, ESMFold2, ESMFold2-fast, Protenix-v2,
OpenDDE, BoltzGen designability, and ESMC-300m/600m reference parity. It folds
7ROA at production sampling settings, parses every written mmCIF, and checks the
confidence-selected structure against these regression limits:

| model | maximum CA-RMSD | minimum TM-score |
|---|---:|---:|
| Boltz-2 | 3.0 Å | 0.75 |
| ESMFold2 | 4.0 Å | 0.65 |
| ESMFold2-fast | 4.5 Å | 0.60 |
| Protenix-v2 | 6.0 Å | 0.50 |
| OpenDDE | 6.0 Å | 0.50 |

BoltzGen passes when at least half of four generated binders refold within
2 Å scRMSD. ESMC passes at per-residue PCC ≥0.99 against upstream ESM.
OpenDDE-abag parity on 1AHW is tracked in `docs/pharma-benchmark.md`.

The performance gate measures warm throughput for every shipped architecture
— the fold models, the ESMC embed path, and the BoltzGen design pipeline
(`tt-bio gen run` on `examples/binder.yaml`, reported as designs/s) — and
compares each with the matching card-type baseline in
`docs/perf_baselines.json`. A slowdown beyond 15% fails. Update a baseline only
for an intentional performance change:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/perf_regression.py --update-baseline --note "reason"
```

The UX gate checks CLI help, live progress phase ordering, strict output parsing,
and results or manifest shape for every user-facing architecture — the fold
models, ESMC embed, and BoltzGen (exercised via `tt-bio gen run`, whose progress
is the gen pipeline's own stdout stage stream under `--debug --log`). BoltzGen
therefore has full three-leg coverage: designability accuracy, designs/s perf,
and gen-run UX plumbing.

Also run the documented supported-size and multi-card smoke cases for the target
hardware. Record hard limits in the changelog; do not infer OOM safety from the
small gate inputs.

If the public ColabFold service is unavailable, place the previously generated
`{sha256(sequence)[:16]}.a3m` in the gate output's `msa/` directory and rerun.

## Cut the release

1. Add a dated changelog section with the measured accuracy, performance, UX,
   and supported-size results.
2. Set the version in `pyproject.toml` and update the README install tag.
3. Tag and push:

```bash
git tag vX.Y.Z
git push origin main --tags
```

The release workflow builds the wheel and source archive, verifies that the tag
matches the package version, checks both artifacts, publishes to PyPI, and
creates the GitHub release.
