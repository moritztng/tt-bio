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

TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
  PYTHONPATH="$PWD" \
  python3 scripts/full_parity_gate.py --workers pc:0

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/perf_regression.py

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/ux_regression.py
```

The parity gate is `scripts/full_parity_gate.py` — the FULL
`docs/implementation-parity.md` story (every leg, every model/target, 5-seed
depth) as one command. It reuses the committed reference fixtures under
`docs/implementation-parity-data/ref-fixtures/` and only re-runs the device side
plus the comparison, so it finishes in well under an hour when references are
cached and cards are free. Fan it across every card that is up for parallelism:

```bash
python3 scripts/full_parity_gate.py --workers pc:0,qb1:0,qb1:1,qb2:0
```

Each leg's reference fixture carries a `meta.json` pinning the reference
implementation, version, commit, and settings; the runner fingerprints that
meta and compares it to
`docs/implementation-parity-data/ref-fixture-fingerprints.json`. A match takes
the fast path (device-only); a mismatch means the model code, weights, or test
settings changed and the reference must be regenerated, so the leg is flagged
`BLOCKED-REF-REGEN-NEEDED` (the slow opt-in path — run
`scripts/pharma_harvest_ref_fixtures.py` to re-harvest it, then
`scripts/full_parity_gate.py --init-fingerprints` to refresh the index). The
runner never silently overwrites `docs/implementation-parity.md`: a leg that
reproduces within its recorded noise floor is marked `REPRODUCES`; a leg that
drifts outside the floor is flagged `DRIFT — investigate` and exits non-zero.

`scripts/release_gate.py` remains as a fast single-target smoke proxy (one
7ROA fold per model + a BoltzGen/OpenDDE-abag/ESMC quick check) for a quick
sanity look, but it is no longer the parity gate of record — `full_parity_gate.py`
is the command that must pass before a tag.

The accuracy gate covers Boltz-2, ESMFold2, ESMFold2-fast, Protenix-v2,
OpenDDE, BoltzGen designability, OpenDDE-abag antibody-antigen docking, and
ESMC-300m/600m reference parity. It folds 7ROA at production sampling settings,
parses every written mmCIF, and checks the confidence-selected structure against
these regression limits:

| model | maximum CA-RMSD | minimum TM-score |
|---|---:|---:|
| Boltz-2 | 3.0 Å | 0.75 |
| ESMFold2 | 4.0 Å | 0.65 |
| ESMFold2-fast | 4.5 Å | 0.60 |
| Protenix-v2 | 6.0 Å | 0.50 |
| OpenDDE | 6.0 Å | 0.50 |

BoltzGen passes when at least half of four generated binders refold within
2 Å scRMSD. ESMC passes at per-residue PCC ≥0.99 against upstream ESM.
OpenDDE-abag co-folds the 1AHW Fab + antigen complex and passes when the
confidence-selected complex scores global DockQ ≥0.50 against the experimental
1AHW structure (a floor that catches a gross mis-dock; the measured baseline is
0.863 best-confidence). DockQ is an eval-time requirement, not a project runtime
dep — set `OPENDDE_DOCKQ_PYTHON` to a venv with DockQ (==2.1.3) installed if the
gate venv does not carry it. The 1AHW implementation-parity detail stays in
`docs/implementation-parity.md`.

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
