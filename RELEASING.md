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

# Packaging guard — catches a dropped data file in the wheel/sdist before it
# ships to PyPI (the v0.3.3 bug class: protenix-v2/opendde/boltzgen crashed on
# a clean `pip install` because the package-data globs were missing). Card-free.
python3 scripts/packaging_smoke.py

# Card-free preflight — validates every leg's yaml / fixture+fingerprint / committed-JSON /
# target-id / MSA wiring in seconds. Run it first; it catches a misconfigured leg before a
# device turn is wasted on it. (It also runs automatically at the start of the gate below.)
python3 scripts/full_parity_gate.py --check

TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
  PYTHONPATH="$PWD" \
  python3 scripts/full_parity_gate.py --workers pc:0

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/perf_regression.py

TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/ux_regression.py
```

The packaging guard (`scripts/packaging_smoke.py`) builds the wheel and sdist
from the current tree and asserts every non-`.py` data file under `tt_bio/`
ships in both artifacts and lands on disk after a clean `pip install --no-deps
--target` of the wheel. The expected file set is derived from the repo, so a
newly committed data file is automatically required to ship — no allowlist to
forget. Pass `--fold` to also install the wheel into a deps-inheriting venv and
run one protenix-v2 + one opendde + one boltzgen fold on a card, asserting each
gets past the missing-data-file gate. The card-free default is the required
pre-tag step; `--fold` is the deeper on-device confirmation.

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
is the command that must pass before a tag. The two do not overlap awkwardly:
the full gate runs the BoltzGen designability and OpenDDE-abag DockQ legs by
calling `release_gate`'s vetted `run_boltzgen` / `run_opendde_abag` **in-process**
(capturing their real scRMSD/DockQ numbers), so there is one implementation of
each leg, not two.

### Gate behavior you can rely on

- **Resume is the default.** The gate reuses completed device folds and per-leg
  reports already in `--workdir`, so a run interrupted partway (a bounded shell,
  a lost connection) resumes where it stopped instead of re-folding everything.
  Use a fresh `--workdir` per release commit, or pass `--fresh`, for a clean
  from-scratch run.
- **No leg can hang forever.** Every device fold is bounded by `--fold-timeout`
  (default 2400 s); a fold that never produces `results.json` in that window
  (e.g. a flaky MSA server) is killed with a clear error. A fold that succeeds
  but hangs on shutdown is reaped once its `results.json` is written.
- **Offline MSA fallback.** When the public ColabFold service is down or flaky,
  set `RELEASE_GATE_MSA_DIR` to a directory holding the cached
  `{sha256(sequence)[:16]}.a3m` files; the network-MSA legs then fold with
  `--msa_dir` and never touch the network. `RELEASE_GATE_FOLD_TIMEOUT` tunes the
  `release_gate.py` fold timeout for a slow host.

### Verdict semantics

| verdict | meaning | gate effect |
|---|---|---|
| `PASS` | diffusion leg: every metric within the measured bf16 envelope (§ below); other legs: metric within its threshold / recorded noise floor | pass |
| `PASS-caveated` | (legacy R/D/X only) gate metric passes; a documented secondary metric GAPs on a known bf16 floor | pass (equivalent to PASS for drift) |
| `GAP` | diffusion leg: a metric exceeds the bf16 envelope — a real residual to hunt; other legs: metric outside its floor | **fail** — unless it reproduces a committed `GAP-evidenced` |
| `GAP-evidenced` | a GAP proven to be a genuine bf16-backend floor, accepted in `docs/implementation-parity.md` (a committed verdict only) | a live GAP that matches it reproduces (pass) |
| `DRIFT` | live verdict does not reproduce the committed one and is not an improvement | **fail**; never silently overwrites the doc |
| `BLOCKED-REF-REGEN-NEEDED` | reference missing or its fingerprint changed — diffusion legs need `ref_fp32`+`ref_bf16` CPU references (`--regen-refs`) | not a failure — the slow opt-in regen path, reported separately |
| `ERROR` | the fold or scorer produced no report | **fail** |
| `NO-DATA` | a report with no comparable metric | drift check skipped; a live NO-DATA still fails |

### Integration-parity envelope — the correctness test (supersedes R/D/X)

The correctness question "is this port numerically right end-to-end" is answered by a
DETERMINISTIC shared-draws, measured-bf16-envelope integration test — NOT by the old R/D/X
same-backend self-consistency floor, which is unsound for that question (it compares independent
stochastic samples against a guessed self-spread floor, so it cannot separate a real backend bug
from ordinary sample-to-sample diffusion noise; a correct port could fail it and a subtle bug
could hide in it).

A diffusion model is a deterministic function of its input noise. So the test feeds byte-identical
noise (initial coords + every per-step eps) to three CLOSED-LOOP runs and compares their FINAL
structures with the same per-leg distance the leg already uses (CA/ligand Kabsch-RMSD,
pocket-lDDT, |Δ| for the affinity scalar):

- `device_bf16`    — tt-bio on Tenstorrent (the port under test)
- `reference_fp32` — tt-bio on CPU, `--no_kernels`, fp32 (ground truth)
- `reference_bf16` — tt-bio on CPU, `--no_kernels`, `TT_BIO_REF_BF16=1` (bf16 autocast)

Because the reference is tt-bio's OWN torch path, all three are the same code with a backend/dtype
toggle and draw their diffusion `torch.randn` on CPU MT19937 from the one `--seed`, so shared
draws hold by construction (the only difference between any two runs is arithmetic). Pass, per leg
per metric `d(.,.)`:

    d(device_bf16, reference_fp32)  <=  d(reference_bf16, reference_fp32) * (1 + margin) + abs_floor

The device may differ from the fp32 reference by no more than a bf16 recomputation of the reference
differs from itself (plus a small honest residual for TT-bf16 vs torch-bf16 accumulation,
absorbed by `margin`). The floor is MEASURED per leg, not guessed. If the numerator blows well
past the envelope, that is an unambiguous bug signal — surfaced, never excused as "floor". Scorer:
`scripts/integration_envelope.py`; bf16 reference hook: `tt_bio/worker.py:_maybe_ref_bf16`.

This is the DEFAULT correctness criterion for every diffusion (structure/affinity) leg in
`full_parity_gate.py`. Per leg the gate folds the device once at the reference seed, reads the
leg's two cached CPU references under `<fixture>/ref_fp32/` and `<fixture>/ref_bf16/`, and scores
via `integration_envelope.py` through the one `finalize_leg` path (PASS iff every metric is within
the envelope; else GAP — a real residual to hunt). The CPU references are the cached fixture,
fingerprinted like the old ones, so only the device fold + scoring re-run per release. Generate or
regenerate them (2 CPU folds per leg, run serially — concurrent pure-torch CPU folds oversubscribe
the host) with:

```
# one leg (or drop --leg for every envelope leg); ~2 CPU folds/leg, slow but cached
PYTHONPATH="$PWD" python3 scripts/full_parity_gate.py --regen-refs --leg boltz2-affinity-fkbp12-nomsa
```

A leg whose `ref_fp32`/`ref_bf16` are absent (or whose fingerprint drifted) reports
`BLOCKED-REF-REGEN-NEEDED` and does not fail the gate — regenerate rather than trust a false pass.
The retired R/D/X floor stays available as an opt-in device self-consistency (`D`) DIAGNOSTIC via
`--legacy-rdx`; it is no longer a pass criterion. `--margin` overrides the envelope margin
(default 0.50, justified in `~/.coworker/state/tt-bio-integration-parity-gate.md §4`).

Landed: the scorer, the CPU bf16-reference hook (boltz2 + affinity path), the `--regen-refs`
reference generator, and the envelope verdict wired into `full_parity_gate.py:finalize_leg`. Proven
end-to-end on FKBP12 (no-MSA affinity, seed 0): `full_parity_gate.py --leg
boltz2-affinity-fkbp12-nomsa` folds the device (136 s) and returns `PASS` (all four metrics within
envelope; device-vs-fp32 affinity residual 0.0227 log10(IC50) vs a measured bf16 envelope of 0.0620,
ratio 0.37) with no manual intervention. See `docs/implementation-parity.md` for the head-to-head.
Remaining (CPU-bound, not a code gap): regenerate the cached CPU references for the rest of the leg
matrix — DHFR / trypsin / the MSA legs / Protenix-v2 HSA — so those legs score instead of blocking.
Gate of record: needs Moritz's OK before merge.

### Trusting a new or changed gate

**Any new gate-of-record script, or a significant change to one, must be dry-run
end-to-end on a real release candidate before it is trusted to gate a tag.** The
v0.3.3 release learned this the hard way: `full_parity_gate.py` was made the
parity gate of record and first exercised during the release itself, so a string
of harness/config bugs (leg-id mismatches, a live-vs-committed shape mismatch, no
resume, no network timeout, broken remote fan-out) surfaced one device turn at a
time — an all-day thrash with zero model-numerics problems. Run `--check` and a
one-leg `--dry-run`/fold smoke first; they catch that whole class in minutes.

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

If the public ColabFold service is unavailable, use the offline MSA fallback
described above: set `RELEASE_GATE_MSA_DIR` to a directory holding the previously
generated `{sha256(sequence)[:16]}.a3m` files and rerun. (Equivalently, for a
single leg, drop the a3m into the gate output's `msa/` directory, which is
`predict`'s default `msa_dir`.)

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
