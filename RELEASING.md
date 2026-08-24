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

Run from the repository root, and run them with the interpreter that carries the
**pinned** TT-NN, not whatever the gate host's system `python3` happens to have.
`pyproject.toml` pins an exact `ttnn==` version and that is what a user gets from
`pip install tt-bio`; a gate run against a different TT-NN certifies a runtime
nobody ships. The difference is not cosmetic: on a boltz2 no-MSA target, ttnn
0.67.4 and 0.68.0 disagree by several angstrom of Kabsch RMSD on identical code
(see the note in `docs/implementation-parity-data/boltz2-9ncy.json`), which reads
as a parity regression that is not one. Check before you start:

```bash
python3 -c "import importlib.metadata as m; print(m.version('ttnn'))"   # must equal the pyproject pin
```

The surest way is to build the release artifacts first and run every gate from the
venv you installed the wheel into, with `PYTHONPATH="$PWD"` so the tree under test
stays the repository:

```bash
python3 -m build && python3 -m venv /tmp/relvenv
/tmp/relvenv/bin/pip install dist/tt_bio-*.whl
PYTHONPATH="$PWD" /tmp/relvenv/bin/python3 scripts/full_parity_gate.py ...
```

`full_parity_gate.py`, `perf_regression.py` and `ux_regression.py` all spawn their
folds and scorers as `sys.executable`, so the choice propagates to every leg.

```bash
# Pin the card. Much of the suite opens a device, and with TT_VISIBLE_DEVICES unset
# it takes the whole mesh, which collides with anything else running on the host.
TT_VISIBLE_DEVICES=0 python3 -m pytest -v --tb=short

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

# Size-generality arm: folds every structure model at 256/512/768 aa and fails
# if the fired/dark perf-lever set or the runtime scaling exponent drifted from
# docs/size_ladder_baseline.json. A perf lever may not ship default-ON on the
# strength of one sequence length; re-record after an intentional size-affecting
# change with --size-ladder-record (once per card type — a card with no baseline
# fails loudly). See docs/size-generality.md.
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/release_gate.py --model size-ladder

TT_VISIBLE_DEVICES=0 OF3_CKPT=/path/to/of3-p2-155k.pt PYTHONPATH="$PWD" \
  python3 scripts/ux_regression.py
```

OpenFold3 and OpenBind are the two checkpoints tt-bio does not download, so the
accuracy, perf and UX gates need them on disk before they can run those legs.
OpenFold3 is a release-host prerequisite: set `OF3_CKPT` or place the file at
`~/.boltz/of3-p2-155k.pt`, and `ux_regression.py` refuses to start without it
rather than skipping the leg (see `docs/openfold3-port.md`). OpenBind is not a
prerequisite, so its UX leg is skipped with the reason printed on its own row and
again in the verdict line when the checkpoint is absent; fetch it per
`docs/weights.md` to gate it. `ux_regression.py` derives its gated set from the
`--model` choice lists in `tt_bio/main.py` and refuses to start if a shipped model
has no leg, so a new port cannot reach a tag with zero UX coverage.

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

Two guards can stop either gate before it folds anything:

* **Card grant.** `TT_VISIBLE_DEVICES` is the set of cards the run may open. Ask `--workers` for
  a card outside it and the gate refuses in preflight rather than taking a card a sibling job
  holds. A release run leaves it unset, which means the whole box and is the unchanged path; the
  fan-out line above needs it unset. A leg that needs more cards than the grant is skipped as
  `SKIPPED-CARD-GRANT` and listed under `COVERAGE REDUCED`. **A release run must show zero of
  those**: a skipped leg is not coverage, and an all-skipped run reports `GATE INCONCLUSIVE`.
* **Load ceiling.** Both gates refuse to start when the 1-min loadavg is above 1.5x nproc, since
  the numbers would be noise and the gate's own fan-out on top of a loaded box is how a QuietBox
  stops answering. Wait for the box to settle, move hosts, or pass `--load-ceiling 0`.

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

The gate also carries a **capacity** leg, which is about device memory rather
than accuracy. Every other leg compares numbers, so a change that grows the
footprint is invisible to it: the fold either still fits and the numbers match,
or it runs out of memory on a target the gate never folds. The capacity leg folds
the largest supported target (`examples/abag_pilot_expansion/9j4c_abag.yaml`,
1095 tokens) at 50 diffusion samples, measures the peak device DRAM, and fails if
it exceeds the budget or if the run writes fewer structures than it was asked
for. Set `RELEASE_GATE_CAPACITY_MAX_GIB` to gate a card with a different budget.

The **l1-budget** leg gates a part rather than a number. Every L1-edge budget in
`tt_bio/tenstorrent.py` was fitted on a 130-core p150a, and `_apply_grid_thresholds`
keeps those values on any grid of 110 cores or more, so a P300's 110 cores ran budgets
fitted for 130 and a mid-size target died at program creation with an L1
circular-buffer clash (#11). No other leg could see it: they compare numbers, and a
part that dies before any kernel runs produces none. The leg runs the trimul chunk-width
arithmetic for every part class in `L1_BUDGET_PARTS` and folds the target from that
issue across the grid ladder the running part can express, checking that a clash can
always be narrowed out of and that the narrow path returns the same bytes as a run that
never clashed. **A part-specific resource figure entering `tenstorrent.py` gets a row in
`L1_BUDGET_PARTS` in the same commit** — the leg fails if a selectable grid has no row.
See `docs/part-l1-budgets.md` for the measured figures and their provenance.

The **batch-position** leg gates a job shape rather than a target. Every other leg
folds one target per process, so a result that depends on where a target sits in the
batch has nothing to differ from and none of them can see it. v0.6.4 shipped exactly
that: three byte-identical Boltz-2 affinity targets in one job scored 0.648724 /
0.722511 / 0.687149, because unseeded RDKit ETKDG redrew the ligand conformer on every
parse and the affinity checkpoint's lazy load advanced the RNG the first target's
diffusion drew from. The leg folds three identical targets plus one genuinely different
control in a single process and requires the three to agree exactly on the structure and
both affinity heads while the control does not. The control is what keeps the leg honest:
without it, a run where every fold collapsed to one constant would also pass.

`scripts/release_gate.py` remains as a fast single-target smoke proxy (one
7ROA fold per model + a BoltzGen/OpenDDE-abag/ESMC quick check) for a quick
sanity look, but it is no longer the parity gate of record — `full_parity_gate.py`
is the command that must pass before a tag. The two do not overlap awkwardly:
the full gate runs the BoltzGen designability and OpenDDE-abag DockQ legs by
calling `release_gate`'s vetted `run_boltzgen` / `run_opendde_abag` / `run_nesso1`
**in-process** (capturing their real scRMSD/DockQ/scalar numbers), so there is one
implementation of each leg, not two.

Nesso-1 is scored by `release_gate.py --model nesso1`, which needs no ccd.pkl setup:
`find_ccd` downloads the 413 MB file from the checkpoint repo on a miss, so `NESSO_CACHE`
is an override rather than a prerequisite. Run it standalone with:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/release_gate.py --model nesso1
```

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
  The boltz2 affinity legs carry a higher floor of 7200 s
  (`Leg.min_fold_timeout`): their affinity trunk runs in fp32 on the host, so a
  co-tenanted host makes them slow rather than wrong — the wider window keeps
  that from reading as an ERROR.
- **Offline MSA fallback.** When the public ColabFold service is down or flaky,
  set `RELEASE_GATE_MSA_DIR` to a directory holding the cached
  `{sha256(sequence)[:16]}.a3m` files; the network-MSA legs then fold with
  `--msa_dir` and never touch the network. It must cover **every** chain of every
  MSA-dependent target you select, `examples/prot.yaml` and `examples/1ahw_abag.yaml`
  both, not just the single-chain fold legs. `release_gate.py` checks this up front and
  prints the exact files to seed; a dir seeded for one target only used to fail an hour
  in, as a missed accuracy floor. `RELEASE_GATE_FOLD_TIMEOUT` tunes the
  `release_gate.py` fold timeout for a slow host.

### Verdict semantics

| verdict | meaning | gate effect |
|---|---|---|
| `PASS` | diffusion leg: every metric within the measured bf16 envelope (§ below); other legs: metric within its threshold / recorded noise floor | pass |
| `PASS-caveated` | (legacy R/D/X only) gate metric passes; a documented secondary metric GAPs on a known bf16 floor | pass (equivalent to PASS for drift) |
| `GAP` | diffusion leg: a metric exceeds the bf16 envelope — a real residual to hunt; other legs: metric outside its floor | **fail** — unless it reproduces a committed `GAP-evidenced` |
| `GAP-evidenced` | a GAP proven to be a genuine bf16-backend floor, accepted in `docs/implementation-parity.md` (a committed verdict only) | a live GAP that matches it reproduces (pass) |
| `DRIFT` | live verdict does not reproduce the committed one and is not an improvement | **fail**; never silently overwrites the doc |
| `BLOCKED-REF-REGEN-NEEDED` | reference missing or its fingerprint changed — diffusion legs need `ref_fp32`+`ref_bf16` CPU references (`--regen-refs`) | not a failure — the slow opt-in regen path, reported separately; if EVERY leg is blocked the run prints `GATE INCONCLUSIVE` and exits nonzero |
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

All three are tt-bio's OWN torch path (CPU references via `--accelerator cpu --no_kernels`), so
they are the same code with a backend/dtype toggle. But shared draws do NOT hold from the single
`--seed` alone: the device (ttnn) trunk and the CPU (torch) trunk consume the global RNG
differently between that seed and the sampler, so the device would otherwise draw DIFFERENT
diffusion noise than the reference (measured 2026-07-23 — the device vs CPU initial noise diverged
completely). The gate therefore sets `TT_BIO_SHARED_DRAW_SEED` on all three runs, which re-seeds in
`AtomDiffusion.sample` immediately before the first `torch.randn` (covering the structure AND
affinity samplers); with it, device and reference draw byte-identical noise (verified bit-exact) so
the only difference between any two runs is arithmetic. `--regen-refs` and the device fold both set
it automatically; it is unset in production. Pass, per leg per metric `d(.,.)`:

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
If EVERY leg in a run is blocked this way, the gate prints `GATE INCONCLUSIVE` and exits
nonzero — zero scored legs is no evidence of parity.
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
OpenFold3, OpenDDE, BoltzGen designability, OpenDDE-abag antibody-antigen
docking, and
ESMC-300m/600m reference parity. It folds 7ROA at production sampling settings,
parses every written mmCIF, and checks the confidence-selected structure against
these regression limits:

| model | maximum CA-RMSD | minimum TM-score |
|---|---:|---:|
| Boltz-2 | 3.0 Å | 0.75 |
| ESMFold2 | 8.0 Å | 0.40 |
| ESMFold2-fast | 4.5 Å | 0.60 |
| Protenix-v2 | 6.0 Å | 0.50 |
| OpenFold3 | 3.5 Å | 0.70 |
| OpenDDE | 6.0 Å | 0.50 |

ESMFold2's floor is loose because it is anchored to the **default single-sequence**
fold (measured 5.80 Å / TM 0.508 on Blackhole), not the MSA-on fold the old 4.0 Å /
0.65 came from. Needing an MSA on 7ROA is a model-quality property of that
checkpoint, not a port defect — its tight numerics live in `full_parity_gate.py`'s
esmfold2 leg. `scripts/release_gate.py:MODELS` is the source of truth for these
numbers; keep this table in sync with it.

BoltzGen passes when at least half of four generated binders refold within
2 Å scRMSD. ESMC passes at per-residue PCC ≥0.99 against upstream ESM. Nesso-1 predicts no
coordinates, so it is not in that table: it passes when the worst of its eleven output
scalars stays inside 5.0x upstream's own run-to-run spread and the device repeats agree to
1e-6. Floors live in `scripts/nesso1_port/device_parity.py`.
OpenDDE-abag co-folds the 1AHW Fab + antigen complex and passes when the
confidence-selected complex scores global DockQ ≥0.50 against the experimental
1AHW structure (a floor that catches a gross mis-dock; the measured baseline is
0.863 best-confidence). DockQ is an eval-time requirement, not a project runtime
dep — set `OPENDDE_DOCKQ_PYTHON` to a venv with DockQ (==2.1.3) installed if the
gate venv does not carry it. The 1AHW implementation-parity detail stays in
`docs/implementation-parity.md`.

The performance gate measures warm throughput for every shipped architecture
— the fold models, the ESMC embed path, and both design pipelines
(BoltzGen via `tt-bio design --model boltzgen` on `examples/binder.yaml` and
RFD3 via `tt-bio design --model rfd3`, each reported as designs/s) — and
compares each with the matching card-type baseline in
`docs/perf_baselines.json`. A slowdown beyond 15% fails.

"Every shipped architecture" is enforced, not aspirational: `perf_regression.py`
cross-checks its `SPECS` dict against `tt_bio.main.PREDICT_MODELS` /
`EMBED_MODELS` / `SAPROT_MODELS` (the same lists each CLI `--model` choice is
built from) before running anything, and refuses to start if any shipped model
has neither a `SPECS` entry nor a documented `SPECS_EXEMPT` reason. This closes
the gap that let OpenDDE's antibody-antigen checkpoint (`opendde-abag`) ship a
>60x diffusion-precision slowdown in v0.3.3/v0.3.4 with zero perf coverage — it
shared its implementation class with the already-covered `opendde` entry, so a
per-model perf gap for an "already covered" model was invisible until a real
fold caught it days after release. Two models sharing one class (`opendde` /
`opendde-abag`, like `boltz2` / `boltz2-affinity` already did) now get
independent `SPECS` entries. Adding a new `--model` choice anywhere in
`tt_bio/main.py` and forgetting its perf entry is now a loud gate failure
naming the exact model, not a silent gap.

Update a baseline only for an intentional performance change:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python3 scripts/perf_regression.py --update-baseline --note "reason"
```

The UX gate checks CLI help, live progress phase ordering, strict output parsing,
and results or manifest shape for every user-facing architecture — the fold
models, ESMC embed, and both design models (BoltzGen via
`tt-bio design --model boltzgen`, whose progress is the pipeline's own stdout
stage stream under `--debug --log`, and RFD3 via `tt-bio design --model rfd3`).
BoltzGen therefore has full three-leg coverage: designability accuracy,
designs/s perf, and design UX plumbing.

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
