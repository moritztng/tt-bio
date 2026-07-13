# Releasing TT-Bio

Releases are versioned, tested on real Tenstorrent hardware, and installable without pinning
to an arbitrary `main` commit. `main` is the development branch (may contain untested work);
**a tagged release is the tested artifact users should install.**

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged — a release is a promise to customers that it works.

Run `ux_regression.py` and `perf_regression.py` on a **quiet card** (nothing else contending for
it) — a first predict on a card with other jobs mid-run can cold-start/time out and read as a
false gate failure. Re-run on an idle card before treating a failure there as real.

1. **Accuracy / correctness** — full test suite green **and** numerical parity vs the reference /
   paper numbers within tolerance (PCC/RMSD) for every model (Boltz-2, ESMFold2, Protenix-v2,
   BoltzGen). **No accuracy regression** vs the previous release.

   **REQUIRED — ground-truth fold + designability gate** (`scripts/release_gate.py`): folds one
   easy target end-to-end on the card with production sampling (200 steps / 5 samples) for the
   four fold models, verifies the written mmCIF parses under a strict `Bio.PDB.MMCIFParser`
   (catches writer/format regressions), and gates the confidence-selected structure against a
   per-model ground-truth CA-RMSD / TM-score floor. For **BoltzGen** (a design, not fold, model —
   no ground truth to compare to) it instead runs `tt-bio gen` on `examples/binder.yaml` (n=4,
   production 500-step sampling), parses every written mmCIF the same way, and gates on
   designability: self-consistency RMSD (scRMSD) of each design refolded in isolation with
   Boltz-2, reusing `scripts/boltzgen_designability.py` (see `docs/boltzgen-designability.md`).
   **No tag ships unless it exits 0.**
   ```bash
   TT_VISIBLE_DEVICES=<card> python scripts/release_gate.py   # all 5; exit 0 == all PASS
   ```
   Self-consistency (seed-vs-reference RMSD) is **not** sufficient for the four fold models — it
   passes even when the fold is wrong. Reference floors on 7ROA (`examples/prot.yaml`), best-of-N
   by confidence, at the tag: Boltz-2 ~1.6 Å, ESMFold2 ~2.2 Å, ESMFold2-fast ~1.7 Å
   (single-sequence), Protenix-v2 ~3.5 Å (weak confidence head — correct topology, see
   `docs/protenix-accuracy-investigation.md`). The floors are deliberately generous to absorb TT
   diffusion's seed-to-seed variance; tighten per model as baselines firm up, never below what a
   correct fold hits. BoltzGen's designability floor is scRMSD ≤ 2 Å (BoltzGen's own designable
   bar) on ≥ 50% of the 4 designs — same "catch a gross failure, not a tight target" philosophy.
2. **No UX regression** — the user-facing plumbing every release ships with must keep
   working: the `tt-bio predict` live progress view advances through every real phase
   (load → trunk recycling → diffusion → output) with no phase skipped for every model,
   the emitted CIF/npz parse under a strict standard parser, and `tt-bio predict`/
   `embed --help` + the results/manifest shape hold. This is the guard against the
   "0 → diffusion" / "loading → diffusion" progress-jump class and the malformed-output
   class (e.g. the missing `_atom_site.occupancy` bug). **No tag ships unless it exits 0.**
   ```bash
   TT_VISIBLE_DEVICES=<card> /path/to/env/bin/python scripts/ux_regression.py   # all surfaces; exit 0 == all PASS
   /path/to/env/bin/python scripts/ux_regression.py --cli-only                   # no card; GitHub CI smoke
   ```
   It folds `examples/trpcage.yaml` with minimal steps (UX plumbing, not accuracy), so it
   runs in ~2 min on a card and complements (does not duplicate) the accuracy + perf gates.
   A UX regression blocks a tag on the same standing as an accuracy one. Whenever a new
   user-facing surface ships, extend this guard to cover it.
3. **No OOM** — run the full supported sequence/complex-size range on the target card(s),
   single- and multi-card, to completion. No out-of-memory. Document any hard size limit in the
   release notes rather than letting a customer hit it.
4. **No perf regression** — run the standing perf gate against the committed baselines and
   paste its table into the release notes:
   ```bash
   TT_VISIBLE_DEVICES=<card> PYTHONPATH=<worktree> python3 scripts/perf_regression.py   # exit 0 == no model regressed beyond ±15%
   ```
   `scripts/perf_regression.py` measures WARM steady-state throughput (structures/s for the
   fold models, seq/s for the ESMC embed) for every shipped model on a fixed small input
   (trpcage, 1 recycle / 10 steps / 1 sample; model load + first-compile excluded), compares
   to the per-model baselines in `docs/perf_baselines.json`, and FAILS any model beyond the
   noise threshold (`--threshold`, default 15%). An **intentional** perf change (a landed
   optimization, or a deliberate accuracy/perf tradeoff) updates the baseline explicitly:
   `python3 scripts/perf_regression.py --update-baseline --note "<why>"` and commit the
   baseline diff alongside the change that justifies it — never silently. A regression the
   author didn't intend fails the gate. Add a spec + baseline entry for each new model as it
   ships.

If any of these fails, it does not ship — fix it or hold the release. `main` may be
experimental; the tag is the promise.

## Cut a release

1. Run the gate above on hardware; capture the accuracy table + benchmark numbers.
2. **Bump the version** in `pyproject.toml` (SemVer) and add a dated section to `CHANGELOG.md`
   (include the measured accuracy + perf numbers). Update the install tag in `README.md`.
3. **Tag and push:**
   ```bash
   git tag v0.2.0
   git push origin main --tags
   ```
4. CI (`.github/workflows/release.yaml`) then builds the sdist + wheel, checks the tag matches
   the `pyproject` version, and publishes a **GitHub Release** with the changelog notes and the
   wheel attached. If PyPI is enabled (below), it also publishes there.

## Enabling PyPI (one-time, maintainer)

Do this once, before the first tag. Until it exists, releases go to GitHub only (the
`pypi-publish` job fails harmlessly and the GitHub Release still publishes).

On <https://pypi.org/manage/account/publishing/> add a **pending Trusted Publisher** for
project `tt-bio`: owner `moritztng`, repository `tt-bio`, workflow `release.yaml`, environment
`pypi`. (No API token is created or stored — GitHub authenticates via OIDC.) The next `v*` tag
then publishes to PyPI automatically.
