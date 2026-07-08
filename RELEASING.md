# Releasing TT-Bio

Releases are versioned, tested on real Tenstorrent hardware, and installable without pinning
to an arbitrary `main` commit. `main` is the development branch (may contain untested work);
**a tagged release is the tested artifact users should install.**

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged — a release is a promise to customers that it works:

1. **Accuracy / correctness** — full test suite green **and** numerical parity vs the reference /
   paper numbers within tolerance (PCC/RMSD) for every model (Boltz-2, ESMFold2, Protenix-v2,
   BoltzGen). **No accuracy regression** vs the previous release.
2. **No OOM** — run the full supported sequence/complex-size range on the target card(s),
   single- and multi-card, to completion. No out-of-memory. Document any hard size limit in the
   release notes rather than letting a customer hit it.
3. **No perf regression** — benchmark the release commit against the previous release; latency
   and throughput must not regress beyond noise. Record the numbers in the release notes.

If any of the three fails, it does not ship — fix it or hold the release. `main` may be
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
4. CI (`.github/workflows/release.yml`) then builds the sdist + wheel, checks the tag matches
   the `pyproject` version, and publishes a **GitHub Release** with the changelog notes and the
   wheel attached. If PyPI is enabled (below), it also publishes there.

## Enabling PyPI (one-time, maintainer)

Do this once, before the first tag. Until it exists, releases go to GitHub only (the
`pypi-publish` job fails harmlessly and the GitHub Release still publishes).

On <https://pypi.org/manage/account/publishing/> add a **pending Trusted Publisher** for
project `tt-bio`: owner `moritztng`, repository `tt-bio`, workflow `release.yml`, environment
`pypi`. (No API token is created or stored — GitHub authenticates via OIDC.) The next `v*` tag
then publishes to PyPI automatically.
