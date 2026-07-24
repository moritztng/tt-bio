#!/usr/bin/env python3
"""Packaging regression guard — the "never ship a dropped data file again" leg.

Builds the wheel and sdist from the current source tree and asserts that every
non-``.py`` runtime data file tracked under ``tt_bio/`` ships in BOTH artifacts.
This is the exact bug class that broke every clean ``pip install tt-bio==0.3.3``:
``[tool.setuptools.package-data]`` listed only the two vendored LICENSEs, so the
13 files the package loads by path (``tt_bio/data/protein_ref_conformers.json``,
the ``tt_bio/boltzgen/resources/**`` tree) were silently dropped from the
published wheel and sdist, and protenix-v2 / opendde / boltzgen crashed at
featurization / ``_configure`` on a fresh install.

The expected file set is derived from the repo itself (``find tt_bio -type f
! -name "*.py"``), so it stays in sync as data files are added — a new data file
committed under ``tt_bio/`` is automatically required to ship, no allowlist to
forget. Exit 0 iff every expected file is present in the wheel AND the sdist AND
on disk after a clean ``pip install --no-deps --target`` of the wheel; 1 otherwise.

Optional ``--fold`` mode goes deeper: installs the wheel WITH deps into the
scratch venv and runs one protenix-v2 fold, one opendde covalent-bond fold, and
one ``tt-bio gen run`` boltzgen design, asserting each gets past the
FileNotFoundError class (succeeds, or fails for an unrelated reason). This needs
a Tenstorrent card and the full dep tree; the default artifact-contents check is
the fast, card-free guard that catches the bug class on its own.

    # fast card-free guard (run before every tag, also in CI)
    python3 scripts/packaging_smoke.py
    # deeper on-device check (needs a card + full deps)
    TT_VISIBLE_DEVICES=0 python3 scripts/packaging_smoke.py --fold

Wire into RELEASING.md as a required pre-tag step alongside the accuracy / perf /
UX gates. See the v0.3.4 changelog for the incident this prevents.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = "tt_bio"

# Wall-clock ceilings so a hung pip/network step or a wedged on-device fold can
# never stall a release (standing gate rule: every external / long step gets a
# timeout + honest fallback). Generous vs the real cost so a timeout means stuck,
# not slow. build/pip are network-bound; the --fold leg runs real inference.
PIP_TIMEOUT_S = 600
BUILD_TIMEOUT_S = 600
FOLD_TIMEOUT_S = 1800


def _expected_data_files() -> list[str]:
    """Every non-.py file under tt_bio/ (the set the wheel/sdist must ship).

    Derived from the repo so a newly committed data file is automatically
    required — no allowlist to forget, which is exactly how 0.3.3 slipped.
    """
    files = []
    for p in sorted((REPO_ROOT / PKG).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".py":
            continue
        if "__pycache__" in p.parts:
            continue
        files.append(p.relative_to(REPO_ROOT).as_posix())
    return files


def _build() -> tuple[Path, Path]:
    """Build wheel + sdist into ./dist, return their paths."""
    dist = REPO_ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", "build"],
                   check=True, timeout=PIP_TIMEOUT_S)
    subprocess.run([sys.executable, "-m", "build", "--quiet"], cwd=REPO_ROOT,
                   check=True, timeout=BUILD_TIMEOUT_S)
    wheels = sorted(dist.glob("tt_bio-*.whl"))
    sdists = sorted(dist.glob("tt_bio-*.tar.gz"))
    if not wheels or not sdists:
        sys.exit(f"build produced no wheel/sdist in {dist}")
    return wheels[-1], sdists[-1]


def _wheel_names(whl: Path) -> set[str]:
    with zipfile.ZipFile(whl) as z:
        return set(z.namelist())


def _wheel_requires(whl: Path) -> set[str]:
    """Runtime dependency NAMES declared in the wheel's METADATA (Requires-Dist),
    normalized (lowercased, extras/version/markers stripped). This is what a
    ``pip install tt-bio`` would pull. Compared against pyproject's declared
    dependencies so a dependency dropped from ``[project.dependencies]`` — which
    would make a fresh install import-fail at runtime — fails the gate at the
    artifact level, cheaply and card-free (no need to actually resolve the tree).
    """
    import re
    names: set[str] = set()
    with zipfile.ZipFile(whl) as z:
        meta = next((n for n in z.namelist()
                     if n.endswith(".dist-info/METADATA")), None)
        if meta is None:
            return names
        for line in z.read(meta).decode().splitlines():
            if line.startswith("Requires-Dist:"):
                spec = line.split(":", 1)[1].strip()
                # strip environment markers (after ';') and extras/version
                spec = spec.split(";", 1)[0].strip()
                m = re.match(r"[A-Za-z0-9._-]+", spec)
                if m:
                    names.add(m.group(0).lower().replace("_", "-"))
    return names


def _pyproject_requires() -> set[str]:
    """Runtime dependency names declared in pyproject's [project.dependencies]."""
    import re
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11 fallback
        import tomli as tomllib  # type: ignore
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    deps = data.get("project", {}).get("dependencies", []) or []
    names: set[str] = set()
    for d in deps:
        m = re.match(r"[A-Za-z0-9._-]+", d.strip())
        if m:
            names.add(m.group(0).lower().replace("_", "-"))
    return names


def _check_dependencies(whl: Path) -> list[str]:
    """Assert the wheel's declared runtime deps match pyproject's. Catches a
    dependency dropped from the published metadata (a fresh install would then
    import-fail) without installing the tree. Returns failures.

    Note: this catches a *declaration* drop (pyproject vs wheel metadata). It does
    NOT catch a dependency the code imports but nobody ever declared — that needs a
    truly isolated install and is documented as the --fold leg's job / a follow-up
    --isolated mode, not this fast card-free guard."""
    pj = _pyproject_requires()
    wl = _wheel_requires(whl)
    failures = []
    for miss in sorted(pj - wl):
        failures.append(f"wheel METADATA missing declared dependency: {miss}")
    return failures


def _sdist_names(sdist: Path) -> set[str]:
    import tarfile
    with tarfile.open(sdist) as t:
        return {m.name for m in t.getmembers() if m.isfile()}


def _check_artifacts(whl: Path, sdist: Path, expected: list[str]) -> list[str]:
    """Assert every expected data file ships in both artifacts. Returns failures."""
    whl_names = _wheel_names(whl)
    sdist_names = _sdist_names(sdist)
    failures = []
    for rel in expected:
        # wheel stores files under tt_bio/... directly
        whl_hit = rel in whl_names
        # sdist stores files under tt_bio-<ver>/tt_bio/...
        sdist_hit = any(n.endswith("/" + rel) for n in sdist_names)
        if not whl_hit:
            failures.append(f"wheel missing: {rel}")
        if not sdist_hit:
            failures.append(f"sdist missing: {rel}")
    return failures


def _check_install(whl: Path, expected: list[str]) -> list[str]:
    """Install the wheel --no-deps into an isolated target dir, assert files land on disk.

    Uses ``pip install --target`` rather than a venv so the check works on any
    interpreter with pip — including uv-managed CPython builds that ship no
    ``ensurepip`` wheels (a fresh ``venv.create(with_pip=True)`` raises there).
    The target dir holds only the wheel's own contents, so it is as clean as a
    fresh venv for the file-presence assertion without the ensurepip dependency.
    """
    with tempfile.TemporaryDirectory(prefix="tt-bio-pkg-smoke-") as tmp:
        target = Path(tmp) / "site"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--no-deps", "--target", str(target), str(whl)],
                       check=True, timeout=PIP_TIMEOUT_S)
        failures = []
        for rel in expected:
            # rel is "tt_bio/..."; --target lays the package out as <target>/tt_bio/...
            if not (target / rel).exists():
                failures.append(f"installed missing: {rel}")
        return failures


def _make_venv(venv_dir: Path) -> Path:
    """Create a venv whose deps + pip come from the parent interpreter.

    ``with_pip=False`` + ``system_site_packages=True`` inherits the parent's pip
    and dependency tree, so this works on interpreters without ``ensurepip``
    (uv-managed CPython) and avoids re-resolving the heavy TT dep tree. The
    wheel is installed --no-deps into the venv afterwards, so the venv's own
    ``tt_bio`` (and its ``tt-bio`` console script) shadow any inherited copy.
    """
    venv.create(venv_dir, with_pip=False, clear=True, system_site_packages=True)
    return venv_dir / "bin" / "python"


def _fold_check(whl: Path) -> int:
    """Install the wheel into a deps-inheriting venv and run one protenix-v2 +
    one opendde + one boltzgen call.

    Asserts each gets past the FileNotFoundError class (the 0.3.3 failure mode).
    A fold that succeeds, or fails for an unrelated reason, passes this guard; a
    fold that fails with a missing-data-file error fails it. Needs the parent
    interpreter to already carry the TT dep tree (run on a card host).
    """
    examples = REPO_ROOT / "examples"
    cases = [
        ("protenix-v2", ["predict", str(examples / "trpcage_no_msa.yaml"),
                         "--model", "protenix-v2", "--single_sequence"]),
        ("opendde", ["predict", str(examples / "opendde_covalent_bond.yaml"),
                     "--model", "opendde", "--single_sequence"]),
        ("boltzgen", ["gen", "run", str(examples / "binder.yaml"),
                      "--num_designs", "1", "--fast"]),
    ]
    with tempfile.TemporaryDirectory(prefix="tt-bio-pkg-fold-") as tmp:
        venv_dir = Path(tmp) / "venv"
        py = _make_venv(venv_dir)
        ttbio = venv_dir / "bin" / "tt-bio"
        print("installing wheel --no-deps into deps-inheriting venv...", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--no-deps",
                        str(whl)], check=True, timeout=PIP_TIMEOUT_S)
        failures = 0
        for name, args in cases:
            print(f"\n{'='*70}\n[fold] {name}: tt-bio {' '.join(args)}\n{'='*70}", flush=True)
            work = Path(tmp) / name
            work.mkdir()
            try:
                proc = subprocess.run([str(ttbio), *args], cwd=work,
                                      capture_output=True, text=True,
                                      timeout=FOLD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"FAIL [{name}]: exceeded {FOLD_TIMEOUT_S}s timeout "
                      f"(possible device wedge / hung dependency)", file=sys.stderr)
                failures += 1
                continue
            out = proc.stdout + proc.stderr
            if "FileNotFoundError" in out and ("protein_ref_conformers.json" in out
                                               or "resources/config/" in out
                                               or "resources/splits/" in out
                                               or "/data/" in out):
                print(f"FAIL [{name}]: still hits a missing-data-file error:\n"
                      f"{out[-800:]}", file=sys.stderr)
                failures += 1
            else:
                print(f"PASS [{name}]: past the missing-data-file gate "
                      f"(exit {proc.returncode})", flush=True)
        return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", action="store_true",
                    help="Also install with deps and run one protenix-v2 + opendde + "
                         "boltzgen fold (needs a Tenstorrent card + full dep tree).")
    args = ap.parse_args()

    expected = _expected_data_files()
    print(f"expecting {len(expected)} non-.py data file(s) under {PKG}/:")
    for f in expected:
        print(f"  {f}")

    whl, sdist = _build()
    print(f"\nbuilt: {whl.name}\n       {sdist.name}")

    failures = _check_artifacts(whl, sdist, expected)
    failures += _check_install(whl, expected)
    dep_failures = _check_dependencies(whl)

    print(f"\n{'#'*70}\nPACKAGING SMOKE — artifact + install contents + deps\n{'#'*70}")
    if failures or dep_failures:
        for f in failures:
            print(f"  FAIL {f}")
        for f in dep_failures:
            print(f"  FAIL {f}")
        if failures:
            print(f"\nGATE FAIL — {len(failures)} data file(s) missing from the built "
                  f"wheel/sdist/install. A clean `pip install` will crash. Fix "
                  f"[tool.setuptools.package-data] / MANIFEST.in before tagging.")
        if dep_failures:
            print(f"GATE FAIL — {len(dep_failures)} declared dependency(ies) dropped "
                  f"from the wheel METADATA. A clean `pip install` won't pull them. "
                  f"Fix [project.dependencies] before tagging.")
        return 1
    print(f"  PASS all {len(expected)} expected data files ship in wheel + sdist "
          f"and land on disk after a clean install.")
    print(f"  PASS all {len(_pyproject_requires())} declared runtime dependencies "
          f"ship in the wheel METADATA.")
    print("GATE PASS — no dropped data files or dependencies.")

    if args.fold:
        print(f"\n{'#'*70}\nPACKAGING SMOKE — on-device fold check\n{'#'*70}")
        if _fold_check(whl) != 0:
            print("GATE FAIL — a fold still hits a missing-data-file error.")
            return 1
        print("GATE PASS — protenix-v2 + opendde + boltzgen past the missing-data gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
