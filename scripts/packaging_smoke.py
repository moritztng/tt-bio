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

Dependencies get the same treatment from both ends: every name in
``[project.dependencies]`` must survive into the wheel's ``Requires-Dist``, and
every third-party module the source imports at module level must be declared in
the first place (or listed in ``_UNDECLARED_OK`` with the reason). The second half
is what a metadata-only comparison cannot see — an undeclared import agrees with
itself in pyproject and the wheel, and only crashes on someone else's machine.

Optional ``--fold`` mode goes deeper: installs the wheel WITH deps into the
scratch venv and runs one protenix-v2 fold, one opendde covalent-bond fold, and
one ``tt-bio design --model boltzgen`` design, asserting each gets past the
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


# Import names spelled differently from the distribution that provides them.
_IMPORT_ALIAS = {
    "bio": "biopython",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

# Module-level imports [project.dependencies] deliberately does not declare, and why.
# Anything else the tree imports at module level and nobody declared fails the gate, so
# a new third-party import cannot reach a tag undeclared. That is how rf3 shipped needing
# zstandard / beartype / jaxtyping / pyarrow with none of them declared: each was found by
# a fold crashing in featurization, one package at a time, not by a gate.
# Grandfathered 2026-09-02, checked one at a time; reachability by import of tt_bio.main
# and tt_bio.rf3 measured, not assumed.
_UNDECLARED_OK = {
    "ttnn": "installed out-of-band from Tenstorrent's index, never a PyPI dependency",
    "pygments": "provided by rich; reached only through atomworks' error formatter",
    "msgpack": "provided by msgpack-numpy; vendored esm serialization only",
    "openbabel": "atomworks symmetry/rf2aa path, not reached by importing tt_bio.main or tt_bio.rf3",
    "py3Dmol": "atomworks notebook visualiser, not reached by importing tt_bio.main or tt_bio.rf3",
    "sympy": "atomworks conditions path, not reached by importing tt_bio.main or tt_bio.rf3",
    "redis": "tt_bio/boltzgen/data/parse/a3m.py is unreferenced; tt_bio/data/parse.py is the live a3m parser",
}


def _source_imports() -> dict[str, str]:
    """Third-party module names the package imports at module level, each mapped to
    the first file that imports it.

    Only statements at the top level of a file count. An import nested in a ``try``,
    an ``if``, a function or a class is optional or lazy by construction, so its
    absence is not the import-time crash this guards against.
    """
    import ast
    pkg = REPO_ROOT / PKG
    # A name that resolves to a module of this package is not a third-party import.
    intree = {PKG, "__future__"}
    for p in pkg.iterdir():
        if p.is_dir() and (p / "__init__.py").exists():
            intree.add(p.name)
        elif p.suffix == ".py":
            intree.add(p.stem)
    found: dict[str, str] = {}
    for f in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for n in names:
                if n in sys.stdlib_module_names or n in intree:
                    continue
                found.setdefault(n, f.relative_to(REPO_ROOT).as_posix())
    return found


def _check_undeclared_imports() -> list[str]:
    """Assert every third-party module imported at module level is a declared
    dependency. Returns failures.

    This is the half ``_check_dependencies`` cannot see. That one compares pyproject
    against the wheel's own metadata, so a dependency the code imports but nobody ever
    declared agrees in both places and passes: the wheel is a faithful copy of an
    incomplete declaration. Reading the source instead catches it before the build.
    """
    declared = _pyproject_requires()
    failures = []
    for name, where in sorted(_source_imports().items()):
        if name in _UNDECLARED_OK:
            continue
        dist = _IMPORT_ALIAS.get(name.lower(), name.lower().replace("_", "-"))
        if dist not in declared:
            failures.append(f"undeclared dependency: `import {name}` at module level in "
                            f"{where}, and {dist} is not in [project.dependencies]")
    return failures


def _sdist_names(sdist: Path) -> set[str]:
    import tarfile
    with tarfile.open(sdist) as t:
        return {m.name for m in t.getmembers() if m.isfile()}


def _sdist_root(names: set[str]) -> str | None:
    """The single top-level directory an sdist packs its tree under, or None.

    setuptools writes every member under one ``tt_bio-<version>/`` directory, and
    the membership test below anchors on it. Matching any member that merely ENDS
    with the file would let a copy somewhere else in the tarball
    (``tt_bio-0.7.2/docs/tt_bio/data/x.json``) stand in for the real one.
    """
    roots = {n.split("/", 1)[0] for n in names if "/" in n}
    return roots.pop() if len(roots) == 1 else None


def _check_artifacts(whl: Path, sdist: Path, expected: list[str]) -> list[str]:
    """Assert every expected data file ships in both artifacts. Returns failures."""
    whl_names = _wheel_names(whl)
    sdist_names = _sdist_names(sdist)
    root = _sdist_root(sdist_names)
    failures = []
    if root is None:
        failures.append("sdist has no single root directory; cannot locate its package tree")
    for rel in expected:
        # wheel stores files under tt_bio/... directly
        whl_hit = rel in whl_names
        # sdist stores files under tt_bio-<ver>/tt_bio/...
        sdist_hit = (f"{root}/{rel}" in sdist_names if root is not None
                     else any(n.endswith("/" + rel) for n in sdist_names))
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


# Path fragments that identify a data file missing from the INSTALLED package. They are
# matched against the ``FileNotFoundError:`` line only, which is where CPython puts the
# path that was not found. Matching the whole output instead reads traceback frames too,
# and every frame of a packaged run sits under .../site-packages/tt_bio/..., so any
# FileNotFoundError at all -- a user's missing input file -- would look like this bug.
_MISSING_DATA_MARKERS = (
    "tt_bio/data/",                # protein_ref_conformers.json (protenix-v2, opendde)
    "tt_bio/boltzgen/resources/",  # config/design.yaml, splits (boltzgen)
    "/tt_bio/",                    # any other file the installed package loads by path
)


def _is_missing_data_error(out: str) -> bool:
    """True when a fold's output shows the 0.3.3 failure mode: a file the installed
    package loads by path is not on disk."""
    return any(any(m in line for m in _MISSING_DATA_MARKERS)
               for line in out.splitlines() if "FileNotFoundError" in line)


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
        ("protenix-v2", examples / "trpcage_no_msa.yaml",
         ["predict", "--model", "protenix-v2", "--single_sequence"]),
        ("opendde", examples / "opendde_covalent_bond.yaml",
         ["predict", "--model", "opendde", "--single_sequence"]),
        ("boltzgen", examples / "binder.yaml",
         ["design", "--model", "boltzgen", "--num_designs", "1", "--fast"]),
    ]
    # A renamed example would make every fold die on a FileNotFoundError for the YAML,
    # which is not the marker shape below, so each one would print PASS having tested
    # nothing. Check the inputs first and say so.
    absent = [str(inp) for _, inp, _ in cases if not inp.exists()]
    if absent:
        print("FAIL: fold input(s) missing, the folds below would prove nothing:\n  "
              + "\n  ".join(absent), file=sys.stderr)
        return len(absent)
    with tempfile.TemporaryDirectory(prefix="tt-bio-pkg-fold-") as tmp:
        venv_dir = Path(tmp) / "venv"
        py = _make_venv(venv_dir)
        ttbio = venv_dir / "bin" / "tt-bio"
        print("installing wheel --no-deps into deps-inheriting venv...", flush=True)
        # --force-reinstall matters: system_site_packages means pip can see a tt_bio of
        # the same version in the PARENT interpreter and report "already satisfied", so
        # the child venv gets no tt_bio and no `tt-bio` console script, and every fold
        # below dies on FileNotFoundError for the script itself instead of testing the
        # wheel. That reads as a packaging failure and is not one.
        subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--no-deps",
                        "--force-reinstall", str(whl)],
                       check=True, timeout=PIP_TIMEOUT_S)
        if not ttbio.exists():
            print(f"FAIL: the wheel installed but left no console script at {ttbio}; "
                  f"check [project.scripts] in pyproject.toml", file=sys.stderr)
            return 1
        failures = 0
        for name, inp, flags in cases:
            args = [flags[0], str(inp), *flags[1:]]
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
            if _is_missing_data_error(out):
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
    import_failures = _check_undeclared_imports()

    print(f"\n{'#'*70}\nPACKAGING SMOKE — artifact + install contents + deps\n{'#'*70}")
    if failures or dep_failures or import_failures:
        for f in failures + dep_failures + import_failures:
            print(f"  FAIL {f}")
        if failures:
            print(f"\nGATE FAIL — {len(failures)} data file(s) missing from the built "
                  f"wheel/sdist/install. A clean `pip install` will crash. Fix "
                  f"[tool.setuptools.package-data] / MANIFEST.in before tagging.")
        if dep_failures:
            print(f"GATE FAIL — {len(dep_failures)} declared dependency(ies) dropped "
                  f"from the wheel METADATA. A clean `pip install` won't pull them. "
                  f"Fix [project.dependencies] before tagging.")
        if import_failures:
            print(f"GATE FAIL — {len(import_failures)} module-level import(s) nobody "
                  f"declares. A clean `pip install` won't pull them and the import "
                  f"crashes. Add them to [project.dependencies], or to _UNDECLARED_OK "
                  f"with the reason if they genuinely need no declaration.")
        return 1
    print(f"  PASS all {len(expected)} expected data files ship in wheel + sdist "
          f"and land on disk after a clean install.")
    print(f"  PASS all {len(_pyproject_requires())} declared runtime dependencies "
          f"ship in the wheel METADATA.")
    print(f"  PASS all {len(_source_imports())} third-party module-level imports are "
          f"declared, or listed as deliberately undeclared with a reason.")
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
