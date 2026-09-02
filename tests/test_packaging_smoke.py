"""What the packaging gate would still catch after a refactor, and what it would not.

`scripts/packaging_smoke.py` is the fix for the 0.3.3 incident (CHANGELOG 0.3.4): the
package-data globs shipped two LICENSEs and none of the 13 runtime data files, so every
clean `pip install tt-bio` crashed protenix-v2, opendde and boltzgen. It has guarded every
release since with no test of its own, and three of its checks are the shape that stops
meaning anything quietly.

Four defects found by writing these tests, each fixed in the script and pinned below.

The sdist membership test matched any member ENDING with the file, so a stray copy
anywhere in the tarball satisfied it; it now anchors on the sdist's single root directory.
The --fold leg took its three example YAMLs by path and called any failure that was not a
missing-data error a PASS, so renaming an example would have made all three folds die on
the YAML and all three print PASS having run no model at all; the inputs are checked first
now. That leg also paired "FileNotFoundError" anywhere in the output with "/data/" anywhere
in it, and every traceback frame of a packaged run sits under site-packages/tt_bio/, so a
user's missing input file read as a packaging failure; the markers are matched against the
FileNotFoundError line, which is where the missing path is. And `_check_dependencies`
compared pyproject against the wheel's own metadata, which by construction cannot see a
dependency nobody ever declared -- the script said so in its own docstring and left it. It
is checked now, from the source imports, and `test_live_tree_has_no_undeclared_imports`
holds the line.

Everything here is hermetic except the three tests that read the real tree on purpose
(named `live_`). No card, no wheel build, no network.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "packaging_smoke.py"

VERSION = "9.9.9"
DIST_INFO = f"tt_bio-{VERSION}.dist-info"
SDIST_ROOT = f"tt_bio-{VERSION}"


@pytest.fixture()
def mod():
    """A fresh module per test, so a monkeypatched REPO_ROOT cannot leak between them."""
    spec = importlib.util.spec_from_file_location("packaging_smoke", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def make_wheel(path: Path, files=(), requires=(), metadata=True) -> Path:
    """A minimal but real wheel: the named members plus valid dist-info."""
    whl = path / f"tt_bio-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        for name in files:
            z.writestr(name, "x")
        if metadata:
            meta = ["Metadata-Version: 2.1", "Name: tt-bio", f"Version: {VERSION}"]
            meta += [f"Requires-Dist: {r}" for r in requires]
            z.writestr(f"{DIST_INFO}/METADATA", "\n".join(meta) + "\n")
        z.writestr(f"{DIST_INFO}/WHEEL",
                   "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{DIST_INFO}/RECORD", "")
    return whl


def make_sdist(path: Path, files=(), root: str = SDIST_ROOT, extra=()) -> Path:
    """A tarball laid out the way setuptools writes one: everything under one root dir."""
    sd = path / f"tt_bio-{VERSION}.tar.gz"
    with tarfile.open(sd, "w:gz") as t:
        for name in [f"{root}/{f}" for f in files] + list(extra):
            data = b"x"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return sd


def fake_repo(root: Path, data_files=(), deps=(), sources=None) -> Path:
    """A repo tree with a pyproject and a tt_bio package, for REPO_ROOT."""
    pkg = root / "tt_bio"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for rel in data_files:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    for rel, text in (sources or {}).items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(text))
    body = "".join(f'    "{d}",\n' for d in deps)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "tt-bio"\nversion = "{VERSION}"\ndependencies = [\n{body}]\n')
    return root


# --------------------------------------------------------------------------- expected set

def test_expected_set_is_the_non_py_files(mod, tmp_path, monkeypatch):
    repo = fake_repo(tmp_path, data_files=["tt_bio/data/conformers.json",
                                           "tt_bio/boltzgen/resources/config/design.yaml"])
    (repo / "tt_bio/main.py").write_text("")
    (repo / "tt_bio/__pycache__").mkdir()
    (repo / "tt_bio/__pycache__/main.cpython-312.pyc").write_text("x")
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    assert mod._expected_data_files() == ["tt_bio/boltzgen/resources/config/design.yaml",
                                          "tt_bio/data/conformers.json"]


def test_expected_set_is_the_working_tree_not_git(mod, tmp_path, monkeypatch):
    """Known and deliberate: the set comes from the filesystem, so an untracked stray
    under tt_bio/ is required to ship too and fails the gate until it is cleaned up.

    Deriving it from `git ls-files` instead would drop the stray, but the script has to
    work against an unpacked sdist and a `git archive` export, where nothing is tracked
    and the fallback is this walk anyway. A loud failure naming the stray file is the
    better half of that trade, so this pins the behaviour rather than calling it a bug.
    """
    repo = fake_repo(tmp_path, data_files=["tt_bio/data/conformers.json"])
    (repo / "tt_bio/data/conformers.json.orig").write_text("x")
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    assert "tt_bio/data/conformers.json.orig" in mod._expected_data_files()


# --------------------------------------------------------------------------- artifacts

def test_file_in_both_artifacts_is_not_a_false_positive(mod, tmp_path):
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=[rel])
    sdist = make_sdist(tmp_path, files=[rel])
    assert mod._check_artifacts(whl, sdist, [rel]) == []


def test_data_file_dropped_from_wheel_is_caught(mod, tmp_path):
    """The 0.3.3 bug itself: the .py files ship, the data file does not."""
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=["tt_bio/__init__.py"])
    sdist = make_sdist(tmp_path, files=[rel])
    assert mod._check_artifacts(whl, sdist, [rel]) == [f"wheel missing: {rel}"]


def test_data_file_dropped_from_sdist_is_caught(mod, tmp_path):
    rel = "tt_bio/boltzgen/resources/config/design.yaml"
    whl = make_wheel(tmp_path, files=[rel])
    sdist = make_sdist(tmp_path, files=["tt_bio/__init__.py"])
    assert mod._check_artifacts(whl, sdist, [rel]) == [f"sdist missing: {rel}"]


def test_sdist_hit_must_be_under_the_root_not_anywhere(mod, tmp_path):
    """A copy elsewhere in the tarball does not count as shipping the file.

    The check used to be `any(n.endswith("/" + rel))`, which a docs copy of the same
    path satisfies while the package tree is missing it -- and a `pip install` of that
    sdist crashes exactly like 0.3.3.
    """
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=[rel])
    sdist = make_sdist(tmp_path, files=["tt_bio/__init__.py"],
                       extra=[f"{SDIST_ROOT}/docs/{rel}"])
    assert mod._check_artifacts(whl, sdist, [rel]) == [f"sdist missing: {rel}"]


def test_sdist_without_a_single_root_is_reported(mod, tmp_path):
    """No root means the layout is not what the membership test assumes, so it says so
    instead of quietly falling back to a check that proves less."""
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=[rel])
    sdist = make_sdist(tmp_path, files=[rel], extra=[f"other-1.0/{rel}"])
    failures = mod._check_artifacts(whl, sdist, [rel])
    assert any("no single root directory" in f for f in failures)


# --------------------------------------------------------------------------- dependencies

def test_declared_dependency_missing_from_metadata_is_caught(mod, tmp_path, monkeypatch):
    """The direction the script already covered: pyproject declares it, the wheel does not."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(tmp_path, deps=["torch", "rdkit"]))
    whl = make_wheel(tmp_path, requires=["torch"])
    assert mod._check_dependencies(whl) == ["wheel METADATA missing declared dependency: rdkit"]


def test_matching_dependencies_are_clean(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(tmp_path, deps=["torch", "rdkit"]))
    assert mod._check_dependencies(make_wheel(tmp_path, requires=["torch", "rdkit"])) == []


def test_dependency_names_normalize_across_spellings(mod, tmp_path, monkeypatch):
    """Version bounds, extras, markers and _/- spelling are not differences."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(
        tmp_path, deps=["biotite<1.7", "huggingface_hub>=1.5.0,<2.0", "Msgpack-Numpy"]))
    whl = make_wheel(tmp_path, requires=[
        "biotite (<1.7)", "huggingface-hub>=1.5.0,<2.0", 'msgpack_numpy ; python_version >= "3.10"'])
    assert mod._check_dependencies(whl) == []


def test_wheel_without_metadata_fails_loudly(mod, tmp_path, monkeypatch):
    """An unreadable wheel reports every declared dependency, not a silent pass."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(tmp_path, deps=["torch", "rdkit"]))
    whl = make_wheel(tmp_path, requires=["torch", "rdkit"], metadata=False)
    assert len(mod._check_dependencies(whl)) == 2


# ------------------------------------------------------------------- undeclared imports

DECLARED_SRC = {"tt_bio/main.py": "import torch\nimport tt_bio.data\n"}


def test_undeclared_import_is_caught(mod, tmp_path, monkeypatch):
    """The gap `_check_dependencies` documented and left open.

    pyproject and the wheel agree with each other and both omit the package, so the
    metadata comparison passes; only the source says `import scipy`. This is the rf3
    shape: zstandard, beartype, jaxtyping and pyarrow were each found by a fold crashing.
    """
    repo = fake_repo(tmp_path, deps=["torch"],
                     sources={"tt_bio/main.py": "import torch\nimport scipy\n"})
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    whl = make_wheel(tmp_path, requires=["torch"])
    assert mod._check_dependencies(whl) == []          # metadata is internally consistent
    failures = mod._check_undeclared_imports()
    assert len(failures) == 1
    assert "scipy" in failures[0] and "tt_bio/main.py" in failures[0]


def test_declared_imports_are_clean(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REPO_ROOT",
                        fake_repo(tmp_path, deps=["torch"], sources=DECLARED_SRC))
    assert mod._check_undeclared_imports() == []


def test_optional_and_lazy_imports_are_not_flagged(mod, tmp_path, monkeypatch):
    """An import inside a try, an if, a function or a class has a fallback or never runs
    at import time, so its absence is not the crash this guards against. Vendored trees
    are full of them and flagging those would make the gate cry wolf on every release."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(tmp_path, deps=["torch"], sources={
        "tt_bio/opt.py": """
            import torch

            try:
                import flash_attn
            except ImportError:
                flash_attn = None

            if False:
                import deepspeed

            def plot():
                import logomaker
                return logomaker

            class C:
                import wandb
        """}))
    assert mod._check_undeclared_imports() == []


def test_stdlib_relative_and_in_tree_imports_are_not_flagged(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(tmp_path, deps=[], sources={
        "tt_bio/data/__init__.py": "",
        "tt_bio/data/parse.py": "import json\nfrom pathlib import Path\n",
        "tt_bio/main.py": "import data\nfrom tt_bio.data import parse\nfrom . import data\n"}))
    assert mod._check_undeclared_imports() == []


def test_import_name_maps_to_its_distribution(mod, tmp_path, monkeypatch):
    """`import yaml` is satisfied by pyyaml, `from Bio import` by biopython."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(
        tmp_path, deps=["pyyaml", "biopython", "scikit-learn"],
        sources={"tt_bio/main.py": "import yaml\nfrom Bio import SeqIO\nimport sklearn\n"}))
    assert mod._check_undeclared_imports() == []


def test_deliberately_undeclared_import_needs_a_written_reason(mod, tmp_path, monkeypatch):
    """ttnn is installed from Tenstorrent's index, not PyPI, so it is never declared.
    The escape hatch is a table entry carrying the reason, not a silent skip."""
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo(
        tmp_path, deps=[], sources={"tt_bio/main.py": "import ttnn\n"}))
    assert mod._check_undeclared_imports() == []
    assert all(reason.strip() for reason in mod._UNDECLARED_OK.values())


def test_live_tree_has_no_undeclared_imports(mod):
    """The ratchet. A new third-party import in tt_bio/ fails the release gate until it
    is declared in pyproject or written into _UNDECLARED_OK with why it need not be."""
    assert mod._check_undeclared_imports() == []


def test_live_undeclared_allowlist_has_no_dead_entries(mod):
    """An entry that no longer matches any import is a stale exemption, and a stale
    exemption is how a real one gets waved through later."""
    imported = set(mod._source_imports())
    assert set(mod._UNDECLARED_OK) <= imported


# --------------------------------------------------------------------------- fold parsing

INSTALLED = "/tmp/venv/lib/python3.12/site-packages"
MISSING_CONFORMERS = f"""\
Traceback (most recent call last):
  File "{INSTALLED}/tt_bio/main.py", line 660, in predict
    feats = featurize(target)
  File "{INSTALLED}/tt_bio/data/featurizer.py", line 88, in featurize
    ref = json.loads(Path(REF).read_text())
FileNotFoundError: [Errno 2] No such file or directory: '{INSTALLED}/tt_bio/data/protein_ref_conformers.json'
"""
MISSING_DESIGN_YAML = f"""\
Traceback (most recent call last):
  File "{INSTALLED}/tt_bio/boltzgen/cli/boltzgen.py", line 210, in _configure
    cfg = OmegaConf.load(path)
FileNotFoundError: [Errno 2] No such file or directory: '{INSTALLED}/tt_bio/boltzgen/resources/config/design.yaml'
"""


@pytest.mark.parametrize("out", [MISSING_CONFORMERS, MISSING_DESIGN_YAML])
def test_incident_signature_is_detected(mod, out):
    assert mod._is_missing_data_error(out)


def test_successful_fold_is_not_flagged(mod):
    out = "predicting trpcage...\nwrote predictions/trpcage_model_0.cif\ndone in 41.2s\n"
    assert not mod._is_missing_data_error(out)


def test_unrelated_failure_is_not_flagged(mod):
    """A wedged card or an OOM is not a packaging failure, and the leg says so."""
    out = ("RuntimeError: device 0 did not respond within 300s\n"
           f'  File "{INSTALLED}/tt_bio/tenstorrent.py", line 44, in get_device\n')
    assert not mod._is_missing_data_error(out)


def test_missing_user_input_is_not_read_as_a_packaging_failure(mod):
    """The false positive the old whole-output match produced. Every frame of a packaged
    run sits under site-packages/tt_bio/, so pairing "FileNotFoundError" anywhere with
    "/data/" anywhere flagged a user's own missing file as a dropped package data file."""
    out = f"""\
Traceback (most recent call last):
  File "{INSTALLED}/tt_bio/main.py", line 640, in predict
    target = parse_yaml(path)
  File "{INSTALLED}/tt_bio/data/parse.py", line 401, in parse_yaml
    text = path.read_text()
FileNotFoundError: [Errno 2] No such file or directory: '/home/me/data/my_protein.yaml'
"""
    assert not mod._is_missing_data_error(out)


def test_live_markers_still_name_paths_that_exist(mod):
    """If a data file moves, the marker that names it stops matching and the leg goes
    quietly green on a fold that never touches the moved file. This is the tripwire."""
    for marker in mod._MISSING_DATA_MARKERS:
        if marker == "/tt_bio/":       # the catch-all, not a path
            continue
        assert (REPO / marker.rstrip("/")).exists(), f"{marker} names nothing in the repo"


def test_live_fold_inputs_exist(mod):
    """Renaming one of these three examples would make its fold die on the YAML, which
    is not a missing-data error, so the leg would print PASS having run no model."""
    for name in ("trpcage_no_msa.yaml", "opendde_covalent_bond.yaml", "binder.yaml"):
        assert (REPO / "examples" / name).exists()


def test_fold_check_refuses_to_run_without_its_inputs(mod, tmp_path, monkeypatch, capsys):
    """The same defect from the other side: with the examples gone the leg fails and
    names them, instead of running three folds that prove nothing."""
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    rc = mod._fold_check(tmp_path / "nonexistent.whl")
    err = capsys.readouterr().err
    assert rc == 3
    assert "FAIL" in err and "trpcage_no_msa.yaml" in err
    assert "PASS" not in err


# --------------------------------------------------------------------------- install leg

@pytest.mark.skipif(importlib.util.find_spec("pip") is None, reason="needs pip")
def test_install_leg_sees_the_files_that_land_on_disk(mod, tmp_path):
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=["tt_bio/__init__.py", rel])
    assert mod._check_install(whl, [rel]) == []


@pytest.mark.skipif(importlib.util.find_spec("pip") is None, reason="needs pip")
def test_install_leg_catches_a_file_that_never_lands(mod, tmp_path):
    """The check that would have caught 0.3.3 on the installed tree, not just the archive."""
    rel = "tt_bio/data/conformers.json"
    whl = make_wheel(tmp_path, files=["tt_bio/__init__.py"])
    assert mod._check_install(whl, [rel]) == [f"installed missing: {rel}"]


# --------------------------------------------------------------------------- build + main

def test_build_exits_when_it_produces_nothing(mod, tmp_path, monkeypatch):
    """A build that silently produced no artifacts must not leave the later checks
    reading a stale dist/ from the last release."""
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)
    stale = tmp_path / "dist"
    stale.mkdir()
    (stale / "tt_bio-0.0.1-py3-none-any.whl").write_text("stale")
    with pytest.raises(SystemExit):
        mod._build()
    assert not (stale / "tt_bio-0.0.1-py3-none-any.whl").exists()


def _wire_main(mod, monkeypatch, repo, whl, sdist, install_failures=()):
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    monkeypatch.setattr(mod, "_build", lambda: (whl, sdist))
    monkeypatch.setattr(mod, "_check_install", lambda w, e: list(install_failures))
    monkeypatch.setattr(sys, "argv", ["packaging_smoke.py"])


def test_gate_passes_on_a_clean_tree(mod, tmp_path, monkeypatch, capsys):
    rel = "tt_bio/data/conformers.json"
    repo = fake_repo(tmp_path / "repo", data_files=[rel], deps=["torch"], sources=DECLARED_SRC)
    whl = make_wheel(tmp_path, files=[rel], requires=["torch"])
    _wire_main(mod, monkeypatch, repo, whl, make_sdist(tmp_path, files=[rel]))
    assert mod.main() == 0
    assert "GATE PASS" in capsys.readouterr().out


def test_gate_fails_on_a_dropped_data_file(mod, tmp_path, monkeypatch, capsys):
    rel = "tt_bio/data/conformers.json"
    repo = fake_repo(tmp_path / "repo", data_files=[rel], deps=["torch"], sources=DECLARED_SRC)
    whl = make_wheel(tmp_path, files=["tt_bio/__init__.py"], requires=["torch"])
    _wire_main(mod, monkeypatch, repo, whl, make_sdist(tmp_path, files=[rel]))
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "GATE FAIL" in out and f"wheel missing: {rel}" in out and "GATE PASS" not in out


def test_gate_fails_on_a_dropped_dependency(mod, tmp_path, monkeypatch, capsys):
    """Every data file ships and the install is clean; only the metadata is wrong."""
    rel = "tt_bio/data/conformers.json"
    repo = fake_repo(tmp_path / "repo", data_files=[rel], deps=["torch", "rdkit"],
                     sources={"tt_bio/main.py": "import torch\nimport rdkit\n"})
    whl = make_wheel(tmp_path, files=[rel], requires=["torch"])
    _wire_main(mod, monkeypatch, repo, whl, make_sdist(tmp_path, files=[rel]))
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "declared dependency: rdkit" in out and "GATE PASS" not in out


def test_gate_fails_on_an_undeclared_import(mod, tmp_path, monkeypatch, capsys):
    """Artifacts, install and metadata all clean; the source imports something nobody
    declared. Before this leg existed the gate passed and the install crashed."""
    rel = "tt_bio/data/conformers.json"
    repo = fake_repo(tmp_path / "repo", data_files=[rel], deps=["torch"],
                     sources={"tt_bio/main.py": "import torch\nimport scipy\n"})
    whl = make_wheel(tmp_path, files=[rel], requires=["torch"])
    _wire_main(mod, monkeypatch, repo, whl, make_sdist(tmp_path, files=[rel]))
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "undeclared dependency" in out and "scipy" in out and "GATE PASS" not in out
