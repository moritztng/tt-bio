"""A recorded claim and the thing it describes, kept equal.

Two claims in this tree drifted from what they describe, and neither drift was visible from the
claim itself.

``perf/abb3_port/model_gate_qb1c3.txt`` was recorded at ``ca7a7573e``. Six later commits on the
same branch touched the model it scores, including the device-to-host convention the gate itself
calls, and the file kept reading 0.014 A max where the head reads 0.018 A. That difference reads
exactly like a hardware difference, and it was nearly filed as one -- p150a against p300c -- before
two cards agreed with each other and the file turned out to be the outlier.

``scripts/abb3_port/step_gate.py`` shipped ``--micro 8`` as its default and quoted the same value
in its run line, and it OOMs in the first backward. Every recorded run overrode it, so the default
was never the thing that was measured.

So: an artifact names the commit it was recorded at, and this file fails when the code that feeds
it has moved since. A docstring's run line quotes flag values, and this file fails when they are
not the defaults.

The sources that feed an artifact are NOT listed by hand. A hand list is the next thing to go
stale, and it goes stale silently in exactly the same way. They are the producing script plus the
transitive closure of its first-party imports, read out of the import graph, so a gate that grows
a dependency starts watching it on the same commit.

Run: python3 -m pytest tests/test_recorded_claims.py, or as part of the release suite.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: ``RECORDED-AT: <40-hex commit> <producing script, repo-relative>``. Any file under ``perf/``
#: carrying it is checked; these must carry it, so that stripping the stanza or deleting the file
#: fails rather than quietly removing the check.
REQUIRED = ("perf/abb3_port/model_gate_qb1c1.txt",
            "perf/abb3_port/step_gate_default_qb1c1.txt")

STANZA = re.compile(r"^RECORDED-AT:\s*([0-9a-f]{7,40})\s+(\S+)\s*$", re.M)


def _git(*args):
    """Git output, or None outside a work tree -- a release gate runs this suite on an export."""
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.decode().strip()


def _resolve(module: str, level: int, path: Path):
    """Repo files an import in *path* can mean: a package module, or a sibling of the script.

    The sibling case is not exotic here -- every script in ``scripts/abb3_port/`` puts its own
    directory on ``sys.path`` and imports its neighbours by bare name, and those neighbours are
    exactly the shared instruments whose movement invalidates a recording. Relative imports are
    resolved too, because that is how everything inside ``tt_bio`` imports everything else.
    """
    parts = module.split(".") if module else []
    base = path.parents[level - 1] if level else REPO
    for candidate in (base.joinpath(*parts).with_suffix(".py"),
                      base.joinpath(*parts, "__init__.py"),
                      path.parent / f"{parts[0]}.py" if parts and not level else None):
        if candidate is not None and candidate.is_file() and candidate.is_relative_to(REPO):
            yield candidate


def _imports(path: Path):
    """Module-level imports only, as ``(module, level, names)``.

    Not ``ast.walk``: a function-local import is a deliberately lazy one and following it makes
    the closure the whole package. ``tt_bio/size_limits.py`` imports ``tt_bio.main`` inside two
    functions, and through it the CLI reaches every model in the tree -- 259 files instead of the
    30 that can actually move a number here.
    """
    body = list(ast.parse(path.read_text()).body)
    while body:
        node = body.pop()
        if isinstance(node, (ast.If, ast.Try)):
            body.extend(node.body + node.orelse + getattr(node, "finalbody", [])
                        + [n for h in getattr(node, "handlers", []) for n in h.body])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield (f"{node.module}.{alias.name}" if node.module else alias.name), node.level
            if node.module:
                yield node.module, node.level


def _closure(script: Path) -> list:
    """The script and every first-party module reachable from its imports, repo-relative."""
    seen, queue = set(), [script]
    while queue:
        path = queue.pop().resolve()
        rel = str(path.relative_to(REPO))
        if rel in seen:
            continue
        seen.add(rel)
        if path.name == "__init__.py":
            # Watched, not traversed: a package init re-exports its whole package, so following
            # one makes an artifact depend on models that cannot reach its number.
            continue
        for module, level in _imports(path):
            queue.extend(_resolve(module, level, path))
    return sorted(seen)


def _moved_since(sha: str, paths) -> list:
    """Commits after *sha* that touched *paths*, newest first."""
    log = _git("log", "--format=%h %s", f"{sha}..HEAD", "--", *paths)
    return [line for line in (log or "").splitlines() if line]


def _artifacts():
    """Every file under ``perf/`` carrying the stanza, with the commit and script it names."""
    found = []
    for path in sorted((REPO / "perf").rglob("*.txt")):
        match = STANZA.search(path.read_text(errors="ignore"))
        if match:
            found.append((str(path.relative_to(REPO)), match.group(1), match.group(2)))
    return found


def test_required_artifacts_carry_the_stanza():
    """Refreshing an artifact is the fix; deleting it or its stanza is the check removed."""
    carried = {name for name, _, _ in _artifacts()}
    assert set(REQUIRED) <= carried, (
        f"{sorted(set(REQUIRED) - carried)} must carry a `RECORDED-AT: <commit> <script>` line. "
        f"A recorded number without the commit it was recorded at cannot be told from a stale one")


@pytest.mark.parametrize("artifact,sha,script", _artifacts(),
                         ids=[a for a, _, _ in _artifacts()])
def test_recorded_artifact_is_not_stale(artifact, sha, script):
    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("not a git work tree: there is no history to compare against")
    assert (REPO / script).is_file(), f"{artifact} names a producing script that is gone: {script}"
    assert _git("cat-file", "-e", f"{sha}^{{commit}}") is not None, (
        f"{artifact} names commit {sha}, which is not in this repository")
    assert _git("merge-base", "--is-ancestor", sha, "HEAD") is not None, (
        f"{artifact} was recorded at {sha}, which is not an ancestor of HEAD. The numbers in it "
        f"were produced by code this branch does not contain")
    moved = _moved_since(sha, _closure(REPO / script))
    assert not moved, (
        f"{artifact} was recorded at {sha} and {len(moved)} commit(s) have touched the code that "
        f"feeds it since:\n  " + "\n  ".join(moved)
        + f"\nRe-run {script} and update both the numbers and the RECORDED-AT line. The file is "
        f"not evidence about this branch until you do")


def test_the_staleness_check_can_fail():
    """The negative control, taken from live history rather than a pinned commit.

    Rewind one artifact's recorded commit to just before the last change to its own sources and
    the check must report that change. A pinned sha would decay into a skip the first time
    somebody rebased it away, and a control that skips is not a control.
    """
    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("not a git work tree: there is no history to compare against")
    artifact, _, script = next(a for a in _artifacts() if a[0] == REQUIRED[0])
    sources = _closure(REPO / script)
    last = _git("log", "-1", "--format=%h", "--", *sources)
    assert last, f"no commit in this history touches any of {len(sources)} sources of {artifact}"
    moved = _moved_since(f"{last}^", sources)
    assert any(line.startswith(last) for line in moved), (
        f"the check did not notice {last}, which touched {script}'s own sources. It reads "
        f"something other than what it claims to read, and every pass above is worth nothing")


#: ``[--flag value]`` in a docstring's run line.
QUOTED = re.compile(r"\[--([a-z0-9-]+)\s+([^\]\s]+)\]")


def _defaults(tree):
    """``ap.add_argument("--flag", ..., default=X)`` as written, without importing the script."""
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        flag = getattr(node.args[0], "value", None)
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                out[flag[2:]] = kw.value.value
    return out


SCRIPTS = sorted(p.name for p in (REPO / "scripts/abb3_port").glob("*.py"))


@pytest.mark.parametrize("name", SCRIPTS)
def test_run_line_quotes_the_shipped_defaults(name):
    """A run line is the command a reader will actually type, so it has to be the shipped one.

    ``step_gate.py`` quoted ``--micro 8`` in both its run line and its default and the value OOMs.
    Agreement between the two is not the same as either being right -- what makes it worth testing
    is that the default is now the configuration a recorded run was measured at, so a run line
    that disagrees with it sends a reader to a number nobody measured.
    """
    path = REPO / "scripts/abb3_port" / name
    tree = ast.parse(path.read_text())
    doc = ast.get_docstring(tree) or ""
    run = doc[doc.index("Run:"):] if "Run:" in doc else ""
    defaults = _defaults(tree)
    wrong = []
    for flag, quoted in QUOTED.findall(run):
        key = flag.replace("-", "_")
        if key not in defaults and flag not in defaults:
            continue  # a flag the run line names but argparse gives no default: nothing to agree
        shipped = defaults.get(key, defaults.get(flag))
        if quoted != str(shipped):
            wrong.append(f"--{flag}: run line says {quoted}, the default is {shipped}")
    assert not wrong, (f"{name}'s run line and its defaults disagree:\n  " + "\n  ".join(wrong))
