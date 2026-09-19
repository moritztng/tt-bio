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

What this file asks is "has the code that feeds this number changed?", and the answer is yes most
of the time. Measured on origin/main: 370 commits in 30 days touch model_gate's 28-file closure,
332 of them change executable source, and over the last week the median gap between two of them
is 15 minutes. 244 land in tt_bio/tenstorrent.py, the 13,059-line shared op layer. That is the
artifact's shelf life, not a defect here, and it is why the affordable place to re-record is the
release commit rather than every work branch.

The question a reader actually wants answered is "do the recorded numbers still reproduce?", and
that one needs a card, so a test cannot ask it. What this file can do is be exact about the
cheaper question: it reads the CONTENT of the closure rather than which commits came near it, and
it names the files that differ. A failure here means the numbers are unconfirmed, not wrong.

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


def _git(*args, cwd=REPO):
    """Git output, or None outside a work tree -- a release gate runs this suite on an export."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)
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


def _drifted(sha: str, paths, head="HEAD", cwd=REPO):
    """Which of *paths* differ from *sha*, and the commits that changed those files.

    Content, not commit adjacency. ``git log <sha>..HEAD -- <paths>`` also counts a merge that
    brings the recording's own branch in: its diff against its first parent touches those paths
    even though every line it moves was already at its recorded state. Composing main with
    train-q/k/i reported two commits for one real change and put that merge at the top of the
    list, which sends a reader to a commit that changed nothing. Asking `git diff` first and
    attributing only the files that actually differ drops it, and keeps a merge that really did
    carry content -- a conflict resolution, say -- because that merge is the only commit touching
    the file it changed.
    """
    changed = (_git("diff", "--name-only", sha, head, "--", *paths, cwd=cwd) or "").split()
    if not changed:
        return [], []
    log = _git("log", "--format=%h %s", f"{sha}..{head}", "--", *changed, cwd=cwd)
    return changed, [line for line in (log or "").splitlines() if line]


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
    changed, commits = _drifted(sha, _closure(REPO / script))
    assert not changed, (
        f"{artifact} was recorded at {sha}, and {len(changed)} file(s) that feed it differ from "
        f"what it was recorded against:\n  " + "\n  ".join(changed)
        + f"\nchanged by:\n  " + "\n  ".join(commits)
        + f"\nThe numbers are unconfirmed, not known wrong. Re-run {script} and update both the "
        f"numbers and the RECORDED-AT line; until then the file is a historical measurement "
        f"rather than evidence about this branch. Expect this on a work branch: this "
        f"closure takes ~19 code-changing commits a day, so a recording goes unconfirmed within "
        f"a median of 15 minutes and the affordable place to re-record is the release commit")


def test_the_staleness_check_can_fail():
    """The negative control, taken from live history rather than a pinned commit.

    Rewind one artifact's recorded commit to just before the last change to its own sources and
    the check must report that change. A pinned sha would decay into a skip the first time
    somebody rebased it away, and a control that skips is not a control.

    ``head`` is that commit rather than ``HEAD``: the check now reads content, and a later commit
    that put the file back would make a HEAD-anchored control pass while the mechanism it is
    guarding was broken.
    """
    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("not a git work tree: there is no history to compare against")
    artifact, _, script = next(a for a in _artifacts() if a[0] == REQUIRED[0])
    sources = _closure(REPO / script)
    last = _git("log", "-1", "--no-merges", "--format=%h", "--", *sources)
    assert last, f"no commit in this history touches any of {len(sources)} sources of {artifact}"
    changed, commits = _drifted(f"{last}^", sources, head=last)
    assert changed and any(line.startswith(last) for line in commits), (
        f"the check did not notice {last}, which touched {script}'s own sources. It reads "
        f"something other than what it claims to read, and every pass above is worth nothing")


def test_a_merge_that_carries_nothing_is_not_staleness(tmp_path):
    """The positive control: the check must be SEEN to pass across a change that cannot matter.

    Built rather than searched for, because the shape does not occur on ``main`` -- 115 merges in
    60 days, none of them log-counted with an empty content diff -- but occurs immediately on the
    composed trees workers actually build. ``q`` carries a recording; ``main`` moves ``feeder.py``
    under it and separately moves ``other.py`` away and back; the merge brings ``q`` in.

    ``recorded.py`` is byte-identical to its recorded state and ``other.py`` came back to it, so
    neither can move a number, yet ``git log`` counts the merge and the round trip for both. The
    real change to ``feeder.py`` must still be caught, and must be attributed to the commit that
    made it rather than to the merge.
    """
    repo = tmp_path / "r"
    repo.mkdir()

    def g(*args):
        out = _git(*args, cwd=repo)
        assert out is not None, f"git {' '.join(args)} failed"
        return out

    g("init", "-q", "-b", "main")
    g("config", "user.email", "t@t"), g("config", "user.name", "t")

    def commit(message, **files):
        for name, body in files.items():
            (repo / f"{name}.py").write_text(body)
        g("add", "-A")
        g("commit", "-q", "-m", message)
        return g("rev-parse", "--short", "HEAD")

    base = commit("base", feeder="F0", recorded="R0", other="O0")
    g("branch", "q", base)
    commit("another campaign moves the shared feeder", feeder="F1")
    commit("a change to other", other="O1")
    commit("and it comes back", other="O0")
    g("checkout", "-q", "q")
    recording = commit("the recording's own branch", recorded="R1")
    g("checkout", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "Merge q", "q")
    head = g("rev-parse", "--short", "HEAD")

    closure = ["feeder.py", "recorded.py", "other.py"]
    counted = g("log", "--format=%h", f"{recording}..{head}", "--", *closure).split()
    assert len(counted) >= 3, "the fixture must reproduce a history the commit-touch test misreads"

    unchanged = ["recorded.py", "other.py"]
    assert _drifted(recording, unchanged, head=head, cwd=repo) == ([], []), (
        f"the check called a merge and a round trip a change. git log counts {len(counted)} "
        f"commits here and not one of them leaves {unchanged} different from {recording}")

    changed, commits = _drifted(recording, closure, head=head, cwd=repo)
    assert changed == ["feeder.py"], f"the real change was missed or over-reported: {changed}"
    assert len(commits) == 1 and "shared feeder" in commits[0], (
        f"the real change must be attributed to the commit that made it, not to the merge that "
        f"brought the recording's branch in: {commits}")


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
