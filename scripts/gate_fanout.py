#!/usr/bin/env python3
"""Run the release gate as independent legs across many cards and both architectures, one verdict.

The gate scripts (full_parity_gate.py, release_gate.py, capacity_gate.py, ux_regression.py,
perf_regression.py, pytest) are unchanged. This runner splits them into legs that do not depend on
each other, runs every leg once per architecture on whatever granted card of that architecture is
free, and writes one verdict. The parity gate alone used to take 7-8 h on one card because its 44
legs run one after another; spread over N cards the gate takes about as long as its longest leg.

    PYTHONPATH="$PWD" python3 scripts/gate_fanout.py --sha <commit> --hosts hosts.json \\
        --workers qb1:0,qb1:1,qb1:3,g114:5,g114:6 --timed qb1:3,g114:6 --out ~/gate-<sha9>

Legs (enumerated from the tree under test, so a new arm or model is picked up without editing here):
  parity:<leg>       full_parity_gate.py --leg <leg>                     every non-opt-in leg
  rg:<arm>           release_gate.py --model <arm>                       every arm but size-ladder
  ladder:<model>     release_gate.py --model size-ladder --size-ladder-models <model>
  capacity:<model>   capacity_gate.py --models <model>
  ux:<model>         ux_regression.py --model <model>
  pytest_device:i/K  pytest over every K-th test file, starting at i
  bc2:boundary       two BindCraft 2 trajectories from the installed wheel
  perf               perf_regression.py, TIMED (see below)
plus three card-free legs run once on the first host: check, packaging_smoke, pytest_cpu.
A ladder leg whose model's baseline has no row for a lever its fold resolves (owed_levers) is not
folded: it reads OWED, failing the verdict, which names the `--record-lever` run that clears it.
  record:<model>     release_gate.py --size-ladder-record-lever <FLAGS>  only with --record-lever
                     release_gate.py --size-ladder-record                 only with --record-full

TIMED legs measure speed, so they must not share a host with this gate's own load. They run only
on a --timed card, after every correctness leg of that arch has been handed out, and while one runs
no other leg of this gate starts on that host (running ones finish first). Everything else is a correctness leg: an independent fold whose answer does not
depend on what the next card is doing, which is why they can run side by side.

REUSE. Every leg result is keyed by
    sha256(content of every tracked file except *.md, pyproject's version line, recorded
           measurements and gate baselines, interpreter's installed distributions except
           tt-bio, card type, leg name, leg argv, and the baseline files the leg compares
           against: docs/size_ladder_baseline* for ladder legs, docs/perf_baselines.json for
           perf, docs/capacity_gate_baseline.json for capacity; for the pytest legs, the
           tests/test_*.py files no Python file outside tests/ loads)
and written to --ledger. A leg whose key already holds a PASS is not run again; the verdict names
the evidence it reused (its log, host, card, date). So a crossmodel or suite run done through this
runner during grading counts toward the release on the same code, and a docs-only commit on top of
a gated tree does not repeat the gate, and a fix to a test file reruns the pytest legs only. Any
change to code, scripts, data, fixtures, the interpreter's packages or the card type is a different
key and runs fresh.

hosts.json names, per host, how to reach it and what to run with (no host facts live in this file):
    {"qb1": {"ssh": "qb1", "arch": "bh", "card_type": "p150a", "root": "/home/ttuser/gate",
             "lock": "/home/ttuser/spd_qb1_card{card}.lock",
             "prep_env": {"GATE_PYTHON": "/usr/bin/python3", "BC2": "/home/ttuser/bcx_e2e/bc2"},
             "env": {"ESM_ROOT": "/home/ttuser/esm", "AF2_PARAMS_DIR": "...", ...}}}
Under `root` the runner keeps a clone (root/repo, which must exist), one tree per commit
(root/trees/<sha12>) and the interpreters scripts/gate_host_prep.sh builds from that commit's own
wheel: root/venv-<sha12> for every leg, root/venv312-<sha12> (with BindCraft 2) for pytest and the
BindCraft 2 leg. `python` / `python_for` override those. Every `env` value is exported to the leg
and also fills `{NAME}` in a leg's argv. `args` adds flags to one leg family on that host, and they
are part of the leg's key.

POOL. A host shared through a chip pool (a queue directory a pool runner drains onto idle healthy
chips, starting each job as `CHIP=<id> bash <job>`) names it, `"pool": {"queue": "<dir>", "prio": 5,
"row": "<holder>"}`, and is granted as `host:pool`, once per leg it may run at a time (`g114:pool`
four times = at most four legs queued or running there). Each leg becomes one job file; the pool
picks the chip, the leg still takes that chip's flock, and the runner waits for the job's exit code.

FIRST. A release never waits on our own experiments. A pool host queues its legs at the prio
it names (0, ahead of every grading job), and a flock host shared with other rows sets
`"first": true`: each leg then takes its card through scripts/flock_first.sh, which stops the
other waiters on that card's lock (never the holder) until the gate is done with the card.

SEEDING. perf_regression.py fails NO BASELINE on a card type with no baseline. That stays a
failure. A host seeding one names it explicitly, `"args": {"perf": ["--update-baseline", "--note",
"<why>"]}`: the leg then reports SEEDED, not PASS (there was nothing to regress against), and the
runner copies the seeded docs/perf_baselines.json to <out>/seeded/<arch>/ to commit. Drop the
`args` entry once it is committed, and the next gate on that card type is a regression check.
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import functools
import hashlib
import inspect
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Per-leg budgets (s). A budget is a hang guard, not an expected time.
BUDGET = {"parity": 14400, "rg": 7200, "ladder": 14400, "capacity": 14400, "ux": 3600,
          "pytest_device": 7200, "perf": 7200, "check": 900, "packaging_smoke": 1800,
          "pytest_cpu": 5400, "bc2": 3600, "record": 14400}
# Expected wall (s) per family when the ledger has no history for a leg, from the 0.13.1 and
# 0.14.0 chains. Legs start longest first so the gate ends close to its longest leg.
EXPECT = {"parity": 900, "rg": 900, "ladder": 5400, "capacity": 1800, "ux": 300,
          "pytest_device": 3000, "perf": 3600, "check": 60, "packaging_smoke": 600,
          "pytest_cpu": 1800, "bc2": 1200, "record": 2700}
TIMED = {"perf"}
#: Results that do not fail the gate. REUSED is a ledger hit of one of the others.
OK = ("PASS", "REUSED", "BLOCKED", "SEEDED")
LOG_CAP = 20 << 20
POOL = "pool"            # --workers host:pool, a slot in the host's chip pool
POOL_CARD = "@@CARD@@"   # stands for $CHIP in a pool job, filled in when the pool starts it
POOL_POLL_S = 30
LOAD_REFUSAL = "refusing to run the gate. host load:"
LOAD_RETRY_S, LOAD_WAIT_MAX_S = 300, 8 * 3600
QUEUED: dict = {}         # pool job file -> host, taken back if the runner is stopped
CARD_FREE = ("check", "packaging_smoke", "pytest_cpu")
PY312 = {"pytest_device", "pytest_cpu", "bc2"}
# Arms the parity gate already runs in-process by calling release_gate's own runner with the
# same arguments (full_parity_gate.py: run_inprocess -> release_gate.run_<arm>). Running the rg
# arm as well folds the same thing twice.
PARITY_COVERS_RG = {"boltzgen", "opendde-abag", "capacity", "nesso1", "rf3-1024aa"}

def owed_by(roster: dict, card_type: str, lg) -> list[str]:
    """Levers a ladder leg's baseline lacks on this card type; every other leg owes none."""
    if lg.family != "ladder":
        return []
    return roster.get("owed", {}).get(card_type, {}).get(lg.name.split(":", 1)[1], [])


def owed_levers(docs, levers, models) -> dict:
    """{card_type: {model: [flag, ...]}}: census levers a ladder model's fold will resolve that its
    size-ladder baseline has no row for, so its ladder leg fails "new lever not in the baseline" at
    every rung after folding them all. A lever's module is imported in a model's fold exactly when
    another lever of that module resolved there. Runs inside the tree under test (ENUMERATE)."""
    import json
    from pathlib import Path
    docs, module = Path(docs), {flag: mod for flag, mod, *_ in levers}
    entries = {}
    for f in [docs / "size_ladder_baseline.json", *sorted((docs / "size_ladder_baseline.d").glob("*.json"))]:
        if f.exists():
            for card, c in json.loads(f.read_text()).get("cards", {}).items():
                entries.update({(card, m): e for m, e in c.get("models", {}).items()})
    out = {}
    for (card, model), e in sorted(entries.items()):
        if model not in models:
            continue
        new = set()
        for rows in (e.get("levers") or {}).values():
            imported = {module.get(f) for f, r in rows.items() if r.get("resolved") != "not-imported"}
            new |= {f for f, mod in module.items() if f not in rows and mod in imported}
        if new:
            out.setdefault(card, {})[model] = sorted(new)
    return out


ENUMERATE = inspect.getsource(owed_levers) + r"""
import json, full_parity_gate as f, release_gate as r, capacity_gate as c, ux_regression as u
import lever_census
print(json.dumps({"parity": [l.id for l in f.LEGS if not l.opt_in], "rg": r.default_arms(),
                  "ladder": list(r.SIZE_LADDER_MODELS), "capacity": c.roster(), "ux": u.ALL_LEGS,
                  "owed": owed_levers("docs", lever_census.LEVERS, r.SIZE_LADDER_MODELS)}))
"""
ENV_PROBE = r"""
import json, sys, importlib.metadata as m
d = sorted({f"{x.metadata['Name'].lower()}=={x.version}" for x in m.distributions()
            if x.metadata['Name'] and x.metadata['Name'].lower() not in ("tt-bio", "tt_bio")})
print(json.dumps({"python": sys.version.split()[0], "dists": d}))
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------------------------
# The baselines a leg family compares against. They are measurements of the code, re-recorded
# when a lever lands, so they leave the content key and enter only their own family's leg key:
# re-recording the size ladder reruns the ladder legs, not the other 230, and splicing one
# model's fragment reruns that model's ladder leg only.
BASELINES = {"ladder": ("docs/size_ladder_baseline.json", "docs/size_ladder_baseline.d/"),
             "perf": ("docs/perf_baselines.json",),
             "capacity": ("docs/capacity_gate_baseline.json",)}
_BASELINE_PATHS = tuple(p for ps in BASELINES.values() for p in ps)


def _commit(sha: str, repo: Path) -> str:
    """The commit id `sha` names, so a cache keyed on it survives a moving ref like HEAD."""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{sha}^{{commit}}"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _ls_tree(sha: str, repo: Path) -> list[str]:
    return _ls_tree_at(_commit(sha, repo), repo)


@functools.lru_cache(maxsize=None)
def _ls_tree_at(sha: str, repo: Path) -> list[str]:
    return subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--full-tree", sha], check=True,
                          capture_output=True, text=True).stdout.splitlines()


def _same_code(sha: str, content: str, repo: Path = REPO) -> bool:
    """True when `sha` is in this repo and holds the code `content` hashes."""
    try:
        return content_hash(sha, repo) == content
    except subprocess.CalledProcessError:
        return False


def baseline_hash(sha: str, family: str, repo: Path = REPO, model: str = "") -> str:
    """Blob ids of the baseline files `family` compares against at `sha` ('' for none).

    A ladder leg for `model` reads the monolith plus that model's own fragment, so splicing one
    model's rows reruns that model's ladder leg only."""
    ps = BASELINES.get(family)
    if not ps:
        return ""
    if family == "ladder" and model:
        ps = (ps[0], f"{ps[1]}{model}.json")
    rows = [ln for ln in _ls_tree(sha, repo) if ln.split("\t", 1)[-1] in ps or (
        not model and ln.split("\t", 1)[-1].startswith(ps))]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


#: The leg families that run the test suite, and so read the test files the content key leaves out.
PYTEST = ("pytest_cpu", "pytest_device")


def test_only(sha: str, repo: Path = REPO) -> frozenset:
    return _test_only_at(_commit(sha, repo), repo)


@functools.lru_cache(maxsize=None)
def _test_only_at(sha: str, repo: Path) -> frozenset:
    """tests/test_*.py that only pytest reads: no Python file outside tests/ names them in code.
    Every leg but the pytest ones runs Python, so a gate script that loads a test (release_gate
    imports tests/test_structure.py by path) names it, and that file stays in the content key
    with the code that runs it. Only string literals and imports count: a comment or docstring
    pointing at the test that covers a function does not load it, and most of tt_bio's do."""
    stems = {Path(t).stem: t for t in test_files(sha, repo)}
    if not stems:
        return frozenset()
    git = ["git", "-C", str(repo)]
    files = subprocess.run([*git, "grep", "-l", "-w", "-F", *[x for t in stems for x in ("-e", t)], sha,
                            "--", "*.py", ":(exclude)tests/"],
                           capture_output=True, text=True).stdout.splitlines()
    word = re.compile(r"\b(" + "|".join(map(re.escape, stems)) + r")\b")
    named = set()
    for f in files:
        text = subprocess.run([*git, "show", f], capture_output=True, text=True).stdout
        named |= set(word.findall("\n".join(_code_strings(text))))
    return frozenset(t for stem, t in stems.items() if stem not in named)


def _code_strings(src: str) -> list:
    """String literals and imported module names of a Python file, docstrings and comments left
    out. Unparseable source is returned whole, which can only keep a test in the content key."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [src]
    docs = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            out.append(n.value)
        elif isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out += [n.module or ""] + [a.name for a in n.names]
    return out


def tests_hash(sha: str, family: str, repo: Path = REPO) -> str:
    """Blob ids of the test-only files at `sha` for a pytest leg ('' for any other family), so a
    test fix reruns the pytest legs and not the folds."""
    if family not in PYTEST:
        return ""
    only = test_only(sha, repo)
    rows = [ln for ln in _ls_tree(sha, repo) if ln.split("\t", 1)[-1] in only]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


RUNNER_FILES = ("scripts/gate_fanout.py", "scripts/release_next.py", "scripts/splice_ladder_fragments.py",
                "scripts/gate_host_prep.sh",
                "tests/test_gate_fanout.py", "tests/test_release_next.py")


def content_hash(sha: str, repo: Path = REPO, baselines: bool = False, tests: bool = False) -> str:
    """Hash of every tracked file at `sha` except Markdown, from git's own blob ids.

    pyproject.toml is hashed without its `version =` line: the release commit bumps it, and that
    alone must not throw away evidence gathered on the identical code the line before. Recorded
    measurements (perf/**.txt carrying a `RECORDED-AT:` line, tests/test_recorded_claims.py) are
    left out for the same reason: the release commit re-records them, and they are numbers about
    the code, not code. The one test that reads them is in pytest_cpu, which is never reused.
    Gate baselines (BASELINES) are left out too and keyed per leg family instead;
    `baselines=True` keeps them, the key every ledger row before 2026-10-09 15Z was written under.
    So is this runner (RUNNER_FILES): it decides where a leg runs, not what the leg computes, and
    a fix to it must not throw away a gate's worth of evidence. Test files only pytest reads
    (test_only) are keyed on the pytest legs instead; `tests=True` keeps them, the key every
    ledger row before 2026-10-09 21Z was written under."""
    def git(*a, ok=(0,)):
        p = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
        if p.returncode not in ok:
            raise subprocess.CalledProcessError(p.returncode, a, p.stdout, p.stderr)
        return p.stdout
    recorded = {ln.split(":", 1)[1] for ln in git("grep", "-l", "^RECORDED-AT:", sha, "--", "perf/*.txt",
                                                  ok=(0, 1)).splitlines()}
    only = set() if tests else test_only(sha, repo)
    rows = []
    for ln in _ls_tree(sha, repo):
        path = ln.split("\t", 1)[-1]
        if (path.endswith(".md") or path in recorded or path in RUNNER_FILES or path in only
                or (not baselines and path.startswith(_BASELINE_PATHS))):
            continue
        if path == "pyproject.toml":
            body = "".join(x for x in git("show", f"{sha}:pyproject.toml").splitlines(True)
                           if not x.startswith("version ="))
            ln = f"pyproject.toml {hashlib.sha256(body.encode()).hexdigest()}"
        rows.append(ln)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def env_hash(probe: dict) -> str:
    return sha256(probe["python"], probe["dists"])


def leg_key(content: str, env: str, card_type: str, leg: "Leg", *inputs: str) -> str:
    """`inputs` are the leg family's own files outside the content key (baselines, test files);
    an empty one drops out, so a family that reads none keeps the key it always had."""
    return sha256(content, env, card_type, leg.name, leg.argv, *[x for x in inputs if x])


# ---------------------------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------------------------
@dataclass
class Leg:
    name: str
    argv: list            # run in the tree, argv[0] "PY" is the host's interpreter
    family: str
    timed: bool = False
    card: bool = True     # opens a card
    workdir: bool = False  # argv takes the leg's own out dir as "{OUT}"
    tree_on_path: bool = True  # False: import tt_bio from the installed wheel, not the tree
    setup: str = ""        # shell run in the tree before argv, with the same {OUT} substitution

    @property
    def budget(self) -> int:
        return BUDGET[self.family]


def build_legs(roster: dict, test_files: list, shards: int, record_lever: str = "",
               record_full: str = "") -> list:
    legs = [Leg("check", ["PY", "scripts/full_parity_gate.py", "--check", "--workdir", "{OUT}",
                         "--workers", "localhost:0"], "check",
                card=False),
            Leg("packaging_smoke", ["PY", "scripts/packaging_smoke.py"], "packaging_smoke", card=False),
            Leg("pytest_cpu", ["PY", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short"],
                "pytest_cpu", card=False)]
    for lid in roster["parity"]:
        legs.append(Leg(f"parity:{lid}", ["PY", "scripts/full_parity_gate.py", "--leg", lid,
                                          "--workers", "localhost:{CARD}", "--workdir", "{OUT}",
                                          "--out", "{OUT}/report.json"], "parity", workdir=True))
    for arm in roster["rg"]:
        if arm == "size-ladder" or arm in PARITY_COVERS_RG:
            continue
        legs.append(Leg(f"rg:{arm}", ["PY", "scripts/release_gate.py", "--model", arm], "rg"))
    for m in roster["ladder"]:
        legs.append(Leg(f"ladder:{m}", ["PY", "scripts/release_gate.py", "--model", "size-ladder",
                                        "--size-ladder-models", m], "ladder"))
    for m in roster["capacity"]:
        legs.append(Leg(f"capacity:{m}", ["PY", "scripts/capacity_gate.py", "--models", m,
                                          "--workers", "localhost:{CARD}", "--no-card-reset",
                                          "--work-dir", "{OUT}", "--report", "{OUT}/report.json"],
                        "capacity", workdir=True))
    for m in roster["ux"]:
        legs.append(Leg(f"ux:{m}", ["PY", "scripts/ux_regression.py", "--model", m], "ux"))
    for i in range(shards):
        part = test_files[i::shards]
        if part:
            legs.append(Leg(f"pytest_device:{i}/{shards}",
                            ["PY", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", *part],
                            "pytest_device"))
    # The wheel itself on a card: two short BindCraft 2 trajectories with tt_bio imported from the
    # installed wheel and BindCraft 2 on the path, so a file the wheel drops fails here.
    legs.append(Leg("bc2:boundary", ["PY", "-c", "from tt_bio.main import ensure_p300_mesh_descriptor; "
                                     "ensure_p300_mesh_descriptor(); import runpy, sys; sys.argv = "
                                     "['boundary.py', '--params', '{AF2_PARAMS_DIR}', '--trajectories', '2', "
                                     "'--out', '{OUT}/bc2.json', '--project', '{OUT}/project']; "
                                     "runpy.run_path('perf/bc2_memory/boundary.py', run_name='__main__')"],
                    "bc2", tree_on_path=False))
    legs.append(Leg("perf", ["PY", "scripts/perf_regression.py"], "perf", timed=True))
    # --record-lever: add census levers to each ladder model's baseline (one fold per rung, refused
    # unless every other lever still matches), on a copy in the leg's out dir so the tree the
    # ladder legs read stays the commit's. The runner fetches each fragment to <out>/recorded/.
    # --record-full: the models whose splice was refused because something else moved get a full
    # re-record of their own fragment instead (every rung, timings included), which also carries
    # the new levers.
    full = [m for m in record_full.split(",") if m]
    unknown = [m for m in full if m not in roster["ladder"]]
    if unknown:
        raise SystemExit(f"--record-full: not ladder models: {', '.join(unknown)}")
    for m in roster["ladder"]:
        if m in full:
            how = ["--size-ladder-record"]
        elif record_lever:
            how = ["--size-ladder-record-lever", record_lever]
        else:
            continue
        legs.append(Leg(f"record:{m}", ["PY", "scripts/release_gate.py", "--model", "size-ladder",
                                        "--size-ladder-models", m, *how, "--size-ladder-baseline",
                                        "{OUT}/size_ladder_baseline.json"], "record", workdir=True,
                        setup="cp -r docs/size_ladder_baseline.json docs/size_ladder_baseline.d {OUT}/"))
    return legs


def test_files(sha: str, repo: Path = REPO) -> list:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", sha, "tests/"],
                         check=True, capture_output=True, text=True).stdout
    return sorted(p for p in out.splitlines() if Path(p).name.startswith("test_") and p.endswith(".py"))


def select(legs: list, patterns: list) -> list:
    """Legs matching any pattern (all when there is none) and no `!pattern`."""
    take = [p for p in patterns if not p.startswith("!")]
    drop = [p[1:] for p in patterns if p.startswith("!")]
    return [lg for lg in legs if (not take or any(fnmatch.fnmatch(lg.name, p) for p in take))
            and not any(fnmatch.fnmatch(lg.name, p) for p in drop)]


# ---------------------------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------------------------
@dataclass
class Host:
    name: str
    cfg: dict
    sha: str
    lock: threading.Condition = field(default_factory=threading.Condition)
    running: int = 0
    timed_running: bool = False

    @property
    def arch(self) -> str:
        return self.cfg["arch"]

    @property
    def root(self) -> str:
        return self.cfg["root"]

    @property
    def tree(self) -> str:
        return f"{self.root}/trees/{self.sha[:12]}"

    def ssh(self, cmd: str, **kw) -> subprocess.CompletedProcess:
        if self.cfg.get("ssh") in (None, "", "localhost"):
            return subprocess.run(["bash", "-c", cmd], text=True, **kw)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60",
                               self.cfg["ssh"], cmd], text=True, **kw)

    def pins(self) -> dict:
        """This host's resolved packages per venv, minus tt-bio and BindCraft 2 (installed from
        the wheel and a checkout), as constraints for the other hosts."""
        out = {}
        for v in (f"venv-{self.sha[:12]}", f"venv312-{self.sha[:12]}"):
            r = self.ssh(f"uv=$(command -v uv || echo ~/.local/bin/uv); $uv pip freeze -p {self.root}/{v}/bin/python",
                         capture_output=True, timeout=300)
            if r.returncode:
                raise SystemExit(f"{self.name}: cannot freeze {v}: {r.stderr.strip()[-300:]}")
            out[v.split("-")[0]] = "".join(
                ln for ln in r.stdout.splitlines(True)
                if not ln.startswith("-e") and " @ " not in ln
                and not re.match(r"(tt[-_]bio|bindcraft)\b", ln, re.I))
        return out

    def prepare(self, pins: dict | None = None) -> None:
        """A tree of its own at exactly `sha` (never a shared checkout moved under someone), with
        the parity fixtures restored, and the interpreters built from that commit's wheel, held
        to `pins` (another host's resolution) when given."""
        r, t = self.root, self.tree
        for v, text in (pins or {}).items():
            self.ssh(f"mkdir -p {r}/pins-{self.sha[:12]} && cat > {r}/pins-{self.sha[:12]}/{v}-{self.sha[:12]}.txt",
                     input=text, check=True, timeout=120)
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.cfg.get("prep_env", {}).items())
        cmd = (f"set -e; if [ ! -d {t} ]; then git -C {r}/repo fetch -q origin; "
               f"git -C {r}/repo worktree add -q --detach {t} {self.sha}; "
               f"cd {t}; bash scripts/fetch_parity_fixtures.sh >/dev/null; fi; "
               f"test \"$(git -C {t} rev-parse HEAD)\" = {self.sha}")
        p = self.ssh(cmd, capture_output=True, timeout=1800)
        if not p.returncode:
            # The runner's own copy of the recipe: the commit under test may predate it.
            p = self.ssh(f"cd {t} && env {envs} bash -s -- {r} {self.sha}", capture_output=True,
                         timeout=3600, input=(REPO / "scripts" / "gate_host_prep.sh").read_text())
        if p.returncode:
            raise SystemExit(f"{self.name}: cannot prepare {t} at {self.sha}: "
                             f"{(p.stderr or p.stdout).strip()[-600:]}")
        print(f"{self.name}: {p.stdout.strip()}")

    def python(self, family: str) -> str:
        """The interpreter a leg family runs under (pytest and the BindCraft 2 leg need the
        Python 3.12 venv that also carries BindCraft 2)."""
        s = self.sha[:12]
        default = (f"{self.root}/venv312-{s}/bin/python" if family in PY312
                   else f"{self.root}/venv-{s}/bin/python")
        return self.cfg.get("python_for", {}).get(family, self.cfg.get("python", default))

    def leg(self, leg: Leg) -> Leg:
        """`leg` with this host's extra flags for its family (hosts.json `args`)."""
        extra = self.cfg.get("args", {}).get(leg.family, [])
        return dataclasses.replace(leg, argv=leg.argv + extra) if extra else leg

    def seeding(self, leg: Leg) -> bool:
        return "--update-baseline" in self.cfg.get("args", {}).get(leg.family, [])

    def run_py(self, code: str, python: str | None = None) -> dict:
        cmd = (f"cd {self.tree} && TT_VISIBLE_DEVICES= PYTHONPATH={self.tree}:{self.tree}/scripts "
               f"timeout 300 {python or self.python('')} -c {shlex.quote(code)}")
        r = self.ssh(cmd, capture_output=True, timeout=400)
        if r.returncode:
            raise SystemExit(f"{self.name}: probe failed: {r.stderr.strip()[-400:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    def command(self, leg: Leg, card, out: str) -> str:
        c = self.cfg
        pp = ":".join([self.tree] * leg.tree_on_path + c.get("pythonpath_extra", []))
        # Every leg folds in one shared tree per host, so release_gate's size-ladder scratch must be
        # the leg's own: the first ladder or record leg to finish rmtrees the shared default under
        # the others, and a fold whose census dumps vanished reads every lever "not-imported".
        env = {"PYTHONPATH": pp, "TT_VISIBLE_DEVICES": "" if card is None else str(card),
               "RELEASE_GATE_SIZE_WORKDIR": f"{out}/sizegate-work", **c.get("env", {})}
        subst = {"{CARD}": str(card), "{OUT}": out,
                 **{f"{{{k}}}": v for k, v in c.get("env", {}).items()}}

        def fill(a: str) -> str:
            for k, v in subst.items():
                a = a.replace(k, v)
            return a
        argv = [self.python(leg.family) if a == "PY" else fill(a) for a in self.leg(leg).argv]
        body = (f"echo \"##LEG-START $(date +%s)\"; exec timeout -s INT {leg.budget} nice -n 10 "
                + " ".join(shlex.quote(a) for a in argv))
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        run = f"env {envs} bash -c {shlex.quote(body)}"
        if card is not None:
            lock = "bash scripts/flock_first.sh" if c.get("first") else "flock"
            run = f"{lock} {shlex.quote(c['lock'].format(card=card))} {run}"
        setup = f"{fill(leg.setup)} && " if leg.setup else ""
        return f"mkdir -p {shlex.quote(out)} && cd {self.tree} && {setup}{run}"


# ---------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------
def classify(leg: Leg, rc: int, report: dict | None) -> str:
    """PASS / FAIL / BLOCKED. A parity leg the gate itself reports BLOCKED-REF-REGEN-NEEDED does
    not fail the gate (RELEASING.md, verdict semantics); a one-leg run of it exits nonzero as
    GATE INCONCLUSIVE, so read the report rather than the exit code."""
    if leg.family == "parity" and report:
        verdicts = {r.get("verdict") for r in report.get("legs", [])}
        if verdicts == {"BLOCKED-REF-REGEN-NEEDED"}:
            return "BLOCKED"
    return "PASS" if rc == 0 else "FAIL"


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        p = self.path / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def shas(self) -> set:
        """Every commit a row was recorded at."""
        out = set()
        for p in self.path.glob("*.json"):
            try:
                out.add(json.loads(p.read_text())["sha"])
            except (ValueError, KeyError):
                continue
        return out

    def history(self) -> dict:
        """leg name -> the latest wall clock any run of it took, for ordering only."""
        seen = {}
        for p in self.path.glob("*.json"):
            try:
                r = json.loads(p.read_text())
            except ValueError:
                continue
            if r.get("wall_s") and r.get("ended", "") >= seen.get(r["leg"], ("", 0))[0]:
                seen[r["leg"]] = (r["ended"], r["wall_s"])
        return {k: v[1] for k, v in seen.items()}

    def put(self, key: str, rec: dict) -> None:
        p = self.path / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
        tmp.replace(p)


def write_verdict(out: Path, sha: str, archs: list, results: list) -> bool:
    by = {(r["leg"], r["arch"]): r for r in results}
    legs = sorted({r["leg"] for r in results})
    ok = all(r["verdict"] in OK for r in results)
    lines = [f"# Gate {sha[:12]}: {'PASS' if ok else 'FAIL'}", "",
             f"{len(legs)} legs x {', '.join(archs)}; written {now()}.", "",
             "| leg | " + " | ".join(archs) + " |", "|---|" + "---|" * len(archs)]
    for lg in legs:
        cells = []
        for a in archs:
            r = by.get((lg, a)) or by.get((lg, "any"))
            if r is None:
                cells.append("-")
            elif r["verdict"] == "REUSED":
                cells.append(f"REUSED ({r['evidence']})")
            elif r["verdict"] == "OWED":
                cells.append(f"OWED {','.join(r['owed'])}")
            else:
                cells.append(f"{r['verdict']} {r.get('wall_s', 0) / 60:.0f} min {r.get('worker', '')}")
        lines.append(f"| {lg} | " + " | ".join(cells) + " |")
    seeded = [r for r in results if r["verdict"] == "SEEDED"]
    if seeded:
        lines += ["", "SEEDED: no baseline existed on these card types, so these legs recorded one "
                  "instead of checking against one. Commit the files and drop the host's `args`:"]
        lines += [f"- {r['leg']} {r['card_type']}: {r.get('seeded', '?')}" for r in seeded]
    owed = sorted({f for r in results if r["verdict"] == "OWED" for f in r["owed"]})
    if owed:
        lines += ["", "OWED: the size-ladder baseline at this commit has no row for a lever these "
                  "models resolve, so their ladder legs were not folded. Record it, commit the "
                  "fragments, and gate that commit (RELEASING.md, Baselines a new lever owes):",
                  f"`gate_fanout.py --legs 'record:*' --record-lever {','.join(owed)} ...`"]
    recorded = [r for r in results if r["leg"].startswith("record:") and r.get("seeded")]
    if recorded:
        lines += ["", "RECORDED: copy these over docs/size_ladder_baseline.d/ in the release commit:"]
        lines += [f"- {r['leg']} {r['card_type']}: {r['seeded']}" for r in recorded]
    (out / "VERDICT.md").write_text("\n".join(lines) + "\n")
    (out / "verdict.json").write_text(json.dumps({"sha": sha, "pass": ok, "archs": archs,
                                                  "results": results}, indent=1))
    return ok


# ---------------------------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------------------------
class Gate:
    """One worker thread per host:card; each takes the next leg of its arch. A timed leg runs only
    on a --timed card and holds its host quiet (no new leg starts there while it runs)."""

    def __init__(self, todo: dict, workers: list, timed: set, execute, out: Path):
        self.todo = todo          # arch -> list of Leg (card legs), consumed in order
        self.workers = workers    # (Host, card)
        self.timed = timed        # {(host name, card)}
        self.execute = execute    # (host, card, leg) -> result dict
        self.out = out
        self.results: list = []
        self.mu = threading.Lock()

    @staticmethod
    def _queue(host: Host, card) -> str:
        return host.arch if card is not None else "any"

    def _take(self, host: Host, card) -> Leg | None:
        is_timed = (host.name, card) in self.timed
        with self.mu:
            q = self.todo.get(self._queue(host, card), [])
            # Correctness legs first, timed legs last: by then the host's other cards have run out
            # of work, so holding the host quiet idles nothing.
            for i, lg in enumerate(q):
                if not lg.timed:
                    return q.pop(i)
            for i, lg in enumerate(q):
                if lg.timed and is_timed:
                    return q.pop(i)
        return None

    def _pending(self, queue: str) -> bool:
        with self.mu:
            return bool(self.todo.get(queue))

    def worker(self, host: Host, card) -> None:
        while True:
            with host.lock:
                while host.timed_running:
                    host.lock.wait()
                leg = self._take(host, card)
                if leg is None:
                    # nothing left this worker may run (what remains, if anything, is timed)
                    return
                if leg.timed:
                    host.timed_running = True
                    while host.running:
                        host.lock.wait()
                host.running += 1
            try:
                res = self.execute(host, card, leg)
            finally:
                with host.lock:
                    host.running -= 1
                    if leg.timed:
                        host.timed_running = False
                    host.lock.notify_all()
            with self.mu:
                self.results.append(res)
                with open(self.out / "results.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")

    def run(self) -> list:
        ts = [threading.Thread(target=self.worker, args=w, daemon=True) for w in self.workers]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return self.results


def pool_job(host: Host, leg: Leg, rdir: str, name: str) -> str:
    """A job file for the host's chip pool (CHIPS.md "Chip pool"): the pool starts it as
    `CHIP=<umd> bash <job>` on an idle healthy chip, and it leaves its log and exit code in rdir."""
    cmd = host.command(leg, POOL_CARD, rdir)
    q = shlex.quote
    return (f"#!/bin/bash\n# {name}: gate leg {leg.name} at {host.sha[:12]} (gate_fanout.py)\n"
            f"mkdir -p {q(rdir)}; rm -f {q(rdir)}/rc\ncmd={q(cmd)}\n"
            f"{{ echo \"##CHIP $CHIP\"; eval \"${{cmd//{POOL_CARD}/$CHIP}}\"; }} > {q(rdir)}/leg.log 2>&1\n"
            f"echo $? > {q(rdir)}/rc.tmp && mv {q(rdir)}/rc.tmp {q(rdir)}/rc\n")


def run_in_pool(host: Host, card, leg: Leg, rdir: str, f) -> int:
    """Queue `leg` in the host's pool, wait for its exit code, copy its log (capped) into f."""
    pool = host.cfg["pool"]
    run, _, tag = Path(rdir).parts[-3:]     # <root>/out-<sha12>/<run>/<arch>/<leg>
    name = f"{pool.get('prio', 5)}-{pool.get('row', 'gate')}-{run}-{tag}"
    job, q = f"{pool['queue']}/{name}.sh", shlex.quote
    r = host.ssh(f"mkdir -p {q(rdir)} && rm -f {q(rdir)}/rc && cat > {q(job)}.tmp && mv {q(job)}.tmp {q(job)}",
                 input=pool_job(host, leg, rdir, name), capture_output=True, timeout=120)
    if r.returncode:
        f.write(f"cannot queue {job}: {r.stderr.strip()}\n")
        return 1
    f.write(f"# queued {host.name}:{job}\n")
    f.flush()
    QUEUED[job] = host
    while True:
        time.sleep(POOL_POLL_S)
        r = host.ssh(f"cat {q(rdir)}/rc", capture_output=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            rc = int(r.stdout.strip())
            break
    QUEUED.pop(job, None)
    r = host.ssh(f"L={q(rdir)}/leg.log; head -c {LOG_CAP} $L; [ $(stat -c%s $L) -le {LOG_CAP} ] || "
                 f"{{ echo; echo '# ... log capped at {LOG_CAP} bytes; last 400 lines:'; tail -n 400 $L; }}",
                 capture_output=True, timeout=300)
    f.write(r.stdout)
    return rc


def run_over_ssh(host: Host, card, leg: Leg, rdir: str, f) -> int:
    """Run `leg` on a named card, streaming its log into f."""
    cmd = host.command(leg, card, rdir)
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60", host.cfg["ssh"], cmd]
                         if host.cfg.get("ssh") not in (None, "", "localhost") else ["bash", "-c", cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The runner's host keeps every leg's log; cap each at LOG_CAP bytes (head kept, last lines
    # kept in memory) so one chatty leg cannot fill its disk.
    written, tail = 0, []
    for line in p.stdout:
        if written < LOG_CAP:
            f.write(line)
            written += len(line)
        else:
            tail = (tail + [line])[-400:]
    if tail:
        f.write(f"\n# ... log capped at {LOG_CAP} bytes; last {len(tail)} lines:\n" + "".join(tail))
    return p.wait()


def make_executor(sha: str, out: Path, ledger: Ledger, keys: dict, remote_out: str):
    def execute(host: Host, card, leg: Leg) -> dict:
        tag = leg.name.replace(":", "_").replace("/", "-")
        log = out / "logs" / host.arch / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        rdir = f"{host.root}/{remote_out}/{host.arch}/{tag}"
        t0 = time.time()
        run = run_in_pool if card == POOL else run_over_ssh
        while True:
            with open(log, "w") as f:
                f.write(f"# {leg.name} on {host.name}:{card} ({host.cfg['card_type']}) tree {sha}\n")
                f.flush()
                rc = run(host, card, leg, rdir, f)
            text = log.read_text(errors="replace")
            # The gate scripts refuse to start on an overloaded host (gate_guard's load ceiling).
            # That is not a verdict: wait for the load to drop and start the leg again.
            if not (rc and LOAD_REFUSAL in text and time.time() - t0 < LOAD_WAIT_MAX_S):
                break
            time.sleep(LOAD_RETRY_S)
        m = re.search(r"^##LEG-START (\d+)", text, re.M)
        start = float(m.group(1)) if m else None
        m = re.search(r"^##CHIP (\d+)", text, re.M)
        if m:
            card = f"{POOL}/{m.group(1)}"
        report = None
        if leg.workdir:
            r = host.ssh(f"cat {shlex.quote(rdir + '/report.json')}", capture_output=True, timeout=120)
            if r.returncode == 0:
                try:
                    report = json.loads(r.stdout)
                except ValueError:
                    pass
        verdict = classify(leg, rc, report)
        seeded = None
        if verdict == "PASS" and host.seeding(leg):
            verdict = "SEEDED"
            dst = out / "seeded" / host.arch / "perf_baselines.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = host.ssh(f"cat {host.tree}/docs/perf_baselines.json", capture_output=True, timeout=120)
            if r.returncode == 0:
                dst.write_text(r.stdout)
                seeded = str(dst)
        if verdict == "PASS" and leg.family == "record":
            m = leg.name.split(":", 1)[1]
            dst = out / "recorded" / host.cfg["card_type"] / f"{m}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = host.ssh(f"cat {shlex.quote(f'{rdir}/size_ladder_baseline.d/{m}.json')}",
                         capture_output=True, timeout=120)
            if r.returncode == 0:
                dst.write_text(r.stdout)
                seeded = str(dst)
            else:
                verdict = "FAIL"
        end = time.time()
        res = {"leg": leg.name, "arch": host.arch if leg.card else "any", "verdict": verdict,
               "rc": rc, "worker": f"{host.name}:{card}", "card_type": host.cfg["card_type"],
               "queued_s": round((start or end) - t0), "wall_s": round(end - (start or t0)),
               "log": str(log), "remote_dir": f"{host.name}:{rdir}", "ended": now(), "sha": sha,
               "key": keys[(leg.name, host.arch if leg.card else "any")]}
        if seeded:
            res["seeded"] = seeded
        if verdict in OK:
            ledger.put(res["key"], res)
        return res
    return execute


def parse_workers(spec: str, hosts: dict) -> list:
    out = []
    for w in filter(None, spec.split(",")):
        h, c = w.rsplit(":", 1)
        if h not in hosts:
            raise SystemExit(f"--workers names {h!r}, not in the hosts file")
        if c == POOL and "pool" not in hosts[h].cfg:
            raise SystemExit(f"--workers {w}: {h} has no pool in the hosts file")
        out.append((hosts[h], c if c == POOL else int(c)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sha", required=True, help="commit under test (pushed, every host fetches it)")
    ap.add_argument("--hosts", required=True, type=Path, help="hosts.json (see module docstring)")
    ap.add_argument("--workers", required=True, help="host:card[,host:card...] granted to this gate")
    ap.add_argument("--timed", default="", help="host:card per arch that runs the timed legs")
    ap.add_argument("--legs", default="", help="comma-separated globs, e.g. 'rg:*,parity:boltz2-*'; "
                                               "'!ladder:*' leaves legs out")
    ap.add_argument("--record-lever", default="", metavar="FLAG[,FLAG...]",
                    help="add record:<model> legs that splice these census levers into each ladder "
                         "model's size-ladder baseline per card type; fragments land in <out>/recorded/")
    ap.add_argument("--record-full", default="", metavar="MODEL[,MODEL...]",
                    help="fully re-record these ladder models' baselines per card type (for a model "
                         "whose --record-lever splice was refused); fragments land in <out>/recorded/")
    ap.add_argument("--arch", default="", help="limit to these archs (default: every arch in --workers)")
    ap.add_argument("--shards", type=int, default=4, help="pytest_device shards")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".tt-bio-gate-ledger")
    ap.add_argument("--no-reuse", action="store_true", help="run every leg even if its key passed")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", args.sha], check=True,
                         capture_output=True, text=True).stdout.strip()
    cfg = json.loads(args.hosts.read_text())
    workers = parse_workers(args.workers, {n: Host(n, c, sha) for n, c in cfg.items()})
    hosts = {}
    for h, _ in workers:
        hosts.setdefault(h.name, h)
    workers = [(hosts[h.name], c) for h, c in workers]
    timed = {(h.name, c) for h, c in parse_workers(args.timed, hosts)}
    archs = sorted({h.arch for h in hosts.values()})
    if args.arch:
        archs = [a for a in archs if a in args.arch.split(",")]
    for a in archs:
        if not any(h.arch == a for h, c in workers if (h.name, c) in timed):
            print(f"note: no --timed card for {a}; its timed legs will not run", file=sys.stderr)

    # The first host resolves the packages; every other host installs exactly that resolution.
    lead, *rest = hosts.values()
    lead.prepare()
    pins = lead.pins()
    with ThreadPoolExecutor() as ex:          # a failure re-raises here
        list(ex.map(lambda h: h.prepare(pins), rest))
    first = {a: next(h for h in hosts.values() if h.arch == a) for a in archs}
    roster = first[archs[0]].run_py(ENUMERATE)
    legs = select(build_legs(roster, test_files(sha), args.shards, args.record_lever, args.record_full),
                  [p for p in args.legs.split(",") if p])
    content, legacy = content_hash(sha), content_hash(sha, baselines=True, tests=True)
    probes = {}
    for a in archs:
        for py in {first[a].python(lg.family) for lg in legs}:
            probes[f"{a} {py}"] = first[a].run_py(ENV_PROBE, py)
    envs = {k: env_hash(p) for k, p in probes.items()}
    # A leg's key carries its arch's environment, read on the first host of that arch; every
    # other host of the arch must hold the same one, or a result would be filed under a key that
    # does not describe where it ran.
    for h in hosts.values():
        if h.arch in archs and h is not first[h.arch]:
            pairs = {(h.python(lg.family), first[h.arch].python(lg.family), lg.family) for lg in legs}
            for py, ref, fam in {(py, ref): (py, ref, fam) for py, ref, fam in pairs}.values():
                if env_hash(h.run_py(ENV_PROBE, py)) != envs[f"{h.arch} {ref}"]:
                    raise SystemExit(f"{h.name} and {first[h.arch].name} hold different packages for "
                                     f"{fam} legs; prepare both from the same commit")
    ctype = {a: first[a].cfg["card_type"] for a in archs}

    args.out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.ledger)
    # Rows written while test files were inside the content key (until 2026-10-09 21Z), at this
    # commit or at any commit with the same code, which a leg accepts when its own inputs match too.
    tests_in = {s: content_hash(s, tests=True) for s in {sha} | ledger.shas()
                if s == sha or _same_code(s, content)}
    keys, results, todo = {}, [], {a: [] for a in archs}
    for lg in legs:
        for a in (archs if lg.card else [archs[0]]):
            slot = a if lg.card else "any"
            env, ct, hleg = (envs[f"{a} {first[a].python(lg.family)}"], ctype[a] if lg.card else "cpu",
                             first[a].leg(lg))
            model = lg.name.split(":", 1)[1] if lg.family == "ladder" else ""
            base, tst = baseline_hash(sha, lg.family, model=model), tests_hash(sha, lg.family)
            k = leg_key(content, env, ct, hleg, base, tst)
            keys[(lg.name, slot)] = k
            # Card-free legs always run: they are cheap, and pytest_cpu checks the recorded
            # measurements the key leaves out.
            # A row under an older key scheme proved the same code against the same inputs, so it
            # is as good as a row under the new one: baselines inside the content hash (legacy),
            # or test files inside it, at any commit whose code and this leg's inputs match.
            olds = [leg_key(legacy, env, ct, hleg)] + [
                leg_key(c, env, ct, hleg, base) for s, c in tests_in.items()
                if s == sha or (baseline_hash(s, lg.family, model=model) == base
                                and tests_hash(s, lg.family) == tst)]
            hit = (None if args.no_reuse or not lg.card or lg.family == "record"
                   else next(filter(None, map(ledger.get, [k, *olds])), None))
            owes = owed_by(roster, ctype[a], lg)
            if owes:
                results.append({"leg": lg.name, "arch": slot, "verdict": "OWED", "owed": owes,
                                "card_type": ctype[a]})
            elif hit and hit.get("verdict") in OK:
                results.append({"leg": lg.name, "arch": slot, "verdict": "REUSED", "key": k,
                                "evidence": f"{hit['worker']} {hit['ended']} {hit['sha'][:9]} {hit['log']}",
                                "reused_verdict": hit["verdict"]})
            elif lg.card:
                todo[a].append(lg)
            else:
                todo.setdefault("any", []).append(lg)
    hist = ledger.history()
    for q in todo.values():
        # Record legs first: the ladder legs of the release commit wait on their fragments.
        q.sort(key=lambda lg: (lg.family != "record", -hist.get(lg.name, EXPECT[lg.family])))
    plan = {"sha": sha, "content": content, "envs": envs, "card_types": ctype, "archs": archs,
            "workers": [f"{h.name}:{c}" for h, c in workers], "timed": sorted(map(list, timed)),
            "env_probe": probes, "run": {a: [lg.name for lg in q] for a, q in todo.items()},
            "reused": [r["leg"] + "@" + r["arch"] for r in results if r["verdict"] == "REUSED"],
            "owed": [r["leg"] + "@" + r["arch"] for r in results if r["verdict"] == "OWED"],
            # A dry run, or one limited by --legs or --arch, is not a release verdict.
            "partial": bool(args.legs or args.arch or args.dry_run or args.record_lever
                       or args.record_full),
            # release_next.py tells a live run from one that was stopped before its verdict.
            "pid": os.getpid(), "host": socket.gethostname()}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"gate {sha[:12]}: {sum(len(q) for q in todo.values())} legs to run, "
          f"{len(plan['reused'])} reused, {len(plan['owed'])} owed, archs {archs}, {len(workers)} cards")
    if args.dry_run:
        for a, q in todo.items():
            print(f"  {a}: " + " ".join(lg.name for lg in q))
        return 0

    # Per run, so a job a stopped run left in a pool can never share a directory with a new one.
    execute = make_executor(sha, args.out, ledger, keys, f"out-{sha[:12]}/{args.out.name}")
    # Card-free legs run one at a time on the first host, as a worker without a card, so they
    # also stand aside while a timed leg holds that host quiet.
    # Started in the background by a non-interactive shell, SIGINT arrives ignored and the
    # cleanup below never ran; SIGTERM never reached it at all. Both stop the runner the same way.
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        results += Gate(todo, workers + [(first[archs[0]], None)], timed, execute, args.out).run()
    except KeyboardInterrupt:
        # A stopped runner takes back what it queued; a job the pool already started has left
        # the queue and finishes on its own.
        for job, host in list(QUEUED.items()):
            host.ssh(f"rm -f {shlex.quote(job)}", timeout=120)
        raise
    ok = write_verdict(args.out, sha, archs, results)
    print(f"GATE {'PASS' if ok else 'FAIL'}: {args.out / 'VERDICT.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
