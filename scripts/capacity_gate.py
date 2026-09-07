#!/usr/bin/env python3
"""Capacity gate: can every shipped model ALLOCATE and COMPLETE at 1536 tokens on this card?

THIS IS NOT A PARITY GATE AND CANNOT SUBSTITUTE FOR ONE. It answers one question -- does the shape
fit and does the pipeline finish -- and says nothing about whether the numbers are right. Its
fixture is a tandem-repeated chain whose structure is meaningless by construction, and it is meant
to run on cards (pc card 0 among them) that are known to miscompute matmuls. Correctness is
scripts/full_parity_gate.py's job, against real targets with cached references. A model that clears
this gate and fails that one has shipped a torn structure, which is worse than an OOM, not better.

WHY IT EXISTS. Three production failures this month were all the same hole: the platform's coverage
matrix sizes each cell at the model's own advertised ceiling, so while the ceilings were 576/576/627
it folded 576 and correctly passed. Raising every ceiling to 1024 moved the gate's target to 1024
and nothing re-ran it, so the failures surfaced in real traffic at 40-50%. Nothing in tt-bio tested
a size axis at all: full_parity_gate.py uses fixed targets, release_gate.py's ladder tops out at 768
single-sequence, and the only max-residues coverage anywhere ran against the live API on Wormhole.

THE TWO FAILURE CLASSES, AND WHY ONE TIER CANNOT SEE BOTH

  Class A, one oversized tensor. Shape-determined, so it appears on the FIRST execution of the op:
  OuterProductMean's 2 GiB single-shot z, the 8790736896 B DRAM buffer behind all three production
  failures, the 1.86 GB [depth, tokens, c_m] that still refuses at 960. One block exposes every one
  of these.

  Class B, cumulative residency and fragmentation. RF3 at 630-656 tokens dies LATE on a request as
  small as 103 MB with DRAM already 99% full: the named allocation is the last straw, not the
  problem. A single block cannot see this, because the residency has not been built up yet.
  OpenFold3's second wall is the same shape from the other side -- an L1 refusal inside the
  diffusion transformer, retried rather than fatal, so it presents as a STALL and not an error. A
  gate that only watches for exceptions scores that as a pass.

So a layer-subset screen is necessary and not sufficient, and this runs both:

  TIER 1, screen. One block per stack (scripts/capacity_hook.py truncates every block list to its
  first element) plus one diffusion step, at the target token count. Seconds per model. A FAIL here
  is definitive: a shape that cannot allocate once cannot allocate ever, so Tier 2 is skipped.
  A PASS here is NOT A VERDICT and is never reported as one -- see _screen.

  TIER 2, residency. The full pipeline at the target size. Minutes per model. Catches Class B and
  the diffusion-side L1 wall. Carries a STALL DETECTOR, because that failure mode is a hang.

EFFICIENCY THAT COSTS NO COVERAGE, all of it taken: a committed MSA fixture instead of an alignment
search (device memory does not care how the alignment was found); target-FIRST rather than
laddering up, so a passing model costs one run and only a failure pays for a bisect;
diffusion_samples=1; and one model per card over --workers.

EFFICIENCY THAT COSTS COVERAGE is a named decision, written into the report, never silent. Every
reduction a run applies lands in report["reductions"] and in the printed summary. Two are hard
rules: the diffusion/structure stage is never skipped (OpenFold3's remaining wall lives inside it),
and MSA depth is never reduced for an MSA model without saying so, because for the OF3 family the
failing tensor scales with tokens x rows -- at 14190 rows OpenFold3 folds 576 and dies at 614,
while single-sequence it folds 768 in 301 s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import capacity_fixture                                                    # noqa: E402

# ---------------------------------------------------------------------------------------------
# THE BAR
# ---------------------------------------------------------------------------------------------
#: The bar is 1536 TOKENS. The token axis buckets to a multiple of 32 and an unmasked tail is a
#: ~72x error, so the number tested has to be the number the hardware sees: 1536 is exactly
#: 48 x 32 and needs no padding at all, where a "1500" bar would pad to 1504 and report a size
#: 4 tokens smaller than what actually ran. Blackhole-only -- 1536 is out of reach on a 12 GiB
#: Wormhole chip for several of these models, so nothing here is a Wormhole statement.
TOKEN_BAR = 1536
TOKEN_BUCKET = 32
assert TOKEN_BAR % TOKEN_BUCKET == 0

#: Bisect rungs, used ONLY after a failure, to report where the real ceiling sits. A passing model
#: never runs any of these. All bucket-aligned for the same reason the bar is.
BISECT_RUNGS = (1408, 1280, 1024, 896, 768, 640, 512)

#: No forward progress for this long is a FAIL, not a wait. OpenFold3's diffusion-side L1 refusal
#: is retried rather than raised and sat for 2289 s at diffusion step 0; a gate that waits for an
#: exception scores that green.
#:
#: Progress is the CLI's own structured event stream (TT_BIO_PROGRESS_CAPTURE, the same stream the
#: UX-regression guard reads), with log growth as a fallback. It is deliberately not CPU time: the
#: OF3 stall is a RETRY loop, so it burns CPU while going nowhere.
STALL_S = 900
#: Before the first progress event, though, silence is normal and long. The JIT kernel cache is
#: keyed by shape, so the first run at a new token count compiles every kernel from scratch with
#: no output at all: measured on esmfold2 at 1504 tokens, 7.5 minutes of CPU and zero log bytes.
#: Applying the post-first-event threshold from t=0 would score that cold compile as a hang.
WARMUP_S = 2700
#: Total wall-clock ceiling per run. A 1536-token deep-MSA fold is minutes, not an hour.
RUN_TIMEOUT_S = 5400
#: Host RAM headroom below which a death is recorded as HOST_OOM and not as a device wall. A
#: 1536-token deep-MSA fold can OOM the HOST, which is a different failure and must not be
#: reported as a capacity ceiling.
HOST_RAM_FLOOR_MB = 700

VERDICTS = ("PASS", "FAIL", "STALL", "HOST_OOM", "NO_WEIGHTS", "CONTENDED", "CARD_DIRTY",
            "ERROR", "SKIPPED")

# ---------------------------------------------------------------------------------------------
# THE MODEL LIST -- DERIVED, NEVER HARDCODED
# ---------------------------------------------------------------------------------------------
# A hand-typed list is exactly how a new port gets shipped with no capacity coverage; that class
# has bitten this repo repeatedly (release_gate.py's SIZE_LADDER tuple, perf_regression.py's SPECS,
# the platform's own model list). So the roster comes from main.py's own `*_MODELS` tuples via
# size_limits.shipped_models(), and `coverage_gaps()` FAILS the gate when a shipped model is
# neither runnable here nor carrying a written exemption. A new model appears automatically.


def _verbs() -> dict[str, str]:
    """model id -> the CLI verb that runs it, read off main.py's tuples."""
    from tt_bio import main as m
    out = {}
    for verb, models in (("predict", m.PREDICT_MODELS), ("embed", m.EMBED_MODELS),
                         ("saprot", m.SAPROT_MODELS), ("design", m.DESIGN_MODELS),
                         ("affinity", m.AFFINITY_MODELS)):
        for name in models:
            out[name] = verb
    return out


#: Models this gate cannot drive, each with the reason written down. A reason is not a pass: it
#: records what is NOT covered so the gap is readable, the same discipline as
#: release_gate.SIZE_LADDER_EXEMPT. Anything here is reported SKIPPED with its reason.
EXEMPT = {
    "boltzgen": "design, not a fold: its input is a target plus a binder spec and its measured "
                "cap is atom-denominated (between 3158 and 4651 atoms in the trunk Pairformer's "
                "triangle attention), which a token bar cannot express. Needs an atom-denominated "
                "cell of its own.",
    "rfd3":     "design, not a fold: sized on DESIGN_TOTAL (motif plus designed) from a contig "
                "spec, so the 1536-token fixture here is not a valid input. Its wall is "
                "fragmentation rather than capacity and wants its own cell.",
    "pxdesign": "design, not a fold: sized on DESIGN_TARGET from a target STRUCTURE, so it needs a "
                "1536-residue PDB rather than a sequence. The shipped ladder's own fixture source "
                "(1DP0 chain A, 1011 residues) cannot reach the bar either.",
    "saprot-1.3b": "structure-aware embeddings, and the checkpoint is not in the local weights "
                   "cache on this host. saprot-35m and saprot-650m carry the same code path at "
                   "the bar; this one is a weights gap, not a code gap.",
}


def roster() -> list[str]:
    from tt_bio import size_limits as sl
    return sorted(sl.shipped_models())


def coverage_gaps() -> list[str]:
    """Shipped models that are neither runnable by this gate nor exempted in writing."""
    verbs = _verbs()
    return sorted(m for m in roster()
                  if m not in EXEMPT and verbs.get(m) not in ("predict", "embed", "saprot",
                                                              "affinity"))


# ---------------------------------------------------------------------------------------------
# PER-MODEL CELLS -- THE BAR IS IN TOKENS, RESIDUES ARE DERIVED
# ---------------------------------------------------------------------------------------------
# Ligand atoms are tokens on top of the residue count, so OpenBind at 1536 RESIDUES is more than
# 1536 tokens and a residue-defined bar silently under-tests exactly the model that failed hardest
# in production. The bar is therefore in tokens and each cell derives its own residue count:
#
#     residues = TOKEN_BAR - ligand_tokens
#
# The gate's fixture is polymer-only and every residue in it is a standard amino acid, so for a
# ligand-free cell tokens == residues exactly (AF3-style tokenisation gives one token per standard
# residue). That is a property of THIS fixture, stated so the next person does not generalise it:
# a cell that adds a ligand must set ligand_tokens, and the ligand-token path is otherwise
# UNCOVERED here (named in report["reductions"]).


class Cell:
    """One model's capacity cell: how to drive it, and at what size and depth."""

    def __init__(self, model, verb, *, ligand_tokens=0, depth=None, recycling=None,
                 msa=True, reason_depth=None):
        self.model, self.verb = model, verb
        self.ligand_tokens = ligand_tokens
        self.depth = depth                 # None -> the committed source's full depth
        self.recycling = recycling         # None -> the model's production default
        self.msa = msa
        self.reason_depth = reason_depth

    def residues(self, tokens: int) -> int:
        return tokens - self.ligand_tokens

    def padded(self, tokens: int) -> int:
        """What the hardware actually sees, after this model's own token bucketing."""
        try:
            from tt_bio import token_axis
            return int(token_axis.bucketed_width(tokens, token_axis.bucket_multiple(self.model)))
        except Exception:
            return tokens


def cells(models: list[str], *, depth=None, recycling=None) -> list[Cell]:
    """Build a cell per model. `depth`/`recycling` are run-wide named reductions."""
    verbs, out = _verbs(), []
    from tt_bio.main import MSA_DEFAULT_MODELS
    for m in models:
        verb = verbs.get(m)
        # embed/saprot take a bare sequence and have no MSA track at all, so a depth reduction is
        # not a reduction for them and recycling does not exist.
        msa = verb == "predict" and m in MSA_DEFAULT_MODELS
        out.append(Cell(m, verb, depth=depth if msa else None,
                        recycling=recycling if verb == "predict" else None, msa=msa))
    return out


# ---------------------------------------------------------------------------------------------
# BOARD AND BANK GEOMETRY -- READ OUT OF THE ALLOCATOR, NOT A SPEC SHEET
# ---------------------------------------------------------------------------------------------
# "Blackhole" is not one board. p150a (pc, qb1) and p300c (qb2) differ, and an interleaved
# allocation is refused on BANK SIZE and not total capacity. Measured on pc's p150a: 8 DRAM banks
# of 4278190016 B, 31.875 GiB, against Wormhole's 12 x ~1 GiB. L1 bank count is host-specific too
# -- pc runs custom 130-core firmware where a stock part presents 140 -- so both are read from the
# live allocator. Without this the numbers are not comparable across hosts.

_GEOM_PROBE = r"""
import json, ttnn
d = ttnn.open_device(device_id=0)
o = {"arch": ttnn.get_arch_name(),
     "grid": str(d.compute_with_storage_grid_size())}
for tag, bt in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
    v = ttnn.get_memory_view(d, bt)
    o[tag] = {"banks": int(v.num_banks),
              "bytes_per_bank": int(v.total_bytes_per_bank),
              "largest_contiguous_free_per_bank":
                  int(v.largest_contiguous_bytes_free_per_bank)}
    o[tag]["total_bytes"] = o[tag]["banks"] * o[tag]["bytes_per_bank"]
ttnn.close_device(d)
print("CAPGATE_GEOM " + json.dumps(o))
"""


def board_type() -> str | None:
    """The canonical board_type, from tt-smi. p150a and p300c are both 'blackhole' to ttnn."""
    for exe in (Path.home() / ".local/bin/tt-smi", Path("/usr/local/bin/tt-smi")):
        if not exe.exists():
            continue
        try:
            r = subprocess.run([str(exe), "-s"], capture_output=True, text=True, timeout=60)
            info = json.loads(r.stdout)["device_info"]
            return info[0].get("board_info", {}).get("board_type")
        except Exception:
            continue
    return None


def geometry(worker) -> dict:
    """Open the card once, read arch + bank geometry off the allocator, close it."""
    rc, out, _ = worker.run([sys.executable, "-c", _GEOM_PROBE], timeout=600)
    for line in (out or "").splitlines():
        if line.startswith("CAPGATE_GEOM "):
            g = json.loads(line[len("CAPGATE_GEOM "):])
            g["board_type"] = board_type() if worker.is_local else None
            g["host"] = worker.host
            g["card"] = worker.card
            return g
    return {"error": f"geometry probe failed (rc={rc})", "host": worker.host, "card": worker.card}


# ---------------------------------------------------------------------------------------------
# RUNNING ONE CELL
# ---------------------------------------------------------------------------------------------
# The gate drives the real CLI. A capacity check that reimplemented the load-and-fold path would
# be testing its own reimplementation, and the failures it exists to catch live in the production
# path's allocation ORDER (ttnn.slice is not a view, and allocation order decides whether a block
# big enough exists).

#: Cheap host-side signals that a fold hit the device wall rather than anything else. Matched
#: against the run log so a verdict names its own mechanism instead of just "exited nonzero".
MECHANISM_PATTERNS = (
    ("dram",          re.compile(r"Out of Memory: Not enough space to allocate.*DRAM", re.I)),
    ("l1",            re.compile(r"Out of Memory: Not enough space to allocate.*L1", re.I)),
    ("fragmentation", re.compile(r"largest free block", re.I)),
    ("dram",          re.compile(r"Statically allocated circular buffers.*exceed", re.I)),
    ("oom",           re.compile(r"Out of Memory|bad_alloc|std::bad_alloc", re.I)),
    # Not a capacity result at all: another process holds the card. Scoring this as FAIL would
    # publish a ceiling that was never measured -- and it is easy to hit, because a killed leg
    # whose spawned fold worker outlived the kill keeps the lease.
    ("contention",    re.compile(r"DeviceInUseError|device contention, nothing ran"
                                 r"|is in use by", re.I)),
)


def classify(log_text: str) -> str | None:
    for name, pat in MECHANISM_PATTERNS:
        if pat.search(log_text):
            return name
    return None


def tree_cpu_s(pid: int) -> float:
    """CPU seconds burned by a process and its children, from /proc. Used only to ANNOTATE a
    stall, never to decide one: the OF3 failure this gate watches for is a retry loop, so it burns
    CPU going nowhere, and treating CPU as progress would hide exactly that case. But a stall that
    was idle throughout and one that was computing throughout are different findings, and the
    report should say which it saw."""
    total, seen = 0.0, set()
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            f = Path(f"/proc/{p}/stat").read_text().rsplit(") ", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
            stack += [int(c) for c in
                      Path(f"/proc/{p}/task/{p}/children").read_text().split()]
        except (OSError, IndexError, ValueError):
            continue
    return total


def host_ram_free_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


class Worker:
    """One host:card slot. Mirrors scripts/full_parity_gate.py's vetted worker, whose --workers
    round-robin is exactly right for one-model-per-card, without importing 2500 lines of
    correctness harness to get it."""

    def __init__(self, host, card, is_local, remote_cwd=None, remote_python=None):
        self.host, self.card, self.is_local = host, card, is_local
        self.remote_cwd, self.remote_python = remote_cwd, remote_python

    def __repr__(self):
        return f"{self.host}:{self.card}"

    def env(self, extra: dict | None = None) -> dict:
        # TT_BIO_LEASE_CARDS is the card grant and tt-bio enforces it at the device open: an
        # UNPINNED open is refused too, because a process that can see four cards brings up all
        # four (UMD starts every visible chip, not just the one it computes on).
        e = {
            "TT_VISIBLE_DEVICES": str(self.card),
            "TT_BIO_LEASE_CARDS": str(self.card),
            # Inherit the fleet's holder identity when there is one: it is what the dispatcher's
            # running-task check reads, and overwriting it with a bare gate pid makes a live task
            # look idle. Only name ourselves when nothing else has.
            "TT_BIO_LEASE_HOLDER": os.environ.get("TT_BIO_LEASE_HOLDER")
                                   or f"capacity_gate:{os.getpid()}",
            # The env has tt_bio installed EDITABLE against the shared checkout, and running a
            # script puts scripts/ on sys.path[0] with cwd absent -- so `import tt_bio` silently
            # loads the stale shared tree. Every leg of this gate must score THIS worktree.
            "PYTHONPATH": str(REPO_ROOT),
        }
        e.update(extra or {})
        return e

    def cmd(self, argv: list[str], extra_env: dict | None = None) -> list[str]:
        env = self.env(extra_env)
        assign = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        work = self.remote_cwd or str(REPO_ROOT)
        if self.is_local:
            return ["sh", "-c", f"{assign} exec " + " ".join(shlex.quote(c) for c in argv)]
        remote = f"cd {shlex.quote(work)} && {assign} exec " + " ".join(
            shlex.quote(c) for c in argv)
        return ["ssh", "-o", "ConnectTimeout=5", self.host, remote]

    def run(self, argv, *, timeout, extra_env=None):
        p = subprocess.run(self.cmd(argv, extra_env), capture_output=True, text=True,
                           cwd=REPO_ROOT, timeout=timeout)
        return p.returncode, p.stdout, p.stderr


HOST_ALIASES = {"qb1": "tt-quietbox", "qb2": "tt-quietbox2"}


def local_host() -> str:
    return (os.environ.get("HOSTNAME") or socket.gethostname()).split(".")[0]


def _is_local(host: str, this_host: str) -> bool:
    if host in ("localhost", "127.0.0.1", this_host):
        return True
    return HOST_ALIASES.get(host) == this_host or HOST_ALIASES.get(this_host) == host


def parse_workers(spec: str) -> list[Worker]:
    """'--workers host:card[:remote_cwd[:remote_python]][,...]'. A name can lie in both
    directions: an alias that exists only in one user's ssh config resolves nowhere else, and one
    that resolves to THIS box silently doubles a card's load instead of fanning out."""
    out, this = [], local_host()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        host, _, rest = part.partition(":")
        card, _, rest2 = rest.partition(":")
        cwd, _, py = rest2.partition(":")
        out.append(Worker(host, int(card or 0), _is_local(host, this), cwd or None, py or None))
    return out or [Worker(this, 0, True)]


# ---------------------------------------------------------------------------------------------
# THE SITECUSTOMIZE BOOTSTRAP
# ---------------------------------------------------------------------------------------------


def hook_dir(work: Path) -> Path:
    """Scratch dir holding the sitecustomize that arms the hook, and a copy of the hook itself.

    `predict` folds in a SPAWNED worker, so a patch installed in the launcher would never reach
    the process that opens the device. CPython imports `sitecustomize` in every interpreter it
    starts, and PYTHONPATH is inherited across spawn, so this reaches all of them.

    The hook is COPIED here rather than imported from scripts/, because it has to be importable
    from a directory the child is guaranteed to have. It was not: the child's PYTHONPATH carried
    the repo root and this dir but never repo/scripts, so `import capacity_hook` raised
    ModuleNotFoundError, sitecustomize's bare `except` swallowed it, and every "screen" silently
    ran at full depth for a whole campaign. Which is why the failure is now RECORDED and the gate
    refuses to call a leg a screen unless the hook says it installed.
    """
    d = work / "_hook"
    d.mkdir(parents=True, exist_ok=True)
    (d / "capacity_hook.py").write_text((Path(__file__).parent / "capacity_hook.py").read_text())
    (d / "sitecustomize.py").write_text(
        "# Generated by scripts/capacity_gate.py. Inert unless TT_BIO_CAPACITY_HOOK is set.\n"
        "import os\n"
        "if os.environ.get('TT_BIO_CAPACITY_HOOK'):\n"
        "    try:\n"
        "        import capacity_hook\n"
        "        capacity_hook.install()\n"
        "    except BaseException as exc:\n"
        "        # NEVER silent: a hook that did not install turns a screen into a full-depth run\n"
        "        # reported as a screen, and a residency peak into a number nobody measured.\n"
        "        out = os.environ.get('TT_BIO_CAPACITY_HOOK_OUT')\n"
        "        if out:\n"
        "            try:\n"
        "                with open('%s.install-failed.%d' % (out, os.getpid()), 'w') as fp:\n"
        "                    fp.write('%s: %s' % (type(exc).__name__, exc))\n"
        "            except OSError:\n"
        "                pass\n")
    return d


def hook_findings(out_prefix: Path) -> dict:
    """Merge what the hook reported from every interpreter in the run. The launcher patches and
    folds nothing, so its empty result must not overwrite the worker's real one."""
    failed = sorted(out_prefix.parent.glob(out_prefix.name + ".install-failed.*"))
    if failed:
        return {"install_failed": failed[0].read_text()[:200], "truncated": [],
                "instrumented": []}
    best = {}
    for p in sorted(out_prefix.parent.glob(out_prefix.name + ".*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        score = len(d.get("truncated", [])) + len(d.get("instrumented", []))
        if score >= len(best.get("truncated", [])) + len(best.get("instrumented", [])):
            if score or not best:
                best = d
    return best


# ---------------------------------------------------------------------------------------------
# ONE RUN
# ---------------------------------------------------------------------------------------------


def build_argv(cell: Cell, fixture: dict, out_dir: Path, *, tier: str) -> list[str]:
    """The CLI invocation for one cell. `tier` picks screen depth from residency depth."""
    py = [sys.executable, "-m", "tt_bio.main"]
    if cell.verb in ("embed", "saprot"):
        # No MSA track and no diffusion stage: one pass at the bar is the whole test. These verbs
        # want a FASTA, not predict's `sequences:` document -- handing them the latter parses to
        # nothing and the cell would pass having embedded zero residues.
        return py + [cell.verb, str(fixture["fasta"]), "--model", cell.model,
                     "--out_dir", str(out_dir)]
    if cell.verb == "affinity":
        return py + ["affinity", str(fixture["yaml"]), "--model", cell.model,
                     "--recycling_steps", "1" if tier == "screen" else "5",
                     "--out_dir", str(out_dir)]
    argv = py + ["predict", str(fixture["yaml"]), "--model", cell.model,
                 "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out_dir)]
    if cell.msa:
        argv += ["--msa_dir", str(fixture["msa_dir"]), "--msa_cache_only"]
    else:
        argv += ["--single_sequence"]
    if tier == "screen":
        # One diffusion step and one recycle. The stage is NEVER skipped -- OpenFold3's remaining
        # wall lives inside it -- only shortened, and the shapes it allocates are step-independent.
        argv += ["--sampling_steps", "1", "--recycling_steps", "1"]
    elif cell.recycling is not None:
        argv += ["--recycling_steps", str(cell.recycling)]
    return argv


def execute(worker: Worker, argv: list[str], log: Path, *, mode: str,
            hook_out: Path, hookdir: Path, timeout=RUN_TIMEOUT_S, stall_s=STALL_S) -> dict:
    """Run one leg with a stall detector and a host-RAM watch.

    The stall detector is the reason this is not a plain subprocess.run: OpenFold3's diffusion-side
    L1 refusal is RETRIED rather than raised, so the process stays alive and busy and produces no
    output. Watching for an exception scores that as a pass; watching for forward progress does not.
    """
    events = log.with_suffix(".events.jsonl")
    events.unlink(missing_ok=True)
    beat = log.with_suffix(".beat")
    for old in beat.parent.glob(beat.name + ".*"):
        old.unlink(missing_ok=True)
    env = {"TT_BIO_CAPACITY_HOOK": mode,
           # repo/scripts too: the gate's own sys.path tweak does not reach a spawned child.
           "TT_BIO_CAPACITY_HOOK_OUT": str(hook_out),
           "TT_BIO_CAPACITY_HOOK_BEAT": str(beat),
           "TT_BIO_PROGRESS_CAPTURE": str(events),
           "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": f"{hookdir}:{REPO_ROOT}:{REPO_ROOT / 'scripts'}"}
    ram_floor = host_ram_free_mb()
    t0 = time.monotonic()
    with open(log, "w") as fp:
        proc = subprocess.Popen(worker.cmd(argv, env), stdout=fp, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, start_new_session=True)

    def progress() -> tuple[int, bool]:
        """(a monotonically growing progress counter, whether real work has started yet).

        Three signals summed. The heartbeat is the sharp one: the CLI's progress stream is per
        recycle, and at 1536 tokens ONE trunk block can run for minutes, so the coarse signal alone
        would read a legitimately grinding block as a hang.
        """
        n = events.stat().st_size if events.exists() else 0
        beats = sum(f.stat().st_size for f in beat.parent.glob(beat.name + ".*"))
        return (n + beats + (log.stat().st_size if log.exists() else 0)), (n > 0 or beats > 0)

    last, last_move, stalled, warm = -1, time.monotonic(), False, False
    cpu_at_quiet, cpu_now = tree_cpu_s(proc.pid), tree_cpu_s(proc.pid)
    try:
        while proc.poll() is None:
            time.sleep(5)
            ram_floor = min(ram_floor, host_ram_free_mb())
            cpu_now = tree_cpu_s(proc.pid)
            n, seen = progress()
            if n != last:
                last, last_move, cpu_at_quiet = n, time.monotonic(), cpu_now
            if seen and not warm:
                # First real progress event: the cold compile is behind us, so tighten up.
                warm, last_move = True, time.monotonic()
            now = time.monotonic()
            if now - last_move > (stall_s if warm else max(stall_s, WARMUP_S)):
                stalled = True
                break
            if now - t0 > timeout:
                break
        if proc.poll() is None:
            # The CLI spawns worker processes; killing only the outer pid orphans the engine.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=60)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    wall = time.monotonic() - t0
    text = log.read_text(errors="replace") if log.exists() else ""
    return {"rc": proc.returncode, "wall_s": round(wall, 1), "stalled": stalled,
            "warmed": warm, "quiet_s": round(time.monotonic() - last_move, 1),
            "progress_events": events.stat().st_size if events.exists() else 0,
            "block_calls": sum(f.stat().st_size for f in beat.parent.glob(beat.name + ".*")),
            # Evidence for a STALL: was it grinding or waiting? Annotation only, never the verdict.
            "cpu_s_while_quiet": round(max(0.0, cpu_now - cpu_at_quiet), 1),
            "host_ram_floor_mb": ram_floor, "mechanism": classify(text),
            "hook": hook_findings(hook_out),
            "tail": "\n".join(text.splitlines()[-25:])}


#: Opening the device is NOT enough to prove the card is usable. A chip left dirty by a killed
#: fold opens fine and then hangs on the first program dispatch -- measured here: a SIGKILLed
#: large fold left card 0 in a state where the next run sat forever inside tt-bio's own
#: dispatch probe, all threads idle, with no error. So this probe dispatches and synchronizes.
_CARD_PROBE = (
    "import torch, ttnn\n"
    "d = ttnn.open_device(device_id=0)\n"
    "t = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16),\n"
    "                    layout=ttnn.TILE_LAYOUT, device=d)\n"
    "ttnn.add(t, t)\n"
    "ttnn.synchronize_device(d)\n"
    "ttnn.close_device(d)\n"
    "print('CARD_HEALTHY')\n")


def card_healthy(worker: Worker, *, timeout=420) -> bool:
    """Can this card open AND dispatch a program?

    Zero processes is not proof of a clean chip. A killed leg leaves both possibilities: the lease
    still held by a spawned fold worker that outlived the kill, and a chip that accepts an open and
    then never dispatches. Both turn every leg after them into a spurious FAIL, which would publish
    a ceiling nobody walked -- so the gate checks before it believes a failure.
    """
    try:
        _, out, _ = worker.run([sys.executable, "-c", _CARD_PROBE], timeout=timeout)
        return "CARD_HEALTHY" in (out or "")
    except subprocess.TimeoutExpired:
        return False


def wait_for_card(worker: Worker, *, timeout=600) -> bool:
    """Block until the card is free and dispatching again."""
    deadline = time.monotonic() + timeout
    while True:
        if card_healthy(worker, timeout=300):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(15)


#: Resetting a card is the ONE operation in this gate that can hurt somebody else's run, so it is
#: allowed only when this run owns the host outright. On a p300c a `tt-smi -r` resets the BOARD
#: PAIR and not the chip, so a reset issued for card 0 also takes down card 1 -- a card this gate
#: may not even be scheduling on.
def _may_reset(worker: Worker, workers: list[Worker]) -> bool:
    return worker.is_local and sum(1 for w in workers if w.host == worker.host) == 1


def recover_card(worker: Worker, workers: list[Worker]) -> tuple[bool, str]:
    """Bring a wedged card back, or say why not.

    This exists because provoking an out-of-memory refusal is THE JOB of this gate, and a
    device-side TT_FATAL can leave the chip accepting an open and then never dispatching. Without
    recovery the first model that legitimately fails the bar wedges the card and every model after
    it reads as a failure too, so a one-line real result would arrive wrapped in a cascade of
    invented ones. Polling cannot fix a wedge; only a reset can.
    """
    if card_healthy(worker, timeout=300):
        return True, "still dispatching"
    if not _may_reset(worker, workers):
        return False, ("card stopped dispatching and this run does not own the host exclusively, "
                       "so it must not reset (a reset takes the board pair down with it)")
    smi = os.path.expanduser("~/.local/bin/tt-smi")
    try:
        subprocess.run([smi, "-r", str(worker.card)], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"reset failed to run: {type(exc).__name__}"
    if card_healthy(worker, timeout=420):
        return True, f"recovered by tt-smi -r {worker.card}"
    return False, f"tt-smi -r {worker.card} did not restore dispatch"


def _screen(worker, cell, fixture, work, hookdir) -> dict:
    """TIER 1. A FAIL here is definitive and short-circuits Tier 2. A pass is INCONCLUSIVE.

    That asymmetry is what makes the shortcut safe. The truncation can only reduce what runs, so
    it can miss a Class B failure but cannot invent one; a screen that passed therefore proves
    nothing and is never reported as PASS. Which means an over-permissive screen costs wall-clock
    and nothing else -- including the case where the hook matched no block list at all and the
    "screen" quietly ran full depth, which the hook records and the report prints.
    """
    log = work / f"screen_{cell.model}.log"
    hook_out = work / f"hook_screen_{cell.model}"
    argv = build_argv(cell, fixture, work / f"out_screen_{cell.model}", tier="screen")
    r = execute(worker, argv, log, mode="screen", hook_out=hook_out, hookdir=hookdir,
                timeout=RUN_TIMEOUT_S, stall_s=STALL_S)
    trunc = r["hook"].get("truncated") or []
    r["stacks_truncated"] = len(trunc)
    r["truncated"] = trunc[:40]
    # The screen's whole claim is "one block per stack". If the hook did not install or matched
    # nothing, this leg ran the FULL model and calling it a screen would misreport both what ran
    # and what a clean result means.
    r["hook_installed"] = "install_failed" not in r["hook"]
    if not r["hook_installed"]:
        r["note"] = (f"the block-truncation hook did not install "
                     f"({r['hook']['install_failed']}), so this leg ran at FULL depth. Not a "
                     f"screen.")
    elif not trunc:
        r["note"] = ("the hook installed but matched no block stack, so this leg ran at FULL "
                     "depth. Not a screen.")
    if r["rc"] == 0 and not r["stalled"]:
        r["verdict"] = "INCONCLUSIVE"          # deliberately not PASS
        return r
    if r["stalled"]:
        # A stall under truncation is not a definitive shape verdict; let Tier 2 decide.
        r["verdict"] = "INCONCLUSIVE"
        r["note"] = f"screen made no progress for {STALL_S}s; deferring to the residency run"
        return r
    if r["mechanism"] == "contention":
        r["verdict"] = "CONTENDED"
        return r
    if not r["mechanism"] and (r["host_ram_floor_mb"] < HOST_RAM_FLOOR_MB
                               or _host_killed(r["rc"], r["tail"])):
        # No device-side error and the worker died on a signal: the HOST ran out of memory. A
        # 1504-token fold that the kernel kills at 22.8 GiB anon-rss on a 30 GB box never reached
        # the device wall, and recording it as one publishes a ceiling nobody walked.
        r["verdict"] = "HOST_OOM"
        r["host_oom_evidence"] = _oom_killer_fired(r.get("model", ""))
        return r
    r["verdict"] = "FAIL"
    return r


def _residency(worker, cell, fixture, work, hookdir, tokens) -> dict:
    """TIER 2. The full pipeline at the target size. This is the only tier that can say PASS."""
    log = work / f"resid_{cell.model}_{tokens}.log"
    hook_out = work / f"hook_resid_{cell.model}_{tokens}"
    argv = build_argv(cell, fixture, work / f"out_resid_{cell.model}_{tokens}", tier="residency")
    r = execute(worker, argv, log, mode="residency", hook_out=hook_out, hookdir=hookdir)
    h = r["hook"]
    r["dram_peak_bytes"] = h.get("dram_peak_bytes")
    r["dram_total_bytes"] = h.get("dram_total_bytes")
    r["dram_largest_free_at_peak"] = h.get("dram_largest_free_at_peak")
    r["blocks_instrumented"] = len(h.get("instrumented") or [])
    if r["mechanism"] == "contention":
        r["verdict"] = "CONTENDED"
    elif r["stalled"]:
        r["verdict"] = "STALL"
        r["stall_kind"] = ("compute-active" if r["cpu_s_while_quiet"] > 0.5 * STALL_S
                           else "idle")
    elif not r["mechanism"] and (r["host_ram_floor_mb"] < HOST_RAM_FLOOR_MB
                                 or _host_killed(r["rc"], r["tail"])):
        r["verdict"] = "HOST_OOM"
        r["host_oom_evidence"] = _oom_killer_fired(r.get("model", ""))
    elif r["rc"] == 0:
        r["verdict"] = "PASS"
    else:
        r["verdict"] = "FAIL"
    return r


def fixture_for(cell: Cell, tokens: int, work: Path, depth) -> dict:
    """Build (or reuse) the fixture for one cell at `tokens`, and wire its cached MSA.

    `predict --msa_cache_only` resolves an alignment as `<msa_dir>/<seq_hash(sequence)>.a3m`, so
    the generated a3m is filed under that name. No network, no search, and the SAME alignment
    every run.
    """
    res = cell.residues(tokens)
    f = capacity_fixture.build(res, work / "fixtures", depth=depth)
    if cell.msa:
        from tt_bio.cache import seq_hash
        seq = [l for l in f["yaml"].read_text().splitlines() if "sequence:" in l][0]
        seq = seq.split("sequence:")[1].strip()
        msa_dir = work / "msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        target = msa_dir / f"{seq_hash(seq)}.a3m"
        target.write_text(f["a3m"].read_text())
        f["msa_dir"] = msa_dir
        f["msa_file"] = target
    f["tokens_requested"] = tokens
    f["tokens_padded"] = cell.padded(tokens)
    f["residues"] = res
    return f


def run_cell(worker: Worker, cell: Cell, work: Path, hookdir: Path, *, depth,
             bisect: bool) -> dict:
    """Screen at the bar, then the residency run, then bisect DOWN only if it failed.

    Target-first is the single biggest efficiency win here and it is the opposite of a ladder: a
    model that clears 1536 costs exactly one screen and one residency run, and nobody pays for
    640/768/896 to learn something the 1536 pass already proved.
    """
    rec = {"model": cell.model, "verb": cell.verb, "worker": repr(worker),
           "bar_tokens": TOKEN_BAR, "legs": []}
    if cell.model in EXEMPT:
        rec.update(verdict="SKIPPED", reason=EXEMPT[cell.model])
        return rec
    try:
        f = fixture_for(cell, TOKEN_BAR, work, depth)
    except Exception as exc:
        rec.update(verdict="ERROR", reason=f"fixture: {type(exc).__name__}: {exc}")
        return rec
    rec.update(tokens_requested=f["tokens_requested"], tokens_padded=f["tokens_padded"],
               residues=f["residues"], msa_rows_in_file=f["file_depth"] if cell.msa else 0,
               msa_rows_effective=f["effective_depth"] if cell.msa else 0)

    scr = _screen(worker, cell, f, work, hookdir)
    rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR))
    if scr["verdict"] == "CONTENDED" and wait_for_card(worker):
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR, retry=True))
    if scr["verdict"] == "CONTENDED":
        rec.update(verdict="CONTENDED", decided_by="screen", wall_s=scr["wall_s"],
                   reason="another process held the card; nothing was measured")
        return rec
    if scr["verdict"] in ("FAIL", "HOST_OOM"):
        # Definitive: a shape that cannot allocate once cannot allocate ever.
        rec.update(verdict=scr["verdict"], decided_by="screen",
                   mechanism=scr["mechanism"], wall_s=scr["wall_s"])
        if _no_weights(scr["tail"]):
            rec["verdict"] = "NO_WEIGHTS"
        elif bisect:
            rec["ceiling_tokens"] = _bisect(worker, cell, work, hookdir, depth, rec)
        return rec

    res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
    rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR))
    # A STALL is only the MODEL's stall if the chip is still dispatching. A card left dirty by an
    # earlier killed fold accepts an open and then hangs, with all threads idle and no error --
    # indistinguishable from a model hang from the outside, and attributing it to the model would
    # publish a ceiling that is really a housekeeping bug.
    if res["verdict"] in ("CONTENDED", "STALL") and not card_healthy(worker):
        rec["legs"][-1]["card_unhealthy_after"] = True
        if wait_for_card(worker):
            res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
            rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR, retry=True))
        else:
            rec.update(verdict="CARD_DIRTY", decided_by="residency", wall_s=res["wall_s"],
                       reason="the card stopped dispatching and did not recover; nothing was "
                              "measured. Reset it (tt-smi -r) and re-run this cell.")
            return rec
    elif res["verdict"] == "CONTENDED" and wait_for_card(worker):
        res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
        rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR, retry=True))
    rec.update(verdict=res["verdict"], decided_by="residency", mechanism=res["mechanism"],
               wall_s=res["wall_s"], dram_peak_bytes=res["dram_peak_bytes"],
               dram_total_bytes=res["dram_total_bytes"],
               dram_largest_free_at_peak=res["dram_largest_free_at_peak"],
               blocks_instrumented=res["blocks_instrumented"],
               host_ram_floor_mb=res["host_ram_floor_mb"])
    if res["verdict"] != "PASS" and _no_weights(res["tail"]):
        rec["verdict"] = "NO_WEIGHTS"
    elif res["verdict"] in ("FAIL", "STALL") and bisect:
        rec["ceiling_tokens"] = _bisect(worker, cell, work, hookdir, depth, rec)
    return rec


_NO_WEIGHTS = re.compile(
    r"could not (be )?(download|fetch)|No such file or directory.*\.(pt|safetensors|ckpt)"
    r"|HFValidationError|RepositoryNotFound|GatedRepo|401 Client Error|Connection error"
    r"|Weights .* not found|missing weights"
    # An UNUSABLE checkpoint is the same non-result as an absent one: the model was never
    # constructed, so nothing was ever asked of the device. Measured on nesso1, whose artifact on
    # this host carries atom-encoder layers the module does not declare, so from_pretrained dies
    # in load_state_dict(strict=True) before a single tensor reaches the card.
    r"|Error\(s\) in loading state_dict|Unexpected key\(s\) in state_dict"
    r"|Missing key\(s\) in state_dict|size mismatch for ", re.I)


def _no_weights(tail: str) -> bool:
    """A model whose checkpoint is absent or unusable on this host has NOT failed the bar.
    Recording that as a capacity failure would be the same lie in the other direction as scoring
    it a pass: it publishes a ceiling nobody walked."""
    return bool(_NO_WEIGHTS.search(tail or ""))


#: A leg whose worker died on SIGKILL with no device-side error is the HOST out of memory, not the
#: card. Measured: esmfold2 at 1504 residues was killed by the kernel OOM killer at 22.8 GiB
#: anon-rss on this 30 GB host, and the launcher only ever saw "SpawnProcess-1 exit -9". The
#: RSS-floor sampler can miss it outright, because the kill happens between two samples.
_SIGKILLED = re.compile(r"exit -9\b|exit 137\b|Killed\b|SIGKILL")


def _host_killed(rc: int, tail: str) -> bool:
    return rc in (-9, 137) or bool(_SIGKILLED.search(tail or ""))


def _oom_killer_fired(model: str) -> str | None:
    """The kernel's own record of the kill, so HOST_OOM is evidence and not an inference."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    hits = [ln for ln in out.splitlines() if "Out of memory: Killed process" in ln]
    return hits[-1].strip()[-200:] if hits else None


def _bisect(worker, cell, work, hookdir, depth, rec) -> int | None:
    """After a failure only: walk DOWN to report the real ceiling. Screen-first at each rung, so a
    rung that cannot even allocate costs seconds."""
    for rung in BISECT_RUNGS:
        try:
            f = fixture_for(cell, rung, work, depth)
        except Exception:
            continue
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=rung))
        if scr["verdict"] in ("FAIL", "HOST_OOM"):
            continue
        res = _residency(worker, cell, f, work, hookdir, rung)
        rec["legs"].append(dict(res, tier="residency", tokens=rung))
        if res["verdict"] == "PASS":
            return rung
    return None


# ---------------------------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------------------------


def reductions(depth, recycling, models) -> list[str]:
    """Every coverage reduction this run applied, by name. Silence here is the failure mode."""
    out = []
    if depth is not None:
        out.append(f"MSA depth truncated to {depth} rows. For the OF3 family the failing tensor "
                   f"scales with tokens x rows, so this REDUCES exposure to the exact wall that "
                   f"took OpenFold3 down at 614; the default is the committed source's full depth.")
    if recycling is not None:
        out.append(f"recycling_steps={recycling} instead of the production default. Peak tensor "
                   f"SHAPE is preserved (buffers are reused) but the number of alloc/free cycles "
                   f"drops, so Class B exposure is REDUCED. For RF3 recycling also drives MSA "
                   f"sampling (one i.i.d. sample per recycle), so it is not a free knob there.")
    out.append("diffusion_samples=1. Samples are independent and add no peak, so this costs no "
               "coverage.")
    out.append("The MSA is a committed alignment rather than a fresh search. Device memory does "
               "not care how the alignment was found; this costs no coverage.")
    out.append("Target-first: 1536 runs FIRST and a pass ends the cell. Only a failure pays for a "
               "downward bisect. Costs no coverage.")
    out.append("The fixture is polymer-only, so the LIGAND-TOKEN path is uncovered: no cell here "
               "sets ligand_tokens, and a model whose production input carries ligands is tested "
               "at 1536 polymer tokens only.")
    skipped = sorted(set(models) & set(EXEMPT))
    if skipped:
        out.append(f"Not driven by this gate at all: {', '.join(skipped)} (see EXEMPT for each "
                   f"reason). Reported SKIPPED, never PASS.")
    return out


def render(report: dict) -> str:
    g = report["geometry"]
    L = []
    L.append(f"CAPACITY GATE -- {report['bar_tokens']} tokens -- "
             f"{g.get('board_type') or '?'} / {g.get('arch') or '?'} on {g.get('host')}:"
             f"{g.get('card')}")
    if "dram" in g:
        d, l1 = g["dram"], g["l1"]
        L.append(f"  DRAM {d['banks']} banks x {d['bytes_per_bank']} B = "
                 f"{d['total_bytes'] / 2**30:.3f} GiB   "
                 f"L1 {l1['banks']} banks x {l1['bytes_per_bank']} B   grid {g.get('grid')}")
    L.append("  CAPACITY ONLY: allocates and completes. It does not and cannot check correctness "
             "-- that is scripts/full_parity_gate.py.")
    L.append("")
    hdr = f"  {'model':<14} {'verdict':<10} {'tok':>5} {'pad':>5} {'rows':>6} {'peak DRAM':>12} " \
          f"{'wall':>8}  by/mech"
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for r in report["results"]:
        peak = r.get("dram_peak_bytes")
        tot = r.get("dram_total_bytes")
        peak_s = "-" if not peak else (f"{peak / 2**30:.2f}G" +
                                       (f"/{100 * peak / tot:.0f}%" if tot else ""))
        L.append(f"  {r['model']:<14} {r['verdict']:<10} {r.get('tokens_requested', '-'):>5} "
                 f"{r.get('tokens_padded', '-'):>5} {r.get('msa_rows_effective', '-'):>6} "
                 f"{peak_s:>12} {str(r.get('wall_s', '-')):>8}  "
                 f"{r.get('decided_by', '-')}/{r.get('mechanism') or '-'}"
                 + (f"  ceiling~{r['ceiling_tokens']}" if r.get("ceiling_tokens") else "")
                 + (f"  [{r['reason'][:60]}]" if r.get("reason") else ""))
    L.append("")
    L.append("  Coverage reductions applied by this run:")
    for x in report["reductions"]:
        L.append(f"    - {x}")
    if report["coverage_gaps"]:
        L.append(f"  COVERAGE GAP: shipped models neither runnable nor exempted: "
                 f"{report['coverage_gaps']}")
    n = report["counts"]
    L.append("")
    L.append(f"  {n['PASS']} pass, {n['fail_like']} fail, {n['INCONCLUSIVE']} inconclusive, "
             f"{n['SKIPPED']} skipped, {n['NO_WEIGHTS']} no-weights, "
             f"{n['CONTENDED']} unmeasured of {len(report['results'])} cells")
    if n["INCONCLUSIVE"]:
        L.append("  INCONCLUSIVE is a Tier 1 result and is NOT a pass: one block per stack cannot "
                 "see the cumulative-residency class. Run without --tier screen for a verdict.")
    return "\n".join(L)


def main(argv=None) -> int:
    global STALL_S, TOKEN_BAR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=None,
                    help="comma-separated subset; default is every shipped model (derived)")
    ap.add_argument("--tokens", type=int, default=TOKEN_BAR,
                    help=f"the bar, in tokens; must be a multiple of {TOKEN_BUCKET}")
    ap.add_argument("--workers", default=None, help="host:card[,host:card...]")
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "perf" / "capacity" / "work")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="truncate MSA depth (a NAMED reduction, written into the report)")
    ap.add_argument("--recycling", type=int, default=None,
                    help="override recycling_steps (a NAMED reduction)")
    ap.add_argument("--tier", choices=("screen", "both"), default="both",
                    help="'screen' runs Tier 1 only; it can never report PASS")
    ap.add_argument("--no-bisect", action="store_true",
                    help="do not walk down to find the real ceiling after a failure")
    ap.add_argument("--stall-s", type=int, default=STALL_S)
    ap.add_argument("--no-card-reset", action="store_true",
                    help="never tt-smi -r a wedged card; report the rest of the run CARD_DIRTY "
                         "instead. A reset takes the board pair down, so use this when anything "
                         "else on the host is in flight.")
    ap.add_argument("--list", action="store_true", help="print the derived roster and exit")
    ap.add_argument("--record", action="store_true",
                    help="write docs/capacity_gate_baseline.json, pinning the ceiling table this "
                         "run measured against. tests/test_capacity_gate.py fails until this is "
                         "re-recorded after any ceiling change.")
    a = ap.parse_args(argv)

    STALL_S = a.stall_s
    if a.tokens % TOKEN_BUCKET:
        print(f"--tokens {a.tokens} is not a multiple of {TOKEN_BUCKET}: it would pad to "
              f"{TOKEN_BUCKET * -(-a.tokens // TOKEN_BUCKET)} internally and the gate would "
              f"report a size the hardware never saw.", file=sys.stderr)
        return 2
    TOKEN_BAR = a.tokens

    models = [m.strip() for m in a.models.split(",")] if a.models else roster()
    if a.list:
        for m in models:
            print(f"{m:<16} {_verbs().get(m, '?'):<9} "
                  f"{'EXEMPT: ' + EXEMPT[m][:70] if m in EXEMPT else 'covered'}")
        print(f"\ncoverage gaps: {coverage_gaps() or 'none'}")
        return 0

    unknown = sorted(set(models) - set(roster()))
    if unknown:
        print(f"not shipped models: {unknown}", file=sys.stderr)
        return 2

    work = a.work_dir
    work.mkdir(parents=True, exist_ok=True)
    hookdir = hook_dir(work)
    workers = parse_workers(a.workers) if a.workers else [Worker(local_host(), 0, True)]

    if not card_healthy(workers[0]):
        print(f"{workers[0]} cannot dispatch a trivial program. Every leg would fail or hang and "
              f"the gate would record capacity failures nobody walked. Reset it "
              f"(tt-smi -r {workers[0].card}) and re-run.", file=sys.stderr)
        return 3

    report = {
        "bar_tokens": TOKEN_BAR, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                     capture_output=True, text=True).stdout.strip()),
        "workers": [repr(w) for w in workers],
        "geometry": geometry(workers[0]),
        "coverage_gaps": coverage_gaps(),
        "reductions": reductions(a.depth, a.recycling, models),
        "results": [],
        "note": "CAPACITY ONLY. Allocates and completes. Says nothing about whether the output is "
                "correct; that is scripts/full_parity_gate.py, which this does not substitute for.",
    }
    # Flushed: a gate run is watched through a redirected log, where an unflushed header sits in
    # the buffer for the whole campaign and the board geometry it carries is what a reader needs
    # FIRST to know the numbers are comparable.
    print(render(dict(report, results=[], counts=dict.fromkeys(
        ("PASS", "fail_like", "SKIPPED", "NO_WEIGHTS", "INCONCLUSIVE", "CONTENDED"), 0))),
        flush=True)

    #: Cards this run has given up on: a wedge that a reset could not clear. Every cell still
    #: owed on such a card is reported CARD_DIRTY, because nothing was measured on it.
    dead: dict[str, str] = {}
    all_cells = list(cells(models, depth=a.depth, recycling=a.recycling))

    for i, cell in enumerate(all_cells):
        w = workers[i % len(workers)]
        if repr(w) in dead:
            r = {"model": cell.model, "verdict": "CARD_DIRTY", "worker": repr(w), "wall_s": 0.0,
                 "reason": f"not run: {dead[repr(w)]}"}
            report["results"].append(r)
            print(f"  -> {r['model']:<14} {r['verdict']:<10} 0s  {r['reason'][:80]}", flush=True)
            _finish(report)
            (a.report or work / "report.json").write_text(
                json.dumps(report, indent=1, default=str))
            continue
        t0 = time.monotonic()
        try:
            r = run_cell(w, cell, work, hookdir, depth=a.depth, bisect=not a.no_bisect) \
                if a.tier == "both" else _screen_only(w, cell, work, hookdir, a.depth)
        except Exception as exc:
            r = {"model": cell.model, "verdict": "ERROR",
                 "reason": f"{type(exc).__name__}: {exc}"}
        r.setdefault("wall_s", round(time.monotonic() - t0, 1))
        report["results"].append(r)
        print(f"  -> {r['model']:<14} {r['verdict']:<10} {r.get('wall_s')}s "
              f"{r.get('mechanism') or ''} {str(r.get('reason', ''))[:80]}", flush=True)

        # A device-side fatal can leave the chip open-able but not dispatching, so the NEXT cell
        # would hang in tt-bio's dispatch probe and be recorded as this gate's own kind of
        # failure. Checked only after a failure, so a clean run pays nothing for it.
        if r["verdict"] in ("FAIL", "STALL", "ERROR", "CARD_DIRTY") and not a.no_card_reset:
            ok, how = recover_card(w, workers)
            r["card_after"] = how
            if not ok:
                dead[repr(w)] = how
                print(f"     {w}: {how}", flush=True)
            elif how != "still dispatching":
                print(f"     {w}: {how}", flush=True)

        _finish(report)
        (a.report or work / "report.json").write_text(json.dumps(report, indent=1, default=str))

    _finish(report)
    print(flush=True)
    print(render(report), flush=True)
    out = a.report or work / "report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nreport: {out}")
    if a.record:
        print(record_baseline(report, partial=bool(a.models)))
    return 0 if report["counts"]["fail_like"] == 0 and not report["coverage_gaps"] else 1


BASELINE = REPO_ROOT / "docs" / "capacity_gate_baseline.json"


def ceilings_fingerprint() -> str:
    """A stable hash over every published ceiling, so moving any row is detectable.

    The published ceilings a user sees live in the serving platform, which tt-bio does not import.
    The engine's own copy is tt_bio.size_limits.CEILINGS, and that is what a ceiling change edits
    here, so that is what gets pinned.
    """
    import hashlib
    from tt_bio import size_limits as sl
    rows = []
    for model in sorted(sl.CEILINGS):
        for arch in sorted(sl.CEILINGS[model]):
            c = sl.CEILINGS[model][arch]
            rows.append([model, arch, c.residues, c.pass_at, c.binds, c.mechanism,
                         c.msa_rows, c.counts])
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:16]


def record_baseline(report: dict, *, partial: bool) -> str:
    """Merge this run's cells into docs/capacity_gate_baseline.json.

    A partial run (--models) updates only the cells it measured and leaves the rest standing, so
    re-measuring one model does not silently erase the others' recorded results.
    """
    prior = {}
    if BASELINE.exists():
        try:
            prior = json.loads(BASELINE.read_text())
        except ValueError:
            prior = {}
    cells = prior.get("cells", {}) if partial else {}
    # A partial run must not carry cells measured at a DIFFERENT bar across into this baseline.
    # Measured: raising the bar 1504 -> 1536 and re-recording six cells left the two esmfold2
    # cells from the 1504 run inside a file stamped 1536, where they read as current evidence.
    # Pair tensors scale roughly quadratically, so a 1504 result is not a 1536 result.
    cells = {m: c for m, c in cells.items()
             if (c or {}).get("tokens_requested") in (None, TOKEN_BAR)}
    for r in report["results"]:
        cells[r["model"]] = {k: r.get(k) for k in
                             ("verdict", "tokens_requested", "tokens_padded", "residues",
                              "msa_rows_effective", "dram_peak_bytes", "dram_total_bytes",
                              "wall_s", "mechanism", "decided_by", "ceiling_tokens", "reason")}
    BASELINE.write_text(json.dumps({
        "bar_tokens": report["bar_tokens"],
        "recorded": report["started"],
        "tree": report["tree"],
        "dirty_tree": report["dirty"],
        "geometry": report["geometry"],
        "ceilings_fingerprint": ceilings_fingerprint(),
        "reductions": report["reductions"],
        "cells": cells,
        "note": "CAPACITY ONLY: allocates and completes. Not a correctness record.",
    }, indent=1, default=str) + "\n")
    return f"recorded {BASELINE} ({len(cells)} cells, ceilings {ceilings_fingerprint()})"


def _screen_only(worker, cell, work, hookdir, depth) -> dict:
    rec = {"model": cell.model, "verb": cell.verb, "worker": repr(worker), "legs": []}
    if cell.model in EXEMPT:
        rec.update(verdict="SKIPPED", reason=EXEMPT[cell.model])
        return rec
    f = fixture_for(cell, TOKEN_BAR, work, depth)
    rec.update(tokens_requested=f["tokens_requested"], tokens_padded=f["tokens_padded"],
               residues=f["residues"],
               msa_rows_effective=f["effective_depth"] if cell.msa else 0)
    scr = _screen(worker, cell, f, work, hookdir)
    rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR))
    if scr["verdict"] == "CONTENDED" and wait_for_card(worker):
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR, retry=True))
    rec.update(verdict=scr["verdict"], decided_by="screen", mechanism=scr["mechanism"],
               wall_s=scr["wall_s"], stacks_truncated=scr.get("stacks_truncated"),
               block_calls=scr.get("block_calls"), note=scr.get("note"))
    # Not `!= "FAIL"`: an absent or unusable checkpoint is exactly what makes a leg exit
    # nonzero, so guarding this on "not already a FAIL" would skip every case it is for.
    if scr["verdict"] in ("FAIL", "HOST_OOM") and _no_weights(scr["tail"]):
        rec["verdict"] = "NO_WEIGHTS"
    return rec


def _finish(report: dict) -> None:
    v = [r["verdict"] for r in report["results"]]
    report["counts"] = {
        "PASS": v.count("PASS"),
        "fail_like": sum(v.count(x) for x in ("FAIL", "STALL", "HOST_OOM", "ERROR")),
        "CONTENDED": v.count("CONTENDED") + v.count("CARD_DIRTY"),
        "SKIPPED": v.count("SKIPPED"),
        "NO_WEIGHTS": v.count("NO_WEIGHTS"),
        "INCONCLUSIVE": v.count("INCONCLUSIVE"),
    }


if __name__ == "__main__":
    sys.exit(main())
