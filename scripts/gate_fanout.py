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
  ladder:<model>@<lo>-<hi>  release_gate.py --model size-ladder --size-ladder-models <model> --size-ladder-rungs <group>
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
  record:abb3        perf/abb3_port's model and step gate records (ABB3_RECORD), only with
                     --record-abb3, on a p300c card; the updated files land in <out>/abb3/

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

READS. That key changes with any code change, so on its own every staging reruns every leg. Each
gate venv therefore carries scripts/gate_reads.py, which records every tree file the leg's Python
processes import, open for reading or list. A PASS row keeps that list and a scope key: the key
above with only the tracked non-Python files in place of the whole tree (C++ kernels and shell
scripts are read by something other than Python, so any change to one still reruns every leg).
A later commit in the same scope reuses the PASS when every file the leg read is the same blob
there, so a staging that touched one model reruns that model's legs and the ones that import its
files, not the gate. A row without a list reuses on identical code only.

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
Pool legs skip gate_guard's load ceiling (see LOAD_GUARDED).

FIRST. A release never waits on our own experiments. A pool host queues its legs at the prio
it names (0, ahead of every grading job), and a flock host shared with other rows sets
`"first": true`: each leg then takes its card through scripts/flock_first.sh, which stops the
other waiters on that card's lock (never the holder) until the gate is done with the card.

PERF A/B. Committed perf cells age: a cell recorded in August on another software stack fails a
release that changed nothing. `"perf_against": "<tag>"` on a host checks the timed leg out at the
last release (root/trees/<tag>) and runs perf_regression.py --against it: the release's own
numbers are measured on the same card in the same session, then the commit is gated on them.
`"perf_against": "last"` is the newest v* tag, so the reference moves with every release and a
card type with no committed cell is compared with a real measurement rather than seeded.
On a PASS the runner writes the reference's re-measured machine block into <out>/seeded/<arch>/
perf_baselines.json for the release commit; the A/B leg's key leaves the committed baseline out.
Ladder legs on such a host time their exponent rungs against the same tree, folds interleaved on
their chip (release_gate.py --size-ladder-against), and their lever census still reads the
committed baseline.

SEEDING. A perf leg on a card type and machine with no baseline has nothing to regress against:
perf_regression.py records the run as the first baseline and prints GATE SEEDED. The leg then
reports SEEDED, not PASS, and the runner copies the seeded docs/perf_baselines.json to
<out>/seeded/<arch>/ to commit; the next gate on that card is a regression check. A host can also
re-seed on purpose with `"args": {"perf": ["--update-baseline", "--note", "<why>"]}`.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
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

# Per-leg budgets (s). A budget is a hang guard, not an expected time, so a leg known to need longer
# gets BUDGET_X times that instead (Leg.known_s): rf3's Wormhole record took 4.6 h of folds that each
# ran at its baseline's speed and was stopped at the flat 4 h (rec-c2b4b898e, 10-14).
BUDGET = {"parity": 14400, "rg": 7200, "ladder": 14400, "capacity": 14400, "ux": 3600,
          "pytest_device": 7200, "perf": 7200, "check": 900, "packaging_smoke": 1800,
          "pytest_cpu": 5400, "bc2": 3600, "record": 14400}
# Expected wall (s) per family when the ledger has no history for a leg, from the 0.13.1 and
# 0.14.0 chains. Legs start longest first so the gate ends close to its longest leg.
EXPECT = {"parity": 900, "rg": 900, "ladder": 5400, "capacity": 1800, "ux": 300,
          "pytest_device": 3000, "perf": 3600, "check": 60, "packaging_smoke": 600,
          "pytest_cpu": 1800, "bc2": 1200, "record": 2700}
BUDGET_X = 1.5
#: timeout(1)'s exit status when it stopped a leg at its budget.
TIMED_OUT = 124
TIMED = {"perf"}
#: Results that do not fail the gate. REUSED is a ledger hit of one of the others.
OK = ("PASS", "REUSED", "BLOCKED", "SEEDED")
SEEDED_FIRST = "\nGATE SEEDED"   # perf_regression.py recorded a card's first baseline
LOG_CAP = 20 << 20
POOL = "pool"            # --workers host:pool, a slot in the host's chip pool
POOL_CARD = "@@CARD@@"   # stands for $CHIP in a pool job, filled in when the pool starts it
POOL_POLL_S = 30
#: A chip marked bad by hand. feed.py skips a chip with ~/spd/locks/chipN.BAD; fleet.sh skips a card
#: with state/cardblock-<host>-<N>, on pc or on the host itself. A worker reads all three before
#: each leg: rec-c2b4b898e (10-14) ran protenix-v2 on qb2:0, the chip it deadlocks on, because
#: gate.json listed it as a worker though both marks were set.
BAD_MARKS = ("~/spd/locks/chip{card}.BAD", "~/.coworker/state/cardblock-{host}-{card}")
LOCAL_BLOCKS = Path(os.environ.get("COWORKER_DIR", Path.home() / ".coworker")) / "state"
LOAD_REFUSAL = "refusing to run the gate. host load:"
#: The scripts that refuse to start above gate_guard's load ceiling. On a pool host the ceiling is
#: the wrong guard: the pool's per-chip queue bounds the fan-out, and a busy pool keeps a Galaxy's
#: 1-min loadavg above 1.5x nproc on its own (g114 at ~110 on 64 cores, 10-12), so every leg
#: refused and was requeued for hours. Timing stays honest without it: release_gate samples the
#: load beside the clock on every fold and the speed bar voids a rung read above the ceiling.
LOAD_GUARDED = ("scripts/release_gate.py", "scripts/full_parity_gate.py")
LOAD_RETRY_S, LOAD_WAIT_MAX_S = 300, 8 * 3600
#: ssh's own words when the connection, not the leg, failed (exit 255). qb1 dropped off the LAN for
#: minutes on 2026-10-09 and five BH legs were recorded FAIL without running. A Galaxy reached
#: through the japanfold-ssh jump host reports a dropped hop as "Connection closed by UNKNOWN port
#: 65535": rel-412d01dfe (10-17 08:4xZ) recorded five WH legs FAIL in 7-12 s that way.
SSH_LOST = re.compile(r"^(ssh: connect to host .*|Timeout, server .* not responding\.|"
                      r"client_loop: send disconnect: .*|Connection to .* closed by remote host\.|"
                      r"Connection closed by \S+ port \d+|"
                      r"kex_exchange_identification: .*|ssh_dispatch_run_fatal: Connection to .*)$", re.M)
SSH_LOST_RETRY_S, SSH_LOST_TRIES = 120, 30
#: A full disk on the leg's host. Another job filled qb1's root on 2026-10-10 and 116 BH legs were
#: recorded FAIL in an hour, each at its first mkdir. Wait for space and run the leg again.
DISK_FULL = re.compile(r"No space left on device")
DISK_FULL_RETRY_S, DISK_FULL_TRIES = 300, 12
#: Another process held the leg's card through its device lease without taking the card's flock.
#: The timed perf leg of gate 92d5924f5 lost two models to it on qb1 card 3 (2026-10-10 18:4xZ) and
#: a third read -19.8 %. Nothing about the tree was measured, so the whole leg runs again.
CONTENDED = re.compile(r"device contention, nothing ran|Refusing to open it concurrently")
CONTENDED_RETRY_S, CONTENDED_TRIES = 300, 3
#: release_gate stops when a fold timed out and the card's reset was refused (its board-pair sibling
#: was busy). Opening that card again from the same leg is unsafe, but a pool probes, and if need be
#: resets, every chip before it hands it out: g108 chip 0 did this on ladder:boltz2 on 2026-10-10 and
#: probed OK minutes later. Run such a pool leg once more; a second stop is the leg's result.
CARD_STUCK = re.compile(r"a fold timed out and its card could not be reset")
#: A pool leg that died opening its chip measured nothing about the tree. .108's pool hands out chips
#: an earlier job left dirty and resets them after the job that finds out: gate 412d01dfe recorded six
#: WH legs FAIL on 2026-10-17 09:00-09:30Z this way (chips 3, 9, 10, 13, 19), each the first job on its
#: chip, each chip "probe: OK after reset" a minute later. Run such a leg again, a bounded number of times.
INIT_FAILED = re.compile(r"failed to initialize FW|ARC core .* failed to start|"
                         r"Failed to set initial power state|Read unexpected run_mailbox value")
INIT_FAILED_TRIES = 2
#: A pool job notes the host's boot id when it starts. g114 rebooted under five legs of two gates on
#: 2026-10-11 19:30Z; they never wrote an exit code and their runners waited for one for over an
#: hour. A different boot id and no exit code means the leg died with the host: run it once more.
BOOT_ID = "/proc/sys/kernel/random/boot_id"
REBOOTED = "rebooted under the leg"
#: A Galaxy lent out of the pool (spd/galaxy/lend_cfd.sh writes the window's start epoch here and
#: PAUSEs the pool through it). pool.sh starts no job whose cap reaches the window, so a leg queued
#: there waits out the whole lend: capacity:opendde of gate cb55af219 sat on .114 for hours on
#: 2026-10-10 while .107 and .108 had chips. Such a host takes no leg the lend would hold, and a leg it
#: queued before the lend was booked is taken back and handed to another worker.
LEND_AT = "~/spd/galaxy/LEND-AT"
LEND_READ_S = 300
REQUEUED = "requeued"     # run_in_pool's note for an attempt the pool put back after a chip recall
REQUEUE_GRACE_S = 3       # the pool requeues right after the job exits; give it that long before reading rc
WITHDRAWN = -1           # run_in_pool's code for "taken back from the queue before it started"
#: glx_elastic gives a lent Galaxy back to JapanFold by SIGINTing every pool job and touching the pool's
#: RETURNED. .107 went back at 07:34Z on 2026-10-16, 35 min into the WH legs of gate 5e250928a, and 14 of
#: them were recorded FAIL (folds "exited -2"). The job notes a give-back that came after it started; such
#: a leg measured nothing about the tree, so it goes back on the queue and waits for the next lend.
GIVEN_BACK = "the Galaxy was given back under the leg"
GIVEN_BACK_TRIES = 3
#: A leg failed on a named card and the card is what failed: qb2 chip 1 stopped initialising its
#: firmware at 06:07Z on 2026-10-12 and the runner handed it 40 more legs of gate 026b5c71c, each
#: recorded FAIL in 15-90 s. After a failed leg on a named card the runner probes the card
#: (capacity_gate's open + dispatch probe, under the card's flock). A card that fails the probe is
#: retired for the run and its leg goes back on the queue, once: a leg that kills a second card is
#: recorded as it ran. A leg no card is left to run is recorded UNRUN, which fails the gate.
CARD_PROBE = ("import sys; sys.path.insert(0, 'scripts'); "
              "from capacity_gate import _CARD_PROBE; exec(_CARD_PROBE)")
CARD_PROBE_S, CARD_PROBE_LOCK_S = 300, 1800
#: A card is dead only when it fails every one of these probes, a gap apart. One failed probe right
#: after a failed leg is not enough: qb1 card 2 failed one on 2026-10-12 08:01Z after a protenix-v2
#: record leg's fold workers died, was retired, and was running an SPD bench minutes later.
CARD_PROBE_TRIES, CARD_PROBE_GAP_S = 3, 60
#: perf/abb3_port/{model_gate,step_gate_default}_qb1c3.txt are records with a RECORDED-AT line, and
#: test_recorded_claims fails the release commit when their scripts' import closure moved since, which
#: it does in most stagings. They were re-run by hand on qb2 card 3 for every release since 0.13.0.
#: `--record-abb3` runs them as a leg on a p300c card (the step gate's DRAM peak and its --micro 8
#: OOM are that board's) and checks them against the committed records: every ok/FAIL and gradient
#: row identical, the same taped parameter count and DRAM peak, losses within ABB3_LOSS_TOL, and
#: --micro 8 still out of memory. On a match it writes both files with RECORDED-AT at the gate's
#: commit and a RE-RECORDED stanza; the runner fetches them to <out>/abb3/ for the release commit.
#: Any difference fails the leg and names it, for a human to judge.
ABB3_LEG, ABB3_CARD_TYPES, ABB3_LOSS_TOL = "record:abb3", ("p300c",), 2e-4
ABB3_FILES = ("perf/abb3_port/model_gate_qb1c3.txt", "perf/abb3_port/step_gate_default_qb1c3.txt")
ABB3_RECORD = r"""
import re, subprocess, sys, textwrap, time
from pathlib import Path
out, sha, worker, card_type, tol = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
MODEL, STEP = (Path(f) for f in sys.argv[6:8])

def run(*args):
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode

t0 = time.gmtime()
arms = []
for sigma in ("0.05", "0.3"):
    for tokens in ("64", "256"):
        text, rc = run("scripts/abb3_port/model_gate.py", "--tokens", tokens, "--sigma", sigma)
        arms.append((sigma, tokens, text, rc))
step, step_rc = run("scripts/abb3_port/step_gate.py")
micro8, micro8_rc = run("scripts/abb3_port/step_gate.py", "--micro", "8")
t1 = time.gmtime()
(out / "model_gate.txt").write_text("".join(f"== model_gate --tokens {t} --sigma {s}\n{x}\nrc={rc}\n"
                                             for s, t, x, rc in arms))
(out / "step_gate.txt").write_text(f"{step}\nrc={step_rc}\n")
(out / "step_gate_micro8.txt").write_text(f"{micro8}\nrc={micro8_rc}\n")

ROW = re.compile(r"^  (?:ok|FAIL) .*$|^  grad (?:on|off) .*$", re.M)
rows = lambda t: ROW.findall(t)
def steps(t):
    return {int(n): float(v) for n, v in re.findall(r"^  step +(\d+) .* loss ([\d.]+)", t, re.M)}
def one(pat, t):
    m = re.search(pat, t, re.M)
    return m.group(1) if m else None
mt, st = MODEL.read_text(), STEP.read_text()
st_default = st.split("== default:", 1)[-1]
new_rows = [r for _, _, x, _ in arms for r in rows(x)]
bad = []
if not new_rows or new_rows != rows(mt):
    bad.append(f"model gate rows differ from {MODEL}: {len(new_rows)} rows against {len(rows(mt))}")
    for a, b in zip(rows(mt), new_rows):
        if a != b:
            bad.append(f"  was {a.strip()}\n  now {b.strip()}")
    bad += [f"  now {b.strip()}" for b in new_rows[len(rows(mt)):]]
loss, was = steps(step), steps(st_default)
if step_rc or not loss or set(loss) != set(was) or any(abs(loss[n] - was[n]) > tol for n in was):
    bad.append(f"step gate rc {step_rc}, losses {loss} against {was} (tolerance {tol})")
for what, pat in (("taped parameters", r"taped device parameters: (\d+)"), ("DRAM peak", r"^DRAM: peak (\d+) MB")):
    if one(pat, step) is None or one(pat, step) != one(pat, st_default):
        bad.append(f"step gate {what} {one(pat, step)} against {one(pat, st_default)}")
oom = one(r"Not enough space to allocate (\d+) B", micro8)
if not micro8_rc or oom is None:
    bad.append(f"step gate --micro 8 exited {micro8_rc} without the out-of-memory the record carries")
if bad:
    sys.exit("abb3 records do not match:\n" + "\n".join(bad))

prev = one(r"^RECORDED-AT: ([0-9a-f]{9})", mt)
when = f"{time.strftime('%H:%M', t0)}-{time.strftime('%H:%M', t1)}Z {time.strftime('%Y-%m-%d', t1)}"
where = (f"RE-RECORDED by the release gate (gate_fanout.py record:abb3) at {sha[:9]}, {when}, "
         f"{worker}, {card_type}, the gate's wheel venv with the tree on PYTHONPATH.")
fails = [f"{r.split()[1]} at {r.split()[2]} (sigma {s}, {t} tokens)" for s, t, x, _ in arms
         for r in rows(x) if r.startswith("  FAIL")]
model_note = (f"{where} Every ok/FAIL row and every gradient row is identical to the {prev} record"
              + (f", which still FAILs {'; '.join(fails)}" if fails else "") + ". NO TIMING CLAIM.")
clock = re.search(r"AICLK during the run: card \d+ min (\d+) median (\d+)", step)
median = one(r"^STEP: median ([\d.]+) s", step)
taped, peak = one(r"taped device parameters: (\d+)", step), one(r"^DRAM: peak (\d+) MB", step)
losses = " / ".join(f"{loss[n]:.4f}" for n in sorted(loss))
step_note = (f"{where} Losses {losses} at steps {', '.join(map(str, sorted(loss)))}, {taped} taped "
             f"parameters and the {peak} MB DRAM peak as in the {prev} record; "
             f"--micro 8 still OOMs, on a {int(oom):,} B request. Median {median} s"
             + (f" at AICLK median {clock.group(2)} (min {clock.group(1)})" if clock else "")
             + ". A fit record, not a perf claim.")

def rerecord(text, note):
    text = re.sub(r"^RECORDED-AT: [0-9a-f]+ ", f"RECORDED-AT: {sha} ", text, count=1, flags=re.M)
    lines = text.splitlines(True)
    i = next(i for i, ln in enumerate(lines) if i and not ln.startswith("#") and not ln.startswith("RECORDED-AT:"))
    stanza = [f"# {ln}\n" for ln in textwrap.wrap(note, 104)] + ["#\n"]
    return "".join(lines[:i] + stanza + lines[i:])
for src, note in ((MODEL, model_note), (STEP, step_note)):
    (out / src.name).write_text(rerecord(src.read_text(), note))
print(f"abb3 records match {prev}; re-recorded at {sha[:9]} in {out}")
"""
QUEUED: dict = {}         # pool job file -> host, taken back if the runner is stopped
RUNNING: dict = {}        # leg out dir -> host, its attempt ended if the runner is stopped
CARD_FREE = ("check", "packaging_smoke", "pytest_cpu")
PY312 = {"pytest_device", "pytest_cpu", "bc2"}
# Arms the parity gate already runs in-process by calling release_gate's own runner with the
# same arguments (full_parity_gate.py: run_inprocess -> release_gate.run_<arm>). Running the rg
# arm as well folds the same thing twice.
PARITY_COVERS_RG = {"boltzgen", "opendde-abag", "capacity", "nesso1", "rf3-1024aa"}

CAP_OWED = "host-threads"   # an OWED entry: no granted host times this ladder at its baseline's cap


#: A size ladder runs as one leg per rung group, so the longest ladder no longer sets the gate's
#: wall clock: Wormhole opendde was a single 13592 s leg of gate 026b5c71c (2026-10-12). The first
#: group keeps 256, 512 and 768 together, the two intervals the exponent check reads. release_gate's
#: `--size-ladder-rungs` filters the model's own ladder and the check scores only the rungs it walked.
#: A release_gate refuses a rung subset in check mode (a lone subset run would pass without the
#: rungs where levers go dark), so the runner splits only a tree whose release_gate sets
#: SIZE_LADDER_PARTS and takes `--size-ladder-part`; the groups together cover every rung.
LADDER_GROUPS = ((0, 768), (769, 1100), (1101, 1300), (1301, 1 << 20))


def ladder_model(name: str) -> str:
    """'ladder:opendde@896-1024' -> 'opendde' (also 'record:m' -> 'm')."""
    return name.split(":", 1)[1].split("@", 1)[0]


def ladder_rungs(roster: dict, card_type: str, model: str) -> list:
    """The rungs `model`'s ladder walks on `card_type` (ENUMERATE's "rungs"; "*" for a board with
    no rungs of its own)."""
    rungs = roster.get("rungs", {})
    return (rungs.get(card_type) or rungs.get("*", {})).get(model, [])


def rungs_fit(roster: dict, host, lg) -> bool:
    """Whether `host`'s card walks at least one of a ladder group leg's rungs. A p300c walks only
    the shared ladder, so the 1152-1536 groups go to a p150a; an empty group would fold nothing."""
    if lg.family != "ladder" or not lg.rungs:
        return True
    return bool(set(lg.rungs) & set(ladder_rungs(roster, host.cfg["card_type"], ladder_model(lg.name))))


def type_fits(host, lg) -> bool:
    """Whether `host`'s card type may run `lg` (Leg.card_types)."""
    return not lg.card_types or host.cfg["card_type"] in lg.card_types


def owed_by(roster: dict, card_type: str, lg) -> list[str]:
    """Levers a ladder leg's baseline lacks on this card type; every other leg owes none."""
    if lg.family != "ladder":
        return []
    return roster.get("owed", {}).get(card_type, {}).get(ladder_model(lg.name), [])


def ladder_caps(docs, models) -> dict:
    """{card_type: {model: host_threads}}: the cap each ladder baseline was timed at (None = every
    core). release_gate refuses to score a ladder run at another cap, so a ladder leg may only run
    on a host whose `--host-threads` matches. A model's own fragment wins over the merged file, as
    in release_gate. Runs inside the tree under test (ENUMERATE)."""
    import json
    from pathlib import Path
    docs, out = Path(docs), {}
    for f in [docs / "size_ladder_baseline.json", *sorted((docs / "size_ladder_baseline.d").glob("*.json"))]:
        if f.exists():
            for card, c in json.loads(f.read_text()).get("cards", {}).items():
                for m in c.get("models", {}):
                    if m in models:
                        out.setdefault(card, {})[m] = c.get("host_threads")
    return out


def cap_fits(roster: dict, host, lg) -> bool:
    """Whether `host` times `lg` at the cap its baseline was timed at (always, for a non-ladder leg
    or a model with no baseline on this card type: that absence is reported by the leg itself)."""
    if lg.family != "ladder":
        return True
    caps = roster.get("caps", {}).get(host.cfg["card_type"], {})
    model = ladder_model(lg.name)
    return model not in caps or caps[model] == host.ladder_cap


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


ENUMERATE = inspect.getsource(owed_levers) + inspect.getsource(ladder_caps) + r"""
import json, full_parity_gate as f, release_gate as r, capacity_gate as c, ux_regression as u
import lever_census
print(json.dumps({"parity": [l.id for l in f.LEGS if not l.opt_in], "rg": r.default_arms(),
                  "ladder": list(r.SIZE_LADDER_MODELS), "capacity": c.roster(), "ux": u.ALL_LEGS,
                  "owed": owed_levers("docs", lever_census.LEVERS, r.SIZE_LADDER_MODELS),
                  "caps": ladder_caps("docs", r.SIZE_LADDER_MODELS),
                  "rungs": {ct: {m: list(r._size_ladder_model_rungs(m, card=ct)) for m in r.SIZE_LADDER_MODELS}
                            for ct in [*r.SIZE_LADDER_CARD_RUNGS, "*"]}
                           if getattr(r, "SIZE_LADDER_PARTS", False) else {}}))
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


@functools.lru_cache(maxsize=None)
def ladder_ab(sha: str, repo: Path = REPO) -> bool:
    """Whether `sha`'s release_gate.py takes --size-ladder-against. This runner gates commits older
    than itself (a fix chain's base, a re-gate), and their ladder legs keep the recorded reference."""
    p = subprocess.run(["git", "-C", str(repo), "grep", "-q", "-e", "--size-ladder-against", sha, "--",
                        "scripts/release_gate.py"], capture_output=True)
    return p.returncode == 0


def _same_code(sha: str, content: str, repo: Path = REPO) -> bool:
    """True when `sha` is in this repo and holds the code `content` hashes."""
    try:
        return content_hash(sha, repo) == content
    except subprocess.CalledProcessError:
        return False


def rerecorded(tree: str, reference: str, note: str) -> str:
    """`tree`'s perf_baselines.json with every machine block the A/B reference run rewrote
    (perf_regression.py notes it "A/B reference for ...") taken from `reference`: the last
    release measured on that card in the gate's own session, so the committed cells are current
    instead of dating from whenever the machine was first seeded."""
    t, ref = json.loads(tree), json.loads(reference)
    ab = lambda x: str(x.get("note", "")).startswith("A/B reference for")
    for ct, entry in ref.get("cards", {}).items():
        for mid, blk in entry.get("machines", {}).items():
            got = {m: {**v, "note": note} for m, v in blk.get("models", {}).items() if ab(v)}
            if not got:
                continue
            mine = t.setdefault("cards", {}).setdefault(ct, {}).setdefault("machines", {}).setdefault(mid, {})
            mine.setdefault("models", {}).update(got)
            if ab(blk):
                mine.update(date=blk.get("date"), tt_bio_version=blk.get("tt_bio_version"), note=note)
    return json.dumps(t, indent=2) + "\n"


def baseline_hash(sha: str, family: str, repo: Path = REPO, model: str = "", cards=()) -> str:
    """Blob ids of the baseline files `family` compares against at `sha` ('' for none).

    A ladder leg for `model` reads the monolith plus that model's own fragment, so splicing one
    model's rows reruns that model's ladder leg only. Given the card types the leg can run on,
    it hashes only what a check on them reads (each file's card stamp and `model` entry for
    those cards), so recording Wormhole rows does not rerun the Blackhole legs."""
    ps = BASELINES.get(family)
    if not ps:
        return ""
    if family == "ladder" and model:
        ps = (ps[0], f"{ps[1]}{model}.json")
        if cards:
            return _ladder_view_hash(_commit(sha, repo), repo, ps, model, tuple(sorted(cards)))
    rows = [ln for ln in _ls_tree(sha, repo) if ln.split("\t", 1)[-1] in ps or (
        not model and ln.split("\t", 1)[-1].startswith(ps))]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _ladder_view_hash(sha: str, repo: Path, paths: tuple, model: str, cards: tuple) -> str:
    view = []
    for path in paths:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{path}"], capture_output=True, text=True)
        blocks = (json.loads(r.stdout) if r.returncode == 0 else {}).get("cards") or {}
        view.append({c: {**{k: v for k, v in blocks[c].items() if k != "models"},
                         "models": (blocks[c].get("models") or {}).get(model)}
                     for c in cards if c in blocks})
    return sha256(view)


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
                "scripts/gate_host_prep.sh", "scripts/flock_first.sh", "scripts/gate_reads.py",
                "scripts/release_train.py", "tests/test_gate_fanout.py", "tests/test_release_next.py",
                "tests/test_release_train.py")
#: The runner before flock_first.sh joined it, the key every ledger row before 2026-10-10 03Z
#: was written under.
RUNNER_FILES_0 = tuple(f for f in RUNNER_FILES if f not in ("scripts/flock_first.sh", "scripts/gate_reads.py",
                                                         "scripts/release_train.py", "tests/test_release_train.py"))


#: Data formats only Python parses in this tree: gate_reads.py sees every read of one, so the
#: reuse scope leaves them to the leg's recorded reads like a .py file.
PY_DATA = (".json", ".jsonl", ".yaml", ".yml", ".csv", ".tsv")


def content_hash(sha: str, repo: Path = REPO, baselines: bool = False, tests: bool = False,
                 runner: tuple = RUNNER_FILES, python: bool = True, data: bool = True) -> str:
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
    ledger row before 2026-10-09 21Z was written under. `python=False` drops the .py files and
    `data=False` the PY_DATA files too: the reuse scope, whose rows carry the files they read."""
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
        if (path.endswith(".md") or path in recorded or path in runner or path in only
                or (not python and path.endswith(".py")) or (not data and path.endswith(PY_DATA))
                or (not baselines and path.startswith(_BASELINE_PATHS))):
            continue
        if path == "pyproject.toml":
            ln = f"pyproject.toml {dep_spec(sha, repo)}"
        rows.append(ln)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def reads_match(a: str, b: str, reads: list, repo: Path = REPO) -> bool:
    """True when every file a leg read at `a` (gate_reads.py) is the same blob at `b`, and every
    directory it listed holds the same names. A file absent at both is the same file."""
    def tree(sha):
        return {ln.split("\t", 1)[1]: ln.split()[2] for ln in _ls_tree(sha, repo)}
    ta, tb = tree(a), tree(b)
    for r in reads:
        if r.startswith("dir:"):
            d = r[4:].strip("/")
            d = d + "/" if d else ""
            kids = [{p[len(d):].split("/", 1)[0] for p in t if p.startswith(d)} for t in (ta, tb)]
            if kids[0] != kids[1]:
                return False
        elif ta.get(r) != tb.get(r):
            return False
    return True


def dep_spec(sha: str, repo: Path = REPO) -> str:
    """pyproject.toml at `sha` without its `version =` line: what a resolution depends on."""
    body = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:pyproject.toml"], check=True,
                          capture_output=True, text=True).stdout
    return hashlib.sha256("".join(x for x in body.splitlines(True)
                                  if not x.startswith("version =")).encode()).hexdigest()


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
    rungs: tuple = ()      # a ladder group leg's rungs (any card type's), () for a whole ladder
    card_types: tuple = ()  # the card types that may run it, () for any
    known_s: int = 0       # the longest this leg is known to need: its last pass, or a run stopped at its budget

    @property
    def budget(self) -> int:
        return max(BUDGET[self.family], round(BUDGET_X * self.known_s))


#: The staleness check of the abb3 records. A gate with --record-abb3 answers it on a card instead
#: (record:abb3), and the release commit carries the re-stamped files, so its pytest legs leave it out.
STALE_TEST = "tests/test_recorded_claims.py::test_recorded_artifact_is_not_stale"


def build_legs(roster: dict, test_files: list, shards: int, record_lever: str = "",
               record_full: str = "", record_abb3: bool = False, record_cards: tuple = ()) -> list:
    skip = ["--deselect", STALE_TEST] if record_abb3 else []
    legs = [Leg("check", ["PY", "scripts/full_parity_gate.py", "--check", "--workdir", "{OUT}",
                         "--workers", "localhost:0"], "check",
                card=False),
            Leg("packaging_smoke", ["PY", "scripts/packaging_smoke.py"], "packaging_smoke", card=False),
            Leg("pytest_cpu", ["PY", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", *skip],
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
        argv = ["PY", "scripts/release_gate.py", "--model", "size-ladder", "--size-ladder-models", m]
        every = sorted({n for ct in roster.get("rungs", {}) for n in ladder_rungs(roster, ct, m)})
        if not every:       # a tree whose ENUMERATE predates the groups
            legs.append(Leg(f"ladder:{m}", argv, "ladder"))
        for lo, hi in LADDER_GROUPS:
            g = tuple(n for n in every if lo <= n <= hi)
            if g:
                legs.append(Leg(f"ladder:{m}@{g[0]}-{g[-1]}", [*argv, "--size-ladder-part", "--size-ladder-rungs",
                                                               ",".join(map(str, g))], "ladder", rungs=g))
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
                            ["PY", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", *skip, *part],
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
        # A record writes only the entry of the card type it ran on, and a ladder leg checks the
        # entry of whichever card it lands on, so each card type the gate holds gets its own record:
        # gate 29e9c9d44 recorded four BH models on p300c only and left their p150a entries stale.
        for ct in record_cards or ("",):
            legs.append(Leg(f"record:{m}@{ct}" if ct else f"record:{m}",
                            ["PY", "scripts/release_gate.py", "--model", "size-ladder",
                             "--size-ladder-models", m, *how, "--size-ladder-baseline",
                             "{OUT}/size_ladder_baseline.json"], "record", workdir=True,
                            # The tree's files can be frozen read-only (Galaxy dedupe); a plain cp
                            # keeps that mode and the record then cannot write its own copy.
                            setup="cp -r --no-preserve=mode docs/size_ladder_baseline.json "
                                  "docs/size_ladder_baseline.d {OUT}/",
                            card_types=(ct,) if ct else ()))
    if record_abb3:
        legs.append(Leg(ABB3_LEG, ["PY", "-c", ABB3_RECORD, "{OUT}", "{SHA}", "{WORKER}", "{CARD_TYPE}",
                                   str(ABB3_LOSS_TOL), *ABB3_FILES], "record", workdir=True,
                        card_types=ABB3_CARD_TYPES))
    return legs


def test_files(sha: str, repo: Path = REPO) -> list:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", sha, "tests/"],
                         check=True, capture_output=True, text=True).stdout
    return sorted(p for p in out.splitlines() if Path(p).name.startswith("test_") and p.endswith(".py"))


def last_tag(repo: Path = REPO) -> str:
    """The newest v* tag, fetched first: the last release, wherever it was cut."""
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--tags", "origin"], capture_output=True)
    tags = subprocess.run(["git", "-C", str(repo), "tag", "-l", "v*", "--sort=-v:refname"],
                          capture_output=True, text=True).stdout.split()
    if not tags:
        raise SystemExit('perf_against "last": this clone has no v* tag')
    return tags[0]


def known_walls(hist: dict, prior: list) -> dict:
    """{leg name: wall s} from the ledger's passes (`hist`), raised by any row of this --out's earlier
    run (`prior`) that was stopped at its budget: it needed at least that long, so its resume gets more."""
    out = dict(hist)
    for r in prior:
        if r.get("rc") == TIMED_OUT and r.get("wall_s"):
            out[r["leg"]] = max(out.get(r["leg"], 0), r["wall_s"])
    return out


def select(legs: list, patterns: list) -> list:
    """Legs matching any pattern (all when there is none) and no `!pattern`. A pattern that matches
    no leg is an error: `--legs record:rf3` (the leg is `record:rf3@<card>`) ran nothing on 10-23 and
    wrote an empty PASS verdict over the record run rec-437a23413."""
    take = [p for p in patterns if not p.startswith("!")]
    drop = [p[1:] for p in patterns if p.startswith("!")]
    if none := [p for p in take if not any(fnmatch.fnmatch(lg.name, p) for lg in legs)]:
        raise SystemExit(f"--legs {','.join(none)} matches no leg; legs are e.g. "
                         + ", ".join(lg.name for lg in legs[:6]))
    return [lg for lg in legs if (not take or any(fnmatch.fnmatch(lg.name, p) for p in take))
            and not any(fnmatch.fnmatch(lg.name, p) for p in drop)]


# ---------------------------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------------------------
def out_group(leg_name: str) -> str:
    """Legs that write the same output dir in the commit's shared tree on a host. release_gate
    names its results dir after the model family: a --fast arm shares its model's (rg:esmfold2 and
    rg:esmfold2-fast on g107, 10-10, the fast arm found its CIFs deleted) and so do a model's
    versions (rg:protenix-v2 deleted rg:protenix-v1's on qb1, 10-13). Trees since
    RELEASE_GATE_RUN_DIR write per leg; this keeps older trees, a fix-forward base say, apart."""
    return re.sub(r"-(fast|v\d+)$", "", leg_name)


@dataclass
class Host:
    name: str
    cfg: dict
    sha: str
    lock: threading.Condition = field(default_factory=threading.Condition)
    running: int = 0
    timed_running: bool = False
    outputs: set = field(default_factory=set)
    lend: tuple = (float("-inf"), None, False)   # (read at, LEND-AT epoch, pool PAUSEd)

    def read_lend(self) -> None:
        """Refresh `lend` from the host's pool at most every LEND_READ_S."""
        if "pool" not in self.cfg or time.time() - self.lend[0] < LEND_READ_S:
            return
        pause = shlex.quote(str(Path(self.cfg["pool"]["queue"]).parent / "PAUSE"))
        r = self.ssh(f"cat {LEND_AT} 2>/dev/null; echo; test -e {pause} && echo PAUSE",
                     capture_output=True, timeout=120)
        if r.returncode in (0, 1):
            words = r.stdout.split()
            at = next((int(w) for w in words if w.isdigit()), None)
            self.lend = (time.time(), at, "PAUSE" in words)

    def lent(self, budget: int) -> bool:
        """True while this host's pool would not start a job capped at `budget` s (from `lend`)."""
        _, at, paused = self.lend
        return paused or (at is not None and time.time() + budget >= at)

    @property
    def key_card_type(self) -> str:
        return self.cfg.get("key_card_type", self.cfg["card_type"])

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
        # Twenty workers connecting at once can pass a Galaxy's sshd MaxStartups, which drops the
        # handshake ("kex_exchange_identification: Connection closed"), and qb1 drops off the LAN
        # for minutes at a time. Either way the command did not answer, so it is retried, never
        # recorded as the leg's result; a timeout that outlasts every retry reads as exit 255.
        check = kw.pop("check", False)
        for wait in (5, 15, 45, 120, 300, None):
            try:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60",
                                    self.cfg["ssh"], cmd], text=True, **kw)
            except subprocess.TimeoutExpired as e:
                r = subprocess.CompletedProcess(e.cmd, 255, "", f"Timeout, server {self.cfg['ssh']} not responding.")
            if wait is None or r.returncode != 255 or not SSH_LOST.search(r.stderr or ""):
                break
            time.sleep(wait)
        if check and r.returncode:
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
        return r

    def answers(self) -> bool:
        """One quick connection, not ssh()'s minutes of retries: a host that does not answer now
        sits out this gate run."""
        if self.cfg.get("ssh") in (None, "", "localhost"):
            return True
        try:
            return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.cfg["ssh"], "true"],
                                  capture_output=True, timeout=60).returncode == 0
        except subprocess.TimeoutExpired:
            return False

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
               f"git -C {r}/repo worktree add -q --detach {t} {self.sha}; fi; "
               # Every time, not only on a new tree: a prep that died after the worktree add left
               # a tree without the binaries, and the next prep skipped the fetch (1064f95a1, qb1).
               f"cd {t}; bash scripts/fetch_parity_fixtures.sh >/dev/null; "
               f"test \"$(git -C {t} rev-parse HEAD)\" = {self.sha}")
        if self.cfg.get("perf_against"):
            ref, a = shlex.quote(self.cfg["perf_against"]), self.against
            cmd += (f"; [ -d {a} ] || {{ git -C {r}/repo fetch -q --tags origin; "
                    f"git -C {r}/repo worktree add -q --detach {a} {ref}; }}")
        # One prep at a time per host: two runners preparing at once (a record run beside its
        # gate, 10-11 on g114) raced `git fetch` on the shared clone's ref locks and one died.
        lock = f"flock {r}/prep.lock"
        p = self.ssh(f"mkdir -p {r} && {lock} bash -c {shlex.quote(cmd)}", capture_output=True, timeout=1800)
        if not p.returncode:
            # The runner's own copy of the recipe: the commit under test may predate it.
            p = self.ssh(f"cd {t} && {lock} env {envs} bash -s -- {r} {self.sha}", capture_output=True,
                         timeout=3600, input=(REPO / "scripts" / "gate_host_prep.sh").read_text())
        if p.returncode:
            raise SystemExit(f"{self.name}: cannot prepare {t} at {self.sha}: "
                             f"{(p.stderr or p.stdout).strip()[-600:]}")
        print(f"{self.name}: {p.stdout.strip()}")
        # The runner's copy of gate_reads.py into every venv a leg runs under (READS), as a .pth
        # hook so it loads whatever PYTHONPATH a gate script gives its children. Not into a system
        # interpreter: a leg on one records nothing and reuses on identical code only.
        hook = (REPO / "scripts" / "gate_reads.py").read_text()
        site = ("import sys, sysconfig; sys.prefix != sys.base_prefix and print(sysconfig.get_paths()['purelib'])")
        for py in sorted({self.python(f) for f in EXPECT}):
            self.ssh(f"s=$({py} -c {shlex.quote(site)}) && [ -n \"$s\" ] && cat > $s/_gate_reads.py && "
                     f"echo 'import _gate_reads' > $s/_gate_reads.pth", input=hook, timeout=120)

    def python(self, family: str) -> str:
        """The interpreter a leg family runs under (pytest and the BindCraft 2 leg need the
        Python 3.12 venv that also carries BindCraft 2)."""
        s = self.sha[:12]
        default = (f"{self.root}/venv312-{s}/bin/python" if family in PY312
                   else f"{self.root}/venv-{s}/bin/python")
        return self.cfg.get("python_for", {}).get(family, self.cfg.get("python", default))

    def leg(self, leg: Leg) -> Leg:
        """`leg` with this host's extra flags for its family (hosts.json `args`)."""
        extra = list(self.cfg.get("args", {}).get(leg.family, []))
        if self.cfg.get("perf_against") and (leg.family == "perf" or (
                leg.family == "ladder" and ladder_ab(self.sha))):
            extra += ["--against" if leg.family == "perf" else "--size-ladder-against", self.against]
        return dataclasses.replace(leg, argv=leg.argv + extra) if extra else leg

    @property
    def against(self) -> str:
        return f"{self.root}/trees/{self.cfg['perf_against']}"

    @property
    def ladder_cap(self) -> int | None:
        """The `--host-threads` this host's ladder legs run at (None = every core)."""
        a = self.cfg.get("args", {}).get("ladder", [])
        return int(a[a.index("--host-threads") + 1]) if "--host-threads" in a else None

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
        # Every leg folds in one shared tree per host, so release_gate's scratch and run outputs must
        # be the leg's own: the first ladder or record leg to finish rmtrees the shared default under
        # the others (a fold whose census dumps vanished reads every lever "not-imported"), and
        # rg:protenix-v2 rmtree'd the protenix_results_prot/ rg:protenix-v1 was parsing (qb1 10-13).
        env = {"PYTHONPATH": pp, "TT_VISIBLE_DEVICES": "" if card is None else str(card),
               "RELEASE_GATE_SIZE_WORKDIR": f"{out}/sizegate-work", "RELEASE_GATE_RUN_DIR": f"{out}/run",
               **c.get("env", {}),
               "GATE_READS": f"{out}/reads", "GATE_TREE": self.tree}   # gate_reads.py
        subst = {"{CARD}": str(card), "{OUT}": out, "{SHA}": self.sha, "{WORKER}": f"{self.name}:{card}",
                 "{CARD_TYPE}": c["card_type"],
                 **{f"{{{k}}}": v for k, v in c.get("env", {}).items()}}

        def fill(a: str) -> str:
            for k, v in subst.items():
                a = a.replace(k, v)
            return a
        argv = [self.python(leg.family) if a == "PY" else fill(a) for a in self.leg(leg).argv]
        if "pool" in c and argv[1:2] and argv[1] in LOAD_GUARDED:
            argv += ["--load-ceiling", "0"]   # not in the leg's key: it changes nothing measured
        legpid = shlex.quote(f"{out}/leg.pid")
        body = (f"echo \"##LEG-START $(date +%s)\"; echo $$ > {legpid}; exec timeout -s INT -k 600 {leg.budget} "
                "nice -n 10 " + " ".join(shlex.quote(a) for a in argv))
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        run = f"env {envs} bash -c {shlex.quote(body)}"
        if card is not None:
            lock = "bash scripts/flock_first.sh" if c.get("first") else "flock"
            run = f"{lock} {shlex.quote(c['lock'].format(card=card))} {run}"
        setup = f"{fill(leg.setup)} && " if leg.setup else ""
        # Each attempt is its own session, its pid in attempt.pid and its marker in the environment
        # (which survives bash's exec of the last command). When the runner's connection drops the
        # remote side keeps waiting on the card, so a rerun first ends the attempt it replaces;
        # otherwise both run the leg into one directory (qb1, 10-10: five BH legs queued twice).
        #
        # The session also hides the leg from a signal sent to the pool job: a JapanFold recall of a lent
        # chip SIGINTs the job's process group, and on .110 on 2026-10-23 the gate drivers ran on and
        # started their next fold on the chip being reset. So the job shell passes SIGINT/SIGTERM on to
        # the leg's `timeout` (leg.pid), which hands it to the driver, and waits for the leg to finish.
        # Still in flock or setup, there is no driver yet: the session is ended outright.
        mark = shlex.quote(f"GATE_ATTEMPT={out}")
        attempt = shlex.quote(f"echo $$ > {shlex.quote(f'{out}/attempt.pid')}; {run}")
        fwd = shlex.quote(f"kill -INT $(cat {legpid} 2>/dev/null) 2>/dev/null || pkill -TERM -s $a")
        return (f"mkdir -p {shlex.quote(out)} && {self.end_attempt(out)} && rm -f {legpid} && cd {self.tree} && "
                f"{setup}{{ env {mark} setsid -w bash -c {attempt} & a=$!; trap {fwd} INT TERM; "
                f"while :; do wait $a; rc=$?; kill -0 $a 2>/dev/null || break; done; (exit $rc); }}")
                # no exec: a pool job writes rc after it

    def marked_bad(self, card) -> str | None:
        """The mark that takes `card` out of service (BAD_MARKS), or None. A host that does not
        answer reads as unmarked: the leg then meets the same lost connection and is retried."""
        local = LOCAL_BLOCKS / f"cardblock-{self.name}-{card}"
        if local.exists():
            return str(local)
        marks = " ".join(m.format(card=card, host=self.name) for m in BAD_MARKS)
        r = self.ssh(f"for f in {marks}; do [ -e \"$f\" ] && echo \"$f\"; done; true",
                     capture_output=True, timeout=120)
        found = (r.stdout or "").split()
        return found[0] if r.returncode == 0 and found else None

    def card_dead(self, card) -> str | None:
        """Why `card` failed CARD_PROBE_TRIES probes in a row, or None once one passes."""
        why = None
        for i in range(CARD_PROBE_TRIES):
            if i:
                time.sleep(CARD_PROBE_GAP_S)
            why = self.probe_card(card)
            if why is None:
                return None
        return why

    def probe_card(self, card) -> str | None:
        """Why `card` cannot open and dispatch a program, or None when it can. A probe that never
        got the card's flock, or never started, says nothing about the card and reads as None."""
        env = {"PYTHONPATH": self.tree, "TT_VISIBLE_DEVICES": str(card), **self.cfg.get("env", {})}
        body = (f"echo CARD_PROBE_START; env {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
                f"timeout -s KILL {CARD_PROBE_S} {self.python('parity')} -c {shlex.quote(CARD_PROBE)} 2>&1 "
                f"| grep -v -e '| DEBUG' -e '^Config{{' | tail -n 12")
        lock = self.cfg.get("lock")
        lock = f"flock -w {CARD_PROBE_LOCK_S} {shlex.quote(lock.format(card=card))} " if lock else ""
        r = self.ssh(f"cd {self.tree} && {lock}bash -c {shlex.quote(body)}",
                     capture_output=True, timeout=CARD_PROBE_LOCK_S + CARD_PROBE_S + 120)
        out = r.stdout or ""
        if "CARD_PROBE_START" not in out or "CARD_HEALTHY" in out:
            return None
        return out.split("CARD_PROBE_START", 1)[1].strip()[-1200:] or "the probe printed nothing"

    @staticmethod
    def end_attempt(out: str) -> str:
        """Shell that ends the attempt recorded in out/attempt.pid if it is still that leg's, and
        every process that carries its marker: a child that starts its own session (lever_census's
        fold) escapes `pkill -s` but not its inherited environment (rel-c829d6e72, 10-13)."""
        pidf, mark = shlex.quote(f"{out}/attempt.pid"), shlex.quote(f"GATE_ATTEMPT={out}")
        return (f"{{ p=$(cat {pidf} 2>/dev/null) && tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null "
                f"| grep -qxF -- {mark} && pkill -TERM -s $p; "   # -s: timeout moves the leg to its own group
                f"for e in /proc/[0-9]*/environ; do tr '\\0' '\\n' 2>/dev/null < $e | grep -qxF -- {mark} "
                f"&& q=${{e#/proc/}} && kill -TERM ${{q%/environ}} 2>/dev/null; done; true; }}")


# ---------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------
def classify(leg: Leg, rc: int, report: dict | None, baseline: dict | None = None) -> str:
    """PASS / FAIL / BLOCKED. A parity leg the gate itself reports BLOCKED-REF-REGEN-NEEDED does
    not fail the gate (RELEASING.md, verdict semantics); a one-leg run of it exits nonzero as
    GATE INCONCLUSIVE, so read the report rather than the exit code.

    A capacity cell the engine's own size guard refuses (opendde on Blackhole: 1536 freezes the
    trunk, so the cap is 1024) exits 1 on every run. It passes when the committed baseline for
    this card type (`baseline`, its "cells") records the same refusal; anything else still fails."""
    if leg.family == "parity" and report:
        verdicts = {r.get("verdict") for r in report.get("legs", [])}
        if verdicts == {"BLOCKED-REF-REGEN-NEEDED"}:
            return "BLOCKED"
    if rc and leg.family == "capacity" and report and baseline is not None:
        c = report.get("counts", {})
        failed = [r for r in report.get("results", []) if r.get("verdict") != "PASS"]
        if (failed and not c.get("GATE_BUG") and not c.get("BAD_FIXTURE")
                and not report.get("coverage_gaps")
                and all(r.get("verdict") == "FAIL" and r.get("mechanism") == "size_guard"
                        and (baseline.get(r.get("model")) or {}).get("verdict") == "FAIL"
                        and (baseline.get(r.get("model")) or {}).get("mechanism") == "size_guard"
                        for r in failed)):
            return "PASS"
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

    def pins(self, sha: str, spec: str) -> dict | None:
        """The package set `sha` was gated with, else the newest one recorded for the same
        dependency spec: installing it keeps every leg's environment key, so a PyPI patch
        release between two commits' preps does not rerun the gate (beartype 0.23.1, 10-11)."""
        own = self.path / "pins" / sha[:12]
        if not (own / "spec").exists():
            same = [d for d in (self.path / "pins").glob("*/spec") if d.read_text().strip() == spec]
            if not same:
                return None
            own = max(same, key=lambda d: d.stat().st_mtime).parent
        return {f.stem: f.read_text() for f in own.glob("*.txt")}

    def save_pins(self, sha: str, spec: str, pins: dict) -> None:
        d = self.path / "pins" / sha[:12]
        if (d / "spec").exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        for v, text in pins.items():
            (d / f"{v}.txt").write_text(text)
        (d / "spec").write_text(spec + "\n")

    def history(self) -> dict:
        """leg name -> the latest wall clock any passing run of it took, for ordering and budgets."""
        seen = {}
        for p in self.path.glob("*.json"):
            try:
                r = json.loads(p.read_text())
            except ValueError:
                continue
            if r.get("wall_s") and r.get("ended", "") >= seen.get(r["leg"], ("", 0))[0]:
                seen[r["leg"]] = (r["ended"], r["wall_s"])
        return {k: v[1] for k, v in seen.items()}

    def read_hit(self, scope: str, sha: str, repo: Path = REPO) -> dict | None:
        """The newest passing row in `scope` whose recorded reads are unchanged at `sha`."""
        if not hasattr(self, "_scopes"):
            self._scopes = {}
            for p in self.path.glob("*.json"):
                try:
                    r = json.loads(p.read_text())
                except ValueError:
                    continue
                if r.get("scope") and r.get("reads") and r.get("verdict") in OK:
                    self._scopes.setdefault(r["scope"], []).append(r)
        for r in sorted(self._scopes.get(scope, []), key=lambda r: r.get("ended", ""), reverse=True):
            try:
                if reads_match(r["sha"], sha, r["reads"], repo):
                    return r
            except subprocess.CalledProcessError:
                continue                       # a commit this clone does not have
        return None

    def put(self, key: str, rec: dict) -> None:
        p = self.path / f"{key}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
        tmp.replace(p)

    # A leg runs under a claim, so a second runner on the same commit (more cards joining a gate
    # already running) skips what the first holds and waits for its row instead of running it twice.
    ME = {"host": socket.gethostname(), "pid": os.getpid()}

    def claim(self, key: str) -> dict | None:
        """Take `key` for this runner: None when taken, else the live holder. A holder on this host
        whose pid is gone is stale and is replaced."""
        p = self.path / f"{key}.claim"
        while True:
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                held = self.holder(key)
                if held is None:
                    time.sleep(0.05)       # mid-write, just released, or stale and dropped
                    continue
                return None if held == self.ME else held
            with os.fdopen(fd, "w") as f:
                json.dump(self.ME, f)
            return None

    def holder(self, key: str) -> dict | None:
        """The live holder of `key`; a claim left on this host by a pid that is gone is dropped."""
        p = self.path / f"{key}.claim"
        try:
            held = json.loads(p.read_text())
        except ValueError:
            # empty or torn: a runner died between creating and writing it, or is writing it now
            if time.time() - p.stat().st_mtime > 10:
                p.unlink(missing_ok=True)
            return None
        except OSError:
            return None
        if held.get("host") == self.ME["host"] and not _alive(held.get("pid")):
            p.unlink(missing_ok=True)
            return None
        return held

    def release(self, key: str) -> None:
        if self.holder(key) == self.ME:
            (self.path / f"{key}.claim").unlink(missing_ok=True)


def _reused(leg: str, slot: str, key: str, row: dict) -> dict:
    return {"leg": leg, "arch": slot, "verdict": "REUSED", "key": key,
            "evidence": f"{row['worker']} {row['ended']} {row['sha'][:9]} {row['log']}",
            "reused_verdict": row["verdict"]}


class CardDead(Exception):
    """execute() found the leg's card dead after the leg failed; `res` is the leg's result."""

    def __init__(self, res: dict, why: str):
        super().__init__(why)
        self.res, self.why = res, why


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        pass
    return True


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
    owed = sorted({f for r in results if r["verdict"] == "OWED" for f in r["owed"]} - {CAP_OWED})
    capped = sorted({ladder_model(r["leg"]) for r in results
                     if r["verdict"] == "OWED" and CAP_OWED in r["owed"]})
    if capped:
        lines += ["", "OWED: no granted host runs these ladders at the host-thread cap their baseline "
                  "was timed at, so they were not folded. Re-record them on the gate's hosts, commit "
                  "the fragments, and gate that commit:",
                  f"`gate_fanout.py --legs 'record:*' --record-full {','.join(capped)} ...`"]
    if owed:
        lines += ["", "OWED: the size-ladder baseline at this commit has no row for a lever these "
                  "models resolve, so their ladder legs were not folded. Record it, commit the "
                  "fragments, and gate that commit (RELEASING.md, Baselines a new lever owes):",
                  f"`gate_fanout.py --legs 'record:*' --record-lever {','.join(owed)} ...`"]
    recorded = [r for r in results if r["leg"].startswith("record:") and r.get("seeded")]
    if recorded:
        lines += ["", "RECORDED: copy these over docs/size_ladder_baseline.d/ in the release commit:"]
        lines += [f"- {r['leg']} {r['card_type']}: {r['seeded']}" for r in recorded]
    ab = [r for r in results if r["leg"] == "perf" and r["verdict"] == "PASS" and r.get("seeded")]
    if ab:
        lines += ["", "RE-RECORDED: the A/B reference's cells; copy over docs/perf_baselines.json in the release commit:"]
        lines += [f"- perf {r['card_type']}: {r['seeded']}" for r in ab]
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

    def __init__(self, todo: dict, workers: list, timed: set, execute, out: Path,
                 ledger: "Ledger | None" = None, keys: dict | None = None, roster: dict | None = None):
        self.todo = todo          # arch -> list of Leg (card legs), consumed in order
        self.workers = workers    # (Host, card)
        self.timed = timed        # {(host name, card)}
        self.execute = execute    # (host, card, leg) -> result dict
        self.out = out
        self.ledger = ledger      # with keys: claim each card leg, so another runner can join
        self.keys = keys or {}    # (leg name, arch) -> key
        self.roster = roster or {}  # ENUMERATE's output; its "caps" keep a ladder leg on a host of its cap
        self.held: dict = {}      # queue -> [Leg] another runner holds
        self.bounced: set = set()  # (leg name, queue) already put back once off a dead card
        self.inflight: dict = {}   # queue -> legs started and not yet recorded or put back
        self.results: list = []
        self.mu = threading.Lock()

    @staticmethod
    def _queue(host: Host, card) -> str:
        return host.arch if card is not None else "any"

    def _take(self, host: Host, card) -> Leg | None:
        """The next leg this card may run, or None. A leg whose output dir is in use on the host
        stays queued (see `out_group`); `_blocked` tells the caller to wait for it."""
        is_timed = (host.name, card) in self.timed
        host.read_lend()
        with self.mu:
            q = self.todo.get(self._queue(host, card), [])
            # Correctness legs first, timed legs last: by then the host's other cards have run out
            # of work, so holding the host quiet idles nothing.
            for timed in (False, True):
                for i, lg in enumerate(q):
                    if lg.timed == timed and (is_timed or not timed) \
                            and out_group(lg.name) not in host.outputs and not host.lent(lg.budget) \
                            and cap_fits(self.roster, host, lg) and rungs_fit(self.roster, host, lg) \
                            and type_fits(host, lg):
                        return q.pop(i)
        return None

    def _lent_out(self, host: Host, card) -> bool:
        """True when a leg this card could take is held back only by its host's lend."""
        is_timed = (host.name, card) in self.timed
        with self.mu:
            return any((is_timed or not lg.timed) and host.lent(lg.budget)
                       for lg in self.todo.get(self._queue(host, card), []))

    def _blocked(self, host: Host, card) -> bool:
        with self.mu:
            return any(out_group(lg.name) in host.outputs
                       for lg in self.todo.get(self._queue(host, card), []))

    def _pending(self, queue: str) -> bool:
        with self.mu:
            return bool(self.todo.get(queue))

    def _record(self, res: dict) -> None:
        with self.mu:
            self.results.append(res)
            with open(self.out / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")

    def _joined(self, leg: Leg, queue: str) -> bool:
        """True when another runner holds `leg` or has passed it (then recorded as REUSED)."""
        if self.ledger is None or not leg.card or leg.family == "record":
            return False
        key = self.keys[(leg.name, queue)]
        row = self.ledger.get(key)
        if row is None and self.ledger.claim(key) is None:
            return False
        if row is None:
            with self.mu:
                self.held.setdefault(queue, []).append(leg)
        else:
            self._record(_reused(leg.name, queue, key, row))
        return True

    def _reuse_passed(self, queue: str) -> None:
        """Record as REUSED each queued leg another runner has passed meanwhile. A leg held back by
        a lend is never taken, so `_joined` would never see that row."""
        if self.ledger is None:
            return
        with self.mu:
            legs = [lg for lg in self.todo.get(queue, []) if lg.card and lg.family != "record"]
        for leg in legs:
            key = self.keys[(leg.name, queue)]
            row = self.ledger.get(key)
            if not (row and row.get("verdict") in OK):
                continue
            with self.mu:
                if leg not in self.todo.get(queue, []):
                    continue
                self.todo[queue].remove(leg)
            self._record(_reused(leg.name, queue, key, row))

    def _wait_held(self, queue: str) -> bool:
        """Wait until a leg another runner held is released: its row is REUSED, a leg it left
        without a passing row goes back on the queue (True). False once nothing is held."""
        while True:
            with self.mu:
                held = list(self.held.get(queue, []))
            if not held:
                return False
            for leg in held:
                key = self.keys[(leg.name, queue)]
                if self.ledger.holder(key) is not None:
                    continue
                with self.mu:
                    if leg not in self.held.get(queue, []):
                        continue
                    self.held[queue].remove(leg)
                row = self.ledger.get(key)
                if row and row.get("verdict") in OK:
                    self._record(_reused(leg.name, queue, key, row))
                else:
                    with self.mu:
                        self.todo.setdefault(queue, []).insert(0, leg)
                    return True
            time.sleep(POOL_POLL_S)

    def worker(self, host: Host, card) -> None:
        queue = self._queue(host, card)
        while True:
            if card not in (None, POOL) and (mark := host.marked_bad(card)):
                print(f"skipped {host.name}:{card} for this run, its chip is marked bad: {mark}", flush=True)
                return
            with host.lock:
                while host.timed_running:
                    host.lock.wait()
                leg = self._take(host, card)
                while leg is None and self._blocked(host, card):
                    host.lock.wait()
                    leg = self._take(host, card)
                if leg is not None and self._joined(leg, queue):
                    continue
                if leg is not None:
                    with self.mu:
                        self.inflight[queue] = self.inflight.get(queue, 0) + 1
                    host.outputs.add(out_group(leg.name))
                    if leg.timed:
                        host.timed_running = True
                        while host.running:
                            host.lock.wait()
                    host.running += 1
            if leg is None:
                # nothing left this worker may run (what remains, if anything, is timed), unless
                # a leg another runner held comes back
                if self._wait_held(queue):
                    continue
                if self._lent_out(host, card):
                    self._reuse_passed(queue)   # a sibling runner may pass it first (rel-9b16, 10-18)
                    time.sleep(POOL_POLL_S)     # another host may take it first; else the lend ends
                    continue
                with self.mu:
                    busy = self.inflight.get(queue, 0) > 0
                if busy:
                    time.sleep(1)       # a sibling's leg may come back (withdrawn, or off a dead card)
                    continue
                return
            dead = None
            try:
                res = self.execute(host, card, leg)
            except CardDead as e:
                res, dead = e.res, e.why
            finally:
                with host.lock:
                    host.running -= 1
                    host.outputs.discard(out_group(leg.name))
                    if leg.timed:
                        host.timed_running = False
                    host.lock.notify_all()
                if self.ledger is not None and leg.card:
                    self.ledger.release(self.keys[(leg.name, queue)])
            if res is None:
                with self.mu:
                    self.todo.setdefault(queue, []).insert(0, leg)
                    self.inflight[queue] -= 1
                continue
            if dead:
                with self.mu:
                    again = (leg.name, queue) not in self.bounced
                    self.bounced.add((leg.name, queue))
                    if again:
                        self.todo.setdefault(queue, []).insert(0, leg)
                    self.inflight[queue] -= 1
                print(f"retired {host.name}:{card} for this run, its card is dead: {dead.splitlines()[-1]}"
                      + (f"; {leg.name} goes back on the queue" if again else ""), flush=True)
                if not again:
                    self._record(res)
                with host.lock:
                    host.lock.notify_all()
                return
            self._record(res)
            with self.mu:
                self.inflight[queue] -= 1

    def run(self) -> list:
        ts = [threading.Thread(target=self.worker, args=w, daemon=True) for w in self.workers]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for queue, legs in self.todo.items():
            for leg in legs:
                self._record({"leg": leg.name, "arch": queue, "verdict": "UNRUN", "rc": None,
                              "worker": "none", "wall_s": 0, "ended": now(),
                              "key": self.keys.get((leg.name, queue))})
        return self.results


def cardfree(workers: list, arch: str) -> list:
    """One cardless worker per host of `arch`, all drawing on the shared "any" queue, so the
    card-free legs run on whichever host is free. Each stands aside while a timed leg holds its
    host quiet. They used to run only on the first host: pre2-e6ae723c4 (10-11) listed g107
    first, g107 went back to JapanFold, and check/packaging_smoke/pytest_cpu waited 5 h."""
    hosts = {h.name: h for h, _ in workers if h.arch == arch}
    return [(h, None) for h in hosts.values()]


def pool_job(host: Host, leg: Leg, rdir: str, name: str) -> str:
    """A job file for the host's chip pool (CHIPS.md "Chip pool"): the pool starts it as
    `CHIP=<umd> bash <job>` on an idle healthy chip, and it leaves its log and exit code in rdir."""
    cmd = host.command(leg, POOL_CARD, rdir)
    q = shlex.quote
    returned = q(str(Path(host.cfg["pool"]["queue"]).parent / "RETURNED"))
    return (f"#!/bin/bash\n# {name}: gate leg {leg.name} at {host.sha[:12]} (gate_fanout.py)\n"
            f"mkdir -p {q(rdir)}; rm -f {q(rdir)}/rc; cat {BOOT_ID} > {q(rdir)}/boot\ncmd={q(cmd)}\n"
            f"{{ echo \"##CHIP $CHIP\"; eval \"${{cmd//{POOL_CARD}/$CHIP}}\"; }} > {q(rdir)}/leg.log 2>&1\n"
            f"rc=$?; [ {returned} -nt {q(rdir)}/boot ] && echo \"##{GIVEN_BACK}\" >> {q(rdir)}/leg.log\n"
            f"echo $rc > {q(rdir)}/rc.tmp && mv {q(rdir)}/rc.tmp {q(rdir)}/rc\n")


def run_in_pool(host: Host, card, leg: Leg, rdir: str, f) -> int:
    """Queue `leg` in the host's pool, wait for its exit code, copy its log (capped) into f."""
    pool = host.cfg["pool"]
    run, _, tag = Path(rdir).parts[-3:]     # <root>/out-<sha12>/<run>/<arch>/<leg>
    name = f"{pool.get('prio', 5)}-{pool.get('row', 'gate')}-{run}-{tag}"
    job, q = f"{pool['queue']}/{name}.sh", shlex.quote
    # The boot id goes too: a leg re-queued after a reboot reuses rdir, and the last run's boot
    # file read as "started under another boot" at once (rel-5e250928a, 10-16 11:01Z: nine WH legs
    # FAILed in seconds and left their jobs queued for the pool to run with nobody waiting).
    r = host.ssh(f"mkdir -p {q(rdir)} && rm -f {q(rdir)}/rc {q(rdir)}/boot && cat > {q(job)}.tmp && mv {q(job)}.tmp {q(job)}",
                 input=pool_job(host, leg, rdir, name), capture_output=True, timeout=120)
    if r.returncode:
        # ssh's own message on its own line, so a lost connection reads as one and is retried.
        f.write(f"cannot queue {job}:\n{r.stderr.strip()}\n")
        return r.returncode if r.returncode == 255 else 1
    f.write(f"# queued {host.name}:{job}\n")
    f.flush()
    QUEUED[job] = host
    RUNNING[rdir] = host
    while True:
        time.sleep(POOL_POLL_S)
        host.read_lend()
        if host.lent(leg.budget) and host.ssh(f"mv {q(job)} {q(job)}.lent", capture_output=True,
                                              timeout=120).returncode == 0:
            # the pool had not started it (it moves a job out of the queue when it does)
            QUEUED.pop(job, None)
            RUNNING.pop(rdir, None)
            f.write(f"# taken back: {host.name}'s pool is lent or paused past this leg's cap\n")
            return WITHDRAWN
        # A chip the agent recalls is SIGINTed under the leg, which still writes an rc; the pool then copies the
        # job back into the queue within milliseconds, so a queued job no older than the rc is a requeue, not a
        # verdict (file times tick per jiffy, hence "no older"). rel-f2fd17ff3, 10-22 21:22Z: ladder:openbind
        # recorded FAIL from a recalled attempt while its rerun went unread.
        rc_f = f"{q(rdir)}/rc"
        try:
            r = host.ssh(f"cat {BOOT_ID}; cat {q(rdir)}/boot 2>/dev/null || echo; [ -s {rc_f} ] && sleep {REQUEUE_GRACE_S} && "
                         f"if [ -e {q(job)} ] && ! [ {rc_f} -nt {q(job)} ]; then rm -f {rc_f}; echo {REQUEUED}; else cat {rc_f}; fi",
                         capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            continue        # a busy sshd; the job runs on regardless, ask again
        if r.returncode == 255:
            continue        # the connection, not the job
        boot, started, rc = (r.stdout.split("\n") + ["", ""])[:3]
        if rc.strip() == REQUEUED:
            f.write(f"# {REQUEUED}: {host.name} recalled the chip under this attempt; waiting for the rerun\n")
            f.flush()
            continue
        if rc.strip():
            rc = int(rc)
            break
        if started.strip() and started.strip() != boot.strip():
            QUEUED.pop(job, None)
            RUNNING.pop(rdir, None)
            f.write(f"# {host.name} {REBOOTED} (boot id changed) and the leg left no exit code\n")
            return 1
    QUEUED.pop(job, None)
    RUNNING.pop(rdir, None)
    for _ in range(SSH_LOST_TRIES):
        try:
            r = host.ssh(f"L={q(rdir)}/leg.log; head -c {LOG_CAP} $L; [ $(stat -c%s $L) -le {LOG_CAP} ] || "
                         f"{{ echo; echo '# ... log capped at {LOG_CAP} bytes; last 400 lines:'; tail -n 400 $L; }}",
                         capture_output=True, timeout=300)
            break
        except subprocess.TimeoutExpired:
            time.sleep(POOL_POLL_S)
    else:
        f.write(f"# could not read {host.name}:{rdir}/leg.log\n")
        return rc
    f.write(r.stdout)
    return rc


def run_over_ssh(host: Host, card, leg: Leg, rdir: str, f) -> int:
    """Run `leg` on a named card, streaming its log into f."""
    cmd = host.command(leg, card, rdir)
    RUNNING[rdir] = host
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
    rc = p.wait()
    RUNNING.pop(rdir, None)
    return rc


def record_fragment(model_file: str, written: str, card_type: str, model: str):
    """The model's baseline file with `card_type`'s entry replaced by the one the record leg wrote.

    release_gate --size-ladder-record writes its card's entry into the top-level baseline file it
    was pointed at, not into size_ladder_baseline.d/. Until 2026-10-10 the runner returned the .d
    copy staged before the leg, so 14 record legs of gate 92d5924f5 "passed" with the old entry.
    None when the record wrote no entry for this model, which fails the leg."""
    entry = json.loads(written).get("cards", {}).get(card_type)
    if not entry or model not in entry.get("models", {}):
        return None
    base = json.loads(model_file)
    base["cards"][card_type] = entry
    return json.dumps(base, indent=2) + "\n"


def make_executor(sha: str, out: Path, ledger: Ledger, keys: dict, remote_out: str, scopes: dict | None = None):
    def execute(host: Host, card, leg: Leg) -> dict:
        tag = leg.name.replace(":", "_").replace("/", "-")
        log = out / "logs" / host.arch / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        # A second runner on the same --out reruns a leg that FAILed: keep the first run's log.
        n = 1
        while log.exists():
            log, n = log.with_name(f"{tag}.{n}.log"), n + 1
        rdir = f"{host.root}/{remote_out}/{host.arch}/{tag}"
        t0 = time.time()
        run = run_in_pool if card == POOL else run_over_ssh
        lost = full = stuck = contended = rebooted = given_back = init_failed = 0
        while True:
            with open(log, "w") as f:
                f.write(f"# {leg.name} on {host.name}:{card} ({host.cfg['card_type']}) tree {sha}\n")
                f.flush()
                rc = run(host, card, leg, rdir, f)
            if rc == WITHDRAWN:
                return None     # the worker puts it back on the queue for another host
            text = log.read_text(errors="replace")
            # The gate scripts refuse to start on an overloaded host (gate_guard's load ceiling).
            # That is not a verdict: wait for the load to drop and start the leg again.
            if rc == 255 and SSH_LOST.search(text) and lost < SSH_LOST_TRIES:
                # The connection dropped, not the leg: run it again once the host answers.
                lost += 1
                time.sleep(SSH_LOST_RETRY_S)
                continue
            if rc and DISK_FULL.search(text) and full < DISK_FULL_TRIES:
                full += 1
                time.sleep(DISK_FULL_RETRY_S)
                continue
            if rc and CONTENDED.search(text) and contended < CONTENDED_TRIES:
                contended += 1
                time.sleep(CONTENDED_RETRY_S)
                continue
            if rc and card == POOL and REBOOTED in text and not rebooted:
                rebooted += 1
                continue
            if rc and card == POOL and GIVEN_BACK in text and given_back < GIVEN_BACK_TRIES:
                given_back += 1
                continue
            if rc and card == POOL and INIT_FAILED.search(text) and init_failed < INIT_FAILED_TRIES:
                init_failed += 1
                continue
            if rc and card == POOL and CARD_STUCK.search(text) and not stuck:
                stuck += 1
                continue
            if not (rc and LOAD_REFUSAL in text and time.time() - t0 < LOAD_WAIT_MAX_S):
                break
            time.sleep(LOAD_RETRY_S)
        # A leg that stopped because its card is dirty and the reset was refused has already proved
        # what a probe would guess at, and a probe can pass on that card (it reads None when it
        # cannot get the flock): qb2:1 passed one after ladder:opendde stopped this way on 10-13,
        # and ladder:boltz2, next on it, died at its first open with a sysmem TT_THROW.
        dead = None if not rc or card in (None, POOL) else (
            m[0] if (m := CARD_STUCK.search(text)) else host.card_dead(card))
        if dead:
            with open(log, "a") as f:
                f.write(f"\n# {host.name}:{card} failed the card probe after this leg: {dead}\n")
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
        baseline = None
        if rc and leg.family == "capacity" and report:
            r = host.ssh(f"cat {host.tree}/docs/capacity_gate_baseline.json", capture_output=True, timeout=120)
            if r.returncode == 0:
                baseline = (json.loads(r.stdout).get("cards", {}).get(host.cfg["card_type"]) or {}).get("cells", {})
        verdict = classify(leg, rc, report, baseline)
        # What the leg's Python read (gate_reads.py), so a later commit that changed none of it
        # reuses this result. No list (an interpreter without the hook) reuses on the same code only.
        scope = (scopes or {}).get((leg.name, host.arch)) if leg.card else None
        reads = []
        if scope:
            r = host.ssh(f"cat {shlex.quote(rdir)}/reads/*.txt 2>/dev/null | sort -u", capture_output=True,
                         timeout=120)
            reads = r.stdout.splitlines() if r.returncode == 0 else []
        seeded = None
        if verdict == "PASS" and (host.seeding(leg) or (leg.family == "perf" and SEEDED_FIRST in text)):
            verdict = "SEEDED"
            dst = out / "seeded" / host.arch / "perf_baselines.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = host.ssh(f"cat {host.tree}/docs/perf_baselines.json", capture_output=True, timeout=120)
            if r.returncode == 0:
                dst.write_text(r.stdout)
                seeded = str(dst)
        if verdict == "PASS" and leg.family == "perf" and host.cfg.get("perf_against"):
            r = [host.ssh(f"cat {d}/docs/perf_baselines.json", capture_output=True, timeout=120)
                 for d in (host.tree, host.against)]
            if not any(x.returncode for x in r):
                dst = out / "seeded" / host.arch / "perf_baselines.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(rerecorded(r[0].stdout, r[1].stdout, f"re-recorded {now()[:10]} on "
                                          f"{host.name} as the A/B reference ({host.cfg['perf_against']}) "
                                          f"of the {sha[:9]} release gate"))
                seeded = str(dst)
        if verdict == "PASS" and leg.name == ABB3_LEG:
            for f in ABB3_FILES:
                r = host.ssh(f"cat {shlex.quote(f'{rdir}/{Path(f).name}')}", capture_output=True, timeout=120)
                if r.returncode:
                    verdict = "FAIL"
                    break
                dst = out / "abb3" / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(r.stdout)
                seeded = str(out / "abb3")
        elif verdict == "PASS" and leg.family == "record":
            m = ladder_model(leg.name)
            dst = out / "recorded" / host.cfg["card_type"] / f"{m}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = [host.ssh(f"cat {shlex.quote(f'{rdir}/{f}')}", capture_output=True, timeout=120)
                 for f in (f"size_ladder_baseline.d/{m}.json", "size_ladder_baseline.json")]
            frag = (record_fragment(r[0].stdout, r[1].stdout, host.cfg["card_type"], m)
                    if not any(x.returncode for x in r) else None)
            if frag:
                dst.write_text(frag)
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
        if reads:
            res["reads"], res["scope"] = reads, scope
        if verdict in OK:
            ledger.put(res["key"], res)
        elif dead:
            raise CardDead(res, dead)
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


def hold_out(out: Path):
    """Create `out` and lock it for this runner: a second runner on it would rerun the same legs
    beside the first. The returned file holds the lock until it is closed or the process exits."""
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit(f"another gate_fanout.py is running on {out}")
    return lock


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
    ap.add_argument("--record-abb3", action="store_true",
                    help="add record:abb3, perf/abb3_port's records re-run on a p300c card and checked "
                         "against the committed ones; updated files land in <out>/abb3/")
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
    for c in cfg.values():
        if c.get("perf_against") == "last":
            c["perf_against"] = last_tag()
    workers = parse_workers(args.workers, {n: Host(n, c, sha) for n, c in cfg.items()})
    hosts = {}
    for h, _ in workers:
        hosts.setdefault(h.name, h)
    workers = [(hosts[h.name], c) for h, c in workers]
    timed = {(h.name, c) for h, c in parse_workers(args.timed, hosts)}
    # A dark host would fail its prepare below and take the whole gate with it: qb1 fell off the
    # LAN on 2026-10-16 22:3xZ and the train restarted, and lost, the next release's gate every
    # 20 minutes. Run on the hosts that answer; a leg only a dark host could run is UNRUN.
    # An arch whose every host is dark fails the gate (UNRUN) rather than drop out of it.
    asked = sorted({h.arch for h in hosts.values()})
    with ThreadPoolExecutor() as ex:
        up = dict(zip(hosts, ex.map(Host.answers, hosts.values())))
    for n in [n for n, ok in up.items() if not ok]:
        print(f"note: {n} does not answer ssh; running without its cards", file=sys.stderr)
        del hosts[n]
    if not hosts:
        raise SystemExit("no host in --workers answers ssh")
    workers = [(h, c) for h, c in workers if h.name in hosts]
    archs = sorted({h.arch for h in hosts.values()})
    if args.arch:
        archs = [a for a in archs if a in args.arch.split(",")]
    for a in archs:
        if not any(h.arch == a for h, c in workers if (h.name, c) in timed):
            print(f"note: no --timed card for {a}; its timed legs will not run", file=sys.stderr)

    # Every host installs one resolution: the one this commit, or an earlier one with the same
    # dependencies, was gated with; failing that the first host resolves from PyPI and records it.
    ledger = Ledger(args.ledger)
    spec = dep_spec(sha)
    rest = list(hosts.values())
    pins = ledger.pins(sha, spec)
    if pins is None:
        lead = rest.pop(0)
        lead.prepare()
        pins = lead.pins()
    ledger.save_pins(sha, spec, pins)
    with ThreadPoolExecutor() as ex:          # a failure re-raises here
        list(ex.map(lambda h: h.prepare(pins), rest))
    first = {a: next(h for h in hosts.values() if h.arch == a) for a in archs}
    roster = first[archs[0]].run_py(ENUMERATE)
    legs = select(build_legs(roster, test_files(sha), args.shards, args.record_lever, args.record_full,
                             args.record_abb3, tuple(sorted({h.cfg["card_type"] for h, _ in workers}))),
                  [p for p in args.legs.split(",") if p])
    content = content_hash(sha)
    legacy = content_hash(sha, baselines=True, tests=True, runner=RUNNER_FILES_0)
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
    # A host's `key_card_type` files its results under another card type whose results it
    # reproduces exactly (qb2's p300c keys as p150a: digests equal seed for seed). Timed legs
    # measure the card itself, so only a host whose card type is the key's may run them.
    ctype = {a: first[a].key_card_type for a in archs}
    # The baseline blocks a leg of each arch can read: every card type of that arch in hosts.json.
    reads = {a: {ctype[a]} | {c["card_type"] for c in cfg.values() if c.get("arch") == a} for a in archs}
    for h in hosts.values():
        if h.arch in archs and h.key_card_type != ctype[h.arch]:
            raise SystemExit(f"{h.name} keys as {h.key_card_type}, {first[h.arch].name} as {ctype[h.arch]}")
    for h, c in workers:
        if (h.name, c) in timed and h.cfg["card_type"] != h.key_card_type:
            raise SystemExit(f"--timed {h.name}:{c}: a {h.cfg['card_type']} keyed as {h.key_card_type} "
                             "cannot run timed legs")

    lock = hold_out(args.out)  # noqa: F841 (held until the runner exits)
    # Rows written while test files were inside the content key (until 2026-10-09 21Z), at this
    # commit or at any commit with the same code, which a leg accepts when its own inputs match too.
    same = {s for s in {sha} | ledger.shas() if s == sha or _same_code(s, content)}
    tests_in = {s: content_hash(s, tests=True, runner=RUNNER_FILES_0) for s in same}
    runner_in = {s: content_hash(s, runner=RUNNER_FILES_0) for s in same}
    keys, scopes, results, todo = {}, {}, [], {a: [] for a in archs}
    results += [{"leg": "every leg", "arch": a, "verdict": "UNRUN", "rc": None, "why": "no host of this arch answers ssh"}
                for a in asked if a not in archs and (not args.arch or a in args.arch.split(","))]
    # A record run whose runner died (pc rebooted 2026-10-13 18:13Z with 10 of 27 legs recorded)
    # resumes in the same --out: a record leg that passed there at this commit, with its fragment
    # still on disk, keeps that row. The ledger never reuses record legs, so nothing else would.
    prior = [json.loads(ln) for ln in (args.out / "results.jsonl").read_text().splitlines()] \
        if (args.out / "results.jsonl").exists() else []
    kept = {(r["leg"], r["arch"]): r for r in prior
            if r.get("sha") == sha and r["verdict"] == "PASS" and r["leg"].startswith("record:")
            and r.get("seeded") and Path(r["seeded"]).is_file()}
    # A fixture JSON changed in a fix-forward put every leg in a new scope (rel-c2b4b898e, 10-14:
    # 240 legs planned, 0 reused, though 125 read nothing the fix touched). Rows filed in the scope
    # that still keyed data files (until 2026-10-14) stay good where those files did not change.
    nonpy, nonpy_old = content_hash(sha, python=False, data=False), content_hash(sha, python=False)
    for lg in legs:
        on = [a for a in archs if not lg.card_types
              or any(h.arch == a and type_fits(h, lg) for h, _ in workers)] if lg.card else [archs[0]]
        if not on:
            results.append({"leg": lg.name, "arch": "any", "verdict": "UNRUN", "rc": None,
                            "why": f"no worker on a {'/'.join(lg.card_types)} card"})
        for a in on:
            slot = a if lg.card else "any"
            env, ct, hleg = (envs[f"{a} {first[a].python(lg.family)}"], ctype[a] if lg.card else "cpu",
                             first[a].leg(lg))
            model = ladder_model(lg.name) if lg.family == "ladder" else ""
            cards = reads[a] if lg.card else ()
            base, tst = baseline_hash(sha, lg.family, model=model, cards=cards), tests_hash(sha, lg.family)
            # An A/B perf leg reads the reference tree's numbers, not the committed baseline, so
            # committing a re-recorded baseline does not rerun it.
            ab = lg.family == "perf" and bool(first[a].cfg.get("perf_against"))
            if ab:
                base = ""
            k = leg_key(content, env, ct, hleg, base, tst)
            keys[(lg.name, slot)] = k
            # The same leg on any commit with the same non-Python files, whose Python read nothing
            # this commit changed (gate_reads.py), proved this commit's result too.
            scopes[(lg.name, slot)] = scope = leg_key(nonpy, env, ct, hleg, base, tst)
            scope_old = leg_key(nonpy_old, env, ct, hleg, base, tst)
            # Card-free legs always run: they are cheap, and pytest_cpu checks the recorded
            # measurements the key leaves out.
            # A row under an older key scheme proved the same code against the same inputs, so it
            # is as good as a row under the new one: baselines inside the content hash (legacy),
            # or test files inside it, at any commit whose code and this leg's inputs match.
            match = {s for s in same if s == sha or (baseline_hash(s, lg.family, model=model, cards=cards) == base
                                                     and tests_hash(s, lg.family) == tst)}
            olds = ([leg_key(legacy, env, ct, hleg)]
                    # rows filed when the ladder key was the files' blob ids (until 2026-10-11)
                    + [leg_key(content, env, ct, hleg, baseline_hash(s, lg.family, model=model), tst)
                       for s in match if cards and model]
                    + [leg_key(tests_in[s], env, ct, hleg, base) for s in match]
                    + [leg_key(runner_in[s], env, ct, hleg, base, tst) for s in match]
                    + [leg_key(content, env, ct, hleg, baseline_hash(s, "perf"), tst) for s in same if ab])
            hit = (None if args.no_reuse or not lg.card or lg.family == "record"
                   else next(filter(None, map(ledger.get, [k, *olds])), None) or ledger.read_hit(scope, sha)
                   or ledger.read_hit(scope_old, sha))
            owes = owed_by(roster, ctype[a], lg)
            if not owes and not any(h.arch == a and cap_fits(roster, h, lg) for h, _ in workers):
                owes = [CAP_OWED]
            if owes:
                results.append({"leg": lg.name, "arch": slot, "verdict": "OWED", "owed": owes,
                                "card_type": ctype[a]})
            elif hit and hit.get("verdict") in OK:
                results.append(_reused(lg.name, slot, k, hit))
            elif (lg.name, slot) in kept:
                results.append(kept[lg.name, slot])
            elif lg.card:
                todo[a].append(lg)
            else:
                todo.setdefault("any", []).append(lg)
    hist = known_walls(ledger.history(), prior)
    for q in todo.values():
        for lg in q:
            lg.known_s = hist.get(lg.name, 0)
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

    # Per run, so a job a stopped run left in a pool can never share a directory with a new one,
    # and a stopped run's cleanup (pool job names, end_attempt's GATE_ATTEMPT mark) can never
    # reach a later run on the same --out: on 10-11 one ended five legs of its successor.
    execute = make_executor(sha, args.out, ledger, keys, f"out-{sha[:12]}/{args.out.name}-{os.getpid()}", scopes)
    # Card-free legs run on a worker without a card on each host of the first arch (`cardfree`).
    # Started in the background by a non-interactive shell, SIGINT arrives ignored and the
    # cleanup below never ran; SIGTERM never reached it at all. Both stop the runner the same way.
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        results += Gate(todo, workers + cardfree(workers, archs[0]), timed, execute, args.out,
                        None if args.no_reuse else ledger, keys, roster).run()
    except KeyboardInterrupt:
        # A stopped runner takes back what it queued and ends every leg it started: a remote leg
        # outlives its ssh, and on 10-11 a stopped gate left 86 processes holding 23 cards.
        for job, host in list(QUEUED.items()):
            host.ssh(f"rm -f {shlex.quote(job)}", timeout=120)
        hosts, cmds = {}, {}
        for rdir, host in list(RUNNING.items()):
            hosts[host.name] = host
            cmds.setdefault(host.name, []).append(Host.end_attempt(rdir))
        for name, c in cmds.items():
            hosts[name].ssh("; ".join(c), timeout=120)
        raise
    ok = write_verdict(args.out, sha, archs, results)
    print(f"GATE {'PASS' if ok else 'FAIL'}: {args.out / 'VERDICT.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
