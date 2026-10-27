#!/usr/bin/env python3
"""Keep every SPD pool fed from experiment grids, so chips never wait for an agent turn.

Each row declares what it wants measured in state/spd/grids/<row>.yaml:

    row: spd-gap-of3
    tier: red            # red | amber | fast | fill: queue order across rows (GAP RED cells first)
    experiments:
      - name: wsd
        arch: wh         # wh = held Galaxies, bh = qb1/qb2
        prio: 4          # pool prio 3-8 (outside it is clamped)
        mem: 24
        cap: 3600
        cpu: 5           # optional: host cores the job really uses beyond its thread share; pool.sh limits such jobs
        hosts: [113]     # optional: only these pools, while SPD holds one of them (else any pool of the arch)
        active: true     # false parks the experiment without deleting it
        axes: {seed: [103, 104], cpx: [9W8A, 28VJ]}     # product, first axis outermost
        vars: {out: "~/spd/spd-gap-of3/wsd/s{seed}/{cpx}"}
        requires: ["~/spd/spd-gap-of3/trees/763b2621c/perf/spd/run_chip.sh"]
        also_named: ["4-spd-gap-of3-wsd-openfold3-{cpx}-s{seed}.sh"]   # hand-queued twin: skip while it is queued/running
        done: ["{out}/b/bench.jsonl", "{out}/w/bench.jsonl"]
        command: |
          bash ... run_chip.sh $CHIP {out} "b:... w:..." {cpx} 1800 --seed {seed}

A cell is done once every `done` path exists and is non-empty on any pool of its arch, so cells already measured on
the current tree (its paths name the tree) are skipped. Every pass (tick.d, ~10 min) feed tops each pool up to
2 x healthy chips of queued prio 0-8 work (a cell whose runs take under a pass counts as that fraction of a job),
cells in tier order, rows round-robin inside a tier.

A cell whose job ended twice within 60 s without its done files, or three times in all, is quarantined in
state/spd/QUARANTINE.md with its log tail, and its row is woken. The job name carries a hash of the rendered command,
so the cell comes back by itself once the row changes its grid entry.

Galaxies: jobs go to ~/spd/pool/queue/ and pool.sh starts them. qb1/qb2 have no pool: feed starts each job detached as
a waiter that takes the first usable card of its box to come free (its CHIPS.md flock free, no release leg on it);
the job itself gets a private LOCK so run_chip.sh does not wait on its own card. Waiters queue for cards through one
dispatch flock per box, so they start in about the order they were placed. A BH job still waiting after STEAL_S moves
to a free card on another BH box its cell allows; a Galaxy that offers no chip for STEAL_S hands its queued feed jobs
back to the fill, which places them on a pool with room.
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

D = Path("/home/moritz/.coworker")
sys.path.insert(0, str(D / "scripts"))
try:
    from glx_elastic import CHIP_LENDING  # JapanFold boxes that lend single chips to the SPD pool
except ImportError:  # no coworker checkout on this host
    CHIP_LENDING = set()
TIERS = {"red": 0, "amber": 1, "fast": 2, "fill": 3}
FAST_S = 60
TICK_S = 600   # feed's own cadence (tick.d). A queued job expected to run shorter fills that fraction of a depth slot:
               # 10-16 qb1 ran 8 one-to-five-minute cells dry in 5 min and idled 2 cards until the next pass.
INIT_FAIL = ("failed to initialize FW|ARC core .* failed to start|Failed to set initial power state"
             "|waiting for active ethernet core")   # 10-16 qb2 board (0,1) after a TERMed mesh run
MAX_FAILS = 3
STEAL_S = 900   # a BH job still waiting this long moves to a free card on another BH box its cell allows
# BH boxes: SPD cards and their flocks (state/spd/CHIPS.md). A chip kept off the feed has ~/spd/locks/chip<N>.BAD on its
# host (qb2 chip 0 since the 2026-10-14 slot swap: marginal at 1350 MHz, and feed trees predate the AICLK cap).
# `reset` resets a card after a capped job (see run_cmd). qb1 only: each p150a is its own board, while a qb2 p300c
# board carries two chips and a reset there would kill the other chip's job. A box with state/<name>-offline is no
# pool: its pins clear for cells of an experiment that has nothing done there, the rest wait for it (one experiment, one box), and so does an
# experiment with `pinned: true`, which measures the box it names (spd_backfill's per-box ladders and seeds).
BH_POOLS = {
    "qb1": {"ssh": "qb1", "cards": [0, 1, 2, 3], "lock": "/home/ttuser/spd_qb1_card{c}.lock",
            "reset": "/home/ttuser/.local/bin/tt-smi -r"},
    "qb2": {"ssh": "qb2", "cards": [0, 1, 2, 3], "lock": "/home/ttuser/spd_qb2_chip{c}.lock"},
}

# pgrep pattern for a release-gate leg on card %s, running or waiting for its flock. A leg that runs a perf script through
# runpy (bc2_boundary, parity_*) names the gate only in its RELEASE_GATE_RUN_DIR env, so the pattern keys on that too.
# Everything else that wants a card books it in the box's booking file (book.py: CFD windows, the gapboard column). A
# waiter takes a card only if its job's worst-case end (start + cap + 600 s of grace and reset, written to <job>.until)
# comes before the card's next booking; a waiter too long for the gap hands the dispatch flock on each poll, so a shorter
# job behind it can take the card. A renamed `sleep` that matches GATE_LEG is an old-style sentinel: it blocks for
# SENT_S (600 s) after it started and is ignored after that (10-16 qb1: cfd3 rt4 sentinels re-raised every 5 min kept
# three cards idle for hours with 21 jobs queued). Takers read worst-case ends from ~/spd/feed/jobs/*.until.
GATE_LEG = "(spd_qb1_card|spd_qb2_chip|TT_VISIBLE_DEVICES=)%s[. ].*(RELEASE_GATE_RUN_DIR|release_gate|gate_fanout|perf_ab)"

# Runs on the pool host: report state, existence of paths, and sweep dead entries. Payload is appended as JSON.
PROBE = r'''
import glob, json, os, shutil, subprocess, sys, time
A = json.loads(sys.stdin.readline())
H = os.path.expanduser("~"); now = time.time(); out = {"exists": {}, "jobs": {}}
def ok(p):
    p = os.path.expanduser(p)
    if any(c in p for c in "*?["):
        return any(ok(q) for q in glob.glob(p))
    # a result file must be non-empty; a marker (`touch $O/done`, `touch $O/$arm.done`, `touch $T.ok`) is empty by design
    # bench.py writes a "start" record before the device opens, so a bench.jsonl is done only once it holds a rep
    b = os.path.basename(p)
    if b == "bench.jsonl" and os.path.isfile(p):
        with open(p, "rb") as f:
            return b'"ev": "rep"' in f.read()
    return (os.path.isfile(p) and (os.path.getsize(p) > 0 or "." not in b or b.endswith((".done", ".ok")))
            or os.path.isdir(p) and bool(os.listdir(p)))
for p in A["paths"]:
    out["exists"][p] = ok(p)
def tail(p):
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 1500)); return f.read().decode(errors="replace")
    except OSError:
        return ""
def ended(sh, end, start_ref):
    rc = open(end).read().strip(); n = os.path.basename(sh).split("-", 2)[-1] if A["kind"] == "galaxy" else os.path.basename(sh)
    t = tail(sh + ".log")
    if len(t.strip()) < 200:   # most jobs write their own logs next to their results: tail the newest one there
        logs = [q for d in A.get("outdirs", {}).get(n, []) for q in glob.glob(os.path.expanduser(d) + "/*.log")
                + glob.glob(os.path.expanduser(d) + "/*/*.log") if os.path.getsize(q)]
        if logs:
            q = max(logs, key=os.path.getmtime); t += "\n[" + q + "]\n" + tail(q)
    return {"state": "ended", "dur": int(os.path.getmtime(end) - os.path.getmtime(start_ref)), "rc": rc,
            "at": int(os.path.getmtime(end)), "tail": t}
if A["kind"] == "galaxy":
    P = H + "/spd/pool"
    if A.get("lend"):  # a JapanFold box lending single chips: only the chips it lends now are ours
        lent = [json.load(open(f)) for f in glob.glob(H + "/japanfold/state/agent/lend/*.lent")]
        out["healthy"] = sum(1 for x in lent if x.get("recall_at") is None)
    else:
        bad = len(glob.glob(H + "/spd/locks/chip*.BAD"))
        out["healthy"] = len([n for n in os.listdir("/dev/tenstorrent") if n.isdigit()]) - bad
    q = [os.path.basename(f) for f in glob.glob(P + "/queue/*.sh")]
    out["queued"] = sum(1 for n in q if n[0] in "012345678")
    out["qprio"] = [sum(1 for n in q if n[0] == str(d)) for d in range(9)]
    out["paused"] = os.path.exists(P + "/PAUSE") or not A.get("lend") and os.path.exists(P + "/RETURNED")
    out["live"] = q + [os.path.basename(r).split("-", 2)[2] for r in glob.glob(P + "/running/*.sh")
                       if not os.path.exists(r + ".end")]
    for n in q:
        if "-f-" in n: out["jobs"][n] = {"state": "queued"}
    for sh in sorted(glob.glob(P + "/running/*-f-*.sh"), key=os.path.getmtime):
        n = os.path.basename(sh).split("-", 2)[2]          # <stamp>-c<u>-<name>
        out["jobs"][n] = ended(sh, sh + ".end", sh) if os.path.exists(sh + ".end") else {"state": "running"}
    # sweep: renamed-dead entries (.moved/.stopped/.dup/...) older than 2 h go to archive/<day>/<dir>/
    arch = P + "/archive/" + time.strftime("%Y%m%d"); moved = 0
    for d in [P + "/queue", P + "/running"] + glob.glob(P + "/backlog/*"):
        if not os.path.isdir(d): continue
        for e in os.scandir(d):
            n = e.name
            if not e.is_file() or n.endswith((".sh", ".sh.log", ".sh.end")) or n.startswith("."): continue
            if not A["sweep"] or now - e.stat().st_mtime < 7200: continue
            t = os.path.join(arch, os.path.relpath(d, P)); os.makedirs(t, exist_ok=True)
            shutil.move(e.path, os.path.join(t, n)); moved += 1
    out["swept"] = moved
else:
    J = H + "/spd/feed/jobs"; os.makedirs(J, exist_ok=True)
    bk = H + "/spd/feed/book.py"   # the box's card arbiter (book.py), shipped by the feed so every runner reads one version
    if A.get("book") and (not os.path.exists(bk) or open(bk).read() != A["book"]):
        open(bk + ".tmp", "w").write(A["book"]); os.replace(bk + ".tmp", bk)
    # A row that concludes leaves its booking standing: 10-23 rfd4-land8 and rfd4-perf-ptrans held qb2 cards 3 and 1
    # for ~4 h after concluding, no process on them, 14 waiters held off. Drop a booking whose owner has concluded.
    W = H + "/spd/feed/windows.tsv"
    if A.get("sweep") and A.get("concluded") and os.path.exists(W):
        for o in sorted({l.split("\t")[2] for l in open(W) if l.count("\t") >= 3} & set(A["concluded"])):
            subprocess.run(["python3", bk, "rm", o], capture_output=True); out.setdefault("unbooked", []).append(o)
    cards = [c for c in A["cards"] if not os.path.exists(H + "/spd/locks/chip%d.BAD" % c)]
    # dead ARC: the class node reads 0xFFFFFFFF (10-15 qb1 card 2 for 70+ min while jobs kept dying there in init).
    # Logical card c is the c-th class node in PCI order, the order TT_VISIBLE_DEVICES and tt-smi use.
    sy = os.environ.get("TT_SYSFS", "/sys/class/tenstorrent")
    nodes = [d for _, d in sorted((os.path.basename(os.path.realpath(d + "/device")), d) for d in glob.glob(sy + "/*"))]
    def dead(c):
        return c < len(nodes) and any(open(f).read().strip() == "4294967295" for f in
                                      (nodes[c] + "/tt_aiclk", nodes[c] + "/tt_heartbeat") if os.path.exists(f))
    out["dead"] = {str(c): int(nodes[c].rsplit("!", 1)[1]) for c in cards if dead(c)}
    cards = [c for c in cards if str(c) not in out["dead"]]
    AD = H + "/spd/feed/arc"; out["arc"] = []; out["arc_busy"] = []
    for f in glob.glob(AD + "/card*.pid"):
        b = f[:-4]; c = int(os.path.basename(b)[4:].split(".")[0])
        if os.path.exists(b + ".result"):
            out["arc"].append({"card": c, "at": int(b.rsplit(".", 1)[1]), "result": open(b + ".result").read().strip()})
        elif os.path.exists("/proc/" + open(f).read().strip()):
            out["arc_busy"].append(c)
    out["healthy"] = len(cards); out["cards"] = cards; queued = 0; held = set(); out["qprio"] = [0] * 9
    # an ended job older than a day was accounted long ago (the feed reads every 10 min); tailing every one of them
    # each tick timed the qb2 probe out at 120 s on 10-25 with 3357 records, and the box got no work. Archive them.
    JA = H + "/spd/feed/archive/" + time.strftime("%Y%m%d") + "/jobs"
    for sh in glob.glob(J + "/*.sh"):
        n = os.path.basename(sh); card = int(open(sh + ".card").read()) if os.path.exists(sh + ".card") else -1
        if A["sweep"] and os.path.exists(sh + ".end") and now - os.path.getmtime(sh + ".end") > 86400:
            os.makedirs(JA, exist_ok=True)
            for f in [sh] + glob.glob(glob.escape(sh) + ".*"):
                shutil.move(f, os.path.join(JA, os.path.basename(f)))
            continue
        if os.path.exists(sh + ".end"):
            out["jobs"][n] = ended(sh, sh + ".end", sh + ".start" if os.path.exists(sh + ".start") else sh)
        elif not os.path.exists(sh + ".pid") or not os.path.exists("/proc/" + open(sh + ".pid").read().strip()):
            open(sh + ".end", "w").write("lost\n"); out["jobs"][n] = ended(sh, sh + ".end", sh)
        elif os.path.exists(sh + ".start"):
            out["jobs"][n] = {"state": "running"}; held.add(card)
        else:
            out["jobs"][n] = {"state": "queued", "age": int(now - os.path.getmtime(sh))}; queued += 1
            if n[0].isdigit(): out["qprio"][int(n[0])] += 1
    import fcntl, subprocess
    def lock_free(c):
        try:
            with open(A["lock"].format(c=c), "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN); return True
        except (OSError, KeyError):
            return False
    rel = lambda c: subprocess.run(["pgrep", "-f", GATE_LEG % c], capture_output=True).returncode == 0
    # a box's own waiters take its free cards within a poll, so only a box with none waiting offers cards to steal
    out["free"] = [] if queued else [c for c in cards if c not in held and lock_free(c) and not rel(c)]
    out["live"] = []
    out["queued"] = queued; out["paused"] = os.path.exists(J + "/PAUSE")
print(json.dumps(out))
'''
PROBE = f"GATE_LEG = {GATE_LEG!r}\n" + PROBE

# Runs a BH job under its cap: SIGINT at the cap, then SIGTERM to the job's whole process group after a grace period.
# SIGINT alone cannot interrupt a native wait: on 10-15 a device-init hang outlived its bash wrapper as an orphan and
# kept the card's flock (inherited fd 9) for 2 h. The group is the inner timeout's pid, kept in $1.pg. The .end says
# what ended a capped job: "124 INT" or "124 INT+TERM". Never SIGKILL a device job.
# A capped job leaves its card dirty: on 10-15 qb1 three PV2 fast seed cells hung, and every later job on those three
# cards failed device init ("failed to initialize FW", "ARC core failed to start") until reset. So once the group is
# gone, and while the card's flock is still held, $6 (the box's reset command, empty = none) resets the card; the .end
# then ends in "reset" or "reset-failed", or "noreset" if the group outlived SIGTERM too.
# The reset names the card's /dev node (`node`, from NODE, which the caller defines first): a bare logical id needs
# tt-smi to enumerate every card, which a dead ARC can hang. `dead` is true while the card's class node reads 0xFFFFFFFF.
# A card can also come out dirty from something that was not capped (10-16 00:12Z qb1 card 0: six jobs of four rows failed
# init in 15 s each, "failed to initialize FW! Try resetting the board", sysfs healthy). Two DIFFERENT jobs failing fast
# back to back on one card is the card, not the cells: the second resets it and its .end says "fastfail2 reset", which
# the ledger does not count against the cell. The first leaves its name in fastfail.<card>; a clean exit clears it.
# Only a device-init failure (INIT_FAIL in the job's log) counts, so two broken cells never reset a healthy card. qb2 has no
# reset (p300c board pair): the second marks ~/spd/locks/chip<N>.BAD instead and its .end says "fastfail2 bad".
# Most cells send the model's output to their own log, not stdout (10-16 qb2: 12 init failures on board (0,1) in 7 min, none
# seen, four cells quarantined for the board's fault), so the check also reads the files the job wrote since its .start in
# the cell's output dirs, which PLACE lists in $1.outs.
NODE = ('node() { for s in "${TT_SYSFS:-/sys/class/tenstorrent}"/*; do echo "$(basename "$(readlink -f "$s/device")") '
        '${s##*!}"; done | sort | sed -n "$(($1 + 1))s/.* //p"; }; '
        'dead() { local n=$(node $1); [ -n "$n" ] && cat "${TT_SYSFS:-/sys/class/tenstorrent}/tenstorrent!$n/tt_aiclk" '
        '"${TT_SYSFS:-/sys/class/tenstorrent}/tenstorrent!$n/tt_heartbeat" 2>/dev/null | grep -qx 4294967295; }; '
        'gate() { local p n=$(($(date +%s) + $1 + 600)) b; for p in $(pgrep -f "$G"); do '
        '[ "$(cat /proc/$p/comm 2>/dev/null)" = sleep ] && [ "$(ps -o etimes= -p $p | tr -d " ")" -ge "${SENT_S:-600}" ] 2>/dev/null && continue; '
        'return 0; done; b=$(python3 "${SPD_BOOK:-$HOME/spd/feed/book.py}" next $c 2>/dev/null) || b=; '
        '[ $n -le "${b:-9999999999}" ] && return 1; W=1; return 0; }; ')


def run_cmd(grace=120):
    return ('S=$(date +%s); CHIP=$2 TT_VISIBLE_DEVICES=$2 LOCK="$4" WAIT=60 timeout -s TERM $(($5 + G)) '
            'sh -c \'echo $$ > "$1.pg"; exec timeout -s INT "$2" bash "$1"\' _ "$1" "$5" > "$1.log" 2>&1; rc=$?; '
            'M="$(dirname "$1")/fastfail.$2"; if [ $rc = 0 ]; then rm -f "$M"; '
            'elif [ $rc != 124 ] && [ $rc != 127 ] && [ $(($(date +%s) - S)) -lt FAST ] && { grep -qE "INIT" "$1.log" || { [ -s "$1.outs" ] && while read -r d; do find "$d" -maxdepth 1 -type f -newer "$1.start" 2>/dev/null; done < "$1.outs" | xargs -r -d "\\n" grep -lsE "INIT" | grep -q .; }; }; then '
            'if [ -s "$M" ] && [ "$(cat "$M")" != "$1" ]; then rm -f "$M"; '
            'if [ -z "$6" ]; then L="${SPD_LOCKS:-$HOME/spd/locks}"; mkdir -p "$L"; '
            'echo "$(date -u +%FT%TZ) two jobs failed device init back to back ($(basename "$1")), feed" > "$L/chip$2.BAD"; '
            'rc="$rc fastfail2 bad"; '
            'elif timeout 180 $6 "${TT_DEV:-/dev/tenstorrent}/$(node $2)" >> "$1.log" 2>&1; then rc="$rc fastfail2 reset"; '
            'else rc="$rc fastfail2 reset-failed"; fi; else echo "$1" > "$M"; fi; fi; '
            'if [ "$rc" = 124 ]; then g=$(cat "$1.pg" 2>/dev/null); s=INT; [ $(($(date +%s) - S)) -lt $(($5 + G)) ] || s=INT+TERM; '
            'for i in $(seq G); do kill -0 -- -$g 2>/dev/null || break; sleep 1; done; '
            '[ -n "$g" ] && kill -0 -- -$g 2>/dev/null && { kill -TERM -- -$g; s=INT+TERM; }; '
            'if [ -n "$6" ]; then for i in $(seq G); do kill -0 -- -$g 2>/dev/null || break; sleep 1; done; '
            'if kill -0 -- -$g 2>/dev/null; then s="$s noreset"; elif timeout 180 $6 "${TT_DEV:-/dev/tenstorrent}/$(node $2)" >> "$1.log" 2>&1; then s="$s reset"; '
            'else s="$s reset-failed"; fi; fi; rc="$rc $s"; fi; '
            'echo "$rc" > "$1.end"').replace(" G)", f" {grace})").replace("seq G", f"seq {grace}").replace("-lt FAST ]", f"-lt {FAST_S} ]").replace('"INIT"', f"'{INIT_FAIL}'")


RUN = run_cmd()
BOOK = (Path(__file__).parent / "book.py").read_text() if (Path(__file__).parent / "book.py").exists() else ""


def outs(c):
    """The dirs a cell writes its done files to: its own logs live there (PLACE writes them to <job>.outs)."""
    return sorted({os.path.dirname(x) for x in c["done"]})

# Writes the jobs feed placed. Galaxy: into the pool queue. BH: detached flock waiter per job.
PLACE = r'''
import json, os, subprocess, sys
A = json.loads(sys.stdin.readline()); H = os.path.expanduser("~")
for j in A["jobs"]:
    if A["kind"] == "galaxy":
        # the Galaxy's `python` is the JapanFold venv, on PATH only after env.sh (a bare `python` job fails in 1 s)
        q = H + "/spd/pool/queue"; t = q + "/." + j["name"]
        open(t, "w").write("[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1\n" + j["body"])
        os.rename(t, q + "/" + j["name"])
    else:
        J = H + "/spd/feed/jobs"; sh = J + "/" + j["name"]
        for x in (".end", ".start", ".pid", ".log", ".pg", ".until", ".want", ".w"):
            if os.path.exists(sh + x): os.remove(sh + x)
        open(sh, "w").write(j["body"]); open(sh + ".card", "w").write("-1")
        open(sh + ".outs", "w").write("".join(os.path.expanduser(d) + "\n" for d in j.get("outs", [])))
        # Take the box's dispatch flock in turn, then the first card that is not BAD or dead-ARC, runs no release-gate leg, has
        # no booking before the job's worst-case end (checked again once its flock is held) and whose flock is free.
        # 10-15: waiters bound to one card queued on busy cards 2 and 3 while qb1 card 0 sat free 15+ min.
        # A card goes to the lowest prio digit waiting for it, as pool.sh does on a Galaxy: a waiter leaves a card to a live,
        # unstarted waiter of lower digit that wants it ($1.want) and is not held off by a booking ($1.w), and hands the
        # dispatch flock on. 10-20 qb2: prio-8 backfill placed first held both cards for hours over four prio-3 row jobs.
        w = (NODE + 'hi() { local f p b=${1##*/}; for f in "$J"/[0-9]-*.sh; do p=${f##*/}; [ "${p%%-*}" -lt "${b%%-*}" ] 2>/dev/null || continue; '
             '[ -e "$f.start" ] || [ -e "$f.end" ] || [ -e "$f.w" ] && continue; grep -qw "$2" "$f.want" 2>/dev/null || continue; '
             'kill -0 "$(cat "$f.pid" 2>/dev/null)" 2>/dev/null && return 0; done; return 1; }; '
             'echo "$2" > "$1.want"; exec 8>"$(dirname "$1")/.dispatch"; flock -w 86400 8 || { echo flockbusy > "$1.end"; exit 3; }; '
             'J="$(dirname "$1")"; D=$((SECONDS + 86400)); while :; do W=; Y=; for c in $2; do [ -e ~/spd/locks/chip$c.BAD ] && continue; dead $c && continue; '
             'hi "$1" $c && { Y=1; continue; }; G="' + GATE_LEG % "$c" + '"; '
             'gate "$5" && continue; exec 9>"${3//@/$c}"; '
             'flock -n 9 && ! gate "$5" && break 2; exec 9>&-; done; '
             '[ $SECONDS -lt $D ] || { echo flockbusy > "$1.end"; exit 3; }; '
             'if [ -n "$W" ]; then touch "$1.w"; else rm -f "$1.w"; fi; '
             'if [ -n "$W$Y" ]; then flock -u 8; sleep "${POLL:-20}"; flock -w 86400 8 || { echo flockbusy > "$1.end"; exit 3; }; '
             'else sleep "${POLL:-20}"; fi; done; '
             'exec 8>&-; rm -f "$1.w"; echo $c > "$1.card"; touch "$1.start"; echo $(($(date +%s) + $5 + 600)) > "$1.until"; set -- "$1" $c "$3" "$(dirname "$1")/nolock.$c" "$5" "$6"; '
             'cd "$(dirname "$1")"; ' + RUN)
        p = subprocess.Popen(["setsid", "nohup", "bash", "-c", w, "_", sh, " ".join(map(str, j["cards"])),
                              j["lock"].replace("{c}", "@"), "", str(j["cap"] + 300), A.get("reset", "")],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=J)
        open(sh + ".pid", "w").write(str(p.pid))
for n in A.get("unqueue", []):
    p = H + "/spd/pool/queue/" + n
    if os.path.exists(p): os.makedirs(H + "/spd/pool/quarantine", exist_ok=True); os.rename(p, H + "/spd/pool/quarantine/" + n)
print(len(A["jobs"]))
'''
PLACE = f"NODE = {NODE!r}\nRUN = {RUN!r}\nGATE_LEG = {GATE_LEG!r}\n" + PLACE   # the waiter string above runs on the BH host, where only what PLACE defines exists


# Withdraws BH waiters that have not started: kill the waiter's session (feed started it under setsid), then set the
# job aside as .moved so the probe no longer sees it. Prints the names withdrawn before they started.
WITHDRAW = r'''
import json, os, signal, sys, time
A = json.loads(sys.stdin.readline()); J = os.path.expanduser("~/spd/feed/jobs"); got = []
if A.get("kind") == "galaxy":   # pool.sh starts only queue/*.sh; a job it already moved to running/ fails the rename
    for n in A["names"]:
        q = os.path.expanduser("~/spd/pool/queue/" + n)
        try: os.rename(q, q + ".moved"); got.append(n)
        except OSError: pass
    A["names"] = []
for n in A["names"]:
    sh = J + "/" + n
    if not os.path.exists(sh) or os.path.exists(sh + ".start") or os.path.exists(sh + ".end"): continue
    try: os.killpg(int(open(sh + ".pid").read()), signal.SIGTERM)
    except (OSError, ValueError): pass
    time.sleep(0.5)
    if os.path.exists(sh + ".start"): continue          # took its card in that half second: let it run
    os.rename(sh, sh + ".moved")
    for x in (".pid", ".card"):
        if os.path.exists(sh + x): os.remove(sh + x)
    got.append(n)
print(json.dumps(got))
'''


# One bounded reset of a dead-ARC qb1 card, detached on the host: hold the card's flock (no job lands meanwhile), wait
# until no process has the node open, reset by node path, then verify the class node no longer reads 0xFFFFFFFF.
# card<c>.<at>.result says ok / still-dead / reset-failed / busy (node never closed) / nolock, or "died" if a signal ended
# the script first. Every exit (a signal included) appends one "end" line with the reset's rc and the sysfs reading;
# a signal during tt-smi is handled once tt-smi returns, so a reset is never cut off half way.
ARC_SH = ('N="${TT_DEV:-/dev/tenstorrent}/$3"; C="${TT_SYSFS:-/sys/class/tenstorrent}/tenstorrent!$3"; '
          'trap \'[ -s "$1.result" ] || echo died > "$1.result"; echo "$(date -u +%FT%TZ) end rc=${rc:-none} '
          'aiclk=$(cat "$C/tt_aiclk") hb=$(cat "$C/tt_heartbeat") result=$(cat "$1.result")" >> "$1.log"\' EXIT; '
          'trap "exit 143" TERM INT HUP; rm -f "$1.result"; '
          'exec 9>"$2"; flock -w 7200 9 || { echo nolock > "$1.result"; exit; }; '
          'for i in $(seq ${ARC_WAIT:-360}); do fuser "$N" >/dev/null 2>&1 || break; sleep 10; done; '
          'fuser "$N" >/dev/null 2>&1 && { echo busy > "$1.result"; exit; }; '
          'echo "$(date -u +%FT%TZ) before aiclk=$(cat "$C/tt_aiclk") hb=$(cat "$C/tt_heartbeat")" >> "$1.log"; '
          'timeout 180 $4 "$N" >> "$1.log" 2>&1; rc=$?; [ $rc = 0 ] || { echo reset-failed > "$1.result"; exit; }; '
          'sleep ${ARC_SETTLE:-10}; '
          'cat "$C/tt_aiclk" "$C/tt_heartbeat" | grep -qx 4294967295 && r=still-dead || r=ok; echo $r > "$1.result"')
ARCFIX = r'''
import json, os, subprocess, sys
A = json.loads(sys.stdin.readline()); D = os.path.expanduser("~/spd/feed/arc"); os.makedirs(D, exist_ok=True)
b = "%s/card%d.%d" % (D, A["card"], A["at"])
p = subprocess.Popen(["setsid", "nohup", "bash", "-c", A["sh"], "_", b, A["lock"].format(c=A["card"]), str(A["node"]),
                      A["reset"]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=D)
open(b + ".pid", "w").write(str(p.pid)); print(p.pid)
'''
BAD = r'''
import json, os, sys
A = json.loads(sys.stdin.readline()); D = os.path.expanduser("~/spd/locks"); os.makedirs(D, exist_ok=True)
open("%s/chip%d.BAD" % (D, A["card"]), "w").write(A["why"] + "\n"); print(1)
'''


def arc(R, st, P, log, now=None):
    """Dead-ARC cards (class node reads 0xFFFFFFFF). qb1: one bounded reset; a second death within 24 h cardblocks the
    card and wakes the orchestrator. qb2: chipN.BAD, never a reset (a p300c board carries two chips). Every step is a
    fleet.log line. History: state/spd/feed/arc.json."""
    now = int(now or time.time()); f = R / "state/spd/feed/arc.json"
    H = json.loads(f.read_text()) if f.exists() else {}

    def fleet(m):
        log(m)
        with open(R / "fleet.log", "a") as fl:
            fl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} spd feed: {m}\n")
    for n, s in st.items():
        for r in s.get("arc", []):
            h = H.setdefault(f"{n}:{r['card']}", {"resets": [], "seen": []})
            if r["at"] not in h["seen"]:
                h["seen"].append(r["at"]); fleet(f"{n} card {r['card']} dead-ARC reset: {r['result']}")
        for c, node in s.get("dead", {}).items():
            c = int(c); h = H.setdefault(f"{n}:{c}", {"resets": [], "seen": []})
            if c in s.get("arc_busy", []):
                continue
            why = f"dead ARC (class node {node} reads 0xFFFFFFFF)"
            try:
                if not P[n].get("reset"):
                    remote(P[n]["ssh"], BAD, {"card": c, "why": f"{time.strftime('%FT%TZ', time.gmtime(now))} {why}, feed"})
                    fleet(f"{n} chip {c} {why}: marked chip{c}.BAD, no reset (p300c board pair)")
                elif [t for t in h["resets"] if now - t < 86400]:
                    remote(P[n]["ssh"], BAD, {"card": c, "why": f"{time.strftime('%FT%TZ', time.gmtime(now))} {why} again "
                                                              f"within 24 h of a reset, feed"})
                    (R / f"state/cardblock-{n}-{c}").write_text(f"{why} again within 24 h of a reset "
                                                                f"({time.strftime('%FT%TZ', time.gmtime(now))}); "
                                                                "set by perf/spd/feed.py; needs RMA triage\n")
                    fleet(f"{n} card {c} {why} again within 24 h of a reset: cardblocked, chip{c}.BAD")
                    wake(R, "spd-orchestrator", f"{n} card {c} dead ARC twice in 24 h, cardblocked", log)
                else:
                    remote(P[n]["ssh"], ARCFIX, {"card": c, "node": node, "at": now, "lock": P[n]["lock"],
                                                  "reset": P[n]["reset"], "sh": ARC_SH})
                    h["resets"].append(now); fleet(f"{n} card {c} {why}: one bounded reset under its card flock")
            except Exception as e:
                log(f"{n}: dead-ARC handling of card {c} failed: {e}")
    write_json(f, H)


def remote(ssh, script, payload, timeout=120):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh,
                        "python3 -c " + _q(script)], input=json.dumps(payload) + "\n",
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{ssh}: rc {r.returncode}: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _q(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


class Fmt(dict):
    def __missing__(self, k):
        raise KeyError(f"grid placeholder {{{k}}} is not an axis or var")


def render(s, ctx):
    # {name} placeholders only; shell ${VAR} and $VAR pass through untouched.
    return re.sub(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(Fmt(ctx)[m.group(1)]), s)


# Paths a command reads before it can do anything: what it cd's into, sources, runs with bash, or names as its tree.
# A job placed on a pool that lacks one ends in a second; on a Galaxy two such ends fence a healthy chip.
INPUT = re.compile(r"(?:\bcd|(?:^|[;&|(]\s*)\.|\bsource|\bbash|\bTREE=|--tree|\bPYTHONPATH=)[ =]?\s*((?:~|\$HOME|/home)/[^\s;:&|)\"']+)", re.M)


# bench.py's inputs by name: run_chip.sh CHIP OUT ARMS INPUTS ... or bench.py --inputs A,B. They resolve to
# ~/spd-data/inputs/<name>.yaml with the MSA cache beside it (cache-only), so a box without them fails every cell in a
# second: .103 10-24 (refcache) and .108 10-25 (9*_512 inputs, ~40 spd-m-protenix cells in 5 min).
DATA = "~/spd-data"
BENCH_INPUTS = re.compile(r"run_chip\.sh\s+\S+\s+\S+\s+(?:\"[^\"]*\"|'[^']*'|\S+)\s+([\w.,-]+)|--inputs[ =]([\w.,-]+)")


def inputs(cmd):
    optional = set(re.findall(r"\[ -[efsd] (\S+) \] &&", cmd))      # `[ -f X ] && . X` runs without X
    guarded = re.findall(r"\[ -[efsd] ((?:~|\$HOME|/home)/\S+) \] \|\|", cmd)   # `[ -s X ] || exit 1` cannot
    written = set(re.findall(r">>?\s*((?:~|\$HOME|/home)/[^\s;:&|)\"']+)", cmd))  # `cat > X; bash X` writes X itself
    data = set() if "--data" in cmd else {f"{DATA}/inputs/{n}.yaml" for m in BENCH_INPUTS.findall(cmd)
                                          for n in (m[0] or m[1]).split(",") if n and not n.isdigit()}
    return data | {re.sub(r"^\$HOME/", "~/", p) for p in INPUT.findall(cmd) + guarded
                   if p not in optional | written and "$" not in p.replace("$HOME/", "", 1)}


def lead_rows(root):
    """Live rows dedicated to a steer directive's group (state/spd/steer/config.json "directives"). They may queue at
    prio 2: at 3 Protenix's bhcen3 tied with esmc's prio-3 jobs on qb2 and raced them for the dispatch flock (10-20).
    A row shared with another group does not hold the directive, as in spd_steer.py."""
    try:
        cfg = json.loads((root / "state/spd/steer/config.json").read_text())
        lines = (root / "state/spd/OWNERS.tsv").read_text().splitlines()
    except (OSError, ValueError):
        return set()
    group = {m: g for g, ms in cfg.get("groups", {}).items() for m in ms}
    lead, out = set(cfg.get("directives", {})), set()
    for line in lines:
        p = line.split("\t")
        if line.startswith("#") or len(p) < 3 or (root / "state/concluded" / p[0]).exists():
            continue
        gs = {group.get(m, m) for m in p[2].split(",") if m}
        if len(gs) == 1 and gs <= lead:
            out.add(p[0])
    return out


def cells(grid_dir, log=print, lead=frozenset(), done=frozenset()):
    """All cells of all active experiments, ordered by tier, rows round-robin within a tier. Rows in `lead` may use
    prio 2 (lead_rows). A row in `done` has concluded and nobody reads its cells: on 10-24 spd-gap-esmf (concluded
    10-15) and spd-pairchain still held ~45 jobs and 6 chips of the one lent Galaxy."""
    by_tier = {}
    for f in sorted(Path(grid_dir).glob("*.yaml")):
        try:  # one unreadable grid file must not stop every row's feed
            g = yaml.safe_load(f.read_text()) or {}
            row, tier = g["row"], TIERS[g.get("tier", "amber")]
        except Exception as x:
            log(f"skip {f.name}: {type(x).__name__}: {str(x).splitlines()[0] if str(x) else ''}")
            continue
        if row in done:
            continue
        for e in g.get("experiments", []):
            if not e.get("active", True):
                continue
            try:  # one bad experiment must not stop every row's feed (10-15 19:02-19:20Z)
                prio = int(e.get("prio", 5))
            except (TypeError, ValueError):
                log(f"skip {f.name}:{e['name']}: prio {e.get('prio')!r} is not a number")
                continue
            lo = 2 if row in lead else 3  # 0-1 (2 outside a directive) is hand-queued release/integrate work
            if not lo <= prio <= 8:  # a skip left wsv unqueued (10-17)
                log(f"clamp {f.name}:{e['name']}: prio {prio} -> {min(max(prio, lo), 8)} (pool prio {lo}-8)")
                prio = min(max(prio, lo), 8)
            axes = e.get("axes", {})
            out = []
            for vals in itertools.product(*axes.values()):
                ctx = dict(zip(axes, vals))
                for k, v in (e.get("vars") or {}).items():
                    ctx[k] = render(str(v), ctx)
                cmd = render(e["command"], ctx)
                body = (f"# spd-feed: {row} {e['name']} {' '.join(f'{k}={v}' for k, v in zip(axes, vals))} "
                        f"(grid state/spd/grids/{f.name})\n# mem: {e.get('mem', 24)}\n# cap: {e.get('cap', 7200)}\n"
                        + (f"# cpu: {int(e['cpu'])}\n" if e.get("cpu") else "") + f"export TT_BIO_LEASE_HOLDER={row}\n{cmd.rstrip()}\n")
                h = hashlib.sha1(body.encode()).hexdigest()[:6]
                cid = re.sub(r"[^A-Za-z0-9_.]+", "_", "-".join(map(str, vals))) or "x"
                key = f"{row}/{e['name']}/{cid}"
                out.append({"key": key, "row": row, "hash": h, "arch": e.get("arch", "wh"), "pinned": bool(e.get("pinned")),
                            "hosts": [str(x) for x in e.get("hosts", [])], "cap": int(e.get("cap", 7200)),
                            "name": f"{prio}-{row}-f-{e['name']}-{cid}-{h}.sh", "body": body,
                            "done": [render(p, ctx) for p in e["done"]],
                            "also": [render(p, ctx) for p in e.get("also_named", [])],
                            "requires": sorted(set(render(p, ctx) for p in e.get("requires", [])) | inputs(cmd))})
            by_tier.setdefault(tier, []).append(out)
    ordered = []
    for t in sorted(by_tier):
        for group in itertools.zip_longest(*by_tier[t]):
            ordered += [c for c in group if c]
    return ordered


def donors(root):
    """Every Galaxy, whoever holds it: a file is only read there, never a device touched."""
    return [f"ubuntu@172.16.102.{ip}" for ip in json.loads((root / "state/glx/ALLOC.json").read_text())["galaxies"]]


def pools(root):
    a = json.loads((root / "state/glx/ALLOC.json").read_text())["galaxies"]
    p = {ip: {"name": ip, "ssh": f"ubuntu@172.16.102.{ip}", "kind": "galaxy", "arch": "wh"}
         for ip, g in a.items() if g.get("owner") == "spd" and g.get("phase") == "lent"}
    # A JapanFold box in glx_elastic.CHIP_LENDING lends single idle chips to its pool (state/glx/DESIGN-shared-chips.md).
    # Without it every WH grid waited for a whole-box lend while .107 lent ~20 chips (10-18).
    p.update({ip: {"name": ip, "ssh": f"ubuntu@172.16.102.{ip}", "kind": "galaxy", "arch": "wh", "lend": True}
              for ip, g in a.items() if g.get("owner") == "japanfold" and ip in CHIP_LENDING})
    for name, b in BH_POOLS.items():
        if not (root / f"state/{name}-offline").exists():
            p[name] = dict(b, name=name, kind="bh", arch="bh")
    return p


def runs(p, c):
    """Pool `p` can take cell `c`: its arch and its pins."""
    return p["arch"] == c["arch"] and (not c["hosts"] or p["name"] in c["hosts"])


SHA_DIR = re.compile(r"[a-z-]{0,8}?([0-9a-f]{7,40})")


def tree_root(path, repo, _seen={}):
    """(tree dir, commit) if `path` lies in a tt-bio tree named by its commit (trees/<sha>, tree-<sha>, t<sha>, m<sha>...)."""
    parts = path.split("/")
    for i, comp in enumerate(parts):
        m = SHA_DIR.fullmatch(comp)
        if not m:
            continue
        if m.group(1) not in _seen:
            r = subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "-q", m.group(1) + "^{commit}"],
                               capture_output=True, text=True)
            _seen[m.group(1)] = r.stdout.strip() if r.returncode == 0 else None
        if _seen[m.group(1)]:
            return "/".join(parts[:i + 1]), m.group(1)
    return None


SMALL = (".sh", ".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".env")   # copyable by name; builds and venvs are not


def provision(ssh, trees, files, donors, repo, lockdir):
    """Ship what a pool lacks, detached, one shipper per pool: missing trees as a git archive of the named commit
    (+ TREE_SHA), and small files (scripts, sourced configs, <= 1 MB) copied from the first other box that has them.
    A box lent after a row set up its jobs has neither, and every job of that row would end in a second there."""
    steps = []
    for d, sha in sorted(trees.items()):
        # Main moves ~50 times a day and two trees differ in ~10 files, yet streaming the archive sent all ~880 MB over
        # pc's Wi-Fi each time and starved every other ssh (10-27). So extract on pc and rsync by checksum (git archive
        # stamps every file with the commit time, a quick check matches nothing) against the box's newest tree.
        rd = d[2:] if d.startswith("~/") else d
        d = d.replace("~", "$HOME", 1) if d.startswith("~/") else d
        q, part = f'"{d}"', f'"{d}.part"'
        probe = (f"[ -e {q} ] && echo HAVE && exit 0; mkdir -p \"$(dirname {q})\"; "
                 f"ls -dt \"$(dirname {q})\"/*/TREE_SHA 2>/dev/null | grep -v '[.]part/' | head -1")
        steps.append(f"b=$(ssh -o BatchMode=yes {ssh} {shlex.quote(probe)}); [ \"$b\" = HAVE ] || {{ t=$(mktemp -d) && "
                     f"git -C {shlex.quote(repo)} archive {sha} | tar -x -C $t && echo {sha} > $t/TREE_SHA && "
                     f"rsync -a -c --delete -e 'ssh -o BatchMode=yes' ${{b:+--copy-dest=${{b%/TREE_SHA}}}} $t/ {ssh}:{rd}.part/ && "
                     f"ssh -o BatchMode=yes {ssh} {shlex.quote(f'mv -T {part} {q}')}; rm -rf $t; }}")
    if any(f.startswith(DATA + "/") for f in files):
        # an input is useless without its MSAs, so bench data ships whole (~250 MB) and never overwrites a file
        need = shlex.quote(next(f for f in sorted(files) if f.startswith(DATA + "/")).replace("~", "$HOME", 1))
        steps.append(f"for d in {' '.join(donors)}; do if ssh -o BatchMode=yes -o ConnectTimeout=10 $d test -s {need}; then "
                     f"ssh -o BatchMode=yes $d 'tar -C ~/spd-data -c .' | ssh -o BatchMode=yes {ssh} "
                     f"'mkdir -p ~/spd-data && tar -C ~/spd-data -x --skip-old-files'; break; fi; done")
        files = [f for f in files if not f.startswith(DATA + "/")]
    for f in sorted(files):
        rf = f.replace("~", "$HOME", 1) if f.startswith("~/") else f
        get = shlex.quote(f'f="{rf}"; [ -s "$f" ] && [ -f "$f" ] && [ $(stat -Lc %s "$f") -le 1048576 ] && cat "$f"')
        put = shlex.quote(f'f="{rf}"; [ -e "$f" ] || {{ mkdir -p "$(dirname "$f")"; cat > "$f.part" && mv "$f.part" "$f"; }}')
        steps.append(f"for d in {' '.join(donors)}; do t=$(mktemp); if ssh -o BatchMode=yes -o ConnectTimeout=10 $d {get} > $t; "
                     f"then ssh -o BatchMode=yes {ssh} {put} < $t; rm -f $t; break; fi; rm -f $t; done")
    sh = "; ".join(steps)
    lock = f"{lockdir}/provision.{ssh.split('@')[-1]}.lock"
    subprocess.Popen(["setsid", "flock", "-n", lock, "timeout", "3000", "bash", "-c", sh],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wake(root, row, why, log):
    nb = root / "state/notbefore" / row
    if nb.exists():
        (root / "state/notbefore/cleared").mkdir(parents=True, exist_ok=True)
        nb.rename(root / "state/notbefore/cleared" / f"{row}.spd-feed-{int(time.time())}")
    log(f"woke {row}: {why}")


def write_json(f, obj):
    """Write through a temp file and a rename: the tick kills a slow pass at 900 s, and a kill inside write_text left
    a page-aligned half ledger that crashed every later pass until someone repaired it (10-23, 20:05Z-21:2xZ)."""
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, f)


def steal(cs, ledger, st, P, stamp, dry, log):
    """Move BH jobs that waited STEAL_S on a busy card to a free card, oldest first, one per card. The free card may be
    on the job's own box: 10-14 two OF3 deadline A/Bs sat 2.5 h behind qb1's gapboard column and a release leg while
    qb1 cards 0 and 3 went free and took jobs placed after them."""
    by = {c["name"]: c for c in cs}
    for b in [n for n in st if P[n]["kind"] == "bh"]:
        for card in list(st[b].get("free", [])):
            cand = [(j["age"], n, name) for n in st if P[n]["kind"] == "bh"
                    if n != b for name, j in st[n]["jobs"].items()
                    if j["state"] == "queued" and j.get("age", 0) >= STEAL_S and name in by
                    and runs(P[b], by[name])
                    and all(st[b]["exists"].get(x) for x in by[name]["requires"])]
            if not cand:
                break
            _, n, name = max(cand); c = by[name]
            st[n]["jobs"][name]["state"] = "moving"
            log(f"steal {c['key']}: waited {max(cand)[0] // 60} min on {n} -> {b} card {card}")
            if dry:
                continue
            try:
                if name not in remote(P[n]["ssh"], WITHDRAW, {"names": [name]}):
                    continue
                remote(P[b]["ssh"], PLACE, {"kind": "bh", "reset": P[b].get("reset", ""), "jobs": [{"name": name, "body": c["body"], "cap": c["cap"],
                                                                   "cards": [card], "lock": P[b]["lock"], "outs": outs(c)}]})
            except Exception as e:   # withdrawn but not placed: the next pass finds the cell lost and places it again
                log(f"steal {c['key']} failed: {e}")
                continue
            st[n]["queued"] -= 1; st[b]["queued"] += 1
            ledger[c["key"]] = dict(ledger.get(c["key"], {}), host=b, state="queued", placed=stamp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(D), help="coworker dir: ALLOC.json, grids, ledger, QUARANTINE.md, notbefore")
    ap.add_argument("--depth", type=float, default=2.0, help="queued jobs per healthy chip to keep")
    ap.add_argument("--repo", default="/home/moritz/tt-bio", help="tt-bio clone that trees missing on a pool are archived from")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    R = Path(a.root); S = R / "state/spd/feed"; S.mkdir(parents=True, exist_ok=True)
    ledger_f = S / "ledger.json"
    ledger = json.loads(ledger_f.read_text()) if ledger_f.exists() else {}
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logf = open(S / "feed.log", "a")

    def log(m):
        logf.write(f"{stamp} {m}\n"); print(m)

    cs = cells(R / "state/spd/grids", log, lead_rows(R),
               {f.name for f in (R / "state/concluded").iterdir()} if (R / "state/concluded").is_dir() else set())
    P = pools(R)
    # Galaxies rotate: a pin to one SPD no longer holds lets the cell run on any held pool of its arch. A dark BH box
    # keeps its results: an experiment with a cell done there waits for it, so its A/B arms stay on one box.
    dark = {n for n in BH_POOLS if n not in P}
    held = {k.rsplit("/", 1)[0] for k, L in ledger.items() if L.get("state") == "done" and L.get("host") in dark}
    for c in cs:
        if c["hosts"] and not set(c["hosts"]) & set(P) and not c["pinned"] and c["key"].rsplit("/", 1)[0] not in held:
            c["hosts"] = []
    roots = {p: tree_root(p, a.repo) for c in cs for p in c["requires"]}
    paths = sorted({p for c in cs for p in c["done"] + c["requires"]} | {r[0] for r in roots.values() if r})
    outdirs = {c["name"]: outs(c) for c in cs}
    st = {}
    now = time.time()
    concluded = [f.name for f in (R / "state/concluded").iterdir() if now - f.stat().st_mtime < 7 * 86400] \
        if (R / "state/concluded").is_dir() else []
    for name, p in P.items():
        try:
            st[name] = remote(p["ssh"], PROBE, {"kind": p["kind"], "paths": [x for x in paths], "cards": p.get("cards", []), "book": BOOK,
                                             "concluded": concluded, "lock": p.get("lock", ""), "lend": p.get("lend", False), "sweep": not a.dry_run, "outdirs": outdirs})
        except Exception as e:  # an unreachable box is skipped, never fatal
            log(f"{name}: probe failed: {e}")
    # a box whose probe failed hides its done files, so its ledger counts as for a dark box: 10-25 13:44Z a qb2 probe timeout
    # turned 825 qb2-done cells undone and feed queued 29 of them, graded days before, on qb1's four chips
    dark |= set(P) - set(st)
    # 1. account every ledger entry: done, pending, or a failed attempt
    for c in cs:
        L = ledger.get(c["key"])
        hosts = [n for n in st if P[n]["arch"] == c["arch"]]
        c["is_done"] = (any(all(st[n]["exists"].get(x) for x in c["done"]) for n in hosts)
                        or bool(L) and L.get("state") == "done" and L.get("host") in dark and L.get("hash") == c["hash"])
        if not L:
            continue
        if L.get("hash") != c["hash"]:          # the row changed the cell: fresh start
            L.update(hash=c["hash"], fails=[], quarantined=False)
        j = st.get(L.get("host"), {}).get("jobs", {}).get(L.get("name"))
        if c["is_done"]:
            if j and j["state"] == "ended" and str(j["rc"]).split()[:1] == ["0"]:
                L["dur"] = j["dur"]
            L.update(state="done", quarantined=False)
        elif j and j["state"] in ("queued", "running"):
            L["state"] = j["state"]
        elif j and j["state"] == "ended" and j["at"] not in [f["at"] for f in L["fails"]]:
            L["fails"].append({"at": j["at"], "dur": j["dur"], "rc": j["rc"], "host": L["host"], "tail": j["tail"][-1200:]})
            L["state"] = "failed"
            if "fastfail2" in str(j["rc"]) and not a.dry_run:
                with open(R / "fleet.log", "a") as fl:
                    fl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} spd feed: {L['host']} {L['name']} was the "
                             f"second device-init failure in a row on its card: {j['rc']}\n")
            # a dirty card's failure is not the cell's, nor is a waiter's that never took a card (flockbusy)
            own = [f for f in L["fails"] if "fastfail2" not in str(f["rc"]) and f["rc"] != "flockbusy"]
            fast = sum(1 for f in own if f["dur"] < FAST_S)
            if fast >= 2 or len(own) >= MAX_FAILS:
                L["quarantined"] = True
                log(f"QUARANTINE {c['key']} ({fast} fast of {len(L['fails'])} fails), last {L['name']} on {L['host']}")
                if not a.dry_run:
                    wake(R, c["row"], f"{c['key']} quarantined", log)
        elif L.get("state") in ("queued", "running") and L.get("host") in st:
            L["state"] = "lost"                  # vanished from queue/running without an end: place again
        elif L.get("state") in ("queued", "running") and L.get("host") not in P:
            L["state"] = "lost"                  # its box went dark or back to JapanFold: 10-14 qb2 held 11 cells 1.5 h
    keys = {c["key"] for c in cs}           # a cell its row dropped from the grid while its box was dark: nobody's now
    for k, L in ledger.items():
        if k not in keys and L.get("state") in ("queued", "running") and L.get("host") not in P:
            L["state"] = "lost"
    # a box back from dark still queues the cells placed elsewhere or dropped meanwhile: withdraw those before they run
    owner = {L["name"]: L["host"] if L.get("state") in ("queued", "running") else None for L in ledger.values() if L.get("name")}
    for n, s in st.items():
        twins = [m for m, j in s.get("jobs", {}).items() if j["state"] == "queued" and owner.get(m, n) != n]
        if twins and not a.dry_run:
            try:
                gone = remote(P[n]["ssh"], WITHDRAW, {"names": twins, "kind": P[n]["kind"]})
            except Exception as e:
                log(f"{n}: twin withdraw failed: {e}"); continue
            for m in gone:
                s["jobs"].pop(m); s["queued"] -= 1
            log(f"{n}: withdrew {len(gone)} twin(s) placed elsewhere while it was dark")
    # a Galaxy that has offered no chip for STEAL_S (lending switched off, every chip recalled) strands its queue: withdraw
    # its feed jobs and mark them lost, so step 2 places them on a pool with room (10-24: .107 lent 0 chips from 01:10Z
    # with 92 cells queued while .103 sat lent with 20 healthy)
    zf = S / "zero.json"
    zero = json.loads(zf.read_text()) if zf.exists() else {}
    zero = {n: zero.get(n, now) for n, s in st.items() if P[n]["kind"] == "galaxy" and s["healthy"] <= 0}
    if not a.dry_run:
        write_json(zf, zero)
    for n in [n for n in zero if now - zero[n] >= STEAL_S]:
        stuck = [m for m, j in st[n]["jobs"].items() if j["state"] == "queued"]
        if not stuck or a.dry_run:
            continue
        try:
            gone = remote(P[n]["ssh"], WITHDRAW, {"names": stuck, "kind": "galaxy"})
        except Exception as e:
            log(f"{n}: stranded withdraw failed: {e}"); continue
        for m in gone:
            st[n]["jobs"].pop(m); st[n]["queued"] -= 1
        for L in ledger.values():
            if L.get("host") == n and L.get("name") in gone:
                L["state"] = "lost"
        log(f"{n}: no chip for {int(now - zero[n]) // 60} min, withdrew {len(gone)} queued job(s) to place elsewhere")
    if not a.dry_run:
        arc(R, st, P, log)
    live = {n for s in st.values() for n in s.get("live", [])}
    steal(cs, ledger, st, P, stamp, a.dry_run, log)
    # 2. fill each pool up to depth x healthy, cells in priority order, the pool with most room first. A cell's room
    # counts only queued work it does not jump: pool.sh starts the lowest prio digit first, so prio-6 jobs hand-queued
    # past the depth must not lock prio-3 cells out (10-15: 211 queued on .107, every grid cell refused for hours).
    # Depth counts expected chip time: a job whose experiment's median done run is under TICK_S weighs that fraction
    # (at least 0.1), so the queue covers two passes of short cells too. A BH waiter leaves a card to a lower digit (hi()),
    # so a BH box counts its queue by digit too: counted at prio 0, five prio-4/8 waiters on qb2 locked out every
    # prio-3 Protenix BH cell (10-21).
    exp = lambda k: k.rsplit("/", 1)[0]
    durs = {}
    for k, L in ledger.items():
        if L.get("dur") is not None:
            durs.setdefault(exp(k), []).append(L["dur"])
    est = {e: sorted(v)[len(v) // 2] for e, v in durs.items()}
    weight = lambda k: min(1.0, max(0.1, est[exp(k)] / TICK_S)) if exp(k) in est else 1.0
    key_of = {L["name"]: k for k, L in ledger.items() if L.get("name")}
    slot = lambda n, p: p if st[n].get("qprio") else 0
    qp = {n: list(s.get("qprio") or [s["queued"]] + [0] * 8) for n, s in st.items()}
    for n, s in st.items():
        for m, j in s.get("jobs", {}).items():
            if j["state"] == "queued" and m in key_of and m[0].isdigit():
                qp[n][slot(n, int(m[0]))] -= 1 - weight(key_of[m])
    room = lambda n, p: 0 if st[n].get("paused") else int(a.depth * st[n]["healthy"]) - sum(qp[n][:p + 1])
    place = {n: [] for n in st}
    for c in cs:
        L = ledger.get(c["key"], {})
        if c["is_done"] or L.get("quarantined") or L.get("state") in ("queued", "running") or live & set(c["also"]):
            continue
        p = int(c["name"][0])
        ok = [n for n in st if runs(P[n], c) and room(n, p) > 0 and all(st[n]["exists"].get(x) for x in c["requires"])]
        if not ok:
            continue
        n = max(ok, key=lambda n: room(n, p))
        job = {"name": c["name"], "body": c["body"], "cap": c["cap"]}
        if P[n]["kind"] == "bh":
            job.update(cards=st[n]["cards"], lock=P[n]["lock"], outs=outs(c))
        place[n].append(job); qp[n][slot(n, p)] += weight(c["key"])
        ledger[c["key"]] = dict(L, host=n, name=c["name"], hash=c["hash"], state="queued", placed=stamp,
                                fails=L.get("fails", []), quarantined=False, row=c["row"])
    # 3. trees and small files a pending cell needs and a pool of its arch lacks: ship them so a later pass places there
    # a cell whose requires no pool of its arch has and feed cannot ship waits forever: say so every pass
    ship, copy, blocked = {n: {} for n in st}, {n: set() for n in st}, {}
    for c in cs:
        L = ledger.get(c["key"], {})
        if c["is_done"] or L.get("quarantined") or L.get("state") in ("queued", "running"):
            continue
        hosts, stuck, fine = [n for n in st if runs(P[n], c)], set(), False
        for n in hosts:
            lack = set()
            for x in c["requires"]:
                r = roots[x]
                if st[n]["exists"].get(x):
                    continue
                if r and not st[n]["exists"].get(r[0]):
                    ship[n][r[0]] = r[1]
                elif not r and P[n]["kind"] == "galaxy" and x.endswith(SMALL) and "/trees/" not in x:
                    # a file under trees/ is part of a tree feed cannot name: a lone copy is a stub that satisfies `requires`
                    copy[n].add(x)
                else:
                    lack.add(x)
            fine |= not lack
            stuck |= lack
        if hosts and not fine:
            blocked.setdefault((c["row"], " ".join(sorted(stuck))), []).append(c["name"])
    for (row, paths), names in sorted(blocked.items()):
        log(f"blocked {row}: {len(names)} cell(s) ({min(names)}...), no host has {paths}")
    for n in st:
        if ship[n] or copy[n]:
            log(f"{n}: shipping {len(ship[n])} tree(s), {len(copy[n])} file(s): {' '.join(sorted(ship[n]) + sorted(copy[n]))}")
            if not a.dry_run:
                provision(P[n]["ssh"], ship[n], copy[n], [d for d in donors(R) if d != P[n]["ssh"]], a.repo, S)
    qnames = {L["name"] for L in ledger.values() if L.get("quarantined")}
    for n, jobs in place.items():
        s = st[n]
        if s.get("unbooked"):
            log(f"{n}: dropped the booking of concluded {', '.join(s['unbooked'])}")
        log(f"{n}: healthy {s['healthy']} queued {s['queued']} -> +{len(jobs)} feed jobs; swept {s.get('swept', 0)}")
        if a.dry_run:
            continue
        unq = [q for q in qnames if s["jobs"].get(q, {}).get("state") == "queued"] if P[n]["kind"] == "galaxy" else []
        if jobs or unq:
            try:
                remote(P[n]["ssh"], PLACE, {"kind": P[n]["kind"], "jobs": jobs, "unqueue": unq,
                                            "reset": P[n].get("reset", "")})
            except Exception as e:  # the next pass finds these cells lost and places them again
                log(f"{n}: place failed: {e}")
    if a.dry_run:
        log(f"dry run: cells {len(cs)}, done {sum(c['is_done'] for c in cs)}, twin queued/running {sum(bool(live & set(c['also'])) for c in cs)}")
        return 0
    write_json(ledger_f, ledger)
    q = [(k, L) for k, L in ledger.items() if L.get("quarantined") and k in keys]
    lines = ["# SPD quarantine (perf/spd/feed.py; a cell returns once its row changes the grid entry)", "",
             f"Updated {stamp}. {len(q)} cell(s).", ""]
    for k, L in q:
        f = L["fails"][-1]
        lines += [f"## {k}", f"row {L['row']}, job {L['name']} on {f['host']}, {len(L['fails'])} fails, "
                  f"durations {[x['dur'] for x in L['fails']]} s, last rc {f['rc']}", "```", f["tail"].strip()[-1200:], "```", ""]
    (R / "state/spd/QUARANTINE.md").write_text("\n".join(lines))
    log(f"cells {len(cs)}: done {sum(c['is_done'] for c in cs)}, twin queued/running {sum(bool(live & set(c['also'])) for c in cs)}")
    tot = {s: sum(1 for L in ledger.values() if L.get("state") == s) for s in ("queued", "running", "done", "failed")}
    log(f"cells {len(cs)}, ledger {tot}, quarantined {len(q)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
