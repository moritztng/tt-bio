#!/usr/bin/env python3
"""Keep every SPD pool fed from experiment grids, so chips never wait for an agent turn.

Each row declares what it wants measured in state/spd/grids/<row>.yaml:

    row: spd-gap-of3
    tier: red            # red | amber | fast | fill: queue order across rows (GAP RED cells first)
    experiments:
      - name: wsd
        arch: wh         # wh = held Galaxies, bh = qb1/qb2
        prio: 4          # pool prio 3-8
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
2 x healthy chips of queued prio 0-8 work, cells in tier order, rows round-robin inside a tier.

A cell whose job ended twice within 60 s without its done files, or three times in all, is quarantined in
state/spd/QUARANTINE.md with its log tail, and its row is woken. The job name carries a hash of the rendered command,
so the cell comes back by itself once the row changes its grid entry.

Galaxies: jobs go to ~/spd/pool/queue/ and pool.sh starts them. qb1/qb2 have no pool: feed starts each job detached as
a waiter that takes the first usable card of its box to come free (its CHIPS.md flock free, no release leg on it);
the job itself gets a private LOCK so run_chip.sh does not wait on its own card. Waiters queue for cards through one
dispatch flock per box, so they start in about the order they were placed. A BH job still waiting after STEAL_S moves
to a free card on another BH box its cell allows.
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
TIERS = {"red": 0, "amber": 1, "fast": 2, "fill": 3}
FAST_S = 60
INIT_FAIL = "failed to initialize FW|ARC core .* failed to start|Failed to set initial power state"
MAX_FAILS = 3
STEAL_S = 900   # a BH job still waiting this long moves to a free card on another BH box its cell allows
# BH boxes: SPD cards and their flocks (state/spd/CHIPS.md). A chip kept off the feed has ~/spd/locks/chip<N>.BAD on its
# host (qb2 chip 0 since the 2026-10-14 slot swap: marginal at 1350 MHz, and feed trees predate the AICLK cap).
# `reset` resets a card after a capped job (see run_cmd). qb1 only: each p150a is its own board, while a qb2 p300c
# board carries two chips and a reset there would kill the other chip's job.
BH_POOLS = {
    "qb1": {"ssh": "qb1", "cards": [0, 1, 2, 3], "lock": "/home/ttuser/spd_qb1_card{c}.lock",
            "reset": "/home/ttuser/.local/bin/tt-smi -r"},
    "qb2": {"ssh": "qb2", "cards": [0, 1, 2, 3], "lock": "/home/ttuser/spd_qb2_chip{c}.lock"},
}

# pgrep pattern for a release-gate leg on card %s, running or waiting for its flock. A leg that runs a perf script through
# runpy (bc2_boundary, parity_*) names the gate only in its RELEASE_GATE_RUN_DIR env, so the pattern keys on that too.
# A multi-card take (cfd2/cfd3 window scripts) raises a `release_gate` sentinel per card and waits until every card is
# free, so the cards that free first idle behind it (10-16 qb2: chips 0, 2 and 3 for up to 6,000 s). A sentinel is a
# placeholder process named `<who> spd_qb<N>_<chip|card><c>.lock <kind> <tag> ...`; its take is every card whose sentinel
# carries the same who/kind/tag. The take cannot start before the feed jobs on its other cards end, so a waiter backfills
# a sentinel's card with a job whose worst-case end (start + cap + 600 s of grace and reset, written to <job>.until) comes
# before the latest .until on those cards; the take never starts later for it. A sentinel may name its own bound with
# `until=<epoch>`, which then wins. A one-card sentinel (gapboard) and any real gate leg (not named like a sentinel) block.
# A waiter that is only too long for such a window hands the dispatch flock on each poll, so a shorter job behind it
# can take the card. Takers can size their wait from ~/spd/feed/jobs/*.until (card in the matching .card).
GATE_LEG = "(spd_qb1_card|spd_qb2_chip|TT_VISIBLE_DEVICES=)%s[. ].*(RELEASE_GATE_RUN_DIR|release_gate|gate_fanout|perf_ab)"

# Runs on the pool host: report state, existence of paths, and sweep dead entries. Payload is appended as JSON.
PROBE = r'''
import glob, json, os, shutil, sys, time
A = json.loads(sys.stdin.readline())
H = os.path.expanduser("~"); now = time.time(); out = {"exists": {}, "jobs": {}}
def ok(p):
    p = os.path.expanduser(p)
    if any(c in p for c in "*?["):
        return any(ok(q) for q in glob.glob(p))
    # a result file must be non-empty; a marker (`touch $O/done`, `touch $O/$arm.done`) is empty by design
    b = os.path.basename(p)
    return (os.path.isfile(p) and (os.path.getsize(p) > 0 or "." not in b or b.endswith(".done"))
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
    bad = len(glob.glob(H + "/spd/locks/chip*.BAD"))
    out["healthy"] = len([n for n in os.listdir("/dev/tenstorrent") if n.isdigit()]) - bad
    q = [os.path.basename(f) for f in glob.glob(P + "/queue/*.sh")]
    out["queued"] = sum(1 for n in q if n[0] in "012345678")
    out["qprio"] = [sum(1 for n in q if n[0] == str(d)) for d in range(9)]
    out["paused"] = os.path.exists(P + "/PAUSE") or os.path.exists(P + "/RETURNED")
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
    out["healthy"] = len(cards); out["cards"] = cards; queued = 0; held = set()
    for sh in glob.glob(J + "/*.sh"):
        n = os.path.basename(sh); card = int(open(sh + ".card").read()) if os.path.exists(sh + ".card") else -1
        if os.path.exists(sh + ".end"):
            out["jobs"][n] = ended(sh, sh + ".end", sh + ".start" if os.path.exists(sh + ".start") else sh)
        elif not os.path.exists(sh + ".pid") or not os.path.exists("/proc/" + open(sh + ".pid").read().strip()):
            open(sh + ".end", "w").write("lost\n"); out["jobs"][n] = ended(sh, sh + ".end", sh)
        elif os.path.exists(sh + ".start"):
            out["jobs"][n] = {"state": "running"}; held.add(card)
        else:
            out["jobs"][n] = {"state": "queued", "age": int(now - os.path.getmtime(sh))}; queued += 1
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
NODE = ('node() { for s in "${TT_SYSFS:-/sys/class/tenstorrent}"/*; do echo "$(basename "$(readlink -f "$s/device")") '
        '${s##*!}"; done | sort | sed -n "$(($1 + 1))s/.* //p"; }; '
        'dead() { local n=$(node $1); [ -n "$n" ] && cat "${TT_SYSFS:-/sys/class/tenstorrent}/tenstorrent!$n/tt_aiclk" '
        '"${TT_SYSFS:-/sys/class/tenstorrent}/tenstorrent!$n/tt_heartbeat" 2>/dev/null | grep -qx 4294967295; }; '
        'take() { pgrep -af "[.]lock (release_gate|perf_ab) " | awk -v a="$1" -v r="$2" -v c="$c" -v J="$J" \'function rd(p, v) { if ((getline v < p) <= 0) v = ""; close(p); return v } $2 == a && $3 ~ /^spd_qb[0-9]_(chip|card)[0-9]+[.]lock$/ { l = $3; $1 = $2 = $3 = ""; sub(/^ +/, ""); if ($0 != r) next; sub(/[.]lock$/, "", l); sub(/.*[a-z]/, "", l); if (l != c) k[l] } END { m = 0; while (("ls " J " 2>/dev/null") | getline f) { j = J "/" f; if (f ~ /[.]sh$/ && (rd(j ".card") in k) && system("[ -e \\"" j ".start\\" ] && [ ! -e \\"" j ".end\\" ]") == 0 && rd(j ".until") + 0 > m) m = rd(j ".until") + 0 } print m }\'; }; '
        'gate() { local m p a l r e x=1 n=$(($(date +%s) + $1 + 600)); m=$(pgrep -af "$G") || return 1; '
        'while read -r p a l r; do [[ $l =~ ^spd_qb[0-9]_(chip|card)[0-9]+[.]lock$ ]] || return 0; '
        'e=$(grep -o "until=[0-9]*" <<<"$r" | cut -d= -f2); [ -n "$e" ] || e=$(take "$a" "$r"); [ $n -gt "${e:-0}" ] && x=0; '
        'done <<<"$m"; [ $x = 1 ] || W=1; return $x; }; ')


def run_cmd(grace=120):
    return ('S=$(date +%s); CHIP=$2 TT_VISIBLE_DEVICES=$2 LOCK="$4" WAIT=60 timeout -s TERM $(($5 + G)) '
            'sh -c \'echo $$ > "$1.pg"; exec timeout -s INT "$2" bash "$1"\' _ "$1" "$5" > "$1.log" 2>&1; rc=$?; '
            'M="$(dirname "$1")/fastfail.$2"; if [ $rc = 0 ]; then rm -f "$M"; '
            'elif [ $rc != 124 ] && [ $rc != 127 ] && [ $(($(date +%s) - S)) -lt FAST ] && grep -qE "INIT" "$1.log"; then '
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
        for x in (".end", ".start", ".pid", ".log", ".pg", ".until"):
            if os.path.exists(sh + x): os.remove(sh + x)
        open(sh, "w").write(j["body"]); open(sh + ".card", "w").write("-1")
        # Take the box's dispatch flock in turn, then the first card that is not BAD or dead-ARC, runs no release-gate leg and no
        # gapboard sentinel (the scoreboard measuring main; checked again once its flock is held) and whose flock is
        # free. 10-15: waiters bound to one card queued on busy cards 2 and 3 while qb1 card 0 sat free 15+ min.
        w = (NODE + 'exec 8>"$(dirname "$1")/.dispatch"; flock -w 86400 8 || { echo flockbusy > "$1.end"; exit 3; }; '
             'J="$(dirname "$1")"; D=$((SECONDS + 86400)); while :; do W=; for c in $2; do [ -e ~/spd/locks/chip$c.BAD ] && continue; dead $c && continue; '
             'G="' + GATE_LEG % "$c" + '"; '
             'gate "$5" && continue; exec 9>"${3//@/$c}"; '
             'flock -n 9 && ! gate "$5" && break 2; exec 9>&-; done; '
             '[ $SECONDS -lt $D ] || { echo flockbusy > "$1.end"; exit 3; }; '
             'if [ -n "$W" ]; then flock -u 8; sleep "${POLL:-20}"; flock -w 86400 8 || { echo flockbusy > "$1.end"; exit 3; }; '
             'else sleep "${POLL:-20}"; fi; done; '
             'exec 8>&-; echo $c > "$1.card"; touch "$1.start"; echo $(($(date +%s) + $5 + 600)) > "$1.until"; set -- "$1" $c "$3" "$(dirname "$1")/nolock.$c" "$5" "$6"; '
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
    f.write_text(json.dumps(H, indent=1))


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


def inputs(cmd):
    optional = set(re.findall(r"\[ -[efsd] (\S+) \] &&", cmd))      # `[ -f X ] && . X` runs without X
    guarded = re.findall(r"\[ -[efsd] ((?:~|\$HOME|/home)/\S+) \] \|\|", cmd)   # `[ -s X ] || exit 1` cannot
    return {re.sub(r"^\$HOME/", "~/", p) for p in INPUT.findall(cmd) + guarded
            if p not in optional and "$" not in p.replace("$HOME/", "", 1)}


def cells(grid_dir, log=print):
    """All cells of all active experiments, ordered by tier, rows round-robin within a tier."""
    by_tier = {}
    for f in sorted(Path(grid_dir).glob("*.yaml")):
        g = yaml.safe_load(f.read_text()) or {}
        row, tier = g["row"], TIERS[g.get("tier", "amber")]
        for e in g.get("experiments", []):
            if not e.get("active", True):
                continue
            prio = int(e.get("prio", 5))
            if not 3 <= prio <= 8:  # one bad experiment must not stop every row's feed (10-15 19:02-19:20Z)
                log(f"skip {f.name}:{e['name']}: prio {prio} outside 3-8")
                continue
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
                out.append({"key": key, "row": row, "hash": h, "arch": e.get("arch", "wh"),
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
    p = {ip: {"ssh": f"ubuntu@172.16.102.{ip}", "kind": "galaxy", "arch": "wh"}
         for ip, g in a.items() if g.get("owner") == "spd" and g.get("phase") == "lent"}
    for name, b in BH_POOLS.items():
        p[name] = dict(b, kind="bh", arch="bh")
    return p


SHA_DIR = re.compile(r"(?:tree-|t)?([0-9a-f]{7,40})")


def tree_root(path, repo, _seen={}):
    """(tree dir, commit) if `path` lies in a tt-bio tree named by its commit (trees/<sha>, tree-<sha>, t<sha>)."""
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
        d = d.replace("~", "$HOME", 1) if d.startswith("~/") else d
        q, part = f'"{d}"', f'"{d}.part"'
        remote_cmd = (f"set -e; [ -e {q} ] && exit 0; rm -rf {part}; mkdir -p {part}; tar -x -C {part}; "
                      f"echo {sha} > {part}/TREE_SHA; mv -T {part} {q}")
        steps.append(f"git -C {shlex.quote(repo)} archive {sha} | ssh -o BatchMode=yes {ssh} {shlex.quote(remote_cmd)}")
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
                    and (not by[name]["hosts"] or b in by[name]["hosts"])
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
                                                                   "cards": [card], "lock": P[b]["lock"]}]})
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

    cs = cells(R / "state/spd/grids", log)
    P = pools(R)
    for c in cs:  # Galaxies rotate: a pin to one SPD no longer holds lets the cell run on any held pool of its arch
        if c["hosts"] and not set(c["hosts"]) & set(P):
            c["hosts"] = []
    roots = {p: tree_root(p, a.repo) for c in cs for p in c["requires"]}
    paths = sorted({p for c in cs for p in c["done"] + c["requires"]} | {r[0] for r in roots.values() if r})
    outdirs = {c["name"]: sorted({os.path.dirname(x) for x in c["done"]}) for c in cs}
    st = {}
    for name, p in P.items():
        try:
            st[name] = remote(p["ssh"], PROBE, {"kind": p["kind"], "paths": [x for x in paths], "cards": p.get("cards", []),
                                             "lock": p.get("lock", ""), "sweep": not a.dry_run, "outdirs": outdirs})
        except Exception as e:  # an unreachable box is skipped, never fatal
            log(f"{name}: probe failed: {e}")
    # 1. account every ledger entry: done, pending, or a failed attempt
    for c in cs:
        L = ledger.get(c["key"])
        hosts = [n for n in st if P[n]["arch"] == c["arch"]]
        c["is_done"] = any(all(st[n]["exists"].get(x) for x in c["done"]) for n in hosts)
        if not L:
            continue
        if L.get("hash") != c["hash"]:          # the row changed the cell: fresh start
            L.update(hash=c["hash"], fails=[], quarantined=False)
        j = st.get(L.get("host"), {}).get("jobs", {}).get(L.get("name"))
        if c["is_done"]:
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
        elif L.get("state") in ("queued", "running") and (L.get("host") not in P or (R / f"state/{L['host']}-offline").exists()):
            L["state"] = "lost"                  # its box went dark or back to JapanFold: 10-14 qb2 held 11 cells 1.5 h
    # a box back from dark still queues the cells placed elsewhere meanwhile: withdraw those twins before they run
    owner = {L["name"]: L["host"] for L in ledger.values() if L.get("name")}
    for n, s in st.items():
        twins = [m for m, j in s.get("jobs", {}).items() if j["state"] == "queued" and owner.get(m, n) != n]
        if twins and not a.dry_run:
            try:
                gone = remote(P[n]["ssh"], WITHDRAW, {"names": twins})
            except Exception as e:
                log(f"{n}: twin withdraw failed: {e}"); continue
            for m in gone:
                s["jobs"].pop(m); s["queued"] -= 1
            log(f"{n}: withdrew {len(gone)} twin(s) placed elsewhere while it was dark")
    if not a.dry_run:
        arc(R, st, P, log)
    live = {n for s in st.values() for n in s.get("live", [])}
    steal(cs, ledger, st, P, stamp, a.dry_run, log)
    # 2. fill each pool up to depth x healthy, cells in priority order, the pool with most room first. A cell's room
    # counts only queued work it does not jump: pool.sh starts the lowest prio digit first, so prio-6 jobs hand-queued
    # past the depth must not lock prio-3 cells out (10-15: 211 queued on .107, every grid cell refused for hours).
    qp = {n: list(s.get("qprio") or [s["queued"]] + [0] * 8) for n, s in st.items()}
    room = lambda n, p: 0 if st[n].get("paused") else int(a.depth * st[n]["healthy"]) - sum(qp[n][:p + 1])
    place = {n: [] for n in st}
    for c in cs:
        L = ledger.get(c["key"], {})
        if c["is_done"] or L.get("quarantined") or L.get("state") in ("queued", "running") or live & set(c["also"]):
            continue
        p = int(c["name"][0])
        ok = [n for n in st if P[n]["arch"] == c["arch"] and room(n, p) > 0 and (not c["hosts"] or n in c["hosts"])
              and all(st[n]["exists"].get(x) for x in c["requires"])]
        if not ok:
            continue
        n = max(ok, key=lambda n: room(n, p))
        job = {"name": c["name"], "body": c["body"], "cap": c["cap"]}
        if P[n]["kind"] == "bh":
            job.update(cards=st[n]["cards"], lock=P[n]["lock"])
        place[n].append(job); qp[n][p] += 1
        ledger[c["key"]] = dict(L, host=n, name=c["name"], hash=c["hash"], state="queued", placed=stamp,
                                fails=L.get("fails", []), quarantined=False, row=c["row"])
    # 3. trees and small files a pending cell needs and a pool of its arch lacks: ship them so a later pass places there
    ship, copy = {n: {} for n in st}, {n: set() for n in st}
    for c in cs:
        L = ledger.get(c["key"], {})
        if c["is_done"] or L.get("quarantined") or L.get("state") in ("queued", "running"):
            continue
        for n in st:
            if P[n]["arch"] == c["arch"] and (not c["hosts"] or n in c["hosts"]):
                for x in c["requires"]:
                    r = roots[x]
                    if st[n]["exists"].get(x):
                        continue
                    if r and not st[n]["exists"].get(r[0]):
                        ship[n][r[0]] = r[1]
                    elif not r and P[n]["kind"] == "galaxy" and x.endswith(SMALL):
                        copy[n].add(x)
    for n in st:
        if ship[n] or copy[n]:
            log(f"{n}: shipping {len(ship[n])} tree(s), {len(copy[n])} file(s): {' '.join(sorted(ship[n]) + sorted(copy[n]))}")
            if not a.dry_run:
                provision(P[n]["ssh"], ship[n], copy[n], [d for d in donors(R) if d != P[n]["ssh"]], a.repo, S)
    qnames = {L["name"] for L in ledger.values() if L.get("quarantined")}
    for n, jobs in place.items():
        s = st[n]
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
    ledger_f.write_text(json.dumps(ledger, indent=1))
    keys = {c["key"] for c in cs}           # a cell its row dropped from the grid is no longer anyone's to fix
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
