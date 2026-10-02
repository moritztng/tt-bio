#!/bin/bash
# rel0121: the v0.12.0 legs that the qb2 03:34Z hang killed, re-run at a46281e91 on one card. Copy of rel012/chain.sh.
# Usage: chain.sh <card> <arm-list-name>. Two chains run in parallel (orchestrator order 21:43Z),
# so each waits only on its OWN card being free of foreign holders. Interpreter: the release venv
# the gated wheel was installed into (ttnn 0.68.0 = the pyproject pin). rc 0/1 is a verdict and the
# chain continues so every arm reports; any other rc is recorded and the chain also continues.
CARD=${1:?card}; SET=${2:?set}
CARDS=${CARDS:-$CARD}  # every card an arm folds on; the env sampler logs each one
T=/home/ttuser/scratch/rel0121/tree
L=/home/ttuser/scratch/rel0121
PY=/home/ttuser/scratch/rel012/relvenv/bin/python3
G=$(git -C /home/ttuser/scratch/rel0121/tree rev-parse 265cd26b8)
BT=/home/ttuser/scratch/rel0121/bcptree
LOG=$L/CHAIN_c$CARD.log
cd "$T" || exit 1
export PYTHONPATH="$T"
export ESM_ROOT=/home/ttuser/esm
export AF2IG_PARAMS=/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/.coworker/dockq-venv/bin/python
export TT_BIO_LEASE_HOLDER=worker:rel0121
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
clock() { cat "/sys/class/tenstorrent/tenstorrent!$1/tt_aiclk" 2>/dev/null; }

foreign() {  # device holders that are not descendants of this chain
    CHAIN_PID=$$ "$PY" - <<"PYX"
import glob, os
chain = int(os.environ["CHAIN_PID"])
def anc(pid):
    s = set()
    while pid > 1 and pid not in s:
        s.add(pid)
        try: pid = int(open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[1])
        except Exception: break
    return s
out = {}
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try: t = os.readlink(fd)
    except OSError: continue
    if not t.startswith("/dev/tenstorrent/"): continue
    pid = int(fd.split("/")[2])
    if chain in anc(pid) or pid == os.getpid(): continue
    out.setdefault(t.rsplit("/", 1)[1], set()).add(pid)
print(" ".join("c%s:%s" % (c, ",".join(map(str, sorted(p)))) for c, p in sorted(out.items())) or "-")
PYX
}
card_busy() { foreign | grep -qE "(^| )c$CARD:"; }

wait_card() {  # own card quiet on 2 reads 60 s apart; give up after 2 h
    waited=0; quiet=0
    while [ $quiet -lt 2 ]; do
        if card_busy; then
            quiet=0
            [ $waited -ge 7200 ] && { note "STOP: card $CARD still held after 2 h ($(foreign))"; return 1; }
            note "waiting for card $CARD: $(foreign)"
            sleep 300; waited=$((waited + 300))
        else
            quiet=$((quiet + 1)); [ $quiet -lt 2 ] && sleep 60
        fi
    done
    return 0
}

arm() {  # arm <name> <timeout_s> -- <cmd...>
    name=$1; tmo=$2; shift 3
    "$PY" -c "import ttnn" >/dev/null 2>&1 || { note "$name NOT RUN: venv cannot import ttnn -- CHAIN STOP"; exit 1; }
    wait_card || { note "$name NOT RUN (card busy) -- CHAIN STOP"; exit 1; }
    ( while :; do echo "$(date -u +%FT%TZ) $(for c in ${CARDS//,/ }; do echo -n "aiclk$c=$(clock $c) "; done)load=$(cut -d" " -f1-3 /proc/loadavg) foreign=$(foreign)" >> "$L/env_$name.log"; sleep 30; done ) &
    sampler=$!
    note "$name START"
    timeout "$tmo" "$@" > "$L/$name.log" 2>&1
    rc=$?
    kill "$sampler" 2>/dev/null
    note "$name rc=$rc"
    echo "$rc" > "$L/$name.rc"
}

note "CHAIN START c$CARD set=$SET pid $$ at $(git -C "$T" rev-parse HEAD), interpreter $PY"
[ "$(git -C "$T" rev-parse HEAD)" = "$G" ] || { note "tree not at $G -- CHAIN STOP"; exit 1; }
test -z "$(git -C "$T" status --porcelain -- tt_bio scripts pyproject.toml tests)" || { note "tree dirty -- CHAIN STOP"; exit 1; }
E="env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD"
case $SET in
C)  # the legs rel012 D18 lost, shortest first; parity is --fresh: rel012 legs hash to other code
    arm ux_regression 7200 -- $E "$PY" scripts/ux_regression.py
    arm pytest_device 10800 -- $E "$PY" -m pytest -q --tb=short -p no:cacheprovider
    arm parity 28800 -- $E "$PY" scripts/full_parity_gate.py --workers localhost:$CARD --workdir "$L/parity_work" --fresh
    arm capacity 28800 -- env TT_BIO_LEASE_CARDS=$CARD "$PY" scripts/capacity_gate.py --workers localhost:$CARD --no-card-reset --models opendde-abag,openfold3,protenix-v1,protenix-v2,rf3,saprot-35m,saprot-650m
    ;;
P)  # parity after the 09:37Z hang: tree 265cd26b8 = a46281e91 + the gate reaping fix (tt_bio identical)
    arm parity2 28800 -- $E "$PY" scripts/full_parity_gate.py --workers localhost:$CARD --workdir "$L/parity_work2"
    ;;
B)  # bcp stage 14 bench at wk/bcp-evo 2628733f1, then the capacity cells after opendde
    arm bcp14 3600 -- $E PYTHONPATH="$BT" "$PY" "$BT/perf/bcp_evo/inproj_gated_bench.py" --out "$L/bcp14.json"
    arm capacity 28800 -- env TT_BIO_LEASE_CARDS=$CARD "$PY" scripts/capacity_gate.py --workers localhost:$CARD --no-card-reset --models opendde-abag,openfold3,protenix-v1,protenix-v2,rf3,saprot-35m,saprot-650m
    ;;
P2) # parity2 resumed on two cards (CARDS=3,0): the gate spreads seeds across --workers
    arm parity2 28800 -- env TT_BIO_LEASE_CARDS=$CARDS "$PY" scripts/full_parity_gate.py $(for c in ${CARDS//,/ }; do echo -n "--workers localhost:$c "; done | sed "s/ --workers /,/g; s/^--workers /--workers /") --workdir "$L/parity_work2"
    ;;
esac
note "CHAIN_DONE c$CARD"
