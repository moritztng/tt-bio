#!/bin/bash
# chain.sh "<candidate cards>" <model> ...: the constraint set, then the base cell, for each
# model in turn, each batch on whichever candidate card is free at that moment.
#
# A card is free when its lease file in ~/leases is released or its holder pid is gone. The
# chain writes its own lease (holder worker:mgx-constraints, this pid) the moment it takes one,
# and again as soon as tt-bio releases it after a batch, so the card is held for the whole run
# rather than only while a fold has the device open. The cardblocked whglx chips (1, 24-27)
# are never candidates.
set -u
cd "$(dirname "$0")"
POOL=$1; shift
L=$HOME/leases; ME=worker:mgx-constraints
free() { $HOME/env/bin/python - "$L/j10glx02-card$1.json" "$ME" "$$" <<'P'
import fcntl, json, os, sys
f, me, pid = sys.argv[1:]
try:
    d = json.load(open(f))
except (OSError, ValueError):
    d = {"released": 1}
mine = d.get("holder") == me and str(d.get("pid")) == pid
if not (mine or d.get("released") or not os.path.exists(f"/proc/{d.get('pid')}")):
    sys.exit(1)
with open(f, "a") as fh:                 # a live fold holds the flock whatever the json says
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(1)
P
}
claim() { printf '{"host": "j10glx02", "card": "%s", "holder": "%s", "pid": %s, "acquired": %s, "released": null, "note": "held between folds by a mgx-constraints chain"}' \
    "$1" "$ME" "$$" "$(date +%s)" > "$L/j10glx02-card$1.json"; }
release() { $HOME/env/bin/python - "$L/j10glx02-card$1.json" <<'P'
import json, sys, time
d = json.load(open(sys.argv[1])); d["released"] = time.time(); json.dump(d, open(sys.argv[1], "w"))
P
}
C=
declare -A AVOID      # card -> epoch until which it is skipped, after someone else opened it
take() {
  while :; do
    for c in $C $POOL; do
      case " 1 24 25 26 27 " in *" $c "*) continue;; esac
      [ "${AVOID[$c]:-0}" -gt "$(date +%s)" ] && continue
      if free "$c"; then C=$c; claim "$c"; return; fi
    done
    sleep 30
  done
}
# fold <model> <tag> <run.sh args...>: rerun on another card while the open loses the card to a
# co-tenant (exit 75, nothing ran). A JSON claim holds no flock, so another row's fold that
# opens the chip regardless wins it; that is a scheduling miss, not a result.
fold() {
  local M=$1 T=$2; shift 2
  while :; do
    take; echo "$(date -u +%FT%TZ) $M $T card $C"
    ./run.sh "$M" "$C" "$T" "$@"
    grep -q "^EXIT=75 " "out/$M/$T/run.log" || break
    AVOID[$C]=$(( $(date +%s) + 600 )); C=
  done
  [ -n "$C" ] && claim "$C"
}
I=inputs
for M in "$@"; do
  fold "$M" cons --diffusion_samples 5 -- $I/sfti_cyclic_ss.yaml $I/sfti_cyclic.yaml \
      $I/sfti_ss.yaml $I/sfti_linear.yaml $I/cyclic.yaml $I/bond_ligand.yaml \
      $I/bond_protein_cys.yaml $I/modification.yaml
  fold "$M" base -- $I/base.yaml
done
[ -n "$C" ] && release "$C"
echo CHAIN_DONE
