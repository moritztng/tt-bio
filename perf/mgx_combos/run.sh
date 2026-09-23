#!/bin/bash
# Combination folds on whglx: run.sh "<candidate cards>" <model>:<input>[:<samples>] ...
# Each job folds in its own process into out/<model>/<stem>_s<samples>/, log alongside, ending
# EXIT=<rc> WALL=<s>. Samples default to 5. Every fold runs at --host_threads 2 (the campaign's
# whglx load policy), so wall times here are outcomes, not speed measurements.
#
# A card is taken only when its lease is released or its holder is dead, and the chain writes its
# own lease between folds (tt-bio releases the card when each fold exits), so a sibling row cannot
# take the chip mid-chain. A fold that loses the open anyway (DeviceInUseError, or another row's
# lease) reruns elsewhere, and the chain re-takes its card after a fold only if it is still free.
# A job whose results.json already says ok is skipped, so a restarted chain keeps finished logs.
# The cardblocked chips (1, 24-27) are never candidates. POLL (s, default 30) is how often a waiting
# chain looks for a free card. Runs the tree this script lives in.
set -u
cd "$(dirname "$0")/../.."
POOL=$1; shift
L=$HOME/leases; ME=worker:mgx-combos
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
with open(f, "a") as fh:
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(1)
P
}
claim() { printf '{"host": "j10glx02", "card": "%s", "holder": "%s", "pid": %s, "acquired": %s, "released": null, "note": "held between folds by a mgx-combos chain"}\n' \
    "$1" "$ME" "$$" "$(date +%s)" > "$L/j10glx02-card$1.json"; }
release() { $HOME/env/bin/python - "$L/j10glx02-card$1.json" <<'P'
import json, sys, time
d = json.load(open(sys.argv[1])); d["released"] = time.time(); json.dump(d, open(sys.argv[1], "w"))
P
}
C=
declare -A AVOID
take() {
  while :; do
    for c in $C $POOL; do
      case " 1 24 25 26 27 " in *" $c "*) continue;; esac
      [ "${AVOID[$c]:-0}" -gt "$(date +%s)" ] && continue
      if free "$c"; then C=$c; claim "$c"; return; fi
    done
    sleep "${POLL:-30}"
  done
}
export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=$ME TT_BIO_LEASE_DIR=$L
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxc TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_TIMEOUT=60
for job in "$@"; do
  IFS=: read -r m f n <<< "$job"; n=${n:-5}
  s=$(basename "${f%.*}")_s$n; out=perf/mgx_combos/out/$m; mkdir -p "$out"
  grep -qs '"status": "ok"' "$out/$s"/*_results_*/results.json && continue
  while :; do
    take; start=$(date +%s)
    echo "START $(date -u +%FT%TZ) card=$C commit=$(git rev-parse --short HEAD) size_limit=${TT_BIO_SIZE_LIMIT:-on} job=$job" > "$out/$s.log"
    TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C $HOME/env/bin/python -m tt_bio.main predict "$f" \
        --model "$m" --out_dir "$out/$s" --diffusion_samples "$n" --host_threads 2 \
        --accelerator tenstorrent >> "$out/$s.log" 2>&1
    rc=$?
    echo "EXIT=$rc WALL=$(( $(date +%s) - start ))s" >> "$out/$s.log"
    if free "$C"; then claim "$C"; else C=; fi
    grep -qE 'DeviceInUseError|is in use by worker:' "$out/$s.log" || break
    [ -n "$C" ] && AVOID[$C]=$(( $(date +%s) + 600 )); C=
  done
done
[ -n "$C" ] && release "$C"
echo CHAIN_DONE
