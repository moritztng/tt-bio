#!/bin/bash
# Repeat one boltz2 input on one chip: run.sh <inproc|fresh> <N> <yaml> <tag> [card pool]
#   inproc  one process folds N copies of the input (model loaded once, program cache warm)
#   fresh   N processes fold one copy each
# Every fold is seed 0, --host_threads 2, one diffusion sample. Output: $OUT/<tag>/, with
# hash.txt from sitecustomize.py (DET_LEVEL, default stage) and one CIF per fold.
# A card is taken only when its lease is released or its holder is dead and no live process has
# it in TT_VISIBLE_DEVICES; the chain holds the lease between folds. Waits while the MGX quiet
# window file exists. Never a cardblocked chip (1, 24-27) or chip 4.
set -u
cd "$(dirname "$0")/../.."
MODE=$1 N=$2 Y=$3 TAG=$4; POOL=${5:-"0 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 28 29 30 31"}
L=$HOME/leases ME=worker:mgx-determinism OUT=${OUT:-$HOME/scratch/mgxdet}
QUIET=$HOME/mgx-quiet-window
D=$OUT/$TAG; mkdir -p "$D/in"
free() { $HOME/env/bin/python - "$L/j10glx02-card$1.json" "$ME" "$$" "$1" <<'P'
import fcntl, json, os, sys
f, me, pid, card = sys.argv[1:]
try:
    d = json.load(open(f))
except (OSError, ValueError):
    d = {"released": 1}
if not (d.get("holder") == me and str(d.get("pid")) == pid):
    if not (d.get("released") or not os.path.exists(f"/proc/{d.get('pid')}")):
        sys.exit(1)
with open(f, "a") as fh:                          # a fold that holds the card holds this flock
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(1)
for p in os.listdir("/proc"):                     # no live process may have the card pinned
    try:
        env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
    except OSError:
        continue
    if f"TT_VISIBLE_DEVICES={card}".encode() in env and p != pid:
        sys.exit(1)
P
}
claim() { printf '{"host": "j10glx02", "card": "%s", "holder": "%s", "pid": %s, "acquired": %s, "released": null, "note": "held between folds by a mgx-determinism chain"}\n' \
    "$1" "$ME" "$$" "$(date +%s)" > "$L/j10glx02-card$1.json"; }
release() {  # only a lease this chain still holds: tt-bio rewrites it with the fold's pid at open
  $HOME/env/bin/python -c "import json,sys,time; f=sys.argv[1]; d=json.load(open(f))
if str(d.get('pid')) == sys.argv[2] and not d.get('released'):
    d['released']=time.time(); json.dump(d,open(f,'w'))" "$L/j10glx02-card$1.json" "$$"; }
C=
take() {
  while :; do
    if [ ! -e "$QUIET" ]; then
      for c in $C $POOL; do
        case " 1 4 24 25 26 27 " in *" $c "*) continue;; esac
        if free "$c"; then C=$c; claim "$c"; return; fi
      done
    fi
    sleep "${POLL:-30}"
  done
}
trap '[ -n "$C" ] && release "$C"' EXIT
stem=$(basename "${Y%.*}")
for i in $(seq -w 1 "$N"); do
  sed "s#: perf/#: $PWD/perf/#g" "$Y" > "$D/in/${stem}_r$i.yaml"
done
export PYTHONPATH=$PWD:$PWD/perf/mgx_det TT_BIO_LEASE_HOLDER=$ME TT_BIO_LEASE_DIR=$L DET_HASH=$D/hash.txt
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxdet TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_TIMEOUT=60
fold() {  # <input file or dir> <log>
  take
  echo "START $(date -u +%FT%TZ) card=$C commit=$(git rev-parse --short HEAD) level=${DET_LEVEL:-stage} load=$(cut -d' ' -f1 /proc/loadavg) in=$1" >> "$2"
  local t=$(date +%s)
  TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C $HOME/env/bin/python -m tt_bio.main predict "$1" \
      --model boltz2 --diffusion_samples 1 --seed 0 --host_threads 2 --accelerator tenstorrent \
      --out_dir "$D/out" ${EXTRA:-} >> "$2" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - t ))s load=$(cut -d' ' -f1 /proc/loadavg)" >> "$2"
}
if [ "$MODE" = inproc ]; then
  fold "$D/in" "$D/run.log"
else
  for f in "$D"/in/*.yaml; do fold "$f" "$D/run.log"; done
fi
echo CHAIN_DONE >> "$D/run.log"
