#!/bin/bash
# ref_esmfold2.sh <esmfold2|esmfold2-fast> "<candidate cards>" [input.yaml ...]:
# esmfold2_upstream.py on one free whglx chip (the ttnn ESMC-6B needs it), holding a
# mgx-constraints lease for the run. With no inputs it runs the six the row scores.
# The thread count stays at 8 so a later input is comparable to the ones already in ref/.
set -u
cd "$(dirname "$0")"
M=$1 POOL=$2 L=$HOME/leases ME=worker:mgx-constraints
shift 2
while [ -e "$L/.mgx-quiet-window" ]; do sleep 60; done
C=
while [ -z "$C" ]; do
  for c in $POOL; do
    case " 1 24 25 26 27 " in *" $c "*) continue;; esac
    if $HOME/env/bin/python - "$L/j10glx02-card$c.json" <<'Q'
import fcntl, json, os, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    d = {"released": 1}
if not (d.get("released") or not os.path.exists(f"/proc/{d.get('pid')}")):
    sys.exit(1)
with open(sys.argv[1], "a") as fh:
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(1)
Q
    then C=$c; break; fi
  done
  [ -z "$C" ] && sleep 30
done
printf '{"host": "j10glx02", "card": "%s", "holder": "%s", "pid": %s, "acquired": %s, "released": null}' \
    "$C" "$ME" "$$" "$(date +%s)" > "$L/j10glx02-card$C.json"
echo "$(date -u +%FT%TZ) $M ref card $C"
I=perf/mgx_constraints/inputs
cd ../..
INS=("$@")
[ ${#INS[@]} -eq 0 ] && INS=($I/cyclic.yaml $I/linear13.yaml $I/bond_ligand.yaml
                              $I/bond_protein_cys.yaml $I/sfti_cyclic.yaml $I/sfti_linear.yaml)
TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_DIR=$L TT_BIO_LEASE_HOLDER=$ME \
TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH=$PWD \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  $HOME/env/bin/python perf/mgx_constraints/esmfold2_upstream.py "$M" perf/mgx_constraints/ref \
    "${INS[@]}"
echo "EXIT=$?"
$HOME/env/bin/python - "$L/j10glx02-card$C.json" <<'Q'
import json, sys, time
d = json.load(open(sys.argv[1])); d["released"] = time.time(); json.dump(d, open(sys.argv[1], "w"))
Q
