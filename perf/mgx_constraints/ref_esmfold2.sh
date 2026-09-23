#!/bin/bash
# ref_esmfold2.sh <esmfold2|esmfold2-fast> "<candidate cards>": esmfold2_upstream.py on one
# free whglx chip (the ttnn ESMC-6B needs it), holding a mgx-constraints lease for the run.
set -u
cd "$(dirname "$0")"
M=$1 POOL=$2 L=$HOME/leases ME=worker:mgx-constraints
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
I=inputs
cd ../..
TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_DIR=$L TT_BIO_LEASE_HOLDER=$ME \
TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH=$PWD \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  $HOME/env/bin/python perf/mgx_constraints/esmfold2_upstream.py "$M" perf/mgx_constraints/ref \
    perf/mgx_constraints/$I/cyclic.yaml perf/mgx_constraints/$I/linear13.yaml \
    perf/mgx_constraints/$I/bond_ligand.yaml perf/mgx_constraints/$I/bond_protein_cys.yaml \
    perf/mgx_constraints/$I/sfti_cyclic.yaml perf/mgx_constraints/$I/sfti_linear.yaml
echo "EXIT=$?"
$HOME/env/bin/python - "$L/j10glx02-card$C.json" <<'Q'
import json, sys, time
d = json.load(open(sys.argv[1])); d["released"] = time.time(); json.dump(d, open(sys.argv[1], "w"))
Q
