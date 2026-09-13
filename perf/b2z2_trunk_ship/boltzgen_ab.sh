#!/bin/bash
# Is the gate boltzgen leg 75% / 0.7428 a lever effect or leg variance? Same leg, same seed, same
# card, three runs: flags off (what main scored 100% / 0.80851), flags on, flags off again.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-trunk-byte-round2-ship
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
run() { # $1 = tag, $2 = flag value
  rm -rf "$WT/boltzgen_gate_binder"
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
  TT_BIO_LEASE_HOLDER=worker:b2z2-trunk-byte-round2-ship PYTHONPATH="$WT" \
  TT_BIO_TRIATT_FUSED_QKVG="$2" TT_BIO_TRIATT_FUSED_QKVGB="$2" TT_BIO_TRIMUL_FUSED_GOUT="$2" \
  "$PY" scripts/full_parity_gate.py --leg boltzgen --fresh --workers localhost:0 \
    --load-ceiling 999 --workdir "perf/b2z2_gate/bg_${1}_work" \
    --out "perf/b2z2_gate/gate_bg_${1}.json" > "perf/b2z2_gate/bg_${1}.log" 2>&1
  echo "== $1 rc=$? $(cat perf/b2z2_gate/bg_${1}_work/boltzgen.json 2>/dev/null | tr -d "\n")"
}
run off1 0
run on1 1
run off2 0
echo ALLDONE
