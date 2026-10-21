#!/bin/bash
# rfd4-size device phase on qb2: the 256 and 512 censuses, the 329-token antibody census on this
# branch, and the same antibody design on the code that measured 231 s/design (e6240569d), so the
# gap can be split into fixed cost, batching and code. Usage: run_qb2.sh <card> <out_dir>
set -u
CARD=$1 OUT=$2
WT=$HOME/.coworker/wt/rfd4-size OLD=$HOME/rfd4-size-old
PY=$HOME/tt-bio-dev/env/bin/python
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:rfd4-size
export TT_BIO_RFD4_WEIGHTS=$HOME/rfd4-bundle
C="$WT/perf/rfd4/e2e/census.py --weights $HOME/rfd4-bundle --work $OUT/work"
AB=$WT/perf/rfd4/size/specs/antibody.yaml
step() {  # name, pythonpath, args...
  local name=$1 pp=$2; shift 2
  echo "== $name start $(date -u +%FT%TZ)"
  (cd "$pp" && PYTHONPATH=$pp timeout 5400 $PY $C --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1)
  echo "== $name end rc=$? $(date -u +%FT%TZ)"
}
echo "start $(date -u +%FT%TZ) card $CARD $(git -C $WT log --oneline -1)"
step e2e-256 "$WT" --spec $HOME/rfd4-inputs/pdl1-256.yaml
step e2e-512 "$WT" --spec $HOME/rfd4-inputs/pdl1-512.yaml
step antibody-352 "$WT" --spec "$AB" --recipe high/front_loaded
step antibody-w2 "$WT" --spec "$AB" --recipe high/front_loaded --designs 2 --width 2 --no-instrument
# The code and the default (max_parallel_samples=5, so two designs batch at width 2) the 231 s ran on.
step antibody-old "$OLD" --spec "$AB" --recipe high/front_loaded --designs 2 --width 5 --no-instrument
echo "end $(date -u +%FT%TZ)"
