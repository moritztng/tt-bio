#!/bin/bash
# bc4-bet-fuse sweep: one arm env against off at BindCraft 2's shape. ARM is the arm env (required), TAG names
# the out dir, DIGEST=0 skips the digest pair. A/B/A x3 on one card, devprof seam walls, a digest pair and one
# census per arm to show which bytes left DRAM.
#   CARD=<umd index> ARM=TT_BIO_AF2_T2CM=1 TAG=t2cm bash perf/bc4_fuse/l1_aba.sh   (worktree root, detached)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
ARM=${ARM:?set ARM to the arm env, e.g. TT_BIO_AF2_T2CM=1}
D=$WT/_bc4out/${TAG:-arm}aba_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fuse TT_BIO_LEASE_TIMEOUT=3600
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"
( while :; do printf "%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
run() { arm=$1; shift; echo "$arm start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
  env "$@" timeout -s INT -k 120 6000 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/$arm" \
      --warm 2 --reps 6 $EXTRA > "$D/$arm.log" 2>&1
  echo "$arm rc=$? end $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"; }
if [ "${DIGEST:-1}" = 1 ]; then EXTRA="--digest --reps 2"; run off_dig X=0; run l1_dig $ARM; fi
EXTRA=--census
run off_cen X=0
run l1_cen $ARM
EXTRA=
for r in 0 1 2; do run off$r X=0; run l1$r $ARM; done
run off3 X=0
kill $P; touch "$D/done"
