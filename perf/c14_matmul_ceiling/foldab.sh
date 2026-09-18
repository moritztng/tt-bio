#!/usr/bin/env bash
# The fold A/B for the fitted in0_block_w on the DiT's short-M token projections.
#
# Session f1 is why this script has a precondition block. It held benchlock and still measured a
# contaminated fold: the RELEASE GATE was folding on node 0 (board ...4103) throughout, and the
# gate does not take benchlock, so benchlock serialised nothing -- it excludes benchlock users, not
# device users. Different board, same host, same PSU, same host DRAM and PCIe. The session's own
# A/A floor caught it (half-width 0.3724 s against a 0.251 s effect) and the reading was refused.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c14_matmul_ceiling
NODE="${NODE:-3}"
TAG="${1:-f3}"; shift || true
cd "$WT" || exit 1

gate=$(ps -eo args= | grep "[r]elease_gate" | wc -l)
if [ "$gate" -ne 0 ]; then
  echo "REFUSING: $gate release_gate process(es) live; f1 was contaminated by exactly this"
  ps -eo pid,etime,args= | grep "[r]elease_gate" | cut -c1-120
  exit 75
fi
others=$(ps -eo args= | grep -c "[p]redict_one\|[f]old_compose\|[t]rain_distogram" || true)
echo "other fold-shaped processes: $others"
python3 perf/c12_orchestrator/pair_guard/pair_idle.py --card "$NODE" || { echo "PAIR GUARD REFUSED"; exit 75; }
load=$(cut -d' ' -f1 /proc/loadavg)
echo "loadavg at launch: $load  (f1 ran at 3.43-4.90 and could not resolve a 0.25 s lever)"

exec /home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- env \
  TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/fold_ab_$TAG.json" --cifs "$OUT/fold_ab_${TAG}_cifs" \
    --arms base,bw,base --reps 9 --size 512 --mhz 1350 --palindrome "$@"
