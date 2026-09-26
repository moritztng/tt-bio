#!/usr/bin/env bash
# of3t-p10fp32bw: the diffusion scope pair and the two clause readings, unattended.
# Each step is skipped when its output is already there, so a relaunch resumes.
set -uo pipefail
O=/home/ttuser/of3t_p10fp32bw
cd "$O/tree"
exec >> "$O/chain.log" 2>&1
echo "=== chain start $(date -u +%FT%TZ) pid $$ ==="
for FP in off on; do
  G=$O/device_grads_rc_refatom_fp32bw_$FP.pt
  if [ -s "$G" ]; then echo "SKIP arm $FP, $G exists"; else
    bash "$O/diffarm.sh" "$FP" || { echo "ARM $FP FAILED"; exit 1; }
  fi
  C=$O/tree/perf/of3t_p10fp32bw/CLAUSE_DIFF_$FP.json
  if [ -s "$C" ]; then echo "SKIP score $FP"; else
    bash "$O/score.sh" "DIFF_$FP" "$G" || { echo "SCORE $FP FAILED"; exit 1; }
  fi
done
echo "=== chain done $(date -u +%FT%TZ) ==="
