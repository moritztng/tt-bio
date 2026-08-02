#!/bin/bash
# p2 step 1 driver: warm private kernel caches (cards 1+2), then timed RFD3 A/B legs (card 0).
# Detached campaign: logs per leg in /tmp/rfd3_*.log, DONE marker at the end.
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
export TT_BIO_TRACE_REGION_SIZE=$((1<<30))
rm -f /tmp/rfd3_step1_DONE

SCOUT_CARD=1 scripts/scout_run_leg.sh 68 scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 --legs eager both --alternations 1 --timesteps 3 --json_out /tmp/rfd3_warm68.json >/tmp/rfd3_warm68.log 2>&1 &
W68=$!
SCOUT_CARD=2 scripts/scout_run_leg.sh 75 scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 --legs eager both --alternations 1 --timesteps 3 --json_out /tmp/rfd3_warm75.json >/tmp/rfd3_warm75.log 2>&1 &
W75=$!
wait $W68 $W75
echo "warmups done: $(date -u)" >> /tmp/step1_driver.log

for p in 1 2; do
  SCOUT_CARD=0 scripts/scout_run_leg.sh 68 scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 --legs eager both --alternations 2 --json_out artifacts/ttnn075-scout/rfd3_68_p$p.json >/tmp/rfd3_68_p$p.log 2>&1
  echo "leg 68 p$p rc=$? $(date -u)" >> /tmp/step1_driver.log
  SCOUT_CARD=0 scripts/scout_run_leg.sh 75 scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 --legs eager both --alternations 2 --json_out artifacts/ttnn075-scout/rfd3_75_p$p.json >/tmp/rfd3_75_p$p.log 2>&1
  echo "leg 75 p$p rc=$? $(date -u)" >> /tmp/step1_driver.log
done
touch /tmp/rfd3_step1_DONE
