#!/bin/bash
# mol1-bh-prof: OPM contraction plan on Blackhole, in the fold. Arms in separate processes, interleaved base plan base plan,
# each 1 cold + 2 warm folds of a frozen PROTOCOL input through perf/mol1/fair/run.py (--part dev, AICLK sampled during
# every fold). base = TT_BIO_OPM_CFG=0 (ttnn's own plan, what main runs on Blackhole), plan = opm_contract_config.
# Then grade_ab.py plan vs base per size.  Usage: CARD=1 opm_ab.sh OUT REPS SIZE [SIZE ...]
set -u
W=/home/ttuser/.coworker/wt/mol1-bh-prof; PY=/home/ttuser/tt-bio-dev/env/bin/python
O=$1; N=$2; shift 2; mkdir -p "$O"; C=${CARD:-1}
cd /home/ttuser/mol1-fair/inputs
exec 9>/home/ttuser/spd_qb1_card$C.lock; flock -w 3600 9 || { echo "no flock" >> "$O/ab.log"; exit 1; }
for S in "$@"; do
  for r in $(seq 1 "$N"); do
    for arm in base plan; do
      cfg=1; [ $arm = base ] && cfg=0
      echo "$(date -u +%FT%TZ) $S rep $r arm $arm" >> "$O/ab.log"
      TT_BIO_OPM_CFG=$cfg TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-bh-prof PYTHONPATH=$W \
        timeout 3000 $PY $W/perf/mol1/fair/run.py --input mol1-$S.json --chips $C --part dev --rounds 2 --max-attempts 2 \
        --seed $((100 + r)) --arm $arm --out "$O/$S/$arm/r$r" >> "$O/ab.log" 2>&1
      echo "$(date -u +%FT%TZ) rc=$? $S $arm r$r" >> "$O/ab.log"
    done
  done
done
touch "$O/DONE"
