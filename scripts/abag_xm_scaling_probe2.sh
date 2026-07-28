#!/usr/bin/env bash
# The owed 2-card leg of the Phase-1 acceptance (Moritz asked for 1/2/4 cards), plus a 1-card
# capped rerun as a drift control against turn 11's 185.3 s. Same target, sample count, seed and
# warm kernel cache as turn 11, so the numbers are directly comparable. Order is L1c-again,
# L2u, L2c: the drift control runs FIRST so a drifted host is caught before the 2-card legs,
# not explained away after them.
set -u
WT=/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4
PY=/home/ttuser/tt-bio/env/bin/python3
OUT=/home/ttuser/abag_xm/scaling
YAML=examples/abag_xm/9w14.yaml
N=8
JSONL=$OUT/scaling.jsonl
mkdir -p "$OUT"
cd "$WT" || exit 1

fold() {  # fold <leg> <card> <host_threads|->
  local leg=$1 card=$2 ht=$3 extra=()
  [ "$ht" != "-" ] && extra=(--host_threads "$ht")
  local t0 t1
  t0=$(date +%s.%N)
  TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_HOLDER=worker:abag-xm-crossmodel-ranking-dataset-p4 \
  PYTHONPATH=$WT "$PY" -m tt_bio.main predict "$YAML" --model protenix-v2 \
    --out_dir "$OUT/$leg-c$card" --diffusion_samples $N --max_parallel_samples 5 \
    --msa_dir /home/ttuser/abag_xm/msa_cache --msa_db_path /home/ttuser/.boltz/msa_db \
    --seed 42 --override --write_pae "${extra[@]}" > "$OUT/$leg-c$card.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  echo "{\"leg\":\"$leg\",\"card\":$card,\"host_threads\":\"$ht\",\"n\":$N,\"rc\":$rc,\"wall_s\":$(echo "$t1-$t0"|bc)}" >> "$JSONL"
}

sample() { # sample <leg> <seconds>
  local leg=$1 dur=$2 i=0
  while [ $i -lt "$dur" ]; do
    echo "$(date +%s) $leg loadavg=$(cut -d' ' -f1 /proc/loadavg) runnable=$(ps -eLo stat | grep -c '^R')" \
      >> "$OUT/hostsample2.log"
    sleep 10; i=$((i+10))
  done
}

echo "=== L1c2: 1 card cap 8 (drift control vs turn 11's 185.3 s) $(date +%T) ==="
sample L1c2 300 & SP=$!
fold L1c2 0 8
kill $SP 2>/dev/null

echo "=== L2u: 2 cards uncapped $(date +%T) ==="
sample L2u 400 & SP=$!
PIDS=(); for c in 0 1; do fold L2u $c - & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SP 2>/dev/null

echo "=== L2c: 2 cards --host_threads 16 $(date +%T) ==="
sample L2c 400 & SP=$!
PIDS=(); for c in 0 1; do fold L2c $c 16 & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $SP 2>/dev/null

echo "TWO_CARD PROBE COMPLETE $(date +%T)"
