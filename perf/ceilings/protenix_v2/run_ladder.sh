#!/bin/bash
# protenix-v2 ceiling ladder. Claims only chips outside the JapanFold production pool, one
# rung per chip, and re-checks occupancy with `sudo lsof` immediately before every claim so a
# chip the pool or a sibling task takes back is never stolen. Rungs run in priority order:
# 1024 is the answer, 1056 is its negative control, 980 is the same-family positive control.
set -u
H=/home/cust-team/mthuening
B=$H/ceilpx2
SRC=$H/tt-bio
PY=$SRC/env/bin/python3.10
MSA=$H/abag_xm/msa_cache
ALLOWED="1 2 3 4 5 6 7 10"      # /dev nodes not in the production pool at 2026-09-02 22:26Z
RUNGS="${1:-1024 1056 980 1088}"
MAXCONC=3
mkdir -p "$B/claims" "$B/out"

# UMD enumerates by PCI BDF ascending; on this box that is nodes 16..31 then 8..15 then 0..7.
umd_of() { local n=$1
  if   [ "$n" -le 7  ]; then echo $((n+24))
  elif [ "$n" -le 15 ]; then echo $((n+8))
  else echo $((n-16)); fi; }

held_nodes() { sudo lsof /dev/tenstorrent/* 2>/dev/null | tail -n +2 | awk '{print $9}' \
  | grep '^/dev/tenstorrent/' | sed 's|.*/||' | sort -u; }

free_allowed() { local held; held=$(held_nodes)
  for n in $ALLOWED; do echo "$held" | grep -qx "$n" || echo "$n"; done; }

launch() { # <rung> <node>
  local N=$1 n=$2 u; u=$(umd_of "$n")
  echo "$(date -u +%FT%TZ) claim rung=$N node=$n umd=$u" >> "$B/sched.log"
  setsid env N="$N" U="$u" NODE="$n" H="$H" B="$B" SRC="$SRC" PY="$PY" MSA="$MSA" bash -c '
    cd "$SRC"; s=$(date +%s)
    TT_VISIBLE_DEVICES=$U PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      HF_HUB_CACHE=$H/models \
      timeout -k 30 7200 $PY -u -m tt_bio.main predict $B/inputs/px$N.yaml \
      --model protenix-v2 --out_dir $B/out/$N --override --fast \
      --diffusion_samples 1 --max_parallel_samples 1 --seed 42 --host_threads 2 \
      --msa_dir $MSA --msa_cache_only > $B/px$N.log 2>&1
    rc=$?; secs=$(( $(date +%s) - s ))
    cifs=$(ls $B/out/$N/*results_*/structures/*.cif 2>/dev/null | wc -l)
    oom=$(grep -c "Out of Memory" $B/px$N.log 2>/dev/null)
    printf "{\"rung\":%s,\"node\":%s,\"umd\":%s,\"rc\":%s,\"secs\":%s,\"cifs\":%s,\"oom\":%s}\n" \
      "$N" "$NODE" "$U" "$rc" "$secs" "${cifs:-0}" "${oom:-0}" >> $B/results.jsonl
    touch $B/claims/$N.done
  ' </dev/null >/dev/null 2>&1 &
}

running() { local c=0 r
  for r in $RUNGS; do
    [ -e "$B/claims/$r" ] && [ ! -e "$B/claims/$r.done" ] && c=$((c+1))
  done; echo $c; }

for N in $RUNGS; do
  [ -e "$B/claims/$N" ] && continue
  while [ "$(running)" -ge "$MAXCONC" ]; do sleep 30; done
  node=""
  while [ -z "$node" ]; do
    a=$(free_allowed); [ -z "$a" ] && { sleep 45; continue; }
    sleep 15
    b=$(free_allowed)
    for n in $a; do echo "$b" | grep -qx "$n" && { node=$n; break; }; done
    [ -z "$node" ] && sleep 45
  done
  mkdir "$B/claims/$N" 2>/dev/null || continue
  launch "$N" "$node"
  sleep 20
done
echo "$(date -u +%FT%TZ) all rungs dispatched" >> "$B/sched.log"
