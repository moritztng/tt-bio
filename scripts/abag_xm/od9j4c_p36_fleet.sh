#!/bin/bash
# opendde-abag 9j4c, all eight chunks, at the campaign config, on the row-blocked trimul input
# norm (tt_bio/tenstorrent.py TriangleMultiplication._in_proj_rows). One worker per chip,
# directory-mkdir claims so a worker that dies does not strand a chunk twice.
#
# Starts at mps=5, the campaign default, and not at the bottom of a ladder: the refused buffer
# was 3479248896 B at mps=5, 2 and 1 alike (state §18.1), so mps was never the constraint and
# there is no reason to pay 5x the diffusion cost for it.
#
# The ladder is kept only as a safety net, with the defect §17.4 named fixed: od9j4c_fleet.sh
# laddered on `oom>0` and took `break` on anything else, so a timeout or a signal read as a
# terminal verdict and stranded the chunk (c1's mps=5 record is rc=137, oom=0, REJECT, and it
# needed a whole second fleet to pick it up). Here a non-OOM failure retries the SAME rung once;
# only an OOM moves down one.
set -u
H=$HOME/mthuening
B=$H/p34d/od9j4c                 # same ledger od9j4c_accept.sh reads
SRC=${SRC:-$H/p35_src}
MSA=$H/abag_xm/msa_cache
CHUNKS=${CHUNKS:-"0 1 2 3 4 5 6 7"}
CHIPS="${1:-26 28 29 30 31}"
mkdir -p "$B" "$B/claims"

fold_one() { # <chunk> <chip>
  local c=$1 u=$2 seed=$((20000 + 1000 * c)) mps try s rc n d oom ob lg
  ob=$B/9j4c_c$c
  for mps in 5 2 1; do
    for try in 1 2; do
      lg=$B/9j4c_c${c}_p36_mps${mps}_t${try}.log
      s=$(date +%s)
      ( cd "$SRC" && TT_VISIBLE_DEVICES=$u PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
          timeout -k 30 28800 /usr/bin/python3.10 -u -m tt_bio.main predict \
          examples/abag_xm/9j4c.yaml --model opendde-abag --out_dir "$ob" --override \
          --diffusion_samples 64 --max_parallel_samples $mps --seed $seed --host_threads 2 \
          --msa_dir "$MSA" --msa_cache_only > "$lg" 2>&1 )
      rc=$?
      n=$(ls $ob/*results_9j4c/structures/*.cif 2>/dev/null | wc -l)
      d=$(md5sum $ob/*results_9j4c/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
      oom=$(grep -ac 'Out of Memory' "$lg" 2>/dev/null)
      printf '{"model":"opendde-abag","target":"9j4c","rung":512,"seed":%s,"chunk":%s,"chunks":8,"mps":%s,"umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
        $seed $c $mps $u $rc $(( $(date +%s) - s )) ${n:-0} ${d:-0} ${oom:-0} >> "$B/results.jsonl"
      if [ "${n:-0}" -eq 64 ] && [ "${d:-0}" -eq 64 ]; then
        touch "$B/claims/$c/ok"; return
      fi
      [ "${oom:-0}" -gt 0 ] && break        # capacity: drop a rung
    done                                    # otherwise retry this rung once
    [ "${oom:-0}" -gt 0 ] || return         # two non-OOM failures here: stop, do not ladder
  done
}

worker() { # <chip>
  local u=$1 c
  while true; do
    for c in $CHUNKS; do
      [ -e "$B/claims/$c/ok" ] && continue
      mkdir "$B/claims/$c" 2>/dev/null || continue
      echo "$(date -u +%FT%TZ) P36 CLAIM c$c on UMD $u" >> "$B/slots.log"
      fold_one $c $u
      continue 2
    done
    break
  done
  echo "$(date -u +%FT%TZ) P36 worker UMD $u drained" >> "$B/slots.log"
}

for u in $CHIPS; do worker $u & done
wait
echo "$(date -u +%FT%TZ) P36_DONE" >> "$B/results.jsonl"
