#!/bin/bash
# opendde-abag 9j4c chunks 1-7 at the campaign config, on main + the tri_att host-assembly
# hunk and the OPM accumulator (odlev_src). One worker per chip, directory-mkdir claims so a
# worker that dies does not strand a chunk twice. mps narrows 5->2->1 exactly as p32's fold_od
# does. Chunk 0 is already folding on UMD 31 as p34d/odcamp.
set -u
H=$HOME/mthuening
B=$H/p34d/od9j4c; mkdir -p "$B" "$B/claims"
MSA=$H/abag_xm/msa_cache
SRC=$H/odlev_src
CHUNKS="1 2 3 4 5 6 7"
CHIPS="${1:-27 28 29 30}"

fold_one() { # <chunk> <chip>
  local c=$1 u=$2 seed=$((20000 + 1000 * c)) mps s rc n d oom ob
  ob=$B/9j4c_c$c
  for mps in 5 2 1; do
    s=$(date +%s)
    ( cd "$SRC" && TT_VISIBLE_DEVICES=$u PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        timeout -k 30 28800 /usr/bin/python3.10 -u -m tt_bio.main predict \
        examples/abag_xm/9j4c.yaml --model opendde-abag --out_dir "$ob" --override \
        --diffusion_samples 64 --max_parallel_samples $mps --seed $seed --host_threads 2 \
        --msa_dir "$MSA" --msa_cache_only > "$B/9j4c_c${c}_mps${mps}.log" 2>&1 )
    rc=$?
    n=$(ls $ob/*results_9j4c/structures/*.cif 2>/dev/null | wc -l)
    d=$(md5sum $ob/*results_9j4c/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
    oom=$(grep -c 'Out of Memory' "$B/9j4c_c${c}_mps${mps}.log" 2>/dev/null)
    printf '{"model":"opendde-abag","target":"9j4c","rung":512,"seed":%s,"chunk":%s,"chunks":8,"mps":%s,"umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
      $seed $c $mps $u $rc $(( $(date +%s) - s )) ${n:-0} ${d:-0} ${oom:-0} >> "$B/results.jsonl"
    [ "${oom:-0}" -gt 0 ] && [ "${n:-0}" -eq 0 ] && [ "$mps" != 1 ] || break
  done
  [ "${n:-0}" -eq 64 ] && [ "${d:-0}" -eq 64 ] && touch "$B/claims/$c/ok"
}

worker() { # <chip>
  local u=$1 c
  while true; do
    for c in $CHUNKS; do
      mkdir "$B/claims/$c" 2>/dev/null || continue
      echo "$(date -u +%FT%TZ) CLAIM c$c on UMD $u" >> "$B/slots.log"
      fold_one $c $u
      continue 2
    done
    break
  done
  echo "$(date -u +%FT%TZ) worker UMD $u drained" >> "$B/slots.log"
}

for u in $CHIPS; do worker $u & sleep 8; done
wait
echo "OD9J4C_DONE $(date -Is)" >> "$B/results.jsonl"
