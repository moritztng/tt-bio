#!/bin/bash
# Re-fold pass for the opendde-abag 9j4c chunks the first fleet pass (od9j4c_fleet.sh) could not
# land. Three things this runner does that the first one did not, each from a measured failure:
#
#   * It starts at mps=2. At mps=5 this target reaches the end of the trunk and then dies in
#     diffusion: LayerNorm asks for 3479248896 B across 12 banks (277 MB per bank) while 654 MB
#     per bank is free but the largest free block is 218 MB. Fragmentation, not exhaustion, and it
#     is not marginal enough to win by retrying. Both c0 and c3 paid a 1.7 h trunk for that
#     refusal before falling back, so the mps=5 rung is pure loss on this cell.
#
#   * It smoke-tests a chip before committing a 2.5 h fold to it. A chip that is wedged stays
#     wedged, and inheriting one costs a whole chunk.
#
#   * Its watchdog reads log-line advance, not CPU time. Chunk 1 sat at 100 pct CPU inside a hung
#     ttnn.chunk for 97 minutes with 98 minutes of CPU time against a healthy sibling's 103 --
#     indistinguishable from working by every CPU measure, and obvious in one line of its log.
#
# Chunk state is a claim directory, so this can run alongside the first fleet script without the
# two of them folding the same chunk twice. It writes the same results.jsonl record format.
set -u
# Job control on, so every background fold lands in its own process group and the watchdog can
# signal the whole tree. setsid is not a substitute: it forks only sometimes, so $! is not reliably
# the group leader.
set -m
H=$HOME/mthuening
B=$H/p34d/od9j4c
MSA=$H/abag_xm/msa_cache
SRC=$H/odlev_src
CL=$B/claims_r2
LOG=$B/slots_r2.log
mkdir -p "$CL" "$B/chiplock" "$B/badchip" "$B/giveups"

CHUNKS="${CHUNKS:-1 5 6 7}"
# Prod (the japanfold.com tunnel) holds a worker on all 32 chips. Chips 0-25 are its live pool and
# are not ours to take; 26-31 are the ones this campaign has been folding on since 19:13Z.
# 27 is out: it wedged inside ttnn.chunk, survived SIGKILL, and now fails firmware init.
CANDIDATES="${CANDIDATES:-28 29 30 31 26}"
STALL=${STALL:-1500}          # 25 min with no new log line. Recycles run 10.3 min under co-tenancy.
BADCHIP_COOL=${BADCHIP_COOL:-1800}

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# A chip is taken if a live fold is pinned to it, which is what TT_VISIBLE_DEVICES says. Do not
# read this off /dev/tenstorrent: prod holds every chip open and idle, so an fd count says every
# chip is busy and no chunk ever runs.
chip_busy() { # <umd>
  local u=$1 p
  for p in $(pgrep -f "tt_bio.main predict" 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "TT_VISIBLE_DEVICES=$u" && return 0
  done
  return 1
}

smoke() { # <umd> -- one small matmul. A wedged chip fails or hangs here instead of eating a fold.
  TT_VISIBLE_DEVICES=$1 timeout -k 10 240 "$H/tt-bio/env/bin/python3.10" -c '
import torch, ttnn
d = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
ttnn.to_torch(ttnn.matmul(a, a))
ttnn.close_device(d)
' >/dev/null 2>&1
}

acquire() { # -> echoes a healthy, free, locked chip; blocks until one exists
  local u age
  while true; do
    for u in $CANDIDATES; do
      if [ -e "$B/badchip/$u" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$B/badchip/$u") ))
        [ "$age" -lt "$BADCHIP_COOL" ] && continue
        rm -f "$B/badchip/$u"
      fi
      chip_busy "$u" && continue
      mkdir "$B/chiplock/$u" 2>/dev/null || continue
      # Two observations, not one. The first fleet script relaunches the same chunk on the same
      # chip within a minute of an OOM, and for those seconds the chip looks free. Taking it there
      # would put two 10 GiB folds on one chip and lose both.
      sleep 90
      if chip_busy "$u"; then rmdir "$B/chiplock/$u"; continue; fi
      if smoke "$u"; then echo "$u"; return 0; fi
      rmdir "$B/chiplock/$u"
      touch "$B/badchip/$u"
      say "SMOKE FAIL UMD $u, cooling ${BADCHIP_COOL}s"
    done
    sleep 120
  done
}

fold_one() { # <chunk> <umd> <mps> -> 0 if the chunk landed 64 distinct cifs
  local c=$1 u=$2 mps=$3 seed=$((20000 + 1000 * c))
  local ob=$B/9j4c_c$c lg=$B/9j4c_c${c}_r2_mps${mps}.log rn=$B/run_c${c}_r2_mps${mps}.sh
  local s fpid rc n d oom age
  [ -d "$ob" ] && mv "$ob" "$ob.stale.$(date +%s)"
  cat > "$rn" <<EOF
cd $SRC || exit 9
TT_VISIBLE_DEVICES=$u PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  exec timeout -k 30 28800 /usr/bin/python3.10 -u -m tt_bio.main predict \\
  examples/abag_xm/9j4c.yaml --model opendde-abag --out_dir $ob --override \\
  --diffusion_samples 64 --max_parallel_samples $mps --seed $seed --host_threads 2 \\
  --msa_dir $MSA --msa_cache_only > $lg 2>&1
EOF
  s=$(date +%s)
  bash "$rn" & fpid=$!                   # set -m gives this its own process group == $!
  say "START c$c UMD $u mps=$mps pgid=$fpid"
  while kill -0 "$fpid" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "$lg" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$STALL" ]; then
      say "STALL c$c UMD $u: no log line in ${age}s -- killing the group and blacklisting the chip"
      kill -INT -- "-$fpid" 2>/dev/null; sleep 45
      kill -KILL -- "-$fpid" 2>/dev/null
      touch "$B/badchip/$u"
      break
    fi
  done
  wait "$fpid"; rc=$?
  n=$(ls $ob/*results_9j4c/structures/*.cif 2>/dev/null | wc -l)
  d=$(md5sum $ob/*results_9j4c/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
  oom=$(grep -c 'Out of Memory' "$lg" 2>/dev/null)
  printf '{"model":"opendde-abag","target":"9j4c","rung":512,"seed":%s,"chunk":%s,"chunks":8,"mps":%s,"umd":%s,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
    "$seed" "$c" "$mps" "$u" "$rc" "$(( $(date +%s) - s ))" "${n:-0}" "${d:-0}" "${oom:-0}" >> "$B/results.jsonl"
  say "END c$c UMD $u mps=$mps rc=$rc cifs=${n:-0} distinct=${d:-0} oom=${oom:-0}"
  [ "${n:-0}" -eq 64 ] && [ "${d:-0}" -eq 64 ]
}

worker() {
  local c got u mps try landed
  while true; do
    got=""
    for c in $CHUNKS; do
      [ -e "$CL/$c/ok" ] && continue
      # A chunk given up on three separate chips is a finding, not a retry. Stop asking, so the
      # mechanism reaches the state doc instead of a worker looping on it all night.
      [ "$(cat "$B/giveups/$c" 2>/dev/null || echo 0)" -ge 3 ] && continue
      mkdir "$CL/$c" 2>/dev/null || continue
      got=$c; break
    done
    [ -z "$got" ] && break

    landed=0
    for try in 1 2 3; do
      u=$(acquire)
      # mps ladder on one chip. A stall blacklists the chip, and then the ladder is the wrong
      # answer -- go get another chip instead of asking the wedged one again.
      for mps in 2 1; do
        if fold_one "$got" "$u" "$mps"; then landed=1; break; fi
        [ -e "$B/badchip/$u" ] && break
      done
      rmdir "$B/chiplock/$u" 2>/dev/null
      [ "$landed" = 1 ] && break
      say "c$got did not land on UMD $u (attempt $try/3)"
    done

    if [ "$landed" = 1 ]; then
      touch "$CL/$got/ok"
    else
      say "GIVE UP c$got after 3 chips, releasing the claim for another worker"
      echo $(( $(cat "$B/giveups/$got" 2>/dev/null || echo 0) + 1 )) > "$B/giveups/$got"
      rm -rf "$CL/$got"
      sleep 300
    fi
  done
  say "worker drained"
}

N=$(echo $CHUNKS | wc -w)
for i in $(seq "$N"); do worker & sleep 5; done
wait
echo "OD9J4C_R2_DONE $(date -Is)" >> "$B/results.jsonl"
say "OD9J4C_R2_DONE"
