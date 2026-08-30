#!/usr/bin/env bash
# Binary search the first-parent leg 2ce29a75(GOOD 0.725106) .. 626be0b5(BAD 0.725015).
# Threshold 0.72506, midway between the two clusters.
set -u
WT=/home/moritz/.coworker/wt/opendde-512aa-residual-drift-bisect
OUT=$WT/.bisect-out
cd "$WT" || exit 1
mapfile -t C < <(git rev-list --first-parent --reverse 2ce29a75..626be0b5)
N=${#C[@]}
LO=0; HI=$N          # 1-based into C: idx i means C[i-1]; LO=0 means 2ce29a75
THR=0.72506
log(){ echo "$@" | tee -a "$OUT/search2.log"; }
log "=== $(date -u +%FT%TZ) leg has $N commits, LO=$LO(GOOD 2ce29a75) HI=$HI(BAD 626be0b5) thr=$THR ==="
while [ $((HI-LO)) -gt 1 ]; do
  MID=$(( (LO+HI)/2 ))
  SHA=${C[$((MID-1))]}; SHORT=${SHA:0:8}
  log "--- $(date -u +%FT%TZ) step idx=$MID $SHORT $(git log -1 --format=%s "$SHA" | cut -c1-70)"
  R=$("$OUT/fold_at2.sh" "T_$SHORT" "$SHA" 2>&1 | tee -a "$OUT/T_$SHORT.log" | grep -E '^(RESULT|SHA-MISMATCH|CHECKOUT-FAILED|NOJSON|TIMEOUT)')
  log "    $R"
  case "$R" in
    RESULT*) ;;
    *) log "ABORT: arm $SHORT did not produce a trusted result"; exit 1 ;;
  esac
  P=$(echo "$R" | sed -n "s/.*plddt=\['\([0-9.]*\)'\].*/\1/p")
  [ -z "$P" ] && P=$(echo "$R" | sed -n "s/.*plddt=\[\([0-9.]*\)\].*/\1/p")
  if [ -z "$P" ]; then log "ABORT: could not parse plddt from: $R"; exit 1; fi
  if awk "BEGIN{exit !($P > $THR)}"; then
    log "GOOD idx=$MID $SHORT plddt=$P"; LO=$MID
  else
    log "BAD  idx=$MID $SHORT plddt=$P"; HI=$MID
  fi
  log "    window LO=$LO HI=$HI"
done
CUL=${C[$((HI-1))]}
log "=== CULPRIT idx=$HI $(git log -1 --format='%h %ad %s' --date=short "$CUL") ==="
if [ $LO -eq 0 ]; then log "last good: 2ce29a75"; else log "last good: $(git log -1 --format='%h %s' "${C[$((LO-1))]}")"; fi
