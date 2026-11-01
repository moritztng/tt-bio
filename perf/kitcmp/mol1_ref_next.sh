#!/bin/bash
# Continue the reference predictions on a fresh box once the previous session ends, run DETACHED on pc:
#   QB2=user@host:dir bash mol1_ref_next.sh <prev-label> <next-label> <cap_h> [query]
# Waits for results/<prev>/SESSION-END, hardlinks the previous box's outputs on qb2 into inputs/_ref (box_mol1.sh
# copies _ref into its tree and mol1_ref.py skips every job with a summary), then starts mol1_session.sh REF=1.
# Written 11-01 when the only single-B200 offer besides the running box never came up on ssh: one box in series.
set -u
PREV=${1:?prev label} NEXT=${2:?next label} CAP=${3:?cap h} QB2=${QB2:?QB2=user@host:dir}
P=$(cd "$(dirname "$0")" && pwd); log(){ echo "$(date -u +%FT%TZ) $*"; }
until [ -f "$P/results/$PREV/SESSION-END" ]; do sleep 120; done
log "$PREV ended: $(cat "$P/results/$PREV/SESSION-END")"
ssh -o BatchMode=yes "${QB2%%:*}" "cd ${QB2#*:} && for d in $PREV/ref/*/; do mkdir -p inputs/_ref/\$(basename \$d) && cp -aln \$d. inputs/_ref/\$(basename \$d)/; done && find inputs/_ref -name '*summary*' | wc -l" ||
  { log "seeding qb2 _ref from $PREV failed"; exit 1; }
log "seeded _ref, launching $NEXT"
exec env QB2="$QB2" INPUTS="${INPUTS:-}" BOXENV="REF=1 ${REVERSE:+REF_REVERSE=1}" \
  bash "$P/mol1_session.sh" "$NEXT" "$CAP" "${4:-gpu_name=B200 num_gpus=1 cuda_max_good>=13.0 inet_down>=200 reliability>=0.90 dph<=8}"
