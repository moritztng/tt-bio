#!/bin/bash
# Host cost of AF2's structure module inside one BindCraft 2 iteration, measured as a removal:
# census A/B/A on one card, B = --stub-structure (timing only). Read with stub_read.py OUT.
#   usage: stub_aba.sh CARD OUT [LOCKFILE]     env: PY (python), BC2 (checkout at 7a2dfdb),
#   B (the B arm, default "stub --stub-structure"; "sm --structure card" for the structure module on card)
set -u
CARD=$1; OUT=$2; LOCK=${3:-}
if [ -n "$LOCK" ] && [ -z "${STUB_ABA_LOCKED:-}" ]; then
    STUB_ABA_LOCKED=1 exec flock "$LOCK" "$0" "$CARD" "$OUT"
fi
cd "$(dirname "$0")/../.."
PY=${PY:-python3}; BC2=${BC2:-$HOME/bcx_e2e/bc2}
mkdir -p "$OUT"
echo "start $(date -u +%FT%TZ) sha $(git rev-parse HEAD) card $CARD host $(hostname)" >> "$OUT/chain.log"
fail=0
run() {
    local name=$1; shift
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-headdev \
        timeout 3600 "$PY" perf/spd/bc2_census.py --out "$OUT/$name" --bc2 "$BC2" --seed 101 --at 40 \
        --profile 0 "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "$name rc=$rc $(date -u +%FT%TZ)" >> "$OUT/chain.log"
    [ $rc -eq 0 ] || fail=1
}
run eager-a
run ${B:-stub --stub-structure}
run eager-b
# done only if every leg ran: a chain that failed must not read as finished
if [ $fail -eq 0 ]; then touch "$OUT/done"; else touch "$OUT/failed"; fi
