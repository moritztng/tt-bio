#!/usr/bin/env bash
# Keep the burst-clock soak running until the epoch in until.txt, across host resets.
#
# qb2 resets on its own from the open PCIe/NoC fault (state/qbroot-verdict-pcie-link-failure.md),
# so a three-hour soak cannot be one process. Cron runs this every minute; each role takes its own
# non-blocking lock, so a role that is already up is left alone and a role the reset killed comes
# back. Once the deadline passes this removes its own cron line, because a leftover entry outlives
# the worktree it points into.
#
# Card 1 only. A governor arm on card 0 of the same board would have been the better control, but
# ttx-a3-sdpa-ship-remerge2 holds card 0 for a release gate for the length of this soak, so the
# not-held comparison comes from cards 0, 2 and 3 in the same window of ~/qbcard/cardtel.tsv.
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/out"
UNTIL_FILE="$DIR/until.txt"
HOLDER="worker:b2z2-aiclk-default-decision"
PY=/home/ttuser/tt-bio-dev/env/bin/python3

[ -r "$UNTIL_FILE" ] || exit 0
UNTIL="$(tr -dc '0-9.' < "$UNTIL_FILE")"
mkdir -p "$OUT"

if [ "$(printf '%.0f' "$UNTIL")" -le "$(date +%s)" ]; then
    # Deadline passed: stop coming back, and take the cron line with us.
    if crontab -l 2>/dev/null | grep -q 'b2z2_aiclk_soak/soak.sh'; then
        crontab -l 2>/dev/null | grep -v 'b2z2_aiclk_soak/soak.sh' | crontab -
        echo "$(date -u +%FT%TZ) cron line removed, deadline $UNTIL passed" >> "$OUT/soak.sh.log"
    fi
    exit 0
fi

start() {   # start <role-name> <env-assignments...> -- <soak.py args...>
    local name="$1"; shift
    local lock="$OUT/$name.lock"
    flock -n "$lock" true 2>/dev/null || return 0        # already held: that role is up
    echo "$(date -u +%FT%TZ) starting $name (boot $(cut -c1-8 /proc/sys/kernel/random/boot_id))" \
        >> "$OUT/soak.sh.log"
    setsid flock -n "$lock" env "$@" >> "$OUT/$name.out" 2>&1 &
}

cd "$DIR" || exit 0

start hold \
    "$PY" "$DIR/soak.py" --role hold --node 1 --mhz 1350 \
    --log "$OUT/hold.jsonl" --until "$UNTIL"

start fold-held \
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER="$HOLDER" \
    "$PY" "$DIR/soak.py" --role fold --arm held --interval 240 \
    --log "$OUT/folds.jsonl" --until "$UNTIL"
