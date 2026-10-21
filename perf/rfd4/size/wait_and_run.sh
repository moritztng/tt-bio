#!/bin/bash
# Start run_qb2.sh on CARD once rfd4-size's booked window on it is open and no process holds the
# card's device node. Gives up at the window's end. Usage: wait_and_run.sh <card> <out_dir>
CARD=$1 OUT=$2 BOOK=$HOME/spd/feed/book.py
END=$(python3 $BOOK list >/dev/null; awk -v c="$CARD" '$3=="rfd4-size" {print $2}' $HOME/spd/feed/windows.tsv)
echo "waiter $$ card $CARD window end $END $(date -u +%FT%TZ)"
while [ "$(date +%s)" -lt "${END:-0}" ]; do
  if [ "$(python3 $BOOK active "$CARD")" = "rfd4-size" ] && ! fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; then
    echo "go $(date -u +%FT%TZ)"
    exec "$(dirname "$0")/run_qb2.sh" "$CARD" "$OUT"
  fi
  sleep 60
done
echo "window closed without a free card $(date -u +%FT%TZ)"
