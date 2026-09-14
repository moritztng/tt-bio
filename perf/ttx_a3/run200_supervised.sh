#!/usr/bin/env bash
# run200_supervised.sh — run the paired 1536 aa 200-step A/B and survive the qb2 dispatch wedge.
#
# Three launches of this measurement died to the same failure: the host spins in
# `fetch_queue_reserve_back` (gdb stack, twice) while the card's dispatcher is dead, and the fold
# never returns. It is not card-specific — cards 2 and 3, one of them reset one minute earlier —
# and tt-kmd logs nothing, so nothing notices. Recovery is a board-pair reset, which means the only
# missing piece was a watchdog. Progress is the mtime of the harness's JSON: fold_ab_a3.py rewrites
# it after every fold, so a stalled mtime is a stalled fold.
#
# Run it UNDER benchlock, not the other way round: the lock has to be held across a retry.
#
#   STALL_S=600 run200_supervised.sh <card> <out.json> <attempts>
set -u
CARD="${1:?card}"; OUT="${2:?out json}"; MAX="${3:-3}"
STALL_S="${STALL_S:-600}"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}"
TTSMI="${TTSMI:-$HOME/.local/bin/tt-smi}"

for att in $(seq 1 "$MAX"); do
  [ -f "$OUT" ] && mv -f "$OUT" "$OUT.att$((att-1))"
  echo "[sup] attempt $att/$MAX on card $CARD, stall budget ${STALL_S}s, $(date -Is)"
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:ttx-a3-1536-200step-fold \
    "$PY" "$WT/perf/ttx_a3/fold_ab_a3.py" --fixture cdk2x2_1536 --reps 1 \
      --steps 200 --recycles 3 --out "$OUT" &
  pid=$!
  last=$(date +%s)
  while :; do
    sleep 30
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      echo "[sup] attempt $att exited rc=$rc, $(date -Is)"
      [ "$rc" -eq 0 ] && exit 0
      break
    fi
    now=$(date +%s)
    [ -f "$OUT" ] && last=$(stat -c %Y "$OUT")
    age=$((now - last))
    if [ "$age" -ge "$STALL_S" ]; then
      echo "[sup] STALL: no fold written for ${age}s, killing pid $pid and resetting card $CARD"
      kids=$(pgrep -P "$pid" 2>/dev/null)
      kill -9 "$pid" 2>/dev/null
      for k in $kids; do kill -9 "$k" 2>/dev/null; done
      sleep 5
      # tt-smi -r resets the BOARD PAIR, not the chip, so it takes the sibling card down with it.
      # On 2026-09-14 this row reset pair 2/3 while another worker's fold held card 2 and
      # contaminated its run. Never reset while the sibling has a foreign holder: retry without it.
      sib=$(( CARD ^ 1 ))
      foreign=$(lsof -t "/dev/tenstorrent/$sib" 2>/dev/null | grep -v "^$pid$" | head -1)
      if [ -n "$foreign" ]; then
        echo "[sup] card $sib held by pid $foreign, NOT resetting the pair; retrying cold"
      else
        timeout 240 "$TTSMI" -r "$CARD" 2>&1 | tail -2
      fi
      sleep 20
      break
    fi
  done
done
echo "[sup] giving up after $MAX attempts"
exit 1
