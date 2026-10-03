#!/usr/bin/env bash
# Reset the board that carries these chips, bounded, and check the chips answer afterwards.
#
#   reset_board.sh 2,3
#
# qb2's p300 boards carry chips (0,1) and (2,3), and a reset takes both. The engine calls this
# with every chip of one board after it has stopped their workers with SIGINT. It refuses to
# reset a board while any process outside the demo still has one of its chips open: that is
# somebody else's work, and the reset would wedge it.
set -uo pipefail
ids=${1:?usage: reset_board.sh CHIP[,CHIP]}
smi=${TT_SMI:-$HOME/.local/bin/tt-smi}
holder=${TT_BIO_LEASE_HOLDER:-worker:sc26-demo}
log(){ echo "$(date -u +%FT%TZ) reset $ids: $*"; }
targets=()
for c in ${ids//,/ }; do
  for p in $(fuser "/dev/tenstorrent/$c" 2>/dev/null); do
    if ! tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qxF "TT_BIO_LEASE_HOLDER=$holder"; then
      log "chip $c is open by pid $p, which is not the demo's; not resetting"
      exit 3
    fi
  done
  targets+=("/dev/tenstorrent/$c")
done
log "tt-smi -r ${targets[*]}"
timeout -s INT 120 "$smi" -r "${targets[@]}" < /dev/null
rc=$?
log "tt-smi rc=$rc"
[ $rc = 0 ] || exit $rc
# A reset chip must answer: a readable clock (a dead ARC answers 0xFFFFFFFF) and a moving heartbeat.
for c in ${ids//,/ }; do
  d="/sys/class/tenstorrent/tenstorrent!$c"
  ok=0
  for _ in $(seq 30); do
    a=$(cat "$d/tt_aiclk" 2>/dev/null); h1=$(cat "$d/tt_heartbeat" 2>/dev/null)
    sleep 1
    h2=$(cat "$d/tt_heartbeat" 2>/dev/null)
    if [ -n "$a" ] && [ "$a" != 4294967295 ] && [ -n "$h1" ] && [ "$h1" != "$h2" ]; then ok=1; break; fi
  done
  [ $ok = 1 ] || { log "chip $c does not answer after the reset (aiclk=$a heartbeat $h1->$h2)"; exit 4; }
  log "chip $c back: aiclk $a MHz, heartbeat moving"
done
