#!/usr/bin/env bash
# abag_xm_host_hold.sh -- tell the fleet dispatcher this whole host is busy folding.
#
# Why a HOST hold and not a card lease. tt-bio keys two different things to one flock on
# ~/.coworker/state/leases/<host>-card<N>.json: the fleet's "is this card free" check
# (fleet.sh remote_busy_cards) and the device-open mutex (tt_bio/device_lease.py). So a
# campaign parent cannot claim a card between folds:
#   * hold the flock  -> the fold's own DeviceLease.acquire() polls the same path and dies
#                        after TT_BIO_LEASE_TIMEOUT. That IS the failure this is meant to fix.
#   * write the JSON without flocking -> fleet.sh pick_card runs `flock -n` on it, succeeds,
#                        rm -f's the file and takes the card anyway.
# The dispatcher does honour a separate, lock-free host hold (fleet.sh host_blocked reads
# $HOME/.coworker/state/<h>-hold as a unix expiry timestamp), and this campaign occupies every
# card on both boxes, so host granularity is exactly right.
#
# The hold file lives on the host that RUNS fleet.sh, not on the folding host -- ~/.coworker
# gitignores state/, so the two are unrelated directories and writing it locally on a
# QuietBox does nothing at all. Hence the ssh.
#
# Expiry is short and refreshed rather than long and absolute: a campaign that crashes must
# release the fleet by itself within the hour instead of blocking it until someone notices.
#
#   Usage:  abag_xm_host_hold.sh refresh    # extend the hold (idempotent; call on a timer)
#           abag_xm_host_hold.sh release    # campaign over -- give the cards back
#           abag_xm_host_hold.sh status
set -u
FLEET_HOST="${ABAG_XM_FLEET_HOST:-moritz@pc}"
TTL="${ABAG_XM_HOLD_TTL:-3600}"
case "$(hostname)" in
  tt-quietbox)  ALIAS=qb1 ;;
  tt-quietbox2) ALIAS=qb2 ;;
  pc)           ALIAS=pc  ;;
  *)            ALIAS="$(hostname)" ;;
esac
HOLD="\$HOME/.coworker/state/$ALIAS-hold"
say(){ echo "[host-hold $(date '+%H:%M:%S')] $*"; }

# Never fail silently. A hold that quietly stopped working looks exactly like a hold that is
# working, and the failure it prevents (a stolen card) shows up hours later as a fold_failed
# record with no hint of the cause.
run(){ timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$FLEET_HOST" "$1"; }

case "${1:-refresh}" in
  refresh)
    if run "mkdir -p \$HOME/.coworker/state && echo \$(( \$(date +%s) + $TTL )) > $HOLD"; then
      say "$ALIAS held on $FLEET_HOST for ${TTL}s"
    else
      say "WARNING: could not reach $FLEET_HOST -- $ALIAS is NOT held and the fleet may take"
      say "  a card between folds. Fix ssh $FLEET_HOST or set ABAG_XM_FLEET_HOST."
      exit 1
    fi ;;
  release)
    if run "rm -f $HOLD"; then say "$ALIAS released"; else say "WARNING: release failed"; exit 1; fi ;;
  status)
    run "if [ -f $HOLD ]; then echo \"held until \$(date -d @\$(cat $HOLD) '+%F %T') (\$(( \$(cat $HOLD) - \$(date +%s) ))s left)\"; else echo 'no hold'; fi" ;;
  *) echo "usage: $0 {refresh|release|status}" >&2; exit 2 ;;
esac
