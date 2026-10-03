#!/bin/sh
# The login session GDM starts for the booth. sway runs in a loop, so a compositor crash comes
# back to the demo rather than ending the session at a login screen.
export XDG_CURRENT_DESKTOP=sway XDG_SESSION_DESKTOP=sc26
while :; do
  sway -c @CFG@/sway.config >>"$HOME/sc26-logs/sway.log" 2>&1
  echo "$(date -Is) sway exited rc=$?" >>"$HOME/sc26-logs/sway.log"
  sleep 2
done
