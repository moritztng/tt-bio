#!/bin/sh
# The login session GDM starts for the booth. sway runs in a loop, so a compositor crash comes
# back to the demo rather than ending the session at a login screen.
export XDG_CURRENT_DESKTOP=sway XDG_SESSION_DESKTOP=booth
# headless too, so display.sh can give sway a screen to draw on when none is plugged in.
export WLR_BACKENDS=drm,libinput,headless
# Between two sways a real screen shows the empty text console, black, so start the next one at
# once. Only a sway that dies within 10 s of starting waits 2 s, so a crash loop cannot spin.
while :; do
  start=$(date +%s)
  sway -c @CFG@/sway.config >>"$HOME/booth-logs/sway.log" 2>&1
  echo "$(date -Is) sway exited rc=$?" >>"$HOME/booth-logs/sway.log"
  [ $(($(date +%s) - start)) -ge 10 ] || sleep 2
done
