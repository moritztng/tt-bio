#!/usr/bin/env bash
# 65 min of the page through a shaped link: 3mbit/40ms/1%, a 60 s cut at 20 min, 1mbit/80ms/2% from
# 30 to 40 min, a 90 s cut at 50 min. Logs link bytes, engine RSS each minute, a still every 5 min.
set -u
D=/tmp/bthsoak OPS=/home/ttuser/.coworker/wt/bth-stream/demo/booth/ops/thin_link.sh
APP=/home/ttuser/.coworker/wt/bth-stream/demo/booth/web/app/tools/look.sh
URL="http://10.77.0.2:8637/app/?wire=1"
bash $OPS shape 3mbit 40ms 1%
(LOOK_TAG=bths bash $APP rec 1920x1080 "$URL" $D/soak.mp4 3900 > $D/look.log 2>&1; echo "rec ended $(date -u +%T)" >> $D/events.log) &
t0=$(date +%s); echo "start $(date -u +%T)" > $D/events.log
ev() { echo "$(date -u +%T) +$(( ($(date +%s)-t0)/60 ))min $*" >> $D/events.log; }
for m in $(seq 0 66); do
  pid=$(pgrep -f "server.py --chips 0,1,2,3 --worker engine/tests/recorded_worker.py --port 8637" | head -1)
  echo -e "$(date -u +%T)\t$m\t$(bash $OPS bytes)\t$(ps -o rss= -p $pid 2>/dev/null)" >> $D/link.tsv
  [ $((m % 5)) = 2 ] && XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-9bths grim $D/still-$(printf %02d $m).png 2>/dev/null
  case $m in
    20) ev cut; bash $OPS shape 3mbit 40ms 100%; sleep 60; bash $OPS shape 3mbit 40ms 1%; ev restored; continue ;;
    30) ev 1mbit; bash $OPS shape 1mbit 80ms 2% ;;
    40) ev 3mbit; bash $OPS shape 3mbit 40ms 1% ;;
    50) ev cut90; bash $OPS shape 3mbit 40ms 100%; sleep 90; bash $OPS shape 3mbit 40ms 1%; ev restored; continue ;;
  esac
  sleep 60
done
wait
grep -a "wire \|error\|rror" $HOME/boothbths/firefox.log > $D/wire.log
ev done
