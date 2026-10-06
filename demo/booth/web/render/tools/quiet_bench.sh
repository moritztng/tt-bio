#!/usr/bin/env bash
# waits until no other program uses the iGPU (foreign firefox / wf-recorder / sway-exp clients), then benches
cd /home/ttuser/.coworker/wt/booth-render/demo/booth/web/render
for i in $(seq 1 360); do
  busy=$(pgrep -af "[f]irefox|[w]f-recorder" | grep -v boothsway/ffprofile | wc -l)
  if [ "$busy" = 0 ]; then
    q=$((q+1)); [ "$q" -ge 6 ] && break
  else q=0; fi
  sleep 10
done
echo "quiet after $i checks at $(date -u +%FT%TZ), load $(cut -d" " -f1-3 /proc/loadavg)"
tools/bench.sh fixtures/9d3j-397.json 20 3840x2160 "" 1920x1080 ""
tools/bench.sh fixtures/prot-117.json 20 3840x2160 "" 1920x1080 ""
echo "done $(date -u +%FT%TZ), load $(cut -d" " -f1-3 /proc/loadavg); foreign GPU clients during run:"; pgrep -af "[f]irefox|[w]f-recorder" | grep -v boothsway/ffprofile | cut -c1-80
