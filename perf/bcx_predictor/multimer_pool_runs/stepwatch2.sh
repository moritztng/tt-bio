#!/bin/bash
# Log BindCraft 2's own step counter off the live frame with clock and load until step 6 or exit.
P=$1; OUT=$2
while [ -d /proc/$P ]; do
  d=$(/home/ttuser/.local/bin/py-spy dump --pid $P --locals 2>/dev/null)
  u=$(echo "$d" | grep -m1 "sequence_updates:" | awk '{print $2}')
  l=$(echo "$d" | grep -m1 " design_loss:" | awk '{print $2}')
  clk=$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c "
import json,sys
for dev in json.load(sys.stdin)['device_info']:
    if dev['board_info'].get('bus_id')=='0000:03:00.0': print(dev['telemetry']['aiclk'].strip(), dev['telemetry']['power'].strip())" 2>/dev/null)
  echo "$(date -u +%s) upd=$u loss=$l aiclk_pwr=[$clk] load=$(cut -d' ' -f1 /proc/loadavg)" >> $OUT
  [ -n "$u" ] && [ "$u" -ge 6 ] && break
  sleep 20
done
echo "END $(date -u +%s) alive=$([ -d /proc/$P ] && echo 1 || echo 0)" >> $OUT
