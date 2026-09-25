#!/bin/bash
# Log BindCraft 2's own step counter off the live frame, with clock and load, until 3 more steps land.
OUT=/home/ttuser/bcx_multimer_art/pool_l96_steps.txt
start=""
for i in $(seq 1 90); do
  d=$(/home/ttuser/.local/bin/py-spy dump --pid 23215 --locals 2>/dev/null)
  [ -z "$d" ] && { echo "$(date -u +%s) PROCESS-GONE" >> $OUT; break; }
  u=$(echo "$d" | grep -m1 "sequence_updates:" | awk '{print $2}')
  l=$(echo "$d" | grep -m1 " design_loss:" | awk '{print $2}')
  clk=$(~/.local/bin/tt-smi -s 2>/dev/null | python3 -c "
import json,sys
for dev in json.load(sys.stdin)['device_info']:
    if dev['board_info'].get('bus_id')=='0000:03:00.0': print(dev['telemetry']['aiclk'].strip(), dev['telemetry']['power'].strip())" 2>/dev/null)
  echo "$(date -u +%s) upd=$u loss=$l aiclk_pwr=[$clk] load=$(cut -d' ' -f1 /proc/loadavg)" >> $OUT
  [ -z "$start" ] && start=$u
  [ -n "$u" ] && [ "$u" -ge $((start+3)) ] && break
  sleep 20
done
echo DONE >> $OUT
