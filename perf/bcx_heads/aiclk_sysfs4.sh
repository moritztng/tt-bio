#!/usr/bin/env bash
# sysfs AICLK of node 0 (TT_VISIBLE_DEVICES=3) and node 3 (TT_VISIBLE_DEVICES=2) every 4 s, until
# /dev/shm/bcx-heads-clk.stop exists.
while [ ! -e /dev/shm/bcx-heads-clk.stop ]; do
  echo "$(date -u +%T) $(cat "/sys/class/tenstorrent/tenstorrent!0/tt_aiclk") $(cat "/sys/class/tenstorrent/tenstorrent!3/tt_aiclk") $(cut -d" " -f1 /proc/loadavg)"
  sleep 4
done
