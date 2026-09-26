#!/bin/bash
# Sample tt_aiclk off the sysfs class node for one card, n samples at a fixed interval.
# Reads telemetry only, so it is safe to run against a card another process is computing on.
card=${1:-1}; n=${2:-60}; iv=${3:-1}
node=/sys/class/tenstorrent/tenstorrent!${card}/tt_aiclk
for ((i=0;i<n;i++)); do
  printf "%s %s\n" "$(date -u +%FT%T.%3NZ)" "$(cat "$node")"
  sleep "$iv"
done
