#!/bin/bash
# Cards some live python process on this host is pinned to (TT_VISIBLE_DEVICES), one per line.
for p in $(pgrep -u "$USER" -f python); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^TT_VISIBLE_DEVICES=\(.\+\)$/\1/p'
done | tr ',' '\n' | sort -n | uniq -c
