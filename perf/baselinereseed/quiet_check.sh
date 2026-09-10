#!/bin/bash
# No draw is recorded unless this prints QUIET.
holders=$(lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u | tr "\n" " ")
la=$(cut -d" " -f1 </proc/loadavg)
mine=$$
other=""
for p in $holders; do
  case " $(ps -o pid= --ppid $mine 2>/dev/null) " in *" $p "*) continue;; esac
  other="$other $p($(ps -o comm= -p $p 2>/dev/null))"
done
if [ -n "$other" ]; then
  echo "BUSY loadavg=$la device-holders:$other"
  exit 1
fi
echo "QUIET loadavg=$la no /dev/tenstorrent holders"
