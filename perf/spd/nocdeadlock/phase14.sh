#!/bin/bash
# Phase 14 (2026-10-14, spd-nocdeadlock2): proof soak on a main sha that carries the deadlock fix. Waits for phase 13
# (slot test) to finish, then c730 Protenix-v2 exact x20 then fast x20 on chips 2 and 3 (board 4103, slot 2) and chip 1
# (control, board 410D, slot 1), in parallel. Usage: phase14.sh <tree dir> ; index phase14.log, soak s8.
cd ~/spd/spd-nocdeadlock; t=$1
until grep -q P13-DONE phase13.log 2>/dev/null; do sleep 60; done
echo "phase13 done, soak s8 on $t ($(cat $t/.tree_sha)) $(date -u +%FT%TZ)"
./soak.sh s8 none 2=$t:exact,$t:fast 3=$t:exact,$t:fast 1=$t:exact,$t:fast
echo "P14-DONE $(date -u +%FT%TZ)"
