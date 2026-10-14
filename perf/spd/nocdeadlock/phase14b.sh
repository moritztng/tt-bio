#!/bin/bash
# Phase 14b (2026-10-14, spd-nocdeadlock2), replaces phase14.sh. After phase 13: proof soak s8 on main tree <t> (fix, no
# forced grid) on chips 2+3 (board 4103) exact x20 + fast x20; meanwhile phase 15 finds chip 0's AICLK cap, then s8 on
# control chip 1 and soak s9 on chip 0 capped through tt-bio's own AICLK_CAP path (tree <capt>, main with the cap code).
# Usage: phase14b.sh <t> <capt>. Index phase14.log; per-group soak_s8_23.idx, soak_s8_1.idx, soak_s9.idx.
cd ~/spd/spd-nocdeadlock; t=$1; ct=$2
until grep -q P13-DONE phase13.log 2>/dev/null; do sleep 60; done
echo "phase13 done, s8 on $t ($(cat $t/.tree_sha)) chips 2,3 + phase15 on chip 0 $(date -u +%FT%TZ)"
./soak.sh s8 none 2=$t:exact,$t:fast 3=$t:exact,$t:fast > soak_s8_23.idx 2>&1 & p23=$!
./phase15.sh > phase15.log 2>&1
echo "$(grep P15-CAP phase15.log), s8 control chip 1 + s9 chip 0 on $ct ($(cat $ct/.tree_sha)) $(date -u +%FT%TZ)"
./soak.sh s8 none 1=$t:exact,$t:fast > soak_s8_1.idx 2>&1 & p1=$!
grep -q "P15-CAP none" phase15.log || ./soak.sh s9 none 0=$ct:exact,$ct:fast > soak_s9.idx 2>&1
wait $p1 $p23; echo "P14-DONE $(date -u +%FT%TZ)"
