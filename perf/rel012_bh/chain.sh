#!/bin/bash
# The pc card 0 order: the ceiling walk (already running), then the memory-mode matrix at the
# rungs that refused, then the paired 288-token round against v0.11.0. One card, one arm at a
# time: a round timed while another process holds the same card is not a round time.
wt=/home/moritz/.coworker/wt/rel012-verify-bh
bash $wt/perf/rel012_bh/modes.sh
bash $wt/perf/rel012_bh/round.sh
