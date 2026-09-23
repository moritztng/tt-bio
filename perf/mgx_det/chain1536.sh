#!/bin/bash
# The 1536-bucket repeat set on the fixed build, two chips at once: eal_1536 (3ABQ a2b2 with its
# crystal as a two-copy template and two ethanolamines, 1526 tokens) in-process x10 beside fresh x10.
# $1 = the fixed tree.
set -u
F=$1
Y=perf/mgx_combos/inputs/eal_1536.yaml
"$F/perf/mgx_det/run.sh" inproc 10 "$Y" fix1536_inproc &
sleep 90
"$F/perf/mgx_det/run.sh" fresh 10 "$Y" fix1536_fresh &
wait
