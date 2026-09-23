#!/bin/bash
# The 640-token repeat set, one chip, in order: fixed build in-process x10, fixed build fresh x10,
# main in-process x5 (main's fresh-process spread is already on record from mgx-combos).
# $1 = the fixed tree, $2 = the main tree, $3 = card pool.
set -u
F=$1 M=$2 P=${3:-}
Y=perf/mgx_combos/inputs/mdh_640.yaml
"$F/perf/mgx_det/run.sh" inproc 10 "$Y" fix640_inproc "$P"
"$F/perf/mgx_det/run.sh" fresh 10 "$Y" fix640_fresh "$P"
"$M/perf/mgx_det/run.sh" inproc 5 "$Y" main640_inproc "$P"
