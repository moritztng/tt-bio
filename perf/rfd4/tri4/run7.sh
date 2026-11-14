#!/bin/bash
# tri4 run7: docs/rfd4-design.md's per-family cost table on this tree, main's default set (cand) vs this branch's
# default (dflt), through perf/rfd4/small/dev5.sh.   perf/rfd4/tri4/run7.sh <card>
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4
CAND=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,pair_bias_rms,pair_write_res \
HOLDER=worker:rfd4-perf-tri4 exec $W/perf/rfd4/small/dev5.sh ${1:?card} $W $W/artifacts/tri4/run7 $HOME/rfd4-oracle-inputs
