#!/bin/bash
# disto_score.py hardcodes ARMS=["def","hifi"] and the 9bk6 anchor has three arms, so score
# it as two pairs. The lever's margin is B vs A' (both padded); A' vs A is the pad's own effect.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT
A=perf/obfused/anchor/9bk6
pair(){   # <name> <def-arm> <hifi-arm>
  P=$A/pair_$1; rm -rf $P; mkdir -p $P
  ln -s ../$2 $P/def; ln -s ../$3 $P/hifi
  echo "=== pair $1: def=$2 hifi=$3"
  PYTHONPATH=$WT $PY perf/fused_sdpa/disto_score.py --anchor 9bk6_164 --dir $P \
      --out perf/obfused/anchor_$1.json
}
pair lever Ap B
pair pad   A  Ap
