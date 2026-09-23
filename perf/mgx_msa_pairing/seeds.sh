#!/bin/bash
# seeds.sh <card> <batch> <target> <model> <seed>... : more BEFORE/AFTER seeds on one target, to tell a
# per-target shift from seed noise. Shares run.sh's claiming, so two chips can split one list.
C=$1 B=$2 T=$3 M=$4; shift 4
R=$HOME/wt-mgx-msa-pairing; MAIN=$HOME/wt-mgx-msa-pairing-main
D=$R/perf/mgx_msa_pairing
for S in "$@"; do
  $D/run.sh $C $MAIN $B/$M-before-s$S $M $S --msa_cache_only $D/inputs/$T.yaml
  $D/run.sh $C $R $B/$M-after-s$S $M $S --msa_cache_only $D/inputs/$T.yaml
done
echo SEEDS_DONE $(date -u +%FT%TZ)
