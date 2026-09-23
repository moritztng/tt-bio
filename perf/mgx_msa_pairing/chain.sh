#!/bin/bash
# chain.sh <card> <model>... : BEFORE (main) and AFTER (this branch), seeds 0 and 1, per model.
C=$1; shift
R=$HOME/wt-mgx-msa-pairing; MAIN=$HOME/wt-mgx-msa-pairing-main
D=$R/perf/mgx_msa_pairing; I=$D/inputs
for M in "$@"; do
  IN="$I/1emv.yaml $I/1brs.yaml"
  BFLAGS=--msa_cache_only
  if [ "${M#opendde}" != "$M" ]; then
    IN="$IN $I/1hvr.yaml"
    # main's OpenDDE pairs only when it has a search source, and only its offline branch
    # reads a cached paired MSA; any existing path serves, nothing is searched.
    BFLAGS="--msa_db_path /tmp"
  fi
  for S in 0 1; do
    X=""; [ $S = 0 ] && X=$I/im9.yaml
    $D/run.sh $C $MAIN $M-before-s$S $M $S "$BFLAGS" $IN $X
    $D/run.sh $C $R $M-after-s$S $M $S --msa_cache_only $IN $X
  done
done
echo CHAIN_DONE $(date -u +%FT%TZ)
