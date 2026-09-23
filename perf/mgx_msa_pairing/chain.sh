#!/bin/bash
# chain.sh <card> <batch> <targets> <model>... : BEFORE (main) and AFTER (this branch), seeds 0
# and 1, per model, into out/<batch>/. <targets> is a comma list of inputs/ stems. im9 (the
# monomer control) folds on seed 0 only and 1hvr (the homodimer, OpenDDE's trigger) on OpenDDE only.
# Two chips can share one batch (run.sh skips a run dir another chain created).
# First batch: chain.sh <card> . 1emv,1brs,1hvr,im9 <models>; second: chain.sh <card> b2 8pu1,8wt4 <models>.
C=$1 B=$2 T=$3; shift 3
R=$HOME/wt-mgx-msa-pairing; MAIN=$HOME/wt-mgx-msa-pairing-main
D=$R/perf/mgx_msa_pairing; I=$D/inputs
for M in "$@"; do
  BFLAGS=--msa_cache_only
  if [ "${M#opendde}" != "$M" ]; then
    # main's OpenDDE pairs only when it has a search source, and only its offline branch
    # reads a cached paired MSA; any existing path serves, nothing is searched.
    BFLAGS="--msa_db_path /tmp"
  fi
  for S in 0 1; do
    IN=""
    for t in ${T//,/ }; do
      [ $t = im9 ] && [ $S != 0 ] && continue
      [ $t = 1hvr ] && [ "${M#opendde}" = "$M" ] && continue
      IN="$IN $I/$t.yaml"
    done
    # shellcheck disable=SC2086
    $D/run.sh $C $MAIN $B/$M-before-s$S $M $S "$BFLAGS" $IN
    # shellcheck disable=SC2086
    $D/run.sh $C $R $B/$M-after-s$S $M $S --msa_cache_only $IN
  done
done
echo CHAIN_DONE $(date -u +%FT%TZ)
