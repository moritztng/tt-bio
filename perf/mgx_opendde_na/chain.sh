#!/bin/bash
# chain.sh <card> <model>... runs this row's calls for each model on one chip, in order.
C=$1; shift
R=$HOME/wt-mgx-opendde-na; D=$R/perf/mgx_opendde_na; I=$D/inputs; MI=$R/perf/mgx_opendde_na/inputs
MX=$HOME/wt-mgx-matrix/perf/mgx_matrix/inputs
for M in "$@"; do
  if [ "$M" = protenix-v2 ]; then
    $D/run.sh $C $R $M-s0 $M 0 $I/1lmb.yaml $I/1urn.yaml
    $D/run.sh $C $R $M-s1 $M 1 $I/1lmb.yaml $I/1urn.yaml
    continue
  fi
  $D/run.sh $C $HOME/wt-mgx-opendde-na-main $M-main-s0 $M 0 $MX/base.yaml $MX/ligand_ccd.yaml
  $D/run.sh $C $R $M-s0 $M 0 $MX/base.yaml $MX/ligand_ccd.yaml $MX/rna.yaml $MX/dna.yaml $MX/rna_only.yaml $I/1lmb.yaml $I/1urn.yaml
  $D/run.sh $C $R $M-s1 $M 1 $I/1lmb.yaml $I/1urn.yaml
done
echo CHAIN_DONE $(date -u +%FT%TZ)
