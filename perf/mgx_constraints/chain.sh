#!/bin/bash
# chain.sh <card> <model> ...: the constraint set then the base cell for each model, in turn.
set -u
cd "$(dirname "$0")"
C=$1; shift
I=inputs
for M in "$@"; do
  ./run.sh "$M" "$C" cons --diffusion_samples 5 -- $I/sfti_cyclic_ss.yaml $I/sfti_cyclic.yaml \
      $I/sfti_ss.yaml $I/sfti_linear.yaml $I/cyclic.yaml $I/bond_ligand.yaml \
      $I/bond_protein_cys.yaml $I/modification.yaml
  ./run.sh "$M" "$C" base -- $I/base.yaml
done
echo CHAIN_DONE
