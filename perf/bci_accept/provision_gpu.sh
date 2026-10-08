#!/bin/bash
# bci-seventeen: provision the rented GPU box for the HOST-JAX arm of #17.
#
# Runs on pc, which already holds exactly what the arm needs: bcx_shipped/bc2 (BindCraft 2 v1.0.1
# with its ProteinMPNN weights) and a 1.8 GB af2_params, the same files bci-accept's host arms
# used. Shipping pc's copy rather than qb1's 5.3 GB superset keeps weight parity with the prior
# host arm and moves a third of the bytes.
set -eu
H=root@ssh4.vast.ai
P=27702
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P"
LOG=/home/moritz/.bci-seventeen-host/.bci/provision.log
exec >> "$LOG" 2>&1
date -u +"=== provision start %Y-%m-%dT%H:%M:%SZ ==="

$SSH $H 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; df -h / | tail -1'

echo "--- bc2 + af2 params ---"
rsync -a --info=progress2 -e "$SSH" \
  /home/moritz/bcx_shipped/ $H:/root/bcx_shipped/
echo "--- tt-bio host path, this branch ---"
rsync -a -e "$SSH" --exclude .git --exclude '*.pyc' \
  /home/moritz/.bci-seventeen-host/tt-bio/ $H:/root/tt-bio/

echo "--- jax[cuda12] ---"
$SSH $H 'pip install -q --no-input "jax[cuda12]" && python -c "import jax;print(\"jax\",jax.__version__,jax.devices())"'
echo "--- bc2 deps ---"
$SSH $H 'pip install -q --no-input biopython pandas scipy matplotlib seaborn tqdm pdbfixer 2>&1 | tail -3; true'

date -u +"=== provision end %Y-%m-%dT%H:%M:%SZ ==="
