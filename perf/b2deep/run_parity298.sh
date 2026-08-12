#!/bin/bash
# The monomeric parity control. cdk2x2_298 is pure CDK2 (PDB 1HCL), ONE real domain -- unlike
# cdk2x2_512, which is CDK2 fused to a truncated second copy of itself and whose dominant degree of
# freedom is an unconstrained hinge between the two pseudo-domains. Every non-bit-exact arm measured
# at 512 (S1, L1, L2, both) lands at 7.7-9.0 A global RMSD purely by flipping that hinge, so the 512
# fixture cannot decide whether a lever perturbs the STRUCTURE. At 298 there is no hinge, so the
# global all-atom RMSD is a valid parity metric again.
WT=/home/ttuser/.coworker/wt/boltz2-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-512aa-deep-perf PYTHONPATH=$WT
export BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=300
/home/ttuser/.coworker/scripts/benchlock.sh boltz2-512aa-deep-perf -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 298 --arms on,s3,l1,both,on,both,l1,s3,on \
    --out perf/b2deep/parity298.json
echo "RC=$?"
