#!/bin/bash
# Post-reboot, quiet box: the A/A null first, then a repeat of the A/B. The headline 1.0566x was
# taken under sibling load 2.26-5.45; this pair says what the harness reads with no effect at all,
# and what the effect reads with no neighbours.
set -u
cd /home/ttuser/.coworker/wt/roof-msa-ladder-bh-ship
P=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:roof-msa-ladder-bh-ship TT_METAL_LOGGER_LEVEL=FATAL
$P -u perf/roof_msa_ladder/ladder_ab.py --out perf/roof_msa_ladder/aa_512_qb2.json \
   --cifdir perf/roof_msa_ladder/cif_qb2_aa --sizes 512 --pairs 5 --aa \
   > perf/roof_msa_ladder/aa_512_qb2.log 2>&1
echo "aa rc=$?"
$P -u perf/roof_msa_ladder/ladder_ab.py --out perf/roof_msa_ladder/ab_512_qb2_quiet.json \
   --cifdir perf/roof_msa_ladder/cif_qb2_quiet --sizes 512 --pairs 6 \
   > perf/roof_msa_ladder/ab_512_qb2_quiet.log 2>&1
echo "quiet rc=$?"
echo CHAIN2_DONE
