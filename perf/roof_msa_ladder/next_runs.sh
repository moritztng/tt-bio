#!/bin/bash
# Chain: 298 aa A/B (second size, and the CIFs the accuracy control re-scores), then the
# A/A null at 512 aa — both arms shipped-setting, same interleave, so the harness reports the
# ratio it reads when there is no effect.
set -u
cd /home/ttuser/.coworker/wt/roof-msa-ladder-bh-ship
P=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:roof-msa-ladder-bh-ship TT_METAL_LOGGER_LEVEL=FATAL
$P -u perf/roof_msa_ladder/ladder_ab.py --out perf/roof_msa_ladder/ab_298_qb2.json \
   --cifdir perf/roof_msa_ladder/cif_qb2_298 --sizes 298 --pairs 3 \
   > perf/roof_msa_ladder/ab_298_qb2.log 2>&1
echo "298 rc=$?"
$P -u perf/roof_msa_ladder/ladder_ab.py --out perf/roof_msa_ladder/aa_512_qb2.json \
   --cifdir perf/roof_msa_ladder/cif_qb2_aa --sizes 512 --pairs 5 --aa \
   > perf/roof_msa_ladder/aa_512_qb2.log 2>&1
echo "aa rc=$?"
echo CHAIN_DONE
