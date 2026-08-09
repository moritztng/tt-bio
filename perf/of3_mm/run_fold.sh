#!/bin/sh
# One OF3 298 aa fold on qb1 card 1. $1 = tag, $2 = 1 to enable the matmul census.
WT=/home/ttuser/.coworker/wt/perfwar-of3-matmul-sites
TAG=$1
HOOK=""
[ "$2" = "1" ] && HOOK="OF3_MM_CENSUS=$WT/perf/of3_mm/census_$TAG"
mkdir -p $WT/perf/of3_mm/census_$TAG
S=$(date +%s.%N)
env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-of3-matmul-sites     OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt     PYTHONPATH=$WT/perf/of3_mm/hook:$WT $HOOK     /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict $WT/examples/prot300.yaml       --model openfold3 --single_sequence --override       --out_dir /home/ttuser/of3_e1/$TAG
RC=$?
E=$(date +%s.%N)
echo "FOLD_TAG=$TAG RC=$RC WALL_S=$(echo "$E - $S" | bc)"
