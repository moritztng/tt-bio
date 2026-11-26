#!/bin/bash
# r4t-scale on anx-glx: the real RFD4 step (trunk_step.py, crop 256, nine upstream batches), one rank per Galaxy chip.
# usage: gx_trunk.sh CARDS OUT STEPS [extra args]   CARDS "0" = one chip, "0,1,2,3" = DP over 4 chips.
[ -e ~/fleet/HANDBACK ] && { echo HANDBACK; exit 4; }
C=$1; O=$2; N=$3; shift 3
W=~/fleet/r4t/r4t-scale; S=~/fleet/r4t/step; cd $W
B=$S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz,$S/batch03.npz,$S/batch06.npz,$S/batch02.npz,$S/batch04.npz,$S/batch07.npz
export PYTHONPATH=$W TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_LEASE_HOLDER=worker:r4t-scale TT_BIO_RFD4=~/fleet/r4t/rfd4w OMP_NUM_THREADS=${OMP:-16}
mkdir -p $O; echo "start $(date -u +%FT%TZ) cards $C load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O/run.log
if [[ $C == *,* ]]; then
  TT_BIO_LEASE_CARDS=$C timeout 3600 ~/fleet/env/bin/python perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B --exempt perf/r4t_step/g2_exempt.tsv --steps $N --dp-cards $C --out $O "$@" > $O/driver.log 2>&1
else
  TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C timeout 3600 ~/fleet/env/bin/python perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B --exempt perf/r4t_step/g2_exempt.tsv --steps $N --out $O/ck "$@" > $O/A.log 2>&1
fi
echo "rc=$? end $(date -u +%FT%TZ)" >> $O/run.log
