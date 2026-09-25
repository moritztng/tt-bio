#!/bin/bash
# After the candidate arms: save their CIFs, run the same arms on origin/main, then the
# af2ig-trunk-device leg of full_parity_gate.py on both trees. Card 3 only, one process at a time.
W=/home/ttuser/.coworker/wt/bcx-land; G=$W/perf/bcx_land/gate
until grep -q "^boltz2 rc=" $G/cand/rc.txt 2>/dev/null; do sleep 20; done
for m in rf3 boltz2; do mkdir -p $G/cand/$m; cp -r $W/${m}_results_prot/structures $W/${m}_results_prot/results.json $G/cand/$m/ 2>/dev/null; done
$W/perf/bcx_land/run_gate.sh /tmp/bcx-land-main main openfold3 rf3 boltz2
for m in openfold3 rf3 boltz2; do mkdir -p $G/main/$m; cp -r /tmp/bcx-land-main/${m}_results_prot/structures /tmp/bcx-land-main/${m}_results_prot/results.json $G/main/$m/ 2>/dev/null; done
for t in cand:$W main:/tmp/bcx-land-main; do tag=${t%%:*}; tree=${t#*:}; cd $tree
  PYTHONPATH=$tree TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-land timeout 2400 \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/full_parity_gate.py --leg af2ig-trunk-device --workers localhost:3 \
    --workdir /tmp/bcx-land-fpg-$tag --out $G/$tag/af2ig-trunk-device.json --load-ceiling 10 > $G/$tag/af2ig.log 2>&1
  echo "af2ig rc=$?" >> $G/$tag/rc.txt
done
echo ALLDONE >> $G/main/rc.txt
