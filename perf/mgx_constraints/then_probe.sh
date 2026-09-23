#!/bin/bash
# after the esmfold2-fast upstream reference: the same upstream torch model on the disulfide
# probe and its control, so an inert bond constraint on TT can be told apart from one the
# checkpoint itself ignores.
cd "$(dirname "$0")"
while pgrep -f "esmfold2_upstream.py esmfold2-fast" > /dev/null; do sleep 30; done
I=perf/mgx_constraints/inputs
for M in esmfold2 esmfold2-fast; do
  ./ref_esmfold2.sh "$M" "3 4 5 6 7 8 9 10 12 13 14 15 16 17 18 19 20 21 22 23 28 29 30 31 0" \
      $I/ss_probe.yaml $I/ss_probe_ctrl.yaml
done
echo PROBE_REF_DONE
