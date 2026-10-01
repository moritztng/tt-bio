#!/bin/bash
# After chain.sh: 288 whole-query bit-exactness, the pre-loop kernel (73ea0acbf) against this tree.
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; log=$out/chain.log
while pgrep -f "perf/bcw_dbias/chain[.]sh" > /dev/null; do sleep 60; done
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcw-dbias
rm -rf /tmp/bcw_old && mkdir -p /tmp/bcw_old && git archive 73ea0acbf tt_bio | tar -x -C /tmp/bcw_old
echo "=== $(date -u +%FT%TZ) bitexact288" >> $log
for tree in old new; do
  root=$PWD; [ $tree = old ] && root=/tmp/bcw_old
  BCW_ROOT=$root timeout 900 ~/tt-bio-dev/env/bin/python perf/bcw_dbias/grade.py --b 288 --n 288 \
     --no-fallback --save /tmp/bcw_be_$tree.pt --out $out/bitexact288_$tree.json > $out/bitexact288_$tree.log 2>&1
  echo "rc=$? $tree" >> $log
done
~/tt-bio-dev/env/bin/python - >> $log 2>&1 <<'PY'
import torch, json
a, b = torch.load("/tmp/bcw_be_old.pt"), torch.load("/tmp/bcw_be_new.pt")
r = {n: {"bit_identical": bool(torch.equal(a[n], b[n])), "max_abs": float((a[n]-b[n]).abs().max())} for n in a}
print(json.dumps(r)); json.dump(r, open("perf/bcw_dbias/out/bitexact288.json", "w"), indent=1)
PY
echo "=== FOLLOWUP DONE $(date -u +%FT%TZ)" >> $log
