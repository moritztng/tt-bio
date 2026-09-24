#!/bin/bash
# first-parent bisect of af2ig's device digest: good=965ff9ae4 digest, bad=f374c883e digest
W=~/.coworker/wt/bcx-land; D=$W/perf/bcx_land/af2ig/bisect; mkdir -p $D
GOOD=12df3a0617cd72ef
mapfile -t C < <(git -C $W rev-list --first-parent --reverse 965ff9ae4..f374c883e)
lo=-1; hi=$((${#C[@]}-1))   # C[hi] known bad (f374c883e), lo=-1 means 965ff9ae4 good
while [ $((hi-lo)) -gt 1 ]; do
  mid=$(((lo+hi)/2)); c=${C[$mid]}; t=/tmp/bcx-land-bis
  rm -rf $t; mkdir $t; git -C $W archive $c | tar -x -C $t; echo $c > $t/.commit
  cd $t && PYTHONPATH=$t TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-land \
    timeout -s INT -k 90 600 /home/ttuser/tt-bio-dev/env/bin/python $W/perf/bcx_land/af2ig_digest.py --out $D/${c:0:9}.json > $D/${c:0:9}.log 2>&1
  rc=$?; dg=$(grep -o "DIGEST [0-9a-f]*" $D/${c:0:9}.log | cut -d" " -f2)
  echo "$mid ${c:0:9} rc=$rc digest=$dg"
  if [ "$dg" = "$GOOD" ]; then lo=$mid; elif [ -n "$dg" ]; then hi=$mid; else echo "no digest, stop"; tail -5 $D/${c:0:9}.log; exit 1; fi
done
echo "FIRST BAD ${C[$hi]} (last good ${C[$lo]:-965ff9ae4})"
git -C $W log -1 --oneline ${C[$hi]}
