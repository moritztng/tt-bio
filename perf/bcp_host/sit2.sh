#!/bin/bash
# bcp-host sitting 2, 512 tokens: N=1 (what auto picks) against N=2 interleaved, alternated.
cd "$(dirname "$0")"
for tag in t1a t2a t1b t2b; do
  case $tag in t1*) w=1; r=7;; t2*) w=i2; r=7;; esac
  ./arm.sh $tag $r $w --binder 370 > out_$tag.log 2>&1 || echo "arm $tag rc=$?"
done
echo SIT_DONE
