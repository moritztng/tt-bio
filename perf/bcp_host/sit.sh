#!/bin/bash
# bcp-host sitting: N=1 (the size where auto resolves to 1) at 288 and ~512 tokens, alternated.
cd "$(dirname "$0")"
for tag in s288a s512a s288b s512b; do
  b=146; case $tag in s512*) b=370;; esac
  ./arm.sh $tag 7 1 --binder $b > out_$tag.log 2>&1 || echo "arm $tag rc=$?"
done
echo SIT_DONE
