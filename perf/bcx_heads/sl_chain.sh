#!/usr/bin/env bash
# SL arm before and after the unified verbs, one card, sequential. The transplant sits in the
# stash of .of3t/se until the pre arm is done.
set -u
BH=/home/ttuser/.coworker/wt/bcx-heads
cd "$BH/.of3t/se"
[ -z "$(git status --porcelain tt_bio)" ] || { echo "se tree not clean before PRE"; exit 2; }
bash "$BH/perf/bcx_heads/sl_arm.sh" SL_PRE 3
git stash pop -q && git -c user.name=moritztng -c user.email=moritz.thuening@gmail.com \
  commit -qam "bcx-heads: the unified head verbs on of3t-stackexact (test tree)" && git log --oneline -1
bash "$BH/perf/bcx_heads/sl_arm.sh" SL_POST 3
cd "$BH/perf/bcx_heads/sl"
for p in "digest_SL_PRE.json digest_SL_POST.json" "digest_banked_SL.json digest_SL_PRE.json" "digest_banked_SL.json digest_SL_POST.json"; do
  python3 "$BH/perf/bcx_heads/ptdigest.py" --cmp $p
done | tee cmp.jsonl
