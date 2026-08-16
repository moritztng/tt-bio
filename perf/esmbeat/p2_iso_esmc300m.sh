#!/usr/bin/env bash
# esmc-300m read -18.6% against its committed baseline. Neither lever can reach ESMC (one is in
# esmfold2_runtime, the other is on TriangleMultiplication, which a language model does not build),
# so this asks the two questions that settle it: is the leg stable, and does origin/main read the
# same?
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
PROBE=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2-esmcprobe
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export ESM_ROOT=/home/ttuser/esm

for r in 1 2 3; do
  echo "########## branch rep $r ##########"
  PYTHONPATH="$WT" $PY -u scripts/perf_regression.py --model esmc-300m 2>&1 \
    | grep -E "^esmc-300m|^\[esmc-300m\]|GATE"
done

git worktree remove --force "$PROBE" 2>/dev/null
git fetch -q origin
git worktree add --detach "$PROBE" origin/main >/dev/null || exit 1
echo "probe HEAD: $(git -C "$PROBE" rev-parse --short HEAD)"
cd "$PROBE" || exit 1
for r in 1 2; do
  echo "########## origin/main rep $r ##########"
  PYTHONPATH="$PROBE" $PY -u scripts/perf_regression.py --model esmc-300m 2>&1 \
    | grep -E "^esmc-300m|^\[esmc-300m\]|GATE"
done
cd "$WT" || exit 1
git worktree remove --force "$PROBE"
echo "ESMC_ISO_DONE, probe removed: $(test -d "$PROBE" && echo NO || echo yes)"
