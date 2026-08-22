#!/usr/bin/env bash
# Pass 7's timed sequence, one benchlock hold, qb1 card 2. Highest-value leg first: the two host
# template arms settle whether pass 6's split was measured at the production dtype, the 848 fold is
# item 1's definitive re-take, and `template_split --device` prices item 2's lever. `fold_timing`'s
# own split reports the device stacks per pass, so no separate `trunk_timing` cross-check is needed.
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-perf-p7
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p7
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs

say() { echo "=== $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) $* ==="; }
run() { say "START $1"; shift; timeout 2400 "$@"; echo "rc=$?"; }

say "1/7 template split 848, host, PRODUCTION dtype (model trunk_dtype = bfloat16)"
PYTHONPATH="$WT" timeout 900 "$PY" scripts/af2_port/template_split.py --pdb /tmp/p7_768.pdb \
    --reps 3 --dtype model --out perf/pxdesign/tt_pxd_p7_template_848_bf16.json \
    > logs/p7_template_bf16.out 2>&1; echo "rc=$?"

say "2/7 template split 848, host, float32 (pass 6's arm, reproduced)"
PYTHONPATH="$WT" timeout 900 "$PY" scripts/af2_port/template_split.py --pdb /tmp/p7_768.pdb \
    --reps 3 --dtype float32 --out perf/pxdesign/tt_pxd_p7_template_848_fp32.json \
    > logs/p7_template_fp32.out 2>&1; echo "rc=$?"

say "3/7 fold_timing 848, reps 3 -- THE definitive 848 leg (item 1)"
PYTHONPATH="$WT" timeout 2400 "$PY" scripts/af2_port/fold_timing.py --pdb /tmp/p7_768.pdb \
    --reps 3 --out perf/pxdesign/tt_pxd_p7_fold_848.json > logs/p7_fold_848.out 2>&1; echo "rc=$?"

say "4/7 template pair stack ON CARD, 848 (item 2's cost screen)"
PYTHONPATH="$WT" timeout 1800 "$PY" scripts/af2_port/template_split.py --pdb /tmp/p7_768.pdb \
    --reps 3 --dtype model --device \
    --out perf/pxdesign/tt_pxd_p7_template_device_848.json \
    > logs/p7_template_device.out 2>&1; echo "rc=$?"

for T in 512 256 128; do
  say "rung fold_timing /tmp/p7_$T.pdb, reps 2 (item 4's ladder)"
  PYTHONPATH="$WT" timeout 1800 "$PY" scripts/af2_port/fold_timing.py --pdb /tmp/p7_$T.pdb \
      --reps 2 --out perf/pxdesign/tt_pxd_p7_fold_$T.json > logs/p7_fold_$T.out 2>&1
  echo "rc=$?"
done

say "chain done"
