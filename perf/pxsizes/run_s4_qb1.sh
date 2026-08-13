#!/bin/bash
# Session 4 (qb1): the two missing curve points, 768 and 1024 aa, taken in-process.
#
# Why this is possible here and was not on qb2. Section 2.2 records 1024 as "not obtainable
# in-process" because the second fold in a process stalls the device. Section 10.2 MEASURED that the
# stall does not reproduce on qb1: two 768 aa folds in one process returned, 227.14 s then 155.44 s,
# same plDDT, same 604 calls. So the multi-arm interleaved run that qb2 cannot do, qb1 can, and the
# 768 and 1024 rows of the base4/on curve become takeable.
#
# What it is NOT. qb1 is Blackhole p150a / ttnn 0.67.4 / 13x10; qb2 is P300c / ttnn 0.68.0 / 11x10.
# These two rows are a qb1-INTERNAL curve. They do not extend the 128/256/512/640 table, which is
# qb2, and they must never be plotted on the same axis. Ratios are within-host and each size's arms
# share one process, one device context, one fixture and one weight load, so the ratio is sound even
# though the absolute seconds are not comparable across hosts.
#
# Arms on,on,on,base4 (NOT on,on,base4). --skip-cold makes arm 1 the cold fold, so arm 1 is
# cold-vs-warm and CANNOT serve as an A/A partner. Arms 2 and 3 are both warm `on`: they are the
# real A/A floor, and arm 4 is the base4 leg the ratio is taken against. Section 10.3 is why this
# matters: the cold cost is 71.70 s at 768 and is NOT the additive constant 7.1.1 assumed, so any
# ratio that puts a cold fold on one side of it is compressed by an unknown amount.
#
# Benchlocked, one hold per size so the box is released between them. Last pass's 768 numbers were
# co-tenanted and explicitly labelled indicative; these supersede them for timing.
#
# Order is 768 first ON PURPOSE: it is the cheaper leg (~12 min vs ~19), it banks a complete curve
# point with its own A/A floor, and 1024 is the leg that might hit qb1's crash band. The harness
# writes results after every fold, so a 1024 leg that dies part-way still lands its cold fold, which
# is the capacity answer the brief asks for.
#
# REGISTERED PREDICTIONS (committed before the run; do not edit after it starts):
#   R1  768 warm `on` = 155.44 s +/- 5 s, against last pass's co-tenanted 155.44 s. Benchlocked
#       should land equal or slightly faster. A miss by more than 5 s means co-tenancy was not the
#       only difference and the two passes' 768 numbers must not be pooled.
#   R2  A/A floor at 768 (|on2 - on3|) < 1.0 s. MEASURED 0.052 s at 128 and 0.168 s at 256; scaling
#       by fold wall puts 768 under a second.
#   R3  768 base4/on = 1.05-1.12x. The qb2 curve is monotone decreasing (1.2941 / 1.2463 / 1.1988 /
#       1.1223 at 128 / 256 / 512 / 640) and 768 extrapolates to ~1.08-1.11. Caveat that makes this
#       a genuine prediction and not an extrapolation: E6 serves 0 at 768 on this grid (10.4), and
#       base4 turns E6 off, so E6 cancels between the arms and the ratio here is K1 + K1b + K2 only.
#   R4  1024 cold fold COMPLETES, 320-420 s. qb1 did 768 cold in 227.14 s; qb2's own 768->1024 cold
#       ratio is 315.09/199.63 = 1.578, which puts qb1 at ~358 s.
#   R5  1024 fold 2 RETURNS, extending 10.2 to the size where qb2 stalled worst (78 min). If it
#       stalls here, the stall IS size-dependent on qb1 too and its boundary sits in (768, 1024].
#   R6  Cold cost at 1024 > 71.70 s, the 768 value, per 10.3's "grows with N". Predict 85-150 s.
#   R7  1024 base4/on < the 768 ratio, continuing the monotone decay. Predict 1.02-1.10x.
#   R8  E6 serves 0 at 1024 as it does at 768. If it serves > 0, the gate is non-monotone in N and
#       10.4's decay mechanism needs re-stating.
#   Stop rule: if a size's A/A floor exceeds one third of its (base4 - on) gap, that size's ratio is
#   reported as NOT RESOLVED, not as a number.
#   Stop rule: if the 1024 cold fold dies, the capacity verdict is named from its TT_THROW and no
#   1024 ratio is quoted at all.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/pxsizes

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  start: $(date -Is) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_{768,1024}.{yaml,a3m}

echo "=== 768 leg $(date -Is) ==="
$BL protenix-v2-sizes-perf -- timeout 1500 $PY -u perf/size512/fold_ab512.py \
    --sizes 768 --arms on,on,on,base4 --skip-cold --timers full --out $O/s4_768_qb1.json
echo "RC_768=$?"

echo "=== 1024 leg $(date -Is) ==="
$BL protenix-v2-sizes-perf -- timeout 2400 $PY -u perf/size512/fold_ab512.py \
    --sizes 1024 --arms on,on,on,base4 --skip-cold --timers full --out $O/s4_1024_qb1.json
echo "RC_1024=$?"
echo "=== done: $(date -Is) ==="
