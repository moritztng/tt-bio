#!/bin/bash
# Session 3 (qb1): does the second-fold device stall reproduce on a DIFFERENT board and ttnn?
#
# Why this run exists, and why it is not the S5 curve the plan asked for. qb2 died between passes
# (WiFi only, ethernet unplugged so no WoL, no BMC — unreachable from the laptop and from qb1 on the
# LAN). Every device figure in this task was measured on qb2 card 3, and the four queued 768/1024
# folds died with it. A 768/1024 timing measured here is NOT a curve point against those: qb1 is
# Blackhole p150a / ttnn 0.67.4 / 13x10, qb2 is P300c / ttnn 0.68.0 / 11x10, and mixing grids inside
# one comparison is exactly how the reblock_permute window became a 0.62x loss on the other grid.
#
# But one question is answered BETTER here than on qb2, and it is the open question §9.4 handed over:
# the second fold in a process stalls the device in the MSA stack above ~640 aa (78 min at 1024,
# 11.5 min at 768, py-spy blocked in ttnn.synchronize_device, zero TT_THROWs). On one host that is
# indistinguishable from a P300c or ttnn-0.68.0 artifact. A second, independent stack discriminates
# it. That verdict is BINARY (returns / does not), so it survives the co-tenancy this run has.
#
# Instrument: --arms on,on --skip-cold. Fold 1 is the cold fold and IS arm 1; fold 2 is the second
# fold in the process, which is precisely the position that stalled on qb2.
#
# REGISTERED PREDICTIONS (written and committed before the run; do not edit after it starts):
#   Q1  Fold 1 (cold, 768 aa, qb1 card 1) returns. qb2 did it in 199.63 s; qb1 is a different board
#       so no time is predicted, only that it COMPLETES. If it does not, this run says nothing about
#       the stall and must be reported as inconclusive, not as a reproduction.
#   Q2  THE DISCRIMINATOR. Fold 2 either returns or it does not.
#         returns  -> the stall does NOT reproduce on qb1. It is then specific to the qb2 stack
#                     (P300c and/or ttnn 0.68.0), not a protenix/tt-bio defect, and §9.4's
#                     "no in-process multi-fold workload at 1024 aa completes today" must be
#                     narrowed to that host in the state doc.
#         does not -> the stall is a protenix/tt-bio defect that crosses board and ttnn version.
#                     That is the stronger and more expensive finding, and it makes the
#                     one-arm-per-process workaround permanent rather than a qb2 workaround.
#   Q3  Stall criterion, fixed in advance so it is not chosen after seeing the number: fold 2 is
#       declared STALLED if it exceeds 3x fold 1's wall AND a py-spy dump shows MainThread inside
#       ttnn.synchronize_device under the MSA stack. On qb2 fold 2 ran 3.5x and climbing at 768.
#       Anything under 3x that returns is NOT a stall, it is just a slow warm fold.
#   Q4  If fold 2 returns, it is ALSO this size's A/A partner on qb1 and the pair gives a qb1-local
#       A/A floor at 768 aa. Co-tenanted (boltz2-sizes-perf holds card 0), so that floor is an
#       UPPER bound on the noise, and any timing quoted from it carries that label.
#
# Not benchlocked ON PURPOSE: benchlock is held by boltz2-sizes-perf running a 4-arm 1024 fold, the
# Q2 verdict is binary and immune to co-tenant noise, and waiting would spend the whole pass to buy
# precision this question does not need. Every TIMING below is therefore labelled indicative.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-sizes-perf
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-sizes-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=perf/pxsizes

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  start: $(date -Is) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_768.yaml $WT/perf/size512/fixtures/cdk2x2_768.a3m

timeout 1500 $PY -u perf/size512/fold_ab512.py --sizes 768 --arms on,on --skip-cold \
    --timers full --out $O/s3_768_qb1_aa.json
echo "RC_768_onon=$?"
echo "=== done: $(date -Is) ==="
