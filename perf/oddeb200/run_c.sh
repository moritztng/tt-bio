#!/bin/bash
# Run C (doc section 8.5): capacity at 640 aa and the cross-model byte-identity check, the two
# bars the host-glue lever has to clear before it can be proposed for merge. The glue touches
# `_generate_relp`, which protenix-v2 shares, so a fold that is byte-identical on OpenDDE says
# nothing about protenix-v2 until protenix-v2 is folded.
#
# One benchlock hold, two legs, each writing its JSON before the next starts.
#   leg 1  opendde 640 aa, arms on,glue      -> no allocator refusal, no CB clash, and the 640
#          ratio within ~0.5 % of the 512 ratio, which is what a lever serving the same calls at
#          both sizes should do
#   leg 2  protenix-v2 512 aa, arms on,glue,on -> CIF sha256 identical at the FULL 64-hex digest
#          on all three arms. The harness truncates its own digest to 16 hex
#          (fold_ab_multi.py:160), so the full digest is taken from the kept cif_dir afterwards.
#
# Section 8.5 wrote these arms as `traceglue`. The trace is NO-GO on parity as of pass 4 (only the
# capture fold is correct), so the arm under test is `glue`.
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$WT/perf/oddeb200

foreign () {
  echo "--- foreign fold check ($1) $(date -Is) ---"
  ps -eo pid,pcpu,etime,args | grep -Ei 'tt_baseline|fold_ab|protenix|boltz|opendde|esmfold|openfold' \
    | grep -v grep | grep -v benchlock | head -6
  echo "loadavg: $(cut -d' ' -f1-3 /proc/loadavg)"
}

echo "### run_c start $(date -Is) on $(hostname), card ${TT_VISIBLE_DEVICES}"
echo "### tree $(git log --oneline -1)"

echo "=== leg 1: opendde 640 aa, arms on,glue $(date -Is) ==="
foreign "before 640"
$PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 640 \
    --arms on,glue --out $O/ab_640_runc.json
echo "leg1 RC=$?"
foreign "after 640"
sleep 30

echo "=== leg 2: protenix-v2 512 aa, arms on,glue,on $(date -Is) ==="
foreign "before px"
$PY -u perf/other512/fold_ab_multi.py --model protenix-v2 --sizes 512 \
    --arms on,glue,on --out $O/ab_px512_runc.json
echo "leg2 RC=$?"
foreign "after px"

echo "### run_c done $(date -Is)"
