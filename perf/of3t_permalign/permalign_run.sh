#!/usr/bin/env bash
# Every arm of the D117 probe, in one line each. CPU only, no card.
#
# Ten arms: four positives that show upstream's alignment completing, three controls that
# show the probe still reporting a fallback when one happens, and three shape/variation arms.
# The three `via_forward` arms load the 570 M checkpoint and take a few minutes each; the rest
# need no checkpoint and finish in seconds.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${S:-/home/moritz/.coworker/scratch/of3t-bondcov}"
PY="${PY:-/home/moritz/of3-upstream-venv/bin/python}"
CKPT="${CKPT:-/home/moritz/.boltz/of3-p2-155k.pt}"
OUT="$W/perf/of3t_permalign/runs"
export PYTHONPATH="$S/of3pkg043:$W" OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
cd "$W"
mkdir -p "$OUT"

P="$PY perf/of3t_permalign/permalign_probe.py"
FROZEN="--batch $S/batch_step003.pt"
G5J="--data-dir $S/datasets
     --cache-file $S/datasets/training_cache_with_templates_subset_2.json
     --stage finetune_1 --crop 256 --index 12 --rank-template $S/batch_step003.pt"

run() { tag="$1"; shift; echo "=== $tag"; nice -n 19 $P "$@" --tag "$tag" --out "$OUT/$tag.json"; }

run A_frozen5nw3_s1              $FROZEN --samples 1 --also-naive
run B_frozen_dropkey_CONTROL     $FROZEN --samples 1 --also-naive --drop-key
run C_frozen_nosampledim         $FROZEN --samples 0 --also-naive
run D_frozen5nw3_flip            $FROZEN --samples 1 --also-naive --flip-symmetric --noise 0.05
run E_4g5j_fixed                 $G5J --collate fixed   --samples 1 --also-naive
run F_4g5j_recurse_CONTROL       $G5J --collate recurse --samples 1 --also-naive
run G_4g5j_fixed_flip            $G5J --collate fixed   --samples 1 --also-naive \
                                      --flip-symmetric --noise 0.05
run H_4g5j_via_forward           $G5J --collate fixed --via-forward --checkpoint "$CKPT"
run I_frozen5nw3_via_forward     $FROZEN --via-forward --checkpoint "$CKPT"
run J_frozen5nw3_forward_twice_CONTROL $FROZEN --via-forward --forward-twice --checkpoint "$CKPT"

echo "=== the check"
$PY perf/of3t_permalign/permalign_selftest.py --batch "$S/batch_step003.pt"
echo "=== the table"
python3 perf/of3t_permalign/summarise.py
