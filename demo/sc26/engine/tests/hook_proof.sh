#!/bin/bash
# FRAMES proof for a diffusion model's trajectory hook (TT_BIO_TRAJECTORY_DIR): the same fold,
# same seed, hook off and hook on, each in its own process. The final structures must be
# byte-identical; the "on" arm leaves one .npy per sampler step.
#   bash demo/sc26/engine/tests/hook_proof.sh <chip> <outdir> [boltz2|openfold3] [input.yaml]
set -u
CHIP=$1; OUT=$2; MODEL=${3:-boltz2}
WT=$(cd "$(dirname "$0")/../../../.." && pwd)
PY=~/tt-bio-dev/env/bin/python3
IN=${4:-$WT/examples/prot_no_msa.yaml}
mkdir -p "$OUT"
run() {  # $1 = arm name, rest = env
  local arm=$1; shift
  local t0=$(date +%s.%N)
  env "$@" PYTHONPATH="$WT" TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP \
      TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:sc26-engine} \
      $PY -c "import sys; sys.argv[0]='tt-bio'; from tt_bio.main import cli; cli()" \
      predict "$IN" --model "$MODEL" --accelerator tenstorrent \
      --out_dir "$OUT/$arm" --seed 0 --output_format cif --override > "$OUT/$arm.log" 2>&1
  echo "$arm rc=$? wall_s=$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/result.txt"
}
( while true; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent!$CHIP/tt_aiclk)"; sleep 1; done ) > "$OUT/aiclk.txt" &
CLK=$!
: > "$OUT/result.txt"
run off
run on TT_BIO_TRAJECTORY_DIR="$OUT/traj"
kill $CLK
for arm in off on; do
  f=$(find "$OUT/$arm" -name "*.cif" | sort | head -1)
  echo "$arm structure=$f sha256=$(grep -v -i '^data_\|_audit\|date' "$f" | sha256sum | cut -c1-64) raw_sha256=$(sha256sum < "$f" | cut -c1-64)" >> "$OUT/result.txt"
done
echo "frames_on=$(ls "$OUT/traj" 2>/dev/null | grep -c '^step_') x0_on=$(ls "$OUT/traj" 2>/dev/null | grep -c '^x0_')" >> "$OUT/result.txt"
echo DONE >> "$OUT/result.txt"
