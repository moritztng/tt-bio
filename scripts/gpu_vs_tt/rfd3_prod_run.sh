#!/usr/bin/env bash
# Run gpu_rfd3_prod.py the way the published cell was run, with this box's own power roof and
# power floor measured rather than assumed.
#
#   TAG=b200 bash rfd3_prod_run.sh [batch ...]     # default batch 1, the published cell
#
# gpu_rfd3_prod.py requires --power-limit and --idle-W: a watt figure only means something
# against the part's measured limit and the box's measured idle, and the roof of an SXM part is
# not the roof of the PCIe part with the same name. Both are read here, off an idle card, before
# anything is loaded onto it.
set -uo pipefail
TAG=${TAG:-b200}
BATCHES=${*:-1}
W=${W:-/work}
LIM=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
# timeout rather than `| head -30`: closing the pipe on a -lms loop kills nvidia-smi with
# SIGPIPE, which pipefail then reports as a failure for a command that did its job.
IDLE=$(timeout 6 nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 200 \
       2>/dev/null | sort -n | awk 'NF{a[NR]=$1} END{if(NR)print a[int(NR/2)+1]}')
echo "power limit ${LIM} W, idle ${IDLE} W (median of 30 samples on an empty card)"
[ -n "$IDLE" ] || { echo "could not read idle power"; exit 1; }
# foundry resolves the "input" inside an inputs json relative to THAT FILE'S OWN directory,
# while the paths written in the pinned fixture are repo-root-relative
# ("perf/dsfix/targets/R4_9q6y_A.pdb"). Left in perf/dsfix/fixtures/ the two compose into
# /work/perf/dsfix/fixtures/perf/dsfix/targets/... and the run dies with FileNotFoundError.
# Stage the fixture at the repo root so its relative paths resolve against the root they were
# written against. The bytes are copied unchanged, so the sha256 pin still holds -- it is the
# file's LOCATION that has to match, not its content.
INP=$W/rfd3_R4_gpu.json
cp -f "$W/perf/dsfix/fixtures/rfd3_R4_gpu.json" "$INP"
sha256sum "$INP"

exec "$W/v_head/bin/python" "$W/perf/dsfix/gpu_rfd3_prod.py" \
  --arm head-fast --gpu "${TAG^^}" --runner "$W/v_head/bin/python" \
  --inputs "$(basename "$INP")" \
  --power-limit "$LIM" --idle-W "$IDLE" --batches $BATCHES \
  --out "$W/results/rfd3_prod_${TAG}.jsonl"
