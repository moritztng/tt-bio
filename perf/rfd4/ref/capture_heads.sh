#!/bin/bash
# Third wave: the decoder's output heads. They are bare nn.Sequential modules inside
# DecoderTransformer (logit_linear = residue head, struct_linear, atom_presence_linear),
# so they have no class to select and are hooked by name. Three representative families;
# the heads run once per design, at the end of the trajectory.
set -uo pipefail
ROOT=/home/moritz/rfd4-ref-env
R=$ROOT/harness/run_capture.sh

until grep -q CAPTURE_REST_DONE "$ROOT/logs/capture_rest.log" 2>/dev/null; do sleep 60; done

run () { local f=$1; shift; echo "=== $f $(date -u +%FT%TZ)"; timeout 7200 "$R" "$f" "$@" 2>&1 | tail -20; echo "--- $f rc=$?"; }

export CAPTURE_FLAGS="--per-module 1 --sampler-steps 0 --stop-after-captures --classes DecoderTransformer --module-names logit_linear,struct_linear,atom_presence_linear"
run heads_unconditional unconditional_128.yaml 0 recycle_strategy=medium/flat
run heads_binder        binder.yaml            1 recycle_strategy=medium/front_loaded
run heads_small_molecule small_molecule.yaml   4 recycle_strategy=medium/midpeak
echo "CAPTURE_HEADS_DONE $(date -u +%FT%TZ)"
