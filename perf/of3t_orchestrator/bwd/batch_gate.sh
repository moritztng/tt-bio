#!/usr/bin/env bash
# batch_gate.sh -- the accuracy gate for wk/of3t-bwd, WRITTEN BEFORE THE LEVERS.
#
# A46 clause 6: land several levers, then one strong gate over the batch. Consulting an
# accuracy reference per micro-step produces noise rather than safety -- the fp32 CPU diffusion
# reference is not reproducible run to run (7.17 A against 11.04 A on the same design). And A46
# clause 1: a backward is graded on its VJP, never on a forward. A bfp8 arm on this fleet read
# forward cosine 0.99999 and backward Inf/NaN on the same op.
#
# BARS, FIXED HERE AND NOT BY ME. They are `perf/hallgrad/gradcheck.py`'s own, set from the
# bf16 mantissa before any run: bfloat16 keeps 8 explicit mantissa bits so unit roundoff is
# 2^-9 = 1.95e-3; two rounded operands entering a product give sqrt(2)*u = 2.8e-3, and fp32
# destination accumulation keeps the reduction from adding to it. So the floor is 2.8e-3 and
# the bar is 1.0e-2 relative L2, 3.6x above it, with cosine >= 0.9999 because direction is what
# an optimiser consumes. Nothing in this sprint may move either number.
#
# WHERE IT RUNS: qb2's tt-bio-dev env. NOT pc `python3`, which has no torch.
#
# Usage:  bash batch_gate.sh            # full gate
#         bash batch_gate.sh --quick    # cases + chunk invariance, skip the controls
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# Every case, not gradcheck.py's default eight. A backward sprint touches layernorm, softmax,
# the triangle attentions, both matmul transposes, permute and pair contraction, and the
# default list omits five of those. A gate that does not run the op you changed is decoration.
CASES=linear,chain,fanin,layernorm,softmax,triatt,triatt_chunked,triatt_gated,mm_tb,mm_ta,permute,paircontract,paircontract_in,lora,lora_pair,lora_bias

fail=0
run() { echo; echo "### $*"; "$@" || { echo "^^ NON-ZERO EXIT"; fail=1; }; }

run python3 perf/hallgrad/gradcheck.py --cases "$CASES"

# Differences two chunkings of the attention backward against each other rather than against a
# reference, so a chunked recompute that DROPS or DOUBLE-COUNTS a block shows up here and
# nowhere else. This is the check that already validated the dbias term J3 is going to fuse.
# `--chunk-invariance` builds its own `case_triatt` inputs regardless of `--cases`, so the case
# list here is the cheapest one that satisfies argparse, not a second run of the eight above.
run python3 perf/hallgrad/gradcheck.py --cases linear --chunk-invariance

if [ "${1:-}" != "--quick" ]; then
  # CONTROL 1 -- more precision must COLLAPSE the error. If a formula is wrong rather than
  # imprecise, fp32 does not fix it, and that is the only way to tell the two apart.
  run python3 perf/hallgrad/gradcheck.py --cases "$CASES" --dtype float32

  # CONTROL 2 -- the gate must be able to FAIL. A deliberately wrong reduction axis has to be
  # refused; if this passes, every green above means nothing. Note the inversion: a ZERO exit
  # here is the failure. And it must FAIL BY MEASURING, not by raising before it measures --
  # a break control that raises has tested nothing.
  echo; echo "### negative control: --break-layernorm-axis MUST be refused"
  if python3 perf/hallgrad/gradcheck.py --cases layernorm --break-layernorm-axis; then
    echo "^^ THE NEGATIVE CONTROL PASSED. The gate cannot fail, so it is not a gate."; fail=1
  else
    echo "refused, as required"
  fi

  # CONTROL 3 -- LoRA's frozen base weight must receive NO gradient. Declared rather than
  # omitted because a gradient on W is full fine-tuning wearing LoRA's name, and it is
  # invisible in every error metric: the gradient it computes is correct.
  run python3 perf/hallgrad/gradcheck.py --cases lora,lora_pair,lora_bias --lora-init
fi

# INFERENCE. Moritz, 2026-09-21: "make sure regular inference is not changed ... i dont want to
# see regression in inference." This is a hard stop, not a bar to trade. It is only owed if a
# lever on this branch is reachable from an inference fold -- as of the first lever
# (of3t-tapedfwd's `if ops.taping(): return None`) nothing is, because the guard is false
# untaped. Any lever that changes that owes `perf/allm_gates/` on every model executing the
# site, with the A/A floor reported FIRST and byte-identical digests.
echo
echo "INFERENCE A/B: owed only for a lever reachable from a fold. Check before concluding:"
echo "  git diff origin/main...HEAD -- tt_bio/ | grep -v 'ops.taping'"

echo; [ "$fail" = 0 ] && echo "BATCH GATE: green" || echo "BATCH GATE: RED"
exit "$fail"
