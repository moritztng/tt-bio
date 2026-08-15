"""Bound the fp32-softmax lever end to end, before anyone argues about its accuracy.

p49 measured `ttnn.softmax` plus the bf16 cast that follows it at 43.7 ms/step in the decoder, 43.6 %
of that region, over six calls on the fp32 score tensor. Whether the softmax may run in bf16 is an
accuracy question -- the encoder's PairformerAttention already does and says it matches the
reference, while the decoder's fp32 is deliberate and commented at model.py:1520-1533. This does not
answer that question. It prices it, the same way `--freeze-indices` priced attn_indices: run a
deliberately WRONG arm that skips the fp32 round trip, and read the wall.

The arm: every `ttnn.softmax` on an fp32 input casts down to bf16 first and returns bf16. Designs
from this run are not valid and nothing here is shippable.

Baseline for the comparison is the same instrument's 615.3 ms/step (perf/p47/ledger_R4_qb2.json).

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p50_bf16_softmax_bound.py
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import ttnn  # noqa: E402

_softmax = ttnn.softmax


def _bf16_softmax(x, *a, **k):
    if x.dtype == ttnn.float32:
        xb = ttnn.typecast(x, ttnn.bfloat16, memory_config=x.memory_config())
        return _softmax(xb, *a, **k)
    return _softmax(x, *a, **k)


ttnn.softmax = _bf16_softmax
print("[p50] ttnn.softmax now demotes fp32 inputs to bf16 -- WRONG DESIGN, timing arm only",
      flush=True)

sys.argv = ["p35_host_ledger.py",
            "--spec", "perf/dsfix/fixtures/rfd3_R4.json",
            "--num_timesteps", "12",
            "--out", "perf/p50/ledger_R4_bf16_softmax.json"]
runpy.run_path("scripts/rfd3_port/p35_host_ledger.py", run_name="__main__")
