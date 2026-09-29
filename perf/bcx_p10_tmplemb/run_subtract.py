#!/usr/bin/env python3
"""The ceiling on the multimer template embedding: what does the round do without it?

`bcx-HOSTMAP` charges the template embedding 3.66 s of main's 20.831 host seconds off the
optimised HLO. A profiled share is not seconds you can bank until a subtraction says the round
moves, and this campaign has been wrong here before: on `bcx-seam` the profiler charged the
template embedding 3.78 s and replacing its movable part with a 0.210 s device call moved a
real round by nothing measurable.

`--template-const 1` patches `modules_multimer.TemplateEmbedding.__call__` to return zeros of
its own output shape. XLA then dead-codes the whole module, forward and backward, and what the
round loses is the floor a perfect port could reach. **That arm is wrong numerically and must
never be proposed for merge.** It computes less of the model than the model asks for; it is a
measurement of the ceiling and nothing else.

Everything else is `perf/bcx_round/run_round.py` verbatim -- same campaign, same pool, same
meter, same seed -- so the two arms differ in one patched method and nothing more.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--template-const", type=int, required=True,
                 help="1 replaces the template embedding with a constant; 0 is the model")
_ap.add_argument("--out", required=True)
_known, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

import bc2_state  # noqa: E402,F401  (puts BindCraft 2 on the path)
import jax.numpy as jnp  # noqa: E402
from bindcraft.af.alphafold.model import modules_multimer  # noqa: E402

#: How many times the patched method was traced. A lever that turns out to move nothing has
#: two explanations and this separates them: zero here means the patch never fired.
FIRED = [0]

if _known.template_const:
    def _constant(self, query_embedding, template_batch, padding_mask_2d,
                  multichain_mask_2d, use_dropout, safe_key=None):
        """`TemplateEmbedding.__call__`'s shape and dtype, none of its work."""
        FIRED[0] += 1
        return jnp.zeros(query_embedding.shape, dtype=query_embedding.dtype)

    modules_multimer.TemplateEmbedding.__call__ = _constant

import run_round  # noqa: E402


def main():
    try:
        run_round.main()
    finally:
        # The arm has to be readable off the artifact, not off the directory it landed in.
        events = pathlib.Path(_known.out) / "round_events.json"
        if events.exists():
            doc = json.loads(events.read_text())
            doc["stamp"]["template_const"] = bool(_known.template_const)
            doc["stamp"]["template_patch_fired"] = FIRED[0]
            events.write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
