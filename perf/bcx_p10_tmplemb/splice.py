#!/usr/bin/env python3
"""Can the multimer template pair stack be intercepted in the JAX trace, and with its mask?

The port's cut is the two blocks between `construct_input` and `output_layer_norm` inside
`modules_multimer.SingleTemplateEmbedding.__call__`. Unlike `extra_msa_stack_fn`, they are not
reachable by closure rewriting: the call is inline in a haiku method, as
`template_stack((act, safe_subkey))`.

The hook this installs swaps the module-global `layer_stack` for a shim, and only while the
template embedder is tracing. The shim reads `padding_mask_2d` out of `template_iteration_fn`'s
closure, the way `bindcraft2.find_extra_msa_masks` reads the extra-MSA stack's masks, and hands
both to a replacement.

`--mode identity` is the control and it is the point of this file: the replacement returns the
two blocks' own output, so a correct hook must leave every number in the round exactly where it
was while still proving it ran. A hook that is reachable but inert reports the same round as
the model arm for the wrong reason, so the shim counts its firings and the mask it captured,
and both are stamped into the artifact. `--mode zeros` is `run_subtract.py`'s ceiling arm
reached through this hook instead of by replacing the whole module, which says the hook lands
where the seconds are.
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
_ap.add_argument("--mode", required=True, choices=["identity", "zeros"])
_ap.add_argument("--out", required=True)
_known, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

import bc2_state  # noqa: E402,F401  (puts BindCraft 2 on the path)
import jax.numpy as jnp  # noqa: E402
from bindcraft.af.alphafold.model import modules_multimer  # noqa: E402

#: What the hook saw, so "it changed nothing" and "it never ran" cannot be confused.
SEEN = {"stacks_built": 0, "stacks_called": 0, "mask_shape": None, "act_shape": None,
        "num_block": None}


def _mask_of(fn, depth=0, seen=None):
    """`padding_mask_2d` off `template_iteration_fn`'s closure, however deep haiku wrapped it.

    `gc.use_remat` puts `hk.remat` around the function before `layer_stack` ever sees it, so at
    the shim the only free variable is remat's own `dec_stateful_fun`. Recursing is what
    `bindcraft2._free_variable` does for the extra-MSA masks, for the same reason.
    """
    if depth > 6 or not callable(fn):
        return None
    seen = seen if seen is not None else set()
    if id(fn) in seen:
        return None
    seen.add(id(fn))
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    inner = []
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:                      # an empty cell, still being built
            continue
        if name == "padding_mask_2d":
            return value
        inner.append(value)
    for value in inner:
        got = _mask_of(value, depth + 1, seen)
        if got is not None:
            return got
    return None


class _Shim:
    """Stands in for the `layer_stack` module, for the template embedder's trace only."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):          # everything else stays AlphaFold's own
        return getattr(self._real, name)

    def layer_stack(self, num_block):
        SEEN["num_block"] = int(num_block)

        def build(fn):
            SEEN["stacks_built"] += 1
            mask = _mask_of(fn)
            if mask is None:
                raise ValueError("padding_mask_2d is not reachable from the template stack's "
                                 "closure; the splice point moved")
            SEEN["mask_shape"] = list(mask.shape)

            def run(carry):
                act, key = carry
                SEEN["stacks_called"] += 1
                SEEN["act_shape"] = list(act.shape)
                if _known.mode == "zeros":
                    return jnp.zeros(act.shape, dtype=act.dtype), key
                # identity: AlphaFold's own two blocks, reached through the hook
                return self._real.layer_stack(num_block)(fn)(carry)

            return run

        return build


_orig_call = modules_multimer.SingleTemplateEmbedding.__call__


def _patched(self, *args, **kwargs):
    saved = modules_multimer.layer_stack
    modules_multimer.layer_stack = _Shim(saved)
    try:
        return _orig_call(self, *args, **kwargs)
    finally:
        modules_multimer.layer_stack = saved


modules_multimer.SingleTemplateEmbedding.__call__ = _patched

import run_round  # noqa: E402


def main():
    try:
        run_round.main()
    finally:
        events = pathlib.Path(_known.out) / "round_events.json"
        if events.exists():
            doc = json.loads(events.read_text())
            doc["stamp"]["splice"] = {"mode": _known.mode, **SEEN}
            events.write_text(json.dumps(doc))
        print(json.dumps({"splice": {"mode": _known.mode, **SEEN}}), flush=True)


if __name__ == "__main__":
    main()
