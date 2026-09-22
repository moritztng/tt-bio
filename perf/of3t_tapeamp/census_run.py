#!/usr/bin/env python3
"""Call census of the tape's dtype reconciliation, on the diffusion scope, on device.

D196 is the defect where a source-read route with a measured share attached became a brief's
mechanism and turned out never to execute. So this counts CALLS. What it counts is the one place
the tape reconciles a cotangent's dtype to its forward value's, `tt_bio/autograd.py`:

    if g.dtype != t.value.dtype:
        g = ttnn.typecast(g, t.value.dtype)
    t.node.fn(_unshard(g))

That single branch is the whole D206 surface inside the shipped tape: there is no per-verb
typecast to go hunting for, because every closure gets its gradient here. The question a census
answers and a source read cannot is how often it fires, and in which DIRECTION -- a
float32 -> bfloat16 reconciliation rounds a cotangent away at every node it fires on, an
upcast does not.

Two wrappers, neither of which changes any arithmetic:

  `ttnn.typecast` records (caller file:lineno, in dtype, out dtype) and calls through. The
  reconciliation's own call site is `autograd.py`'s `backward`, so it separates itself from
  every other typecast in the tape by frame rather than by name.

  `tt_bio.autograd._unshard` records the qualname of the closure about to fire and the dtype
  of the gradient going into it. It is called exactly once per node firing, at the line above,
  so it is the denominator: reconciliations over firings.

Because neither wrapper changes arithmetic, the arm it drives must produce a gradient
bit-identical to the unwrapped arm. That is this census's control, and it is checked rather
than asserted: `--expect-forward` fails the run if the forward median moves at all.
"""
from __future__ import annotations

import collections
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

OUT = Path(os.environ.get("CENSUS_OUT", "/tmp/of3t/tapeamp/CENSUS.json"))

typecasts = collections.Counter()
firings = collections.Counter()
recon_by_verb = collections.Counter()
_state = {"verb": None}


def install():
    import ttnn
    import tt_bio.autograd as ag

    orig_tc = ttnn.typecast
    orig_un = ag._unshard

    def typecast(x, dt, *a, **k):
        f = sys._getframe(1)
        site = f"{Path(f.f_code.co_filename).name}:{f.f_lineno}:{f.f_code.co_name}"
        din = str(getattr(x, "dtype", "?")).replace("DataType.", "")
        dout = str(dt).replace("DataType.", "")
        typecasts[f"{site}|{din}->{dout}"] += 1
        if f.f_code.co_name == "backward" and _state["verb"] is not None:
            recon_by_verb[f"{_state['verb']}|{din}->{dout}"] += 1
        return orig_tc(x, dt, *a, **k)

    def _unshard(t):
        # The caller is `backward`, and the closure it is about to run is `t.node.fn` of the
        # tensor whose loop iteration this is. Read it off the caller's own frame rather than
        # threading it through, so the shipped loop is untouched.
        f = sys._getframe(1)
        tt = f.f_locals.get("t")
        fn = getattr(getattr(tt, "node", None), "fn", None)
        verb = getattr(fn, "__qualname__", "?") if fn is not None else "?"
        dg = str(getattr(t, "dtype", "?")).replace("DataType.", "")
        dv = "?"
        try:
            dv = str(tt.value.dtype).replace("DataType.", "")
        except Exception:
            pass
        firings[f"{verb}|g={dg}|value={dv}"] += 1
        _state["verb"] = verb
        return orig_un(t)

    ttnn.typecast = typecast
    ag._unshard = _unshard
    print(f"census: wrappers installed on ttnn.typecast and tt_bio.autograd._unshard", flush=True)


def dump():
    recon = {k: v for k, v in typecasts.items() if k.startswith("autograd.py:") and ":backward" in k}
    tot_fire = sum(firings.values())
    tot_recon = sum(recon.values())
    down = sum(v for k, v in recon.items()
               if "FLOAT32->BFLOAT16" in k.upper() or "FLOAT32_B->BFLOAT16" in k.upper())
    rep = {
        "what": __doc__.strip().splitlines()[0],
        "node_firings_total": tot_fire,
        "reconciliations_total": tot_recon,
        "reconciliation_share_of_firings": (tot_recon / tot_fire) if tot_fire else None,
        "reconciliations_that_DOWNCAST_float32_to_bfloat16": down,
        "RECONCILIATION_by_direction": dict(sorted(recon.items(), key=lambda kv: -kv[1])),
        "RECONCILIATION_by_verb": dict(sorted(recon_by_verb.items(), key=lambda kv: -kv[1])),
        "node_firings_by_verb_and_dtype": dict(sorted(firings.items(), key=lambda kv: -kv[1])),
        "ALL_typecast_call_sites": dict(sorted(typecasts.items(), key=lambda kv: -kv[1])),
        "NOTE": "node_firings counts calls to tt_bio.autograd._unshard, which the backward loop "
                "calls exactly once per fired closure. reconciliations counts ttnn.typecast "
                "calls whose immediate caller frame is that same loop, so the two are the same "
                "denominator and numerator rather than two different populations.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1) + "\n")
    print("=== CENSUS ===", flush=True)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("node_firings_by_verb_and_dtype", "ALL_typecast_call_sites")},
                     indent=1), flush=True)
    print(f"census written to {OUT}", flush=True)


def main():
    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    install()
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            traceback.print_exc()
    finally:
        dump()


if __name__ == "__main__":
    main()
