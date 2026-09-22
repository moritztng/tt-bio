#!/usr/bin/env python3
"""of3t-tapediverge: run another row's instrument under this row's levers, and prove each fired.

Nothing under `tt_bio/` is edited. Every lever below is installed by monkeypatch here, so the
shipped tree stays byte-identical to `origin/wk/of3t` and no default can move. The price of that
choice is that a lever could silently fail to take -- which is exactly `a-lever-can-fire-and-be-
inert` and D121 -- so each one carries its own counter and a lever that asked to fire and served
zero calls fails the run instead of reporting the shipped number under another name.

    tdrun.py --levers sdpa_selfvalue,sm216_precise -- <instrument.py> [args...]

LEVERS

  sdpa_selfvalue   D31's discriminator. `taped_ttnn._v_sdpa` computes the forward with the stock
                   fused SDPA and hands it to `autograd.triangle_attention(..., value=out_v)`,
                   whose backward recomputes the scores at `precise_config()` and never reads
                   that value. Forcing `value=None` makes the function computed and the function
                   differentiated the same function. Both arms are ours; no reference is needed.

  sm216_precise    AMENDMENT 1's narrow arm. `taped_ttnn.py:216` computes the near-cancellation
                   numerator `sum(g * y)` with no kernel config while the denominator added by
                   of3t-apbgrad's repair on :218-219 gets `precise_config()`. This replaces the
                   softmax verb with a copy that configures both. Backward-only by construction.

  sum_precise_all  AMENDMENT 1's wide arm. Every `ttnn.sum` call that passes no
                   `compute_kernel_config` gets `precise_config()`. That reaches `:216`,
                   `autograd.py:949` (the triangle-attention backward's own near-cancellation
                   numerator, with three configured matmuls around it) and D55's remaining
                   sites -- and also reaches FORWARD reductions, so this arm is expected to move
                   `forward_rel_median` where the narrow arm is not.
"""
from __future__ import annotations

import json
import os
import runpy
import sys

COUNT: dict[str, int] = {}


def _bump(name: str) -> None:
    COUNT[name] = COUNT.get(name, 0) + 1


def install_sdpa_selfvalue() -> None:
    import tt_bio.autograd as ag
    orig = ag.triangle_attention

    def patched(*a, **k):
        if k.get("value") is not None:
            k = dict(k)
            k["value"] = None
            _bump("sdpa_selfvalue")
        else:
            _bump("sdpa_selfvalue_already_none")
        return orig(*a, **k)

    ag.triangle_attention = patched
    import tt_bio.taped_ttnn as tp
    tp.ag.triangle_attention = patched


def install_sm216_precise() -> None:
    """Re-register the softmax verb with the numerator reduction configured.

    A copy of `_v_softmax` rather than a wrapper: the reduction sits inside the backward
    closure, so there is nothing to wrap from outside.
    """
    import ttnn
    import tt_bio.taped_ttnn as tp
    from tt_bio.autograd import precise_config

    def _v_softmax(shipped, args, kwargs):
        x = tp._wrap(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        ra, rk = tp._raw(args, kwargs)
        y0 = ttnn.softmax(*ra, **rk) if shipped is ttnn.softmax_in_place else shipped(*ra, **rk)
        box = [y0]

        def make():
            def bw(g):
                y = box[0]
                _bump("sm216_precise")
                inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True,
                                 compute_kernel_config=precise_config())
                if tp._SOFTMAX_BW_RENORM:
                    inner = ttnn.divide(inner, ttnn.sum(
                        y, dim=dim, keepdim=True, compute_kernel_config=precise_config()))
                x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
            return bw

        out = tp._tape(y0, [x], make)
        if out.node is not None:
            out.box = box
        return out

    for name in ("softmax", "softmax_in_place"):
        tp._VERBS[name] = _v_softmax


def install_sum_precise_all() -> None:
    import ttnn
    from tt_bio.autograd import precise_config
    orig = ttnn.sum

    def patched(*a, **k):
        if k.get("compute_kernel_config") is not None:
            _bump("sum_precise_all_already_configured")
            return orig(*a, **k)
        try:
            out = orig(*a, **dict(k, compute_kernel_config=precise_config()))
        except Exception:
            _bump("sum_precise_all_declined")
            return orig(*a, **k)
        _bump("sum_precise_all")
        return out

    ttnn.sum = patched


INSTALL = {
    "sdpa_selfvalue": install_sdpa_selfvalue,
    "sm216_precise": install_sm216_precise,
    "sum_precise_all": install_sum_precise_all,
}


def main() -> int:
    argv = sys.argv[1:]
    levers: list[str] = []
    if argv and argv[0] == "--levers":
        levers = [s for s in argv[1].split(",") if s]
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    script, rest = argv[0], argv[1:]

    for name in levers:
        if name not in INSTALL:
            print(f"FAILED: unknown lever {name!r}", flush=True)
            return 2

    # ttnn and tt_bio must be imported before a patch can land on them, and importing them here
    # rather than inside each installer keeps the order explicit.
    import ttnn  # noqa: F401
    import tt_bio.taped_ttnn  # noqa: F401
    for name in levers:
        INSTALL[name]()

    want_renorm = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "0").lower() \
        not in ("", "0", "false", "no", "off")
    print(f"LEVERS: {levers} renorm_env={want_renorm}", flush=True)

    sys.argv = [script] + rest
    rc = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        rc = e.code or 0

    ev = {"levers_asked": levers, "reach": dict(COUNT)}
    tp = sys.modules.get("tt_bio.taped_ttnn")
    if tp is not None:
        ev["_SOFTMAX_BW_RENORM"] = bool(tp._SOFTMAX_BW_RENORM)
    tt = sys.modules.get("tt_bio.tenstorrent")
    if tt is not None:
        ev["HOST_F64_SOFTMAX_STATS"] = dict(tt.HOST_F64_SOFTMAX_STATS)
    print("LEVER_EVIDENCE " + json.dumps(ev), flush=True)

    if want_renorm and ev.get("_SOFTMAX_BW_RENORM") is not True:
        print("FAILED: the renorm arm did not reach taped_ttnn", flush=True)
        rc = rc or 3
    for name in levers:
        if COUNT.get(name, 0) == 0:
            print(f"FAILED: lever {name} served 0 calls, so this is the unlevered arm under "
                  f"another name", flush=True)
            rc = rc or 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
