#!/usr/bin/env python3
"""Widen `of3t-blk4544`'s pin instrument onto `_taped_layer_norm` and `_taped_linear`.

`allref` was the ceiling of that instrument: every node it had a float64 VJP for, 1296 of 11856
backward firings, 10.93 %, and `R44` went 2.201450 -> 2.203113. The 89.07 % it could not reach
is what this module halves. `_taped_linear` is 1584 firings (13.36 %) and `_taped_layer_norm` is
816 (6.88 %); together 2400, so `allref2` reaches 3696 of 11856, 31.17 %.

Three things kept deliberately separate from `pinvjp.py`:

  1. `refvjp.py` holds the arithmetic and imports no ttnn, so `fdcheck.py` validates it against
     central finite differences with no device in the loop. `FDCHECK.json`: worst FD
     disagreement 1.76e-08, worst autograd disagreement 1.42e-16 over 36 operand-subset checks
     including the no-bias, no-gamma and rank-normalised branches.
  2. ROLES BY IDENTITY. `parents` is `[t for t in (x, w, bias) if t is not None]`, so a call
     without a bias shifts every later index and a VJP that hands back `dw` where the consumer
     wants `dbias` is wrong in a way no shape check catches when both are rank-1. Each parent's
     role is resolved by `is` against the shipped frame's own locals, and an unmatched parent
     RAISES rather than falling through.
  3. `install_census` is a separate, non-substituting wrapper. It answers the brief's `CENSUS:`
     question -- what `impl` actually wraps, 2448 firings and the largest single entry in the
     residue -- and it also checks `refvjp`'s modelled FORWARD against the device's own `out_v`
     at every sampled firing. That is the one question finite differences cannot answer: FD and
     autograd both differentiate the forward `refvjp.py` writes, so neither can say that forward
     is the one the device computes.

REGISTERING INTO `pinvjp._REF` WIDENS `allref` TOO, because `_sel_allref` is `caller in _REF`.
That is a real footgun -- `allref` is a committed, scored arm of the previous row and its name
must keep meaning what it meant -- so `install` REFUSES the pins `allref`, `all` and `identity`
outright. Re-reading those arms is what `COTSCAN_SAT_{64,384}.json` is for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_blk4544"))

import pinvjp as P                                                     # noqa: E402
import refvjp as R                                                     # noqa: E402

FWD = {"checked": 0, "skipped": 0, "worst": {}}          # the modelled-forward check
SHAPES = {"g_reshaped": 0, "role_raises": 0}


def _roles(parents, named):
    """Resolve each parent's role by object identity. Raises on an unmatched parent."""
    out = []
    for p in parents:
        for nm, obj in named:
            if p is obj:
                out.append(nm)
                break
        else:
            SHAPES["role_raises"] += 1
            raise RuntimeError(
                "pinvjp2: a taped parent matched none of %s by identity; the VJP would be "
                "written against the wrong operand" % [n for n, _ in named])
    return out


# --- what a layer_norm / linear node IS -------------------------------------------------------

_REAL_DESCRIBE = P._describe


def _describe(caller, frame):
    lv = frame.f_locals
    if caller == "_taped_layer_norm":
        return {"verb": "layer_norm", "eps": float(lv["eps"]),
                "roles": _roles(lv["parents"], [("x", lv["x"]), ("gamma", lv["gamma"]),
                                                ("beta", lv["beta"])])}
    if caller == "_taped_linear":
        return {"verb": "linear",
                "roles": _roles(lv["parents"], [("x", lv["x"]), ("w", lv["w"]),
                                                ("bias", lv["bias"])])}
    return _REAL_DESCRIBE(caller, frame)


def _fit(g, x):
    """The cotangent, in the operand's own shape. A matmul normalises rank, so g can arrive one
    axis shorter than the value it differentiates; same volume is a reshape and nothing else is
    allowed to pass silently."""
    if list(g.shape) == list(x.shape):
        return g
    if g.numel() == x.numel():
        SHAPES["g_reshaped"] += 1
        return g.reshape(x.shape)
    return g


def _ref_layer_norm(desc, ops_, g):
    roles = desc["roles"]
    idx = {r: i for i, r in enumerate(roles)}
    x = ops_[idx["x"]]
    gamma = ops_[idx["gamma"]] if "gamma" in idx else None
    beta = ops_[idx["beta"]] if "beta" in idx else None
    dx, dgamma, dbeta = R.layer_norm_vjp(x, gamma, beta, _fit(g, x), desc["eps"])
    got = {"x": dx, "gamma": dgamma, "beta": dbeta}
    return [got[r] for r in roles]


def _ref_linear(desc, ops_, g):
    roles = desc["roles"]
    idx = {r: i for i, r in enumerate(roles)}
    x = ops_[idx["x"]]
    w = ops_[idx["w"]]
    bias = ops_[idx["bias"]] if "bias" in idx else None
    dx, dw, dbias = R.linear_vjp(x, w, bias, g)
    got = {"x": dx, "w": dw, "bias": dbias}
    return [got[r] for r in roles]


# --- the widened selector ----------------------------------------------------------------------

def _sel_allref2(caller, shp, desc):
    """Every node the WIDENED instrument can make exact: `_v_matmul`, `_v_softmax`,
    `triangle_attention`, `_taped_layer_norm`, `_taped_linear`. 3696 of 11856, 31.17 %."""
    return caller in P._REF


def _register():
    P._REF["_taped_layer_norm"] = _ref_layer_norm
    P._REF["_taped_linear"] = _ref_linear
    P._describe = _describe
    P.SELECT["allref2"] = _sel_allref2
    P.SELECT["identity2"] = _sel_allref2
    real_run = P._run

    def _run2(desc, caller, shp, g, realfn, plist, out, pin, chunk):
        # `_run`'s identity branch tests `pin == "identity"` literally, so the widened
        # inertness control has to arrive under that name. STATE["pin"] still records
        # `identity2`, which is what the sidecar reports.
        return real_run(desc, caller, shp, g, realfn, plist, out,
                        "identity" if pin == "identity2" else pin, chunk)

    if getattr(P._run, "_vjpln", False) is False:
        _run2._vjpln = True
        P._run = _run2


_REGISTERED = [False]

REFUSED = ("allref", "all", "identity", "sm16", "sm4", "pairattn", "paircontract")


def install(pin: str = "allref2", blocks=None, chunk: int = 16):
    """Install the widened pin. Refuses `of3t-blk4544`'s pins: registering the two new VJPs
    changes what `caller in _REF` selects, so those names no longer mean what their committed
    scores mean."""
    if pin in REFUSED:
        raise ValueError(
            "pinvjp2 refuses pin %r: importing this module widens `_REF`, so `%s` would no "
            "longer be the arm of3t-blk4544 scored. Read COTSCAN_SAT_{64,384}.json instead."
            % (pin, pin))
    if not _REGISTERED[0]:
        _register()
        _REGISTERED[0] = True
    return P.install(pin=pin, blocks=blocks, chunk=chunk)


def summary():
    s = P.summary()
    s["vjpln"] = {"refs_added": ["_taped_layer_norm", "_taped_linear"],
                  "g_reshaped_to_operand": SHAPES["g_reshaped"],
                  "role_resolution_raises": SHAPES["role_raises"],
                  "forward_model_check": _fwd_summary()}
    return s


# --- the census ---------------------------------------------------------------------------------

CENSUS = {"forward": {}, "backward": {}}
_FWD_CAP = 8


def _fwd_summary():
    rows = sorted(FWD["worst"].items(), key=lambda kv: -kv[1]["worst_rel"])
    return {"what": "refvjp's modelled forward against the device's own out_v, per node key. "
                    "FD and autograd both differentiate the modelled forward, so only this "
                    "says the modelled forward is the shipped one.",
            "sampled_per_key": _FWD_CAP,
            "firings_checked": FWD["checked"], "firings_skipped": FWD["skipped"],
            "worst_rel_l2": (rows[0][1]["worst_rel"] if rows else None),
            "worst_key": (rows[0][0] if rows else None),
            "per_key": {k: v for k, v in rows}}


def _check_forward(caller, frame, out_v, shp):
    key = "%s %s" % (caller, shp)
    seen = FWD["worst"].setdefault(key, {"n": 0, "worst_rel": 0.0})
    if seen["n"] >= _FWD_CAP:
        FWD["skipped"] += 1
        return
    lv = frame.f_locals
    try:
        got = P._t(out_v)
        if caller == "_taped_layer_norm":
            mine = R.layer_norm_forward(
                P._t(lv["x"].value),
                P._t(lv["gamma"].value) if lv["gamma"] is not None else None,
                P._t(lv["beta"].value) if lv["beta"] is not None else None,
                float(lv["eps"]))
        else:
            mine = R.linear_forward(
                P._t(lv["x"].value), P._t(lv["w"].value),
                P._t(lv["bias"].value) if lv["bias"] is not None else None)
        mine = mine.reshape(got.shape)
        nr = float(torch.linalg.vector_norm(got.reshape(-1)))
        rel = float(torch.linalg.vector_norm((mine - got).reshape(-1))) / (nr or 1.0)
    except Exception as e:                                             # noqa: BLE001
        seen["error"] = "%s: %s" % (type(e).__name__, e)
        FWD["skipped"] += 1
        return
    seen["n"] += 1
    seen["worst_rel"] = max(seen["worst_rel"], rel)
    FWD["checked"] += 1


def install_census(check_forward: bool = True):
    """Record what every taped node IS, substituting nothing.

    `impl` is 2448 backward firings, 20.65 %, the largest entry in `allref`'s residue, and the
    brief is explicit that a VJP written against a dispatcher tests nothing. `impl` is not an
    op: it is the closure name `taped_ttnn._unary` and `_binary` give their wrappers, so a
    census keyed on `co_name` collapses every eltwise verb in the model into one row. Keyed on
    the SHIPPED callable instead, each row is an op.

    Nothing is substituted, so this arm's cotangent must come back bit-identical to the shipped
    baseline; that is a second inertness control and it is free.
    """
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt

    real_tape = ag._tape

    def _tape(out_value, parents, make_fn, reads=None):
        frame, depth = sys._getframe(1), 1
        while frame.f_code.co_name.startswith("<") and depth < 6:
            depth += 1
            frame = sys._getframe(depth)
        caller = frame.f_code.co_name
        out = real_tape(out_value, parents, make_fn, reads=reads)
        shp = P._shape(out_value)
        shipped = frame.f_locals.get("shipped")
        nm = getattr(shipped, "__name__", None) if shipped is not None else None
        site = "%s:%d" % (Path(frame.f_code.co_filename).name, frame.f_code.co_firstlineno)
        key = "%s | %s | %s | %s" % (caller, nm or "-", site, shp)
        CENSUS["forward"][key] = CENSUS["forward"].get(key, 0) + 1
        if check_forward and caller in ("_taped_layer_norm", "_taped_linear"):
            _check_forward(caller, frame, out_value, shp)
        if out.node is None:
            return out
        realfn = out.node.fn

        def fn(g, realfn=realfn, key=key):
            CENSUS["backward"][key] = CENSUS["backward"].get(key, 0) + 1
            return realfn(g)

        out.node.fn = fn
        return out

    ag._tape = _tape
    tt._tape = _tape
    return CENSUS


def census_summary():
    return {"pin": "census", "blocks": "",
            "forward": dict(sorted(CENSUS["forward"].items())),
            "backward": dict(sorted(CENSUS["backward"].items())),
            "backward_firings_total": sum(CENSUS["backward"].values()),
            "forward_model_check": _fwd_summary()}
