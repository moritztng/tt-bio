#!/usr/bin/env python3
"""Process-local rules for PARAMETERISED fused unaries, plus a census of every one seen.

WHY THIS EXISTS, and what it is not. Taping the OF3 trunk stops at
`tt_bio.taped_ttnn._activation` (taped_ttnn.py:270) with no backward for
`UnaryWithParam(op_type=UnaryOpType::MUL_UNARY_SFPU, params=[0.25])`, reached from the TEMPLATE
module's attention (`openfold3_template.py:129`) through the shared `_fp32_softmax_tail`
(tenstorrent.py:3758). Raising there is correct and its message says why: forwarding the
activation to the shipped verb and ignoring it gives a right forward and a wrong gradient, "which
is how the TriangleAttention gate read 4.88x high".

The gap is structural rather than one missing entry. `_FUSED_UNARY` is keyed on the bare
`UnaryOpType` and holds `(forward, derivative)` pairs, so it cannot express a unary whose
derivative depends on a PARAMETER: SIGMOID, RELU and SILU need no parameter, and MUL_UNARY_SFPU is
nothing but its parameter. Deciding how `_FUSED_UNARY` should carry parameters is a change to
`taped_ttnn.py`, which is `of3t-tape`'s file, not this row's.

So this shim does the narrow thing a perf row is entitled to do: it patches `_activation` IN THIS
PROCESS ONLY so the cost of a taped cycle and its backward can be measured at all, and it records
what it had to patch so `of3t-tape` gets the full list in one pass instead of one crash at a time.
Nothing here is committed into the tape; importing this module is opt-in behind `--shim`.

IT FAILS CLOSED. A parameterised unary gets a rule only where the rule is exact and obvious:
MUL_UNARY_SFPU(p) is x -> x*p, whose derivative is the constant p, and `ttnn.multiply` takes a
python scalar, so `da * p` is the whole backward. Anything else is recorded and then RE-RAISED,
because a census is not worth a wrong gradient. That means a run that completes under this shim
used exact rules for every activation it met, and a run that stops has the unmodelled one named in
its census.

NO GRADIENT CLAIM IS MADE FROM THIS. It exists to make a timing possible. Gradient values under a
shimmed tape are `of3t-equivalence`'s to verify against the frozen float64 reference at PROTOCOL
3d's bars, after `of3t-tape` lands the real thing in `taped_ttnn.py`.
"""
from __future__ import annotations

import re
import traceback

_PARAMS_RE = re.compile(r"params=\[([^\]]*)\]")

CENSUS: dict = {}
_INSTALLED = [False]


def _model_site():
    """The tt_bio frame that issued the call, skipping the tape's own machinery."""
    out = []
    for fr in reversed(traceback.extract_stack()[:-1]):
        if "/tt_bio/" in fr.filename and "taped_ttnn" not in fr.filename:
            out.append(f"{fr.filename.rsplit('/', 1)[1]}:{fr.lineno}")
            if len(out) == 3:
                break
    return " <- ".join(out)


def _params_of(act):
    """The unary's parameters. ttnn exposes op_type as an attribute but not params -- dir() on a
    UnaryWithParam gives ['op_type'] and .params reads None -- so they are parsed off the repr,
    which is the only place the binding puts them. Returns [] when nothing parses, which makes
    _param_rule decline and the shim fail closed.
    """
    got = getattr(act, "params", None)
    if got:
        try:
            return [float(x) for x in got]
        except Exception:
            pass
    m = _PARAMS_RE.search(repr(act))
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    try:
        return [float(x) for x in body.split(",")]
    except ValueError:
        return []


def _param_rule(act):
    """An exact (forward, derivative) pair for a parameterised unary, or None.

    The derivative may be a python scalar: `_binary` applies it as
    `ttnn.multiply(da, deriv(x, y))`, and ttnn.multiply takes a scalar second operand.
    """
    import ttnn
    u = getattr(ttnn, "UnaryOpType", None)
    op = getattr(act, "op_type", None)
    params = _params_of(act)
    if u is None or op is None or len(params) != 1:
        return None
    p = float(params[0])
    if hasattr(u, "MUL_UNARY_SFPU") and op == u.MUL_UNARY_SFPU:
        # x -> x * p. d/dx = p, a constant, so the backward is one scalar multiply.
        return (lambda x, _p=p: ttnn.multiply(x, _p), lambda x, y, _p=p: _p)
    if hasattr(u, "DIV_UNARY_SFPU") and op == u.DIV_UNARY_SFPU:
        # x -> x / p. Same shape of rule; included because the fp32 softmax tail scales by
        # either sense depending on how the caller pre-baked the bias.
        return (lambda x, _p=p: ttnn.multiply(x, 1.0 / _p), lambda x, y, _p=p: 1.0 / _p)
    return None


def install():
    """Patch `taped_ttnn._activation` for this process. Idempotent."""
    if _INSTALLED[0]:
        return CENSUS
    import tt_bio.taped_ttnn as T

    real = T._activation

    def shim(kwargs, key):
        acts = list(kwargs.get(key) or ())
        try:
            r = real(kwargs, key)
            for a in acts:
                _note(a, key, "already_modelled", True, params=_params_of(a))
            return r
        except NotImplementedError:
            rule = _param_rule(acts[0]) if len(acts) == 1 else None
            _note(acts[0] if acts else None, key, "param_rule_supplied_by_shim",
                  rule is not None,
                  params=_params_of(acts[0]) if acts else [])
            if rule is None:
                # Fail closed. A census is not worth a wrong gradient.
                raise
            return rule

    T._activation = shim
    _INSTALLED[0] = True
    return CENSUS


def _note(act, key, how, resolved, params=None):
    k = repr(act)
    row = CENSUS.setdefault(k, {"activation": k, "operand": key, "how": how,
                                "resolved": bool(resolved), "calls": 0, "sites": {},
                                "params_parsed": params})
    row["calls"] += 1
    s = _model_site()
    row["sites"][s] = row["sites"].get(s, 0) + 1


def summary():
    rows = sorted(CENSUS.values(), key=lambda r: -r["calls"])
    return {"distinct_activations": len(rows),
            "unresolved": [r["activation"] for r in rows if not r["resolved"]],
            "shim_supplied": [r["activation"] for r in rows
                              if r["how"] == "param_rule_supplied_by_shim"],
            "rows": rows}
