#!/usr/bin/env python3
"""Substitute the two routing verb classes, and census them per block.

`of3t-blk4544` chose its targets as "the two largest unreachable verbs that are real arithmetic
rather than data movement". That criterion removed `_identity_grad` and `_sliced`, which are
47.97 % of one taped backward at padded 384 and which every later conclusion points at. This
module reaches them.

Four things it does differently from `pinvjp2`, each for a reason the earlier rows paid for:

  1. IT DOES NOT TOUCH `pinvjp._REF`. Registering there widens `_sel_allref`, so `allref` and
     `allref2` would stop meaning what their committed scores mean. The route references live in
     `_ROUTE` and `_run_route` dispatches on the caller, so the earlier arms are untouched and
     re-runnable.

  2. THE OPERANDS ARE NOT READ. `_identity_grad`'s closure deliberately keeps only the source
     dtype and layout, so `x.value` is dead by the time the backward runs; `pinvjp._run` would
     read it, get None, and the arm would look like it fired. Both route VJPs need only the
     cotangent and the parent's SHAPE, which is captured at tape time.

  3. THE CENSUS MEASURES, IT DOES NOT ONLY COUNT. Keyed on (caller, block, shape) and, for a
     capped sample inside the blocks that matter, it scores the contribution the shipped closure
     actually delivered against the float64 reference on the same cotangent. Nothing is
     substituted, so its cotangent must come back bit-identical -- the counting and the error
     measurement share one inertness control.

  4. THE WRITE-BACK DTYPE IS AN ARM, NOT A CONSTANT. `pinvjp._run` writes the exact contribution
     back in the parent grad's own dtype. For a first contribution that arrived bf16 that
     re-rounds the reference to bf16 and reproduces the shipped value exactly, so the
     same-protocol arm is inert on this class BY CONSTRUCTION. `routef32` writes back in float32
     instead, which is the arm that actually removes the rounding, and it is reported as a
     storage change rather than as a pure VJP substitution.

`nocast` is the fifth mode and the cheapest: `_identity_grad`'s only arithmetic is
`ttnn.typecast(g, src_dtype)`, so a variant of the verb that omits it is the exact VJP computed
on device with no host round trip, and it therefore runs at EVERY block rather than at three.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_blk4544"))

import pinvjp as P                                                     # noqa: E402
import refroute as R                                                   # noqa: E402

ROUTE_CALLERS = ("_identity_grad", "_sliced")

STATE = {
    "shape_mismatch": 0,
    "no_parent_grad": 0,
    "writeback_dtype": {},        # "<from>-><to>" -> n
    "writeback_lossy_f32": 0,
    "writeback_lossy_store": 0,   # the float64 reference against what the STORE can hold
    "per_block": {},              # "<caller> block<k>" -> substitutions
    "moved": [],
}

CENSUS = {"backward": {}, "forward": {}, "err": {}, "sampled": 0, "skipped": 0,
          "errors": {}, "dtype": {}, "narrowing_per_block": {}}


# Bytes per element, so "the cast changed the dtype" can be told from "the cast LOST
# something". `src=FLOAT32 g=BFLOAT16` is a widening cast and is exactly lossless; only
# `width(src) < width(g)` rounds. Counting both as one number would have reported three times
# the lossy population at padded 64 (576 against 384).
_W = {"FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "BFLOAT16": 2, "FLOAT16": 2,
      "BFLOAT8_B": 1, "BFLOAT4_B": 1, "UINT8": 1}


def _narrows(src, got):
    a, b = _W.get(src), _W.get(got)
    if a is None or b is None:
        return src != got          # unknown pair: report it rather than call it lossless
    return a < b


def _dt(v):
    try:
        return str(v.dtype).replace("DataType.", "")
    except Exception:                                            # noqa: BLE001
        return "?"


# --- what a route node IS -----------------------------------------------------------------

_REAL_DESCRIBE = P._describe


def _describe(caller, frame):
    """Read off the shipped frame's own locals at TAPE time, while `x.value` is still alive."""
    lv = frame.f_locals
    if caller == "_identity_grad":
        x = lv["x"]
        return {"verb": "identity", "caller": caller,
                "shape": [int(d) for d in x.value.shape],
                "cast": bool(lv.get("cast")),
                "src_dtype": _dt(x.value), "src_layout": str(x.value.layout)}
    if caller == "_sliced":
        return {"verb": "sliced", "caller": caller,
                "shape": [int(d) for d in lv["shape"]],
                "starts": [int(v) for v in lv["starts"]],
                "ends": [int(v) for v in lv["ends"]]}
    return _REAL_DESCRIBE(caller, frame)


def _ref_route(desc, g64):
    if desc["verb"] == "identity":
        return R.identity_vjp(g64, desc["shape"])
    return R.sliced_vjp(g64, desc["shape"], desc["starts"], desc["ends"])


# --- the selectors -------------------------------------------------------------------------

def _sel_route(caller, shp, desc):
    """Both routing classes, whatever their shape. 28,848 of 60,144 firings at padded 384 when
    unrestricted; `blocks` is what restricts it, and the reach is reported from the run's own
    counters rather than from this predicate's name."""
    return caller in ROUTE_CALLERS


def _sel_identity_only(caller, shp, desc):
    return caller == "_identity_grad"


def _sel_sliced_only(caller, shp, desc):
    return caller == "_sliced"


# --- the substitution ------------------------------------------------------------------------

def _install_run(pin):
    real_run = P._run
    if getattr(real_run, "_route", False):
        real_run = real_run._real

    def _run_route(desc, caller, shp, g, realfn, plist, out, pin_, chunk):
        if caller not in ROUTE_CALLERS:
            return real_run(desc, caller, shp, g, realfn, plist, out, pin_, chunk)
        blk = P.CUR[0]
        p = plist[0] if plist else None
        if p is None:
            STATE["no_parent_grad"] += 1
            return realfn(g)
        try:
            g64 = P._t(g)
            before = P._t(p.grad) if p.grad is not None else None
        except Exception as e:                                   # noqa: BLE001
            P._bump(P.STATE["errors"], "pre %s %s" % (desc["verb"], type(e).__name__))
            return realfn(g)
        realfn(g)
        try:
            if p.grad is None:
                STATE["no_parent_grad"] += 1
                return None
            after = P._t(p.grad)
            contrib = after if before is None else (after - before)
            if pin_ == "identity3":
                new, mv = after, 0.0
            else:
                ref = _ref_route(desc, g64)[0]
                if list(ref.shape) != list(contrib.shape):
                    if ref.numel() != contrib.numel():
                        STATE["shape_mismatch"] += 1
                        return None
                    ref = ref.reshape(contrib.shape)
                mv = P._rel(contrib, ref)
                new = ref if before is None else (before + ref)
            tgt = p.grad
            store = tgt.dtype
            if pin_ == "routef32" and store != ttnn.float32:
                store = ttnn.float32
            P._bump(STATE["writeback_dtype"], "%s->%s" % (_dt(tgt), str(store)
                                                          .replace("DataType.", "")))
            f32 = new.to(torch.float32)
            P.STATE["cast_tensors"] += 1
            if not torch.equal(f32.to(torch.float64), new):
                P.STATE["cast_lossy"] += 1
                STATE["writeback_lossy_f32"] += 1
            dev = P._device()
            back = ttnn.from_torch(f32.reshape([int(d) for d in tgt.shape]),
                                   dtype=store, layout=tgt.layout, device=dev)
            if store != ttnn.float32:
                # what the STORE itself costs, measured rather than assumed: the reference is
                # exact and the container may not hold it.
                got = P._t(back)
                if got is not None and not torch.equal(got, new):
                    STATE["writeback_lossy_store"] += 1
            p.grad = back
            P._bump(P.STATE["fired"], "%s block%s" % (desc["verb"], blk))
            P._bump(STATE["per_block"], "%s block%s" % (desc["caller"], blk))
            if mv is not None and len(STATE["moved"]) < 4000:
                STATE["moved"].append({"verb": desc["verb"], "block": blk, "rel_moved": mv})
        except Exception as e:                                   # noqa: BLE001
            P._bump(P.STATE["errors"], ("post %s %s: %s" % (desc["verb"], type(e).__name__,
                                                            e))[:200])
        return None

    _run_route._route = True
    _run_route._real = real_run
    P._run = _run_route


_REGISTERED = [False]
REFUSED = ("allref", "allref2", "all", "identity", "identity2", "sm16", "sm4", "pairattn",
           "paircontract")


def install(pin: str = "route", blocks=None, chunk: int = 16):
    """Install a route pin. Refuses every earlier row's pin name: this module changes
    `_describe` and `_run`, so those names would no longer be the arms they were scored as."""
    if pin in REFUSED:
        raise ValueError("pinroute refuses pin %r: it patches _describe and _run, so %r would "
                         "not be the arm that name was scored as" % (pin, pin))
    if not _REGISTERED[0]:
        P._describe = _describe
        P.SELECT["route"] = _sel_route
        P.SELECT["routef32"] = _sel_route
        P.SELECT["identity3"] = _sel_route
        P.SELECT["routeid"] = _sel_identity_only
        P.SELECT["routesl"] = _sel_sliced_only
        _REGISTERED[0] = True
    _install_run(pin)
    return P.install(pin=pin, blocks=blocks, chunk=chunk)


def summary():
    s = P.summary()
    s["route"] = {k: (v if k != "moved" else v[:400]) for k, v in STATE.items()}
    s["route"]["substitutions_by_caller_block"] = dict(sorted(STATE["per_block"].items()))
    s["route"]["substitutions_total"] = sum(STATE["per_block"].values())
    bwd = s.get("backward", {})
    tot = sum(bwd.values())
    s["route"]["backward_firings_total"] = tot
    s["route"]["reach_measured_pct"] = (100.0 * sum(STATE["per_block"].values()) / tot
                                        if tot else None)
    return s


# --- the no-cast arm --------------------------------------------------------------------------

NOCAST = {"nodes": 0, "cast_nodes": 0, "fired": 0, "cast_omitted": 0,
          "narrowing_omitted": 0, "coarsened": 0, "pert_failed": 0, "mode": None}


def install_nocast(mode: str = "nocast"):
    """`typecast`'s backward without its narrowing cast: the exact VJP, on device.

    `_identity_grad`'s whole arithmetic is one line -- `x.add_grad(ttnn.typecast(g, src_dtype)
    if cast and g.dtype != src_dtype else g)`. Dropping the typecast IS the float64 VJP for this
    verb, because the map is the identity and the only thing the cast does is round. Doing it on
    device costs no host round trip, so unlike `route` this runs at every one of the 48 blocks
    and its reach is the whole class rather than three blocks of it.

    `add_grad` then does its own `to_layout` into the parent's layout and, from the second
    contribution on, accumulates in fp32. The FIRST contribution is stored as it arrives, so
    this lever's effect is exactly: a first contribution that used to be stored bf16 is stored
    at the cotangent's own width instead.
    """
    import tt_bio.taped_ttnn as tt
    from tt_bio import autograd as ag

    def _identity_grad(shipped, args, kwargs, cast=False):
        # named for its co_name: the census and _describe key on the frame name, so a
        # rename here would silently split this verb into a second row.
        x = ag._wrap(args[0]) if hasattr(ag, "_wrap") else tt._wrap(args[0])
        ra, rk = tt._raw(args, kwargs)
        out_v = shipped(*ra, **rk)
        src_layout = x.value.layout
        src_dtype = x.value.dtype
        desc_cast = bool(cast)
        NOCAST["nodes"] += 1
        NOCAST["mode"] = mode
        if desc_cast:
            NOCAST["cast_nodes"] += 1

        def make():
            def bw(g):
                NOCAST["fired"] += 1
                if g.layout != src_layout:
                    g = ttnn.to_layout(g, src_layout)
                if mode == "double":
                    # The ON-PATH control, and the reason it exists: `nocast` came back
                    # BIT-IDENTICAL at padded 64 after firing 960 times. A lever that fires
                    # and is inert and a lever that is not on the recorded path look the same
                    # from the output (D121). This one DOUBLES the cotangent at exactly the
                    # same sites. Two is exactly representable in bfloat16, so unlike a small
                    # multiplier the perturbation cannot be lost to a downstream bf16 store,
                    # and that matters here: the narrowing cast this row is about moves each
                    # contribution by about 1.6e-3, which is UNDER bfloat16's 3.9e-3 unit
                    # roundoff, so a small probe would test the storage granularity rather
                    # than the path. If the ladder does not move when every one of these
                    # contributions is doubled, the sites are off the recorded path and
                    # nocast's inertness says nothing about the verb.
                    #
                    # Two earlier attempts failed on device rather than quietly. A bfloat8_b
                    # round trip raised on the first firing (SIDE_X64.json: nodes 490,
                    # fired 1, coarsened 0, rc 1). ttnn.multiply(g, 1.001) HUNG the backward
                    # at padded 64, 401 s against a 42 s baseline, killed by timeout, twice
                    # (aiclk_X64B.txt, aiclk_X64C.txt). So the perturbation goes through the
                    # same host round trip `route` already uses, which is measured to work.
                    # `coarsened` is counted apart from `fired` so a control that cannot act
                    # reads as zero rather than as a pass.
                    t = P._t(g)
                    if t is None:
                        NOCAST["pert_failed"] += 1
                    else:
                        g = ttnn.from_torch(
                            (t * 2.0).to(torch.float32).reshape([int(d) for d in g.shape]),
                            dtype=g.dtype, layout=g.layout, device=P._device())
                        NOCAST["coarsened"] += 1
                elif desc_cast:
                    NOCAST["cast_omitted"] += 1
                    if _narrows(str(src_dtype).replace("DataType.", ""),
                                str(g.dtype).replace("DataType.", "")):
                        NOCAST["narrowing_omitted"] += 1
                x.add_grad(g)
            return bw

        return tt._tape(out_v, [x], make, reads=())

    for v in ("clone", "reallocate", "to_layout", "to_memory_config"):
        if v in tt._VERBS:
            tt._VERBS[v] = lambda s, a, k, _f=_identity_grad: _f(s, a, k)
    tt._VERBS["typecast"] = lambda s, a, k, _f=_identity_grad: _f(s, a, k, cast=True)
    return NOCAST


# --- the census -------------------------------------------------------------------------------

def install_census(err_blocks=(44, 4, 0), cap: int = 2, all_callers: bool = True):
    """Record every taped node per BLOCK, substituting nothing, and score a capped sample of
    the route verbs' own contributions against their float64 references.

    The block tag comes from `pinvjp._install_blocks`, i.e. off `ops.checkpoint_segment`, the
    machinery that DEFINES a block -- not off a firing ordinal.

    `of3t-vjpln` paid for a census taken at a convenient width: 11,856 at padded 64 against
    60,144 at padded 384, five times off, and every prior row had planned against the small one.
    So this is run at the width the arm is read at, and the counts are reported per block.
    """
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt

    P._install_blocks()
    real_tape = ag._tape

    def _tape(out_value, parents, make_fn, reads=None):
        frame, depth = sys._getframe(1), 1
        while frame.f_code.co_name.startswith("<") and depth < 6:
            depth += 1
            frame = sys._getframe(depth)
        caller = frame.f_code.co_name
        out = real_tape(out_value, parents, make_fn, reads=reads)
        shp = P._shape(out_value)
        desc = None
        if caller in ROUTE_CALLERS:
            try:
                desc = _describe(caller, frame)
            except Exception as e:                               # noqa: BLE001
                P._bump(CENSUS["errors"], "describe %s" % type(e).__name__)
        shipped = frame.f_locals.get("shipped")
        nm = getattr(shipped, "__name__", None) if shipped is not None else None
        fkey = "%s | %s | %s" % (caller, nm or "-", shp)
        CENSUS["forward"][fkey] = CENSUS["forward"].get(fkey, 0) + 1
        if out.node is None:
            return out
        realfn = out.node.fn
        plist = list(parents)

        def fn(g, realfn=realfn, caller=caller, shp=shp, desc=desc, plist=plist, nm=nm):
            blk = P.CUR[0]
            key = "%s | %s | %s | block%s" % (caller, nm or "-", shp, blk)
            CENSUS["backward"][key] = CENSUS["backward"].get(key, 0) + 1
            if desc is not None:
                # Read off the cotangent HANDLE, so it costs no host transfer and runs
                # on every firing rather than on a capped sample. Whether a firing
                # rounds at all is decided by exactly this triple: `cast and g.dtype !=
                # src_dtype` is the condition in the shipped closure.
                gd = _dt(g)
                narrow = bool(desc.get("cast")) and _narrows(desc.get("src_dtype"), gd)
                dk = ("%s | %s | cast=%s src=%s g=%s | %s"
                      % (caller, shp, desc.get("cast"), desc.get("src_dtype"), gd,
                         "NARROWING" if narrow else
                         ("widening" if bool(desc.get("cast"))
                          and gd != desc.get("src_dtype") else "exact")))
                CENSUS["dtype"][dk] = CENSUS["dtype"].get(dk, 0) + 1
                if narrow:
                    bk = "block%s" % blk
                    CENSUS["narrowing_per_block"][bk] = (
                        CENSUS["narrowing_per_block"].get(bk, 0) + 1)
            if desc is None or blk not in err_blocks or not plist:
                return realfn(g)
            ek = "%s | %s | block%s" % (caller, shp, blk)
            slot = CENSUS["err"].setdefault(ek, {
                "n": 0, "worst_rel": 0.0, "sum_abs_err": 0.0, "sum_ref_sq": 0.0,
                "cast": desc.get("cast"), "src_dtype": desc.get("src_dtype"),
                "g_dtype": None, "narrowing": 0, "exact": 0})
            if slot["n"] >= cap:
                CENSUS["skipped"] += 1
                return realfn(g)
            p = plist[0]
            try:
                g64 = P._t(g)
                slot["g_dtype"] = _dt(g)
                before = P._t(p.grad) if p.grad is not None else None
                realfn(g)
                after = P._t(p.grad) if p.grad is not None else None
                if after is None or g64 is None:
                    CENSUS["skipped"] += 1
                    return None
                contrib = after if before is None else (after - before)
                ref = _ref_route(desc, g64)[0]
                if ref.numel() == contrib.numel():
                    ref = ref.reshape(contrib.shape)
                    d = float(torch.linalg.vector_norm((contrib - ref).reshape(-1)))
                    slot["sum_abs_err"] += d * d
                    slot["sum_ref_sq"] += float(
                        torch.linalg.vector_norm(ref.reshape(-1))) ** 2
                    r = P._rel(contrib, ref)
                    if r is not None:
                        slot["worst_rel"] = max(slot["worst_rel"], r)
                    if torch.equal(contrib, ref):
                        slot["exact"] += 1
                    else:
                        slot["narrowing"] += 1
                slot["n"] += 1
                CENSUS["sampled"] += 1
            except Exception as e:                               # noqa: BLE001
                P._bump(CENSUS["errors"], "score %s: %s" % (type(e).__name__, str(e)[:80]))
                return None
            return None

        out.node.fn = fn
        return out

    ag._tape = _tape
    tt._tape = _tape
    return CENSUS


def census_summary():
    bwd = CENSUS["backward"]
    tot = sum(bwd.values())
    per_caller, per_block = {}, {}
    for k, v in bwd.items():
        c = k.split(" | ")[0]
        b = k.rsplit(" | ", 1)[1]
        per_caller[c] = per_caller.get(c, 0) + v
        per_block[b] = per_block.get(b, 0) + v
    for e in CENSUS["err"].values():
        e["rel_l2_aggregate"] = ((e["sum_abs_err"] ** 0.5) / (e["sum_ref_sq"] ** 0.5)
                                 if e["sum_ref_sq"] else None)
    return {"pin": "census-route", "blocks": "",
            "backward_firings_total": tot,
            "backward_per_caller": dict(sorted(per_caller.items(), key=lambda kv: -kv[1])),
            "backward_per_caller_pct": {k: round(100.0 * v / tot, 4)
                                        for k, v in sorted(per_caller.items(),
                                                           key=lambda kv: -kv[1])} if tot else {},
            "backward_per_block": dict(sorted(per_block.items())),
            "backward": dict(sorted(bwd.items())),
            "forward": dict(sorted(CENSUS["forward"].items())),
            "route_error_sample": dict(sorted(CENSUS["err"].items())),
            "dtype_tally": dict(sorted(CENSUS["dtype"].items())),
            "narrowing_firings_total": sum(CENSUS["narrowing_per_block"].values()),
            "narrowing_per_block": dict(sorted(CENSUS["narrowing_per_block"].items())),
            "sampled": CENSUS["sampled"], "sample_skipped": CENSUS["skipped"],
            "census_errors": CENSUS["errors"],
            "nocast": dict(NOCAST)}
