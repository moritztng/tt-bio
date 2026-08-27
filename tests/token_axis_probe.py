"""Count, per call site, how many tile-reducing calls a real run makes at a RAGGED axis.

`ttnn.TILE_LAYOUT` pads a tensor physically to 32 while its logical shape stays at the true
length, so "ragged" here is the exact mechanism and not a proxy: `t.shape[a] != t.padded_shape[a]`
on the axis the op reduces over. Whether a ragged axis is a bug depends on the op
(`tt_bio/token_axis.py` records which is which, measured in `perf/bucketing_audit/`), so this
module counts and classifies; it does not judge.

Four wrapped entry points, monkeypatched in-process. Production code is untouched: nothing here is
imported by `tt_bio`, and `TT_BIO_TRIATT_DUALPROBE` (which the audit brief names) does not exist on
this tree -- it lives only on `origin/wk/fused-sdpa-fold-level-root-cause`.

  ttnn.transformer.scaled_dot_product_attention   the one primitive MEASURED unsafe at a ragged
                                                  key axis under a caller-sized additive bias
  tt_bio.triatt_sdpa.sdpa                         the fused transcription of it; refuses when ragged
  tt_bio.softmax_generic.softmax_bf16             refuses when ragged, so a ragged call is a PERF
                                                  loss (the fused softmax goes dark), not a wrong
                                                  answer
  ttnn.softmax                                    masks its own ragged tail; counted for context

`tt-bio predict` and `tt-bio design` run the model in a per-card WORKER SUBPROCESS, so patching
in the parent counts exactly nothing -- the first run of this probe read 0 calls on a fold that
made thousands. So the install is driven from `sitecustomize.py`
(`perf/bucketing_audit/censusenv/`) through an `__import__` hook that fires as soon as ttnn and the
three tt_bio modules are all loaded, in every process the run spawns. Each process writes its own
`census-<pid>.json` into `$TOKEN_AXIS_CENSUS_DIR`; `merge()` adds them up.

    census() in perf/bucketing_audit/census.sh   run a real job under it
    python3 tests/token_axis_probe.py <dir>      print the merged table
"""
import atexit
import json
import os
import sys

# "op|site" -> counters. `masked_ragged` is the one that matters: a ragged key axis under an
# additive bias the caller sized to the LOGICAL length is the 71-76x defect.
COUNTS: dict = {}
_PROBE = os.path.abspath(__file__)
_INSTALLED = False  # unused; per-target state lives in _PATCHED

_FIELDS = ("ragged", "aligned", "masked_ragged", "unfused_ragged", "declined")


_N = [0]


def _bump(op, site, ragged, shape="", **extra):
    rec = COUNTS.setdefault(op + "|" + site,
                            dict({"op": op, "site": site}, **{f: 0 for f in _FIELDS}))
    rec["ragged" if ragged else "aligned"] += 1
    for k, v in extra.items():
        if v:
            rec[k] += 1
    sh = rec.setdefault("shapes", [])
    if ragged and shape and shape not in sh and len(sh) < 4:
        sh.append(shape)
    # Dump as we go, not only at exit. `predict` stops its per-card workers with SIGINT and then
    # SIGKILLs the stragglers, and a SIGKILLed worker runs no atexit handler -- which is how the
    # third run of this probe still read zero on a fold that made thousands of calls.
    _N[0] += 1
    if _N[0] % 64 == 0:
        _dump()


def _site():
    """The innermost tt_bio frame, as ``tt_bio/mod.py::qualname``.

    A symbol and not a line number, because these strings do not stay in a JSON file:
    they get pasted into `tt_bio/token_axis.py` as the evidence for a census row, and
    there a line number is wrong the next time anything is inserted above it. All nine
    in the rfd3 row went stale in one merge. `tests/test_citations.py` can check the
    symbol form against the file and cannot check the other one.
    """
    fr = sys._getframe(1)
    while fr is not None:
        f = os.path.abspath(fr.f_code.co_filename)
        i = f.rfind(os.sep + "tt_bio" + os.sep)
        if i >= 0 and f != _PROBE:
            # `<locals>` and `<listcomp>` are frames, not definitions; a citation that
            # keeps them resolves against nothing.
            # co_qualname is 3.11+. On 3.10 (pc's tt-bio env) reading it raises inside the
            # patched op, the exception escapes as `'code' object has no attribute
            # 'co_qualname'`, and the whole fold fails while the census reports 0/0 -- a
            # census that counts nothing looks exactly like a model that is already clean.
            # Fall back to the bare function name, which test_citations.py still resolves.
            qual = ".".join(part for part in
                            getattr(fr.f_code, "co_qualname", fr.f_code.co_name).split(".")
                            if not part.startswith("<"))
            return f[i + 1:] + "::" + qual
        fr = fr.f_back
    return "<non-tt_bio>"


def _ragged(t, axis):
    """True when `t`'s logical extent on `axis` is short of its physical tile extent."""
    try:
        return int(t.shape[axis]) != int(t.padded_shape[axis]), int(t.shape[axis])
    except Exception:
        return False, -1


_PATCHED = set()


def _patch_ttnn(ttnn):
    _sdpa = ttnn.transformer.scaled_dot_product_attention
    _soft = ttnn.softmax

    def sdpa(q, k, v, *a, **kw):
        rk, kl = _ragged(k, -2)
        rq, ql = _ragged(q, -2)
        _bump("ttnn.sdpa", _site(), rk or rq, "q%d k%d" % (ql, kl),
              masked_ragged=(rk or rq) and kw.get("attn_mask") is not None)
        return _sdpa(q, k, v, *a, **kw)

    def softmax(x, *a, **kw):
        dim = kw.get("dim", a[0] if a else -1)
        rw, w = _ragged(x, dim)
        _bump("ttnn.softmax", _site(), rw, "w%d" % w)
        return _soft(x, *a, **kw)

    ttnn.transformer.scaled_dot_product_attention = sdpa
    ttnn.softmax = softmax


def _patch_triatt(mod):
    _fused = mod.sdpa

    def fused_sdpa(q, k, v, bias, *a, **kw):
        rk, kl = _ragged(k, -2)
        out = _fused(q, k, v, bias, *a, **kw)
        # Two things counted here, and neither was what the audit's first pass assumed.
        #
        # `declined`: the fused kernel refuses by returning None and the caller falls through to
        # the stock op, which this probe also sees. Count refusals, so a bucket that turns the
        # fused path off cannot look like a free correctness fix.
        #
        # `masked_ragged`: the fused path does NOT refuse a logically-ragged axis. Its gate reads
        # `sdpa_generic.plan`, and plan derives Sq/Sk -- and therefore `use_padded_mask` -- from
        # `padded_shape`, so at logical 98 over a physical 128 it sees Sk=128, finds a dividing
        # k_chunk and accepts. Measured: 1208 of 1208 ragged protenix-v2 calls SERVED at k98, zero
        # fall-through to the stock op. `sdpa` returns None only when `bias is None`, so a served
        # ragged call is by construction a ragged axis under a caller-sized additive bias -- the
        # same shape as the defect measured at 71-76x on the stock op.
        _bump("triatt_sdpa.sdpa", _site(), rk, "k%d" % kl, declined=out is None,
              masked_ragged=rk and out is not None)
        return out

    mod.sdpa = fused_sdpa


def _patch_softmax_generic(mod):
    _sbf16 = mod.softmax_bf16

    def softmax_bf16(x, dtype):
        rw, w = _ragged(x, -1)
        served = mod.SSTATS[0]
        out = _sbf16(x, dtype)
        _bump("softmax_generic.softmax_bf16", _site(), rw, "w%d" % w,
              unfused_ragged=rw and mod.SSTATS[0] == served)
        return out

    mod.softmax_bf16 = softmax_bf16


# module in sys.modules -> (key, attribute that proves it finished defining the target, patcher)
_TARGETS = (
    ("ttnn", "transformer", _patch_ttnn),
    ("tt_bio.triatt_sdpa", "sdpa", _patch_triatt),
    ("tt_bio.softmax_generic", "softmax_bf16", _patch_softmax_generic),
)


def _try_patch():
    """Patch each target the moment ITS module is loaded, independently of the others.

    Not "once all of them are loaded": `tt_bio.softmax_generic` is imported only by RFD3, so an
    all-of condition never fires for any other model -- which is exactly how the second run of this
    probe also read zero. And not "as soon as the module name appears in sys.modules" either: the
    name lands there before the module body runs, so `tenstorrent.py` importing `triatt_sdpa` at
    line 13 would trip a trigger keyed on the name alone. Key it on the attribute.
    """
    for name, proof, patch in _TARGETS:
        if name in _PATCHED:
            continue
        mod = sys.modules.get(name)
        if mod is None or not hasattr(mod, proof):
            continue
        patch(mod)
        _PATCHED.add(name)
        if len(_PATCHED) == 1:
            atexit.register(_dump)
    return len(_PATCHED) == len(_TARGETS)


def install():
    """Patch everything now. For an in-process caller that has already imported the model."""
    import ttnn  # noqa: F401
    from tt_bio import softmax_generic, tenstorrent, triatt_sdpa  # noqa: F401
    assert tenstorrent._triatt_sdpa is triatt_sdpa, "triatt_sdpa is bound by value, not by module"
    assert _try_patch(), "not every probe target was patched: " + repr(sorted(_PATCHED))


def enable():
    """Arm the probe for THIS process and every process it spawns.

    Called from sitecustomize, which runs before ttnn exists, so the patching has to wait until the
    modules load. An `__import__` hook is the cheapest place to notice that: a few dict lookups per
    import, and imports are a startup cost.
    """
    import builtins
    real = builtins.__import__

    def hook(name, *a, **kw):
        m = real(name, *a, **kw)
        if _try_patch():
            builtins.__import__ = real
        return m

    builtins.__import__ = hook


def _program_cache_entries():
    """Distinct compiled ttnn programs on the open device, without ever opening one.

    The ttnn program cache keys on the LOGICAL shape, so this counts kernel VARIANTS: it is the
    number a token bucket is meant to reduce, and the reason Moritz asked for bucketing at all.
    Read off `tt_bio.tenstorrent._device` rather than by calling get_device(), which would open a
    card inside a probe.
    """
    m = sys.modules.get("tt_bio.tenstorrent")
    dev = getattr(m, "_device", None) if m is not None else None
    if dev is None:
        return None
    try:
        return int(dev.num_program_cache_entries())
    except Exception:
        return None


def _dump():
    d = os.environ.get("TOKEN_AXIS_CENSUS_DIR")
    if not d or not COUNTS:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "census-%d.json" % os.getpid()), "w") as fh:
        json.dump(list(COUNTS.values()), fh)
    # Separate file so `merge()` keeps seeing only census records. Rewritten on every periodic
    # dump, so the last value survives a SIGKILLed worker -- the same reason the census itself
    # does not wait for atexit.
    n = _program_cache_entries()
    m = sys.modules.get("tt_bio.tenstorrent")
    rec = {"pid": os.getpid()}
    if n is not None:
        rec["program_cache_entries"] = n
    if m is not None:
        # [ragged calls the guard PADDED, calls already aligned]. The acceptance test for
        # TT_BIO_SDPA_RAGGED_PAD shipping default-ON is that this reads 0 fired on every bucketed
        # model: a guard that fires on a shipped path means that path is not actually bucketed.
        rec["sdpa_ragged_pad_fired"] = int(getattr(m, "SDPA_RAGGED_PAD_STATS", [0, 0])[0])
        rec["sdpa_ragged_pad_on"] = bool(getattr(m, "_SDPA_RAGGED_PAD", False))
        rec["sdpa_sites"] = {k: list(v) for k, v in
                             getattr(m, "SDPA_RAGGED_SITES", {}).items()}
    if len(rec) > 1:
        with open(os.path.join(d, "pce-%d.json" % os.getpid()), "w") as fh:
            json.dump(rec, fh)


def _pce_records(d):
    for n in sorted(os.listdir(d)):
        if n.startswith("pce-") and n.endswith(".json"):
            yield json.load(open(os.path.join(d, n)))


def program_cache_entries(d):
    """Max program-cache entries any process of the run reached. One worker per card, so the max
    is that worker's final count; the parent never opens a device and writes no file."""
    return max((r.get("program_cache_entries", 0) for r in _pce_records(d)), default=0)


def sdpa_ragged_pad_fired(d):
    """(total ragged calls the guard padded, was the guard on). 0 fired on a bucketed model is the
    acceptance test for shipping TT_BIO_SDPA_RAGGED_PAD default-ON."""
    recs = list(_pce_records(d))
    return (sum(r.get("sdpa_ragged_pad_fired", 0) for r in recs),
            any(r.get("sdpa_ragged_pad_on") for r in recs))


def merge(d):
    """Add up every per-process census file under *d*."""
    out = {}
    for n in sorted(os.listdir(d)):
        if not (n.startswith("census-") and n.endswith(".json")):
            continue
        for r in json.load(open(os.path.join(d, n))):
            k = r["op"] + "|" + r["site"]
            t = out.setdefault(k, dict({"op": r["op"], "site": r["site"], "shapes": []},
                                       **{f: 0 for f in _FIELDS}))
            for f in _FIELDS:
                t[f] += r.get(f, 0)
            for s in r.get("shapes", ()):
                if s not in t["shapes"] and len(t["shapes"]) < 6:
                    t["shapes"].append(s)
    rows = sorted(out.values(), key=lambda r: (-r["ragged"], r["op"], r["site"]))
    tot = {f + "_total": sum(r[f] for r in rows) for f in _FIELDS}
    return dict(tot, rows=rows)


def render(s):
    lines = ["=== token-axis census: %d ragged / %d aligned; %d MASKED-RAGGED (unsafe), "
             "%d ragged-but-unfused (perf) ==="
             % (s["ragged_total"], s["aligned_total"], s["masked_ragged_total"],
                s["unfused_ragged_total"]),
             "%-32s %-42s %8s %8s %7s %7s  %s"
             % ("op", "site", "ragged", "aligned", "unsafe", "unfused", "ragged shapes")]
    for r in s["rows"]:
        lines.append("%-32s %-42s %8d %8d %7d %7d  %s"
                     % (r["op"], r["site"], r["ragged"], r["aligned"], r["masked_ragged"],
                        r["unfused_ragged"], ",".join(r["shapes"])))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(merge(sys.argv[1])))
    print("program cache entries (kernel variants): %d"
          % program_cache_entries(sys.argv[1]))
    fired, on = sdpa_ragged_pad_fired(sys.argv[1])
    print("sdpa ragged-pad guard: %s, fired on %d call(s)"
          % ("ON" if on else "off", fired))
