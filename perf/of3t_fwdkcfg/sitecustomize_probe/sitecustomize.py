"""Every softmax a shipped fold runs, by call site, with the dtype arriving and the config passed.

`of3t-fwdkcfg` needs a per-site answer: the compute kernel config it prices buys 11-12x accuracy
on an fp32 input and 1.96x on a bf16 one for MORE cost, so a site's dtype is which row of the
price table applies to it. Reading the dtype off the source is a guess about what the tree does
at runtime, and two of the five sites take theirs from a module-level override.

Three things make a zero reading attributable rather than mysterious:

  * `ttnn.softmax` is wrapped as well as `site_softmax`. A fold runs softmaxes; a fold that
    records none did not run.
  * the five site-owning modules are reported present-or-absent in `sys.modules`, and each is
    checked for whether its imported `site_softmax` IS this wrapper. Every site does
    `from .tenstorrent import site_softmax` at import, so a wrapper installed after those
    modules load rebinds nothing and would read zero while the sites ran normally.
  * the dump is written to a FILE as well as stdout, because a fold's stdout is filtered.
"""
import atexit
import json
import os
import sys

SITES = {}
SOFTMAX = {}

OWNERS = ("tt_bio.openfold3_diffusion_transformer", "tt_bio.openfold3_atom_transformer",
          "tt_bio.protenix", "tt_bio.tenstorrent")

_WRAPPED = []


def _rec(store, x, kw, depth=2):
    f = sys._getframe(depth)
    try:
        fn = os.path.relpath(f.f_code.co_filename)
    except ValueError:
        fn = f.f_code.co_filename
    e = store.setdefault("%s:%d" % (fn, f.f_lineno),
                         {"calls": 0, "dtype": None, "shape": None, "ckc": None})
    e["calls"] += 1
    if e["dtype"] is None:
        e["dtype"] = str(getattr(x, "dtype", "?"))
        e["shape"] = list(getattr(x, "shape", []))
        if "compute_kernel_config" in kw:
            e["ckc"] = "None" if kw["compute_kernel_config"] is None else "SET"


def _install():
    import ttnn
    import tt_bio.tenstorrent as T

    inner_site = T.site_softmax

    def probed_site(x, dim=-1, **kw):
        _rec(SITES, x, kw)
        return inner_site(x, dim=dim, **kw)

    T.site_softmax = probed_site
    _WRAPPED.append(probed_site)

    # `ttnn.softmax` is not the only softmax a fold runs: `softmax_in_place` is its own op (the
    # tape registers a verb for both) and the fused SDPA does its softmax inside the kernel,
    # where no Python wrapper can see it. Counting all three is what makes "this site is never
    # reached" a statement about the sites rather than about the instrument.
    for owner, name in ((ttnn, "softmax"), (ttnn, "softmax_in_place"),
                        (getattr(ttnn, "transformer", None),
                         "scaled_dot_product_attention")):
        if owner is None or not hasattr(owner, name):
            continue
        inner = getattr(owner, name)

        def probe(*a, __inner=inner, __key=name, **kw):
            _rec(SOFTMAX.setdefault(__key, {}), a[0] if a else None, kw)
            return __inner(*a, **kw)

        setattr(owner, name, probe)


def _dump():
    try:
        from tt_bio.tenstorrent import HOST_F64_SOFTMAX_STATS as s
        stats = dict(s)
    except Exception:
        stats = {}
    owners = {}
    for m in OWNERS:
        mod = sys.modules.get(m)
        owners[m] = ("absent" if mod is None else
                     "wrapped" if getattr(mod, "site_softmax", None) in _WRAPPED else
                     "REBOUND-PAST-PROBE" if hasattr(mod, "site_softmax") else "no-symbol")
    rep = {"pid": os.getpid(), "site_softmax": SITES, "ttnn_softmax": SOFTMAX,
           "stats": stats, "owner_modules": owners}
    print("SOFTMAX_PROBE " + json.dumps(rep), flush=True)
    d = os.environ.get("TT_BIO_SOFTMAX_PROBE_OUT")
    if d:
        try:
            with open(os.path.join(d, "probe_%d.json" % os.getpid()), "w") as fh:
                json.dump(rep, fh, indent=1, sort_keys=True)
        except Exception:
            pass


try:
    _install()
    atexit.register(_dump)
except Exception as exc:
    print("SOFTMAX_PROBE_FAILED " + repr(exc), flush=True)
