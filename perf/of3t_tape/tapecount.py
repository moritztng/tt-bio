"""Count what an OpenFold3 forward asks of the shared tape, at runtime.

`perf/ptx_fastpath/coverage.py` answers the same question for the Pairformer chain by
walking the AST of one file. OF3 spans fifteen modules and reaches the shipped primitives
through three different assemblies, so a static walk would have to model imports, and a
number that depends on how well it modelled them is not evidence. This counts the calls
the forward actually makes.

The seam is `taped_ttnn._taped_verb`, the factory that builds one callable per ttnn verb
for the proxy `tape()` swaps in. Wrapping the factory rather than the proxy means the
count sees exactly what the tape sees, in the same order, with the tape's own `_on_tape`
predicate deciding what "carries a taped tensor" means. Nothing in `tt_bio` changes.

Four buckets per verb:

  taped    at least one operand is an `autograd.Tensor` or a registered parameter, so the
           call reached a tape entry and its gradient is on the graph.
  raw      no taped operand, but at least one raw `ttnn.Tensor`. A call that carries a
           tensor the tape did not follow. Some of these are correct -- a host-precomputed
           mask has no gradient -- and each one has to be named, not counted and forgotten.
  none     no tensor at all: a device query, a config factory, an allocator.
  missing  carried a taped tensor and the verb has no entry. Strict mode lets the tape
           raise, which is what a training run must do. Survey mode records it and falls
           back to the shipped verb on unwrapped operands, so ONE run lists every gap
           instead of one run per gap. A survey run's `taped` count is a lower bound:
           unwrapping at a gap drops the rest of that chain off the tape.
  nested   issued from inside a taped verb rather than from a module. A taped entry
           computes its value by calling the shipped verb on UNWRAPPED operands, and
           `tt_bio.ops.linear`'s fallback is literally `ttnn.linear` inside a module the
           proxy has swapped, so that inner call comes back round. It is the same ttnn
           call as its parent, already routed, and counting it again would report ten
           unrouted linears in a template embedder whose linears are all on the tape.

There are TWO seams into the tape, not one, and a count that watches only the proxy
mislabels the other. `tt_bio/ops.py` routes `linear` and `layer_norm` through
`autograd._hook` at the call site, which tapes them and then calls the shipped body, whose
own `ttnn.linear` arrives at the proxy with unwrapped operands and no sign of where it came
from. So the hook is counted too, and its inner call is nested.

Call sites come from the caller's frame, so a `raw` bucket is a list of source lines to go
and read rather than a verb name to argue about.

`install(l1=True)` additionally samples the L1 allocator either side of every call and
attributes the net growth to the call site. A tape keeps what the forward frees, so an
L1-resident activation the shipped code would have released is still live when the next
program lays out its circular buffers, and the run dies with a clash rather than an OOM.
The watch names the site that kept it. It is a diagnostic: `get_memory_view` drains the
pipeline (~6 us a call), so nothing timed may run under it.
"""
from __future__ import annotations

import collections
import json
import sys

import ttnn

_STATS: dict = {}
_SITES: dict = {}
_MISSING: dict = {}
_DEPTH = [0]
_L1: dict = {}
SURVEY = False
L1_WATCH = False


def _l1_used():
    from tt_bio.tenstorrent import get_device
    mv = ttnn.get_memory_view(get_device(), ttnn.BufferType.L1)
    return (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks


def _bump(qual, bucket, site):
    _STATS.setdefault(qual, collections.Counter())[bucket] += 1
    _SITES.setdefault((qual, bucket), collections.Counter())[site] += 1


_PLUMBING = ("tapecount.py", "ops.py", "autograd.py", "taped_ttnn.py", "dispatch.py")


def _site(depth=2):
    try:
        f = sys._getframe(depth)
    except ValueError:                                                   # pragma: no cover
        return "?"
    return "%s:%d" % (f.f_code.co_filename.rsplit("/", 1)[-1], f.f_lineno)


def _site_outside():
    """The first frame that is neither the tape nor this counter: the module that asked."""
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_filename.rsplit("/", 1)[-1]
        if name not in _PLUMBING:
            return "%s:%d" % (name, f.f_lineno)
        f = f.f_back
    return "?"                                                           # pragma: no cover


def _has_raw(args, kwargs):
    from tt_bio.autograd import _walk
    return any(isinstance(v, ttnn.Tensor) for v in _walk(args, kwargs))


def install(survey: bool = False, l1: bool = False):
    """Patch the verb factory and rebuild the proxy so every verb is counted.

    The proxy caches each verb on first attribute access, so a fresh `_Ttnn` is built
    rather than patched: any verb already resolved would otherwise keep the uncounted
    callable it cached earlier.
    """
    global SURVEY, L1_WATCH
    SURVEY = survey
    L1_WATCH = l1
    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tp

    orig = getattr(tp, "_taped_verb_uncounted", None) or tp._taped_verb
    tp._taped_verb_uncounted = orig

    def counted(qual, shipped):
        inner = orig(qual, shipped)
        impl = tp._VERBS.get(qual)

        def call(*args, **kwargs):
            site = _site()
            if _DEPTH[0]:
                _bump(qual, "nested", site)
                return inner(*args, **kwargs)
            if ag._on_tape(args, kwargs):
                if impl is None:
                    _bump(qual, "missing", site)
                    _MISSING.setdefault(qual, collections.Counter())[site] += 1
                    if not SURVEY:
                        return inner(*args, **kwargs)        # let the tape raise
                    ra, rk = ag._raw(args, kwargs)
                    return shipped(*ra, **rk)
                _bump(qual, "taped", site)
            elif _has_raw(args, kwargs):
                _bump(qual, "raw", site)
            else:
                _bump(qual, "none", site)
            before = _l1_used() if L1_WATCH else 0
            _DEPTH[0] += 1
            try:
                return inner(*args, **kwargs)
            finally:
                _DEPTH[0] -= 1
                if L1_WATCH:
                    r = _L1.setdefault((qual, site), [0, 0])
                    r[0] += _l1_used() - before
                    r[1] += 1

        return call

    tp._taped_verb = counted
    tp._SHIM = tp._Ttnn(ttnn)

    # The other seam. `install()` reads the module global when it sets the hook, so
    # replacing the global here is enough and `tape()` picks this one up.
    hook = getattr(ag, "_hook_uncounted", None) or ag._hook
    ag._hook_uncounted = hook

    def counted_hook(name, shipped, args, kwargs):
        if _DEPTH[0]:
            return hook(name, shipped, args, kwargs)
        site = _site_outside()
        _DEPTH[0] += 1
        try:
            out = hook(name, shipped, args, kwargs)
        finally:
            _DEPTH[0] -= 1
        # None is the hook declining. Production's own body then runs and its `ttnn.`
        # call reaches the proxy, which counts it there; counting it here as well would
        # report one call twice.
        if out is not None:
            _bump(name, "taped", site)
        return out

    ag._hook = counted_hook
    reset()


def reset():
    _STATS.clear()
    _SITES.clear()
    _MISSING.clear()
    _L1.clear()


def totals():
    t = collections.Counter()
    for r in _STATS.values():
        t.update(r)
    return t


def report(label="", path=None, top_sites=6):
    tot = totals()
    carry = tot["taped"] + tot["raw"] + tot["missing"]
    pct = 100.0 * tot["taped"] / carry if carry else 0.0
    out = ["", "=== tape coverage: %s ===" % label,
           "  %5d ttnn calls, %d verbs" % (sum(tot.values()), len(_STATS)),
           "  %5d carry a taped tensor and are routed" % tot["taped"],
           "  %5d carry a taped tensor and have NO tape entry" % tot["missing"],
           "  %5d carry a raw tensor only, not routed" % tot["raw"],
           "  %5d carry no tensor at all" % tot["none"],
           "  %5d nested inside a taped verb (the same call, already counted)"
           % tot["nested"],
           "  coverage %d/%d = %.1f%% of the calls that carry a tensor"
           % (tot["taped"], carry, pct)]
    for bucket, title in (("taped", "ROUTED"),
                          ("missing", "MISSING A TAPE ENTRY"),
                          ("raw", "NOT ROUTED (raw tensor operands only)"),
                          ("nested", "NESTED (inner call of a taped verb)")):
        rows = sorted(((r[bucket], q) for q, r in _STATS.items() if r[bucket]), reverse=True)
        if not rows:
            continue
        out.append("")
        out.append("  %s  (%d calls / %d verbs)"
                   % (title, sum(n for n, _ in rows), len(rows)))
        for n, q in rows:
            s = _SITES.get((q, bucket), {})
            where = "  ".join("%s x%d" % (k, v) for k, v in
                              sorted(s.items(), key=lambda kv: -kv[1])[:top_sites])
            out.append("    %5d  ttnn.%-38s %s" % (n, q, where))
    if _L1:
        rows = sorted(((v[0], v[1], q, s) for (q, s), v in _L1.items()), reverse=True)[:15]
        out.append("")
        out.append("  L1 LEFT BEHIND (net allocator growth attributed to the call site)")
        for net, n, q, s in rows:
            out.append("    %+12d B over %4d calls  ttnn.%-34s %s" % (net, n, q, s))
    text = "\n".join(out)
    print(text, flush=True)
    if path:
        json.dump({"label": label, "totals": dict(tot),
                   "verbs": {q: dict(r) for q, r in _STATS.items()},
                   "sites": {"%s|%s" % (q, b): dict(s) for (q, b), s in _SITES.items()}},
                  open(path, "w"), indent=1, sort_keys=True)
    return dict(tot)
