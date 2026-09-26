#!/usr/bin/env python3
"""Which fused kernels a BC2 gradient round would reach if its tape gate opened.

`grep -n "ops.taping()" tt_bio/*.py` finds 27 sites and twelve of them are a fused kernel
declining. That is a count of SOURCE, and the question a campaign needs answered is how many of
the twelve a round actually calls -- a kernel nothing reaches is worth no entry at all.

So the gates are counted at runtime, on one real round, with nothing changed about what the round
computes. Two kinds:

  * a BOOLEAN gate (`eligible`, `eligible_back`, `eligible_gated`, `_taping`, `_gout_eligible`)
    is re-asked with the tape term lifted, inside `ops.untaped_kernel()`. That answers the real
    question -- would the REST of the gate have served this call -- and not just whether the
    site was reached. Its reject counters are snapshotted and restored, so the second ask leaves
    no trace in the round's own numbers.
  * a gate inside a kernel's body (`mm_dualnoc.in_proj`, `trimul_tail.fused_tail`,
    `softmax_generic.softmax_pv_fused`) cannot be re-asked, because asking it means running the
    kernel. Those are counted as REACHED and reported as reached, not as would-serve.

`would_serve` is therefore an exact count for the first kind and unknown for the second, and the
report says which is which rather than averaging them into one number.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                        # noqa: E402
import run_round as R                    # noqa: E402
from tt_bio import ops                   # noqa: E402
from tt_bio import eltwise_fusion, mm_dualnoc, reblock_permute, softmax_generic  # noqa: E402
from tt_bio import swiglu_fused, tenstorrent as tn, triatt_qkv, triatt_sdpa, trimul_tail  # noqa: E402

#: site -> {taped: calls made while a tape was open, would_serve: of those, how many the rest of
#: the gate accepts, untaped: calls with no tape open}
COUNTS: dict = {}


def _count(site, kind):
    c = COUNTS.setdefault(site, {"kind": kind, "taped": 0, "would_serve": 0, "untaped": 0})
    return c


def _snapshot():
    """Every reject counter a re-ask could touch, so the second ask leaves no trace."""
    return [(d, dict(d)) for d in (reblock_permute.REJECTS, triatt_sdpa.REJECTS,
                                   triatt_sdpa.FUSE_REJECTS)
            if isinstance(d, dict)] + \
           [(s, list(s)) for s in (reblock_permute.STATS, reblock_permute.STATS_BACK,
                                   reblock_permute.STATS_GATED, triatt_sdpa.STATS)]


def _restore(snap):
    for obj, was in snap:
        if isinstance(obj, dict):
            obj.clear()
            obj.update(was)
        else:
            obj[:] = was


def wrap_bool(mod, name, site):
    """A boolean gate: count it, and re-ask it with the tape term lifted."""
    orig = getattr(mod, name)

    def gate(*a, **kw):
        out = orig(*a, **kw)
        c = _count(site, "bool")
        if ops.taping():
            c["taped"] += 1
            snap = _snapshot()
            try:
                with ops.untaped_kernel():
                    if orig(*a, **kw):
                        c["would_serve"] += 1
            except Exception as exc:                      # a gate that needs a live device
                c.setdefault("errors", {})
                c["errors"][repr(exc)[:120]] = c["errors"].get(repr(exc)[:120], 0) + 1
            finally:
                _restore(snap)
        else:
            c["untaped"] += 1
        return out
    setattr(mod, name, gate)


def wrap_reached(mod, name, site):
    """A gate inside a kernel body: count the call, never re-ask it."""
    orig = getattr(mod, name)

    def body(*a, **kw):
        c = _count(site, "reached")
        c["taped" if ops.taping() else "untaped"] += 1
        return orig(*a, **kw)
    setattr(mod, name, body)


SITES_BOOL = [
    (eltwise_fusion, "_taping", "eltwise_fusion._taping"),
    (reblock_permute, "eligible", "reblock_permute.eligible"),
    (reblock_permute, "eligible_back", "reblock_permute.eligible_back"),
    (reblock_permute, "eligible_gated", "reblock_permute.eligible_gated"),
    (softmax_generic, "eligible", "softmax_generic.eligible"),
    (swiglu_fused, "eligible", "swiglu_fused.eligible"),
    (triatt_qkv, "_taping", "triatt_qkv._taping"),
]
SITES_REACHED = [
    (mm_dualnoc, "in_proj", "mm_dualnoc.in_proj"),
    (softmax_generic, "softmax_pv_fused", "softmax_generic.softmax_pv_fused"),
    (trimul_tail, "fused_tail", "trimul_tail.fused_tail"),
    (triatt_sdpa, "sdpa", "triatt_sdpa.sdpa"),
    (triatt_sdpa, "sdpa_fused_qkv", "triatt_sdpa.sdpa_fused_qkv"),
    (tn, "_tri_att_sdpa_hifi", "tenstorrent._tri_att_sdpa_hifi"),
]


def install():
    for mod, name, site in SITES_BOOL:
        wrap_bool(mod, name, site)
    for mod, name, site in SITES_REACHED:
        wrap_reached(mod, name, site)
    # `_gout_eligible` is a method and is wrapped on the class.
    wrap_bool(tn.TriangleMultiplication, "_gout_eligible",
              "tenstorrent.TriangleMultiplication._gout_eligible")


def main():
    install()
    real_dump = M.dump

    def dump(path, stamp):
        stamp["census"] = {k: dict(v) for k, v in COUNTS.items()}
        stamp["census_rounds"] = sum(1 for e in M.EVENTS if e["kind"] == "round_start")
        stamp["TAPED_KERNELS"] = os.environ.get("TT_BIO_TAPED_KERNELS", "")
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump
    R.main()


if __name__ == "__main__":
    main()
