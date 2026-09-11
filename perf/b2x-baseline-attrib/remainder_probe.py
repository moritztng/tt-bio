#!/usr/bin/env python3
"""A2: what the uninstrumented remainder of the 512 aa fold actually is.

Pass 1 bracketed every `tt_bio.tenstorrent` class and synced on both sides of every bracket. That
closes the fold by construction but it drains the pipeline ~33 000 times, so its per-call times
are an upper bound and the campaign's 18.2 % sync-inflation correction was aimed at exactly that.

This run keeps the accounting and drops the draining: the same bracket tree, but
`ttnn.synchronize_device` ONLY at depth 0 and 1, i.e. around the top-level phase calls and
nothing else. A few hundred syncs instead of tens of thousands. So:

* the depth-1 times are clean phase times, comparable with the plain fold's wall;
* the deeper times are host-issue time for that subtree, because with no sync the call returns as
  soon as the work is enqueued -- and that is worth having on a fold that is 84 % main-thread CPU;
* every ttnn call is still charged to its bracket path by `Census`, so the ops and the modelled
  bytes that belong to no phase come out by construction rather than by subtraction.

The residual row this prints -- time, ops, bytes, and both roof fractions -- is the row the
campaign's byte-axis ceiling turns on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(HERE))

import baseline_attrib as BA                                                  # noqa: E402

STREAM_ROOF = 429.9e9
COMPUTE_ROOF = 85.96e12


class TopSyncBrackets(BA.Brackets):
    """Brackets that drain the device only at the top of the tree."""

    def __init__(self, *a, sync_depth=1, **k):
        super().__init__(*a, **k)
        self.sync_depth = sync_depth

    def _sync(self):
        if len(self.stack) <= self.sync_depth:
            super()._sync()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m"
    msa_dir = HERE / ".msa_512"

    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": __import__("os").environ.get("TT_VISIBLE_DEVICES"),
                   "loadavg": open("/proc/loadavg").read().split()[:3]}}
    one_fold, meta, state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    out["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "n_msa", "card_type")
                       if k in meta})
    dev = T.get_device()

    def dump():
        a.out.write_text(json.dumps(out, indent=1))

    print("=== cold (discarded) ===", flush=True)
    t, m = one_fold()
    out["cold_s"] = round(t, 3)
    plain = []
    for i in range(a.reps):
        t, m = one_fold()
        plain.append(round(t, 3))
        print(f"  plain{i} {t:.3f}s plddt={m.get('plddt')}", flush=True)
    out["plain_s"] = plain
    out["plain_median_s"] = sorted(plain)[len(plain) // 2]
    dump()

    br = TopSyncBrackets(dev, ttnn, capture_call=-1, sync_depth=1)
    BA.BR = br
    classes = br.install(T)
    cen = BA.Census(ttnn)
    BA.CEN = cen
    cen.install()
    t0 = time.perf_counter()
    _t, m = one_fold()
    wall = time.perf_counter() - t0
    cen.remove()
    br.remove()
    BA.BR = None

    tree = br.tree()
    rows = cen.rows
    top = {p: v for p, v in tree.items() if "/" not in p}
    top_s = sum(v["incl_s"] for v in top.values())
    n_ops = sum(r["n"] for r in rows.values())

    def path_top(p):
        return p.split("/")[0] if p else "(glue)"

    per_top = {}
    for (p, op), r in rows.items():
        k = path_top(p)
        d = per_top.setdefault(k, {"n": 0, "GB": 0.0})
        d["n"] += r["n"]
        d["GB"] += (r["in_b"] + r["out_b"]) / 1e9
    out["instrumented"] = {
        "fold_s": round(wall, 3), "plddt": m.get("plddt"),
        "top_level_s": round(top_s, 3),
        "residual_s": round(wall - top_s, 3),
        "n_ttnn_calls": n_ops,
        "top_level": {k: {"calls": tree[k]["calls"], "incl_s": tree[k]["incl_s"],
                          "median_ms": tree[k]["median_ms"],
                          "ops": per_top.get(k, {}).get("n", 0),
                          "modelled_GB": round(per_top.get(k, {}).get("GB", 0.0), 2)}
                      for k in sorted(top)},
        "outside_every_class": {"ops": per_top.get("(glue)", {}).get("n", 0),
                                "modelled_GB": round(per_top.get("(glue)", {}).get("GB", 0.0), 3)},
        "classes_bracketed": len(classes),
    }
    out["tree"] = tree
    out["sigs"] = br.sigs()
    out["census_rows"] = [{"path": p, "op": o, "n": r["n"],
                           "in_GB": round(r["in_b"] / 1e9, 4),
                           "out_GB": round(r["out_b"] / 1e9, 4)}
                          for (p, o), r in sorted(rows.items(), key=lambda kv: -(kv[1]["in_b"] + kv[1]["out_b"]))]
    dump()
    print(json.dumps(out["instrumented"], indent=1)[:2000], flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
