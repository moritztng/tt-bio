#!/usr/bin/env python3
"""How many times does each op T1 owns actually run in ONE 298 aa protenix-v2 fold?

CHARTER §4.9 / STATUS Q1: the ledger multiplies a block-level ms by 480 (48 blocks x 10 recycles)
and a live-fold count of `trimul.out_proj` says 524. Neither is assumed here. This counts the real
calls at T1's own sites in a real fold, with a counter cheap enough not to perturb it (no timing,
no synchronise, no extra device work).

Folds once through `scripts/gpu_vs_tt/tt_baseline.build_fold`, the same path the campaign absolutes
use: card opened once, model loaded once, MSA seeded from the committed a3m so no search runs,
production config (10 recycles, 1 sample).
"""
import collections
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import ttnn  # noqa: E402

COUNT = collections.Counter()
SHAPES = {}


def site():
    for fr in reversed(traceback.extract_stack()):
        if "tt_bio/" in fr.filename and "blockcount" not in fr.filename:
            return f"{fr.filename.split('/')[-1]}:{fr.lineno}"
    return "?"


def wrap(ns, nm):
    fn = getattr(ns, nm)

    def inner(*a, **kw):
        s = site()
        sig = "+".join("x".join(str(d) for d in t.padded_shape)
                       for t in list(a) + list(kw.values()) if isinstance(t, ttnn.Tensor))
        key = f"{nm}@{s}|{sig}"
        COUNT[key] += 1
        return fn(*a, **kw)
    setattr(ns, nm, inner)


def main():
    from tt_baseline import build_fold                                       # noqa: PLC0415
    target = REPO / "examples/prot300.yaml"
    a3m = REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
    msa_dir = Path.home() / "t1_blockcount_msa"
    one_fold, meta, _state = build_fold("protenix-v2", msa_dir, target, a3m, samples=1)
    one_fold()                                  # cold fold: warms kernels, seeds the MSA cache
    COUNT.clear(); SHAPES.clear()
    for ns, nm in ((ttnn.transformer, "scaled_dot_product_attention"),
                   (ttnn.experimental, "nlp_create_qkv_heads"),
                   (ttnn.experimental, "minimal_matmul"),
                   (ttnn, "permute")):
        wrap(ns, nm)
    one_fold()                                  # the counted fold
    out = {k: v for k, v in COUNT.most_common(60)}
    print(json.dumps(out, indent=1))
    json.dump({"meta_hw": str(meta.get("hw")), "counts": out},
              open(REPO / "perf/tri_att_298/blockcount_298_pv2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
