"""Which tuned/fused paths the Nesso-1 trunk takes, per size: every `*_STATS` counter in
tt_bio.tenstorrent, read as the delta over one warm trunk pass.

    TT_VISIBLE_DEVICES=<chip> [TT_BIO_LEVERS=...] PYTHONPATH=<tree> python perf/nesso1/path_stats.py --n 256 532 --report r.json

A warm-up pass at each size first (its refusals and latches are reported separately, since a
refusal latches and the warm pass then shows it as `blocked`), then one counted pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch


def snapshot(tt) -> dict:
    return {k: copy.deepcopy(v) for k, v in vars(tt).items()
            if k.endswith("_STATS") and isinstance(v, (dict, list))}


def delta(a, b):
    """b - a over matching numeric leaves; other leaves as they are in b."""
    if isinstance(b, dict):
        out = {k: delta(a.get(k) if isinstance(a, dict) else None, v) for k, v in b.items()}
        return {k: v for k, v in out.items() if v not in (0, [], {}, None)}
    if isinstance(b, list) and all(isinstance(x, (int, float)) for x in b):
        a = a if isinstance(a, list) and len(a) == len(b) else [0] * len(b)
        d = [y - x for x, y in zip(a, b)]
        return d if any(d) else None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        d = b - (a or 0)
        return d or None
    return b if b != a else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[256, 532])
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    from tt_bio import tenstorrent as tt
    from tt_bio.nesso1 import Nesso1

    model = Nesso1.from_pretrained(use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    pm = model.pairformer_module
    rep = {"visible": os.environ.get("TT_VISIBLE_DEVICES"), "levers": sorted(model._levers), "sizes": []}
    with tt.lever_scope(model._levers):
        for n in a.n:
            torch.manual_seed(n)
            z = torch.randn(1, n, n, 128)
            mask = torch.ones(1, n)
            pair = mask[:, :, None] * mask[:, None, :]
            pm.reset_static_cache()
            s0 = snapshot(tt)
            pm(None, z, mask=mask, pair_mask=pair)
            s1 = snapshot(tt)
            pm(None, z, mask=mask, pair_mask=pair)
            s2 = snapshot(tt)
            row = {"n": n, "first_pass": delta(s0, s1), "warm_pass": delta(s1, s2)}
            rep["sizes"].append(row)
            print(json.dumps(row), flush=True)
    Path(a.report).write_text(json.dumps(rep, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
