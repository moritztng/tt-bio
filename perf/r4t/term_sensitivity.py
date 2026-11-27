#!/usr/bin/env python3
"""How much does each G3 loss term amplify noise in the network outputs that feed it?

The bf16 parity bar is relative per term, so a term whose value is small and whose gradient
w.r.t. the outputs is large reads as a big relative miss for a forward difference that is
ordinary. This measures that amplification directly, with no forward in the loop: it scores the
RECORDED outputs of a dump with upstream's own `compute_loss`, then rescores them with relative
Gaussian noise of scale `--eps` on every output tensor, and reports each term's relative move.

`--eps 3.9e-3` is one bf16 ulp at the top of the mantissa (2^-8); `--eps 9.8e-4` is a quarter of
it, roughly what a well-behaved bf16 trunk with fp32 accumulation leaves on an output.

    term_sensitivity.py --ref ~/r4t/accuracy/ref --eps 3.9e-3 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loss_parity_reference import build_model, load_batch  # noqa: E402


def _nest(recorded):
    out = {}
    for k, v in recorded.items():
        head, _, tail = k.partition("/")
        out.setdefault(head, {})[tail] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=3.9e-3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _, model = build_model(Path(os.environ["RFD4_REPO"]).expanduser(), torch.float32)
    fm = model.fm
    only = set(a.only.split(",")) if a.only else None
    per_term: dict[str, list[float]] = {}
    for path in sorted(a.ref.glob("batch*.npz")):
        if only and path.stem not in only:
            continue
        batch, meta, recorded, _ = load_batch(path, torch.float32)
        win = torch.as_tensor(meta["winners"])
        out = {k: {m: torch.as_tensor(v) for m, v in d.items()} for k, d in _nest(recorded).items()}
        with torch.no_grad():
            base = fm.optimization_terms({k: v[win] for k, v in
                                          fm.compute_loss(batch=batch, nn_out=out).items()})
            base = {k: float(v.mean()) for k, v in base.items()}
            rows = {k: [] for k in base}
            for s in range(a.seeds):
                g = torch.Generator().manual_seed(1000 + s)
                noisy = {k: {m: v + a.eps * v.abs() * torch.randn(v.shape, generator=g,
                                                                 dtype=v.dtype)
                             for m, v in d.items()} for k, d in out.items()}
                t = fm.optimization_terms({k: v[win] for k, v in
                                           fm.compute_loss(batch=batch, nn_out=noisy).items()})
                for k, v in t.items():
                    y = base[k]
                    x = float(v.mean())
                    rows[k].append(0.0 if x == y else (abs(x - y) / abs(y) if y else float("inf")))
        rec = dict(batch=path.stem, eps=a.eps, seeds=a.seeds, base=base,
                   rel_median={k: statistics.median(v) for k, v in rows.items()},
                   rel_max={k: max(v) for k, v in rows.items()})
        for k, v in rows.items():
            if base[k] != 0.0:
                per_term.setdefault(k, []).extend(v)
        print(json.dumps(rec), flush=True)
    print(json.dumps(dict(summary=True, eps=a.eps,
                          amplification_median={k: statistics.median(v)
                                                for k, v in per_term.items()},
                          amplification_max={k: max(v) for k, v in per_term.items()})), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
