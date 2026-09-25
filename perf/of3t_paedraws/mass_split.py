#!/usr/bin/env python3
"""of3t-paedraws: where a draw's "mass at or better than bf16" goes, per section.

    mass_split.py K      -> MASS_SPLIT_PD384_sK.json

Recomputes score.py's per-tensor rel for the device arm and for bf16 at draw K, with score.py's
own loaders and definitions (imported, not copied), and splits the float64 squared mass of the
tensors where ours is worse than bf16 by section. Read-only on every input.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "perf/of3t_fullstep64"))
import score  # noqa: E402

P, S = W / "perf/of3t_paedraws", Path("/home/ttuser/of3t_paedraws")


def main(k):
    T = f"PD384_s{k}"
    ref = {n: v.to(torch.float64) for n, v in
           torch.load(S / f"ref_s{k}/f64/grads_f64.pt", weights_only=False).items()
           if v is not None and bool(v.any())}
    bf = {n: v.to(torch.float64) for n, v in
          torch.load(S / f"ref_s{k}/bf16/grads_bf16.pt", weights_only=False).items() if v is not None}
    bij = json.loads((P / f"BIJECTION_{T}.json").read_text())
    shapes = json.loads((P / f"DEVICE_SHAPES_{T}.json").read_text())["shapes"]
    arm, _ = score.to_upstream(score.load_device(S / f"grad_{T}.pt", shapes), bij)
    scored = set(ref) & (set(bij["placements"]) | set(bij["derived"]))
    mass = {n: float((ref[n] ** 2).sum()) for n in scored}
    total = sum(mass.values())

    def rel(x, n):
        return (float(((x[n] - ref[n]) ** 2).sum()) / mass[n]) ** 0.5 if n in x else 1.0

    worse = {n: (rel(arm, n), rel(bf, n)) for n in scored if rel(arm, n) > rel(bf, n)}
    by_sec = defaultdict(lambda: [0.0, 0])
    for n in worse:
        by_sec[score.section_of(n)][0] += mass[n] / total
        by_sec[score.section_of(n)][1] += 1
    out = {"draw": k, "arm": T, "scored_n": len(scored),
           "mass_at_or_better_than_bf16": 1 - sum(mass[n] for n in worse) / total,
           "worse_by_section": {s: {"mass_fraction": m, "n": c}
                                for s, (m, c) in sorted(by_sec.items(), key=lambda x: -x[1][0])},
           "worse_top": [{"tensor": n, "mass_fraction": mass[n] / total, "rel": r, "bf16_rel": b}
                         for n, (r, b) in sorted(worse.items(), key=lambda x: -mass[x[0]])[:25]]}
    (P / f"MASS_SPLIT_{T}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"draw": k, "mass": out["mass_at_or_better_than_bf16"],
                      "worse_by_section": dict(list(out["worse_by_section"].items())[:8])}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
