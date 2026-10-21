"""Diff one upstream seam dump against tt-bio's, tensor by tensor.

    /home/moritz/rfd4-ref-env/venv/bin/python perf/rfd4/e2e/pair_compare.py \
        --upstream <seams.pt> --tt <seam0.pt> [--out <json>]

Reports every batch tensor that differs (NaN-aware), the group ids, and x_0 per modality.
"""
import argparse
import json

import torch


def diff(a: torch.Tensor, b: torch.Tensor):
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        a, b = a.double(), b.double()
        if not torch.isnan(a).eq(torch.isnan(b)).all():
            return "NaN pattern differs"
        d = float((torch.nan_to_num(a) - torch.nan_to_num(b)).abs().max())
        return None if d == 0 else f"maxabs={d:.6g}"
    n = int((a.long() != b.long()).sum())
    return None if n == 0 else f"{n} of {a.numel()} elements differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="/home/moritz/rfd4-pair-runs/upstream/seed0/seams.pt")
    ap.add_argument("--tt", default="/home/moritz/rfd4-pair-runs/tt/seed0/seam0.pt")
    ap.add_argument("--out")
    a = ap.parse_args()

    u = torch.load(a.upstream, weights_only=False)
    t = torch.load(a.tt, weights_only=False)
    report = {"upstream": a.upstream, "tt": a.tt, "batch": {}, "groups": {}, "x_0": {}}

    for key in sorted(t["batch"]):
        hit = [c for c in u["batch"] if c == key or c.endswith("." + key)]
        if not hit:
            report["batch"][key] = "missing upstream"
            continue
        d = diff(u["batch"][hit[0]], t["batch"][key])
        if d:
            report["batch"][key] = d
    report["batch_tensors_compared"] = len(t["batch"])

    for key, v in u["groups"].items():
        w = t["groups"].get(key)
        if v is None or w is None:
            report["groups"][key] = "absent" if v is w else "one side absent"
            continue
        d = diff(v, w)
        if d:
            report["groups"][key] = d

    for mode in u["x_0"]:
        d = diff(u["x_0"][mode], t["x_0"][mode])
        if d:
            report["x_0"][mode] = d

    report["features_identical"] = not report["batch"] and not report["groups"]
    report["x_0_identical"] = not report["x_0"]
    print(json.dumps(report, indent=2))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
