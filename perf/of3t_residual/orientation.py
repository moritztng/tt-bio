#!/usr/bin/env python3
"""Does the device gradient of a SQUARE weight need a transpose the harness never applies?

`device_gradient.py` restores checkpoint orientation with a shape test:

    if tuple(gt.shape) != want and tuple(gt.shape)[::-1] == want: gt = gt.t()

which cannot fire on a square weight. `_w_tt` puts `w.t()` on the card, so the taped dW rule
returns `dW^T` for every weight it loads that way, square or not.

This asks the data, not the code: for every compared tensor, score BOTH orientations against
the reference and report which one wins. A tensor whose transpose is dramatically better is
being scored in the wrong orientation. A tensor where the transpose is worse is being scored
correctly and must not be touched -- which is why the fix cannot be "transpose the square
ones".
"""
import argparse
import hashlib
import json
import sys

import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    d = (a - b).double()
    n = b.double().norm()
    return float(d.norm() / n) if float(n) else float("nan")


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = float(a.norm()), float(b.norm())
    return float(torch.dot(a, b) / (na * nb)) if na and nb else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-sha", default="")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.ref_sha:
        d = digest(a.ref)
        if d != a.ref_sha:
            print(f"FAILED: reference digest {d} != expected {a.ref_sha}", file=sys.stderr)
            return 3
        print(f"reference digest verified {d}")

    dev = torch.load(a.device, map_location="cpu", weights_only=False)
    ref = torch.load(a.ref, map_location="cpu", weights_only=False)
    ref = ref.get("state_dict", ref) if isinstance(ref, dict) and "state_dict" in ref else ref

    rows, n_sq, n_flip = [], 0, 0
    flip_sq, keep_sq, tot_sq = 0.0, 0.0, 0.0
    for k, g in dev.items():
        r = ref.get(k)
        if r is None or not torch.is_tensor(r) or g.shape != r.shape:
            continue
        square = g.dim() == 2 and g.shape[0] == g.shape[1]
        as_is = rel(g, r)
        flipped = rel(g.t().contiguous(), r) if (g.dim() == 2 and g.shape[0] == g.shape[1]) else None
        e_as_is = float((g.double() - r.double()).norm() ** 2)
        e_flip = (float((g.t().contiguous().double() - r.double()).norm() ** 2)
                  if (g.dim() == 2 and g.shape[0] == g.shape[1]) else None)
        tot_sq += e_as_is
        better = flipped is not None and flipped < as_is
        if square:
            n_sq += 1
            if better:
                n_flip += 1
                flip_sq += e_as_is - e_flip
            keep_sq += e_as_is
        rows.append({"param": k, "shape": list(g.shape), "square": square,
                     "rel_as_scored": as_is, "rel_transposed": flipped,
                     "cos_as_scored": cos(g, r),
                     "cos_transposed": cos(g.t().contiguous(), r) if (g.dim() == 2 and g.shape[0] == g.shape[1]) else None,
                     "transpose_is_better": better,
                     "sq_err_as_scored": e_as_is, "sq_err_transposed": e_flip})

    flipped_rows = [r for r in rows if r["transpose_is_better"]]
    nonsq_flip = [r for r in flipped_rows if not r["square"]]
    rep = {"what": __doc__.strip().splitlines()[0], "label": a.label,
           "device": a.device, "ref": a.ref, "ref_sha256": a.ref_sha or None,
           "n_compared": len(rows), "n_square": n_sq,
           "n_square_transpose_better": n_flip,
           "n_nonsquare_transpose_better": len(nonsq_flip),
           "nonsquare_transpose_better": [r["param"] for r in nonsq_flip],
           "total_squared_error_as_scored": tot_sq,
           "squared_error_removed_by_transposing_the_winners": flip_sq,
           "share_of_squared_error_removed_pct": round(100 * flip_sq / tot_sq, 4) if tot_sq else None,
           "rows": rows}
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{a.label}: {len(rows)} compared, {n_sq} square, "
          f"{n_flip} square tensors score better transposed, "
          f"{len(nonsq_flip)} non-square ones do")
    print(f"transposing only the winners removes {rep['share_of_squared_error_removed_pct']} % "
          f"of this arm's total squared error")
    sq = sorted((r for r in rows if r["square"]), key=lambda r: -r["sq_err_as_scored"])
    print(f"\n{'param':<80}{'as scored':>12}{'transposed':>12}{'cos now':>10}{'cos T':>10}")
    for r in sq[:12]:
        print(f"{r['param'][-79:]:<80}{r['rel_as_scored']:>12.4e}{r['rel_transposed']:>12.4e}"
              f"{r['cos_as_scored']:>10.4f}{r['cos_transposed']:>10.4f}")
    bad = [r for r in rows if r["square"] and not r["transpose_is_better"]]
    print(f"\n{len(bad)} square tensors that must NOT be transposed:")
    for r in bad[:10]:
        print(f"  {r['param'][-79:]:<80}{r['rel_as_scored']:>12.4e}{r['rel_transposed']:>12.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
