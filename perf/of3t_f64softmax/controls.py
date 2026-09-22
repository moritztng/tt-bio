#!/usr/bin/env python3
"""Two controls the arm table cannot carry, and one identity check.

A16: the zero model, MEASURED rather than asserted. A gradient of zeros has rel_l2 1 by
arithmetic, but the campaign's rule is that the baseline is read out of the same comparator the
arms are read out of, because an asserted 1.0 cannot catch a comparator that is summing the wrong
tensors.

IDENTITY: of3t-softgrad measured this bound as a rule installed over the tape verb, on card 0.
This row measures it as the shipped call sites with the site flag on, on card 3. If the two
dumps are bit-identical then the code path is the diagnostic's arithmetic exactly, the number is
not a property of the card, and the row's own instrument floor is zero rather than estimated.
"""
import argparse
import json

import torch


def massw(arm, ref, names):
    num = sum(float(torch.linalg.vector_norm(
        arm[n].to(torch.float64).reshape(-1) - ref[n])) ** 2 for n in names)
    den = sum(float(torch.linalg.vector_norm(ref[n])) ** 2 for n in names)
    return (num / den) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--a", required=True, help="this row's host-f64 dump")
    ap.add_argument("--b", required=True, help="of3t-softgrad's host-f64 dump")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    A = torch.load(a.a, map_location="cpu", weights_only=False)
    B = torch.load(a.b, map_location="cpu", weights_only=False)
    F = torch.load(a.f64, map_location="cpu", weights_only=False)
    Bf = torch.load(a.bf16, map_location="cpu", weights_only=False)
    names = sorted(set(A) & set(B))
    names = [n for n in names if F.get(n) is not None and Bf.get(n) is not None]
    f = {n: F[n].to(torch.float64).reshape(-1) for n in names}
    b = {n: Bf[n].to(torch.float64).reshape(-1) for n in names}

    bit, maxabs, worst = 0, 0.0, None
    for n in names:
        x, y = A[n], B[n]
        if x.shape == y.shape and torch.equal(x, y):
            bit += 1
            continue
        d = float((x.double() - y.double()).abs().max())
        if d > maxabs:
            maxabs, worst = d, n
    zeros = {n: torch.zeros_like(F[n]) for n in names}
    rep = {
        "scope_tensors": len(names),
        "a16_zero_model_vs_their_step": massw(zeros, b, names),
        "a16_zero_model_vs_float64": massw(zeros, f, names),
        "bit_identical_tensors": bit,
        "of": len(names),
        "largest_absolute_difference": maxabs,
        "worst_tensor": worst,
        "a": a.a, "b": a.b,
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
