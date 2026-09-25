#!/usr/bin/env python3
"""Two cotangent ladders, compared BITWISE, with the hardware that produced each one recorded.

A control that round-trips every selected node through host float64 and writes the SAME value
back has to come out byte for byte identical. `rel_l2 == 0.0` is the same statement only if the
comparison is done in a wider type than the store, so this counts exactly-equal elements instead
and reports the first tensor that is not.

WHY THIS FILE NAMES ITS HARDWARE (D155). `of3t-d137ab` reported protenix-v2 moving between folds
at a fixed seed and it was filed as a user-facing defect against the model. It was pc card 0, a
faulty card root-caused on 2026-08-17, and nothing in the artifact would have said so: it
recorded `"card": 0` and no host, and **card 0 is a different piece of hardware on pc, qb1 and
qb2**. A bit-identity claim whose value depends on which card produced it is a hard stop for this
campaign, so the card has to be recoverable from the ARTIFACT and not from a log that ages out.

The host is read from `socket.gethostname()` rather than passed in, because a transcribed host is
the same failure one level up. The cards are arguments, because a process that pins
`TT_VISIBLE_DEVICES` cannot see which card the OTHER side of the comparison used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket

import torch

#: The fleet's own names for these boxes. `tt-quietbox` is qb1 and `tt-quietbox2` is qb2, and
#: both spellings turn up in this campaign's prose, so the artifact carries both.
ALIAS = {"tt-quietbox": "qb1", "tt-quietbox2": "qb2"}
BOARD = {"qb1": "p150a Blackhole", "qb2": "p300c Blackhole"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the baseline ladder")
    ap.add_argument("b", help="the ladder under test")
    ap.add_argument("--a-card", required=True, help="the card `a` was produced on")
    ap.add_argument("--b-card", required=True, help="the card `b` was produced on")
    ap.add_argument("--out")
    o = ap.parse_args()

    fqdn = socket.gethostname()
    short = fqdn.split(".")[0]
    host = ALIAS.get(short, short)
    board = BOARD.get(host, "unknown board")
    ident = "%s (%s) card %s vs card %s, %s" % (host, short, o.a_card, o.b_card, board)

    A = torch.load(o.a, map_location="cpu", weights_only=False)["cot"]
    B = torch.load(o.b, map_location="cpu", weights_only=False)["cot"]
    out = {"host": host, "hostname": short, "board": board,
           "card": ident,
           "a_ran_on": "%s card %s, %s" % (host, o.a_card, board),
           "b_ran_on": "%s card %s, %s" % (host, o.b_card, board),
           "a": o.a, "b": o.b, "tensors": 0, "bit_identical": 0, "differing": []}
    h = hashlib.sha256()
    for k in sorted(int(x) for x in A):
        for tr in ("ds", "dz"):
            ta, tb = A[k].get(tr), B.get(k, {}).get(tr)
            if ta is None and tb is None:
                continue
            out["tensors"] += 1
            h.update(ta.contiguous().numpy().tobytes())
            if tb is not None and ta.shape == tb.shape and torch.equal(ta, tb):
                out["bit_identical"] += 1
            else:
                d = (ta.double() - tb.double()) if tb is not None else None
                out["differing"].append(
                    {"rung": k, "track": tr,
                     "n_differing": (int((ta != tb).sum()) if tb is not None else None),
                     "max_abs": (float(d.abs().max()) if d is not None else None),
                     "rel_l2": (float(torch.linalg.vector_norm(d.reshape(-1))
                                      / torch.linalg.vector_norm(ta.double().reshape(-1)))
                                if d is not None else None)})
    out["a_content_sha256_12"] = h.hexdigest()[:12]
    out["verdict"] = ("BIT-IDENTICAL" if out["bit_identical"] == out["tensors"]
                      else "DIFFERS on %d of %d" % (out["tensors"] - out["bit_identical"],
                                                    out["tensors"]))
    print(json.dumps({k: v for k, v in out.items() if k != "differing"}, indent=1))
    for r in out["differing"][:6]:
        print(" ", r)
    if o.out:
        json.dump(out, open(o.out, "w"), indent=1)
        print("wrote", o.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
