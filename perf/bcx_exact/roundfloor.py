#!/usr/bin/env python3
"""What exact_training would add to a BindCraft 2 gradient round on origin/main, as a floor.

Card-free, and it answers a question the earlier legs could not. `blockcount.py` and
`ladder.py` measured the instrument's FORWARD float64 arithmetic per Evoformer block and put
it beside `bcx-round`'s 1.13 s taped forward, where it did not fit by a factor of 48. That
contradiction is resolved and the resolution is in `lineage.py`: the tree `bcx-round`
measured has no exact_training in it at all. So the floor is not a contradiction, it is a
PREDICTION about origin/main, where `tt_bio/bindcraft2.py:404` opens `tape()` at the module
default and nothing in BindCraft 2's path ever calls `exact_training(False)`.

This measures both halves of the instrument's arithmetic, forward and backward, at the traced
census shapes, and composes them into the round structure `bcx-round` measured: 2 taped
forwards and 1 checkpointed backward per round, 48 Evoformer blocks each, the backward
recomputing its block forward. So a round pays 3 forward censuses and 1 backward census per
block.

FLOOR, and every excluded term is real and positive:
  * no `ttnn.to_torch` off the card and no `ttnn.from_torch` back, which is the host round
    trip the docstring names and which needs a card;
  * no tape bookkeeping, no dgamma/dbeta (BindCraft 2 trains no weights, so those sums do not
    run and they are correctly absent, not excluded);
  * nothing outside the 48 Evoformer blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                           # noqa: E402

EPS = 1e-5
# bcx-round's measured call structure, runs/round_seed100_card3_long: 2 taped forwards and 1
# backward per round, 0 primal; the backward is checkpointed over 48 Evoformer blocks.
BLOCKS = 48
FWD_CENSUSES_PER_ROUND = 3      # forward #1, forward #2, the backward's recompute
BWD_CENSUSES_PER_ROUND = 1


def med(ts):
    return st.median(ts)


def t_softmax_fwd(shape, reps):
    """`host_f64_softmax_values`' arithmetic: upcast, float64 softmax, downcast for the card."""
    x = torch.randn(*shape, dtype=torch.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y64 = torch.softmax(x.double(), -1)
        y = y64.float()
        ts.append(time.perf_counter() - t0)
        del y64, y
    return ts


def t_softmax_bw(shape, reps):
    """`host_f64_softmax`'s backward: dx = y * (g - (g*y).sum(dim)), on the float64 forward."""
    y64 = torch.softmax(torch.randn(*shape, dtype=torch.float32).double(), -1)
    g = torch.randn(*shape, dtype=torch.float32)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g64 = g.double()
        dx = y64 * (g64 - (g64 * y64).sum(-1, keepdim=True))
        d = dx.float()
        ts.append(time.perf_counter() - t0)
        del g64, dx, d
    return ts


def t_ln_fwd(shape, reps):
    """`_ln_forward64`'s body, gamma and beta present as every AF2 layer norm has them."""
    x = torch.randn(*shape, dtype=torch.float32)
    gamma = torch.randn(shape[-1], dtype=torch.float64)
    beta = torch.randn(shape[-1], dtype=torch.float64)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        x64 = x.double()
        xc = x64 - x64.mean(-1, keepdim=True)
        y64 = xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + EPS)
        y64 = y64 * gamma + beta
        y = y64.float()
        ts.append(time.perf_counter() - t0)
        del x64, xc, y64, y
    return ts


def t_ln_bw(shape, reps):
    """`_v_exact_layer_norm`'s bw, dx only: BindCraft 2 trains no weights, so dgamma and dbeta
    do not run. mean and rstd are re-derived from x re-read off the card, as the code does."""
    x = torch.randn(*shape, dtype=torch.float32)
    g = torch.randn(*shape, dtype=torch.float32)
    gamma = torch.randn(shape[-1], dtype=torch.float64)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g64 = g.double()
        x64 = x.double()
        xc = x64 - x64.mean(-1, keepdim=True)
        rstd = torch.rsqrt((xc * xc).mean(-1, keepdim=True) + EPS)
        n = xc * rstd
        dn = g64 * gamma
        dx = rstd * (dn - dn.mean(-1, keepdim=True) - n * (dn * n).mean(-1, keepdim=True))
        d = dx.float()
        ts.append(time.perf_counter() - t0)
        del g64, x64, xc, rstd, n, dn, dx, d
    return ts


def census_rows(path):
    blob = json.loads(pathlib.Path(path).read_text())
    return blob["traces"]["n288_evoblock0"]["rows"]


def price(rows, reps):
    out = []
    for r in rows:
        sh = tuple(r["shape"])
        if r["op"].startswith("softmax"):
            f, b = med(t_softmax_fwd(sh, reps)), med(t_softmax_bw(sh, reps))
        else:
            f, b = med(t_ln_fwd(sh, reps)), med(t_ln_bw(sh, reps))
        out.append({**r, "fwd_s_each": round(f, 6), "bwd_s_each": round(b, 6),
                    "fwd_s": round(r["calls"] * f, 4), "bwd_s": round(r["calls"] * b, 4)})
    return out


def reduction(n, reps):
    """ladder.py's validated reduction of the census: 2 x (n,4,n,n) + 8 x (n,n,128)."""
    sf, sb = med(t_softmax_fwd((n, 4, n, n), reps)), med(t_softmax_bw((n, 4, n, n), reps))
    lf, lb = med(t_ln_fwd((n, n, 128), reps)), med(t_ln_bw((n, n, 128), reps))
    fwd, bwd = 2 * sf + 8 * lf, 2 * sb + 8 * lb
    return {"n": n, "softmax_fwd_s_each": round(sf, 6), "softmax_bwd_s_each": round(sb, 6),
            "layer_norm_fwd_s_each": round(lf, 6), "layer_norm_bwd_s_each": round(lb, 6),
            "block_fwd_s": round(fwd, 4), "block_bwd_s": round(bwd, 4),
            "round_added_s": round(BLOCKS * (FWD_CENSUSES_PER_ROUND * fwd
                                              + BWD_CENSUSES_PER_ROUND * bwd), 1),
            "loadavg1": round(os.getloadavg()[0], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ns", default="224,288")
    ap.add_argument("--blockcount", default=str(HERE / "BLOCKCOUNT.json"))
    ap.add_argument("--out", default=str(HERE / "ROUNDFLOOR.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_softmax_fwd((64, 4, 64, 64), 1)          # first-touch, untimed

    load0 = os.getloadavg()
    rows = price(census_rows(a.blockcount), a.reps)
    traced = {"block_fwd_s": round(sum(r["fwd_s"] for r in rows), 4),
              "block_bwd_s": round(sum(r["bwd_s"] for r in rows), 4)}
    traced["round_added_s"] = round(BLOCKS * (FWD_CENSUSES_PER_ROUND * traced["block_fwd_s"]
                                              + BWD_CENSUSES_PER_ROUND * traced["block_bwd_s"]), 1)
    red = [reduction(int(x), a.reps) for x in a.ns.split(",")]

    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "opened_a_device": False, "threads": a.threads, "reps": a.reps,
            "nproc": os.cpu_count(), "loadavg_start": load0, "loadavg_end": os.getloadavg(),
            "blocks": BLOCKS, "fwd_censuses_per_round": FWD_CENSUSES_PER_ROUND,
            "bwd_censuses_per_round": BWD_CENSUSES_PER_ROUND,
            "traced_census_n288": {"rows": rows, **traced},
            "reduction": red}
    pathlib.Path(a.out).write_text(json.dumps(blob, indent=1))
    print(json.dumps(traced))
    for r in red:
        print(json.dumps(r))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
