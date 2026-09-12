#!/usr/bin/env python3
"""Tile-pass ledger for the Boltz-2 pair track, and the one number this row is priced on.

Every algebraic reformulation of a linear layer that keeps the same function comes down to the
same question: which tensor does the chain read more than once, and how many of those reads can a
reassociation delete? The capture answers it directly. `itemize` maps every DRAM buffer to the
ops that consumed it, so a buffer with two or more consuming ops IS a re-read, in bytes, measured.

Reported per pair-track sub-unit at one sequence length:

  real_MB     total DRAM traffic of the call, buffer-address deduped (perf/b2x_difflayer)
  reread_MB   sum over buffers of (n_consumers - 1) * size -- the bytes a perfect fusion of
              every co-input consumer would delete, an UPPER BOUND on this row's whole family
  per-buffer  the re-read table itself: size, allocating op, consuming ops

The upper bound is what makes this instrument worth running before writing any kernel: if the
re-read total is a low single-digit percentage of the sub-unit, no amount of projection fusion
reaches the campaign's target and the row should say so instead of building.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from itemize import itemize                                                   # noqa: E402
from real_traffic import counts                                               # noqa: E402

Z_MB = 67.108864                                       # [512,512,128] bf16, the pair tensor
CZ, HIDDEN, HEADS, HEAD_DIM = 128, 128, 4, 32


def _rng(seed):
    g = torch.Generator().manual_seed(seed)
    return lambda *s: torch.randn(*s, generator=g, dtype=torch.float32) * 0.05


def trimul_weights(seed=0):
    r = _rng(seed)
    return {
        "norm_in.weight": torch.ones(CZ), "norm_in.bias": torch.zeros(CZ),
        "norm_out.weight": torch.ones(HIDDEN), "norm_out.bias": torch.zeros(HIDDEN),
        "g_in.weight": r(2 * HIDDEN, CZ), "p_in.weight": r(2 * HIDDEN, CZ),
        "g_out.weight": r(CZ, CZ), "p_out.weight": r(CZ, HIDDEN),
    }


def triatt_weights(seed=1):
    r = _rng(seed)
    d = HEADS * HEAD_DIM
    return {
        "layer_norm.weight": torch.ones(CZ), "layer_norm.bias": torch.zeros(CZ),
        "linear_q.weight": r(d, CZ), "linear_k.weight": r(d, CZ), "linear_v.weight": r(d, CZ),
        "linear_g.weight": r(d, CZ), "linear_o.weight": r(CZ, d),
        "linear.weight": r(HEADS, CZ),
    }


def transition_weights(seed=2, hidden=4 * CZ):
    r = _rng(seed)
    return {
        "norm.weight": torch.ones(CZ), "norm.bias": torch.zeros(CZ),
        "fc1.weight": r(hidden, CZ), "fc2.weight": r(hidden, CZ), "fc3.weight": r(CZ, hidden),
    }


def build(ck):
    return {
        "trimul_start": (T.TriangleMultiplication(False, trimul_weights(0), ck), "zm"),
        "trimul_end": (T.TriangleMultiplication(True, trimul_weights(3), ck), "zm"),
        "triatt_start": (T.TriangleAttention(HEAD_DIM, HEADS, False, triatt_weights(1), ck), "z"),
        "triatt_end": (T.TriangleAttention(HEAD_DIM, HEADS, True, triatt_weights(4), ck), "z"),
        "transition": (T.Transition(transition_weights(2), ck), "z"),
    }


def call(mod, kind, z, m):
    return mod(z, m) if kind == "zm" else mod(z)


def capture(mod, kind, z, m, sig):
    ttnn.deallocate(call(mod, kind, z, m))             # warm / compile
    ttnn.synchronize_device(z.device())
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    out = call(mod, kind, z, m)
    ttnn.synchronize_device(z.device())
    g = ttnn.graph.end_graph_capture()
    ttnn.deallocate(out)
    return {"sig": sig, "nodes": g}


def reread(c, min_mb=1.0):
    """Bytes a perfect co-input fusion would delete, plus the table it comes from.

    A DRAM buffer consumed by k ops is read k times. Fusing those k consumers into one that
    streams the buffer once deletes (k-1) reads. Weights and other pre-existing buffers count the
    same as intermediates: a projection stack re-reads its input whoever allocated it.
    """
    ops, rows = itemize(c)
    tab, tot = [], 0
    for r in rows:
        if r["kind"] != "DRAM" or r["n_consumers"] < 2:
            continue
        extra = (r["n_consumers"] - 1) * r["size"]
        tot += extra
        if r["size"] >= min_mb * 1e6:
            tab.append({"MB": round(r["size"] / 1e6, 3), "k": r["n_consumers"],
                        "extra_MB": round(extra / 1e6, 3), "alloc": r["alloc_op"],
                        "consumers": r["consumer_names"]})
    tab.sort(key=lambda x: -x["extra_MB"])
    return tot, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="pc")
    ap.add_argument("--card", type=int, default=0)
    a = ap.parse_args()

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    mods = build(ck)

    g = torch.Generator().manual_seed(7)
    z = ttnn.from_torch(torch.randn(1, a.n, a.n, CZ, generator=g), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    m = ttnn.from_torch(torch.ones(1, a.n, a.n), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    res = {"n": a.n, "host": a.host, "card": a.card, "arch": "BH p150a",
           "grid": list(T.COMPUTE_GRID_MAIN), "Z_MB": Z_MB,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": {}}

    for name, (mod, kind) in mods.items():
        c = capture(mod, kind, z, m, name)
        x = counts(c)
        rr, tab = reread(c)
        res["units"][name] = {
            "real_MB": round(x["real_MB"], 3), "w_MB": round(x["real_w_MB"], 3),
            "r_MB": round(x["real_r_MB"], 3), "n_ops": x["n_ops"],
            "real_Z": round(x["real_MB"] / Z_MB, 3),
            "reread_MB": round(rr / 1e6, 3), "reread_Z": round(rr / 1e6 / Z_MB, 3),
            "reread_pct": round(100.0 * (rr / 1e6) / x["real_MB"], 2),
            "reread_table": tab,
            "top": [[round(t / 1e6, 3), nm, round(w / 1e6, 3), round(rd / 1e6, 3)]
                    for t, nm, w, rd in x["per_op"][:14]],
        }
        u = res["units"][name]
        print("%-14s %8.1f MB (%5.2f Z) in %3d ops | re-read %7.1f MB = %5.2f %%"
              % (name, u["real_MB"], u["real_Z"], u["n_ops"], u["reread_MB"], u["reread_pct"]),
              flush=True)

    tot = sum(u["real_MB"] for u in res["units"].values())
    rrt = sum(u["reread_MB"] for u in res["units"].values())
    res["block"] = {"real_MB": round(tot, 3), "reread_MB": round(rrt, 3),
                    "reread_pct": round(100.0 * rrt / tot, 2)}
    print("BLOCK (pair track) %.1f MB, re-read %.1f MB = %.2f %%" % (tot, rrt, 100.0 * rrt / tot))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
