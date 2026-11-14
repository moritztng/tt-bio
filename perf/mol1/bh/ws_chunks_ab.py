"""The pair Transition as Mol-1 BH serves it, equal vs ragged weight-stationary hidden chunks: ms and error vs float64.

    TT_VISIBLE_DEVICES=<card> python perf/mol1/bh/ws_chunks_ab.py OUT.json --cells 256x320,512x448,1024x448 [--reps 10]
        [--arms equal,ragged,12-12-16-16]

Each cell is N x c_z on the served lever set for its size (NORMAL | WIDE_NORMAL | MOL1_BH, plus MOL1_BH_512 from 512
tokens, as `size_levers` adds it), the call the Pairformer makes: `transition_z(z, add_to_input=True)`. Arm `equal`
runs the split `pair_swiglu_ws.weights` took before ragged chunks (the fewest equal chunks that fit), arm `ragged`
`hidden_chunks`, and an arm like `12-12-16-16` those hidden tiles per chunk in that order (cells whose hidden
tiles it does not sum to take ragged). Random weights at 1/sqrt(fan-in), z ~ N(0, 4). Error is rel_rms of the update (out - z) against
float64; AICLK is sampled from sysfs during the timed reps.
"""
import argparse
import glob
import json
import os
import pathlib
import statistics
import sys
import threading
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def equal_chunks(W):
    def f(kt, nh, ct, dtype):
        p = next((p for p in range(1, nh // W._SW + 1) if nh % (p * W._SW) == 0
                  and W._l1_bytes(kt, nh // p, ct, dtype) <= W._L1_BUDGET), 1)
        return [nh // p] * p
    return f


def chunker(W, arm, ragged):
    if arm == "equal":
        return equal_chunks(W)
    if arm == "ragged":
        return ragged
    sizes = [int(v) for v in arm.split("-")]
    return lambda kt, nh, ct, dtype: sizes if sum(sizes) == nh else ragged(kt, nh, ct, dtype)


def ref(x, sd):
    x = x.double()
    xn = torch.nn.functional.layer_norm(x, x.shape[-1:], sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
    h = torch.nn.functional.silu(xn @ sd["fc1.weight"].double().t()) * (xn @ sd["fc2.weight"].double().t())
    return h @ sd["fc3.weight"].double().t()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--cells", default="256x320,512x448,1024x448")
    p.add_argument("--arms", default="equal,ragged")
    p.add_argument("--reps", type=int, default=10)
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import pair_swiglu_ws as W
    from tt_bio import protenix as P
    from tt_bio import tenstorrent as TT
    dev = TT.get_device()
    clk_paths = sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            samples.append([int(open(f).read()) for f in clk_paths])
            time.sleep(0.2)
    threading.Thread(target=poll, daemon=True).start()
    ragged = W.hidden_chunks
    res = {"host": os.uname().nodename, "sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "visible": os.environ.get("TT_VISIBLE_DEVICES"), "cells": {}}
    g = torch.Generator().manual_seed(0)
    for cell in A.cells.split(","):
        N, c = (int(v) for v in cell.split("x"))
        lv = P.NORMAL_LEVERS | P.WIDE_NORMAL | P.MOL1_BH | (P.MOL1_BH_512 if N >= P.SIZE_LEVERS_MIN_TOKENS else set())
        sd = {"norm.weight": 1 + 0.1 * torch.randn(c, generator=g), "norm.bias": 0.1 * torch.randn(c, generator=g),
              "fc1.weight": torch.randn(4 * c, c, generator=g) * c ** -0.5,
              "fc2.weight": torch.randn(4 * c, c, generator=g) * c ** -0.5,
              "fc3.weight": torch.randn(c, 4 * c, generator=g) * (4 * c) ** -0.5}
        z0 = (2 * torch.randn(1, N, N, c, generator=g)).to(torch.bfloat16)
        want = ref(z0, sd)
        row = {"levers": sorted(lv)}
        with TT.levers(lv):
            base = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          fp32_dest_acc_en=True, packer_l1_acc=True)
            ckc = TT.trunk_compute_kernel_config(base)
            mods = {}
            for arm in A.arms.split(","):
                W.hidden_chunks = chunker(W, arm, ragged)
                mods[arm] = TT.Transition(sd, ckc)
                row[arm] = {"ms": [], "clk": []}
            for rep in range(A.reps + 1):
                for arm, m in mods.items():
                    W.hidden_chunks = chunker(W, arm, ragged)
                    z = ttnn.from_torch(z0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                    ttnn.synchronize_device(dev)
                    n0 = len(samples)
                    t0 = time.perf_counter()
                    out = m(z, add_to_input=True)
                    ttnn.synchronize_device(dev)
                    ms = (time.perf_counter() - t0) * 1e3
                    if rep == 0:
                        got = ttnn.to_torch(out).double() - z0.double()
                        row[arm]["chunks"] = [int(w3.shape[0]) // 32 for _, w3 in m._ws] if m._ws else None
                        row[arm]["rel_rms"] = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt())
                        row[arm]["out"] = ttnn.to_torch(out)
                    else:
                        row[arm]["ms"].append(round(ms, 3))
                        row[arm]["clk"] += samples[n0:]
                    ttnn.deallocate(out)
            W.hidden_chunks = ragged
        arms = [a for a in row if a != "levers"]
        for a in arms:
            clk = [v for s in row[a].pop("clk") for v in s]
            row[a]["ms_median"] = statistics.median(row[a]["ms"])
            row[a]["aiclk"] = [min(clk), statistics.median(clk), max(clk)] if clk else None
        row["equal_out"] = bool(torch.equal(row[arms[0]].pop("out"), row[arms[-1]].pop("out")))
        for a in arms[:-1]:
            row[a].pop("out", None)
        res["cells"][cell] = row
        print(cell, {a: (row[a]["chunks"], row[a]["ms_median"], round(row[a]["rel_rms"], 5), row[a]["aiclk"])
                     for a in arms}, "torch.equal", row["equal_out"], flush=True)
    stop.set()
    json.dump(res, open(A.out, "w"), indent=1)


if __name__ == "__main__":
    main()
