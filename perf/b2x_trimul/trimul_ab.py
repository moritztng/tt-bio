#!/usr/bin/env python3
"""P7 step 1: does moving the trimul's pair mask past the channel move delete bytes, and do the
deleted bytes come back as time?

One process, one device open. Three phases, each independently useful:

  parity -- torch.equal of the whole TriangleMultiplication output, arm off vs arm on, with a
            random 0/1 mask (the hard case) and with an all-ones mask (the published cell). The
            commuting argument is the load-bearing claim for steps 2 and 3 as well, so it is
            checked on the module and not on the kernel.
  bytes  -- ttnn.graph capture of one settled call per arm, counted with the CORRECTED rule
            (dedupe on buffer address, views free): perf/b2x_difflayer/real_traffic.py.
  time   -- interleaved A/B/A/B, medians, with the incumbent run twice under two labels so the
            session reports its own A/A floor.

No fold here. The fold arm is a separate run; this is the cheap screen that fires the falsifier.
"""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.reblock_permute as RB
from real_traffic import counts

CZ, HIDDEN = 128, 128


def weights(cz=CZ, hidden=HIDDEN, seed=0):
    g = torch.Generator().manual_seed(seed)
    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float32) * 0.05
    return {
        "norm_in.weight": torch.ones(cz), "norm_in.bias": torch.zeros(cz),
        "norm_out.weight": torch.ones(hidden), "norm_out.bias": torch.zeros(hidden),
        "g_in.weight": r(2 * hidden, cz), "p_in.weight": r(2 * hidden, cz),
        "g_out.weight": r(cz, cz), "p_out.weight": r(cz, hidden),
    }


def make_inputs(dev, N, mask_kind, seed=1):
    g = torch.Generator().manual_seed(seed)
    zt = torch.randn(1, N, N, CZ, generator=g, dtype=torch.float32)
    z = ttnn.from_torch(zt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if mask_kind == "ones":
        mt = torch.ones(1, N, N)
    else:
        s = (torch.rand(1, N, generator=g) > 0.25).float()
        mt = s[:, :, None] * s[:, None, :]
    m = ttnn.from_torch(mt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return z, m


def sha(t):
    return hashlib.sha256(ttnn.to_torch(t).to(torch.bfloat16).contiguous()
                          .view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def run_once(mod, z, m):
    out = mod(z, m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--phases", default="parity,bytes,time")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phases = a.phases.split(",")

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ck = T.trunk_compute_kernel_config(compute_kernel_config())
    res = {"n": a.n, "reps": a.reps, "host": "qb2", "card": 2,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    W = weights()
    mods = {e: T.TriangleMultiplication(e, W, ck) for e in (False, True)}

    # ---------------- parity ----------------
    if "parity" in phases:
        out = {}
        for mk in ("ones", "random"):
            z, m = make_inputs(dev, a.n, mk)
            cell = {}
            for ending in (False, True):
                hs = {}
                for arm in (False, True):
                    prev = T.set_trimul_mask_after_move(arm)
                    r = run_once(mods[ending], z, m)
                    ttnn.synchronize_device(dev)
                    hs["on" if arm else "off"] = sha(r)
                    ttnn.deallocate(r)
                    T.set_trimul_mask_after_move(prev)
                # Reuse control: the same mask tensor, three consecutive arm-on calls. This is
                # what catches an arm that frees a caller-owned buffer -- a view deallocated
                # inside the trimul reads clean on its first call and garbage on every later one,
                # which is precisely the pair mask's lifetime in a fold.
                T.set_trimul_mask_after_move(True)
                reuse = []
                for _ in range(3):
                    r = run_once(mods[ending], z, m)
                    ttnn.synchronize_device(dev)
                    reuse.append(sha(r))
                    ttnn.deallocate(r)
                T.set_trimul_mask_after_move(False)
                cell["ending" if ending else "starting"] = dict(
                    hs, equal=hs["on"] == hs["off"], reuse=reuse,
                    reuse_stable=len(set(reuse)) == 1 and reuse[0] == hs["off"])
            ttnn.deallocate(z); ttnn.deallocate(m)
            out[mk] = cell
        res["parity"] = out
        print(json.dumps(out, indent=1), flush=True)

    z, m = make_inputs(dev, a.n, "ones")

    # ---------------- bytes ----------------
    if "bytes" in phases:
        out = {}
        for ending in (False, True):
            cell = {}
            for arm in (False, True):
                prev = T.set_trimul_mask_after_move(arm)
                ttnn.deallocate(run_once(mods[ending], z, m))   # warm/compile
                ttnn.synchronize_device(dev)
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                r = run_once(mods[ending], z, m)
                ttnn.synchronize_device(dev)
                g = ttnn.graph.end_graph_capture()
                ttnn.deallocate(r)
                c = counts({"sig": "trimul", "nodes": g})
                cell["on" if arm else "off"] = {
                    k: c[k] for k in ("real_MB", "real_w_MB", "real_r_MB", "n_ops")}
                cell[("on" if arm else "off") + "_top"] = [
                    [round(t / 1e6, 3), nm, round(w / 1e6, 3), round(rr / 1e6, 3)]
                    for t, nm, w, rr in c["per_op"][:12]]
                T.set_trimul_mask_after_move(prev)
            d = cell["off"]["real_MB"] - cell["on"]["real_MB"]
            cell["deleted_MB"] = round(d, 3)
            cell["deleted_Z"] = round(d / 67.108864, 3)
            out["ending" if ending else "starting"] = cell
            print(json.dumps({k: v for k, v in cell.items() if not k.endswith("_top")}),
                  flush=True)
        res["bytes"] = out

    # ---------------- time ----------------
    if "time" in phases:
        out = {}
        for ending in (False, True):
            for arm in (False, True):          # compile both arms before timing
                prev = T.set_trimul_mask_after_move(arm)
                for _ in range(2):
                    ttnn.deallocate(run_once(mods[ending], z, m))
                T.set_trimul_mask_after_move(prev)
            ttnn.synchronize_device(dev)
            samples = {"A": [], "B": [], "A2": []}
            order = ["A", "B", "A2"]
            for _ in range(a.reps):
                for lab in order:
                    T.set_trimul_mask_after_move(lab == "B")
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    r = run_once(mods[ending], z, m)
                    ttnn.synchronize_device(dev)
                    samples[lab].append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(r)
                    T.set_trimul_mask_after_move(False)
            med = {k: round(statistics.median(v), 4) for k, v in samples.items()}
            out["ending" if ending else "starting"] = {
                "median_ms": med, "raw_ms": {k: [round(x, 4) for x in v]
                                             for k, v in samples.items()},
                "AA_floor_pct": round(100 * (med["A2"] - med["A"]) / med["A"], 3),
                "speedup_A_over_B": round(med["A"] / med["B"], 4),
                "deleted_ms": round(med["A"] - med["B"], 4),
            }
            print(json.dumps(out["ending" if ending else "starting"]["median_ms"]
                             | {"floor%": out["ending" if ending else "starting"]["AA_floor_pct"],
                                "x": out["ending" if ending else "starting"]["speedup_A_over_B"]}),
                  flush=True)
        res["time"] = out

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
