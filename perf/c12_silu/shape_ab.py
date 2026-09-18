#!/usr/bin/env python3
"""Price every fused activation the fold actually executes, on the shape it actually executes.

The basis this row was opened on priced `Transition.swiglu` fc1 at a=[1,16,512,128] x 8,960 calls.
An instrumented fold (perf/c12_silu/sites_*.json) says that shape ships at no size: the pair-track
chunk height is 64 rows at 298 aa, 47 at 512 and 32 at 768, the site fires on five shapes, and
there are 3,762 fused-silu calls at 512 aa. So the arm list here is DERIVED from those artifacts
rather than written down, and the fold seconds come from each shape's own measured price times its
own executed call count -- no per-element extrapolation, no per-call price carried across a
chunk-height change.

Five arms per shape, interleaved rep by rep, in one process on one device open:

    fused      ttnn.linear(activation=act)              -- what ships
    plain      ttnn.linear(activation=None)
    act        the standalone in-place SFPU pass alone
    unfused    plain + in-place act                     -- the lever's real path, both dispatches
    fused_aa   fused again, later in the rep sequence   -- this shape's own noise floor

`tax = fused - plain` and `win = fused - unfused`. The A/A arm is per shape, so a shape whose win
is smaller than its own floor is reported as noise rather than as a result.

Each site's config is reproduced, because a fused epilogue's price is program-config dependent:
swiglu fc1 is L1-in/L1-out with an explicit dtype and no bias, while the AdaLN output projection
at :9610 and the atom-to-token projection at :10705 take a bias and leave the output in DRAM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clk                                                                    # noqa: E402

# per call site: the production config, read off tenstorrent.py at the line the graph reported
SITES = {
    "tenstorrent.py:8211":  {"l1": True,  "bias": False, "dtype": True,
                             "what": "Transition.swiglu fc1"},
    "tenstorrent.py:9610":  {"l1": False, "bias": True,  "dtype": False,
                             "what": "AdaLN output projection"},
    "tenstorrent.py:10705": {"l1": False, "bias": False, "dtype": False,
                             "what": "atom-to-token projection"},
    "tenstorrent.py:11896": {"l1": False, "bias": False, "dtype": False,
                             "what": "PairConditioningDevice"},
}


def out_elements(a_shape: str, b_shape: str) -> int:
    a = [int(d) for d in a_shape.split("x")]
    b = [int(d) for d in b_shape.split("x")]
    assert a[-1] == b[0], f"contraction mismatch {a_shape} x {b_shape}"
    return math.prod(a[:-1]) * b[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--max-out-mb", type=float, default=64.0,
                    help="skip a shape whose output exceeds this; the two skipped ones fire twice "
                         "per fold and are worth ~0.003 s")
    args = ap.parse_args()

    import torch
    import ttnn
    import tt_bio as _TB
    from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    # the arm list, derived from the executed graphs; one entry per distinct (site, a, b, act)
    combos: dict = {}
    per_size: dict = {}
    for p in args.sites:
        d = json.loads(p.read_text())
        size = d["size"]
        per_size[size] = []
        for r in d["rows"]:
            if r["activation"] in ("None", None):
                continue
            k = (r["site"], r["a_shape"], r["b_shape"], r["activation"])
            combos.setdefault(k, 0)
            combos[k] += r["calls"]
            per_size[size].append({"site": r["site"], "a_shape": r["a_shape"],
                                   "b_shape": r["b_shape"], "activation": r["activation"],
                                   "calls": r["calls"]})
    print(f"{len(combos)} distinct (site, a, b, activation) combos across "
          f"{sorted(per_size)} aa\n", flush=True)

    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    node = nodes[0]
    clk.force(args.mhz, [node])
    sampler = clk.Sampler(node)
    L1 = ttnn.L1_MEMORY_CONFIG
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    ACT = {"silu": ttnn.silu, "sigmoid": ttnn.sigmoid, "relu": ttnn.relu}
    torch.manual_seed(0)

    def dev_tensor(shape, mc):
        return ttnn.to_device(ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16),
                                              layout=ttnn.TILE_LAYOUT), dev, memory_config=mc)

    results = {}
    for (site, a_shape, b_shape, act) in sorted(combos):
        cfg = SITES[site]
        els = out_elements(a_shape, b_shape)
        out_mb = els * 2 / 1e6
        key = f"{site}|{a_shape}|{b_shape}|{act}"
        if out_mb > args.max_out_mb:
            results[key] = {"skipped": f"output {out_mb:.1f} MB over the {args.max_out_mb} MB cap"}
            print(f"  SKIP {key}  output {out_mb:.1f} MB", flush=True)
            continue
        mc = L1 if cfg["l1"] else DRAM
        a_dims = [int(x) for x in a_shape.split("x")]
        b_dims = [int(x) for x in b_shape.split("x")]
        try:
            A = dev_tensor(tuple(a_dims), mc)
            W = dev_tensor(tuple(b_dims), mc)
            BI = dev_tensor((1, b_dims[-1]), mc) if cfg["bias"] else None
            X = dev_tensor(tuple(a_dims[:-1] + [b_dims[-1]]), mc)
        except Exception as e:
            results[key] = {"skipped": f"allocation failed: {str(e)[:200]}"}
            print(f"  SKIP {key}  {str(e)[:120]}", flush=True)
            continue
        kw = dict(compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN, memory_config=mc)
        if cfg["bias"]:
            kw["bias"] = BI
        if cfg["dtype"]:
            kw["dtype"] = ttnn.bfloat16
        f = ACT[act]

        def lin(a):
            o = ttnn.linear(A, W, activation=a, **kw)
            ttnn.deallocate(o)

        def unfused():
            o = ttnn.linear(A, W, activation=None, **kw)
            f(o, memory_config=mc, output_tensor=o)
            ttnn.deallocate(o)

        ARMS = {"fused": lambda: lin(act), "plain": lambda: lin(None),
                "act": lambda: f(X, memory_config=mc, output_tensor=X),
                "unfused": unfused, "fused_aa": lambda: lin(act)}
        ORDER = ["fused", "plain", "act", "unfused", "fused_aa"]
        try:
            for _ in range(2):
                for n in ORDER:
                    ARMS[n]()
                ttnn.synchronize_device(dev)
            res = {n: [] for n in ORDER}
            for _ in range(args.reps):
                for n in ORDER:
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(args.calls):
                        ARMS[n]()
                    ttnn.synchronize_device(dev)
                    res[n].append((time.perf_counter() - t0) / args.calls * 1e3)
        except Exception as e:
            results[key] = {"skipped": f"run failed: {str(e)[:200]}"}
            print(f"  SKIP {key}  {str(e)[:120]}", flush=True)
            for T in (A, W, X):
                ttnn.deallocate(T)
            continue
        m = {n: statistics.median(res[n]) for n in ORDER}
        tax, win = m["fused"] - m["plain"], m["fused"] - m["unfused"]
        results[key] = {
            "site": site, "what": cfg["what"], "a_shape": a_shape, "b_shape": b_shape,
            "activation": act, "memory": "L1" if cfg["l1"] else "DRAM", "bias": cfg["bias"],
            "out_elements": els, "out_MB": round(out_mb, 2),
            "ms_per_call": {n: round(m[n], 5) for n in ORDER},
            "tax_ms": round(tax, 5), "win_ms": round(win, 5),
            "act_ms": round(m["act"], 5),
            "aa_ms": round(abs(m["fused"] - m["fused_aa"]), 6),
            "aa_pct": round(100.0 * abs(m["fused"] - m["fused_aa"]) / m["fused"], 3),
            "win_over_own_floor": (round(win / abs(m["fused"] - m["fused_aa"]), 1)
                                   if m["fused"] != m["fused_aa"] else None),
            "tax_ps_per_element": round(tax / 1e3 / els * 1e12, 3),
            "win_ps_per_element": round(win / 1e3 / els * 1e12, 3),
            "total_calls_seen": combos[(site, a_shape, b_shape, act)],
        }
        print("  %-52s fused %7.4f plain %7.4f act %7.4f unfused %7.4f | tax %7.4f win %7.4f "
              "A/A %.3f%% (%sx)" % (key, m["fused"], m["plain"], m["act"], m["unfused"], tax, win,
                                    results[key]["aa_pct"], results[key]["win_over_own_floor"]),
              flush=True)
        for T in (A, W, X):
            ttnn.deallocate(T)
        if BI is not None:
            ttnn.deallocate(BI)

    aiclk = sampler.stop()
    clk.release()

    # fold seconds per size, from each shape's own measured price and its own executed call count
    folds = {}
    for size, rows in sorted(per_size.items()):
        acc = {}
        for r in rows:
            key = f"{r['site']}|{r['a_shape']}|{r['b_shape']}|{r['activation']}"
            m = results.get(key, {})
            act = r["activation"]
            e = acc.setdefault(act, {"tax_s": 0.0, "win_s": 0.0, "calls": 0,
                                     "priced_calls": 0, "unpriced": []})
            e["calls"] += r["calls"]
            if "tax_ms" in m:
                e["tax_s"] += m["tax_ms"] * r["calls"] / 1e3
                e["win_s"] += m["win_ms"] * r["calls"] / 1e3
                e["priced_calls"] += r["calls"]
            else:
                e["unpriced"].append({"shape": key, "calls": r["calls"],
                                      "why": m.get("skipped", "not measured")})
        for act, e in acc.items():
            e["tax_s"] = round(e["tax_s"], 4)
            e["win_s"] = round(e["win_s"], 4)
            e["win_mcycles"] = round(e["win_s"] * args.mhz, 1)
            e["coverage_pct"] = round(100.0 * e["priced_calls"] / e["calls"], 2)
        folds[size] = acc

    out = {"doc": __doc__, "clock_forced_mhz": args.mhz, "aiclk_during_session": aiclk,
           "calls_per_arm": args.calls, "reps": args.reps, "card_node": node,
           "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
           "shapes": results, "fold_seconds_by_size": folds}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("\nFOLDS " + json.dumps(folds, indent=1))
    print("AICLK " + json.dumps(aiclk))
    print("wrote", args.out)
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
