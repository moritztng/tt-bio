#!/usr/bin/env python3
"""BindCraft2 floor screen: separate the fixed per-step cost from the per-block cost.

One process, one device open, one clock thread, every (n, K, transition) point measured in
it. Sixteen separate processes would pay sixteen device opens and write sixteen files that
cannot be compared, because a timing without its clock, its dims and its git sha is not a
measurement.

Fits t(K) = a + b*K per arm. `a` is everything that is not a repeated unit -- the pair init,
the distogram head, the host loss and the host Adam step -- not "host overhead". `b` is the
per-block cost, and the whole point of the exercise is that `a` does not multiply by depth
while `b` does.

Two discriminators fall out for free:

  1. b(n=256) / b(n=128). One unit's forward is 1536*N^3 + 722944*N^2 FLOPs at AF2's pair
     dims, so a compute-bound `b` must scale by 4.86x. Well below that and `b` is bound by
     per-op dispatch inside the block, which makes trace capture and fusion live levers and
     means the campaign must not be closed on this number.
  2. The achieved rate against a roof measured in THIS process, on the shipped kernel, at the
     shipped shape, on the same card at the same clock. Not a datasheet peak.
"""
import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402
from perf.hallgrad.halloop import run_loop, warm_stats  # noqa: E402

C_Z, HEADS, HEAD_DIM = 128, 4, 32          # AF2's pair dims, tt_bio/af2.py:54-57
AF2_PAIR_BLOCKS = 52                       # 4 extra-MSA + 48 Evoformer, tt_bio/af2.py:587
BC2_GRAD_ROUNDS = 125                      # screen 50 + refine 25 + anneal 45 + harden 5


def block_flops(n, c_z=C_Z, heads=HEADS, head_dim=HEAD_DIM):
    """Forward FLOPs of one `_block` (no transition) at these dims."""
    h = c_z
    trimul = 2 * (2 * n ** 3 * h + 2 * n ** 2 * c_z * (5 * h + c_z))
    triatt = 2 * (4 * n ** 3 * heads * head_dim
                  + 2 * n ** 2 * (4 * c_z * heads * head_dim + c_z * heads
                                  + heads * head_dim * c_z))
    return trimul + triatt


def transition_flops(n, c_z=C_Z):
    """c_z -> 4*c_z twice, 4*c_z -> c_z once, over N^2 positions."""
    return 24 * n ** 2 * c_z ** 2


def measure_matmul_roof(ttnn, device, n, c_z=C_Z, warmup=3, timed=20):
    """[N*N, c_z] @ [c_z, c_z] in bf16 -- the shape the block's linears actually use."""
    import torch
    a = ttnn.from_torch(torch.randn(n * n, c_z).to(torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(c_z, c_z).to(torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(warmup):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(timed):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(device)
    dt = time.perf_counter() - t0
    flops = timed * 2 * (n * n) * c_z * c_z
    return {"n": n, "shape": [n * n, c_z, c_z], "calls": timed, "seconds": dt,
            "tflops": flops / dt / 1e12, "per_call_ms": dt / timed * 1e3}


def fit(ks, ts):
    """Least squares t = a + b*K, with the residual and the two-point slope check."""
    ks, ts = np.asarray(ks, float), np.asarray(ts, float)
    A = np.vstack([np.ones(len(ks)), ks]).T
    (a, b), *_ = np.linalg.lstsq(A, ts, rcond=None)
    pred = a + b * ks
    maxrel = float(np.max(np.abs(ts - pred) / ts))
    ss_res = float(((ts - pred) ** 2).sum())
    ss_tot = float(((ts - ts.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"a": float(a), "b": float(b), "maxrel": maxrel, "r2": float(r2),
           "K": ks.tolist(), "t": ts.tolist()}
    if len(ks) >= 4:
        b_lo = (ts[1] - ts[0]) / (ks[1] - ks[0])
        b_hi = (ts[3] - ts[2]) / (ks[3] - ks[2])
        out["b_lo"], out["b_hi"] = float(b_lo), float(b_hi)
        out["slope_drift"] = float(abs(b_hi / b_lo - 1.0)) if b_lo else float("nan")
        out["linear"] = bool(maxrel <= 0.08 and r2 >= 0.99 and out["slope_drift"] <= 0.20)
    else:
        out["linear"] = None
    return out


def analyze(blob):
    """Fit every arm in a finished sweep and write the arithmetic into the blob."""
    pts = [p for p in blob["points"] if p.get("ok")]
    fits = {}
    for ck in (False, True):
        for bt in (False, True):
            for n in sorted({p["n"] for p in pts}):
                sel = sorted([p for p in pts if p["n"] == n and p["block_transition"] == bt
                              and p["checkpoint"] == ck], key=lambda p: p["blocks"])
                if len(sel) < 2:
                    continue
                ks = [p["blocks"] for p in sel]
                key = f"n{n}_transition{'on' if bt else 'off'}" + ("_ckpt" if ck else "")
                fits[key] = {
                    "n": n, "block_transition": bt, "checkpoint": ck,
                    "step": fit(ks, [p["warm_step"]["mean"] for p in sel]),
                    "fwd": fit(ks, [p["warm_fwd"]["mean"] for p in sel]),
                    "bwd": fit(ks, [p["warm_bwd"]["mean"] for p in sel]),
                    "reached": [p["reached"] for p in sel],
                }
    blob["fits"] = fits

    arith = {}
    for n in sorted({p["n"] for p in pts}):
        key = f"n{n}_transitionon"
        if key not in fits:
            continue
        f = fits[key]
        a, b = f["step"]["a"], f["step"]["b"]
        af, bf_ = f["fwd"]["a"], f["fwd"]["b"]
        t_grad = a + AF2_PAIR_BLOCKS * b
        t_fwd = af + AF2_PAIR_BLOCKS * bf_
        t_bc2 = t_grad + t_fwd
        arith[f"n{n}"] = {
            "a_step_s": a, "b_step_s": b, "a_fwd_s": af, "b_fwd_s": bf_,
            "T_grad_step_s": t_grad, "T_fwd_only_s": t_fwd, "T_bc2_step_s": t_bc2,
            "per_trajectory_s": BC2_GRAD_ROUNDS * t_bc2,
            "per_trajectory_min": BC2_GRAD_ROUNDS * t_bc2 / 60.0,
            "linear": f["step"]["linear"],
        }
    blob["arithmetic"] = arith

    # discriminator 1: does b scale like the FLOPs?
    d1 = {}
    for bt in (False, True):
        tag = "on" if bt else "off"
        k1, k2 = f"n128_transition{tag}", f"n256_transition{tag}"
        if k1 in fits and k2 in fits:
            b1, b2 = fits[k1]["step"]["b"], fits[k2]["step"]["b"]
            unit1 = block_flops(128) + (transition_flops(128) if bt else 0)
            unit2 = block_flops(256) + (transition_flops(256) if bt else 0)
            d1[f"transition_{tag}"] = {
                "b_n128_s": b1, "b_n256_s": b2,
                "measured_ratio": b2 / b1 if b1 else float("nan"),
                "flops_ratio": unit2 / unit1,
                "unit_flops_n128": unit1, "unit_flops_n256": unit2,
            }
    blob["discriminator_1_scaling"] = d1

    # discriminator 2: achieved rate vs the roof measured in this same process
    d2 = {}
    for n in sorted({p["n"] for p in pts}):
        key = f"n{n}_transitionon"
        roof = blob.get("matmul_roof", {}).get(str(n)) or blob.get("matmul_roof", {}).get(n)
        if key not in fits or not roof:
            continue
        b = fits[key]["step"]["b"]
        unit = block_flops(n) + transition_flops(n)
        achieved = 3.0 * unit / b / 1e12 if b else float("nan")
        d2[f"n{n}"] = {"unit_fwd_flops": unit, "b_s": b,
                       "achieved_tflops_fwd_plus_bwd": achieved,
                       "roof_tflops": roof["tflops"],
                       "fraction_of_roof": achieved / roof["tflops"] if roof["tflops"] else None}
    blob["discriminator_2_roof"] = d2
    return blob


def git_sha(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm-from", type=int, default=5)
    ap.add_argument("--ns", default="128,256")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--transitions", default="on,off",
                    help="which transition arms to run, in order")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--checkpoint", action="store_true",
                    help="run every point checkpointed (never fit these with uncheckpointed)")
    ap.add_argument("--analyze", default=None,
                    help="re-fit an existing sweep JSON and exit; no device is opened")
    args = ap.parse_args()

    if args.analyze:
        blob = json.load(open(args.analyze))
        analyze(blob)
        json.dump(blob, open(args.out, "w"), indent=1)
        print(json.dumps({k: blob[k] for k in
                          ("fits", "arithmetic", "discriminator_1_scaling",
                           "discriminator_2_roof") if k in blob}, indent=1))
        return 0

    root = pathlib.Path(__file__).resolve().parents[2]
    ns = [int(x) for x in args.ns.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    arms = [x.strip() == "on" for x in args.transitions.split(",")]

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    clocks = ClockTrace(period=1.0).start()

    blob = {
        "what": "BindCraft2 p2 floor screen: fixed vs per-block cost of a taped pairformer "
                "gradient step at AF2's pair dims.",
        "argv": " ".join(sys.argv), "host": socket.gethostname(),
        "git_sha": git_sha(root), "python": sys.executable,
        "tt_visible_devices": __import__("os").environ.get("TT_VISIBLE_DEVICES"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "c_z": C_Z, "heads": HEADS, "head_dim": HEAD_DIM, "chunk": args.chunk,
        "steps": args.steps, "warm_from": args.warm_from, "seed": args.seed,
        "checkpoint": args.checkpoint, "points": [], "matmul_roof": {},
    }

    def flush():
        blob["clock_meta"] = clocks.summary()
        blob["clock_samples"] = clocks.samples
        json.dump(blob, open(args.out, "w"), indent=1)

    for bt in arms:
        for n in ns:
            for k in ks:
                tag = (f"n={n} K={k} transition={'on' if bt else 'off'}"
                       f"{' ckpt' if args.checkpoint else ''}")
                print(f"\n=== {tag} ===", flush=True)
                t0 = time.time()
                try:
                    r = run_loop(ag, ttnn, device, n=n, c_z=C_Z, c_s=384, heads=HEADS,
                                 head_dim=HEAD_DIM, bins=64, chunk=args.chunk,
                                 steps=args.steps, lr=0.1, contact_bins=20, min_sep=6,
                                 seed=args.seed, checkpoint=args.checkpoint, blocks=k,
                                 block_transition=bt, clocks=clocks, verbose=False)
                except Exception as e:
                    print(f"!!! {tag} FAILED: {type(e).__name__}: {e}", flush=True)
                    blob["points"].append({"n": n, "blocks": k, "block_transition": bt,
                                           "checkpoint": args.checkpoint, "ok": False,
                                           "error": f"{type(e).__name__}: {e}",
                                           "wall_s": time.time() - t0})
                    flush()
                    continue
                wf = args.warm_from
                t_warm0 = r["step_starts"][wf] if len(r["step_starts"]) > wf else r["t_start"]
                pt = {
                    "n": n, "blocks": k, "block_transition": bt,
                    "checkpoint": args.checkpoint, "ok": True,
                    "steps": args.steps, "warm_from": wf,
                    "step_1_s": r["step_times"][0],
                    "warm_step": warm_stats(r["step_times"], wf),
                    "warm_fwd": warm_stats(r["fwd_times"], wf),
                    "warm_bwd": warm_stats(r["bwd_times"], wf),
                    "reached": r["reached"],
                    "loss_first": r["traj"][0], "loss_last": r["traj"][-1],
                    "t_start": r["t_start"], "t_end": r["t_end"],
                    "clock_window_warm": clocks.window(t_warm0, r["t_end"]),
                    "clock_window_all": clocks.window(r["t_start"], r["t_end"]),
                    "step_times": r["step_times"], "fwd_times": r["fwd_times"],
                    "bwd_times": r["bwd_times"], "wall_s": time.time() - t0,
                }
                cw = pt["clock_window_warm"]
                pt["clock_ok"] = bool(cw.get("attributable") and cw.get("median", 0) >= 1200)
                blob["points"].append(pt)
                print(f"    step1 {pt['step_1_s']:.2f} s | warm mean {pt['warm_step']['mean']:.4f} s"
                      f" (fwd {pt['warm_fwd']['mean']:.4f} bwd {pt['warm_bwd']['mean']:.4f})"
                      f" | reached {pt['reached']}"
                      f" | AICLK {cw.get('median')} MHz x{cw.get('samples')}"
                      f" | clock_ok {pt['clock_ok']}", flush=True)
                flush()

    for n in ns:
        blob["matmul_roof"][str(n)] = measure_matmul_roof(ttnn, device, n)
        print(f"roof n={n}: {blob['matmul_roof'][str(n)]['tflops']:.2f} TFLOP/s "
              f"bf16 [{n*n},{C_Z}]@[{C_Z},{C_Z}]", flush=True)

    clocks.stop()
    blob["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    analyze(blob)
    flush()
    print("\n" + json.dumps({k: blob[k] for k in
                             ("fits", "arithmetic", "discriminator_1_scaling",
                              "discriminator_2_roof") if k in blob}, indent=1))
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
