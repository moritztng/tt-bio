#!/usr/bin/env python3
"""per_core_M sweep at the real 298 aa and 512 aa call-site shapes, on this card.

Every legal per_core_M for every class whose `saturating` set has more than one member, i.e. every
class where the shipped `max(saturating)` rule and the occupancy-first `min(saturating)`
alternative disagree. Arms are round-robin inside each rep, the device is synchronised on both
sides of every timed region, and the shipped arm is entered TWICE under two labels so the run
carries its own A/A floor.

Bit-exactness is re-checked here per p, not inherited: the alternative is only interesting if it is
free.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tt_bio.tenstorrent as T          # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32, BF16 = ttnn.float32, ttnn.bfloat16


def sub(p, n):
    w = max(x for x in range(1, min(4, n) + 1) if n % x == 0)
    h = max(x for x in range(1, min(4 // w, p) + 1) if p % x == 0)
    return h, w


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True, help="json list of case dicts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--target-ms", type=float, default=25.0)
    a = ap.parse_args()

    cases = json.loads(a.cases.read_text())
    dev = T.get_device()
    T._configure_active_compute_grid(dev)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {"grid": list(grid), "cores": cores, "l1_unreserved": l1, "reps": a.reps, "cases": []}
    for c in cases:
        sa, sb = tuple(c["a"]), tuple(c["b"])
        dt = F32 if c["dtype"] == "fp32" else BF16
        eb = 4 if dt == F32 else 2
        batch = 1
        for d in sa[:-2]:
            batch *= d
        mt, kt, nt = -(-sa[-2] // 32), -(-sa[-1] // 32), -(-sb[-1] // 32)
        bw = T._batched_matmul_block_w(mt, kt, nt)
        tile, acc = 1024 * eb, 4096
        legal = []
        for p in range(1, mt + 1):
            if mt % p or (p != mt and batch * mt // p > cores):
                continue
            cb = 2 * (p + nt) * bw * tile + p * nt * (tile + acc)
            if cb > l1:
                continue
            legal.append(p)
        shipped = T._batched_matmul_config(batch, mt, kt, nt, eb)
        cur = shipped.per_core_M if shipped is not None else None

        torch.manual_seed(0)
        A = ttnn.from_torch(torch.randn(*sa) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(*sb) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        ref = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc))

        arms = {}
        for p in legal:
            h, w = sub(p, nt)
            arms[f"p{p}"] = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=h,
                out_subblock_w=w, per_core_M=p, per_core_N=nt)
        if cur is not None:
            arms[f"p{cur}_aa"] = arms[f"p{cur}"]

        exact = {}
        for name, cfg in arms.items():
            got = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg))
            exact[name] = bool(torch.equal(ref, got))
            del got

        # calibrate iteration count on the shipped arm
        probe = arms.get(f"p{cur}") or next(iter(arms.values()))
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(5):
            r = ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=probe)
            ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        per = (time.perf_counter() - t0) / 5
        iters = max(20, min(2000, int(a.target_ms * 1e-3 / max(per, 1e-6))))

        samples = {k: [] for k in arms}
        for _ in range(a.reps):
            for name, cfg in arms.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(iters):
                    r = ttnn.matmul(A, B, compute_kernel_config=ckc, program_config=cfg)
                    ttnn.deallocate(r)
                ttnn.synchronize_device(dev)
                samples[name].append((time.perf_counter() - t0) / iters * 1e3)

        ms = {k: round(med(v), 5) for k, v in samples.items()}
        aa = None
        if cur is not None and f"p{cur}_aa" in ms:
            aa = round(abs(ms[f"p{cur}"] - ms[f"p{cur}_aa"]) / ms[f"p{cur}"] * 100, 2)
        rec = dict(c, batch=batch, Mt=mt, Kt=kt, Nt=nt, in0_block_w=bw, legal=legal,
                   shipped_pM=cur, iters=iters, ms=ms,
                   ms_all={k: [round(x, 5) for x in v] for k, v in samples.items()},
                   blocks={f"p{p}": batch * mt // p for p in legal},
                   bit_exact=exact, aa_pct=aa)
        out["cases"].append(rec)
        a.out.write_text(json.dumps(out, indent=2))
        best = min(((v, k) for k, v in ms.items() if not k.endswith("_aa")))
        print(f"{c['label']:34s} legal={legal} shipped=p{cur} A/A={aa}% "
              + " ".join(f"{k}={v}" for k, v in ms.items())
              + f" | best={best[1]} exact={all(exact.values())}", flush=True)
        ttnn.deallocate(A)
        ttnn.deallocate(B)
        del ref

    a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
