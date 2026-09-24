#!/usr/bin/env python3
"""The five batched matmuls of one triangle-attention chunk: shipped vs an explicit batched
program config, graded against float64 and timed in windows long enough to carry an AICLK sample.

fwd  S  = q @ k^T         [B,H,nq,d] x [B,H,nk,d]^T
fwd  O  = P @ v           [B,H,nq,nk] x [B,H,nk,d]
bwd  dV = P^T @ dO        transpose_a
bwd  dP = dO @ v^T        transpose_b
bwd  dQ = dS @ k
bwd  dK = dS^T @ q        transpose_a
"""
import argparse, json, statistics, sys, time, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--nq", type=int, default=128)
    ap.add_argument("--nk", type=int, default=256)
    ap.add_argument("--grad-fp32", action="store_true", help="dO and dS in fp32, as a backward may hand them")
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.autograd import precise_config, bmm_program_config
    dev = tt.get_device()
    clocks = ClockTrace(period=0.5).start()
    hifi = precise_config()
    g = torch.Generator().manual_seed(0)
    B, H, nq, nk, d = args.B, 4, args.nq, args.nk, 32
    host = {}

    def T(name, *s, scale=1.0):
        fp32 = args.grad_fp32 and name in ("dO", "dS")
        x = (torch.randn(*s, generator=g) * scale).to(torch.float32 if fp32 else torch.bfloat16)
        host[name] = x
        return ttnn.from_torch(x, dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)

    q, k, v = T("q", B, H, nq, d), T("k", B, H, nk, d), T("v", B, H, nk, d)
    P = T("P", B, H, nq, nk, scale=1 / 16)  # softmax-sized entries
    dO = T("dO", B, H, nq, d)
    dS = T("dS", B, H, nq, nk, scale=1 / 16)
    cases = {
        "S=qk^T": (q, k, False, True, "q", "k"),
        "O=Pv": (P, v, False, False, "P", "v"),
        "dV=P^TdO": (P, dO, True, False, "P", "dO"),
        "dP=dOv^T": (dO, v, False, True, "dO", "v"),
        "dQ=dSk": (dS, k, False, False, "dS", "k"),
        "dK=dS^Tq": (dS, q, True, False, "dS", "q"),
    }
    res = {"shape": [B, H, nq, nk, d], "cases": {}}
    for name, (a, b, ta, tb, an, bn) in cases.items():
        A64, B64 = host[an].double(), host[bn].double()
        ref = (A64.transpose(-1, -2) if ta else A64) @ (B64.transpose(-1, -2) if tb else B64)
        M, K = ref.shape[-2], A64.shape[-2] if ta else A64.shape[-1]
        N = ref.shape[-1]
        flops = 2.0 * B * H * M * N * K
        variants = {"shipped": lambda: ttnn.matmul(a, b, transpose_a=ta, transpose_b=tb,
                                                   compute_kernel_config=hifi)}
        cfg = bmm_program_config(a, b, ta, tb)
        if cfg is not None:
            variants["bmm_cfg"] = lambda: ttnn.matmul(a, b, transpose_a=ta, transpose_b=tb,
                                                      compute_kernel_config=hifi,
                                                      program_config=cfg)
        out = {}
        base = None
        for vn, fn in variants.items():
            try:
                y = ttnn.to_torch(fn()).double()
                err = (y - ref)
                rel = float(err.norm() / ref.norm())
                if base is None:
                    base = y
                exact = bool(torch.equal(y, base))
                fn(); ttnn.synchronize_device(dev)
                t0 = time.perf_counter(); n = 0
                while time.perf_counter() - t0 < 0.3:
                    fn(); n += 1
                ttnn.synchronize_device(dev)
                per = (time.perf_counter() - t0) / n
                R = max(5, int(0.25 / per))
                ts = []; tw0 = time.time()
                while time.time() - tw0 < args.secs:
                    t0 = time.perf_counter()
                    for _ in range(R):
                        fn()
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / R)
                tw1 = time.time()
                us = statistics.median(ts) * 1e6
                out[vn] = dict(us=us, p10=sorted(ts)[len(ts) // 10] * 1e6,
                               p90=sorted(ts)[9 * len(ts) // 10] * 1e6, n=len(ts), R=R,
                               tflops=flops / us / 1e6, rel_l2_vs_f64=rel,
                               maxabs_vs_f64=float(err.abs().max()), bitexact_vs_shipped=exact,
                               clock=clocks.window(tw0, tw1))
                print(f"{name:10s} {vn:8s} {us:8.1f} us  {out[vn]['tflops']:6.2f} TF/s  "
                      f"relL2(f64)={rel:.3e} exact={exact} clk={out[vn]['clock']}", flush=True)
            except Exception as e:
                out[vn] = dict(error=str(e)[:400])
                print(f"{name:10s} {vn:8s} ERROR {str(e)[:300]}", flush=True)
        res["cases"][name] = out
    clocks.stop()
    res["clock_meta"] = clocks.summary()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
