"""Wrong-value audit of every device matmul a fold runs: each call site against float64, per pixel.

    python perf/spd_wherr/audit.py --audit-out OUT.jsonl [--per-site 2] [--budget-gflop 40] -- <bench.py args>

Wraps ttnn.matmul, ttnn.linear and ttnn.experimental.minimal_matmul, then runs perf/spd/bench.py with the args after
`--`. The first `--per-site` calls of every distinct (caller file:line, shapes, program config, kernel config) are
pulled to the host and recomputed in float64 from the device's own operands (bias and a string activation included).
Calls inside a trace capture are not touched. A call bigger than the budget is checked on a random subset of output
rows, and the record says which fraction.

A pixel is WRONG when its error is above both 8 bf16 ulps of the reference value and 16x the call's rms error. That
is what the Wormhole fp32-acc erratum writes (an O(1) value off by about 1, 2 or 4 among errors of 1e-3); rounding,
accumulation order and bfp8 block exponents stay under it. Each record also carries the max error and the rms error
relative to the reference rms, so a site can be read without trusting the threshold.

Also counts every ttnn.generic_op call by caller site (those kernels are checked by their own tests, not here).
"""
import json, os, runpy, sys, time
from collections import Counter
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
ACT = {"silu": torch.nn.functional.silu, "relu": torch.relu, "sigmoid": torch.sigmoid,
       "gelu": lambda t: torch.nn.functional.gelu(t)}


def site(depth=3):
    """The first `depth` tt_bio frames above the call, innermost first: the first is often a shared wrapper."""
    f, out = sys._getframe(2), []
    while f is not None and len(out) < depth:
        p = f.f_code.co_filename
        if "/tt_bio/" in p:
            out.append(f"{p.rsplit('/tt_bio/', 1)[1]}:{f.f_lineno}:{f.f_code.co_name}")
        f = f.f_back
    return " < ".join(out) or "?"


def ckc_desc(k):
    if k is None:
        return "default"
    return {a: str(getattr(k, a)) for a in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode")
            if hasattr(k, a)}


def pc_desc(pc):
    if pc is None:
        return "auto"
    out = {"type": type(pc).__name__}
    for a in ("in0_block_w", "per_core_M", "per_core_N", "out_subblock_h", "out_subblock_w", "out_block_h",
              "out_block_w", "compute_with_storage_grid_size", "fuse_batch", "mcast_in0", "K_block_size",
              "M_block_size", "N_block_size", "subblock_h", "subblock_w"):
        if hasattr(pc, a):
            out[a] = str(getattr(pc, a))
    return out


class Audit:
    def __init__(self, out, per_site, budget):
        self.out, self.per_site, self.budget = open(out, "a"), per_site, budget * 1e9
        self.seen, self.generic, self.tracing = Counter(), Counter(), False
        torch.set_num_threads(int(os.environ.get("AUDIT_THREADS", "4")))

    def check(self, op, a, b, y, kw, extra):
        key_site = site()
        key = (op, key_site, tuple(a.shape), tuple(b.shape), json.dumps(extra, sort_keys=True))
        self.seen[key] += 1
        if self.seen[key] > self.per_site:
            return
        t0 = time.time()
        A = ttnn.to_torch(a).double(); B = ttnn.to_torch(b).double(); Y = ttnn.to_torch(y).double()
        if kw.get("transpose_a"):
            A = A.transpose(-1, -2)
        if kw.get("transpose_b"):
            B = B.transpose(-1, -2)
        Y = Y.reshape(*torch.broadcast_shapes(A.shape[:-2], B.shape[:-2]), A.shape[-2], B.shape[-1])
        rows = A.reshape(-1, A.shape[-1]).shape[0] if A.dim() >= 2 else 1
        flops = 2.0 * A.numel() * B.shape[-1]
        frac = 1.0
        if flops > self.budget and B.dim() == 2:   # weight matmul: sample rows of A
            keep = max(1, int(rows * self.budget / flops)); frac = keep / rows
            idx = torch.randperm(rows)[:keep]
            A2, Y2 = A.reshape(-1, A.shape[-1])[idx], Y.reshape(-1, Y.shape[-1])[idx]
        elif flops > self.budget:                  # batched: sample leading batch entries
            lead = A.shape[0]; keep = max(1, int(lead * self.budget / flops)); frac = keep / lead
            idx = torch.randperm(lead)[:keep]
            A2, Y2 = A[idx], Y[idx]
            B = B[idx] if B.dim() == A.dim() and B.shape[0] == lead else B
        else:
            A2, Y2 = A, Y
        R = A2 @ B
        bias = kw.get("bias") if op != "minimal_matmul" else kw.get("bias_tensor")
        if bias is not None:
            R = R + ttnn.to_torch(bias).double().reshape(-1)[: R.shape[-1]]
        act = kw.get("activation")
        if act is not None:
            f = ACT.get(act) if isinstance(act, str) else None
            if f is None:
                self.emit(op=op, site=key_site, a=list(A.shape), b=list(B.shape), cfg=extra, skipped=f"activation {act}")
                return
            R = f(R)
        Y2 = Y2.reshape(R.shape)
        err = (Y2 - R).abs()
        fin = torch.isfinite(Y2)
        rms_ref = R.pow(2).mean().sqrt().item() or 1e-30
        rms_err = err[fin].pow(2).mean().sqrt().item() if fin.any() else float("nan")
        ulp = torch.exp2(torch.floor(torch.log2(R.abs().clamp_min(1e-30))) - 7)
        bad = fin & (err > 8 * ulp) & (err > 16 * rms_err)
        nb = int(bad.sum())
        worst = []
        if nb:
            v, i = err[bad].topk(min(3, nb))
            worst = [[round(float(R[bad][j]), 5), round(float(Y2[bad][j]), 5)] for j in i.tolist()]
        self.emit(op=op, site=key_site, call=self.seen[key], a=list(A.shape), b=list(B.shape), cfg=extra,
                  checked_el=int(R.numel()), frac=round(frac, 5), wrong=nb, nonfinite=int((~fin).sum()),
                  max_err=round(float(err[fin].max()) if fin.any() else float("nan"), 5),
                  max_err_rel=round(float(err[fin].max()) / rms_ref if fin.any() else float("nan"), 5),
                  rms_err_rel=round(rms_err / rms_ref, 7), rms_ref=round(rms_ref, 5), worst=worst,
                  host_s=round(time.time() - t0, 2))

    def emit(self, **kw):
        self.out.write(json.dumps(kw) + "\n"); self.out.flush()

    def wrap(self, op, fn, getab):
        def w(*args, **kw):
            y = fn(*args, **kw)
            if self.tracing:
                return y
            try:
                a, b = getab(args, kw)
                extra = {"pc": pc_desc(kw.get("program_config") or kw.get("config")),
                         "ckc": ckc_desc(kw.get("compute_kernel_config")),
                         "dtype": str(kw.get("dtype")), "a_dtype": str(a.dtype), "b_dtype": str(b.dtype),
                         "core_grid": str(kw.get("core_grid"))}
                self.check(op, a, b, y, kw, extra)
            except Exception as e:  # the audit must never change the fold
                self.emit(op=op, site=site(), error=f"{type(e).__name__}: {str(e)[:300]}")
            return y
        return w

    def install(self):
        mm = lambda args, kw: (args[0] if args else kw["input_tensor_a"], args[1] if len(args) > 1 else kw["input_tensor_b"])
        ttnn.matmul = self.wrap("matmul", ttnn.matmul, mm)
        ttnn.linear = self.wrap("linear", ttnn.linear, mm)
        mmm = lambda args, kw: (args[0] if args else kw["input_tensor"], args[1] if len(args) > 1 else kw["weight_tensor"])
        ttnn.experimental.minimal_matmul = self.wrap("minimal_matmul", ttnn.experimental.minimal_matmul, mmm)
        gen = ttnn.generic_op

        def g(*args, **kw):
            if not self.tracing:
                self.generic[site()] += 1
            return gen(*args, **kw)
        ttnn.generic_op = g
        btc, etc = ttnn.begin_trace_capture, ttnn.end_trace_capture

        def b(*args, **kw):
            self.tracing = True
            return btc(*args, **kw)

        def e(*args, **kw):
            r = etc(*args, **kw); self.tracing = False
            return r
        ttnn.begin_trace_capture, ttnn.end_trace_capture = b, e

    def close(self):
        self.emit(generic_op_sites=dict(self.generic))
        self.out.close()


def main():
    argv = sys.argv[1:]
    cut = argv.index("--")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-out", type=Path, required=True)
    ap.add_argument("--per-site", type=int, default=2)
    ap.add_argument("--budget-gflop", type=float, default=40)
    a = ap.parse_args(argv[:cut])
    au = Audit(a.audit_out, a.per_site, a.budget_gflop)
    au.install()
    sys.argv = [str(REPO / "perf/spd/bench.py")] + argv[cut + 1:]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        au.close()


if __name__ == "__main__":
    main()
