"""RFD4's triangle contraction alone, x[c] = a[c] @ b[c]^T (outgoing) or a[c]^T @ b[c] (incoming), over the
256 hidden channels, per arm: ms a call, rel RMS against a float64 product of the unrounded operands, and the
max |diff| against the first arm. Operands sit as T1 leaves them, [1, C, n, n] TILE in DRAM, the matmul
taking the inner transpose (`transpose_b` outgoing, `transpose_a` incoming).

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/tri3/contract.py --out <json> [--n 512] [--rounds 4] [--reps 8]
        [--arms "mm|mm;fid=lofi|mm;in=b8;fid=hifi2|bres"]

An arm is `;`-joined parts: a method (`mm` = the trunk's program config, `mmfull` = all of K in one block,
`plain` = ttnn's own plan, `bres` = bmm_res), `in=b8` (bfp8 operands), `out=b8` (bfp8 output), `fid=hifi2|lofi`.
Arms are interleaved round by round (`--rounds` x `--reps` calls each), so clock and host drift land on all.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

FIDS = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}


def ckc_with(base, fid):
    if fid is None:
        return base
    c = type(base)(math_fidelity=FIDS[fid], math_approx_mode=base.math_approx_mode,
                   fp32_dest_acc_en=base.fp32_dest_acc_en, packer_l1_acc=base.packer_l1_acc)
    c.dst_full_sync_en = base.dst_full_sync_en
    c.throttle_level = base.throttle_level
    return c


def parse(arm):
    o = dict(method="mm", inp="bf16", out="bf16", fid=None)
    for p in arm.split(";"):
        if p in ("mm", "mmfull", "plain", "bres"):
            o["method"] = p
        elif p.startswith("in="):
            o["inp"] = p[3:]
        elif p.startswith("out="):
            o["out"] = p[4:]
        elif p.startswith("fid="):
            o["fid"] = p[4:]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--dirs", default="out,in")
    ap.add_argument("--arms", default="mm|mm;fid=hifi2|mm;fid=lofi|mm;in=b8|mm;in=b8;fid=hifi2|mm;in=b8;fid=lofi"
                                      "|mm;out=b8|mm;in=b8;out=b8;fid=lofi|mmfull|plain|bres")
    a = ap.parse_args()

    from tt_bio import bmm_res
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4.trunk import compute_kernel_config

    dev = T.get_device()
    base_ckc = compute_kernel_config()
    n, C = a.n, a.c
    g = torch.Generator().manual_seed(0)
    # T1's operands: sigmoid(gate) * projection of a layer-normed pair, so ~N(0, 1) scaled by a (0, 1) gate
    a64 = torch.sigmoid(torch.randn(1, C, n, n, generator=g, dtype=torch.float64)) * torch.randn(
        1, C, n, n, generator=g, dtype=torch.float64)
    b64 = torch.sigmoid(torch.randn(1, C, n, n, generator=g, dtype=torch.float64)) * torch.randn(
        1, C, n, n, generator=g, dtype=torch.float64)
    st = (n + 31) // 32
    pc = T._triangle_mul_program_config(st, False, (1, 1), False)
    pc_full = T._triangle_mul_program_config(st, True, (1, 1), False)
    up = {k: (ttnn.from_torch(a64.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev),
              ttnn.from_torch(b64.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev))
          for k, dt in (("bf16", ttnn.bfloat16), ("b8", ttnn.bfloat8_b))}
    arms = [(s, parse(s)) for s in a.arms.split("|")]
    rows = []
    with clocksample.during() as clk:
        for d in a.dirs.split(","):
            ending = d == "in"
            ref = (a64.transpose(-1, -2) @ b64) if ending else (a64 @ b64.transpose(-1, -2))
            rn = ref.pow(2).mean().sqrt()

            def call(o):
                x, y = up[o["inp"]]
                ckc = ckc_with(base_ckc, o["fid"])
                odt = ttnn.bfloat8_b if o["out"] == "b8" else ttnn.bfloat16
                if o["method"] == "bres":
                    return bmm_res.matmul(x, y, compute_kernel_config=T._bres_ckc(ckc),
                                          transpose_a=ending, transpose_b=not ending)
                kw = dict(compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=odt,
                          transpose_a=ending, transpose_b=not ending)
                if o["method"] != "plain":
                    kw["program_config"] = pc_full if o["method"] == "mmfull" else pc
                return ttnn.matmul(x, y, **kw)

            live, times, first = [], {}, None
            for s, o in arms:
                try:
                    r = call(o)
                    if r is None:
                        raise RuntimeError("declined")
                    got = ttnn.to_torch(r).double().reshape(ref.shape)
                    ttnn.deallocate(r)
                except Exception as exc:  # noqa: BLE001
                    rows.append(dict(n=n, dir=d, arm=s, error=str(exc).splitlines()[0][:200]))
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                if first is None:
                    first = got
                live.append((s, o, dict(rel_rms_vs_f64=float((got - ref).pow(2).mean().sqrt() / rn),
                                        max_abs_vs_first=float((got - first).abs().max()))))
                times[s] = []
                del got
            for _ in range(a.rounds):
                for s, o, _e in live:
                    for _ in range(a.reps):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        r = call(o)
                        ttnn.synchronize_device(dev)
                        times[s].append(time.perf_counter() - t0)
                        ttnn.deallocate(r)
            flop = 2.0 * C * n ** 3
            for s, o, e in live:
                ts = sorted(times[s][1:])
                ms = 1e3 * statistics.median(ts)
                rows.append(dict(n=n, dir=d, arm=s, ms=round(ms, 4), p10_ms=round(1e3 * ts[len(ts) // 10], 4),
                                 tflops=round(flop / ms / 1e9, 1), reps=len(ts), **e))
                print(json.dumps(rows[-1]), flush=True)
            del ref
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, aiclk=clk.summary(), program_config=str(pc),
                                           script="perf/rfd4/tri3/contract.py"), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
