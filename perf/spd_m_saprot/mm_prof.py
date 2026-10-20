"""Where a SaProt projection's device time goes, per RISC: the model's call and the shipped pick for each projection at
the page shape (7 x 544 rows), plus a K ladder of fc1's call at fixed M, N, under the device profiler.

fc1 runs at ~half fc2's rate for the same FLOPs on both arches (WH --fast 1.24 vs 0.68 ms, BH 0.56 vs 0.29), and no
program knob moved it (lin_sweep ls1/ls33/ls38/ls45). This reads, per program, the kernel duration of each RISC and the
compute cluster's stall accumulators (TT_METAL_PROFILER_SUM: CB WAIT FRONT on the unpacker, CB RESERVE BACK on the
packer), each divided by the program's own core count, so the per-output-tile cost lands on a thread.

Needs the Tracy tt-metal build and TT_METAL_DEVICE_PROFILER=1 (perf/spd_m_saprot/prof.sh sets the environment).

    python perf/spd_m_saprot/mm_prof.py --model saprot-650m [--fast] --out prof.jsonl
"""
import argparse
import json
import math

import torch
import ttnn

MODELS = {"saprot-35m": (480, 1920), "saprot-650m": (1280, 5120), "saprot-1.3b": (1280, 5120)}
KEYS = {"k": "DEVICE KERNEL DURATION [ns]", "br": "DEVICE BRISC KERNEL DURATION [ns]",
        "nc": "DEVICE NCRISC KERNEL DURATION [ns]", "t0": "DEVICE TRISC0 KERNEL DURATION [ns]",
        "t1": "DEVICE TRISC1 KERNEL DURATION [ns]", "t2": "DEVICE TRISC2 KERNEL DURATION [ns]"}
SUMS = {"wait": "DEVICE COMPUTE CB WAIT FRONT [ns]", "resv": "DEVICE COMPUTE CB RESERVE BACK [ns]"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m", choices=sorted(MODELS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--shapes", default="qkv,out,fc1,fc2")
    ap.add_argument("--kladder", default="320,640,1280,2560", help="fc1's call at these K (N, M fixed); empty = none")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, ops
    from tt_bio import saprot as S
    T.set_fast_mode(a.fast)
    dev = T.get_device()
    cfg, _ = S.CONFIGS[a.model]
    ck = S.Saprot(**cfg).compute_kernel_config
    d, ff = MODELS[a.model]
    dh = d if (d // cfg["n_heads"]) % 64 == 0 else cfg["n_heads"] * 32
    shapes = {"qkv": (d, 3 * dh, None), "out": (dh, d, None), "fc1": (d, ff, "gelu"), "fc2": (ff, d, None),
              "fc1n": (d, ff, None)}  # fc1 without its fused GELU: the activation's device cost
    for k in filter(None, a.kladder.split(",")):
        shapes[f"fc1k{k}"] = (int(k), ff, "gelu")
    wdt = ttnn.bfloat8_b if a.fast else ttnn.bfloat16
    out = open(a.out, "a")
    gen = torch.Generator().manual_seed(0)
    B, L = 7, a.rows // 7

    def flush():
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        return [p for progs in ttnn.get_latest_programs_perf_data().values() for p in progs]

    flush()
    for name in a.shapes.split(",") + [s for s in shapes if s.startswith("fc1k")]:
        K, N, act = shapes[name]
        x = torch.randn(B, L, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        b = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        bd = ttnn.from_torch(b.reshape(1, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        kw = {"activation": act} if act else {}
        lin = S._SaprotLin.__new__(S._SaprotLin)
        lin.compute_kernel_config, lin.norm_weight = ck, torch.empty(d)
        mck = lin._ck(wd)
        arms = {"model": lambda: ops.linear(xd, wd, bias=bd, compute_kernel_config=mck, dtype=ttnn.bfloat16,
                                            core_grid=T.CORE_GRID_MAIN, **kw)}
        hifi2 = mck.math_fidelity == ttnn.MathFidelity.HiFi2
        if not name.startswith("fc1k") and S._saprot_prog(xd, wd, act, hifi2) is not None:
            arms["pick"] = lambda: lin._lin(xd, wd, bias=bd, **kw)
        for arm, fn in arms.items():
            try:
                for _ in range(2):
                    ttnn.deallocate(fn())
                flush()
                for _ in range(a.calls):
                    ttnn.deallocate(fn())
                progs = flush()
            except Exception as e:  # noqa: BLE001 -- a refused program is a data point
                rec = dict(model=a.model, fast=a.fast, shape=name, arm=arm, error=str(e).splitlines()[0][:200])
                print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
                continue
            for p in progs:
                r = p.program_analyses_results
                cc = max(1, int(p.core_count))
                rec = dict(model=a.model, fast=a.fast, shape=name, arm=arm, M=a.rows, K=K, N=N, cores=cc,
                           tiles_out=(a.rows // 32) * (N // 32), kt=K // 32,
                           **{s: round(r[k].duration / 1e3, 2) for s, k in KEYS.items() if k in r},
                           **{s: round(r[k].duration / cc / 1e3, 2) for s, k in SUMS.items() if k in r})
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n"); out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
