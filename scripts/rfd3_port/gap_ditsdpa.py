#!/usr/bin/env python3
"""gap_ditsdpa: the DiT token SDPA at the page fixture's shape, swept over chunk configs and padded lengths.

q/k/v [1, 16, 685, 64] bf16 with a [1, 16, n, n] bf16 additive bias (sequence padding masked with -1e4), as
`_dense_sdpa_config` serves it at n = 704 (q 64, k 352: 176 work units on a 72-core Wormhole, three passes).
Each arm pads the key axis to n (704 or 768) and picks (q_chunk, k_chunk) dividing it; ms/call is the median of
N batches of 10 calls between syncs; error is against float64 on the 685 real rows, and `vs_ship_max_abs` against the shipped arm.
    python scripts/rfd3_port/gap_ditsdpa.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio.tenstorrent import get_device, fused_sdpa, _sdpa_program_config  # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
H, L, D = 16, 685, 64
ARMS = [(704, 64, 352), (704, 64, 704), (704, 32, 352), (704, 352, 352), (704, 64, 64),
        (768, 64, 384), (768, 96, 384), (768, 128, 384), (768, 192, 384), (768, 128, 768), (768, 128, 256),
        (768, 96, 256), (768, 64, 768)]
g = torch.Generator().manual_seed(0)


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    q, k, v = (torch.randn(1, H, L, D, generator=g).bfloat16() for _ in range(3))
    bias = (torch.randn(1, H, L, L, generator=g) * 2).bfloat16()
    ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) * D ** -0.5 + bias.double(), -1) @ v.double()
    rec = {"reps": N_REP, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "grid": list(M.COMPUTE_GRID_MAIN) if hasattr(M, "COMPUTE_GRID_MAIN") else None}
    ship = None
    for n, qc, kc in ARMS:
        pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, n - L))        # noqa: E731
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
        m = torch.full((1, H, n, n), -1e4).bfloat16()
        m[:, :, :L, :L] = bias
        m[:, :, L:, :] = 0
        qd, kd, vd, md = up(pad(q)), up(pad(k)), up(pad(v)), up(m)
        pc = _sdpa_program_config(qc, kc)
        run = lambda: fused_sdpa(qd, kd, vd, attn_mask=md, scale=D ** -0.5, program_config=pc,  # noqa: E731
                                 compute_kernel_config=ckc)
        key = f"n{n}_q{qc}_k{kc}"
        try:
            o = run()
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(N_REP):
                t0 = time.perf_counter()
                os_ = [run() for _ in range(10)]
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e2)
                for x in os_:
                    ttnn.deallocate(x)
            out = ttnn.to_torch(o)[:, :, :L].double()
            ttnn.deallocate(o)
            e = (out - ref).abs()
            r = {"ms_median": round(statistics.median(ts), 4), "ms_min": round(min(ts), 4),
                 "max_abs_f64": e.max().item(), "mean_abs_f64": e.mean().item()}
            if ship is None:
                ship = out
            r["vs_ship_max_abs"] = (out - ship).abs().max().item()
        except Exception as ex:                                              # noqa: BLE001
            r = {"error": repr(ex)[:400]}
        for t in (qd, kd, vd, md):
            ttnn.deallocate(t)
        rec[key] = r
        print(key, r, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
