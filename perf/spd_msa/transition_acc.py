"""spd-msa: the MSA transition (LN -> fc1 silu, fc2 -> multiply -> fc3) under fidelity / fp32-dest-acc configs.

One MSA row chunk at the campaign cell, x [ROWS, 736, 128] (c_m 128, hidden 512, as the census shows), issued exactly as
`Transition._transition`'s 3-D path issues it (L1 intermediates, CORE_GRID_MAIN). Per config: median ms of REPS
synced calls and the error of the output against a float64 evaluation of the same bf16 inputs and weights.
lpx-matmul put the transition's cost in fp32 partials (half-size dest, fp32 L1), not math or DRAM; the question
here is what turning fp32 dest accumulation off costs in accuracy at K = 64 and K = 256, i.e. whether it can be a
normal-mode lever for the MSA track alone.

usage: TT_VISIBLE_DEVICES=<chip> python transition_acc.py OUT [ROWS=16] [REPS=10]
The `_unfused` configs issue silu as its own op on fc1's output (TT_BIO_UNFUSED_SILU's form).
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
LOG = open(OUT / "transition.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
F = ttnn.MathFidelity
CONFIGS = [("hifi4_acc", F.HiFi4, True), ("hifi4_noacc", F.HiFi4, False), ("hifi2_acc", F.HiFi2, True),
           ("hifi2_noacc", F.HiFi2, False), ("lofi_noacc", F.LoFi, False), ("hifi4_acc_unfused", F.HiFi4, True),
           ("hifi4_noacc_unfused", F.HiFi4, False)]
T_, C, H = 736, 128, 512

torch.manual_seed(0)
x_h = torch.randn(1, ROWS, T_, C).bfloat16()
lw_h, lb_h = (1 + 0.1 * torch.randn(C)).bfloat16(), (0.1 * torch.randn(C)).bfloat16()
w1_h, w2_h = (torch.randn(C, H) / C ** 0.5).bfloat16(), (torch.randn(C, H) / C ** 0.5).bfloat16()
w3_h = (torch.randn(H, C) / H ** 0.5).bfloat16()


def ref64():
    x = x_h.double()
    xn = torch.nn.functional.layer_norm(x, (C,), lw_h.double(), lb_h.double(), 1e-5)
    return (torch.nn.functional.silu(xn @ w1_h.double()) * (xn @ w2_h.double())) @ w3_h.double()


ref = ref64()
tt = lambda t, **k: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, **k)
x = tt(x_h); lw = tt(lw_h.reshape(1, C)); lb = tt(lb_h.reshape(1, C))
w1, w2, w3 = tt(w1_h), tt(w2_h), tt(w3_h)
L1 = ttnn.L1_MEMORY_CONFIG


def swiglu(k, unfused=False):
    xn = ttnn.layer_norm(x, weight=lw, bias=lb, epsilon=1e-5, compute_kernel_config=k, memory_config=L1)
    x1 = ttnn.linear(xn, w1, activation=None if unfused else "silu", compute_kernel_config=k, memory_config=L1,
                     dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
    if unfused:
        x1 = ttnn.silu(x1, memory_config=L1, output_tensor=x1)
    x2 = ttnn.linear(xn, w2, compute_kernel_config=k, memory_config=L1, dtype=ttnn.bfloat16,
                     core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(xn)
    h = ttnn.multiply_(x1, x2); ttnn.deallocate(x2)
    o = ttnn.linear(h, w3, compute_kernel_config=k, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h)
    return o


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


log(ev="start", rows=ROWS, tokens=T_, c=C, hidden=H, arch=str(dev.arch()), aiclk=aiclk())
for name, fid, acc in CONFIGS:
    uf = name.endswith("_unfused")
    k = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=acc,
                                         packer_l1_acc=True)
    try:
        out = swiglu(k, uf); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(REPS):
            ttnn.deallocate(out)
            t0 = time.perf_counter(); out = swiglu(k, uf); ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        o = ttnn.to_torch(out).double(); ttnn.deallocate(out)
        d = (o - ref).abs()
        log(ev="cfg", cfg=name, ms=ts, ms_med=statistics.median(ts),
            rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.max()),
            finite=bool(torch.isfinite(o).all()), aiclk=aiclk())
    except Exception as e:
        log(ev="cfg_fail", cfg=name, err=str(e)[:300])
log(ev="end")
