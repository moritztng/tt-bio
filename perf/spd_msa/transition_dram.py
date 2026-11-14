"""spd-msa: the MSA transition over one 512-row MSA chunk, L1 intermediates at 16 rows vs DRAM intermediates at taller rows.

transition_acc.py put the 16-row chunk at 0.66-0.88 ms for ~4.6 GFLOP, about 10x off the HiFi4 roof, and 32 rows do
not fit L1. The question: does the per-op cost fall when the intermediates go to DRAM and each op sees 4-32x the rows?
Per arm: median ms for the whole SLAB (512 rows by default, the MSA update chunk), and the error against float64.

usage: TT_VISIBLE_DEVICES=<chip> python transition_dram.py OUT [SLAB=512] [REPS=5]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SLAB = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
LOG = open(OUT / "transition_dram.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
T_, C, H = 736, 128, 512
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
# (name, rows per block, intermediate memory, silu fused into fc1)
ARMS = [("l1_16_fused", 16, L1, True), ("l1_16_unfused", 16, L1, False)]
ARMS += [(f"dram_{h}_{'fused' if f else 'unfused'}", h, DRAM, f) for h in (16, 64, 128, 256, 512) for f in (True, False)]

torch.manual_seed(0)
x_h = torch.randn(1, SLAB, T_, C).bfloat16()
lw_h, lb_h = (1 + 0.1 * torch.randn(C)).bfloat16(), (0.1 * torch.randn(C)).bfloat16()
w1_h, w2_h = (torch.randn(C, H) / C ** 0.5).bfloat16(), (torch.randn(C, H) / C ** 0.5).bfloat16()
w3_h = (torch.randn(H, C) / H ** 0.5).bfloat16()
xd = x_h.double()
xn_r = torch.nn.functional.layer_norm(xd, (C,), lw_h.double(), lb_h.double(), 1e-5)
ref = (torch.nn.functional.silu(xn_r @ w1_h.double()) * (xn_r @ w2_h.double())) @ w3_h.double()
del xd, xn_r

tt = lambda t, **k: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, **k)
x = tt(x_h); lw = tt(lw_h.reshape(1, C)); lb = tt(lb_h.reshape(1, C))
w1, w2, w3 = tt(w1_h), tt(w2_h), tt(w3_h)
K = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                     fp32_dest_acc_en=True, packer_l1_acc=True)


def swiglu(xb, mc, fused):
    xn = ttnn.layer_norm(xb, weight=lw, bias=lb, epsilon=1e-5, compute_kernel_config=K, memory_config=mc)
    x1 = ttnn.linear(xn, w1, activation="silu" if fused else None, compute_kernel_config=K, memory_config=mc,
                     dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
    if not fused:
        x1 = ttnn.silu(x1, memory_config=mc, output_tensor=x1)
    x2 = ttnn.linear(xn, w2, compute_kernel_config=K, memory_config=mc, dtype=ttnn.bfloat16,
                     core_grid=T.CORE_GRID_MAIN)
    ttnn.deallocate(xn)
    h = ttnn.multiply_(x1, x2); ttnn.deallocate(x2)
    o = ttnn.linear(h, w3, compute_kernel_config=K, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN,
                    memory_config=DRAM)
    ttnn.deallocate(h)
    return o


def slab(rows, mc, fused):
    if rows >= SLAB:
        return swiglu(x, mc, fused)
    parts = []
    for r in range(0, SLAB, rows):
        xb = ttnn.slice(x, (0, r, 0, 0), (1, min(r + rows, SLAB), T_, C))
        parts.append(swiglu(xb, mc, fused)); ttnn.deallocate(xb)
    o = ttnn.concat(parts, dim=1)
    for p in parts:
        ttnn.deallocate(p)
    return o


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


log(ev="start", slab=SLAB, tokens=T_, c=C, hidden=H, arch=str(dev.arch()), aiclk=aiclk())
for name, rows, mc, fused in ARMS:
    try:
        out = slab(rows, mc, fused); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(REPS):
            ttnn.deallocate(out)
            t0 = time.perf_counter(); out = slab(rows, mc, fused); ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        clk = aiclk()
        o = ttnn.to_torch(out).double(); ttnn.deallocate(out)
        d = (o - ref).abs()
        log(ev="arm", arm=name, rows=rows, ms=ts, ms_med=statistics.median(ts),
            rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.max()),
            finite=bool(torch.isfinite(o).all()), aiclk=clk)
    except Exception as e:
        log(ev="arm_fail", arm=name, err=str(e)[:300])
log(ev="end")
