"""Stage split of gated_out_proj at Protenix-v2's shape: one process per TT_BIO_TRIATT_TAIL_ABLATE arm.

Times the three forms the Pairformer calls (resid: the starting node's in-place sum; trans: the ending node's
sum into z^T; plain: the update alone) with a bf16 and a bfp8 gate, AICLK sampled during each form. Under an
ablation the values are wrong on purpose and only the time is read.

    TT_VISIBLE_DEVICES=<chip> [TT_BIO_TRIATT_TAIL_ABLATE=READ,WRITE,SIG,MM] python gop_abl.py OUT.jsonl [--n 512] [--cz 256]
"""
import argparse, json, os, statistics, threading, time

os.environ.setdefault("TT_BIO_TRIATT_TAIL", "1")
os.environ.setdefault("TT_BIO_TRIATT_TAIL_T", "1")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--node", default=None, help="/dev/tenstorrent node of the chip, for the AICLK")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=16)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import triatt_qkv as TQ  # noqa: E402

dev = T.get_device()
node = A.node or os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
clk, stop = [], threading.Event()


def _sample():
    while not stop.wait(0.25):
        try:
            clk.append((time.monotonic(), int(open(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read().split()[0])))
        except Exception:
            pass


threading.Thread(target=_sample, daemon=True).start()
N, C, H = A.n, A.cz, 8
torch.manual_seed(0)
put = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)  # noqa: E731
od = put((torch.randn(N, H, N, 32) * 0.5).bfloat16())
wd = put((torch.randn(H * 32, C) / (H * 32) ** 0.5).bfloat16())
g_h = torch.randn(N, H, N, 32).bfloat16()
zd = put(torch.zeros(1, N, N, C).bfloat16())
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def plain(g):
    ttnn.deallocate(TQ.gated_out_proj(od, g, wd, ckc))


forms = {"resid": lambda g: TQ.gated_out_proj(od, g, wd, ckc, resid=zd),
         "trans": lambda g: TQ.gated_out_proj(od, g, wd, ckc, resid_t=zd), "plain": plain}
with open(A.out, "a") as f:
    for gname, gdt in (("g8", ttnn.bfloat8_b), ("g16", ttnn.bfloat16)):
        gd = put(g_h, gdt)
        with T.levers(["triatt_ln_b8g"] if gdt == ttnn.bfloat8_b else []):
            for name, fn in forms.items():
                ms, served0, m0 = [], TQ.GOP_STATS[0], None
                for r in range(A.reps + 1):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(A.calls):
                        fn(gd)
                    ttnn.synchronize_device(dev)
                    if r:   # rep 0 compiles and warms
                        ms.append(1e3 * (time.perf_counter() - t0) / A.calls)
                    else:
                        m0 = time.monotonic()
                c = [v for t, v in clk if t >= m0 and v > 0]
                row = {"abl": ",".join(TQ.TAIL_ABLATE) or "base", "form": name, "g": gname, "n": N, "cz": C,
                       "ms_med": round(statistics.median(ms), 4), "ms_min": round(min(ms), 4), "ms_max": round(max(ms), 4),
                       "served": TQ.GOP_STATS[0] - served0, "calls": A.calls * (A.reps + 1),
                       "aiclk": [min(c), statistics.median(c), max(c)] if c else None,
                       "load": open("/proc/loadavg").read().split()[0], "arch": str(dev.arch())}
                f.write(json.dumps(row) + "\n"); f.flush(); print(json.dumps(row), flush=True)
        ttnn.deallocate(gd)
stop.set()
