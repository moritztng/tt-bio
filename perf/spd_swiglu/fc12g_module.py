"""`swiglu_fc12g` on the real Transition module: error against a float64 Transition, and module time, lever off/on.

    TT_VISIBLE_DEVICES=N PYTHONPATH=. python perf/spd_swiglu/fc12g_module.py --out OUT.json --mode fast|normal

Pair Transition [1, S, S, 256] -> 1024 -> 256 through the sharded path (transition_shard). On Blackhole the
shard path only runs with --bh-shard (fast mode), which adds a sharded arm without fc12g. The float64 reference
uses the device's own weight values (read back), so a difference is arithmetic. Times are pipelined: reps calls,
one sync. AICLK sampled during the timing.
"""
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()  # noqa: E702,I001

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", default="fast", choices=("normal", "fast"))
ap.add_argument("--S", type=int, default=736)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--bh-shard", action="store_true",
                help="Blackhole: arms base, shard (_TRANSITION_SHARD_BH_B8) and shard+fc12g")
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = a.mode

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers(a.mode)
NODES = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.exists(f"/proc/self/fd/{fd}") and
                os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
clk = []


def _sampler():
    while True:
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    clk.append((time.monotonic(), v))
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.2)


threading.Thread(target=_sampler, daemon=True).start()
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
C, HID = 256, 1024
g = torch.Generator().manual_seed(0)
sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm.bias": 0.1 * torch.randn(C, generator=g),
      "fc1.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
      "fc2.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
      "fc3.weight": torch.randn(C, HID, generator=g) / HID ** 0.5}
tr = T.Transition(sd, CKC)
x = ttnn.from_torch(torch.randn(1, a.S, a.S, C, generator=g), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

xd = ttnn.to_torch(x).double()
w = {k: ttnn.to_torch(getattr(tr, k)).double() for k in ("fc1_weight", "fc2_weight", "fc3_weight",
                                                          "norm_weight", "norm_bias")}
ln = torch.nn.functional.layer_norm(xd, (C,), w["norm_weight"].reshape(-1)[:C], w["norm_bias"].reshape(-1)[:C], 1e-5)
ref = (torch.nn.functional.silu(ln @ w["fc1_weight"]) * (ln @ w["fc2_weight"])) @ w["fc3_weight"]
del ln

res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "mode": a.mode,
       "S": a.S, "arms": {}}
outs = {}
ARMS = [("base", a.mode, False), ("fc12g", f"{a.mode}+swiglu_fc12g", a.bh_shard)]
if a.bh_shard:
    ARMS.insert(1, ("shard", a.mode, True))
for arm, lv, bh in ARMS:
    T._TRANSITION_SHARD_BH_B8 = bh
    row = res["arms"][arm] = {}
    T.LATCH_STATS["transition_shard"].update(served=0, refused=0)
    with T.levers(lv):
        o = tr(x)
        got = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        e = got - ref
        row.update(rel_rms=float(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_err=float(e.abs().max()),
                   n_over_0p1=int((e.abs() > 0.1).sum()), served=T.LATCH_STATS["transition_shard"]["served"],
                   refused=T.LATCH_STATS["transition_shard"]["refused"], w12=tr._fc12_weight() is not None)
        outs[arm] = got
        best = []
        for _ in range(a.reps):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            t0 = time.monotonic()
            ttnn.deallocate(tr(x))
            ttnn.synchronize_device(dev)
            best.append((time.perf_counter() - t) * 1e3)
        v = sorted(c for ts, c in clk if ts >= t0)
        row.update(ms_min=round(min(best), 3), ms_med=round(sorted(best)[len(best) // 2], 3),
                   aiclk=dict(median=v[len(v) // 2], min=v[0]) if v else None)
    print(json.dumps({arm: row}), flush=True)
d = (outs["fc12g"] - outs["base"]).abs()
res["fc12g_vs_base"] = dict(max=float(d.max()), n_diff=int((d > 0).sum()))
if a.bh_shard:
    d = (outs["shard"] - outs["base"]).abs()
    res["shard_vs_base"] = dict(max=float(d.max()), n_diff=int((d > 0).sum()))
print(json.dumps(res["fc12g_vs_base"]), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
