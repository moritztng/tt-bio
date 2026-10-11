"""The real Transition module with and without transition_bw / transition_shard: bytes, error, time, size ladder.

    TT_VISIBLE_DEVICES=N PYTHONPATH=. python perf/spd_trans2/module_check.py --out OUT.json [--mode normal]
        [--sizes 256,512,736,1024,1536,2048] [--ref 736]

Arms, each under the mode's lever set:
  old       without transition_bw and transition_shard (staging19 normal)
  bw        transition_bw only (every block interleaved, K blocks fixed)
  new       transition_bw + transition_shard (the shipped path)
Pair Transition [1, S, S, 256] -> 1024 -> 256 and MSA Transition [1, 64, S, 128] -> 512 -> 128, random weights.
`digest` is of the output's bytes: run the script under TT_BIO_FORCE_GRID=8,8 too and compare. bw == new says the
shard is numerically inert on this grid. At --ref sizes the output is graded against float64 on the device's own
input and weight values. Times are warm, `reps` calls, one sync each, AICLK sampled during them.
"""
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()  # noqa: E702,I001

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", default="normal", choices=("normal", "fast"))
ap.add_argument("--sizes", default="256,512,736,1024,1536,2048")
ap.add_argument("--ref", default="736")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--msa-rows", type=int, default=64)
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = a.mode

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
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
MODE = T.parse_levers(a.mode)
ARMS = {"old": MODE - {"transition_bw", "transition_shard"}, "bw": MODE - {"transition_shard"}, "new": MODE}
T._LEVERS = MODE
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))


def module(C, HID, seed):
    g = torch.Generator().manual_seed(seed)
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm.bias": 0.1 * torch.randn(C, generator=g),
          "fc1.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
          "fc2.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
          "fc3.weight": torch.randn(C, HID, generator=g) / HID ** 0.5}
    with T.levers(MODE):
        return T.Transition(sd, CKC), g


def reference(tr, x, C):
    xd = ttnn.to_torch(x).double()
    w = {k: ttnn.to_torch(getattr(tr, k)).double() for k in ("fc1_weight", "fc2_weight", "fc3_weight",
                                                              "norm_weight", "norm_bias")}
    ln = torch.nn.functional.layer_norm(xd, (C,), w["norm_weight"].reshape(-1)[:C], w["norm_bias"].reshape(-1)[:C],
                                        1e-5)
    return (torch.nn.functional.silu(ln @ w["fc1_weight"]) * (ln @ w["fc2_weight"])) @ w["fc3_weight"]


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "mode": a.mode,
       "grid": list(T.COMPUTE_GRID_MAIN), "force_grid": os.environ.get("TT_BIO_FORCE_GRID"), "rows": {}}
refs = {int(s) for s in a.ref.split(",") if s}
for kind, C, HID in (("pair", 256, 1024), ("msa", 128, 512)):
    tr, g = module(C, HID, 0 if kind == "pair" else 1)
    for S in (int(s) for s in a.sizes.split(",")):
        H = S if kind == "pair" else a.msa_rows
        x = ttnn.from_torch(torch.randn(1, H, S, C, generator=g), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        ref = reference(tr, x, C) if S in refs else None
        outs = {}
        for arm, lv in ARMS.items():
            key = f"{kind}.{S}.{arm}"
            row = {}
            for k in ("transition_bw", "transition_shard"):
                T.LATCH_STATS[k].update(served=0, refused=0, declined=0)
            try:
                with T.levers(lv):
                    o = tr(x)
                    got = ttnn.to_torch(o).float()
                    ttnn.deallocate(o)
                    row.update(finite=bool(torch.isfinite(got).all()),
                               digest=hashlib.sha256(got.numpy().tobytes()).hexdigest()[:16],
                               latch={k: {s: T.LATCH_STATS[k].get(s, 0) for s in ("served", "declined", "refused")}
                                      for k in ("transition_bw", "transition_shard")})
                    if ref is not None:
                        e = got.double() - ref
                        row.update(rel_rms=float(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                                   max_err=float(e.abs().max()), n_over_0p25=int((e.abs() > 0.25).sum()))
                    outs[arm] = got
                    ts = []
                    t0 = time.monotonic()
                    for _ in range(a.reps):
                        ttnn.synchronize_device(dev)
                        t = time.perf_counter()
                        ttnn.deallocate(tr(x))
                        ttnn.synchronize_device(dev)
                        ts.append((time.perf_counter() - t) * 1e3)
                    v = sorted(c for tt_, c in clk if tt_ >= t0)
                    row.update(ms_min=round(min(ts), 3), ms_med=round(sorted(ts)[len(ts) // 2], 3),
                               aiclk=dict(median=v[len(v) // 2], min=v[0]) if v else None)
            except Exception as e:  # noqa: BLE001  a size that does not fit is a result
                row["err"] = str(e).splitlines()[0][:200]
            res["rows"][key] = row
            print(json.dumps({key: row}), flush=True)
        if "bw" in outs and "new" in outs:
            d = (outs["bw"] - outs["new"]).abs()
            res["rows"][f"{kind}.{S}.shard_inert"] = dict(n_diff=int((d > 0).sum()), max=float(d.max()))
            print(json.dumps({f"{kind}.{S}.shard_inert": res["rows"][f"{kind}.{S}.shard_inert"]}), flush=True)
        ttnn.deallocate(x)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
