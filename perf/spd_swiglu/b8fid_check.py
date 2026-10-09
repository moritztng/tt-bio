"""Transition fc3 at HiFi2 on bfp8 operands (`bfp8_fidelity`): same bytes as the trunk fidelity, and how much faster.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/b8fid_check.py --out OUT.json [--S 736] [--msa 16]

Fast mode's real Transition module (bfp8 weights and hidden), pair (256 -> 1024) on [1, S, S, 256] and MSA
(128 -> 512) on [1, msa, S, 128], each through the sharded and the interleaved path where the grid has both.
Every cell runs twice in one process, with `bfp8_fidelity` replaced by the identity (before) and as shipped
(after): the output digests must match, the module times are the saving. AICLK sampled during each timing.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--S", type=int, default=736)
ap.add_argument("--msa", type=int, default=16)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--repeat", type=int, default=1, help="outputs per arm: >1 tells a fidelity change from a racy kernel")
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = "fast"

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers("fast")
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
            except Exception:
                pass
        time.sleep(0.2)


threading.Thread(target=_sampler, daemon=True).start()
REAL = T.bfp8_fidelity
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": NODES,
       "fidelity": str(CKC.math_fidelity), "cells": []}
g = torch.Generator().manual_seed(0)
for name, C, HID, rows in (("pair", 256, 1024, a.S), ("msa", 128, 512, a.msa)):
    sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm.bias": 0.1 * torch.randn(C, generator=g),
          "fc1.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
          "fc2.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
          "fc3.weight": torch.randn(C, HID, generator=g) / HID ** 0.5}
    tr = T.Transition(sd, CKC)
    x = ttnn.from_torch(torch.randn(1, rows, a.S, C, generator=g), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    for arm, lv in (("default", "fast"), ("interleaved", "fast-transition_shard")):
        row = dict(shape=name, arm=arm)
        for when, fn in (("before", lambda ckc, *_: ckc), ("after", REAL)):
            T.bfp8_fidelity = fn
            with T.levers(lv):
                outs = []
                for _ in range(a.repeat):
                    o = tr(x)
                    outs.append(ttnn.to_torch(o).float())
                    ttnn.deallocate(o)
                digests = [hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16] for t in outs]
                row[f"digest_{when}"] = digests[0]
                if a.repeat > 1:
                    row[f"digests_{when}"] = digests
                    row[f"ndiff_{when}"] = [int((t != outs[0]).sum()) for t in outs[1:]]
                if when == "before":
                    first = outs[0]
                else:
                    d = (outs[0] - first).abs()
                    row["ndiff_after_vs_before"] = int((d > 0).sum())
                    row["maxdiff_after_vs_before"] = float(d.max())
                del outs
                ts = []
                t0 = time.monotonic()
                for _ in range(a.reps):
                    ttnn.synchronize_device(dev)
                    t = time.perf_counter()
                    ttnn.deallocate(tr(x))
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t) * 1e3)
                v = sorted(c for ts_, c in clk if ts_ >= t0)
            row[f"ms_{when}"] = dict(min=round(min(ts), 3), med=round(st.median(ts), 3),
                                     aiclk=dict(median=v[len(v) // 2], min=v[0]) if v else None)
        T.bfp8_fidelity = REAL
        row["identical"] = row["digest_before"] == row["digest_after"]
        res["cells"].append(row)
        print(json.dumps(row), flush=True)
    ttnn.deallocate(x)
    del tr
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
