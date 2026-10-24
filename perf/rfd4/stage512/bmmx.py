"""stage512's cross-tree check of bmm_res: output sha256 at the trimul shapes the other models run (c 128 / 256, 256
to 1024 tokens, both transposes, the split at 1024), and at RFD4's 512 / 384 that `rot=True` is exactly the plain
output with channel c's tile pages rotated by c inside the channel.

usage: TT_VISIBLE_DEVICES=<chip> timeout 1800 python perf/rfd4/stage512/bmmx.py --out x.jsonl
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
ap = argparse.ArgumentParser()
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, bmm_res as B

dev = T.get_device()
up = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
sha = lambda t: hashlib.sha256(t.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]  # noqa: E731


def emit(rec):
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def run(C, n, ta, tb, dt, rot=False):
    g = torch.Generator().manual_seed(C * 1000 + n + 2 * ta + tb)
    a, b = torch.randn(1, C, n, n, generator=g), torch.randn(1, C, n, n, generator=g)
    y = B.matmul(up(a, dt), up(b, dt), transpose_a=ta, transpose_b=tb, rot=rot)
    return None if y is None else ttnn.to_torch(y).reshape(1, C, n, n)


for C, n, ta, tb, dt in [(128, 256, 0, 0, ttnn.bfloat16), (128, 256, 1, 1, ttnn.bfloat16),
                         (256, 256, 0, 1, ttnn.bfloat8_b), (128, 512, 1, 0, ttnn.bfloat16),
                         (256, 512, 0, 0, ttnn.bfloat8_b), (256, 384, 1, 0, ttnn.bfloat8_b),
                         (128, 1024, 0, 0, ttnn.bfloat16), (256, 1024, 1, 1, ttnn.bfloat8_b)]:
    y = run(C, n, ta, tb, dt)
    rec = {"C": C, "n": n, "ta": ta, "tb": tb, "dtype": str(dt), "sha": None if y is None else sha(y)}
    if y is not None and n in (384, 512):
        r = run(C, n, ta, tb, dt, rot=True)
        St = n // 32
        tiles = y.reshape(C, St, 32, St, 32).permute(0, 1, 3, 2, 4).reshape(C, St * St, 32, 32)
        want = torch.stack([torch.roll(tiles[c], c, 0) for c in range(C)])     # tile q of c lands at q + c
        got = r.reshape(C, St, 32, St, 32).permute(0, 1, 3, 2, 4).reshape(C, St * St, 32, 32)
        rec["rot_equal"] = bool(torch.equal(want, got))
        rec["rot_pays"] = B.rot_pays(dev, n)
    emit(rec)
