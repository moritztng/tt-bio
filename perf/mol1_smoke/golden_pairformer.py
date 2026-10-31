"""The shipped Mol-1 trunk Pairformer (Protenix.load_from_checkpoint) vs mol1-ref's upstream goldens on real
activations: the full 48-block stack with both adapters, and each captured block on its own.
Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/golden_pairformer.py CKPT GOLDEN_SEED_DIR OUT_JSON"""
import copy
import json
import os
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio.protenix import Protenix, bucketed_pairformer  # noqa: E402


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


ckpt, gdir, out = sys.argv[1:4]
trunk = Protenix.load_from_checkpoint(ckpt).trunk
dev = trunk.dev


def run(pf, s, z):
    N = s.shape[0]
    ts = ttnn.from_torch(s.float().reshape(1, N, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tz = ttnn.from_torch(z.float().reshape(1, N, N, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    so, zo = bucketed_pairformer(pf, ts, tz, dev)
    return ttnn.to_torch(so).float().reshape(N, -1), ttnn.to_torch(zo).float().reshape(N, N, -1)


def only(i):
    """trunk.PF reduced to block i, no adapters: same objects, same weights, same flags."""
    pf = copy.copy(trunk.PF)
    pf.blocks, pf.adapters = [trunk.PF.blocks[i]], {}
    return pf


cases = [("stack48_with_adapters", trunk.PF, "pairformer_stack.call0.pt")]
cases += [(f"block{i}", only(i), f"pairformer_stack.blocks.{i}.call0.pt")
          for i in (0, 1, 8, 23, 47) if os.path.exists(os.path.join(gdir, f"pairformer_stack.blocks.{i}.call0.pt"))]
res = {}
for name, pf, f in cases:
    g = torch.load(os.path.join(gdir, f), weights_only=False)
    (s, z), (ws, wz) = g["args"][:2], g["out"]
    so, zo = run(pf, s, z)
    res[name] = {"s_pcc": round(pcc(so, ws), 6), "z_pcc": round(pcc(zo, wz), 6),
                 "s_rel": round(rel(so, ws), 5), "z_rel": round(rel(zo, wz), 5)}
    print(name, res[name], flush=True)
res["N"] = int(s.shape[0])
res["reference"] = f"mol1-ref golden {gdir} (upstream e6e64866, fp32 CPU, first recycle)"
json.dump(res, open(out, "w"), indent=1)
