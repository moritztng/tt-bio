"""tt-bio's Pairformer on Mol-1 weights vs the float64 upstream capture (ref_pairformer.py).
Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/tt_pairformer.py CKPT REF_PT OUT_JSON"""
import json
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

from tt_bio import protenix_weights as PW  # noqa: E402
from tt_bio.tenstorrent import Pairformer, get_device  # noqa: E402


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


ckpt, ref_pt, out = sys.argv[1:4]
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
ref = torch.load(ref_pt, weights_only=False)
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)


def stack(blocks, with_adapters):
    comb, heads = {}, []
    for j, i in enumerate(blocks):
        blk = {k[len(f"pairformer_stack.blocks.{i}."):]: v for k, v in sd.items()
               if k.startswith(f"pairformer_stack.blocks.{i}.")}
        comb.update({f"layers.{j}.{k}": v for k, v in PW.remap_pairformer_block(blk).items()})
        ta = blk.get("tri_att_start.linear.weight")
        heads.append(0 if ta is None else int(ta.shape[0]))
    if with_adapters:
        comb.update(PW.remap_pair_adapters(sd, "pairformer_stack.", len(blocks)))
    return Pairformer(len(blocks), 32, heads, 24, 16, True, comb, ckc)


def run(pf, s, z):
    N = s.shape[0]
    ts = ttnn.from_torch(s.float().reshape(1, N, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tz = ttnn.from_torch(z.float().reshape(1, N, N, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    so, zo = pf(ts, tz)
    return ttnn.to_torch(so).float().reshape(N, -1), ttnn.to_torch(zo).float().reshape(N, N, -1)


res = {}
for name, blocks, adapt, inp, want in (
        ("block0_pairmixer_c448", [0], False, (ref["s"], ref["z"]), ref["b0"]),
        ("block8_pairformer_c320", [8], False, ref["b8_in"], ref["b8"]),
        ("stack48_with_adapters", list(range(48)), True, (ref["s"], ref["z"]), ref["out"])):
    so, zo = run(stack(blocks, adapt), *inp)
    res[name] = {"s_pcc": round(pcc(so, want[0]), 6), "z_pcc": round(pcc(zo, want[1]), 6),
                 "s_rel": round(rel(so, want[0]), 5), "z_rel": round(rel(zo, want[1]), 5)}
    print(name, res[name], flush=True)
res["N"] = int(ref["s"].shape[0])
res["reference"] = "upstream mol.model.modules.pairformer.PairformerStack, float64 CPU, e6e64866"
json.dump(res, open(out, "w"), indent=1)
