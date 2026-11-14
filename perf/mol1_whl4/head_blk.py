"""Which block of Mol-1's confidence Pairformer loses the TT head's single track, against float64 (upstream modules).

    python -I head_blk.py CKPT DUMP_DIR
DUMP_DIR/headblk.pt is the hook's (WHL4_HEADBLK=1) per-block [s_in, z_in, s_out, z_out] of sample 0's head Pairformer,
DUMP_DIR/inputs.pt its inputs. Per block, on the TT block's OWN input: rel err of the TT output vs the float64 block
(local error), fp32 vs float64 (floor), and float64 on the bf16-rounded input (what storing the input in bf16 costs).
Then the chain: the TT output of each block vs the float64 stack run from the head's true input (accumulated error).
Run in the upstream venv, from outside the clone.
"""
import sys
from pathlib import Path

import torch
from mol.model.modules.pairformer import PairformerBlock

torch.set_num_threads(6)
ckpt, d = sys.argv[1], Path(sys.argv[2])
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
recs = torch.load(d / "headblk.pt", weights_only=False)
N = torch.load(d / "inputs.pt", weights_only=False)["s_trunk"].shape[0]


def block(i, dt):
    bp = f"{pre}pairformer_stack.blocks.{i}."
    w = {k[len(bp):]: v for k, v in sd.items() if k.startswith(bp)}
    b = PairformerBlock(n_heads=16, c_z=w["attention_pair_bias.layernorm_z.weight"].shape[0], c_s=384,
                        hidden_scale_up=True)
    b.load_state_dict(w, strict=True)
    return b.eval().to(dt)


def crop(s, z):
    s = s.reshape(-1, s.shape[-2], s.shape[-1])[:, :N]
    z = z.reshape(-1, z.shape[-3], z.shape[-2], z.shape[-1])[:, :N, :N]
    return s.double(), z.double()


def run(b, s, z, dt):
    with torch.no_grad():
        so, zo = b(s.to(dt), z.to(dt), None)
    return so.double(), zo.double()


def rel(x, y): return float((x - y).norm() / y.norm())
def bf(x): return x.to(torch.bfloat16).double()


print(f"N={N} blocks={len(recs)}  TT s_in dtype carried as fp32 dump")
ch = None
for i, (si, zi, so, zo) in enumerate(recs):
    si, zi = crop(si, zi); so, zo = crop(so, zo)
    b64, b32 = block(i, torch.float64), block(i, torch.float32)
    rs, rz = run(b64, si, zi, torch.float64)
    fs, fz = run(b32, si, zi, torch.float32)
    bs, bz = run(b64, bf(si), bf(zi), torch.float64)
    if ch is None:
        ch = (si, zi)
    ch = run(b64, *ch, torch.float64)
    ds, dz = rs - si, rz - zi  # the block's update, which is what it can get wrong
    print(f"blk {i}: s_in bf16-exact {bool((bf(si) == si).all())} z_in bf16-exact {bool((bf(zi) == zi).all())} | "
          f"local s {rel(so, rs):.4g} (upd {rel(so - si, ds):.4g}) z {rel(zo, rz):.4g} (upd {rel(zo - zi, dz):.4g}) | "
          f"fp32 s {rel(fs, rs):.2g} z {rel(fz, rz):.2g} | bf16-in s {rel(bs, rs):.4g} z {rel(bz, rz):.4g} | "
          f"chain s {rel(so, ch[0]):.4g} z {rel(zo, ch[1]):.4g}", flush=True)
torch.save({"N": N}, d / "head_blk.done")
