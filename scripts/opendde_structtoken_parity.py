# SPDX-License-Identifier: Apache-2.0
"""On-device PCC gate for the ttnn StructuralTokenExpander port vs the Phase-0
random-weight golden (scripts/opendde_structtoken_ref.py).

Apples-to-apples device-vs-reference check: same random weights, same synthetic
residue-trunk inputs. Gate: PCC > 0.98 on every output.

Run (qb2, worktree shadowing the editable shared checkout):
  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_structtoken_parity.py
"""
import os
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
import torch
import ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.opendde import StructuralTokenExpander

torch.set_grad_enabled(False)
GOLDEN = os.environ.get("OPENDDE_GOLDEN", "/tmp/opendde_structtoken_golden.pt")


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def main():
    d = torch.load(GOLDEN, map_location="cpu", weights_only=False)
    cfg, sd, inp, out = d["cfg"], d["state_dict"], d["inputs"], d["outputs"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    mod = StructuralTokenExpander(
        sd, ckc, c_s=cfg["c_s"], c_z=cfg["c_z"], c_s_inputs=cfg["c_s_inputs"],
        n_roles=cfg["n_roles"], pair_chunk_size=cfg["pair_chunk_size"])

    si, ss, zs, ab = mod(inp["ifd"], inp["s_inputs_res"], inp["s_res"], inp["z_res"])
    got = {
        "s_inputs_struct": torch.Tensor(ttnn.to_torch(si)).float().reshape(out["s_inputs_struct"].shape),
        "s_struct": torch.Tensor(ttnn.to_torch(ss)).float().reshape(out["s_struct"].shape),
        "z_struct": torch.Tensor(ttnn.to_torch(zs)).float().reshape(out["z_struct"].shape),
        "structural_pair_attn_bias": torch.Tensor(ttnn.to_torch(ab)).float().reshape(out["structural_pair_attn_bias"].shape),
    }
    print("StructuralTokenExpander opendde_v1 device-vs-reference PCC", flush=True)
    worst = 1.0
    for k in ["s_inputs_struct", "s_struct", "z_struct", "structural_pair_attn_bias"]:
        p = pcc(got[k], out[k].float())
        worst = min(worst, p)
        print("  %-28s PCC %.5f   %s" % (k, p, "PASS" if p > 0.98 else "FAIL"), flush=True)
    print("GATE %s (worst %.5f, threshold 0.98)" % ("PASS" if worst > 0.98 else "FAIL", worst), flush=True)

    # multi-chunk self-consistency: the golden used pair_chunk_size=128 (one chunk
    # for N_STRUCT=64). Re-run with a small chunk to force the row-block loop +
    # concat path; row blocks are independent so it must match the golden exactly.
    mod_mc = StructuralTokenExpander(
        sd, ckc, c_s=cfg["c_s"], c_z=cfg["c_z"], c_s_inputs=cfg["c_s_inputs"],
        n_roles=cfg["n_roles"], pair_chunk_size=16)
    _, _, zs_mc, ab_mc = mod_mc(inp["ifd"], inp["s_inputs_res"], inp["s_res"], inp["z_res"])
    z_mc = torch.Tensor(ttnn.to_torch(zs_mc)).float().reshape(out["z_struct"].shape)
    ab2 = torch.Tensor(ttnn.to_torch(ab_mc)).float().reshape(out["structural_pair_attn_bias"].shape)
    pz = pcc(z_mc, out["z_struct"].float()); pa = pcc(ab2, out["structural_pair_attn_bias"].float())
    print("  multi-chunk (chunk=16): z_struct PCC %.5f  attn_bias PCC %.5f  %s"
          % (pz, pa, "PASS" if min(pz, pa) > 0.98 else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
