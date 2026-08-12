#!/usr/bin/env python3
"""Op-level proof for the two levers, at BOTH 512 and 298 aa.

Both changes must be exact, not close: they are a re-association of constant weights and an
elementwise chain, so anything other than `torch.equal` is a bug, not a tolerance. 298 is here
because both preconditions are size-dependent -- E6 needs `mask is None`, which needs
L % PAD_MULTIPLE == 0, and the fc1 split changes the matmul's N-blocking. A 512-only check
would hide either.

Also records `n_pairs // group` for esmfold2's c_z=256 AND boltz2's c_z=128 at both sizes, which
is the shape gate now standing between E6 and every other model.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmc as EC
import tt_bio.reblock_permute as RP


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
         "pad_multiple": getattr(T, "PAD_MULTIPLE", None), "ffn": {}, "trimul": {}, "shape_gate": {}}
    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # ---- lever 2: the pair transition, real class, split vs production ----------------------
    from tt_bio.tenstorrent import WeightScope
    CZ, FF = 256, 1024
    torch.manual_seed(0)
    sd = WeightScope({"0.weight": torch.randn(CZ), "0.bias": torch.randn(CZ),
                      "1.weight": torch.randn(2 * FF, CZ) * 0.02,
                      "3.weight": torch.randn(CZ, FF) * 0.02})
    ffn = EC.SwiGLUFFN(sd, CK, fuse_swiglu=True)
    assert ffn.split_swiglu and not ffn.fuse_swiglu, (ffn.split_swiglu, ffn.fuse_swiglu)
    for L in (512, 298):
        z = f(torch.randn(1, L, L, CZ))
        EC.set_split_swiglu(False)
        ref = ttnn.to_torch(ffn(z))
        EC.set_split_swiglu(True)
        new = ttnn.to_torch(ffn(z))
        ttnn.deallocate(z)
        R["ffn"][str(L)] = {"torch_equal": bool(torch.equal(ref, new)),
                            "max_abs_diff": float((ref - new).abs().max()),
                            "shape": list(ref.shape)}
        print(f"  FFN L={L}: torch.equal={R['ffn'][str(L)]['torch_equal']}", flush=True)
        del ref, new

    # ---- lever 1: the trimul, real class, E6 off vs on, with served/declined --------------
    for cz, name in ((256, "esmfold2_cz256"), (128, "boltz2_cz128")):
        hid = cz
        tsd = WeightScope({
            "norm_in.weight": torch.randn(cz), "norm_in.bias": torch.randn(cz),
            "norm_out.weight": torch.randn(hid), "norm_out.bias": torch.randn(hid),
            "g_in.weight": torch.randn(2 * hid, cz) * 0.02,
            "p_in.weight": torch.randn(2 * hid, cz) * 0.02,
            "g_out.weight": torch.randn(cz, hid) * 0.02,
            "p_out.weight": torch.randn(cz, hid) * 0.02})
        tm = T.TriangleMultiplication(False, tsd, CK, gated_moves=True)
        for L in (512, 320, 298):
            x = f(torch.randn(1, L, L, cz))
            batch = 1
            chunk_size = T._trimul_chunk_size(L, tm._hidden, batch)
            n_pairs = tm._hidden // chunk_size
            large = T._triangle_mul_memory_config(L).buffer_type == ttnn.BufferType.DRAM
            group = T._trimul_inproj_group(L, chunk_size, batch, n_pairs) if large else 1
            key = f"{name}_L{L}"
            R["shape_gate"][key] = {"chunk_size": chunk_size, "n_pairs": n_pairs,
                                    "group": group, "groups": n_pairs // group,
                                    "one_group": n_pairs // group == 1, "dram": large}
            print(f"  gate {key}: n_pairs={n_pairs} group={group} "
                  f"groups={n_pairs//group} one_group={n_pairs//group==1}", flush=True)
            if cz == 256:
                RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
                prev = RP.set_enabled_gated(False)
                ref = ttnn.to_torch(tm(x))
                off = tuple(RP.STATS_GATED)
                RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
                RP.set_enabled_gated(True)
                new = ttnn.to_torch(tm(x))
                on = tuple(RP.STATS_GATED)
                RP.set_enabled_gated(prev)
                R["trimul"][f"L{L}"] = {
                    "torch_equal": bool(torch.equal(ref, new)),
                    "max_abs_diff": float((ref - new).abs().max()),
                    "served_declined_off": list(off), "served_declined_on": list(on)}
                print(f"  TRIMUL L={L}: torch.equal={R['trimul'][f'L{L}']['torch_equal']} "
                      f"served/declined off={off} on={on}", flush=True)
                del ref, new
            ttnn.deallocate(x)
        del tm
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1), flush=True)


if __name__ == "__main__":
    main()
