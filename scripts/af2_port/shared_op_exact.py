"""Prove the AF2 bias plumbing left the shared ops textually where they were.

Builds Boltz-2's `TriangleMultiplication`, `TriangleAttention` and `OuterProductMean` from the
shipping checkpoint -- none of which has a bias in scope -- and dumps their outputs under three
path configurations. Run once per tree; `torch.equal` between the two dumps is the acceptance.

The configurations exist to reach every branch the bias plumbing touched, at a size a card can
actually hold: the whole-tensor tail, the row-blocked tail (`SEQ_LEN_MORE_CHUNKING` lowered) and
the row-blocked input norm (`TRIMUL_IN_NORM_ROWBLOCK_BYTES` lowered, which is `_in_proj_rows`).
Both trees get the identical patch, so the comparison is one path pre against post.

    python3 scripts/af2_port/shared_op_exact.py --tree <tree> --out /tmp/<label>.pt
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.tree)
    import ttnn

    from tt_bio import tenstorrent as T

    ckpt = "/home/ttuser/.boltz/boltz2_conf.ckpt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]

    def scope(prefix, strip=""):
        out = {}
        for k, v in state.items():
            if not k.startswith(prefix):
                continue
            key = k[len(prefix):]
            if strip and key.startswith(strip):
                key = key[len(strip):]
            out[key] = v.float()
        return out

    device = T.get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)

    layer = "pairformer_module.layers.0."
    opm = scope("msa_module.layers.0.outer_product_mean.")
    c_z = scope(layer + "tri_mul_out.")["norm_in.weight"].shape[0]
    c_m = opm["norm.weight"].shape[0]
    print(f"c_z={c_z} c_m={c_m}", flush=True)

    def build():
        return {
            "tri_mul_out": T.TriangleMultiplication(
                False, scope(layer + "tri_mul_out."), ckc),
            "tri_mul_in": T.TriangleMultiplication(
                True, scope(layer + "tri_mul_in."), ckc),
            "tri_att_start": T.TriangleAttention(
                32, 4, False, scope(layer + "tri_att_start.", "mha."), ckc),
            "tri_att_end": T.TriangleAttention(
                32, 4, True, scope(layer + "tri_att_end.", "mha."), ckc),
            "opm": T.OuterProductMean(opm, ckc),
        }

    seq_default = T.SEQ_LEN_MORE_CHUNKING
    bytes_default = T.TRIMUL_IN_NORM_ROWBLOCK_BYTES
    configs = [
        ("whole", 64, seq_default, bytes_default),
        ("rowtail", 192, 128, bytes_default),
        ("rownorm", 192, 128, 1),
    ]
    out = {}
    for label, L, seq_chunk, norm_bytes in configs:
        T.SEQ_LEN_MORE_CHUNKING = seq_chunk
        T.TRIMUL_IN_NORM_ROWBLOCK_BYTES = norm_bytes
        # The persistent-mask triangle-attention kernel throws under a lowered chunking
        # constant on BOTH trees (it is calibrated for the real one), so the chunked
        # configurations run the stock SDPA. Same switch in both dumps.
        T._triatt_sdpa._ENABLED = seq_chunk == seq_default
        # The row chunk has to divide the sequence into at least two parts: at exactly one part
        # the chunked bias loop concats a single tensor and then frees it, which frees the
        # concat's own buffer. Unreachable at the shipped constants (chunk 512 < 1536), so it is
        # a property of this harness, not a regression -- see the state doc.
        T.TRIANGLE_ATT_CHUNK_SIZE = 512 if seq_chunk == seq_default else 64
        mods = build()
        torch.manual_seed(L)
        z = torch.randn(1, L, L, c_z)
        m = torch.randn(1, 4, L, c_m)
        for name, mod in mods.items():
            x = m if name == "opm" else z
            t = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                device=device, dtype=ttnn.bfloat16)
            key = f"{label}/{name}/{L}"
            out[key] = torch.Tensor(ttnn.to_torch(mod(t))).float().clone()
            print(f"{key} {tuple(out[key].shape)}", flush=True)
    T.SEQ_LEN_MORE_CHUNKING = seq_default
    T.TRIMUL_IN_NORM_ROWBLOCK_BYTES = bytes_default
    torch.save(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
