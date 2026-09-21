#!/usr/bin/env python3
"""Which arm is wrong, adjudicated against float64 on identical values.

`c12-diffusion-head-major` pass 10 read `torch.equal(A1, B) = False` at `apb_trunk`, max_abs
0.0703125, and attributed it to the head-major destination. That attribution cannot follow from
that comparison, for two reasons:

  * a wrong DESTINATION permutes whole tiles, so the difference is the size of the values
    themselves; 0.0703 against outputs of order 1 is a handful of bf16 ULP, which is arithmetic;
  * A1 and B do not differ only in their destination. A1 is the WHEEL's `minimal_matmul`; B is
    tt-bio's `generic_op` transcription of it on our own kernels. Two changes, one reading.

So this runs the missing middle arm and adjudicates against float64 rather than against another
device arm:

    REF64  x @ w + bias in float64 on the host, from the same values
    A0     ttnn.linear + nlp_create_qkv_heads            the chain that ships
    A1     wheel minimal_matmul + nlp_create_qkv_heads   op class changed, destination unchanged
    A2     generic_op transcription, PLAIN writer, + nlp_create_qkv_heads   transcription only
    B      generic_op transcription, HEAD-MAJOR writer   destination changed too

A1 -> A2 isolates the transcription. A2 -> B isolates the destination. If B and A2 agree, the
transcription is the defect and the tile map is exonerated; if they disagree, the destination is.

    python3 bisect.py --sig apb_trunk
    python3 bisect.py --sig dit_token --arm B      # one arm per process, so a hang costs an arm
"""
import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
T0 = time.time()

# name -> batch, seq, c_in, heads, head_dim, padded_head_dim, site
#
# `triatt_b2` is NOT part of the lever. It is the same transcription at the key and the core-grid
# orientation the triangle attention SHIPS on by default (`TRIATT_HEAD_MAJOR_QKV = True`), so if
# the transcription loses accuracy generally rather than only in the orientation this lever needs,
# that is a live inference finding and not a lever finding. M is 2048 rows so M > N and
# `transpose_core_grid` is TRUE, which is the orientation the tri-attention runs.
SIGS = {
    "apb_trunk": (1, 512, 384, 16, 32, 32, "apb"),
    "dit_token": (1, 512, 768, 16, 48, 64, "apb"),
    "triatt_b2": (1, 2048, 128, 4, 32, 32, "triatt"),
}
TILE = 32


def say(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def build(device, sig_name):
    import torch
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ
    from tt_bio import mm_generic as MG

    b, s, c_in, heads, hd, phd, site = SIGS[sig_name]
    torch.manual_seed(0)
    x_t = torch.randn(b, s, c_in, dtype=torch.bfloat16)
    w_t = (torch.randn(c_in, 3 * heads * phd, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    bias_t = (torch.randn(3 * heads * phd, dtype=torch.bfloat16) * 0.02).to(torch.bfloat16)
    mk = dict(layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w, bias = (ttnn.from_torch(t, **mk) for t in (x_t, w_t, bias_t))

    # The adjudicator: the same values, contracted in float64 on the host. Every arm is scored
    # against this and never against another arm.
    ref64 = (x_t.double() @ w_t.double() + bias_t.double())
    ref = [ref64[..., c * heads * phd:(c + 1) * heads * phd]
           .reshape(b, s, heads, phd).permute(0, 2, 1, 3).contiguous() for c in range(3)]

    cls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = TT._qkv_mm_config(x, w, site)
    assert cfg is not None, "no block config"
    blk = TT._mm_block_at_site((c_in + 31) // 32, (3 * heads * phd + 31) // 32, site)
    say(f"site={site} block={blk} is_MM_DEFAULT={blk is TT._MM_DEFAULT} "
        f"M={b * s} N={3 * heads * phd} transpose_core_grid={b * s > 3 * heads * phd}")

    def split(full):
        return list(ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(full, 1), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False))

    def a0():
        return split(ttnn.linear(x, w, bias=bias, compute_kernel_config=ckc,
                                 core_grid=TT.CORE_GRID_MAIN))

    def a1():
        return split(ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, bias_tensor=bias, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg))

    def a2():
        full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([b, s, 3 * heads * phd]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
            device, ttnn.DRAM_MEMORY_CONFIG)
        MG.generic_minimal_matmul(device, x, w, [full], (blk, tuple(TT.COMPUTE_GRID_MAIN)),
                                  MG.ckc_args(ckc), {}, TQ.KERNEL_DIR, bias=bias)
        return split(full)

    def arm_b():
        TQ._APB_ENABLED = True
        o = TQ.qkv_heads(x, w, ckc, heads, phd, ttnn.bfloat16, cfg, bias=bias,
                         allow_m_le_n=True, site=site)
        assert o is not None, f"declined: {TQ.APB_REJECTS} {TQ.REJECTS}"
        return list(o)

    return {"A0": a0, "A1": a1, "A2": a2, "B": arm_b}, ref, (b, s, heads, phd)


def score(name, got, ref, geom):
    import torch
    b, s, heads, phd = geom
    shapes = [list(g.shape) for g in got]
    if any(tuple(g.shape) != tuple(r.shape) for g, r in zip(got, ref)):
        return {"arm": name, "shape_mismatch": shapes,
                "expected": [list(r.shape) for r in ref]}
    err = [float((g.double() - r).abs().max()) for g, r in zip(got, ref)]
    scale = max(float(r.abs().max()) for r in ref)
    return {"arm": name, "shapes": shapes, "max_abs_vs_float64": max(err),
            "per_chunk_vs_float64": err, "ref_max_abs": scale,
            "rel_vs_float64": max(err) / scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default="apb_trunk", choices=sorted(SIGS))
    ap.add_argument("--arm", default="", help="run ONE arm (A0|A1|A2|B), so a hang costs one arm")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    import torch
    import ttnn                                                     # noqa: F401
    from tt_bio import tenstorrent as TT
    say(f"{a.sig}: opening device via tt_bio.tenstorrent.get_device()")
    dev = TT.get_device()
    say(f"device open, arch={dev.arch()}")

    arms, ref, geom = build(dev, a.sig)
    say("arms built")
    names = [a.arm] if a.arm else ["A0", "A1", "A2", "B"]
    res = {"sig": a.sig, "arms": []}
    got = {}
    for n in names:
        say(f"running {n}")
        out = [ttnn.to_torch(t) for t in arms[n]()]
        got[n] = out
        r = score(n, out, ref, geom)
        res["arms"].append(r)
        say(f"  {n}: {json.dumps({k: v for k, v in r.items() if k != 'shapes'})}")
    # and the arm-to-arm equalities, which are what the transcription claims
    res["equal"] = {}
    for p, q in (("A0", "A1"), ("A1", "A2"), ("A2", "B"), ("A1", "B")):
        if p in got and q in got and all(tuple(u.shape) == tuple(v.shape)
                                         for u, v in zip(got[p], got[q])):
            res["equal"][f"{p}=={q}"] = all(torch.equal(u, v) for u, v in zip(got[p], got[q]))
    say(f"equal: {res['equal']}")
    path = OUT / (a.out or f"bisect_{a.sig}{'_' + a.arm if a.arm else ''}.json")
    path.write_text(json.dumps(res, indent=1) + "\n")
    say(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
