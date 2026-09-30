#!/usr/bin/env python3
"""Forward AND gradient of BindCraft 2's taped fused-HiFi triangle attention, against float64.

The call is the one a BC2 round makes: a tape open, `TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi`, and
`tenstorrent._tri_att_sdpa_hifi`, whose tape entry takes the fused arm's forward value and puts
`autograd.triangle_attention`'s chunked recompute behind it. Arms per length:

  padup   the shipped route (a 32*p axis now pads one tile up)
  off     `_TRIATT_HIFI_PAD_UP_TILES = 0`, the route before this change (fallback path)

Graded against torch float64 in the kernel's own convention, softmax((qk + bias) * scale) @ v,
with the VJP of the same function. Lengths that serve natively (576, 640, 768) are the controls:
they say what this kernel's own error is one tile away. Leading dim capped (`--batch`): which rung
serves reads shape[2] only (perf/bcx_tapedfwd/khole.py), and the served stat is recorded per arm
so that is checked, not assumed.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi"
H, D = 4, 32


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="544,576,608,640,736,768")
    ap.add_argument("--arms", default="padup,off")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("grade.json"))
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.autograd as ag
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    dev = T.get_device()
    up = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scale = D ** -0.5
    tiles0 = T._TRIATT_HIFI_PAD_UP_TILES
    rows = []
    for n in [int(x) for x in a.ns.split(",")]:
        g_ = torch.Generator().manual_seed(n)
        mk = lambda *s: torch.randn(*s, generator=g_, dtype=torch.float64)
        q, k, v, g = (mk(a.batch, H, n, D) for _ in range(4))
        bias = mk(1, H, n, n)
        # Round the operands to bf16 first, so the reference computes the function of exactly the
        # values the card is given and only the device's own error is left in the grade.
        q, k, v, g, bias = (t.to(torch.bfloat16).double() for t in (q, k, v, g, bias))
        qr, kr, vr, br = (t.clone().requires_grad_(True) for t in (q, k, v, bias))
        o_ref = torch.softmax((qr @ kr.transpose(-1, -2) + br) * scale, -1) @ vr
        o_ref.backward(g)
        ref = {"out": o_ref.detach(), "dq": qr.grad, "dk": kr.grad, "dv": vr.grad, "dbias": br.grad}
        for arm in a.arms.split(","):
            T._TRIATT_HIFI_PAD_UP_TILES = tiles0 if arm == "padup" else 0
            T._TRIATT_HIFI_OVER_L1.clear()
            T.TRIATT_FUSED_HIFI_PADDED.clear()
            s0 = dict(T.TRIATT_FUSED_HIFI_STATS)
            leaves = [ag.Tensor(up(t), requires_grad=True) for t in (q, k, v, bias)]
            try:
                with ag.tape():
                    out = T._tri_att_sdpa_hifi(*leaves, scale)
                    fused = out is not None
                    if out is None:   # the Evoformer's own next rung under a tape
                        out = T._fp32_softmax_attention(
                            *leaves, scale_inv=scale ** -1,
                            compute_kernel_config=None, out_dtype=ttnn.bfloat16)
                ag.backward([out], [up(g)])
                got = {"out": ttnn.to_torch(out.value if hasattr(out, "value") else out).double()}
                for nm, t in zip(("dq", "dk", "dv", "dbias"), leaves):
                    got[nm] = ttnn.to_torch(t.grad).double()
                err = {kk: rel(got[kk], ref[kk]) for kk in ref}
                status = "ok"
            except Exception as exc:  # noqa: BLE001 -- the off arm is expected to refuse at some n
                err, fused, status = None, False, f"{type(exc).__name__}: {str(exc)[:300]}"
            rec = {"n": n, "arm": arm, "batch": a.batch, "fused_forward": fused,
                   "padded_to": T.TRIATT_FUSED_HIFI_PADDED.get(n),
                   "stats": {kk: T.TRIATT_FUSED_HIFI_STATS[kk] - s0[kk] for kk in s0},
                   "rel_l2_vs_f64": err, "status": status}
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            a.out.write_text(json.dumps({"heads": H, "head_dim": D, "rows": rows}, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
