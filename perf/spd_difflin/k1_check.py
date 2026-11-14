"""Float64 error of the token DiT's linears as dit_mm16 calls them, fused silu and bias included.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=normal python perf/spd_difflin/k1_check.py

op_probe.py graded the K block 1 program without a fused activation. dit_mm16 failed its fold grade with
predictions uniformly expanded (CA-CA 3.78 -> 3.80-3.83 A, CA-lDDT -0.08) while fast mode's all-bf16 DiT
costs -0.003, so each call here runs k1_linear and ttnn.linear at the same formats and prints the rms and
max error of both against float64, plus their mean error (a bias, not noise, is what expands a structure).
"""
import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

CASES = [  # name, K, N, bias, activation, out dtype
    ("qkv", 768, 3072, True, None, ttnn.float32),
    ("g", 768, 768, False, None, ttnn.bfloat16),
    ("a1", 768, 1536, False, "silu", ttnn.bfloat16),
    ("a2", 768, 1536, False, None, ttnn.bfloat16),
    ("b", 1536, 768, False, None, ttnn.float32),
]


def main():
    dev = T.get_device()
    print("levers", sorted(T.active_levers()) if hasattr(T, "active_levers") else "?", "arch", T.arch_name())
    for l1acc in (True, False):
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True,
                                                     packer_l1_acc=l1acc)
        for name, k, n, has_bias, act, od in CASES:
            torch.manual_seed(7)
            A = torch.randn(5, 768, k).bfloat16().float()
            W = (torch.randn(k, n) / k ** 0.5).bfloat16().float()
            Bv = (torch.randn(n) * 0.1).bfloat16().float() if has_bias else None
            R = A.double() @ W.double() + (Bv.double() if has_bias else 0)
            if act == "silu":
                R = torch.nn.functional.silu(R)
            ta = ttnn.from_torch(A, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            tw = ttnn.from_torch(W, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            tb = ttnn.from_torch(Bv.reshape(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev) if has_bias else None
            outs = {
                "k1": lambda: T.k1_linear(ta, tw, tb, compute_kernel_config=ckc, dtype=od, activation=act),
                "auto": lambda: ttnn.linear(ta, tw, bias=tb, activation=act, dtype=od,
                                            compute_kernel_config=T.silu_ckc(ckc) if act == "silu" else ckc,
                                            core_grid=T.CORE_GRID_MAIN),
            }
            for lab, fn in outs.items():
                Y = ttnn.to_torch(fn()).double().reshape(R.shape)
                e = Y - R
                print(f"l1acc={int(l1acc)} {name:4s} {lab:4s} rms {e.pow(2).mean().sqrt():.3e} "
                      f"max {e.abs().max():.3e} mean {e.mean():+.3e} ref_rms {R.pow(2).mean().sqrt():.3f}",
                      flush=True)


def elementwise():
    """The two non-matmul ops dit_mm16 changes: the gate multiply (fp32 attention x sigmoid(bf16 gate), bf16
    out) and AdaLN's final add written bf16 from fp32 operands."""
    dev = T.get_device()
    torch.manual_seed(8)
    o = torch.randn(5, 768, 768); g = torch.randn(5, 768, 768).bfloat16().float()
    R = o.double() * torch.sigmoid(g.double())
    to = ttnn.from_torch(o, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    for gd in (ttnn.bfloat16, ttnn.float32):
        tg = ttnn.from_torch(g, dtype=gd, layout=ttnn.TILE_LAYOUT, device=dev)
        for od in (ttnn.bfloat16, ttnn.float32):
            Y = ttnn.to_torch(ttnn.multiply(to, tg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                                            dtype=od)).double()
            e = Y - R
            print(f"gate g={gd} out={od} rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e} "
                  f"mean {e.mean():+.3e}", flush=True)
    # Mixed operand formats, with the sigmoid fused on b and without it (sigmoid done on the host).
    DT = dict(f32=ttnn.float32, bf16=ttnn.bfloat16, b8=ttnn.bfloat8_b)
    for od_, gd_, out_ in (("f32", "bf16", "bf16"), ("bf16", "bf16", "bf16"), ("bf16", "b8", "b8"),
                           ("bf16", "bf16", "b8"), ("f32", "f32", "bf16")):
        tg = ttnn.from_torch(g, dtype=DT[gd_], layout=ttnn.TILE_LAYOUT, device=dev)
        tsg = ttnn.from_torch(torch.sigmoid(g), dtype=DT[gd_], layout=ttnn.TILE_LAYOUT, device=dev)
        tob = ttnn.from_torch(o, dtype=DT[od_], layout=ttnn.TILE_LAYOUT, device=dev)
        Rq = ttnn.to_torch(tob).double() * torch.sigmoid(ttnn.to_torch(tg).double())
        for lab, fn in (("fused", lambda: ttnn.multiply(tob, tg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                                                         dtype=DT[out_])),
                        ("plain", lambda: ttnn.multiply(tob, tsg, dtype=DT[out_]))):
            e = ttnn.to_torch(fn()).double() - Rq
            print(f"mix o={od_} g={gd_} out={out_} {lab} rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e}",
                  flush=True)
    a = torch.randn(5, 768, 768); sb = torch.randn(5, 768, 768)
    R = a.double() + sb.double()
    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ts = ttnn.from_torch(sb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    for od in (ttnn.bfloat16, ttnn.float32):
        Y = ttnn.to_torch(ttnn.add(ta, ts, dtype=od)).double()
        e = Y - R
        print(f"adaln add out={od} rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e} mean {e.mean():+.3e}",
              flush=True)
    Y = ttnn.to_torch(ttnn.typecast(ta, ttnn.bfloat16)).double()
    e = Y - a.double()
    print(f"typecast f32->bf16 rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e} mean {e.mean():+.3e}")


if __name__ == "__main__":
    import sys
    elementwise() if "--elementwise" in sys.argv else main()
