"""pairfuse A: which half of pair_outer_fold departs from the unfused path. Same inputs as the device test (n=256).
Arms: U unfused (outer_residual, then tri_out on z'), F both folds, T T1 fold only (the tail's fold form declined;
its update is added on the host, so U-T is that rounding, not the kernel), a zero product, and the fallback with the
tail declined in both arms. Bit-identical on every arm with kernels/common/outer_add.hpp (10-23).
  TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> PYTHONPATH=.:perf:tests python3 perf/rfd4/pairfuse/diag.py [n]"""
import sys
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TT1
from tt_bio import outer_residual as OR
from tt_bio.rfd4 import trunk as TR
import rfd4_reference as REF

n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
dev = T.get_device()
w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
x, z, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
zb = REF.bf16(z.double())
d = lambda t: ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
h = lambda t: ttnn.to_torch(t).double()
cap = []
fin, ftl = TT1.fused_in_ln, TT1.fused_tail_ln
def in_rec(*a, **k):
    o = fin(*a, **k)
    if o is not None:
        cap.append([h(t) for t in o[:3] if t is not None])
    return o
TT1.fused_in_ln = in_rec
decline = [False]
def tail_dec(*a, **k):
    if decline[0] and k.get("outer") is not None:
        return None
    return ftl(*a, **k)
TT1.fused_tail_ln = tail_dec
with T.lever_scope(frozenset(TR.NORMAL_LEVERS | {"pair_bf16z", "pair_outer_fold"})):
    mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                        TR.compute_kernel_config(), residual_dtype=ttnn.bfloat16)
    pi, pj = mod._factors(d(x), None)
    pj4 = ttnn.reshape(pj, (1, n, 1, 384))
    zp = OR.outer_residual(d(zb), pi, pj4)[0]
    pih, pjh = h(pi).reshape(1, n, 384), h(pj).reshape(1, n, 384)
    zph = REF.bf16(zb + pjh[:, :, None, :] * pih[:, None, :, :])
    print("outer_residual vs host bf16(z + pj*pi): equal", torch.equal(h(zp).reshape(zph.shape), zph),
          "max", (h(zp).reshape(zph.shape) - zph).abs().max().item())
    def arm(fold, dec):
        decline[0] = dec
        for _ in range(2):   # warmup, then the read
            cap.clear()
            t = d(zb) if fold else ttnn.clone(zp)
            u = mod.tri_out(t, None, stream32=t, outer=(pi, pj4) if fold else None)
            took = mod.tri_out.took_residual
        out = h(t) if took else h(t) + h(u)
        return out.reshape(zph.shape), [c for c in cap], took
    U, cu, tu = arm(False, False)
    F, cf, tf = arm(True, False)
    Tn, ct, tt = arm(True, True)
    print("took U/F/T", tu, tf, tt, "T1 calls", len(cu), len(cf), len(ct), "OUTER_STATS", TT1.OUTER_STATS)
    for name, c in (("F", cf), ("T", ct)):
        for k, (a, b) in enumerate(zip(cu[0], c[0])):
            print(f"T1 out[{k}] U vs {name}: equal {torch.equal(a, b)} max {(a - b).abs().max().item():.3e}")
    for name, o in (("F", F), ("T", Tn)):
        print(f"final U vs {name}: equal {torch.equal(U, o)} max {(U - o).abs().max().item():.3e} "
              f"n_diff {(U != o).sum().item()} of {U.numel()}")
    rr = lambda a, b: ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()
    # the scale an index bug would show: the update on z without the product at all
    zp_save = zp
    zp = d(zb)
    U0, cu0, _ = arm(False, False)
    print(f"rel rms  U-F {rr(F, U):.3e}   U-T {rr(Tn, U):.3e}   effect of the product U-U0 {rr(U0, U):.3e}")
    # zero product: the fold must equal the unfused path on z itself, bit for bit
    zero = ttnn.from_torch(torch.zeros(1, n, 384), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pi, pj4 = zero, ttnn.reshape(ttnn.clone(zero), (1, n, 1, 384))
    F0, c0, _ = arm(True, False)
    print(f"zero product: fold vs unfused-on-z equal {torch.equal(F0, U0)} max {(F0 - U0).abs().max().item():.3e}",
          "T1 outs equal", [torch.equal(a, b) for a, b in zip(c0[0], cu0[0])])
    # the fallback (T1 folded, tail declined, z' written in place): against the unfused path with its tail declined
    # too, both return the bare update and leave z' in the stream
    pi, pj = mod._factors(d(x), None)
    pj4 = ttnn.reshape(pj, (1, n, 1, 384))
    zp = zp_save
    TT1.fused_tail_ln = lambda *a, **k: None
    def bare(fold):
        for _ in range(2):
            t = d(zb) if fold else ttnn.clone(zp)
            u = mod.tri_out(t, None, stream32=t, outer=(pi, pj4) if fold else None)
        return h(t), h(u)
    (tu, uu), (tf, uf) = bare(False), bare(True)
    print(f"fallback: stream z' equal {torch.equal(tu, tf)}, bare update equal {torch.equal(uu, uf)} "
          f"max {(uu - uf).abs().max().item():.3e}")
