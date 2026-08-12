#!/usr/bin/env python3
"""The fused SDPA scales the *mask* as well as the scores. openfold3 does not want that.

`s1_fp32_cb_sdpa.py` refuted the reason recorded in `tenstorrent.py:2747`. Widening the
exponentiated-score CBs to fp32 does not recover any accuracy at all -- rmsd/std 0.890 against the
bf16 CBs' 0.685 and the shipped `_fp32_softmax_attention`'s 0.026. Precision is not the defect.

The kernel says what is. `compute_common.hpp:1814` adds the mask into `cb_qk_im` and
`:1843` then exponentiates with `sub_exp_block_bcast_cols_inplace<..., scale_fp32, ...>`, so the
fused op computes

    softmax(scale * (q@k^T + mask))          NOT   softmax(scale * q@k^T + mask)

Boltz and Protenix pre-bake the pair bias by sqrt(head_dim) (`_bias_scale`, tenstorrent.py:2706),
which cancels exactly. openfold3 constructs with `scale_pair_bias=False`, so `_bias_scale` is 1.0 and
routing it through the fused op divides its pair bias by sqrt(32) = 5.657.

This screen tests that directly: same fused arms, bias pre-multiplied by 1/scale. If the error
collapses onto `_fp32_softmax_attention`, the 27.347 A was a bias-weighting defect and the fused
route is open again.
"""
import json, os, statistics, sys, time
import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tt_bio import tenstorrent as T          # noqa: E402
from tt_bio import sdpa_generic as SG        # noqa: E402
from tt_bio import triatt_sdpa as TS         # noqa: E402

H, S, D = 4, 512, 32
B_ACC, B_PERF = 64, 512
SCALE = D ** -0.5
Q_CHUNK = 512
CKC_SHIP = (ttnn.MathFidelity.HiFi2, True, False, False)
CKC_HI = (ttnn.MathFidelity.HiFi4, False, True, False)
WARM, REPS = 3, 7
CKC_TRUNK = None
F32 = ttnn.float32


def host_ref(q, k, v, bias, scale_the_bias):
    """float64 gold. `scale_the_bias` selects which of the two semantics is being modelled."""
    out = torch.empty(q.shape, dtype=torch.float64)
    for s in range(0, q.shape[0], 8):
        e = min(s + 8, q.shape[0])
        sc = q[s:e].to(torch.float64) @ k[s:e].to(torch.float64).transpose(-1, -2)
        b = bias.to(torch.float64)
        sc = (sc + b) * SCALE if scale_the_bias else sc * SCALE + b
        out[s:e] = torch.softmax(sc, dim=-1) @ v[s:e].to(torch.float64)
    return out


def err(got, ref):
    g = got.to(torch.float64)
    d = g - ref
    rmsd = float(torch.sqrt((d ** 2).mean()))
    gc, rc = g.flatten() - g.mean(), ref.flatten() - ref.mean()
    return {"max_abs": round(float(d.abs().max()), 6), "rmsd": round(rmsd, 6),
            "rmsd_over_std": round(rmsd / float(ref.std()), 6),
            "pcc": round(float((gc * rc).sum() / (gc.norm() * rc.norm())), 8)}


def to_dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def arm_fp32_softmax(q, k, v, bias):
    return T._fp32_softmax_attention(q, k, v, bias, scale_inv=SCALE,
                                     compute_kernel_config=CKC_TRUNK,
                                     out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)


def _transcribed(q, k, v, bias, im_dtype, stats_dtype, k_chunk, ckc):
    grid = tuple(T.COMPUTE_GRID_MAIN)
    split = (grid[0] * grid[1] // H, H, 1)
    dev = q.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in q.shape]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    p = SG.plan(q, k, v, bias, out, Q_CHUNK, k_chunk, grid, ckc, SCALE, split)
    SG.sdpa(dev, q, k, v, bias, out, Q_CHUNK, k_chunk, grid, ckc, SCALE, split=split,
            kernel_dir=TS.KERNEL_DIR,
            mask_cb_tiles=p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"],
            defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]},
            im_dtype=im_dtype, stats_dtype=stats_dtype)
    return out


def main():
    global CKC_TRUNK
    torch.manual_seed(0)
    dev = T.get_device()
    kc = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
          else ttnn.types.BlackholeComputeKernelConfig)
    CKC_TRUNK = kc(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                   fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(T.COMPUTE_GRID_MAIN), "shape": {"H": H, "S": S, "D": D},
           "scale": SCALE, "prescale": 1.0 / SCALE, "accuracy": [], "perf": []}

    qh, kh, vh = (torch.randn(B_ACC, H, S, D).to(torch.bfloat16) for _ in range(3))
    bh = torch.randn(1, H, S, S).to(torch.bfloat16)
    ref = host_ref(qh, kh, vh, bh, scale_the_bias=False)          # what openfold3 wants
    ref_scaled_bias = host_ref(qh, kh, vh, bh, scale_the_bias=True)   # what the fused op computes
    res["host_gap"] = {"note": "float64, no device: the two semantics against each other",
                       **err(ref_scaled_bias, ref)}
    print(json.dumps(res["host_gap"]), flush=True)

    # the bias the fused op must be handed so that scale*(qk + bias') == scale*qk + bias
    bh_pre = (bh.to(torch.float32) / SCALE).to(torch.bfloat16)
    qd, kd, vd = (to_dev(t, dev) for t in (qh, kh, vh))
    bd, bd_pre = to_dev(bh, dev), to_dev(bh_pre, dev)

    arms = [
        ("_fp32_softmax_attention (shipped)", lambda b: arm_fp32_softmax(qd, kd, vd, b), bd),
        ("fused bf16 CB, bias as-is", lambda b: _transcribed(qd, kd, vd, b, None, None, 256, CKC_SHIP), bd),
        ("fused bf16 CB, bias PRESCALED", lambda b: _transcribed(qd, kd, vd, b, None, None, 256, CKC_SHIP), bd_pre),
        ("fused bf16 CB, HiFi4+fp32dst, bias PRESCALED",
         lambda b: _transcribed(qd, kd, vd, b, None, None, 256, CKC_HI), bd_pre),
        ("fused fp32 CB k=128, HiFi4+fp32dst, bias PRESCALED",
         lambda b: _transcribed(qd, kd, vd, b, F32, F32, 128, CKC_HI), bd_pre),
    ]
    for name, fn, b in arms:
        row = {"arm": name}
        try:
            o = fn(b)
            ttnn.synchronize_device(dev)
            row.update(err(ttnn.to_torch(o), ref))
            ttnn.deallocate(o)
        except Exception as exc:                                       # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        res["accuracy"].append(row)
        print(json.dumps(row), flush=True)
    for t in (qd, kd, vd, bd, bd_pre):
        ttnn.deallocate(t)

    # ---- perf at the real batch
    z = torch.zeros(B_PERF, H, S, D).to(torch.bfloat16)
    zb = torch.zeros(1, H, S, S).to(torch.bfloat16)
    qd, kd, vd, bd = (to_dev(t, dev) for t in (z, z, z, zb))
    for name, fn in (("_fp32_softmax_attention (shipped)", lambda: arm_fp32_softmax(qd, kd, vd, bd)),
                     ("fused bf16 CB, ship ckc, k=256",
                      lambda: _transcribed(qd, kd, vd, bd, None, None, 256, CKC_SHIP)),
                     ("fused bf16 CB, HiFi4+fp32dst, k=256",
                      lambda: _transcribed(qd, kd, vd, bd, None, None, 256, CKC_HI)),
                     ("fused fp32 CB, HiFi4+fp32dst, k=128",
                      lambda: _transcribed(qd, kd, vd, bd, F32, F32, 128, CKC_HI))):
        row = {"arm": name}
        try:
            for _ in range(WARM):
                o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
            ms = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                o = fn(); ttnn.synchronize_device(dev)
                ms.append((time.perf_counter() - t0) * 1e3); ttnn.deallocate(o)
            row["ms"] = round(statistics.median(ms), 4)
        except Exception as exc:                                       # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        res["perf"].append(row)
        print(json.dumps(row), flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_bias_scale.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
