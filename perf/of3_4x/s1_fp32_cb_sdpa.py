#!/usr/bin/env python3
"""Does a wider intermediate CB inside our own SDPA transcription recover openfold3's accuracy?

`of3-fp32-dst-sdpa` closed the route "run openfold3 triangle attention on the fused SDPA": it is
worth 1.3745x on the fold and moves the structure 27.347 A. The reason recorded in
`tenstorrent.py:2747` is specific -- "sdpa_generic keeps the exponentiated scores in a bf16 circular
buffer, so fp32_dest_acc never reaches them". A compute-kernel *config* cannot fix that. A CB
*format* can, and tt_bio owns the transcription (`tt_bio/sdpa_generic.py`, cb 24/25/26 for the
exponentiated scores and cb 27-31 for the online-softmax running max/sum).

This screen answers, off the fold and before any kernel work:

  1. how far each SDPA arm is from a float64 host reference, on the same bf16 inputs, and
  2. what each arm costs at the openfold3 512 aa triangle-attention shape.

The bar to beat is `_fp32_softmax_attention`, the shipped path, which is what produced plDDT
0.547851 / CIF da9b4ed68f8c0405. An arm that matches its error is a candidate; an arm that sits at
the stock op's error is the 27 A structure again.
"""
import json, os, statistics, sys, time
import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tt_bio import tenstorrent as T          # noqa: E402
from tt_bio import sdpa_generic as SG        # noqa: E402
from tt_bio import triatt_sdpa as TS         # noqa: E402

H, S, D = 4, 512, 32
B_ACC = 64        # accuracy arm: error is per-element, the batch only costs host memory
B_PERF = 512      # perf arm: openfold3's real triangle-attention batch at 512 aa
SCALE = D ** -0.5
Q_CHUNK, K_CHUNK = 512, 256     # what the fold ships (_tri_att_q_chunks widens q, k stays 256)
# (math_fidelity, math_approx, fp32_dest_acc, dst_full_sync)
CKC_SHIP = (ttnn.MathFidelity.HiFi2, True, False, False)    # triatt_sdpa's own default today
CKC_HI = (ttnn.MathFidelity.HiFi4, False, True, False)      # what the trunk asks for elsewhere
CKC = CKC_SHIP
WARM, REPS = 3, 7
CKC_TRUNK = None


def host_ref(q, k, v, bias):
    """float64 softmax attention from the same bf16 inputs, in batch chunks."""
    out = torch.empty(q.shape, dtype=torch.float64)
    for s in range(0, q.shape[0], 8):
        e = min(s + 8, q.shape[0])
        qq = q[s:e].to(torch.float64)
        sc = qq @ k[s:e].to(torch.float64).transpose(-1, -2)
        sc = sc * SCALE + bias.to(torch.float64)
        out[s:e] = torch.softmax(sc, dim=-1) @ v[s:e].to(torch.float64)
    return out


def err(got, ref):
    g = got.to(torch.float64)
    d = g - ref
    rmsd = float(torch.sqrt((d ** 2).mean()))
    std = float(ref.std())
    gc, rc = g.flatten(), ref.flatten()
    gc = gc - gc.mean(); rc = rc - rc.mean()
    pcc = float((gc * rc).sum() / (gc.norm() * rc.norm()))
    return {"max_abs": float(d.abs().max()), "rmsd": rmsd, "rmsd_over_std": rmsd / std,
            "pcc": pcc}


def to_dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------- the arms
def arm_fp32_softmax(q, k, v, bias):
    return T._fp32_softmax_attention(q, k, v, bias, scale_inv=SCALE,
                                     compute_kernel_config=CKC_TRUNK,
                                     out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)


def arm_stock(q, k, v, bias):
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=SCALE,
        program_config=T._sdpa_program_config(q_chunk_size=Q_CHUNK, k_chunk_size=K_CHUNK))


def _transcribed(q, k, v, bias, im_dtype, stats_dtype, k_chunk, ckc=CKC_SHIP, exp_approx=False):
    """triatt_sdpa.sdpa's body, with the CB widths, ckc and k_chunk exposed."""
    grid = tuple(T.COMPUTE_GRID_MAIN)
    split = (grid[0] * grid[1] // H, H, 1)
    dev = q.device()
    shape = [int(d) for d in q.shape]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                         dev, ttnn.DRAM_MEMORY_CONFIG)
    p = SG.plan(q, k, v, bias, out, Q_CHUNK, k_chunk, grid, ckc, SCALE, split)
    assert p["nh_per_core"] == 1 and p["q_per_core"] == 1 and p["bcast_batch"] \
        and not p["use_padded_mask"], p
    persistent = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    SG.sdpa(dev, q, k, v, bias, out, Q_CHUNK, k_chunk, grid, ckc, SCALE, split=split,
            kernel_dir=TS.KERNEL_DIR, mask_cb_tiles=persistent,
            defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]},
            im_dtype=im_dtype, stats_dtype=stats_dtype, exp_approx_mode=exp_approx)
    return out


def l1_bytes(q, k, v, bias, out, im_dtype, stats_dtype, k_chunk):
    grid = tuple(T.COMPUTE_GRID_MAIN)
    split = (grid[0] * grid[1] // H, H, 1)
    p = SG.plan(q, k, v, bias, out, Q_CHUNK, k_chunk, grid, CKC, SCALE, split)
    tb = SG._TILE_BYTES
    im, st = tb[im_dtype or ttnn.bfloat16], tb[stats_dtype or ttnn.bfloat16]
    persistent = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    return (p["q_tiles"] * 2048 + p["k_tiles"] * 2048 + p["v_tiles"] * 2048
            + persistent * 2048 + 2 * 2048 + im
            + p["qk_tiles"] * im + 2 * p["out_im_tiles"] * im
            + 5 * p["statistics_tiles"] * st + p["out0_t"] * 2048)


def _tr(im, st, kc, ckc):
    return lambda q, k, v, b: _transcribed(q, k, v, b, im, st, kc, ckc)


F32 = ttnn.float32
ARMS = [
    ("_fp32_softmax_attention (shipped)", arm_fp32_softmax),
    ("stock ttnn SDPA (the 27.347 A arm)", arm_stock),
    ("transcription bf16 im/stats, ship ckc, k=256", _tr(None, None, 256, CKC_SHIP)),
    ("transcription bf16 im/stats, HiFi4+fp32dst, k=256", _tr(None, None, 256, CKC_HI)),
    ("transcription fp32 im, bf16 stats, HiFi4+fp32dst, k=256", _tr(F32, None, 256, CKC_HI)),
    ("transcription fp32 im+stats, HiFi4+fp32dst, k=256", _tr(F32, F32, 256, CKC_HI)),
    ("transcription fp32 im+stats, HiFi4+fp32dst, k=128", _tr(F32, F32, 128, CKC_HI)),
    ("transcription fp32 im+stats, HiFi4+fp32dst, k=64", _tr(F32, F32, 64, CKC_HI)),
    ("transcription fp32 im+stats, ship ckc, k=128", _tr(F32, F32, 128, CKC_SHIP)),
]


def main():
    torch.manual_seed(0)
    global CKC_TRUNK
    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    CKC_TRUNK = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                           fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(T.COMPUTE_GRID_MAIN), "shape": {"H": H, "S": S, "D": D},
           "q_chunk": Q_CHUNK, "accuracy": [], "perf": [], "l1": []}
    try:
        # ---- accuracy, B=64
        qh = torch.randn(B_ACC, H, S, D).to(torch.bfloat16)
        kh = torch.randn(B_ACC, H, S, D).to(torch.bfloat16)
        vh = torch.randn(B_ACC, H, S, D).to(torch.bfloat16)
        bh = torch.randn(1, H, S, S).to(torch.bfloat16)
        ref = host_ref(qh, kh, vh, bh)
        qd, kd, vd, bd = (to_dev(t, dev) for t in (qh, kh, vh, bh))
        for name, fn in ARMS:
            row = {"arm": name}
            try:
                o = fn(qd, kd, vd, bd)
                ttnn.synchronize_device(dev)
                row.update(err(ttnn.to_torch(o), ref))
                ttnn.deallocate(o)
            except Exception as exc:                                   # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            res["accuracy"].append(row)
            print(json.dumps(row), flush=True)
        for t in (qd, kd, vd, bd):
            ttnn.deallocate(t)

        # ---- L1 arithmetic, stated before the perf arm so a refusal is expected not surprising
        qs = torch.zeros(B_PERF, H, S, D).to(torch.bfloat16)
        bs = torch.zeros(1, H, S, S).to(torch.bfloat16)
        qd, kd, vd, bd = (to_dev(t, dev) for t in (qs, qs, qs, bs))
        od = ttnn.allocate_tensor_on_device(ttnn.Shape([B_PERF, H, S, D]), ttnn.bfloat16,
                                            ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        for im, st, kc in ((None, None, 256), (F32, None, 256),
                           (F32, F32, 256), (F32, F32, 128), (F32, F32, 64)):
            res["l1"].append({"im": str(im), "stats": str(st), "k_chunk": kc,
                              "bytes": l1_bytes(qd, kd, vd, bd, od, im, st, kc)})
        print(json.dumps(res["l1"]), flush=True)

        # ---- perf, B=512
        for name, fn in ARMS:
            row = {"arm": name}
            try:
                for _ in range(WARM):
                    o = fn(qd, kd, vd, bd); ttnn.synchronize_device(dev); ttnn.deallocate(o)
                ms = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    o = fn(qd, kd, vd, bd)
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3)
                    ttnn.deallocate(o)
                row["ms"] = round(statistics.median(ms), 4)
                row["ms_min"] = round(min(ms), 4)
            except Exception as exc:                                   # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            res["perf"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        pass
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s1_fp32_cb_sdpa.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
