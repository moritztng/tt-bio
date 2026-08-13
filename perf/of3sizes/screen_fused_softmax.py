#!/usr/bin/env python3
"""Screen: fuse `multiply` + `add` + `softmax` into `ttnn.scale_mask_softmax_in_place`.

`_fp32_softmax_attention` (tenstorrent.py:745) spends 12 of its ~20.25 whole-tensor passes on three
separate ops -- scale, add the pair bias, softmax -- over a tensor with `n_heads * S**3` elements.
`ttnn.scale_mask_softmax_in_place(x, scale, mask)` is documented as exactly that sequence in one op,
in place. This screens the ACTUAL change at the production shape, and settles the two kill gates the
plan pre-registered:

  gate 1  is it bit-exact?  `torch.equal` against the three-op chain, nothing looser.
  gate 2  does the mask broadcast?  the scores are [S, n_heads, S, S] and the bias is
          [1, n_heads, S, S], a leading-dim broadcast. If it is refused, do NOT materialise a
          broadcast bias -- that adds a whole N**3 pass and inverts the lever. The ladder is
          scale-only fusion (mask=None) with the `add` kept separate.

At S=1024 the three-op chain cannot run at all (16 GiB fp32 score tensor x2 live), so the fused arm
is also a capacity probe: it holds ONE fp32 copy, so it may fit where the chain does not.
"""
import json, os, sys, time
from pathlib import Path

import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.tenstorrent import get_device, batched_matmul  # noqa: E402

SIZES = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("256", "512", "768")))
OUT = Path(__file__).resolve().parent / (
    "screen_fused_softmax_qb1c%s" % os.environ.get("TT_VISIBLE_DEVICES", "0")
    + ("" if SIZES == (256, 512, 768) else "_" + "_".join(str(x) for x in SIZES)) + ".json")
H, DH = 4, 32
SCALE_INV = DH ** -0.5


def med(fn, reps=5, warm=2):
    dev = get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for i in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i < reps - 1:
            ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2], [x * 1e3 for x in ts], o


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": "qb1", "card": os.environ.get("TT_VISIBLE_DEVICES", "0"),
           "ttnn": "0.67.4", "n_heads": H, "head_dim": DH, "rows": []}
    OUT.write_text(json.dumps(res, indent=1))

    for S in SIZES:
        row = {"S": S, "score_fp32_GiB": H * S ** 3 * 4 / 2 ** 30}
        torch.manual_seed(S)
        q, k, v = (ttnn.from_torch(torch.randn(S, H, S, DH, dtype=torch.bfloat16) * 0.1,
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(3))
        bias = ttnn.from_torch(torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.1,
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kt = ttnn.permute(k, (0, 1, 3, 2))
        sc0 = batched_matmul(q, kt, compute_kernel_config=ckc)   # bf16 scores, computed once
        ttnn.deallocate(kt)
        bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config())

        def stock():
            sc = ttnn.typecast(sc0, ttnn.float32, memory_config=sc0.memory_config())
            sc = ttnn.multiply(sc, SCALE_INV)
            sc = ttnn.add(sc, bias_f)
            a = ttnn.softmax(sc, dim=-1)
            ttnn.deallocate(sc)
            return a

        def fused():
            sc = ttnn.typecast(sc0, ttnn.float32, memory_config=sc0.memory_config())
            return ttnn.scale_mask_softmax_in_place(sc, SCALE_INV, bias_f)

        def fused_scale_only():
            sc = ttnn.typecast(sc0, ttnn.float32, memory_config=sc0.memory_config())
            sc = ttnn.add(sc, bias_f)
            return ttnn.scale_mask_softmax_in_place(sc, 1.0, None)

        outs = {}
        for name, fn in (("fused", fused), ("fused_scale_only", fused_scale_only), ("stock", stock)):
            try:
                t, ts, o = med(fn)
                row[f"{name}_ms"] = t * 1e3
                row[f"{name}_all_ms"] = ts
                outs[name] = ttnn.to_torch(o)
                ttnn.deallocate(o)
            except Exception as e:                                          # noqa: BLE001
                row[f"{name}_ms"] = f"REFUSED: {type(e).__name__}: {str(e)[:260]}"
        if "stock" in outs:
            for name in ("fused", "fused_scale_only"):
                if name in outs:
                    row[f"{name}_torch_equal"] = bool(torch.equal(outs["stock"], outs[name]))
                    row[f"{name}_max_abs"] = float(
                        (outs["stock"].float() - outs[name].float()).abs().max())
                    if isinstance(row.get(f"{name}_ms"), float):
                        row[f"{name}_speedup"] = row["stock_ms"] / row[f"{name}_ms"]
        res["rows"].append(row)
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps(row, indent=1), flush=True)
        for t_ in (q, k, v, bias, sc0, bias_f):
            try: ttnn.deallocate(t_)
            except Exception: pass
    print("wrote", OUT)
    return 0


sys.exit(main())
