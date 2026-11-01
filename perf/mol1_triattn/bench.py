#!/usr/bin/env python3
"""Triangle attention at Mol-1's widths against the widths the fused path was built for.

One `TriangleAttention` per (c_z, heads) on synthetic weights, start and ending, under two lever sets:
`mol1_now` (protenix.NORMAL_LEVERS, what a non-256 checkpoint gets from `mode_levers` today) and
`v2_normal` (Protenix-v2's Blackhole normal set), each with and without `_MM_DEFAULT_WIDE` (`*_wide`). Per cell: which fused calls declined and why
(`triatt_ln.REJECTS`), warm op time with the AICLK sampled during it, and parity against a float64
torch reference on a row subset (the reference at 1024 x 448 would be ~2 TFLOP of float64).

Usage: bench.py --sizes 256,512,1024 --widths 256x8,384x12,320x10,448x14 --out cells.jsonl
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))

import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.protenix as P
from tt_bio import triatt_ln as TL
import clocksample

assert Path(T.__file__).resolve().is_relative_to(ROOT)
HD = 32
V2_BH_NORMAL = (P.NORMAL_LEVERS | P.T2_TAIL | P.WS_CHAIN | P.QSPLIT_NORMAL | P.BRES_NORMAL | P.TRIATT_LN
                | P.WS_RESID_NORMAL | P.T1_PSIG | P.TRIATT_K1 | P.BH_V2_NORMAL)
# arm -> (lever set, `_MM_DEFAULT_WIDE`); the *_now arms are the tree before the wide default, `served` is the
# set Mol-1's normal mode takes on this chip today (`protenix.mode_levers`)
ARMS = {"mol1_now": (P.NORMAL_LEVERS, False), "v2_normal": (V2_BH_NORMAL, False),
        "mol1_wide": (P.NORMAL_LEVERS, True), "v2_wide": (V2_BH_NORMAL, True),
        "served": (P.NORMAL_LEVERS | P.WIDE_NORMAL | (P.MOL1_WH if T.is_wormhole() else P.MOL1_BH), True)}


def flops(S, c, H):
    """Matmul FLOPs of one call: bias + q/k/v/g + output projections (S^2) and the attention itself (S^3)."""
    return 2 * S * S * c * H * (1 + 5 * HD) + 4 * S ** 3 * H * HD


def weights(c, H, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) / s[-1] ** 0.5  # noqa: E731
    return {"layer_norm.weight": 1 + 0.1 * torch.randn(c, generator=g), "layer_norm.bias": 0.1 * torch.randn(c, generator=g),
            "linear.weight": r(H, c), "linear_q.weight": r(H * HD, c), "linear_k.weight": r(H * HD, c),
            "linear_v.weight": r(H * HD, c), "linear_g.weight": r(H * HD, c), "linear_o.weight": r(c, H * HD)}


def reference_rows(x, sd, H, ending, rows):
    """float64 triangle attention (Protenix convention: bias b[h,j,k] from the normed pair, shared over rows)."""
    x = x.double()
    if ending:
        x = x.transpose(0, 1)
    w = {k: v.double() for k, v in sd.items()}
    xn = torch.nn.functional.layer_norm(x, x.shape[-1:], w["layer_norm.weight"], w["layer_norm.bias"], 1e-5)
    b = (xn @ w["linear.weight"].t()).permute(2, 0, 1)                     # [H, S, S]
    out = []
    for i in rows:
        xi = xn[i]                                                          # [S, c]
        q, k, v, gt = ((xi @ w[f"linear_{n}.weight"].t()).view(-1, H, HD).transpose(0, 1) for n in "qkvg")
        a = torch.softmax(q @ k.transpose(-1, -2) / HD ** 0.5 + b, -1)
        o = (a @ v) * torch.sigmoid(gt)                                     # [H, S, d]
        out.append(o.transpose(0, 1).reshape(xi.shape[0], H * HD) @ w["linear_o.weight"].t())
    return torch.stack(out)                                                 # [rows, S, c]


def metrics(ref, got):
    ref, got = ref.flatten(), got.double().flatten()
    pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
    rel = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return {"pcc": round(pcc, 7), "rel_rms": round(rel, 6), "max_abs": round((got - ref).abs().max().item(), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--widths", default="256x8,384x12,320x10,448x14")
    ap.add_argument("--arms", default="mol1_now,v2_normal")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ref-max", type=int, default=1024, help="skip the float64 reference above this size")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    sha = __import__("subprocess").run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    for S in map(int, a.sizes.split(",")):
        for wh in a.widths.split(","):
            c, H = map(int, wh.split("x"))
            sd = weights(c, H, 7)
            torch.manual_seed(S + c)
            x = torch.randn(S, S, c).bfloat16()
            rows = sorted(set(torch.linspace(0, S - 1, 16).long().tolist()))
            for ending in (False, True):
                ref = reference_rows(x.float(), sd, H, ending, rows) if S <= a.ref_max else None
                for arm in a.arms.split(","):
                    T._TRIATT_WHOLE_REFUSED.clear(); T._TRIATT_BLOCK_REFUSED.clear()
                    rec = {"S": S, "c_z": c, "heads": H, "ending": ending, "arm": arm, "sha": sha,
                           "arch": str(dev.arch()), "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING, "gflop": round(flops(S, c, H) / 1e9, 2)}
                    try:
                        T._MM_DEFAULT_WIDE = ARMS[arm][1]
                        with T.levers(ARMS[arm][0]):
                            mod = T.TriangleAttention(HD, H, ending, sd, ckc, softmax_site="pairformer")
                            xt = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                 device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            TL.REJECTS.clear(); TL.STATS[:] = [0, 0, 0]
                            o = mod(xt)
                            got = ttnn.to_torch(o).float()[0]
                            ttnn.deallocate(o)
                            rec["triatt_ln_rejects"] = {f"{k[0]}": v for k, v in TL.REJECTS.items()}
                            rec["triatt_ln_stats"] = list(TL.STATS)
                            # a DRAM refusal moves the call onto the row-blocked path for the process
                            rec["row_blocked"] = {str(k): v for m in (T._TRIATT_WHOLE_REFUSED, T._TRIATT_BLOCK_REFUSED)
                                                  for k, v in m.items()}
                            gr = got.transpose(0, 1)[rows] if ending else got[rows]
                            if ref is not None:
                                rec.update(metrics(ref, gr))
                            ttnn.deallocate(mod(xt))
                            ttnn.synchronize_device(dev)
                            ts = []
                            with clocksample.during(period=0.5) as clk:
                                for _ in range(a.reps):
                                    t0 = time.perf_counter()
                                    o = mod(xt)
                                    ttnn.synchronize_device(dev)
                                    ts.append(time.perf_counter() - t0)
                                    ttnn.deallocate(o)
                            ttnn.deallocate(xt)
                            ts.sort()
                            rec["ms_min"] = round(ts[0] * 1e3, 3)
                            rec["ms_med"] = round(ts[len(ts) // 2] * 1e3, 3)
                            rec["aiclk"] = clk.summary()
                            rec["tflops"] = round(rec["gflop"] / rec["ms_min"], 2)
                            del mod
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                    print(json.dumps(rec), flush=True)
                    with a.out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
    T.cleanup()


if __name__ == "__main__":
    main()
