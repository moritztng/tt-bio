#!/usr/bin/env python3
"""Where does the SDPA row-sum deficit come from? A decomposition, with a floor.

`rowsum_probe.py` (committed 8fe4c99f) established the fact this task exists for: with v = all
ones every output element of SDPA is the softmax row sum and must be 1.0, and every arm comes out
SHORT, one-signed, by -0.0005 to -0.0104. That measurement stands. This probe exists because the
first one cannot say WHY, and three things about it have to change before the number can be
attributed to a mechanism.

1. **It has no floor.** Deviation is measured against the exact 1.0, but the arms return bf16, and
   ulp(1.0) in bf16 is 2**-8 = 0.00390625. Every number in the original table is between 0.13 and
   1.92 output ulps. Until an arm that does the arithmetic in fp64 on the host and then stores the
   answer the way the device stores it is on the table, none of those numbers has a scale. The
   `floor_*` arms below are that scale.

2. **Its arms compute different functions.** `_tri_att_sdpa*` gets `scale_inv` as the bias scale,
   so the fused arms evaluate `softmax(s*(qk + bias))`; `_fp32_softmax_attention` is called with
   `bias_scale_inv=1.0` and evaluates `softmax(s*qk + bias)` (the bias-scale fork documented at
   tenstorrent.py:3830). Those are different score distributions, and how peaked a row is changes
   how hard it is to reduce, so the -0.00544 vs -0.00152 gap is confounded. Here the bias is zero
   by default: a softmax row sums to 1 whatever the scores are, so a zero bias costs the probe
   nothing and removes the fork. `--bias` puts a real one back as a check.

3. **Its n is 32x too big and its rows are not independent.** The row sum is one number per
   (batch, head, q) replicated across head_dim, so `mean_over_sem` on flattened elements inflates
   by sqrt(32) and calls perfectly correlated copies independent samples. Every arm here reduces
   head_dim first (and asserts the copies agree), then reports mean/sem over rows.

What the arms are for, in the order they answer the question:

    floor_exact        fp64 softmax, row sum in fp64, stored bf16.  The measurement floor.
    floor_bf16_probs   ... with the probs stored bf16 first.        Prob-storage cost.
    floor_bf16_acc     ... and summed in bf16 sequentially.         What a bf16 accumulator costs.

    softmax_only       ttnn.softmax on fp32 scores, probs read back fp32, summed fp64. No matmul
                       anywhere. This is the ONLY arm that measures normalisation on its own, and
                       it is the one that reproduces or refutes the 0.9769 that
                       tenstorrent.py:1491 records for this kernel.
    softmax_only_acc   `_accurate_softmax` on the same scores. The existing fix, never scored on
                       this instrument.

    fp32 / fp32_acc    the shipped materialised path, with `accurate_softmax` off and on. The off
                       arm is what RF3 ships; the on arm has never been on this table at all.
    default / hifi     the fused kernel at the op default and at HiFi4 + fp32_dest_acc.
    kchunk_<n>         the fused kernel at HiFi4 with k_chunk forced to n. THE discriminator: the
                       online softmax rescales its accumulators once per chunk boundary
                       (compute_common.hpp:1867-1885), so if the deficit tracks the chunk count the
                       rescale is the injector, and if it is flat in chunk count it is not.

Read the kernel before reading the numbers. With v = ones the fused kernel's output is
`mm2_out * recip(cur_sum)` (compute_common.hpp:1956-1959), where `mm2_out` is the exp block reduced
by a MATMUL against v and `cur_sum` is the same exp block reduced by `reduce_c`. Two independent
reductions of one block, and the row sum is their ratio. So "the softmax under-normalises" is not
what this probe measures on the fused path -- it measures how much the matmul reduction and the
reduce reduction disagree. The materialised path is different: it normalises by its own sum, so
there the row sum really is the softmax kernel's own error plus storage.

Usage:
    rowsum_probe2.py [--sizes 320,512] [--batch 8] [--temps 1.0] [--bias]
                     [--out perf/fused_sdpa/rowsum_probe2.json]

`--batch` is the batch dim; the row-sum statistic does not depend on it (the fused kernel's work
split is per (batch, head) and the arithmetic per row is identical), so 8 rows of pair tensor give
16384 rows of statistic for 1/64th of the original probe's DRAM. `--batch 0` uses batch = S, which
is what the trunk actually issues and what the first probe used, for one confirmation run.
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HEADS, HEAD_DIM = 4, 32   # rf3/remap.py PAIRFORMER_DIMS = (32, 4, 24, 16);
                          # --heads/--head_dim override for the other models
BF16_ULP_AT_ONE = 2.0 ** -8


def row_stats(rows, note=""):
    """mean/sem over ROWS, not over elements. `rows` is a 1-D fp64 tensor of row sums."""
    import torch
    d = (rows.double() - 1.0).flatten()
    n = d.numel()
    mean = d.mean().item()
    std = d.std().item() if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n_rows": n,
        "mean": mean,
        "mean_ulps": mean / BF16_ULP_AT_ONE,
        "std": std,
        "sem": sem,
        # |mean| in standard errors. Honest denominator this time: rows, not elements.
        "mean_over_sem": (abs(mean) / sem) if sem > 0 else float("inf"),
        "max_abs": d.abs().max().item(),
        "rms": d.pow(2).mean().sqrt().item(),
        # three buckets, because the original probe's `frac>1` used d>0 and so counted every row
        # that landed exactly on 1.0 as "not above" -- with a bf16 output that is most of them.
        "frac_below": (d < 0).double().mean().item(),
        "frac_exact": (d == 0).double().mean().item(),
        "frac_above": (d > 0).double().mean().item(),
        "note": note,
    }


def rows_from_out(out_t):
    """Device output [B,H,S,D] -> one row sum per (B,H,S), asserting the D copies agree.

    Every column of the output is the same row sum, so a spread across D means the op is not doing
    what this probe assumes and the number must not be reported as a row sum.
    """
    import torch
    t = out_t.double()
    lo, hi = t.min(dim=-1).values, t.max(dim=-1).values
    spread = (hi - lo).max().item()
    return t.mean(dim=-1).flatten(), spread


def main():
    global HEADS, HEAD_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="320,512")
    ap.add_argument("--heads", type=int, default=HEADS,
                    help="tri-att head count: 4 rf3/boltz-2, 8 protenix, 12 opendde")
    ap.add_argument("--head_dim", type=int, default=HEAD_DIM)
    ap.add_argument("--batch", type=int, default=8, help="0 means batch = S")
    ap.add_argument("--temps", default="1.0", help="score multipliers; changes row peakedness")
    ap.add_argument("--bias", action="store_true", help="use a real bias instead of zeros")
    ap.add_argument("--kchunks", default="64,128,256,0", help="0 means one chunk (k_chunk = S)")
    ap.add_argument("--qk", default="64:0,128:0",
                    help="extra explicit q_chunk:k_chunk arms; 0 means the padded length. "
                         "The one-k-chunk arm needs one of these -- --kchunks widens q to S "
                         "as well, and q=k=S is the config that refused L1 by 1.2 percent. "
                         "q chunking is pure parallelism and cannot change a row arithmetic, "
                         "so a narrow q with k_chunk=S buys one k chunk for 8x LESS L1.")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("rowsum_probe2.json"))
    a = ap.parse_args()

    HEADS, HEAD_DIM = a.heads, a.head_dim

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolves to {T.__file__}, not this checkout -- set PYTHONPATH"

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)

    res = {"heads": HEADS, "head_dim": HEAD_DIM, "arch": str(dev.arch()),
           "bf16_ulp_at_one": BF16_ULP_AT_ONE, "bias": bool(a.bias), "sizes": {}}

    for S in [int(x) for x in a.sizes.split(",")]:
        B = S if a.batch == 0 else a.batch
        for temp in [float(x) for x in a.temps.split(",")]:
            key = f"S{S}_B{B}_t{temp:g}"
            arms = {}
            torch.manual_seed(0)
            scale_inv = HEAD_DIM ** -0.5
            # temp rides on q, so the score distribution changes and nothing else does
            q_h = (torch.randn(B, HEADS, S, HEAD_DIM) * temp).to(torch.bfloat16)
            k_h = torch.randn(B, HEADS, S, HEAD_DIM).to(torch.bfloat16)
            b_h = (torch.randn(1, HEADS, S, S) if a.bias
                   else torch.zeros(1, HEADS, S, S)).to(torch.bfloat16)

            # ---- floor: the same question answered in fp64 on the host, then stored the way the
            # device stores it. The device's scores differ from these (bf16 matmul, fp32 dest), but
            # a softmax row sums to 1 for ANY scores, so the floor does not depend on matching them.
            sc64 = (q_h.double() @ k_h.double().transpose(-1, -2)) * scale_inv + b_h.double()
            p64 = torch.softmax(sc64, dim=-1)
            del sc64
            arms["floor_exact"] = row_stats(
                p64.sum(-1).to(torch.bfloat16), "fp64 probs, fp64 sum, bf16 store")
            p16 = p64.to(torch.bfloat16)
            arms["floor_bf16_probs"] = row_stats(
                p16.double().sum(-1).to(torch.bfloat16), "bf16 probs, fp64 sum, bf16 store")
            acc = torch.zeros(p16.shape[:-1], dtype=torch.bfloat16)
            for i in range(S):                       # sequential bf16 accumulation, worst case
                acc = acc + p16[..., i]
            arms["floor_bf16_acc"] = row_stats(acc.double(), "bf16 probs, bf16 sequential sum")
            del p64, p16, acc

            # ---- device
            mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, bias = mk(q_h), mk(k_h), mk(b_h)
            v = mk(torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16))

            def record(name, fn):
                try:
                    out = fn()
                except Exception as exc:                                    # noqa: BLE001
                    arms[name] = {"error": str(exc)[:240]}
                    print(f"{key} {name:20s} RAISED {str(exc)[:110]}", flush=True)
                    return
                if out is None:
                    arms[name] = {"declined": True}
                    print(f"{key} {name:20s} DECLINED", flush=True)
                    return
                rows, spread = rows_from_out(ttnn.to_torch(out))
                ttnn.deallocate(out)
                arms[name] = row_stats(rows, f"head_dim spread {spread:.3e}")
                arms[name]["head_dim_spread"] = spread
                s = arms[name]
                print(f"{key} {name:20s} mean={s['mean']:+.6f} ({s['mean_ulps']:+.2f} ulp) "
                      f"sem={s['sem']:.2e} m/sem={s['mean_over_sem']:.1f} "
                      f"below/exact/above={s['frac_below']:.3f}/{s['frac_exact']:.3f}/"
                      f"{s['frac_above']:.3f}", flush=True)

            # softmax alone, no matmul: the only arm that isolates normalisation. Materialising
            # [B,H,S,S] fp32 is why --batch matters; at B=512,S=512 this is 2 GB and will not fit.
            def softmax_only(accurate):
                kt = ttnn.permute(k, (0, 1, 3, 2))
                sc = T.batched_matmul(q, kt, compute_kernel_config=ckc)
                ttnn.deallocate(kt)
                scf = ttnn.typecast(sc, ttnn.float32, memory_config=sc.memory_config())
                ttnn.deallocate(sc)
                scf = ttnn.multiply(scf, scale_inv)
                p = (T._accurate_softmax(scf) if accurate else ttnn.softmax(scf, dim=-1))
                ttnn.deallocate(scf)
                rows = ttnn.to_torch(p).double().sum(-1)     # fp32 probs, fp64 sum: no storage cost
                ttnn.deallocate(p)
                return rows

            for nm, acc_flag in (("softmax_only", False), ("softmax_only_acc", True)):
                try:
                    rows = softmax_only(acc_flag)
                    arms[nm] = row_stats(rows.flatten(), "probs read fp32, summed fp64, no matmul")
                    s = arms[nm]
                    print(f"{key} {nm:20s} mean={s['mean']:+.6f} ({s['mean_ulps']:+.2f} ulp) "
                          f"sem={s['sem']:.2e} below/exact/above={s['frac_below']:.3f}/"
                          f"{s['frac_exact']:.3f}/{s['frac_above']:.3f}", flush=True)
                except Exception as exc:                                    # noqa: BLE001
                    arms[nm] = {"error": str(exc)[:240]}
                    print(f"{key} {nm:20s} RAISED {str(exc)[:110]}", flush=True)

            # the shipped materialised path, both ways round, and at both output dtypes
            for nm, acc_flag, od in (("fp32", False, ttnn.bfloat16),
                                     ("fp32_acc", True, ttnn.bfloat16),
                                     ("fp32_out32", False, ttnn.float32),
                                     ("fp32_acc_out32", True, ttnn.float32)):
                record(nm, lambda af=acc_flag, o=od: T._fp32_softmax_attention(
                    q, k, v, bias, scale_inv=scale_inv, compute_kernel_config=ckc,
                    out_dtype=o, bias_scale_inv=1.0, accurate_softmax=af))

            record("default", lambda: T._tri_att_sdpa(q, k, v, bias, scale_inv))
            record("hifi", lambda: T._tri_att_sdpa_hifi(q, k, v, bias, scale_inv))

            padded = T._padded_sdpa_len(S)
            configs = []
            for kc in [int(x) for x in a.kchunks.split(",") if x.strip()]:
                configs.append((padded, padded if kc == 0 else kc))
            for pair in a.qk.split(","):
                if not pair.strip():
                    continue
                qs, ks = (int(x) for x in pair.split(":"))
                configs.append((padded if qs == 0 else qs, padded if ks == 0 else ks))
            for qq, kk in configs:
                if padded % kk or padded % qq:
                    continue
                nchunk = padded // kk
                # mask CB tiles the persistent kernel will ask for, so an L1 refusal is priced
                # before it happens rather than read as "one k chunk does not fit"
                mask_tiles = nchunk * (qq // 32) * (kk // 32)
                record(f"q{qq}_k{kk}_n{nchunk}_m{mask_tiles}",
                       lambda qq=qq, kk=kk: TS.sdpa(q, k, v, bias, scale_inv, qq, kk,
                                                    ckc_default=T._TRIATT_FUSED_HIFI_CKC))

            for t in (q, k, v, bias):
                ttnn.deallocate(t)
            res["sizes"][key] = {"S": S, "B": B, "temp": temp, "padded": padded, "arms": arms,
                                 "hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS),
                                 "triatt_stats": list(TS.STATS)}
            a.out.write_text(json.dumps(res, indent=1) + "\n")

    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
