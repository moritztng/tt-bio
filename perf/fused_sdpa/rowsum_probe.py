#!/usr/bin/env python3
"""Is the fused SDPA's error one-signed? Set v to all ones and read the row sums.

Why this probe exists. Two models have now measured the same paradox: the fused triangle-attention
kernel is CLOSER to an fp64 evaluation of its own bf16 operands than the materialised fp32-softmax
path is (rel_rms 0.00470 vs 0.00883 at 512 aa, tt_bio/triatt_sdpa.py:70-95), and yet the fold it
produces is FURTHER from the crystal structure. rel_rms and PCC are both blind to the sign of an
error, so a biased error and a zero-mean error of the same magnitude score identically per call and
diverge over the 2176 chained calls of a fold.

With v = 1 the exact output of attention is the row sum of the softmax weights, which is exactly
1.0 for every element. So the deviation IS the normalisation error, the mean of it over rows is the
coherent component rel_rms cannot see, and no fp64 gold or reference implementation is needed.

The mechanism has a precedent in this codebase: ttnn.softmax returns rows summing to 0.9769, and
tt_bio/rf3/remap.py:198-205 records that this uniform deficit does not cancel in "probs @ v" and
is the whole of AttentionPairBias's 13.43x error on RF3's pairformer. accurate_softmax exists to
fix exactly that. This asks the same question of the fused kernel.

Arms, all on the same operands, at RF3's trunk triangle-attention geometry (head_dim 32, 4 heads,
batch = sequence):

    default   _tri_att_sdpa           fused at the op default (HiFi2, approx on, no fp32_dest_acc)
    hifi      _tri_att_sdpa_hifi      fused at (HiFi4, approx off, fp32_dest_acc)
    fp32      _fp32_softmax_attention the shipped materialised path
    hifi_1chunk                       fused at HiFi4 with k_chunk = padded S, one online-softmax
                                      chunk and therefore no running-max rescale

Usage: rowsum_probe.py [--sizes 320,512] [--out perf/fused_sdpa/rowsum_probe.json]
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HEADS, HEAD_DIM = 4, 32   # rf3/remap.py PAIRFORMER_DIMS = (32, 4, 24, 16)


def stats(t):
    import torch
    d = (t.float() - 1.0).flatten()
    n = d.numel()
    mean = d.mean().item()
    std = d.std().item()
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "max_abs": d.abs().max().item(),
        "frac_above_one": (d > 0).float().mean().item(),
        # how many standard errors the mean sits from zero: the coherence test
        "mean_over_sem": (abs(mean) / (std / math.sqrt(n))) if std > 0 else float("inf"),
        "rel_rms": d.pow(2).mean().sqrt().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="320,512")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("rowsum_probe.json"))
    a = ap.parse_args()

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

    res = {"heads": HEADS, "head_dim": HEAD_DIM, "arch": str(dev.arch()), "sizes": {}}
    for S in [int(x) for x in a.sizes.split(",")]:
        torch.manual_seed(0)
        scale_inv = HEAD_DIM ** -0.5           # TriangleAttention passes self.scale ** -1
        # batch = sequence: one S x S attention per row of the pair tensor, which is the shape the
        # trunk actually issues.
        B = S
        mk = lambda *sh: ttnn.from_torch(
            torch.randn(*sh, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = mk(B, HEADS, S, HEAD_DIM)
        k = mk(B, HEADS, S, HEAD_DIM)
        bias = mk(1, HEADS, S, S)
        v = ttnn.from_torch(
            torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)

        arms = {}

        def record(name, out):
            if out is None:
                arms[name] = {"declined": True}
                print(f"S={S} {name:12s} DECLINED", flush=True)
                return
            arms[name] = stats(ttnn.to_torch(out))
            ttnn.deallocate(out)
            print(f"S={S} {name:12s} mean={arms[name]['mean']:+.6f} "
                  f"std={arms[name]['std']:.6f} mean/sem={arms[name]['mean_over_sem']:.1f} "
                  f"max|d|={arms[name]['max_abs']:.6f} "
                  f"frac>1={arms[name]['frac_above_one']:.4f}", flush=True)

        record("default", T._tri_att_sdpa(q, k, v, bias, scale_inv))
        record("hifi", T._tri_att_sdpa_hifi(q, k, v, bias, scale_inv))
        record("fp32", T._fp32_softmax_attention(
            q, k, v, bias, scale_inv=scale_inv, compute_kernel_config=ckc,
            out_dtype=ttnn.bfloat16, bias_scale_inv=1.0))
        # one k chunk: k_chunk = padded S, so the online softmax never rescales its accumulator
        padded = T._padded_sdpa_len(S)
        try:
            record("hifi_1chunk", TS.sdpa(q, k, v, bias, scale_inv, padded, padded,
                                          ckc_default=T._TRIATT_FUSED_HIFI_CKC))
        except Exception as exc:  # noqa: BLE001 -- an L1 refusal is a result, not a crash
            arms["hifi_1chunk"] = {"error": str(exc)[:200]}
            print(f"S={S} hifi_1chunk raised: {str(exc)[:120]}", flush=True)

        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        res["sizes"][str(S)] = {"padded": padded, "arms": arms,
                                "hifi_stats": dict(T.TRIATT_FUSED_HIFI_STATS)}

    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
