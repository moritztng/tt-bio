#!/usr/bin/env python3
"""Compare the config against the plain call on the REAL operands, inside a real fold.

The op-level test builds clean 608x608 tensors and finds every applied class bit-exact. The fold
does not: its DiT operands are logical 580 padded to 608, so the contraction runs over 28 pad
columns whose content comes from whatever produced the tensor, not from ttnn. This runs both paths
on the same live operands at every applied call and reports where they disagree. It returns the
plain result, so the fold itself is the off arm and stays unperturbed.
"""
import argparse, json, sys, tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force-pcm", type=int, default=0,
                    help="override per_core_M on every applied class, to separate the "
                         "per_core_M value from the operand padding")
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P

    acc = defaultdict(lambda: dict(calls=0, mismatched=0, max_abs=0.0, n_diff=0, n_elem=0))

    def spy(x, y, compute_kernel_config=None):
        sa, sb = tuple(int(d) for d in x.shape), tuple(int(d) for d in y.shape)
        cfg = None
        if len(sa) == 4 and len(sb) == 4 and x.dtype == y.dtype:
            cfg = T._batched_matmul_config(sa[0] * sa[1], -(-sa[2] // 32), -(-sa[3] // 32),
                                          -(-sb[3] // 32), 4 if x.dtype == ttnn.float32 else 2)
        if cfg is not None and a.force_pcm and (-(-sa[2] // 32)) % a.force_pcm == 0:
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN,
                in0_block_w=cfg.in0_block_w, out_subblock_h=1,
                out_subblock_w=cfg.out_subblock_w, per_core_M=a.force_pcm,
                per_core_N=cfg.per_core_N)
        plain = ttnn.matmul(x, y, compute_kernel_config=compute_kernel_config)
        if cfg is None:
            return plain
        k = f"{sa}x{sb} pcm={cfg.per_core_M} bw={cfg.in0_block_w}"
        e = acc[k]
        e["calls"] += 1
        if e["calls"] <= 8:  # to_torch on every call would dominate the fold
            got = ttnn.matmul(x, y, compute_kernel_config=compute_kernel_config, program_config=cfg)
            d = (ttnn.to_torch(plain).float() - ttnn.to_torch(got).float())
            ttnn.deallocate(got)
            nz = int((d != 0).sum())
            e["mismatched"] += nz > 0
            e["n_diff"] += nz
            e["n_elem"] += int(d.numel())
            e["max_abs"] = max(e["max_abs"], d.abs().max().item())
        return plain

    T.batched_matmul = spy
    P.batched_matmul = spy
    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="ktiles-infold-"))
    one_fold, _meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml", Path(B.FIXTURES) / "prot300.a3m")
    one_fold()
    rows = []
    for k, e in sorted(acc.items(), key=lambda kv: -kv[1]["calls"]):
        checked = min(e["calls"], 8)
        print(f"  {k}\n     calls={e['calls']:5d} checked={checked} "
              f"calls_with_a_diff={e['mismatched']} differing elems={e['n_diff']}/{e['n_elem']} "
              f"max|d|={e['max_abs']:.3e}")
        rows.append(dict(key=k, checked=checked, **e))
    a.out.write_text(json.dumps({"model": a.model, "rows": rows}, indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
