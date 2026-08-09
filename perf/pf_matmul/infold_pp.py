#!/usr/bin/env python3
"""Count the pair-track projection calls per fold, and compare the tuned program config
against the production `core_grid=` call on the REAL operands inside a real fold.

D3 found a per_core_M correctness bug that an op-level torch.equal sweep passed and only a
live-operand comparison caught, so the acceptance check for the bit-exact tier is this one.
The fold itself keeps running the production path, so it stays unperturbed.
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
    ap.add_argument("--bw", type=int, default=1)
    ap.add_argument("--max-check", type=int, default=6)
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    T._PAIR_PROJ_BW = a.bw

    acc = defaultdict(lambda: dict(calls=0, checked=0, mismatched=0, max_abs=0.0,
                                   n_diff=0, n_elem=0, cfg=None))

    def spy(x, w, ckc, dtype):
        cfg = T._pair_proj_config(x, w)
        prod = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
                           compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        sa = tuple(int(d) for d in x.shape)
        sw = tuple(int(d) for d in w.shape)
        k = f"{sa}@{sw}"
        e = acc[k]
        e["calls"] += 1
        if cfg is None:
            e["cfg"] = "declined"
            return prod
        e["cfg"] = (f"pcm={cfg.per_core_M} bw={cfg.in0_block_w} obh={cfg.out_block_h} "
                    f"obw={cfg.out_block_w} sh={cfg.out_subblock_h} sw={cfg.out_subblock_w}")
        if e["checked"] < a.max_check:
            e["checked"] += 1
            got = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
                              compute_kernel_config=ckc, program_config=cfg)
            d = ttnn.to_torch(prod).float() - ttnn.to_torch(got).float()
            ttnn.deallocate(got)
            nz = int((d != 0).sum())
            e["mismatched"] += int(nz > 0)
            e["n_diff"] += nz
            e["n_elem"] += int(d.numel())
            e["max_abs"] = max(e["max_abs"], float(d.abs().max()))
        return prod

    T._pair_proj_linear = spy

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="pp-infold-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml",
        Path(B.FIXTURES) / "prot300.a3m")
    one_fold()

    rows = [dict(shape=k, **{kk: vv for kk, vv in e.items()})
            for k, e in sorted(acc.items(), key=lambda kv: -kv[1]["calls"])]
    out = dict(model=a.model, bw=a.bw, n_aa=298, hardware=meta["hardware"],
               applied_calls=sum(r["calls"] for r in rows if r["cfg"] != "declined"),
               declined_calls=sum(r["calls"] for r in rows if r["cfg"] == "declined"),
               any_mismatch=any(r["mismatched"] for r in rows), rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
