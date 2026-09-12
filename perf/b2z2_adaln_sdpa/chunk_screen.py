#!/usr/bin/env python3
"""Both sites are parallelism-starved, and this is the screen that says so.

`sites_census.py` killed the two levers the brief and this row's own prediction proposed:
the token AdaLN's live set in L1 is 0.809x (SLOWER), and dropping the SDPA bias from bf16 to
bfp8_b -- 0.688x of the op's bytes -- buys only 1.070x, so the op is not byte-bound either.

What both sites share is a work-unit count that does not cover the grid.

  token SDPA   q=512, q_chunk=256 -> 2 chunks x 16 heads = 32 work units on 72 cores
  AdaLN norm   [1, 512, 768]      -> 512 rows = 16 tile-rows = 16 cores of 72

q_chunk is a partition of independent query rows, so narrowing it is BIT-EXACT (k_chunk is the
online-softmax reduction order and is not). This sweeps it, checks torch.equal against the
shipped config on every rung, and separately tests the norm's mechanism by giving it 4x the rows:
an op that is core-starved at 512 rows does 4x the work for much less than 4x the time.
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "perf" / "b2z2_step_fusion", ROOT / "scripts" / "gpu_vs_tt",
          ROOT / "perf" / "b2x_difflayer"):
    sys.path.insert(0, str(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    SP.OUT_PATH = a.out
    SP.OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
                     "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                     "loadavg": open("/proc/loadavg").read().split()[:3]}
    out = SP.OUT
    a.out.parent.mkdir(parents=True, exist_ok=True)

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    gs = dev.compute_with_storage_grid_size()
    out["env"].update(arch=str(dev.arch()), cores=f"{gs.x}x{gs.y}", n_cores=gs.x * gs.y,
                      compute_grid_main=list(T.COMPUTE_GRID_MAIN))
    fence = SP.make_fence(ttnn, dev)

    def run():
        return g["obj"](*g["args"], **g["kwargs"])

    # grab one token-DiT SDPA's operands, cloned so the step's own deallocate cannot reach them
    grabbed = {}
    orig_sdpa = ttnn.transformer.scaled_dot_product_attention

    def rec(q, k, v, **kw):
        if "op" not in grabbed and int(q.shape[2]) == int(k.shape[2]) and int(q.shape[1]) == 16:
            grabbed["op"] = (ttnn.clone(q), ttnn.clone(k), ttnn.clone(v),
                             ttnn.clone(kw["attn_mask"]), dict(kw))
        return orig_sdpa(q, k, v, **kw)

    orig_ln = ttnn.layer_norm
    ln_in = {}

    def rec_ln(x, **kw):
        if "a" not in ln_in and list(x.shape) == [1, 512, 768] and kw.get("weight") is None:
            ln_in["a"] = ttnn.clone(x)
            ln_in["kw"] = dict(kw)
        return orig_ln(x, **kw)

    run(); ttnn.synchronize_device(dev)
    ttnn.transformer.scaled_dot_product_attention = rec
    ttnn.layer_norm = rec_ln
    try:
        run(); ttnn.synchronize_device(dev)
    finally:
        ttnn.transformer.scaled_dot_product_attention = orig_sdpa
        ttnn.layer_norm = orig_ln

    q, k, v, bias, kw = grabbed["op"]
    S = int(q.shape[2])
    shipped_q = T._capped_sdpa_chunk_size(S)
    shipped_k = T._capped_sdpa_chunk_size(int(k.shape[2]))
    out["sdpa_shape"] = {"q": list(q.shape), "bias": list(bias.shape),
                         "shipped_q_chunk": shipped_q, "shipped_k_chunk": shipped_k}
    print(f"  token SDPA q={list(q.shape)} bias={list(bias.shape)} shipped "
          f"q_chunk={shipped_q} k_chunk={shipped_k}", flush=True)

    def sdpa_at(qc, kc):
        return orig_sdpa(q, k, v, attn_mask=bias, is_causal=False,
                         scale=kw.get("scale"),
                         program_config=T._sdpa_program_config(qc, kc))

    ref = ttnn.to_torch(sdpa_at(shipped_q, shipped_k))
    rows = []
    for kc in (shipped_k,):
        for qc in (512, 256, 128, 64, 32):
            if S % qc:
                continue
            try:
                got = ttnn.to_torch(sdpa_at(qc, kc))
                t = SP.timed_reps(ttnn, dev, lambda: sdpa_at(qc, kc), (), {}, a.reps, fence, n_med=5)
            except Exception as exc:                       # noqa: BLE001
                rows.append({"q_chunk": qc, "k_chunk": kc, "refused": str(exc)[:160]})
                print(f"    SDPA q_chunk={qc:4d} k_chunk={kc:4d}  REFUSED", flush=True)
                continue
            units = (S // qc) * int(q.shape[1])
            rows.append({"q_chunk": qc, "k_chunk": kc, "work_units": units,
                         "us": round(t["ms_per_call"] * 1e3, 2), "ms_all": t["ms_all"],
                         "bit_exact": bool(torch.equal(ref, got)),
                         "max_abs": float((ref - got).abs().max())})
            print(f"    SDPA q_chunk={qc:4d} k_chunk={kc:4d} units={units:4d} "
                  f"{rows[-1]['us']:8.2f} us  bit_exact={rows[-1]['bit_exact']} "
                  f"max_abs={rows[-1]['max_abs']:.6g}", flush=True)
    out["sdpa_q_chunk_sweep"] = rows
    SP.dump()

    # ---- the norm's mechanism: 4x the rows on the same op ---------------------------------
    a0, lnkw = ln_in["a"], ln_in["kw"]
    big = ttnn.concat([a0, a0, a0, a0], dim=1)
    scale = {}
    for nm, x in (("rows512", a0), ("rows2048", big)):
        t = SP.timed_reps(ttnn, dev, lambda x=x: orig_ln(x, **lnkw), (), {}, a.reps, fence, n_med=5)
        scale[nm] = {"us": round(t["ms_per_call"] * 1e3, 2), "rows": int(x.shape[1]),
                     "tile_rows": int(x.shape[1]) // 32, "ms_all": t["ms_all"]}
        print(f"    LN {nm:9s} tile_rows={scale[nm]['tile_rows']:3d} {scale[nm]['us']:8.2f} us",
              flush=True)
    scale["work_ratio"] = 4.0
    scale["time_ratio"] = round(scale["rows2048"]["us"] / scale["rows512"]["us"], 4)
    out["ln_row_scaling"] = scale
    print(f"    LN 4x the rows costs {scale['time_ratio']:.3f}x the time "
          f"({'core-starved' if scale['time_ratio'] < 3.0 else 'not core-starved'})", flush=True)
    SP.dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
