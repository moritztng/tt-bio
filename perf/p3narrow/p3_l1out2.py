#!/usr/bin/env python3
"""Deliverable 3, candidate 1, priced end to end: does an L1 output remove DRAM traffic?

The pair-track output projection writes [1,298,320,256] = 48.82 MB to DRAM, and its consumer is a
residual `add` that reads it back and writes 48.82 MB of updated pair track to DRAM. If the
projection can put its output in L1 the DRAM write AND the add's operand read both disappear; the
add still writes its 48.82 MB, so the question is whether the pair `projection + add` gets cheaper,
not whether the projection alone does. Measuring the projection alone would price a write that the
next op pays anyway.

Four legs, all producing the same DRAM result, all with a `torch.equal` check against the
production leg:

  dram_cg     production: linear(core_grid) -> DRAM, add -> DRAM
  dram_tuned  production today: linear(_pair_proj config, bw=8) -> DRAM, add -> DRAM
  l1_cg       linear(core_grid) -> L1, add -> DRAM
  l1_tuned    linear(1D config) -> L1, add -> DRAM

Also: the largest L1 output this card accepts for this op, and how much L1 the leg holds live.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import (  # noqa: E402
    CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device, _l1_bank_bytes,
)

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TOK, NPAD, C_Z = 298, 320, 256


def timed(fn, dev, warm=3, pipe=3, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def cfg(m_tiles, n_tiles, bw, obh, out_l1):
    gx, gy = COMPUTE_GRID_MAIN
    nc = gx * gy
    if m_tiles < nc or 8 % bw:
        return None
    pcm = -(-(-(-m_tiles // nc)) // obh) * obh
    if -(-m_tiles // pcm) > nc:
        return None
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, n_tiles), 0, -1) if n_tiles % w == 0)
    need = 2 * bw * (obh + n_tiles) * 2048 + obh * n_tiles * (2048 + 4096) + 128 * 1024
    if out_l1:
        need += pcm * n_tiles * 2048
    if need > _l1_bank_bytes():
        return None
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=obh, out_block_w=n_tiles, per_core_M=pcm,
        per_core_N=n_tiles, fuse_batch=True, fused_activation=None, mcast_in0=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = COMPUTE_GRID_MAIN
    banks = dev.compute_with_storage_grid_size().x * dev.compute_with_storage_grid_size().y
    x = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    z = ttnn.from_torch(torch.randn(1, TOK, TOK, C_Z), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(C_Z, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    m_tiles = TOK * (NPAD // 32)
    out_bytes = m_tiles * 32 * C_Z * 2
    res = {"card": "qb1 card 1", "banks": banks, "l1_bank_bytes": _l1_bank_bytes(),
           "projection_output_bytes": out_bytes,
           "projection_output_kB_per_bank": round(out_bytes / banks / 1024, 1),
           "grid": [gx, gy]}
    mv0 = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
    res["l1_free_before"] = int(mv0.largest_contiguous_bytes_free_per_bank)

    legs = {}
    for lbl, omem, pc in (
            ("dram_cg", DRAM, None),
            ("dram_tuned_bw8_obh5", DRAM, cfg(m_tiles, 8, 8, 5, False)),
            ("l1_cg", L1, None),
            ("l1_tuned_bw8_obh5", L1, cfg(m_tiles, 8, 8, 5, True)),
            ("l1_tuned_bw8_obh2", L1, cfg(m_tiles, 8, 8, 2, True)),
            ("l1_tuned_bw1_obh5", L1, cfg(m_tiles, 8, 1, 5, True))):
        kw = dict(memory_config=omem, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        if pc is None and lbl.endswith("cg"):
            proj = lambda: ttnn.linear(x, w, core_grid=CORE_GRID_MAIN, **kw)
        elif pc is None:
            legs[lbl] = {"err": "config refused by the L1 budget"}
            continue
        else:
            proj = (lambda c: lambda: ttnn.linear(x, w, program_config=c, **kw))(pc)

        def pair():
            p = proj()
            o = ttnn.add(z, p, memory_config=DRAM)
            ttnn.deallocate(p)
            ttnn.deallocate(o)

        row = {}
        try:
            row["proj_only_us"] = round(timed(lambda: ttnn.deallocate(proj()), dev) * 1e6, 2)
            row["proj_write_GBs"] = round(out_bytes / (row["proj_only_us"] / 1e6) / 1e9, 1)
        except Exception as e:                                            # noqa: BLE001
            row["proj_err"] = str(e)[:160]
        try:
            row["proj_plus_add_us"] = round(timed(pair, dev) * 1e6, 2)
        except Exception as e:                                            # noqa: BLE001
            row["pair_err"] = str(e)[:160]
        # L1 held live by the projection output, measured while it exists
        try:
            p = proj()
            ttnn.synchronize_device(dev)
            mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
            row["l1_free_while_output_live"] = int(mv.largest_contiguous_bytes_free_per_bank)
            row["torch_equal_vs_dram_cg"] = None
            row["_out"] = ttnn.to_torch(ttnn.add(z, p, memory_config=DRAM))
            ttnn.deallocate(p)
        except Exception as e:                                            # noqa: BLE001
            row["live_err"] = str(e)[:160]
        legs[lbl] = row
        print(f"  {lbl:22s} proj {row.get('proj_only_us','-'):>9} us   proj+add "
              f"{row.get('proj_plus_add_us','-'):>9} us   "
              f"L1 free while live {row.get('l1_free_while_output_live','-')}", flush=True)

    ref = legs.get("dram_cg", {}).pop("_out", None)
    for lbl, row in legs.items():
        o = row.pop("_out", None)
        if ref is not None and o is not None:
            row["torch_equal_vs_dram_cg"] = bool(torch.equal(ref, o))
            row["max_abs_vs_dram_cg"] = float(
                (ref.to(torch.float64) - o.to(torch.float64)).abs().max())
    if ref is not None:
        legs["dram_cg"]["torch_equal_vs_dram_cg"] = True
        legs["dram_cg"]["max_abs_vs_dram_cg"] = 0.0
    res["legs"] = legs
    print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in legs.items()}, indent=1),
          flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
