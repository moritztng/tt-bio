#!/usr/bin/env python3
"""P1 p2-attention Phase-2: the two experiments that need a REAL Pairformer block.

  L1a  -- how much L1 is free per core at the SDPA call INSIDE a real block, not at an empty
          allocator. W9's bias-once design needs 204.8 kB per core to coexist with whatever the
          block has live at that moment.
  L2b  -- block-level parity of the pair tensor, SDPA chunk 64 (production) vs chunk 320, at the
          fold's true shape [1, 298, 320, 256].
  HD   -- the head-count halving control the first probe lost when the L1-mask arm raised.
  L1L1 -- what `nlp_create_qkv_heads` costs L1-to-L1 at a chunk that fits, as a bound for Phase 3.

EXPERIMENT ONLY. The chunk override is a probe-side monkeypatch of `_tri_att_sdpa_program_config`;
nothing under `tt_bio/` is modified.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "ledger_298"))

import tt_bio.tenstorrent as TT                                       # noqa: E402
from tt_bio.tenstorrent import get_device                             # noqa: E402
from pf_block_ops import build                                        # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
M, N, CZ, NH, HD = 298, 320, 256, 8, 32
RES = {}
DEV = None
L1_SNAPS = []


def save(path):
    json.dump(RES, open(path, "w"), indent=1)


CORES = 130                                    # compute_with_storage_grid_size 13x10 on this card
LADDER_MB = [4, 8, 12, 16, 20, 24, 26.6, 32, 40, 48, 64, 80, 96, 112, 128, 144, 160, 176, 184, 188, 190]


def alloc_l1(mb):
    """An interleaved L1 tensor of `mb` MB. Interleaved spreads over all 130 banks, so mb/130 is
    the per-core footprint -- the quantity W9's resident bias slice competes for."""
    rows = max(32, int(round(mb * 1e6 / 2 / 4096 / 32)) * 32)
    return ttnn.zeros((rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                      device=DEV, memory_config=L1), rows * 4096 * 2


def l1_headroom():
    """The largest interleaved-L1 buffer that allocates RIGHT NOW, and what it is per core.

    This ttnn's MeshDevice exposes no allocator-statistics accessor (checked: no `alloc`/`mem`
    attribute at all), so headroom is measured by allocating it rather than by asking.
    """
    best, best_b, err = 0.0, 0, None
    for mb in LADDER_MB:
        try:
            t, nb = alloc_l1(mb)
            ttnn.deallocate(t)
            best, best_b = mb, nb
        except Exception as e:                                         # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:140]
            break
    return {"largest_l1_alloc_MB": best, "largest_l1_alloc_B": best_b,
            "per_core_B": int(best_b / CORES), "per_core_kB": round(best_b / CORES / 1024, 1),
            "stopped_by": err}


def l1_stats():
    return l1_headroom()


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "perf/p2_attention/block_probe_c2.json"))
    a = ap.parse_args()
    DEV = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["l1_stats_empty"] = l1_stats()
    print("empty allocator:", json.dumps(RES["l1_stats_empty"]), flush=True)

    layer, c_z = build("protenix-v2", ckc)
    print(f"c_z={c_z}", flush=True)
    torch.manual_seed(0)
    s_h = torch.randn(1, M, 384)   # the single track carries the same 298 tokens as the pair track
    # The fold's pair tensor is LOGICALLY [1, 298, 298, c_z]; TILE_LAYOUT pads only the last two
    # dims, so it lands on device as [1, 298, 320, c_z] -- B2's finding, and the shape the block
    # actually runs. Building it as a logical 320 on the column axis is the square-harness defect.
    z_h = torch.randn(1, M, M, c_z)

    def up():
        return (ttnn.from_torch(s_h, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16),
                ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16))

    # ---- L1a: snapshot the allocator at every SDPA entry inside a real block -------------------
    real_sdpa = ttnn.transformer.scaled_dot_product_attention

    def spy(*args, **kw):
        st_ = l1_stats()
        cfg = kw.get("program_config")
        st_["q_chunk"] = getattr(cfg, "q_chunk_size", None)
        st_["q_shape"] = list(args[0].padded_shape) if args else None
        mask = kw.get("attn_mask")
        st_["mask_buf"] = str(mask.memory_config().buffer_type) if mask is not None else None
        st_["mask_shape"] = list(mask.padded_shape) if mask is not None else None
        L1_SNAPS.append(st_)
        return real_sdpa(*args, **kw)

    ttnn.transformer.scaled_dot_product_attention = spy
    sx, zx = up()
    sx, zx = layer(sx, zx)
    ttnn.synchronize_device(DEV)
    ttnn.transformer.scaled_dot_product_attention = real_sdpa
    RES["l1_at_sdpa"] = L1_SNAPS
    print("L1 at SDPA entry:", json.dumps(L1_SNAPS[:2], indent=1), flush=True)
    ttnn.deallocate(sx); ttnn.deallocate(zx)

    # ---- L2b: chunk 64 vs chunk 320, one real block, the fold's true shape ---------------------
    def run_once():
        s0, z0 = up()
        s1, z1 = layer(s0, z0)
        ttnn.synchronize_device(DEV)
        out = (ttnn.to_torch(s1).float(), ttnn.to_torch(z1).float())
        for t in (s1, z1):
            try:
                ttnn.deallocate(t)
            except Exception:                                          # noqa: BLE001
                pass
        return out

    def block_wall():
        s0, z0 = up()
        for _ in range(2):
            s0, z0 = layer(s0, z0)
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(3):
            s0, z0 = layer(s0, z0)
        ttnn.synchronize_device(DEV)
        return (time.perf_counter() - t0) / 3

    s_ref, z_ref = run_once()
    RES["block_wall_ms_chunk64"] = round(block_wall() * 1e3, 3)
    print(f"block wall chunk64 = {RES['block_wall_ms_chunk64']} ms", flush=True)

    saved_cfg = TT._tri_att_sdpa_program_config
    C320 = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                  q_chunk_size=320, k_chunk_size=320)
    TT._tri_att_sdpa_program_config = lambda q_len, k_len: C320
    calls = {"n": 0}
    real2 = ttnn.transformer.scaled_dot_product_attention

    def count(*args, **kw):
        cfg = kw.get("program_config")
        if getattr(cfg, "q_chunk_size", None) == 320:
            calls["n"] += 1
        return real2(*args, **kw)

    ttnn.transformer.scaled_dot_product_attention = count
    s_b, z_b = run_once()
    RES["block_wall_ms_chunk320"] = round(block_wall() * 1e3, 3)
    ttnn.transformer.scaled_dot_product_attention = real2
    TT._tri_att_sdpa_program_config = saved_cfg
    RES["chunk320_sdpa_calls_in_block"] = calls["n"]
    print(f"block wall chunk320 = {RES['block_wall_ms_chunk320']} ms  "
          f"(chunk-320 SDPA calls seen: {calls['n']})", flush=True)

    par = {}
    for lbl, ref, got in (("z", z_ref, z_b), ("s", s_ref, s_b)):
        d = (got - ref)
        par[lbl] = {
            "rms_ref": float(ref.pow(2).mean().sqrt()),
            "rmsd": float(d.pow(2).mean().sqrt()),
            "rel_rmsd": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "max_abs": float(d.abs().max()),
            "max_abs_ref": float(ref.abs().max()),
            "torch_equal": bool(torch.equal(ref, got)),
            "pcc": float(torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1]),
        }
        print(f"  parity {lbl}: {json.dumps(par[lbl])}", flush=True)
    RES["parity_one_block"] = par
    save(a.out)

    # ---- HD: the head-count control -------------------------------------------------------------
    def timed(fn, warm=2, pipe=3, reps=5):
        for _ in range(warm):
            fn()
        ttnn.synchronize_device(DEV)
        o = []
        for _ in range(reps):
            ttnn.synchronize_device(DEV)
            t0 = time.perf_counter()
            for _ in range(pipe):
                fn()
            ttnn.synchronize_device(DEV)
            o.append((time.perf_counter() - t0) / pipe)
        return st.median(o)

    def T(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                               dtype=ttnn.bfloat16, memory_config=mc)

    hd = {}
    for h in (2, 4, 8):
        q, k, v = (T((M, h, N, HD)) for _ in range(3))
        b = T((1, h, N, N))
        c64 = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                     q_chunk_size=64, k_chunk_size=64)
        with_b = timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=b, is_causal=False, scale=HD ** -0.5, program_config=c64)))
        no_b = timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, scale=HD ** -0.5, program_config=c64)))
        hd[h] = {"with_bias_us": round(with_b * 1e6, 1), "nobias_us": round(no_b * 1e6, 1),
                 "bias_leg_us": round((with_b - no_b) * 1e6, 1),
                 "bias_MB": round(M * h * N * N * 2 / 1e6, 1),
                 "bias_leg_GBs": round(M * h * N * N * 2 / (with_b - no_b) / 1e9, 1)}
        print(f"  heads={h}: {json.dumps(hd[h])}", flush=True)
        for t in (q, k, v, b):
            ttnn.deallocate(t)
    RES["head_scaling"] = hd
    save(a.out)

    # ---- L1L1: the head split L1-to-L1 at a chunk that fits ------------------------------------
    ll = {}
    for rows in (16, 32, 64):
        try:
            qkv = T((rows, 1, N, 3 * NH * HD), L1)
            s1 = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=L1)])
            qkv_d = T((rows, 1, N, 3 * NH * HD), DRAM)
            s2 = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
                qkv_d, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)])
            by = rows * N * 3 * NH * HD * 2
            ll[rows] = {"l1_us": round(s1 * 1e6, 1), "dram_us": round(s2 * 1e6, 1),
                        "l1_GBs": round(2 * by / s1 / 1e9, 1), "dram_GBs": round(2 * by / s2 / 1e9, 1),
                        "speedup": round(s2 / s1, 2),
                        "extrapolated_full_l1_us": round(s1 * 1e6 * M / rows, 1)}
            print(f"  split rows={rows}: {json.dumps(ll[rows])}", flush=True)
            ttnn.deallocate(qkv); ttnn.deallocate(qkv_d)
        except Exception as e:                                         # noqa: BLE001
            ll[rows] = {"error": f"{type(e).__name__}: {e}"[:200]}
            print(f"  split rows={rows}: {ll[rows]['error']}", flush=True)
    RES["split_l1_vs_dram"] = ll
    save(a.out)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
