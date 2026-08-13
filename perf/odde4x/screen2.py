#!/usr/bin/env python3
"""Screen 2, corrected: does K1 fire on the op's OWN default block config?

Screen 1 (`perf/odde4x/screen1.json`, same card) settled the parity half: at OpenDDE's (kt=12, nt=36)
and (kt=12, nt=12) pair shapes, `minimal_matmul(config=(8,8,8,2,2))` is `torch.equal` to
`config=None`, max_abs 0.0, ratio 1.001 / 1.0086. That is not a coincidence -- (8,8,8) with subblocks
(2,2) under `fp32_dest_acc_en` is exactly what `determine_default_block_sizes` returns in
tt-metal v0.68.0, and `padded_K_tiles = round_up(12, 8) = 16` is the default's own two-K-block fold.
`K_block = 12`, the only kind of entry the predecessor's sweep tried, is `torch.equal` FALSE at
max_abs 0.5 -- reproduced in screen 1 and consistent with its section 4.

Screen 1's S2 leg could not test K1 because `T._qkv_mm_config` refuses to BUILD that config:
`kt % blk[1]` is 12 % 8 = 4 and `nt % N` is 36 % 8 = 4. Those are tt-bio's guards, not the op's --
the factory pads K to the block and lets the last M/N block be partial. Relaxing them IS the change,
so this screen builds the config directly and asks the only question left.

GATES, written before the run:
  G1  `qkv_heads` returns non-None on (8,8,8,2,2) and its (q,k,v) are `torch.equal` to
      `nlp_create_qkv_heads(minimal_matmul(config=(8,8,8,2,2)))`. If the generic-op transcription
      cannot express two K blocks or a partial N block it will throw or mismatch, and the
      byte-identical K1 route is dead -- the entry would then be worth only the matmul ratio, which
      is 1.001x, i.e. nothing.
  G2  the fused op beats the two stock ops by >= 1.10x. Below that K1 does not pay for the plumbing.
  G3  same two gates for the K1b tail (`gate_proj` + `out_proj`) at (kt=12, nt=12).
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WARM, REPS = 2, 7
DEV = [None]


def _free(o):
    import ttnn
    if o is None:
        return
    for t in (o if isinstance(o, (list, tuple)) else (o,)):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def bench(fn):
    import ttnn
    for _ in range(WARM):
        _free(fn())
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(DEV[0])
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(DEV[0])
        ts.append((time.perf_counter() - t0) * 1e3)
        _free(o)
    return round(st.median(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import importlib.metadata as im
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_qkv as K1
    from tt_bio.tenstorrent import get_device
    assert Path(T.__file__).resolve().is_relative_to(ROOT)
    dev = DEV[0] = get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [gx, gy], "ttnn": im.version("ttnn"),
           "loadavg": open("/proc/loadavg").read().split()[:3], "rows": []}
    torch.manual_seed(0)
    S, CZ, NH = 512, 384, 12

    def cfg(M, K, N, sh, sw):
        return ttnn.MinimalMatmulConfig(
            M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))

    x = ttnn.from_torch(torch.randn(1, S * S, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    w_qkv = ttnn.from_torch(torch.randn(CZ, 3 * NH * 32), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
    w_g = ttnn.from_torch(torch.randn(CZ, NH * 32), layout=ttnn.TILE_LAYOUT, device=dev,
                          dtype=ttnn.bfloat16)
    w_o = ttnn.from_torch(torch.randn(NH * 32, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                          dtype=ttnn.bfloat16)
    saved = dict(T._MM_BLOCK)

    for entry in ((8, 8, 8, 2, 2), (12, 12, 1, 2, 2)):
        T._MM_BLOCK[(12, 36)] = entry
        T._MM_BLOCK[(12, 12)] = entry
        c = cfg(*entry)
        row = {"entry": list(entry),
               "qkv_mm_config_builds_via_tt_bio_guard": T._qkv_mm_config(x, w_qkv) is not None}

        # --- K1a: head-major qkv --------------------------------------------------------------
        def stock_qkv():
            s = ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w_qkv, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=c)
            o = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.unsqueeze(s, 1), num_heads=NH, num_kv_heads=NH, transpose_k_heads=False)
            _free(s)
            return o
        try:
            K1.STATS[:] = [0, 0]
            K1.REJECTS.clear()
            sref = stock_qkv()
            ref = [ttnn.to_torch(t) for t in sref]
            _free(sref)
            out = K1.qkv_heads(x, w_qkv, ckc, NH, 32, ttnn.bfloat16, c)
            row["k1a_served"] = out is not None
            row["k1a_rejects"] = {str(k): v for k, v in K1.REJECTS.items()}
            if out is not None:
                got = [ttnn.to_torch(t) for t in out]
                _free(out)
                row["k1a_torch_equal"] = [bool(torch.equal(g, r)) for g, r in zip(got, ref)]
                row["k1a_max_abs"] = [float((g - r).abs().max()) for g, r in zip(got, ref)]
                row["k1a_ref_absmax"] = float(max(r.abs().max() for r in ref))
                row["k1a_ms"] = bench(lambda: K1.qkv_heads(x, w_qkv, ckc, NH, 32, ttnn.bfloat16, c))
                row["k1a_stock_ms"] = bench(stock_qkv)
                row["k1a_ratio"] = round(row["k1a_stock_ms"] / row["k1a_ms"], 4)
        except Exception as e:
            row["k1a_error"] = f"{type(e).__name__}: {e}"[:250]

        # --- K1b: head-major gate + out -------------------------------------------------------
        try:
            K1.TAIL_STATS[:] = [0, 0]
            K1.TAIL_REJECTS.clear()
            gate = K1.gate_proj(x, w_g, w_o, ckc, NH, 32, ttnn.bfloat16, c)
            row["k1b_served"] = gate is not None
            row["k1b_rejects"] = {str(k): v for k, v in K1.TAIL_REJECTS.items()}
            if gate is not None:
                stock_g = ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w_g, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=c)
                sg = ttnn.to_torch(ttnn.experimental.nlp_create_qkv_heads(
                    ttnn.unsqueeze(stock_g, 1), num_heads=NH, num_kv_heads=NH,
                    transpose_k_heads=False)[0]) if False else ttnn.to_torch(stock_g)
                gg = ttnn.to_torch(gate)
                _free([stock_g, gate])
                # head-major [1,H,S,32] vs [1,S,H*32]: compare after the same reshape
                gg2 = gg.reshape(1, NH, S * S, 32).permute(0, 2, 1, 3).reshape(1, S * S, NH * 32)
                row["k1b_torch_equal"] = bool(torch.equal(gg2, sg.reshape(1, S * S, NH * 32)))
                row["k1b_max_abs"] = float((gg2 - sg.reshape(1, S * S, NH * 32)).abs().max())
                row["k1b_ref_absmax"] = float(sg.abs().max())
                row["k1b_gate_ms"] = bench(
                    lambda: K1.gate_proj(x, w_g, w_o, ckc, NH, 32, ttnn.bfloat16, c))
                row["k1b_stock_gate_ms"] = bench(lambda: ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w_g, compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16, config=c))
        except Exception as e:
            row["k1b_error"] = f"{type(e).__name__}: {e}"[:250]

        res["rows"].append(row)
        print(" S2", json.dumps(row), flush=True)

    T._MM_BLOCK.clear()
    T._MM_BLOCK.update(saved)

    # --- S4: channel_move_back at width 384 vs three 128-wide passes ---------------------------
    cm = {}
    try:
        cc = ttnn.from_torch(torch.randn(1, CZ, S, S), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        mc = ttnn.DRAM_MEMORY_CONFIG
        o = T._channel_move_back(cc, mc)
        ref = ttnn.to_torch(o)
        _free(o)
        cm["wide_384_ms"] = bench(lambda: T._channel_move_back(cc, mc))

        def sliced():
            parts = [T._channel_move_back(cc[:, w:w + 128], mc) for w in range(0, CZ, 128)]
            out = ttnn.concat(parts, dim=-1)
            _free(parts)
            return out
        o2 = sliced()
        got = ttnn.to_torch(o2)
        _free(o2)
        cm["sliced_3x128_ms"] = bench(sliced)
        cm["delta_ms"] = round(cm["wide_384_ms"] - cm["sliced_3x128_ms"], 4)
        cm["shapes_match"] = list(got.shape) == list(ref.shape)
        cm["torch_equal"] = bool(cm["shapes_match"] and torch.equal(got, ref))
        _free(cc)
    except Exception as e:
        cm["error"] = f"{type(e).__name__}: {e}"[:250]
    res["s4_channel_move_back"] = cm
    print(" S4", json.dumps(cm), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
