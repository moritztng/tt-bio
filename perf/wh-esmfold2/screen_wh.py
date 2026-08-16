#!/usr/bin/env python3
"""THE SCREEN for wh-perf-esmfold2: measure the ACTUAL change, not a proxy.

`screen.py` (the recovered Blackhole instrument) prices the pair transition op-by-op and
hand-builds candidate rewrites with `dtype=ttnn.bfloat16` pinned. That was the right screen when
the question was "is a split fc1 worth building". It is the wrong screen here: the question is
"does flipping the small-grid flag on a 72-core Wormhole part, under --fast, pay" -- and under
--fast the two arms do not even carry the same dtype through fc1 (see `dtypes` in the output).
So this screen calls the real `SwiGLUFFN.__call__` and the real `TriangleMultiplication.__call__`
with the real gates flipped, which is the only thing that predicts the fold.

Levers, from state/wh-perf-esmfold2.md:
  A  esmc.SPLIT_SWIGLU_SMALL_GRID   -- reactivate split fc1 + 32-row block + L1 fc1 on a small grid
  C1 TRIANGLE_MULT_L1_MAX_SEQ_FAST  -- 288 -> 320, one tile, so a 298 aa target stays L1-resident
  C2 TRIANGLE_MULT_L1_CHUNK_BUDGET  -- the 64*320*320 width budget, fitted on 13x10, scaled by cores

Every arm is checked with `torch.equal`, not PCC: both levers are claimed bit-exact.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def bench(fn, n=7, warm=3):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return st.median(ts) * 1e3, (max(ts) - min(ts)) * 1e3


def ckc(fid="HiFi4", fp32acc=True):
    cls = (ttnn.types.WormholeComputeKernelConfig if T.is_wormhole()
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
               fp32_dest_acc_en=fp32acc, packer_l1_acc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--levers", default="A,C1,C2")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    # BEFORE any module is constructed: fast mode picks the stored weight dtype at build time.
    T.set_fast_mode(a.fast)
    levers = set(a.levers.split(","))
    L, CZ, FF = a.L, 256, 1024

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    import tt_bio.esmc as EC
    from tt_bio.tenstorrent import WeightScope

    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cores": g.x * g.y, "L": L, "fast": a.fast,
         "small_grid": T._IS_SMALL_GRID, "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
         "gates": {
             "pair_row_tile": T.pair_row_tile(L),
             "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
             "trimul_l1_max_seq": T._trimul_l1_max_seq(),
             "trimul_chunk_size": T._trimul_chunk_size(L, CZ),
             "split_swiglu_min_seq": EC.SPLIT_SWIGLU_MIN_SEQ,
             "pair_ffn_row_block": EC._PAIR_FFN_ROW_BLOCK,
             "pair_ffn_row_block_seq": list(EC.PAIR_FFN_ROW_BLOCK_SEQ),
         },
         "A": {}, "C1": {}, "C2": {}, "dtypes": {}, "exact": {}}

    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # ---------------------------------------------------------------- lever A
    if "A" in levers:
        torch.manual_seed(0)
        w12 = torch.randn(2 * FF, CZ) * 0.02
        w3 = torch.randn(CZ, FF) * 0.02
        sd = WeightScope({"0.weight": torch.randn(CZ), "0.bias": torch.randn(CZ),
                          "1.weight": w12, "3.weight": w3})
        ffn = EC.SwiGLUFFN(sd, CK, fuse_swiglu=True)     # the trunk's construction, esmfold2.py:146
        R["A"]["split_swiglu_built"] = bool(ffn.split_swiglu)
        R["A"]["fc1_weight_dtype"] = str(ffn.fc1_a_weight.dtype if ffn.split_swiglu
                                         else ffn.fc1_weight.dtype)
        z = f(torch.randn(1, L, L, CZ))

        # Record the dtype the fc1 output actually carries in each arm. The base arm goes through
        # `Module._lin`, which resolves `_dtype(ttnn.bfloat16)` -> bf16 even under --fast; the
        # split L1 arm passes `dt = _dtype()` -> bfloat8_b. That is a PRECISION difference, not a
        # memory-config one, and it is why the two arms cannot be assumed torch.equal here.
        seen = {}
        orig_pp = EC._pair_proj_linear
        orig_lin = EC.Module._lin

        def rec_pp(x, w, ckc_, dtype, **kw):
            out = orig_pp(x, w, ckc_, dtype, **kw)
            seen.setdefault("split_fc1_out", str(out.dtype))
            seen.setdefault("split_fc1_req", str(dtype))
            seen.setdefault("split_fc1_l1", str(out.memory_config().buffer_type))
            return out

        def rec_lin(self, x, w, bias=None, dtype=None, **kw):
            out = orig_lin(self, x, w, bias=bias, dtype=dtype, **kw)
            if int(w.shape[-1]) == 2 * FF:
                seen.setdefault("base_fc1_out", str(out.dtype))
            return out

        EC._pair_proj_linear = rec_pp
        EC.Module._lin = rec_lin
        try:
            for arm in (False, True):
                EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
                prev = EC.set_split_swiglu_small_grid(arm)
                seen.clear()
                try:
                    ms, spread = bench(lambda: ffn(z))
                    out = ffn(z)
                    R["A"]["arm_on" if arm else "arm_off"] = {
                        "ms": round(ms, 4), "spread_ms": round(spread, 4),
                        "l1_fc1_served": EC.L1_FC1_STATS[0], "l1_fc1_declined": EC.L1_FC1_STATS[1],
                        "out_dtype": str(out.dtype), "seen": dict(seen),
                    }
                    R["A"]["ref_on" if arm else "ref_off"] = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                except Exception as e:                                            # noqa: BLE001
                    R["A"]["arm_on" if arm else "arm_off"] = f"ERR {type(e).__name__}: {str(e)[:400]}"
                finally:
                    EC.set_split_swiglu_small_grid(prev)
        finally:
            EC._pair_proj_linear = orig_pp
            EC.Module._lin = orig_lin

        ro, rn = R["A"].pop("ref_off", None), R["A"].pop("ref_on", None)
        if ro is not None and rn is not None:
            R["exact"]["A_on_vs_off"] = bool(torch.equal(ro, rn))
            d = (ro.float() - rn.float()).abs()
            R["exact"]["A_ndiff"] = int((d > 0).sum())
            R["exact"]["A_max_abs"] = float(d.max())
            R["exact"]["A_peak"] = float(ro.float().abs().max())
            R["exact"]["A_max_abs_over_peak"] = (float(d.max()) / float(ro.float().abs().max())
                                                 if float(ro.float().abs().max()) else None)
        R["dtypes"] = R["A"].get("arm_off", {}).get("seen", {}) if isinstance(
            R["A"].get("arm_off"), dict) else {}
        ttnn.deallocate(z)

    # ---------------------------------------------------------------- levers C1 / C2
    if "C1" in levers or "C2" in levers:
        torch.manual_seed(1)
        H = CZ
        sd = WeightScope({
            "norm_in.weight": torch.randn(CZ), "norm_in.bias": torch.randn(CZ),
            "norm_out.weight": torch.randn(H), "norm_out.bias": torch.randn(H),
            "g_in.weight": torch.randn(2 * H, CZ) * 0.02,
            "p_in.weight": torch.randn(2 * H, CZ) * 0.02,
            "g_out.weight": torch.randn(CZ, CZ) * 0.02,
            "p_out.weight": torch.randn(CZ, H) * 0.02,
        })
        tm = T.TriangleMultiplication(False, sd, CK, gated_move=True)   # esmfold2.py:140
        zz = f(torch.randn(1, L, L, CZ))
        base_thr = T.TRIANGLE_MULT_L1_MAX_SEQ_FAST
        base_bud = T.TRIANGLE_MULT_L1_CHUNK_BUDGET
        R["C1"]["shipped_threshold"] = base_thr
        R["C1"]["shipped_budget"] = base_bud
        refs = {}
        try:
            arms = []
            if "C1" in levers:
                arms += [("thr288_shipped", base_thr, base_bud),
                         ("thr%d_L1" % ((L + 31) // 32 * 32), (L + 31) // 32 * 32, base_bud)]
            if "C2" in levers:
                # widths 32/64/128 at the raised threshold: budget must admit the width
                for w in (64, 128):
                    need = w * L * L
                    arms.append(("thr_L1_width%d" % w, (L + 31) // 32 * 32,
                                 int(need * (T.COMPUTE_GRID_X_13 * 10) / (g.x * g.y)) + 1))
            for name, thr, bud in arms:
                T.TRIANGLE_MULT_L1_MAX_SEQ_FAST = thr
                T.TRIANGLE_MULT_L1_CHUNK_BUDGET = bud
                width = T._trimul_chunk_size(L, CZ)
                memcfg = "L1" if L <= T._trimul_l1_max_seq() else "DRAM"
                try:
                    ms, spread = bench(lambda: tm(zz), n=5, warm=2)
                    o = tm(zz)
                    refs[name] = ttnn.to_torch(o)
                    ttnn.deallocate(o)
                    R["C1" if name.startswith("thr") and "width" not in name else "C2"][name] = {
                        "ms": round(ms, 4), "spread_ms": round(spread, 4),
                        "threshold": thr, "chunk_width": width, "pair_memcfg": memcfg}
                except Exception as e:                                            # noqa: BLE001
                    R["C1" if "width" not in name else "C2"][name] = \
                        f"ERR {type(e).__name__}: {str(e)[:400]}"
        finally:
            T.TRIANGLE_MULT_L1_MAX_SEQ_FAST = base_thr
            T.TRIANGLE_MULT_L1_CHUNK_BUDGET = base_bud
        keys = list(refs)
        if keys:
            r0 = refs[keys[0]]
            for k in keys[1:]:
                R["exact"]["C_%s_vs_%s" % (k, keys[0])] = bool(torch.equal(r0, refs[k]))
        ttnn.deallocate(zz)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(R, indent=1, default=str))


if __name__ == "__main__":
    main()
