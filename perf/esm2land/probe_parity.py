#!/usr/bin/env python3
"""Which kernel leg does the row-blocked pair FFN actually take, and is it bit-exact at each L?

The claim being audited: `_PAIR_L1_ROWS=32` is `torch.equal` against the shipped pair transition.
It was verified at L=512 only. `_pair_proj_linear(l1_out=True)` picks one of FOUR legs at runtime
(L1 tuned config / minimal_matmul / DRAM tuned config / untuned core_grid) and which one it picks
depends on an L1 allocation that scales with L. So the leg -- and therefore the parity -- may be a
property of L=512 rather than of the change. This probe records the leg per call and the parity
per arm, at three L.

Reference arm is what main SHIPS today (unsplit fc1 -> chunk -> silu -> multiply), not arm A.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T

LEGS = []          # one entry per _pair_proj_linear call: which leg returned
_TAG = [""]


def ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def instrument():
    """Wrap `_pair_proj_linear` so every call records which of its four legs returned."""
    orig_cfg = T._pair_proj_config
    orig_mm = T._pair_proj_minimal_matmul
    orig_lin = ttnn.linear

    def probed(x, w, ckc_, dtype, l1_out=False):
        leg = None
        if l1_out and T._PAIR_PROJ_L1_OUT:
            key = (tuple(x.padded_shape), tuple(w.shape), str(dtype))
            if key not in T._L1_OUT_REFUSED:
                cfg = orig_cfg(x, w, bw_cap=T._PAIR_PROJ_L1_BW, out_l1=True)
                if cfg is not None:
                    try:
                        out = orig_lin(x, w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dtype,
                                       compute_kernel_config=ckc_, program_config=cfg)
                        LEGS.append((_TAG[0], "L1_tuned"))
                        return out
                    except Exception:
                        T._L1_OUT_REFUSED.add(key)
                        leg = "L1_refused->"
                else:
                    leg = "L1_cfg_None->"
            else:
                leg = "L1_cached_refusal->"
        mm = orig_mm(x, w, ckc_, dtype)
        if mm is not None:
            LEGS.append((_TAG[0], (leg or "") + "minimal_matmul"))
            return mm
        cfg = orig_cfg(x, w)
        if cfg is not None:
            LEGS.append((_TAG[0], (leg or "") + "DRAM_tuned"))
            return orig_lin(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
                            compute_kernel_config=ckc_, program_config=cfg)
        LEGS.append((_TAG[0], (leg or "") + "core_grid_untuned"))
        return orig_lin(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype,
                        compute_kernel_config=ckc_, core_grid=T.CORE_GRID_MAIN)

    T._pair_proj_linear = probed


def bench(fn, n=5, warm=2):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return round(st.median(ts) * 1e3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[298, 320, 512])
    ap.add_argument("--rows", type=int, nargs="+", default=[32])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="", help="comma-separated substrings; empty = all arms")
    ap.add_argument("--skip-ref", action="store_true",
                    help="do not build the reference (no parity, but reaches L where the "
                         "full-size fc1 output alone would not fit)")
    a = ap.parse_args()
    want = [x for x in a.arms.split(",") if x]
    CZ, FF = 256, 1024

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    instrument()
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    CK = ckc()
    SILU = [ttnn.UnaryOpType.SILU]
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cz": CZ, "d_ff": FF,
         "PAIR_PROJ_BW": T._PAIR_PROJ_BW, "PAIR_PROJ_L1_BW": T._PAIR_PROJ_L1_BW,
         "sizes": {}}

    f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    for L in a.L:
        torch.manual_seed(0)
        nw, nb = f(torch.randn(CZ)), f(torch.randn(CZ))
        w1_full = (torch.randn(2 * FF, CZ) * 0.02)          # [2*FF, CZ], as the checkpoint stores it
        w1 = f(w1_full.t())                                  # main's fc1_weight
        w1a = f(w1_full[:FF].t())                            # split halves
        w1b = f(w1_full[FF:].t())
        w2 = f((torch.randn(CZ, FF) * 0.02).t())
        z = f(torch.randn(1, L, L, CZ))

        def norm():
            return ttnn.layer_norm(z, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=CK)

        def lin(x, w, **kw):
            return ttnn.linear(x, w, compute_kernel_config=CK, dtype=ttnn.bfloat16,
                               core_grid=T.CORE_GRID_MAIN, **kw)

        def ref():                                           # exactly main's SwiGLUFFN._ffn
            xn = norm()
            h = lin(xn, w1)
            ttnn.deallocate(xn)
            x1, x2 = ttnn.chunk(h, 2, dim=-1)
            ttnn.deallocate(h)
            gt = ttnn.multiply(ttnn.silu(x1), x2)
            ttnn.deallocate(x1); ttnn.deallocate(x2)
            out = lin(gt, w2)
            ttnn.deallocate(gt)
            return out

        def arm_a():                                         # split fc1 + SiLU-in-multiply
            xn = norm()
            h1, h2 = lin(xn, w1a), lin(xn, w1b)
            ttnn.deallocate(xn)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            out = lin(gt, w2)
            ttnn.deallocate(gt)
            return out

        def arm_c(rows):                                     # the branch's row-blocked L1 arm
            def go():
                parts = ttnn.chunk(z, -(-L // rows), dim=1)
                outs = []
                for p in parts:
                    xn = ttnn.layer_norm(p, weight=nw, bias=nb, epsilon=1e-5,
                                         compute_kernel_config=CK)
                    h1 = T._pair_proj_linear(xn, w1a, CK, ttnn.bfloat16, l1_out=True)
                    h2 = T._pair_proj_linear(xn, w1b, CK, ttnn.bfloat16, l1_out=True)
                    ttnn.deallocate(xn)
                    gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU,
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(h1); ttnn.deallocate(h2)
                    outs.append(lin(gt, w2))
                    ttnn.deallocate(gt)
                out = ttnn.concat(outs, dim=1)
                for o in outs:
                    ttnn.deallocate(o)
                return out
            return go

        def arm_e(rows, l1_gated):
            """The recommended shape: row-block, plain `_lin` for fc1 (the engine helper is
            measurably inert here), and the ONE L1 term that is actually served -- the SwiGLU
            product that fc2 reads. `l1_gated=False` isolates whether that term is the win."""
            mc = ttnn.L1_MEMORY_CONFIG if l1_gated else ttnn.DRAM_MEMORY_CONFIG

            def go():
                parts = ttnn.chunk(z, -(-L // rows), dim=1)
                outs = []
                for p in parts:
                    xn = ttnn.layer_norm(p, weight=nw, bias=nb, epsilon=1e-5,
                                         compute_kernel_config=CK)
                    h1, h2 = lin(xn, w1a), lin(xn, w1b)
                    ttnn.deallocate(xn)
                    gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU, memory_config=mc)
                    ttnn.deallocate(h1); ttnn.deallocate(h2)
                    outs.append(lin(gt, w2))
                    ttnn.deallocate(gt)
                out = ttnn.concat(outs, dim=1)
                for o in outs:
                    ttnn.deallocate(o)
                return out
            return go

        # arm D: no L1, no row blocking -- just ask for the DRAM tuned/minimal_matmul leg at full size
        def arm_d():
            xn = norm()
            h1 = T._pair_proj_linear(xn, w1a, CK, ttnn.bfloat16, l1_out=False)
            h2 = T._pair_proj_linear(xn, w1b, CK, ttnn.bfloat16, l1_out=False)
            ttnn.deallocate(xn)
            gt = ttnn.multiply(h1, h2, input_tensor_a_activations=SILU)
            ttnn.deallocate(h1); ttnn.deallocate(h2)
            out = lin(gt, w2)
            ttnn.deallocate(gt)
            return out

        entry = {"arms": {}, "exact_vs_shipped": {}, "legs": {}, "l1_refused_cache": []}
        T._L1_OUT_REFUSED.clear()

        _TAG[0] = f"L{L}:ref"
        if a.skip_ref:
            ref_t = None
        else:
            ref_t = ttnn.to_torch(ref())
            entry["arms"]["ref_shipped"] = bench(ref)

        arms = [("A_split", arm_a), ("D_engine_full", arm_d)]
        arms += [(f"C_rows{r}", arm_c(r)) for r in a.rows]
        arms += [(f"E_lin_L1gated_rows{r}", arm_e(r, True)) for r in a.rows]
        arms += [(f"F_lin_DRAMgated_rows{r}", arm_e(r, False)) for r in a.rows]
        if want:
            arms = [(n, f) for n, f in arms if any(w in n for w in want)]
        for name, fn in arms:
            _TAG[0] = f"L{L}:{name}"
            n0 = len(LEGS)
            try:
                got = ttnn.to_torch(fn())
                entry["exact_vs_shipped"][name] = (
                    None if ref_t is None else bool(torch.equal(ref_t, got))
                )
                entry["arms"][name] = bench(fn)
            except Exception as e:
                entry["exact_vs_shipped"][name] = f"RAISED: {type(e).__name__}: {str(e)[:160]}"
                entry["arms"][name] = None
            legs = [lg for tg, lg in LEGS[n0:] if tg == f"L{L}:{name}"]
            entry["legs"][name] = {lg: legs.count(lg) for lg in sorted(set(legs))}
            print(f"  L={L:4d} {name:16s} {entry['arms'][name]} ms  exact="
                  f"{entry['exact_vs_shipped'][name]}  legs={entry['legs'][name]}", flush=True)

        entry["l1_refused_cache"] = [str(k) for k in T._L1_OUT_REFUSED]
        R["sizes"][str(L)] = entry
        for t in (z, w1, w1a, w1b, w2, nw, nb):
            ttnn.deallocate(t)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
