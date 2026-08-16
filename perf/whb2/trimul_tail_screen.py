"""Lever A's screen: is Boltz-2's trimul tail actually bandwidth-bound, and does F1 decline it?

State doc section 4A prices the shipped tail at 4P read + 3P write per call. Verified against
`tenstorrent.py:2643-2650`, which is `p_out = x @ Wp`, `g_out = x_norm_in @ Wg`, then
`multiply_(p_out, g_out, SIGMOID)` in place -- and x and x_norm_in are two DISTINCT tensors, so the
count is right: read x, write p, read x_norm_in, write g, read p+g, write p. 7P.

At 512 aa, P = 512*512*128*2 B = 67.109 MB, so 7P = 469.8 MB/call. At the measured 218.5 GB/s
Wormhole DRAM roof that is 2.15 ms/call, and 1.075 ms at Blackhole's 409.5 GB/s.

KILL GATE, written before any build: if the three-op sequence measures above ~3.2 ms on Wormhole
(under 67 % of roof) the sequence is not bandwidth-bound, the byte model does not predict the win,
and lever A stops. The same clause on Blackhole is ~1.6 ms.

The script also calls `fused_tail` at the production shape. Before lever A it confirmed on
hardware what the census inferred: F1 declined every Boltz-2 tail because the block was the
single tuple `(4, 8, 1, 4, 1)`, which demands kt == 8, and Boltz-2's tail is 128 wide, kt = 4.
With the block passed in from `_mm_block_for` it serves, and the run then times it and checks
`torch.equal` against the three ops it replaces. Reject reasons are reported verbatim either way.

SECOND KILL GATE, from state doc 4A: the fused call must beat 75 % of the shipped sequence,
i.e. return at least a 25 % improvement. Below that, NO-GO.

Run with TT_VISIBLE_DEVICES=<free id>.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[384, 512, 640, 1024])
    ap.add_argument("--cz", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=5)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.trimul_tail as TT
    assert Path(T.__file__).resolve().is_relative_to(tree)

    device = T.get_device()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    # Exactly the trunk's config: TorchWrapper.__init__ (tenstorrent.py:5367) picks the class off
    # the arch and builds HiFi4 / fp32_dest_acc / packer_l1_acc, then trunk_compute_kernel_config
    # applies TT_BIO_TRUNK_MATH_FIDELITY, which defaults to hifi4 (production unchanged). Restating
    # a fidelity here would price a matmul the fold never runs.
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if device.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = T.trunk_compute_kernel_config(kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True))

    # The measured DRAM roof for each part, from perf/wh-baseline/wh_machine.json and
    # perf/whb2/bh_probe_qb1c1.log. Named, not asserted -- a roof quoted without a measurement
    # behind it is how this lineage published 668 GB/s on a ~400 GB/s card.
    ROOF = {"wormhole_b0": 218.5e9, "blackhole": 409.5e9}
    arch = T.arch_name()
    roof = ROOF.get(arch)

    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(grid), "cores": grid[0] * grid[1], "arch": arch, "c_z": a.cz,
           "roof_gbs": (roof or 0) / 1e9, "cases": []}

    def timed(fn):
        for _ in range(a.warmup):
            o = fn()
            if o is None:
                return None
            ttnn.deallocate(o)
        ttnn.synchronize_device(device)
        ms = []
        for _ in range(a.blocks):
            t0 = time.perf_counter()
            for _ in range(a.iters):
                o = fn()
            ttnn.synchronize_device(device)
            ms.append((time.perf_counter() - t0) * 1e3 / a.iters)
            ttnn.deallocate(o)
        return {"best": round(min(ms), 4), "median": round(statistics.median(ms), 4),
                "all": [round(x, 4) for x in ms]}

    for L in a.sizes:
        cz = a.cz
        P = L * L * cz * 2
        rec = {"L": L, "P_bytes": P, "shipped_bytes": 7 * P, "fused_bytes": 3 * P}
        if roof:
            rec["predicted_shipped_ms"] = round(7 * P / roof * 1e3, 4)
            rec["predicted_fused_ms"] = round(3 * P / roof * 1e3, 4)
            rec["kill_gate_ms"] = round(rec["predicted_shipped_ms"] / 0.67, 4)
        try:
            def dev(shape):
                t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = dev((1, L, L, cz))
            xn = dev((1, L, L, cz))
            wp = dev((cz, cz))
            wg = dev((cz, cz))

            def shipped():
                p = T._trimul_out_proj(x, wp, ckc)
                g = T._trimul_out_proj(xn, wg, ckc)
                r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(g)
                return r
            rec["shipped"] = timed(shipped)
            if rec["shipped"] and roof:
                gbs = 7 * P / (rec["shipped"]["median"] * 1e-3) / 1e9
                rec["achieved_gbs"] = round(gbs, 1)
                rec["frac_of_roof"] = round(gbs / (roof / 1e9), 4)
                rec["kill_gate_passes"] = bool(rec["shipped"]["median"] <= rec["kill_gate_ms"])

            # F1 at the production shape, with the block the rest of the codebase would pick
            # for this weight. Before lever A the block argument did not exist and this
            # declined at ('k_tiles=4', ...).
            block = T._mm_block_for(wp)
            rec["block"] = list(block) if block else None
            before = dict(getattr(TT, "REJECTS", {}))
            cargs = T._mm_generic.ckc_args(ckc)
            f = TT.fused_tail(x, xn, wp, wg, cargs, grid, block)
            rec["fused_served"] = f is not None
            if f is not None:
                # Correctness before speed. The three ops write in place, so the reference is
                # taken on its own copies and both come back to host as bf16 bit patterns.
                pr = T._trimul_out_proj(x, wp, ckc)
                gr = T._trimul_out_proj(xn, wg, ckc)
                ref = ttnn.multiply_(
                    pr, gr, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(gr)
                a_t, b_t = ttnn.to_torch(f), ttnn.to_torch(ref)
                rec["bit_exact"] = bool(torch.equal(a_t, b_t))
                if not rec["bit_exact"]:
                    d = (a_t.float() - b_t.float()).abs()
                    rec["miss"] = {"n": int((d > 0).sum()), "of": int(d.numel()),
                                   "max_abs": float(d.max()),
                                   "ref_max_abs": float(b_t.float().abs().max())}
                ttnn.deallocate(ref)
                ttnn.deallocate(f)
                rec["fused"] = timed(lambda: TT.fused_tail(
                    x, xn, wp, wg, cargs, grid, block))
                if rec["fused"] and rec.get("shipped"):
                    sp = rec["shipped"]["median"] / rec["fused"]["median"]
                    rec["speedup_fused_over_shipped"] = round(sp, 4)
                    rec["gate2_passes"] = bool(sp >= 1.3333)
                    if roof:
                        g2 = 3 * P / (rec["fused"]["median"] * 1e-3) / 1e9
                        rec["fused_achieved_gbs"] = round(g2, 1)
                        rec["fused_frac_of_roof"] = round(g2 / (roof / 1e9), 4)
            after = dict(getattr(TT, "REJECTS", {}))
            rec["new_rejects"] = {str(k): int(after[k] - before.get(k, 0))
                                  for k in after if after[k] - before.get(k, 0)}
            for t in (x, xn, wp, wg):
                ttnn.deallocate(t)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["cases"].append(rec)
        print(json.dumps(rec), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
