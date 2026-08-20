"""Screen F1 at the `k_tiles=4` key, off-fold, one size per process.

    python perf/nesso1/f1_k4_screen.py --n 32 --out perf/nesso1/results/f1_k4_n32.json

Brief item 2. `trimul_tail.F1_BLOCK_KEYS` admits only (8, 8) = c_z 256, so F1 declines every
`c_z=128` model tt-bio ships (Nesso-1, Boltz-2, OpenFold3) on `k_tiles=4`. The recorded reason not
to widen it is the (12, 12) failure of 2026-08-15: wrong numbers at N=32 and N=64, device hang at
N=128, because the kernels are a transcription of `minimal_matmul` swept only at (4, 8, 1, 4, 1)
while (12, 12) resolves to (8, 12, 1, 2, 1) -- `out_block` doubles to 8 tiles and `subblock_h`
halves to 2, and the fork's circular buffers do not follow.

That reason does not obviously transfer to (4, 4), which is why this screens instead of assuming:

    (8, 8)   -> (4, 8, 1, 4, 1)   out_block 4, subblock_h 4     SAFE today
    (4, 4)   -> (4, 4, 1, 4, 1)   out_block 4, subblock_h 4     only K differs, 8 -> 4
    (12, 12) -> (8, 12, 1, 2, 1)  out_block 8, subblock_h 2     hung the device

The fork's own three CBs (c_4, c_5 at `out_block * 2`, c_6 at 2) are K-independent, and
`_MM_BLOCK[(4, 4)]` is already the config production's own Boltz-2 / OpenFold3 projections fold
with. So the prediction is that (4, 4) is numerically fine and the (12, 12) hang is specific to the
block geometry, not to widening the allow-list. A prediction is not a result: N=32, 64 and 128 are
run first because those are exactly where (12, 12) went wrong, one process each so a hang costs one
size and not the pass.

Correctness bar is `torch.equal` against the three ops F1 replaces, at the same block config and
the same grid the fold uses. Timing is a per-call screen, not a fold gain.
"""

import argparse, json, os, statistics, sys, time

import torch

torch.set_grad_enabled(False)
torch.manual_seed(893)

C_Z = 128          # Nesso-1 / Boltz-2 / OpenFold3 token_z -> k_tiles = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--c-z", type=int, default=C_Z)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio import trimul_tail as F1
    from tt_bio import mm_generic as MG

    rec = {"n": args.n, "c_z": args.c_z, "block_key": [args.c_z // 32, args.c_z // 32],
           "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES", "(unset)"),
           "loadavg": os.getloadavg(), "status": "started"}

    def save():
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")
    save()

    dev = TT.get_device()

    # Take the compute kernel config and both weights off a real one-block PairformerModule at
    # token_z = c_z, so the reference arm is the ops F1 actually replaces at the config the fold
    # actually folds with, not a hand-built approximation of it.
    from tt_bio.reference import PairformerNoSeqModule as RefPairformer
    mod = TT.PairformerModule(1, 32, 4, None, None, False, affinity=True)
    mod.load_state_dict(RefPairformer(args.c_z, 1, v2=True).eval().state_dict(), strict=False)
    # PairformerModule registers no child nn.Modules (n_modules == 1), so walk the real path.
    tail = getattr(mod.module.blocks[0], "triangle_multiplication_start", None)
    if tail is None:
        rec["status"] = "NO_TRIMUL_TAIL_FOUND"
        save()
        raise SystemExit(rec["status"])
    ckc = tail.compute_kernel_config
    rec["ckc"] = str(ckc)

    n, cz = args.n, args.c_z
    xa = ttnn.from_torch(torch.randn(1, n, n, cz), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xb = ttnn.from_torch(torch.randn(1, n, n, cz), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    wa, wb = tail.out_p_weight, tail.g_out_weight
    rec["w_shape"] = [int(d) for d in wa.shape]
    rec["w_dtype"] = str(wa.dtype)

    rec["block"] = list(TT._MM_BLOCK.get((cz // 32, cz // 32), ()))
    rec["out_block"] = (rec["block"][0] * rec["block"][2]) if rec["block"] else None
    rec["subblock_h"] = rec["block"][3] if rec["block"] else None

    def reference():
        p = TT._trimul_out_proj(xa, wa, ckc)
        g = TT._trimul_out_proj(xb, wb, ckc)
        return ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

    # ---- reference arm --------------------------------------------------------------------
    ref = reference()
    ref_t = ttnn.to_torch(ref).float()
    ttnn.deallocate(ref)
    ttnn.synchronize_device(dev)

    # ---- F1 arm, allow-list widened in this process only ---------------------------------
    rec["declined_before_widening"] = F1.eligible(xa, xb, wa, wb)
    F1.F1_BLOCK_KEYS = set(F1.F1_BLOCK_KEYS) | {(cz // 32, cz // 32)}
    F1._block_for.cache_clear()
    rec["declined_after_widening"] = F1.eligible(xa, xb, wa, wb)
    save()

    grid = tuple(TT.COMPUTE_GRID_MAIN)
    rec["grid"] = list(grid)
    save()

    fused = F1.fused_tail(xa, xb, wa, wb, MG.ckc_args(ckc), grid)
    if fused is None:
        rec["status"] = "DECLINED"
        rec["rejects"] = {str(k): v for k, v in F1.REJECTS.items()}
        save()
        print(json.dumps(rec, indent=1))
        return
    ttnn.synchronize_device(dev)
    f1_t = ttnn.to_torch(fused).float()
    ttnn.deallocate(fused)

    rec["equal"] = bool(torch.equal(ref_t, f1_t))
    rec["max_abs_diff"] = float((ref_t - f1_t).abs().max())
    rec["ref_abs_max"] = float(ref_t.abs().max())
    rec["mismatch_elems"] = int((ref_t != f1_t).sum())
    rec["n_elems"] = int(ref_t.numel())
    rec["status"] = "numerics_done"
    save()

    # ---- per-call timing screen -----------------------------------------------------------
    def timed(fn, reps):
        ts = []
        for i in range(reps + 1):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            ttnn.deallocate(out)
            if i:
                ts.append(dt)
        return ts

    ref_ts = timed(reference, args.reps)
    f1_ts = timed(lambda: F1.fused_tail(xa, xb, wa, wb, MG.ckc_args(ckc), grid), args.reps)
    rec["ref_ms"] = round(statistics.median(ref_ts) * 1e3, 4)
    rec["f1_ms"] = round(statistics.median(f1_ts) * 1e3, 4)
    rec["ref_ms_all"] = [round(t * 1e3, 4) for t in ref_ts]
    rec["f1_ms_all"] = [round(t * 1e3, 4) for t in f1_ts]
    rec["speedup"] = round(rec["ref_ms"] / rec["f1_ms"], 4) if rec["f1_ms"] else None
    rec["status"] = "ok"
    save()
    print(json.dumps({k: rec[k] for k in ("n", "block", "out_block", "subblock_h", "equal",
                                          "max_abs_diff", "mismatch_elems", "ref_ms", "f1_ms",
                                          "speedup")}, indent=1))


if __name__ == "__main__":
    main()
