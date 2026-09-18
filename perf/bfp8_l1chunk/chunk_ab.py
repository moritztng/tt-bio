#!/usr/bin/env python3
"""bf16 at its best legal chunk against bfp8 at ITS best legal chunk, on the trunk
triangle-attention key.

This is the row's own question and it is not the one bfp8-sdpa-unlock answered. That row held the
chunk fixed and measured the operand narrowing alone: 0.8071x at B = 512, a loss. The L1 premise
says the narrowing pays for itself by buying a chunk bf16 cannot fit, so the arms here are each
kernel's BEST config rather than a shared one.

At padded 512 the surface says that difference is one k_chunk: bf16 holds (q512, k256) and bfp8
additionally holds (q512, k512), which drops k_num_chunks 2 -> 1 and removes the single online
softmax rescale. per_core_cost ranks that at 1.001x, so the prediction registered before the run
is that it cannot cover the operand penalty.

Arms interleaved rep by rep with the interior order reversed on odd reps; the bf16 arm runs twice
a rep as its own A/A twin.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, socket, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_cspec = importlib.util.spec_from_file_location(
    "_clk", REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock" / "clk.py")
CLK = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(CLK)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_sdpa as TS
    import tt_bio.sdpa_generic as SG
    from tt_bio.tenstorrent import get_device
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), _TB.__file__

    dev = get_device()
    S, H, D = a.seq, a.heads, a.head_dim
    cores = TT.COMPUTE_GRID_MAIN[0] * TT.COMPUTE_GRID_MAIN[1]
    scale = D ** -0.5

    def mk(dtype):
        g = torch.Generator().manual_seed(0)
        q, k, v = (ttnn.from_torch(torch.randn(S, H, S, D, generator=g), dtype=dtype,
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(3))
        b = ttnn.from_torch(torch.randn(1, H, S, S, generator=g), dtype=dtype,
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return q, k, v, b

    # each arm's best LEGAL config, from the host surface -- not a shared config
    def best(dtype):
        cands = []
        for qc in SG.chunk_divisors(S):
            for kc in SG.chunk_divisors(S):
                q_pf = TS.q_parallel_factor(S, H, qc, cores, cap=0)
                p = SG.plan_for_shape(S, H, D, qc, kc, grid=(cores, 1), split=(
                    max(cores // (H * q_pf), 1), H, q_pf), dtype=dtype)
                if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
                    continue
                pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
                dts = {f"{o}_dtype": dtype for o in ("q", "k", "v", "mask", "out")}
                if SG.cb_fits_l1(p, mask_cb_tiles=pers, **dts):
                    cands.append((TS.per_core_cost(p, qc, S), qc, kc))
        return min(cands)[1:] if cands else None

    arms = {}
    for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        cfg = best(dt)
        arms[name] = {"dtype": dt, "cfg": cfg, "tensors": mk(dt)}
        print(f"arm {name}: best legal chunk (q, k) = {cfg}")

    def run(name):
        q, k, v, b = arms[name]["tensors"]
        qc, kc = arms[name]["cfg"]
        served = TS.STATS[0]
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters):
            o = TS.sdpa(q, k, v, b, scale, qc, kc)
            assert o is not None, f"{name} declined -- a fallback is a failure, not a cost"
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / a.iters
        return dt, TS.STATS[0] - served

    for name in arms:                                   # warm: each arm compiles its own program
        run(name)

    nodes = CLK.nodes_open_by_this_process()
    held = CLK.force(a.mhz, nodes)
    sampler = CLK.Sampler(nodes[0])
    draws = {"bf16": [], "bfp8": [], "bf16_aa": []}
    serves = {"bf16": 0, "bfp8": 0, "bf16_aa": 0}
    for rep in range(a.reps):
        order = ["bf16", "bfp8", "bf16_aa"] if rep % 2 == 0 else ["bf16_aa", "bfp8", "bf16"]
        for slot in order:
            dt, n = run("bf16" if slot == "bf16_aa" else slot)
            draws[slot].append(dt)
            serves[slot] += n
    clk_stats = sampler.stop()
    CLK.release()

    med = {k: st.median(v) for k, v in draws.items()}
    res = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": list(TT.COMPUTE_GRID_MAIN), "cores": cores,
        "shape": {"S": S, "H": H, "D": D}, "reps": a.reps, "iters": a.iters,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "aiclk_requested_mhz": a.mhz, "aiclk_nodes_held": held,
        "aiclk_sampled_during_run": clk_stats,
        "cfg_bf16": list(arms["bf16"]["cfg"]), "cfg_bfp8": list(arms["bfp8"]["cfg"]),
        "median_ms": {k: round(v * 1e3, 4) for k, v in med.items()},
        "aa_floor_pct": round(100 * abs(med["bf16_aa"] - med["bf16"]) / med["bf16"], 4),
        "ratio_bf16_over_bfp8": round(med["bf16"] / med["bfp8"], 4),
        "served_per_slot": serves,
        "draws_ms": {k: [round(x * 1e3, 4) for x in v] for k, v in draws.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "draws_ms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
