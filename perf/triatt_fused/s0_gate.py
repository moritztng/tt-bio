#!/usr/bin/env python3
"""S0: does the generic_op transcription reproduce the native minimal_matmul, in output and in time?

PREDICTION, WRITTEN BEFORE THE RUN (state/triatt-fused-kernel-final.md 5, and the docstring of
perf/triatt_fused/generic_mm.py):

    2.13-2.35 ms against the native qkv arm measured in the same process, and torch.equal.
    > 2.40 ms, or not torch.equal: the route is dead and this task is a NO-GO.

Both arms run in one process, warm, alternating, with an A/A pair, so the comparison does not carry
a process or a thermal difference.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import ttnn
from tt_bio import tenstorrent as T
import generic_mm as G

RES = {"predictions": __doc__, "arms": []}


def timed(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return st.median(ts), (max(ts) - min(ts)) / st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="perf/triatt_fused/s0_gate.json")
    args = ap.parse_args()
    S, C, H, D = args.n, 256, 8, 32

    dev = T.get_device()
    # exactly TorchWrapper's config (tt_bio/tenstorrent.py:4292), which is what the fold runs
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    RES["meta"] = {"n": S, "grid": list(T.COMPUTE_GRID_MAIN), "loadavg": os.getloadavg(),
                   "card": os.environ.get("TT_VISIBLE_DEVICES")}
    print(json.dumps(RES["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    x = dram(torch.randn(S, S, C).to(torch.bfloat16))
    w = dram(torch.randn(C, 3 * H * D).to(torch.bfloat16))

    cfg = T._qkv_mm_config(x, w)
    RES["meta"]["mm_config"] = str(cfg)
    blk = T._MM_BLOCK[(3 * H * D) // 32]
    gcfg = (blk, tuple(T.COMPUTE_GRID_MAIN))
    gckc = (ttnn.MathFidelity.HiFi4, False, True, False)

    def native():
        return ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg)

    ref = ttnn.to_torch(native())

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, S, 3 * H * D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG)

    t0 = time.perf_counter()
    entry = G.build(dev, x, w, out, gcfg, gckc)
    RES["meta"]["descriptor_build_s"] = time.perf_counter() - t0
    RES["meta"]["dims"] = entry["dims"]
    G._CACHE[G._key(x, w, out, gcfg, gckc)] = entry
    print(json.dumps({"dims": entry["dims"],
                      "descriptor_build_s": RES["meta"]["descriptor_build_s"]}), flush=True)

    def generic():
        return G.generic_minimal_matmul(dev, x, w, out, gcfg, gckc)

    generic()
    got = ttnn.to_torch(out)
    eq = bool(torch.equal(got, ref))
    RES["equal"] = eq
    if not eq:
        d = (got.float() - ref.float()).abs()
        RES["max_abs_err"] = float(d.max())
        RES["mismatch_frac"] = float((d > 0).float().mean())
    print(json.dumps({"torch_equal": eq, **{k: RES[k] for k in
                                            ("max_abs_err", "mismatch_frac") if k in RES}}),
          flush=True)

    # alternating arms, twice round, plus an A/A pair on the native arm
    for rnd in range(2):
        for label, fn in (("native", native), ("generic_op", generic)):
            ms, aa = timed(fn, dev)
            row = {"arm": label, "round": rnd, "ms": ms * 1e3, "aa_spread": aa}
            RES["arms"].append(row)
            print(json.dumps(row), flush=True)

    ms_a, _ = timed(native, dev)
    ms_b, _ = timed(native, dev)
    RES["aa_pair_native_ms"] = [ms_a * 1e3, ms_b * 1e3]
    print(json.dumps({"aa_pair_native_ms": RES["aa_pair_native_ms"]}), flush=True)

    # host cost of re-binding the addresses in the cached descriptor
    t0 = time.perf_counter()
    for _ in range(20):
        G.rebind(entry, x.buffer_address(), w.buffer_address(), out.buffer_address())
    RES["rebind_us"] = (time.perf_counter() - t0) / 20 * 1e6
    print(json.dumps({"rebind_us": RES["rebind_us"]}), flush=True)

    nat = st.median([r["ms"] for r in RES["arms"] if r["arm"] == "native"])
    gen = st.median([r["ms"] for r in RES["arms"] if r["arm"] == "generic_op"])
    RES["summary"] = {"native_ms": nat, "generic_ms": gen, "ratio_generic_over_native": gen / nat,
                      "gate_pass": bool(eq and gen <= 2.40)}
    RES["meta"]["loadavg_end"] = os.getloadavg()
    print(json.dumps(RES["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
