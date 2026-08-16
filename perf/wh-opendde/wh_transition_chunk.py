#!/usr/bin/env python3
"""Screen: the 4D `Transition` row-chunk height on Wormhole at OpenDDE's c_z=384.

`tenstorrent.py:3500-3503` shrinks the row-chunk reference in proportion to the channel's excess
over 128 **only on a small grid**:

    _ref = 1024 * 128 ;  if _IS_SMALL_GRID:  _ref = _ref * 128 // max(128, c)
    h_chunk = max(1, int(TRANSITION_H_CHUNK_SIZE * min(1.0, _ref / (w_eff * c))))

At OpenDDE's c_z=384 and W=512 that is h_chunk = 3 on Wormhole against 10 on Blackhole, i.e. 171
row blocks instead of 52 for the same work. The shrink was added for a measured clash (Protenix-v2
c=256 at W=512), so the question is not "delete it" but "how big can the chunk be on THIS part
before it clashes, and what is the wall between here and there".

Arms are driven by setting `TRANSITION_H_CHUNK_SIZE` so the shipped expression lands on a target
h_chunk, which is asserted rather than assumed -- no arm silently reads as another. Every arm is
`torch.equal`-checked against the shipped one: swiglu is row-local, so a row-block boundary cannot
move a byte, and an arm that is not bit-exact is a bug in this probe.

Usage: TT_VISIBLE_DEVICES=<umd> python3 perf/wh-opendde/wh_transition_chunk.py --out results/x.json
"""
import argparse, json, math, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def build_transition(c, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {"norm.weight": torch.ones(c), "norm.bias": torch.zeros(c),
          "fc1.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc2.weight": torch.randn(n * c, c, generator=g) * (c ** -0.5),
          "fc3.weight": torch.randn(c, n * c, generator=g) * ((n * c) ** -0.5)}
    ckc = ttnn.init_device_compute_kernel_config(
        T.get_device().arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    return T.Transition(sd, ckc)


def ratio_for(W, c):
    """The shipped `min(1.0, _ref / (w_eff * c))` factor, recomputed here so an arm's target
    h_chunk can be hit exactly instead of guessed."""
    thr = (T.SEQ_LEN_MORE_CHUNKING if T.COMPUTE_GRID_MAIN[0] == T.COMPUTE_GRID_X_13
           else T.TRANSITION_W_CHUNKING_THRESHOLD)
    w_eff = min(W, T.TRANSITION_W_CHUNK_SIZE) if W > thr else W
    ref = 1024 * 128
    if T._IS_SMALL_GRID:
        ref = ref * 128 // max(128, c)
    return min(1.0, ref / (w_eff * c)), w_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="320,512,640,995")
    ap.add_argument("--c", type=int, default=384)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--targets", default="3,6,10,16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    base_h = T.TRANSITION_H_CHUNK_SIZE
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "arch": str(dev.arch()).rsplit(".", 1)[-1], "grid": [g.x, g.y],
           "compute_grid_main": list(T.COMPUTE_GRID_MAIN), "is_small_grid": T._IS_SMALL_GRID,
           "l1_per_core": T.ttnn.get_max_worker_l1_unreserved_size(),
           "transition_h_chunk_size": base_h, "c": a.c, "iters": a.iters, "rows": []}
    print(f"grid {g.x}x{g.y} small={T._IS_SMALL_GRID} base_h={base_h}", flush=True)

    tr = build_transition(a.c)
    for W in [int(s) for s in a.sizes.split(",")]:
        H = W
        r, w_eff = ratio_for(W, a.c)
        shipped_h = max(1, int(base_h * r))
        x_t = torch.randn(1, H, W, a.c) * 0.5
        ref_out = None
        print(f"--- W=H={W} c={a.c}: ratio {r:.4f} w_eff {w_eff} shipped h_chunk {shipped_h} "
              f"({-(-H // shipped_h)} blocks) ---", flush=True)
        for target in sorted({shipped_h, *[int(t) for t in a.targets.split(",")]}):
            # Hit `target` exactly through the shipped expression, or skip the arm.
            k = max(1, int(math.ceil(target / r))) if r > 0 else target
            for cand in (k, k - 1, k + 1, k - 2, k + 2):
                if cand >= 1 and max(1, int(cand * r)) == target:
                    k = cand
                    break
            else:
                print(f"  h={target}: unreachable through the shipped expression, skipped",
                      flush=True)
                continue
            T.TRANSITION_H_CHUNK_SIZE = k
            row = {"W": W, "c": a.c, "h_chunk": target, "k": k,
                   "blocks": -(-H // target), "shipped": target == shipped_h}
            try:
                walls = []
                for i in range(a.warm + a.iters):
                    xt = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev,
                                         dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    out = tr(xt)
                    ttnn.synchronize_device(dev)
                    dt = time.perf_counter() - t0
                    if i >= a.warm:
                        walls.append(dt)
                    ho = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                    ttnn.deallocate(xt)
                row["ms"] = round(st.median(walls) * 1e3, 3)
                row["ms_min"] = round(min(walls) * 1e3, 3)
                row["ms_spread_pct"] = round(100 * (max(walls) - min(walls)) / st.median(walls), 2)
                if ref_out is None or row["shipped"]:
                    ref_out = ho if ref_out is None else ref_out
                row["bit_exact_vs_first"] = bool(torch.equal(ho, ref_out))
                if not row["bit_exact_vs_first"]:
                    row["max_abs_diff"] = float((ho - ref_out).abs().max())
                print(f"  h={target:3d} k={k:3d} blocks={row['blocks']:4d}  {row['ms']:9.3f} ms "
                      f"(spread {row['ms_spread_pct']:.1f}%)  exact={row['bit_exact_vs_first']}",
                      flush=True)
            except Exception as e:                                              # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                print(f"  h={target:3d} k={k:3d} FAILED {row['error'][:200]}", flush=True)
            res["rows"].append(row)
            a.out.write_text(json.dumps(res, indent=1))
        T.TRANSITION_H_CHUNK_SIZE = base_h
        # Ratio table against the shipped arm, so the screen answers its own question.
        got = {r0["h_chunk"]: r0.get("ms") for r0 in res["rows"] if r0["W"] == W and "ms" in r0}
        if shipped_h in got:
            for h, ms in sorted(got.items()):
                print(f"    W={W} h={h:3d}: {got[shipped_h]/ms:.4f}x vs shipped h={shipped_h}",
                      flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
