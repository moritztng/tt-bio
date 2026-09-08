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


def shipped_h_for(W, c, hid):
    """The height the ENGINE lands on at this shape, mirroring `Transition.__call__`.

    This used to read `w_eff` off a token-count threshold, which the engine stopped doing: it now
    decides `w_chunked` from whether ANY row fits (`_rows_at(W) < 1.0`), so at OpenDDE's c=384 the
    old rule here predicted w_eff=512 and h=3 at W=896 where the fold actually runs w_eff=896 and
    h=2. A screen that names the wrong shipped arm measures the wrong thing, so the whole chain is
    reproduced -- ratio, the L1 cap, the floor -- and `_SHIPPED_SELFCHECK` below asserts it against
    the two heights the DRAM census read off real folds.
    """
    tile = lambda v: -(-int(v) // 32) * 32
    gx, gy = T.COMPUTE_GRID_MAIN
    base_h = T.TRANSITION_H_CHUNK_SIZE
    ref = 1024 * 128
    if T._IS_SMALL_GRID:
        ref = ref * 128 // max(128, c)
    l1_rows_at = lambda w: (T.TRANSITION_L1_CHUNK_BYTES_PER_CORE * gx * gy
                            / (2 * tile(w) * (tile(c) + 2 * tile(hid))))
    def rows_at(w):
        h = base_h * min(1.0, ref / (w * c))
        return min(h, l1_rows_at(w)) if T._IS_SMALL_GRID else h
    w_chunked = rows_at(W) < 1.0 if T._IS_SMALL_GRID else W > T.TRANSITION_W_CHUNKING_THRESHOLD
    w_eff = min(W, T.TRANSITION_W_CHUNK_SIZE) if w_chunked else W
    h = max(1, int(base_h * min(1.0, ref / (w_eff * c))))
    if T._IS_SMALL_GRID:
        h = min(h, max(1, int(l1_rows_at(w_eff))))
    return h, w_eff, w_chunked


#: (W, c, hid) -> h, as MEASURED on device by the DRAM census (state/opendde-l1-clash-to-1024.md).
#: A drift in the mirrored chain above fails here instead of in a silently mislabelled arm.
_SHIPPED_SELFCHECK = {(896, 384, 1536): 2, (1024, 384, 1536): 1}


def main():
    ap = argparse.ArgumentParser()
    # A width sweep at fixed height, because the height alone is refuted: h=2 covers W=608..896,
    # and 640 aa (h=2, 490 s) and 768 aa (h=2, 876 s) both fold fine while 896 aa (h=2) has never
    # finished. 896 is the only one of them whose width carries a factor of 7 (28 tiles, against
    # 20, 24 and 32), so the sweep steps across it in 32s to see whether the cost spikes at the
    # width rather than at the height.
    ap.add_argument("--sizes", default="768,800,832,864,896,928,960,1024")
    ap.add_argument("--c", type=int, default=384)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--targets", default="1,2")
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
        hid = int(tr.fc1_weight.shape[-1])
        shipped_h, w_eff, w_chunked = shipped_h_for(W, a.c, hid)
        want = _SHIPPED_SELFCHECK.get((W, a.c, hid))
        assert want is None or want == shipped_h, (
            f"mirrored chain says h={shipped_h} at W={W} c={a.c}, census measured {want}")
        r = min(1.0, (1024 * 128 * 128 // max(128, a.c) if T._IS_SMALL_GRID else 1024 * 128)
                / (w_eff * a.c))
        x_t = torch.randn(1, H, W, a.c) * 0.5
        ref_out = None
        print(f"--- W=H={W} c={a.c}: w_eff {w_eff} w_chunked {w_chunked} "
              f"shipped h_chunk {shipped_h} ({-(-H // shipped_h)} blocks) ---", flush=True)
        for target in sorted({shipped_h, *[int(t) for t in a.targets.split(",")]}):
            # The engine's own screen hook sets the height exactly, so an arm cannot land on a
            # neighbour of what it claims -- which the old back-solve through
            # TRANSITION_H_CHUNK_SIZE could, and did whenever the ratio made a target unreachable.
            os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(target)
            row = {"W": W, "c": a.c, "h_chunk": target,
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
                print(f"  h={target:3d} blocks={row['blocks']:4d}  {row['ms']:9.3f} ms "
                      f"(spread {row['ms_spread_pct']:.1f}%)  exact={row['bit_exact_vs_first']}",
                      flush=True)
            except Exception as e:                                              # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                print(f"  h={target:3d} FAILED {row['error'][:300]}", flush=True)
            res["rows"].append(row)
            a.out.write_text(json.dumps(res, indent=1))
        os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
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
