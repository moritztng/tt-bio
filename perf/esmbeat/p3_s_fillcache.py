#!/usr/bin/env python3
"""C-out without a kernel: can `ttnn.fill_cache` write a row block into the pair tensor at an offset?

§8.1 swept the wheel for an op that writes a TILE tensor into another at an offset and dismissed
the `*_cache` family by name as "KV-cache-shaped". That was a name-level dismissal, not a shape
check. `ttnn.fill_cache(cache, input, batch_idx)` fills `cache[batch_idx]` in place from
`input[0]`, i.e. it writes at an offset along dim 0. The pair tensor is `[1, L, L, C]` and the row
block is `[1, R, L, C]`; viewed as `[L/R, R, L, C]` the block offset IS a dim-0 index, so the op's
shape contract and ours are the same object under a metadata-only reshape.

If it serves, C-out lands with no kernel at all: the per-block residual add (lever F) keeps its
output in L1 and `fill_cache` writes it straight into the assembled tensor, deleting the `concat`'s
134 MB read + 134 MB write per call (0.310 s/fold on qb1, §8.1).

Two things have to hold and both are measured here, not argued:
  V1  the view shares the buffer, so the in-place fill is visible through the original tensor.
  V2  `fill_cache` accepts an L1 input. With a DRAM input the block is written to DRAM and read
      back, which is exactly the traffic `concat` already pays -- the saving would be zero.

ARMS (batched, never per-op-synced)
  concat  16 blocks added into DRAM, then `ttnn.concat`. The shipped assembly with lever F.
  fill    16 blocks added into L1, each written in place with `fill_cache`. No concat.

KILL GATES, PRE-COMMITTED
  K1  `fill_cache` raises, or the view does not alias        -> NO-GO, C-out needs its kernel.
  K2  not `torch.equal` to the concat arm                    -> DEAD, do not scope by size.
  K3  (concat - fill) * 538 < 0.15 s                         -> NO-GO on size, kernel or nothing.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC

CALLS_PER_FOLD = 538


def timed(fn, dev, reps=4, batches=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L, C_Z, R = a.size, 256, EC._PAIR_FFN_ROW_BLOCK
    nblk = -(-L // R)

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)
    to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=DR)

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "size": L, "rows": R, "nblk": nblk,
           "calls_per_fold": CALLS_PER_FOLD, "ms": {}, "raw": {}, "notes": {}}

    # Two operands per block, in DRAM. They stand in for lever F's `pair_block` and `fc2_out`, and
    # they stay in DRAM because both arms read them identically -- what this screen compares is the
    # WRITE destination. Sixteen resident 8.39 MB blocks do not fit L1 anyway (§8.3, measured).
    pt = torch.randn(1, L, L, C_Z)
    qt = torch.randn(1, L, L, C_Z)
    p_blocks = [ttnn.from_torch(pt[:, i * R:min((i + 1) * R, L)], dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)
                for i in range(nblk)]
    q_blocks = [ttnn.from_torch(qt[:, i * R:min((i + 1) * R, L)], dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)
                for i in range(nblk)]

    # --- arm: concat -------------------------------------------------------------------------
    def arm_concat(keep=False):
        blocks = [ttnn.add(p_blocks[i], q_blocks[i], memory_config=DR) for i in range(nblk)]
        out = ttnn.concat(blocks, dim=1)
        for b in blocks:
            ttnn.deallocate(b)
        if keep:
            return out
        ttnn.deallocate(out)
        return None

    ref = arm_concat(keep=True)
    ref_t = ttnn.to_torch(ref)

    # --- V1: does a metadata reshape alias the same buffer? ----------------------------------
    dst = to(torch.zeros(1, L, L, C_Z))
    view, view_how, aliases = None, None, None
    for how, fn in (("experimental.view", lambda t: ttnn.experimental.view(t, [nblk, R, L, C_Z])),
                    ("reshape", lambda t: ttnn.reshape(t, [nblk, R, L, C_Z]))):
        try:
            v = fn(dst)
        except Exception as e:                                    # noqa: BLE001
            res["notes"][how] = f"raised: {type(e).__name__}: {str(e)[:200]}"
            continue
        try:
            same = v.buffer_address() == dst.buffer_address()
        except Exception as e:                                    # noqa: BLE001
            same = None
            res["notes"][how + "_addr"] = f"{type(e).__name__}: {str(e)[:120]}"
        res["notes"][how] = f"ok, aliases={same}"
        if same:
            view, view_how, aliases = v, how, same
            break
        if view is None:
            view, view_how, aliases = v, how, same
    res["view_how"], res["view_aliases"] = view_how, aliases
    print("view:", view_how, "aliases:", aliases, res["notes"], flush=True)

    # --- V2: does fill_cache take an L1 input, and does it write where we think? -------------
    fill_ok, fill_err, fill_mc = False, None, None
    if view is not None:
        for mc, tag in ((L1, "L1"), (DR, "DRAM")):
            try:
                b = ttnn.add(p_blocks[0], q_blocks[0], memory_config=mc)
                ttnn.fill_cache(view, b, 0)
                ttnn.synchronize_device(dev)
                ttnn.deallocate(b)
                fill_ok, fill_mc = True, tag
                break
            except Exception as e:                                # noqa: BLE001
                fill_err = f"{tag}: {type(e).__name__}: {str(e)[:300]}"
                print("fill_cache refused", fill_err, flush=True)
    res["fill_ok"], res["fill_input_mc"], res["fill_err"] = fill_ok, fill_mc, fill_err

    if not fill_ok:
        res["verdict"] = "K1 NO-GO: fill_cache cannot serve as an offset write"
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    fmc = L1 if fill_mc == "L1" else DR

    # The production arm has to allocate its own output every call, the way `concat` does. That is
    # `allocate_tensor_on_device`: a bare buffer, no host copy and no zero fill (every one of the
    # nblk blocks is written, so uninitialised is correct). Timing it outside the arm would flatter
    # the lever by the one cost the concat arm cannot avoid.
    def arm_fill(keep=False):
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, L, L, C_Z]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, dev, DR)
        v = ttnn.experimental.view(o, [nblk, R, L, C_Z])
        for i in range(nblk):
            b = ttnn.add(p_blocks[i], q_blocks[i], memory_config=fmc)
            ttnn.fill_cache(v, b, i)
            ttnn.deallocate(b)
        if keep:
            return o
        ttnn.deallocate(o)
        return None

    got = arm_fill(keep=True)
    ttnn.synchronize_device(dev)
    got_t = ttnn.to_torch(got)
    eq = bool(torch.equal(got_t, ref_t))
    res["torch_equal"] = eq
    res["max_abs_diff"] = float((got_t.float() - ref_t.float()).abs().max())
    print("torch.equal", eq, "maxdiff", res["max_abs_diff"], flush=True)

    ttnn.deallocate(got)
    m_c, raw_c = timed(arm_concat, dev, reps=2, batches=5)
    m_f, raw_f = timed(arm_fill, dev, reps=2, batches=5)
    res["ms"]["concat"], res["raw"]["concat"] = round(m_c, 4), [round(v, 4) for v in raw_c]
    res["ms"]["fill"], res["raw"]["fill"] = round(m_f, 4), [round(v, 4) for v in raw_f]
    res["delta_ms"] = round(m_c - m_f, 4)
    res["delta_s_per_fold"] = round((m_c - m_f) * CALLS_PER_FOLD / 1e3, 3)
    res["verdict"] = ("K2 DEAD: not bit-exact" if not eq else
                      "K3 NO-GO: under 0.15 s" if res["delta_s_per_fold"] < 0.15 else
                      "GO: no-kernel C-out")
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
