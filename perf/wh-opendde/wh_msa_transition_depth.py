#!/usr/bin/env python3
"""Screen: the MSA-module `Transition` on a full-depth 4-D representation, 896 aa vs 1024 aa.

This is the call an 896-residue OpenDDE fold sits in for tens of minutes (`py-spy`:
`_msa` -> `Transition.__call__` -> the row listcomp -> `swiglu` -> `ttnn.linear`). It is NOT the
call `wh_transition_chunk.py` measures: that one is the pair track, c=384, H=W. This one is
c=128 with H = the MSA depth (8192 served rows) and W = the token axis, so H is 8-9x W and the
row loop takes the LAZY branch (H > SEQ_LEN_MORE_CHUNKING), where the row blocks are sliced inside
the loop and -- above `concat_host_bytes()` -- assembled on the HOST.

The question is an asymmetry, not a level: the 1024 aa fold makes the same call on a LARGER
representation (2.15 GiB against 1.88 GiB) and completes in ~1553 s, while 896 aa has never
finished in four attempts. So either this op is not where the fold's time goes, or its cost is
non-monotone in W here. A fold cannot answer that -- it cannot be stopped, so an overrunning leg
parks the card for hours (state/opendde-l1-clash-to-1024.md). This probe times the op alone and
exits.

Usage:
  TT_VISIBLE_DEVICES=<umd> TT_BIO_LEASE_CARDS=<grant> PYTHONPATH=$PWD \
    python3 perf/wh-opendde/wh_msa_transition_depth.py --out results/msa_depth.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T
from importlib.machinery import SourceFileLoader

# Reuse the sibling screen's builder so both measure the same module construction.
_sib = SourceFileLoader("wh_transition_chunk",
                        str(Path(__file__).with_name("wh_transition_chunk.py"))).load_module()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="896,1024")
    ap.add_argument("--depth", type=int, default=8192, help="served MSA rows")
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "is_small_grid": T._IS_SMALL_GRID,
           "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
           "transition_h_chunk_size": T.TRANSITION_H_CHUNK_SIZE,
           "depth": a.depth, "c": a.c, "rows": []}
    print(f"grid {g.x}x{g.y} small={T._IS_SMALL_GRID} "
          f"SEQ_LEN_MORE_CHUNKING={T.SEQ_LEN_MORE_CHUNKING} depth={a.depth}", flush=True)

    tr = _sib.build_transition(a.c)
    hid = int(tr.fc1_weight.shape[-1])
    for W in [int(x) for x in a.widths.split(",")]:
        h, w_eff, w_chunked = _sib.shipped_h_for(W, a.c, hid)
        lazy = a.depth > T.SEQ_LEN_MORE_CHUNKING
        gib = a.depth * W * a.c * 2 / 2 ** 30
        row = {"W": W, "depth": a.depth, "c": a.c, "h": h, "w_eff": w_eff,
               "w_chunked": w_chunked, "lazy_branch": lazy, "repr_gib": round(gib, 3),
               "blocks": -(-a.depth // h)}
        print(f"--- W={W} depth={a.depth} c={a.c}: repr {gib:.3f} GiB, h={h}, "
              f"{row['blocks']} row blocks, lazy={lazy} ---", flush=True)
        try:
            x_t = torch.randn(1, a.depth, W, a.c, dtype=torch.float32) * 0.5
            walls = []
            for i in range(a.warm + a.iters):
                xt = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev,
                                     dtype=ttnn.bfloat16,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                out = tr(xt)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                if i >= a.warm:
                    walls.append(dt)
                if isinstance(out, ttnn.Tensor):
                    ttnn.deallocate(out)
                ttnn.deallocate(xt)
                print(f"    iter {i}: {dt:.3f} s", flush=True)
            row["s"] = round(st.median(walls), 3)
            row["s_min"] = round(min(walls), 3)
            del x_t
        except Exception as e:                                                  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"    FAILED {row['error'][:300]}", flush=True)
        res["rows"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
    ok = [r for r in res["rows"] if "s" in r]
    for r in ok:
        print(f"  W={r['W']:5d}  {r['s']:8.3f} s per call  ({r['blocks']} blocks, h={r['h']})",
              flush=True)
    if len(ok) == 2:
        print(f"  ratio W={ok[1]['W']}/W={ok[0]['W']}: {ok[1]['s']/ok[0]['s']:.3f}x "
              f"(element-work ratio {ok[1]['W']/ok[0]['W']:.3f}x)", flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
