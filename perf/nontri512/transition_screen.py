#!/usr/bin/env python3
"""Phase-2 screen for the fused SwiGLU `Transition` kernel (task fold-nontriangle-below-4x).

The screen must predict the outcome of the ACTUAL change, not of a part of it. So every leg runs at
the production pair shape a 512-aa protenix-v2 fold really feeds `Transition` -- x=[1,512,512,256]
bf16 in DRAM, c_hid=1024, h_chunk=16 (32 chunks), HiFi4 + fp32_dest_acc + packer_l1_acc, real
trunk layer-0 weights -- and the legs are INTERLEAVED so a drift in the box cannot rank them.

  A       production `Transition.__call__`                    the 13.25 ms/call being replaced
  B       one chunk's six-op chain alone, x32                 loop overhead vs chain cost
  C       ORACLE FLOOR: the three GEMMs only, h=16,           the prediction: everything a fused
          pre-sliced chunks, hiddens L1, no norm,             kernel could delete, deleted
          no chunk, no concat
  C_ln    leg C + layer_norm                                  how much of A-C the norm is
  D       leg C at h=512 (one chunk, hiddens in DRAM)         the chunk loop vs its DRAM cost
  E_chunk `ttnn.chunk` alone at the shape                     MEASURED mechanism 2 (was DERIVED)
  E_cat   `ttnn.concat` of the 32 output chunks alone         MEASURED mechanism 2

Every timed region synchronises the device immediately before the clock starts and before it stops.
`--mode count` is a structural ttnn.graph tally of one warm call; NEVER quote its durations.
"""

import argparse
import collections
import json
import statistics as st
import time

import torch

import ttnn
from tt_bio import protenix_weights as PW
from tt_bio.tenstorrent import CORE_GRID_MAIN, PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"
TRI_HEAD_DIM = 32
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def build_transition(ckc):
    """The production `Transition` object, with the real protenix-v2 trunk layer-0 weights."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len("pairformer_stack.blocks.0."):]: v
           for k, v in sd.items() if k.startswith("pairformer_stack.blocks.0.")}
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    layer = PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True, remapped, ckc)
    return layer.transition_z, c_z


def chain(tz, c, norm=True, hidden_mem=L1, out_mem=DRAM, act="silu", gate=True):
    """The production swiglu chain, op for op (tt_bio/tenstorrent.py Transition.__call__)."""
    ckc, dtype = tz.compute_kernel_config, ttnn.bfloat16
    if norm:
        src = ttnn.layer_norm(c, weight=tz.norm_weight, bias=tz.norm_bias, epsilon=1e-5,
                              compute_kernel_config=ckc, memory_config=L1)
    else:
        src = c
    x1 = ttnn.linear(src, tz.fc1_weight, activation=act, compute_kernel_config=ckc,
                     memory_config=hidden_mem, dtype=dtype, core_grid=CORE_GRID_MAIN)
    x2 = ttnn.linear(src, tz.fc2_weight, compute_kernel_config=ckc, memory_config=hidden_mem,
                     dtype=dtype, core_grid=CORE_GRID_MAIN)
    if norm:
        ttnn.deallocate(src)
    if gate:
        x1 = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    out = ttnn.linear(x1, tz.fc3_weight, compute_kernel_config=ckc, dtype=dtype,
                      core_grid=CORE_GRID_MAIN, memory_config=out_mem)
    ttnn.deallocate(x1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "count"], default="bench")
    ap.add_argument("--n", type=int, default=512, help="padded token count (pair shape N x N)")
    ap.add_argument("--h", type=int, default=16, help="production row-chunk height")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--legs", type=str, default="A,B,C,C_ln,D,E_chunk,E_cat")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    tz, c_z = build_transition(ckc)
    N = args.n
    print(f"pair shape [1,{N},{N},{c_z}] bf16  h_chunk={args.h}  core_grid={CORE_GRID_MAIN.x}x"
          f"{CORE_GRID_MAIN.y}  ttnn={getattr(ttnn, '__version__', '?')}", flush=True)

    xt = torch.randn(1, N, N, c_z) * 0.5
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=DRAM)

    if args.mode == "count":
        tz(x)  # warm
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = tz(x)
        graph = ttnn.graph.end_graph_capture()
        ttnn.deallocate(out)
        ops = [nd for nd in graph if nd.get("node_type") == "function_start"]
        tally = collections.Counter(nd["params"].get("name", "?") for nd in ops)
        print(f"TTNN_OPS total={len(ops)} nodes={len(graph)}", flush=True)
        for k, v in tally.most_common():
            print(f"  {v:5d}  {k}", flush=True)
        if args.out:
            json.dump({"shape": [1, N, N, c_z], "h": args.h, "ops": len(ops),
                       "nodes": len(graph), "tally": dict(tally)}, open(args.out, "w"), indent=2)
        return

    # Pre-sliced chunks for the oracle legs: the slice is NOT in the timed region, because a fused
    # kernel reads its rows straight out of the input.
    nchunk = -(-N // args.h)
    chunks = [x[:, s:min(s + args.h, N)] for s in range(0, N, args.h)]
    ttnn.synchronize_device(dev)
    # 32 output-shaped chunks, for the concat leg alone.
    out_chunks = [ttnn.from_torch(torch.randn(1, args.h, N, c_z) * 0.1, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                  for _ in range(nchunk)]
    ttnn.synchronize_device(dev)

    def leg_A():
        ttnn.deallocate(tz(x))

    def leg_B():
        for _ in range(nchunk):
            ttnn.deallocate(chain(tz, chunks[0], norm=True, hidden_mem=L1, out_mem=DRAM))

    def leg_C():
        for c in chunks:
            ttnn.deallocate(chain(tz, c, norm=False, hidden_mem=L1, out_mem=DRAM))

    def leg_C_ln():
        for c in chunks:
            ttnn.deallocate(chain(tz, c, norm=True, hidden_mem=L1, out_mem=DRAM))

    def leg_C_strict():
        # The STRICT oracle floor: the three GEMMs and nothing else. No norm, no chunk, no concat,
        # no fused silu, no gate multiply. One measured wall, not a sum of separate legs.
        for c in chunks:
            ttnn.deallocate(chain(tz, c, norm=False, hidden_mem=L1, out_mem=DRAM,
                                  act=None, gate=False))

    def leg_C_nosilu():
        # Leg C with the fused silu removed and the gate kept: the fused-activation penalty in situ.
        for c in chunks:
            ttnn.deallocate(chain(tz, c, norm=False, hidden_mem=L1, out_mem=DRAM, act=None))

    def leg_D():
        ttnn.deallocate(chain(tz, x, norm=False, hidden_mem=DRAM, out_mem=DRAM))

    def leg_E_chunk():
        for c in ttnn.chunk(x, nchunk, dim=1):
            ttnn.deallocate(c)

    def leg_E_cat():
        ttnn.deallocate(ttnn.concat(out_chunks, dim=1))

    legs = {"A": leg_A, "B": leg_B, "C": leg_C, "C_ln": leg_C_ln, "D": leg_D,
            "C_strict": leg_C_strict, "C_nosilu": leg_C_nosilu,
            "E_chunk": leg_E_chunk, "E_cat": leg_E_cat}
    order = [k for k in args.legs.split(",") if k in legs]

    for k in order:  # warm every leg before any leg is timed
        for _ in range(args.warm):
            try:
                legs[k]()
            except Exception as e:
                print(f"  {k}: WARM ERR {str(e)[:160]}", flush=True)
                break
        ttnn.synchronize_device(dev)

    samples = collections.defaultdict(list)
    for r in range(args.rounds):
        for k in order:  # INTERLEAVED: one sample of every leg per round
            try:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                legs[k]()
                ttnn.synchronize_device(dev)
                samples[k].append((time.perf_counter() - t0) * 1e3)
            except Exception as e:
                print(f"  {k}: ERR {str(e)[:160]}", flush=True)
                samples[k].append(float("nan"))
        print(f"  round {r+1}/{args.rounds}: " +
              " ".join(f"{k}={samples[k][-1]:.3f}" for k in order), flush=True)

    res = {}
    print(f"\n=== screen at [1,{N},{N},{c_z}], h={args.h}, {nchunk} chunks, median of "
          f"{args.rounds} interleaved rounds ===", flush=True)
    for k in order:
        v = [s for s in samples[k] if s == s]
        if not v:
            continue
        res[k] = {"ms_median": round(st.median(v), 4), "ms_min": round(min(v), 4),
                  "ms_max": round(max(v), 4), "n": len(v)}
        print(f"  {k:8s} {st.median(v):9.3f} ms   (min {min(v):.3f} max {max(v):.3f})", flush=True)

    if "A" in res and "C" in res:
        a, c = res["A"]["ms_median"], res["C"]["ms_median"]
        print(f"\nLEG_A_PROD {a:.3f} ms/call   LEG_C_ORACLE_FLOOR {c:.3f} ms/call   "
              f"deletable {a - c:.3f} ms/call", flush=True)
    if args.out:
        json.dump({"shape": [1, N, N, c_z], "h": args.h, "chunks": nchunk,
                   "core_grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                   "ttnn": getattr(ttnn, "__version__", "?"),
                   "rounds": args.rounds, "warm": args.warm, "legs": res,
                   "samples": {k: [round(s, 4) for s in v] for k, v in samples.items()}},
                  open(args.out, "w"), indent=2)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
