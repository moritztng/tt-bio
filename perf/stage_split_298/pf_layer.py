#!/usr/bin/env python3
"""One real Protenix-v2 trunk PairformerLayer, standalone: op count + per-block timing.

Step 1a/1b target of the 298-aa perf campaign. Loads layer-0 trunk weights from the
real v2 checkpoint (values are irrelevant to op counts and device time; real weights
kill any shape guessing), builds a single PairformerLayer with the production
HiFi4/fp32_dest_acc config, then:

  --mode count   ttnn.graph capture of ONE warm block call -> ttnn.* op tally.
                 Structural only; NEVER quote its durations (instrumented host wall).
  --mode bench   warm --warm calls, then --iters synced timed calls (per-block wall).

N is the PADDED token count: 298 aa pads to 320, 117 aa pads to 128
(PAIRFORMER_PAD_MULTIPLE=64). c_z: 256 protenix-v2, 384 opendde (same graph, wider
pair track; n_tri_heads = c_z // 32).

    TT_VISIBLE_DEVICES=3 python3 perf/stage_split_298/pf_layer.py --mode count --n 320
    TT_VISIBLE_DEVICES=3 python3 perf/stage_split_298/pf_layer.py --mode bench --n 320
"""

import argparse
import collections
import json
import os
import time

import torch

import ttnn
from tt_bio import protenix_weights as PW
from tt_bio.tenstorrent import PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"
if not os.path.exists(CKPT):
    CKPT = os.path.expanduser("~/.boltz/protenix-v2.pt")
TRI_HEAD_DIM = 32


def build_layer(ckc):
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {
        k[len("pairformer_stack.blocks.0."):]: v
        for k, v in sd.items()
        if k.startswith("pairformer_stack.blocks.0.")
    }
    remapped = PW.remap_pairformer_block(blk)
    # p_in.weight is cat([a_p, b_p]) = [2*c_z, c_z]; the in-features dim is c_z.
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(
        TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True, remapped, ckc
    ), c_z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["count", "bench", "ab_trimul"], required=True)
    ap.add_argument("--n", type=int, default=320, help="padded token count (298 aa -> 320, 117 aa -> 128)")
    ap.add_argument("--trimul-chunk", type=int, default=0,
                    help="if >0, override TRIANGLE_MULT_CHUNK_SIZE before building the layer (M3)")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "ab_trimul":
        import tt_bio.tenstorrent as T
        dev = get_device()
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        layers = {}
        for c in (32, 64, 128):
            T.TRIANGLE_MULT_CHUNK_SIZE = c
            layers[c] = build_layer(ckc)
        T.TRIANGLE_MULT_CHUNK_SIZE = 32
        N = args.n
        torch.manual_seed(0)
        for c, (layer, c_z) in layers.items():
            s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            for _ in range(args.warm):
                s, z = layer(s, z)
            layers[c] = (layer, c_z, s, z)
        ttnn.synchronize_device(dev)
        times = {c: [] for c in layers}
        for _ in range(args.iters):
            for c, (layer, c_z, s, z) in layers.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                s, z = layer(s, z)
                ttnn.synchronize_device(dev)
                times[c].append((time.perf_counter() - t0) * 1e3)
                layers[c] = (layer, c_z, s, z)
        for c, ts in sorted(times.items()):
            med = sorted(ts)[len(ts) // 2]
            print(f"TRIMUL_CHUNK {c}: median {med:.1f} ms series {[round(t, 1) for t in ts]}")
        return

    if args.trimul_chunk:
        import tt_bio.tenstorrent as T
        T.TRIANGLE_MULT_CHUNK_SIZE = args.trimul_chunk

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"layer built: c_z={c_z} tri_heads={c_z // TRI_HEAD_DIM} N={N}", flush=True)

    for _ in range(args.warm):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)

    if args.mode == "count":
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        s, z = layer(s, z)
        graph = ttnn.graph.end_graph_capture()
        ttnn.synchronize_device(dev)
        ops = [nd for nd in graph if nd.get("node_type") == "function_start"]
        tally = collections.Counter(
            (nd.get("params") or {}).get("name") for nd in ops
            if ((nd.get("params") or {}).get("name") or "").startswith("ttnn.")
        )
        total = sum(tally.values())
        print(f"TTNN_OPS total={total} nodes={len(graph)}")
        for name, cnt in tally.most_common():
            print(f"  {name}={cnt}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"n": N, "c_z": c_z, "ttnn_ops": total,
                           "nodes": len(graph), "tally": dict(tally)}, f, indent=2)
        return

    times = []
    for _ in range(args.iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    print(f"BLOCK_MS n={N} c_z={c_z} warm={args.warm} iters={args.iters} "
          f"min={min(times)*1e3:.1f} median={med*1e3:.1f} max={max(times)*1e3:.1f} "
          f"series={[round(t*1e3,1) for t in times]}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n": N, "c_z": c_z, "times_ms": [t * 1e3 for t in times],
                       "median_ms": med * 1e3}, f, indent=2)


if __name__ == "__main__":
    main()
