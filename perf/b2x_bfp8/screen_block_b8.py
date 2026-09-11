#!/usr/bin/env python3
"""One Boltz-2 pairformer block at 512 aa, bf16 pair track against bfloat8_b.

The screen for bet P1: does the block's TIME track its BYTES? bfloat8_b changes the payload per
transaction (2048 -> 1088 B a tile) and not the transaction count, so a path bound by page-read
latency shows the byte cut and none of the time. Falsifier (b) of the bet is exactly that
reading, and it is worth more to the campaign than a hopeful partial.

Real layer-0 weights off the shipped checkpoint, the production shapes, one device open. Per
arm: wall time for the block and for each of its five sub-modules, the graph-capture byte
breakdown, and a census of every `ttnn.typecast` the arm inserted -- the way this bet fails
quietly is a conversion either side of an op that will not take a block-format operand.
"""
from __future__ import annotations

import argparse, json, os, socket, statistics as st, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def graph_bytes(g):
    """DRAM bytes a captured region reads and writes, plus L1, plus the op census.

    Same counter `perf/bioir_roofline/fold_bytes_512.py` produced the campaign's 12.170 GB/block
    with, so the two numbers are comparable.
    """
    out = {"dram_read": 0, "dram_write": 0, "l1_write": 0, "n_ops": 0, "n_tensors": 0}
    ops = Counter()
    seen = set()
    for n in g:
        t = n.get("node_type")
        p = n.get("params") or {}
        if t == "function_start":
            name = str(p.get("name", ""))
            if name.startswith("ttnn."):
                out["n_ops"] += 1
                ops[name] += 1
        elif t == "tensor":
            tid = p.get("tensor_id")
            if tid in seen:
                continue
            seen.add(tid)
            out["n_tensors"] += 1
            if "DRAM" in str(p.get("buffer_type", "")):
                out["dram_read"] += int(p.get("size", 0) or 0)
        elif t == "buffer_allocate":
            if str(p.get("type")) == "DRAM":
                out["dram_write"] += int(p.get("size", 0) or 0)
            else:
                out["l1_write"] += int(p.get("size", 0) or 0)
    out["dram_total"] = out["dram_read"] + out["dram_write"]
    out["ops"] = dict(ops.most_common())
    out["n_typecast"] = ops.get("ttnn.typecast", 0)
    return out


def find_pairformer(model, n_blocks):
    """The trunk `PairformerModule` with `n_blocks` blocks, wherever boltz2 hangs it."""
    import torch.nn as nn
    from tt_bio.tenstorrent import PairformerModule
    for name, m in model.named_modules():
        if isinstance(m, PairformerModule) and m.n_blocks == n_blocks and m.module is not None:
            return name, m
    raise LookupError(f"no PairformerModule with {n_blocks} blocks and a built module")


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    OUT_PATH = args.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()), "seq": args.seq, "reps": args.reps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    cfg = dict(model="boltz2", fast=False, output_format="cif", recycling_steps=3,
               sampling_steps=200, diffusion_samples=1, seed=0, trace=False,
               use_msa_server=False, single_sequence=True)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 2)
    dump()

    name, pf_mod = find_pairformer(state.model, 64)
    OUT["pairformer_path"] = name
    blk = pf_mod.module.blocks[0]
    dump()

    S, C_Z, C_S = args.seq, 128, 384
    torch.manual_seed(0)
    z_t = (torch.randn(1, S, S, C_Z) * 0.35).bfloat16().float()
    z_t = 0.5 * (z_t + z_t.transpose(1, 2))          # z is near-symmetric in a real trunk
    s_t = (torch.randn(1, S, C_S) * 0.6).bfloat16().float()
    mask_t = torch.ones(1, S, S)
    attn_t = torch.zeros(1, 1, 1, S)

    def tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)

    mask = tt(mask_t)
    attn = tt(attn_t)

    SUB = [("trimul_start", "triangle_multiplication_start"),
           ("trimul_end", "triangle_multiplication_end"),
           ("triatt_start", "triangle_attention_start"),
           ("triatt_end", "triangle_attention_end"),
           ("transition_z", "transition_z")]

    def run_block(z, s):
        """The layer's z path, op for op as `PairformerLayer.__call__` runs it."""
        s2, z2 = blk(s, z, mask, attn, attn)
        return s2, z2

    def time_sub(z, sub):
        """One sub-module on a fresh copy of z, timed and byte-counted on its own."""
        op = getattr(blk, sub)
        arg = (z, mask) if sub.startswith("triangle_multiplication") else (
            (z, attn) if sub.startswith("triangle_attention") else (z,))
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        out = op(*arg)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t
        ttnn.deallocate(out)
        return dt

    def arm(pair_b8: bool):
        T._PAIR_B8 = pair_b8
        # Every tuned-width cache the trimul keeps is keyed on shape, not dtype, so an arm flip
        # must not reuse the other arm's fused input weights uploaded at the other dtype.
        for tm in (blk.triangle_multiplication_start, blk.triangle_multiplication_end):
            tm._gp_cache.clear(); tm._gp_bias_cache.clear()
        res = {"pair_b8": pair_b8}

        z0 = tt(z_t); s0 = tt(s_t)
        # warm: first call compiles every program at this shape and dtype
        s1, z1 = run_block(z0, s0)
        ttnn.deallocate(z1)
        if s1 is not None:
            ttnn.deallocate(s1)

        walls = []
        for _ in range(args.reps):
            z0 = tt(z_t); s0 = tt(s_t)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            s1, z1 = run_block(z0, s0)
            ttnn.synchronize_device(dev)
            walls.append(time.perf_counter() - t)
            ttnn.deallocate(z1)
            if s1 is not None:
                ttnn.deallocate(s1)
        res["block_s"] = [round(w, 5) for w in walls]
        res["block_median_s"] = round(st.median(walls), 5)

        # per sub-module, on a z in this arm's storage dtype
        zc = tt(z_t, ttnn.bfloat8_b if pair_b8 else ttnn.bfloat16)
        res["z_dtype"] = str(zc.dtype)
        subs = {}
        for label, sub in SUB:
            time_sub(zc, sub)                                # warm this shape+dtype
            subs[label] = round(st.median([time_sub(zc, sub) for _ in range(3)]), 5)
        res["sub_median_s"] = subs
        res["sub_sum_s"] = round(sum(subs.values()), 5)

        # bytes + op census for one whole block
        z0 = tt(z_t); s0 = tt(s_t)
        ttnn.synchronize_device(dev)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        s1, z1 = run_block(z0, s0)
        ttnn.synchronize_device(dev)
        res["graph"] = graph_bytes(ttnn.graph.end_graph_capture())
        ttnn.deallocate(z1)
        if s1 is not None:
            ttnn.deallocate(s1)
        ttnn.deallocate(zc)
        res["gb_per_block"] = round(res["graph"]["dram_total"] / 1e9, 4)
        res["achieved_gbps"] = round(res["graph"]["dram_total"] / 1e9
                                     / res["block_median_s"], 1)
        return res

    # Arms alternated so session drift cannot be read as an effect, and the incumbent is run
    # twice as its own A/A floor.
    order = [False, True, False, True]
    arms = []
    for i, b8 in enumerate(order):
        a = arm(b8)
        a["order"] = i
        arms.append(a)
        OUT["arms"] = arms
        dump()
        print(f"[{i}] pair_b8={b8}  block {a['block_median_s']*1e3:.2f} ms  "
              f"{a['gb_per_block']:.3f} GB  {a['achieved_gbps']} GB/s  "
              f"typecasts {a['graph']['n_typecast']}", flush=True)

    b16 = [a for a in arms if not a["pair_b8"]]
    b8 = [a for a in arms if a["pair_b8"]]
    t16 = st.median([a["block_median_s"] for a in b16])
    t8 = st.median([a["block_median_s"] for a in b8])
    B16 = st.median([a["graph"]["dram_total"] for a in b16])
    B8 = st.median([a["graph"]["dram_total"] for a in b8])
    aa = abs(b16[0]["block_median_s"] - b16[1]["block_median_s"]) / t16
    OUT["summary"] = {
        "aa_floor_rel": round(aa, 5),
        "block_s_b16": round(t16, 5), "block_s_b8": round(t8, 5),
        "time_cut": round(1 - t8 / t16, 4),
        "gb_b16": round(B16 / 1e9, 4), "gb_b8": round(B8 / 1e9, 4),
        "byte_cut": round(1 - B8 / B16, 4),
        "gbps_b16": round(B16 / 1e9 / t16, 1), "gbps_b8": round(B8 / 1e9 / t8, 1),
        "typecast_b16": b16[0]["graph"]["n_typecast"],
        "typecast_b8": b8[0]["graph"]["n_typecast"],
        "speedup": round(t16 / t8, 4),
    }
    dump()
    print(json.dumps(OUT["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
