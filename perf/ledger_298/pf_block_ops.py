#!/usr/bin/env python3
"""Per-op device time + shapes for one real Pairformer block, measured in place.

Why not the device profiler: the ttnn 0.67.4 wheel on qb1 ships `tracy` the python package but not
the tracy capture binaries, so `python3 -m tracy -r` dies with "Tracy tools were not found" and
there is no ops report to post-process. This measures the same thing a different way.

Every ttnn op the block calls is wrapped. The wrapper lets the model's own call through untouched,
then re-runs that exact call `--reps` more times back to back with one synchronise on each side of
the repeated region, so the per-call time is a device time with the host dispatch amortised rather
than a per-call host round trip. Calls that come out under `--small-us` are re-timed with 8x the
reps, because at that size a single dispatch is a large fraction of the measurement. The extra
outputs are freed by refcount, never by ttnn.deallocate -- see bench() for why that matters.

Recorded per call: op name, call index, the padded shape / dtype / buffer type of every tensor
argument and of the output, the measured seconds (`null` if the re-run failed), the innermost tt_bio
call site and the tt_bio frame chain above it (so a row can be traced to its submodule as well as
its line).

The summary is four numbers and never a single coverage percentage: block wall, per-op sum, ops
DROPPED because they could not be re-run, and the genuinely UNATTRIBUTED remainder. Those last two
are different things and adding them together is how 62 dropped rows once read as inter-op overlap.

    TT_VISIBLE_DEVICES=0 python3 perf/ledger_298/pf_block_ops.py \
        --model protenix-v2 --tokens 298 --out perf/ledger_298/ops_pv2_298.json
"""
import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.tenstorrent import PairformerLayer, get_device  # noqa: E402
from tt_bio import protenix_weights as PW  # noqa: E402

TRI_HEAD_DIM = 32

OPS = ["linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "layer_norm",
       "rms_norm", "softmax", "permute", "transpose", "concat", "reshape", "typecast",
       "to_layout", "to_memory_config", "clone", "relu", "sigmoid", "silu", "gelu", "sum",
       "reciprocal", "pad", "chunk", "unsqueeze", "squeeze", "repeat", "slice"]
# `reallocate` is deliberately NOT wrapped: re-running it moves the model's own buffers.
EXP_OPS = ["minimal_matmul", "nlp_create_qkv_heads", "nlp_concat_heads"]
TF_OPS = ["scaled_dot_product_attention"]

RECORDS = []
STATE = {"on": False, "dev": None, "reps": 4, "small_us": 60.0, "idx": 0}


def desc(t):
    try:
        mc = t.memory_config()
        buf = str(mc.buffer_type).split(".")[-1]
        lay = str(mc.memory_layout).split(".")[-1]
        return {"shape": list(t.padded_shape), "logical": list(t.shape),
                "dtype": str(t.dtype).split(".")[-1], "buf": buf, "layout": lay}
    except Exception:                                            # noqa: BLE001
        return None


DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "uint32": 4, "int32": 4, "uint16": 2, "uint8": 1,
               "bfloat8_b": 1, "bfloat4_b": 1}


def nbytes(rec):
    """Padded bytes the op's tensors carry -- what a dropped row still tells us."""
    ts = list(rec.get("in") or [])
    if rec.get("out"):
        ts.append(rec["out"])
    return sum(math.prod(t["shape"]) * DTYPE_BYTES.get(t["dtype"], 2) for t in ts)


def tensor_args(a, kw):
    out = []
    for v in list(a) + list(kw.values()):
        d = desc(v) if isinstance(v, ttnn.Tensor) else None
        if d:
            out.append(d)
    return out


def call_site():
    """(innermost tt_bio line, the tt_bio frame chain above it).

    Filter this script's own frames out by filename. Slicing a fixed number of frames off the end
    silently ate the real call site and reported the enclosing PairformerLayer line instead.
    """
    chain = [f"{fr.filename.split('/')[-1]}:{fr.lineno}"
             for fr in reversed(traceback.extract_stack())
             if "tt_bio/" in fr.filename and __file__ not in fr.filename]
    return (chain[0] if chain else "?"), chain[:4]


def wrap(name, fn):
    def inner(*a, **kw):
        if not STATE["on"]:
            return fn(*a, **kw)
        site, chain = call_site()
        ins = tensor_args(a, kw)
        STATE["on"] = False                       # nested ttnn calls must not re-enter
        try:
            out = fn(*a, **kw)
            dev = STATE["dev"]
            keep = {id(x) for x in list(a) + list(kw.values()) if isinstance(x, ttnn.Tensor)}
            for o in (out if isinstance(out, (list, tuple)) else [out]):
                keep.add(id(o))

            def bench(reps):
                # The extra outputs are freed by refcount when `extra` goes out of scope. Do NOT
                # call ttnn.deallocate on them: reshape/permute/unsqueeze/to_layout hand back a
                # tensor that SHARES the input's buffer, and deallocating that view frees a buffer
                # the model is still holding -- that is a segfault, and it is how this script died
                # the first time.
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                extra = [fn(*a, **kw) for _ in range(reps)]
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
                del extra
                return dt

            err = None
            try:
                dt = bench(STATE["reps"])
                if dt * 1e6 < STATE["small_us"]:
                    dt = bench(STATE["reps"] * 8)
            except Exception as e:                               # noqa: BLE001
                # An op we cannot safely re-run is still worth a row: shapes are recorded, time is
                # `null`, and the row is DROPPED from the sum. It used to be recorded as 0.0 s with
                # the reason in `error`, which no consumer filtered on, so a dropout was summed as
                # a free op and the coverage line then read as inter-op overlap. 62 of 272 rows on
                # qb2 card 0 at --reps 4. `null` cannot be silently added up.
                err, dt = f"{type(e).__name__}: {e}"[:200], None
                ttnn.synchronize_device(dev)
            first = out[0] if isinstance(out, (list, tuple)) and out else out
            RECORDS.append({"i": STATE["idx"], "op": name, "site": site, "chain": chain, "s": dt,
                            "in": ins, "out": desc(first) if isinstance(first, ttnn.Tensor) else None,
                            **({"error": err} if err else {})})
            STATE["idx"] += 1
            return out
        finally:
            STATE["on"] = True
    return inner


def patch():
    saved = []
    for ns, names in ((ttnn, OPS), (ttnn.experimental, EXP_OPS), (ttnn.transformer, TF_OPS)):
        for nm in names:
            fn = getattr(ns, nm, None)
            if fn is None or not callable(fn):
                continue
            saved.append((ns, nm, fn))
            setattr(ns, nm, wrap(nm, fn))
    return saved


def build(model, ckc):
    if model == "protenix-v2":
        path, prefix = "/home/ttuser/.boltz/protenix-v2.pt", "pairformer_stack.blocks.0."
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("aurekaresearch/OpenDDE", "opendde.pt")
        prefix = "pairformer_stack.blocks.0."
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not blk:
        pfx = sorted({k.split(".blocks.")[0] for k in sd if ".blocks." in k})
        raise SystemExit(f"no keys under {prefix!r}; candidate stacks: {pfx[:20]}")
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True, remapped, ckc), c_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["protenix-v2", "opendde"], default="protenix-v2")
    ap.add_argument("--tokens", type=int, default=298,
                    help="real token count. TILE_LAYOUT pads only the last two dims, so a pair "
                         "tensor built [1, tokens, tokens, c_z] carries the shape a fold carries: "
                         "the column axis pads to a tile multiple and the row axis stays at "
                         "`tokens`. At 298 aa that is [1, 298, 320, c_z], 48.82 MB.")
    ap.add_argument("--n", type=int, default=0,
                    help="build the square block [1, n, n, c_z] instead. This is what the harness "
                         "did before and it is 320/298 = 1.074x heavy on the row axis at 298 aa, "
                         "so every byte model derived from it is 7.4 %% high. Kept for comparison.")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--small-us", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build(args.model, ckc)
    N = args.n or args.tokens
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"model={args.model} c_z={c_z} tokens={N} z padded={list(z.padded_shape)} "
          f"({math.prod(z.padded_shape) * 2 / 1e6:.2f} MB)", flush=True)

    for _ in range(args.warm):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)

    # Block wall time with nothing patched: the denominator every per-op sum is checked against.
    t0 = time.perf_counter()
    for _ in range(3):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)
    block_s = (time.perf_counter() - t0) / 3
    print(f"block wall = {block_s * 1e3:.3f} ms", flush=True)

    saved = patch()
    STATE.update(dev=dev, reps=args.reps, small_us=args.small_us, on=True)
    fatal = None
    try:
        s, z = layer(s, z)
        ttnn.synchronize_device(dev)
    except Exception as e:                                       # noqa: BLE001
        fatal = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()
    STATE["on"] = False
    for ns, nm, fn in saved:
        setattr(ns, nm, fn)
    if fatal:
        print(f"PARTIAL PASS, died after {len(RECORDS)} ops: {fatal}", flush=True)

    # Four numbers, not one coverage percentage. A dropped op and an unattributed remainder have
    # nothing to do with each other: the first is instrument failure, the second is real block time
    # no row accounts for. Reporting their sum as "coverage" is what made 62 dropped rows read as
    # inter-op overlap.
    timed_recs = [r for r in RECORDS if r["s"] is not None]
    dropped = [r for r in RECORDS if r["s"] is None]
    tot = sum(r["s"] for r in timed_recs)
    drop_b = sum(nbytes(r) for r in dropped)
    unattr = block_s - tot
    print(f"ops={len(RECORDS)}  timed={len(timed_recs)}  block wall={block_s * 1e3:.3f} ms  "
          f"per-op sum={tot * 1e3:.3f} ms  dropped={len(dropped)} ops "
          f"({drop_b / 1e6:.2f} MB moved, time unknown)  "
          f"unattributed={unattr * 1e3:.3f} ms ({100 * unattr / block_s:.1f}% of the block wall)",
          flush=True)
    if dropped:
        by_site = defaultdict(int)
        for r in dropped:
            by_site[(r["op"], r["site"])] += 1
        for (op, site), n in sorted(by_site.items(), key=lambda kv: -kv[1]):
            print(f"  dropped x{n:<3} {op} @ {site}", flush=True)
    json.dump({"model": args.model, "n": N, "tokens": N, "c_z": c_z, "block_wall_s": block_s,
               "z_padded_shape": list(z.padded_shape),
               "reps": args.reps, "n_ops": len(RECORDS), "n_timed": len(timed_recs),
               "n_dropped": len(dropped), "dropped_bytes": drop_b,
               "sum_s": tot, "unattributed_s": unattr, "fatal": fatal,
               "records": RECORDS}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
