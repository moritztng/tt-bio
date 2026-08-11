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
argument and of the output, the measured seconds, the innermost tt_bio call site and the tt_bio
frame chain above it (so a row can be traced to its submodule as well as its line).

    TT_VISIBLE_DEVICES=0 python3 perf/ledger_298/pf_block_ops.py \
        --model protenix-v2 --n 320 --out perf/ledger_298/ops_pv2_320.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import PairformerLayer, get_device, set_fast_mode  # noqa: E402
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

            # bench() holds `reps` extra outputs live beside the block, so a wide output OOMs L1
            # and the row lands at 0.0 with its time absorbed by the classes that did time. Nine
            # rows were in that state at 320 aa, five of them matmuls including the largest in the
            # fold, and four fusion go/no-go verdicts were built on the sum that absorbed them
            # (moonshot-4x-k256-kernel-rate.md 10b, pairformer-resident-chunking.md 70). At 512 aa
            # the outputs are 2.6x larger, so fall back down the ladder rather than leave a zero.
            err, dt, used = None, 0.0, 0
            for r in sorted({r for r in (STATE["reps"], 2, 1) if r <= STATE["reps"]},
                            reverse=True):
                try:
                    dt = bench(r)
                    if dt * 1e6 < STATE["small_us"]:
                        dt = bench(r * 8)
                    err, used = None, r
                    break
                except Exception as e:                           # noqa: BLE001
                    # An op we cannot safely re-run even once is still worth a row: shapes are
                    # recorded, time is not, and the row is excluded from the sum rather than
                    # silently guessed.
                    err, dt = f"{type(e).__name__}: {e}"[:200], 0.0
                    ttnn.synchronize_device(dev)
            first = out[0] if isinstance(out, (list, tuple)) and out else out
            RECORDS.append({"i": STATE["idx"], "op": name, "site": site, "chain": chain, "s": dt,
                            "in": ins, "out": desc(first) if isinstance(first, ttnn.Tensor) else None,
                            "reps_used": used,
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
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--small-us", type=float, default=60.0)
    ap.add_argument("--fast", action="store_true",
                    help="census the --fast configuration: bfloat8_b chunks and the 640-token "
                         "trimul L1 window. --fast is banked at 1.2207x and it is spent on the same "
                         "data movement a fused program would remove, so a default-mode census "
                         "measures a prize that is partly already banked.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    if args.fast:
        set_fast_mode(True)
    assert T._FAST_MODE is args.fast, T._FAST_MODE
    layer, c_z = build(args.model, ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"model={args.model} c_z={c_z} N={N}", flush=True)

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

    tot = sum(r["s"] for r in RECORDS)
    print(f"ops={len(RECORDS)}  sum={tot * 1e3:.3f} ms  block={block_s * 1e3:.3f} ms  "
          f"coverage={100 * tot / block_s:.1f}%", flush=True)
    untimed = [r for r in RECORDS if "error" in r]
    if untimed:
        print(f"UNTIMED {len(untimed)} rows -- the sum is not sound, see pairformer-resident-chunking.md 70",
              flush=True)
        for r in untimed:
            print(f"  {r['op']} @ {r['site']} out={r['out']} :: {r['error']}", flush=True)
    json.dump({"model": args.model, "n": N, "c_z": c_z, "block_wall_s": block_s,
               "reps": args.reps, "fast": args.fast, "n_ops": len(RECORDS), "sum_s": tot,
               "n_untimed": len(untimed), "loadavg": open("/proc/loadavg").read().split()[:3],
               "fatal": fatal, "records": RECORDS}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
