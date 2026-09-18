"""Every operand of every program in one PairformerLayer, keyed by BUFFER, with its owner.

The campaign already has a byte counter (`perf/b2z2_tile_census/census_tiles.py`, carried into
`perf/b2z2_redteam2/src/`) and it is the one that produced the 8.0493 GB this row exists to attack.
This does not replace it. It supplies the two columns that counter cannot have, because it counts
from a profiler CSV that carries shapes and no identities:

  OWNER      which `tt_bio` call site issued the program
  BUFFER     the device address the operand lives at, so "the same tensor read twice" is a fact
             about one allocation and not about two tensors that happen to share a shape

Keying on the buffer address and not on the tensor object is not a detail: ttnn hands every
reshape/slice/unsqueeze a fresh tensor id over the same allocation, and a counter that dedupes by
id inflated this model's block by 1.83x once already (memory
`ttnn-graph-byte-count-must-dedupe-buffer-not-tensor-id`).

Addresses are recycled, so an address alone is not an identity. A generation counter advances
whenever an address appears as an op's OUTPUT without also being one of its inputs -- that is a
fresh allocation written from scratch. An in-place op (`ttnn.add_`) keeps the generation, which is
what we want: the residual chain is one buffer read and written many times, not many buffers.

Three layer shapes, one tracer. `--layer pairformer` builds a standalone PairformerLayer with the
Boltz-2 trunk geometry, runs it twice (the first call compiles) and records the second.
`--layer difftx` cannot do that: the diffusion layers take a pair bias sliced per layer, an atom
window index and an AdaLN memo that only exist inside a real rollout, so that mode folds the 512 aa
fixture with a short rollout and arms the recorder on one call of each DiffusionTransformerLayer
shape, which is the same call `perf/b2x_difflayer/capture_difftx.py` hands to `ttnn.graph`. Same
recorder, same ledger, same output schema; only the driver differs.

Usage:  TT_VISIBLE_DEVICES=<n> TT_BIO_LEASE_CARDS=<n> python3 trace_block.py [--layer pairformer]
                                                                             [--tokens 512] [--out P]
        TT_VISIBLE_DEVICES=<n> TT_BIO_LEASE_CARDS=<n> python3 trace_block.py --layer difftx
                                                                             [--steps 4] [--call 3]
                                                                             --out-dir D
A path ending in `.gz` is written gzipped, which is what `fusion_pairs.py` reads.

The recorder (`install`, `ledger`, `dump`, `_owner`, `_stack_tags`) is importable: argument parsing
and the driver dispatch sit under `__main__`, so another row can arm the same ledger on a call of
its own choosing without forking this file. `perf/anthro_zpass/capture.py` does exactly that.
"""
import argparse
import gzip
import json
import os
import sys
import traceback
from collections import defaultdict

import torch


def _args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layer", choices=("pairformer", "difftx"), default="pairformer")
    p.add_argument("--tokens", type=int, default=512, help="pairformer mode: sequence length")
    p.add_argument("--out", default=None, help="pairformer mode: output path")
    p.add_argument("--size", type=int, default=512, help="difftx mode: fixture length")
    p.add_argument("--steps", type=int, default=4, help="difftx mode: sampling steps")
    p.add_argument("--call", type=int, default=3,
                   help="difftx mode: which call of each layer shape to record, 0-based")
    p.add_argument("--out-dir", default=None, help="difftx mode: directory for one file per shape")
    return p.parse_args()


import ttnn  # noqa: E402
from ttnn.decorators import FastOperation, Operation  # noqa: E402

TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088, "UINT32": 4096,
              "UINT16": 2048, "INT32": 4096, "UINT8": 1024, "FLOAT16": 2048}

REC = []
ACTIVE = [False]
DEPTH = [0]


def _tiles(t):
    sh = list(t.padded_shape)
    if len(sh) < 2:
        return 0
    lead = 1
    for d in sh[:-2]:
        lead *= max(int(d), 1)
    return lead * ((int(sh[-2]) + 31) // 32) * ((int(sh[-1]) + 31) // 32)


def _tinfo(t):
    """(address, bytes, 'DRAM'|'L1', dtype, shape) for an allocated device tensor, else None."""
    try:
        if not isinstance(t, ttnn.Tensor):
            return None
        if t.storage_type() != ttnn.StorageType.DEVICE:
            return None
        addr = int(t.buffer_address())
        dt = str(t.dtype).split(".")[-1].upper()
        bt = t.memory_config().buffer_type
        where = "DRAM" if bt == ttnn.BufferType.DRAM else "L1"
        return {"addr": addr, "bytes": _tiles(t) * TILE_BYTES.get(dt, 2048), "where": where,
                "dtype": dt, "shape": "x".join(str(int(d)) for d in t.padded_shape)}
    except Exception:
        return None


def _walk(o, out, seen, depth=0):
    if depth > 3:
        return
    i = _tinfo(o)
    if i is not None:
        out.append(i)
        return
    if isinstance(o, (list, tuple)):
        for e in o:
            _walk(e, out, seen, depth + 1)
    elif isinstance(o, dict):
        for e in o.values():
            _walk(e, out, seen, depth + 1)


_HERE = os.path.dirname(os.path.abspath(__file__))


def _owner():
    """Innermost tt_bio frame, as file:line:function. That is the call site that owns the bytes."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if os.sep + "tt_bio" + os.sep in fr.filename:
            return f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}"
    return "-"


def _stack_tags():
    """The tt_bio call chain, outermost model module first, for grouping by sub-unit."""
    tags = []
    for fr in traceback.extract_stack()[:-2]:
        if os.sep + "tt_bio" + os.sep in fr.filename:
            tags.append(f"{os.path.basename(fr.filename)}:{fr.name}")
    return tags


def _wrap(name, fn):
    def w(*a, **kw):
        if not ACTIVE[0] or DEPTH[0]:
            return fn(*a, **kw)
        ins = []
        _walk(a, ins, None)
        _walk(kw, ins, None)
        owner, tags = _owner(), _stack_tags()
        DEPTH[0] += 1
        try:
            r = fn(*a, **kw)
        finally:
            DEPTH[0] -= 1
        outs = []
        _walk(r, outs, None)
        REC.append({"op": name, "owner": owner, "tags": tags, "in": ins, "out": outs})
        return r
    w.__name__ = getattr(fn, "__name__", name)
    return w


def install():
    """Wrap every `ttnn` Operation, wherever it lives, and return how many.

    The module list used to be hand-written (`ttnn`, `ttnn.experimental`, and one level under
    `ttnn.operations`). That missed `ttnn.transformer`, which is where
    `scaled_dot_product_attention` lives -- so a trace of any stack that reaches the fused SDPA
    recorded its q/k/v as written-and-never-read and dropped the op's own mask read entirely.
    Walking the tree instead of naming it is the fix, and it is the same reason a hardcoded list of
    anything in this repo eventually goes stale.
    """
    n, seen = 0, set()

    def walk(prefix, m, depth):
        nonlocal n
        if id(m) in seen or depth > 3:
            return
        seen.add(id(m))
        for a in dir(m):
            if a.startswith("_"):
                continue
            try:
                o = getattr(m, a)
            except Exception:
                continue
            if isinstance(o, (FastOperation, Operation)):
                try:
                    setattr(m, a, _wrap(f"{prefix}.{a}", o))
                    n += 1
                except Exception:
                    pass
            elif type(o).__name__ == "module" and getattr(o, "__name__", "").startswith("ttnn"):
                walk(f"{prefix}.{a}", o, depth + 1)

    walk("ttnn", ttnn, 0)
    return n


def ledger(rec):
    """Buffer-keyed read/write ledger over an ordered op stream.

    generation: an address that appears as an OUTPUT and was not one of the op's own inputs is a
    fresh allocation. Anything else on that address is the same buffer.
    """
    gen, out_rows, buf = {}, [], {}
    for i, r in enumerate(rec):
        in_addrs = {d["addr"] for d in r["in"]}
        keys_in, keys_out = [], []
        for d in r["in"]:
            k = (d["addr"], gen.get(d["addr"], 0))
            keys_in.append(k)
            b = buf.setdefault(k, {"bytes": d["bytes"], "where": d["where"], "dtype": d["dtype"],
                                   "shape": d["shape"], "readers": [], "writers": []})
            b["bytes"] = max(b["bytes"], d["bytes"])
            b["readers"].append(i)
        for d in r["out"]:
            if d["addr"] not in in_addrs:
                gen[d["addr"]] = gen.get(d["addr"], 0) + 1
            k = (d["addr"], gen.get(d["addr"], 0))
            keys_out.append(k)
            b = buf.setdefault(k, {"bytes": d["bytes"], "where": d["where"], "dtype": d["dtype"],
                                   "shape": d["shape"], "readers": [], "writers": []})
            b["bytes"] = max(b["bytes"], d["bytes"])
            b["writers"].append(i)
        out_rows.append({**{k: r[k] for k in ("op", "owner", "tags")},
                         "in": [f"{a}#{g}" for a, g in keys_in],
                         "out": [f"{a}#{g}" for a, g in keys_out],
                         "in_b": [d["bytes"] for d in r["in"]],
                         "in_w": [d["where"] for d in r["in"]],
                         "in_s": [d["shape"] for d in r["in"]],
                         "out_b": [d["bytes"] for d in r["out"]],
                         "out_w": [d["where"] for d in r["out"]],
                         "out_s": [d["shape"] for d in r["out"]]})
    return out_rows, {f"{a}#{g}": v for (a, g), v in buf.items()}


def dump(path, arch, tokens, rec, extra=None):
    """Ledger `rec` and write it where `fusion_pairs.py` / `census.py` can read it."""
    rows, bufs = ledger(rec)
    blob = {"arch": arch, "tokens": tokens, "n_ops": len(rows), "rows": rows, "buffers": bufs}
    blob.update(extra or {})
    op = gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")
    with op as fh:
        json.dump(blob, fh)
    print(f"{len(rows)} ops, {len(bufs)} buffers -> {path}", flush=True)
    return {"path": path, "n_ops": len(rows), "n_buffers": len(bufs)}


def _arch():
    return str(ttnn.get_arch_name()) if hasattr(ttnn, "get_arch_name") else "?"


def main_pairformer():
    from tt_bio import tenstorrent as tt
    from tt_bio import reference as ref
    dev = tt.get_device()
    KC = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    rl = ref.PairformerLayer(384, 128, 16, 0.25, 32, 4, v2=True)
    weights = {k: v.float() for k, v in rl.state_dict().items()}
    layer = tt.PairformerLayer(32, 4, 24, 16, True, weights, KC)
    S = TOKENS
    s = torch.randn(1, S, 384)
    z = torch.randn(1, S, S, 128)
    m1 = torch.ones(1, S)
    pair_mask = m1[:, :, None] * m1[:, None, :]
    attn = (1 - m1).unsqueeze(1).unsqueeze(1) * -1e9
    f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mask_tt, attn_tt = f(pair_mask), f(attn)

    s_tt, z_tt = f(s), f(z)
    layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)     # compile
    ttnn.synchronize_device(dev)
    ttnn.deallocate(z_tt); ttnn.deallocate(s_tt)

    n = install()
    print(f"wrapped {n} ttnn operations", flush=True)
    s_tt, z_tt = f(s), f(z)
    ACTIVE[0] = True
    layer(s_tt, z_tt, mask_tt, attn_tt, attn_tt)
    ACTIVE[0] = False
    ttnn.synchronize_device(dev)

    elig = {}
    for k in ("qkv_heads", "gate_proj", "out_proj"):
        st = getattr(__import__("tt_bio.triatt_qkv", fromlist=["x"]), "STATS", None)
        if isinstance(st, dict):
            elig = dict(st)
    dump(OUT, _arch(), TOKENS, REC, {"triatt_qkv_stats": elig})


# ---------------------------------------------------------------------------------------------
# difftx: the two diffusion layer shapes, recorded inside a real rollout.

def _sig(args):
    """`difftx|<shape of a>,<shape of s>` -- the key `capture_difftx.py` already dumps under, so a
    tagged trace and a graph capture of the same layer call carry the same name."""
    sh = [x for x in args if getattr(x, "shape", None) is not None]
    return "difftx|" + ",".join("x".join(str(int(d)) for d in x.shape) for x in sh[:2])


def main_difftx():
    out_dir = ARGS.out_dir or "/tmp"
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(os.path.dirname(_HERE))
    for extra in (root, os.path.join(root, "scripts", "gpu_vs_tt"),
                  os.path.join(root, "perf", "other512")):
        sys.path.insert(0, extra)

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import fold_ab_multi as FAM
    from tt_bio.main import _resolve_recycling_steps

    B.SAMPLING_STEPS = ARGS.steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    FAM.patch_boltz2_cfg()

    fix = os.path.join(root, "perf", "size512", "fixtures")
    from pathlib import Path
    one_fold, meta, state = B.build_fold(
        "boltz2", Path(root) / f".msa_bytes_{ARGS.size}",
        Path(fix) / f"cdk2x2_{ARGS.size}.yaml", Path(fix) / f"cdk2x2_{ARGS.size}.a3m")
    dev = T.get_device()
    arch = _arch()

    n = install()
    print(f"wrapped {n} ttnn operations", flush=True)

    seen, done = defaultdict(int), {}
    orig = T.DiffusionTransformerLayer.__call__
    # The flags that decide which attention the token DiT runs and how the atom axis is bucketed.
    # Recorded beside the bytes because the published graph capture was taken with both OFF and a
    # trace that does not say which arm it is cannot be compared with it.
    flags = {"BOLTZ2_TOKEN_DIT_SDPA": bool(T._B2_TOKEN_DIT_SDPA),
             "TT_BIO_ATOM_AXIS_BUCKET": bool(T._ATOM_AXIS_BUCKET)}

    def call(self, *a, **kw):
        sig = _sig(a)
        i = seen[sig]
        seen[sig] = i + 1
        if i != ARGS.call or sig in done or ACTIVE[0]:
            return orig(self, *a, **kw)
        REC.clear()
        ACTIVE[0] = True
        try:
            r = orig(self, *a, **kw)
        finally:
            ACTIVE[0] = False
        ttnn.synchronize_device(dev)
        name = sig.split("|", 1)[1].replace(",", "_")
        done[sig] = dump(os.path.join(out_dir, f"difftx_{name}.json.gz"), arch, ARGS.size, REC,
                         {"sig": sig, "call_index": i, "atom_level": bool(self.atom_level),
                          "steps": ARGS.steps, "flags": flags})
        return r

    T.DiffusionTransformerLayer.__call__ = call
    fold_s, m = one_fold()
    T.DiffusionTransformerLayer.__call__ = orig
    # fold_s is NOT a performance number: every ttnn call in this process is wrapped and two layer
    # calls walk the Python stack per op. It is printed only so a run that folded nothing is obvious.
    print(f"folded (instrumented, not a timing) {fold_s:.1f} s, plddt {m.get('plddt')}", flush=True)
    print("shapes recorded: " + ", ".join(sorted(done)), flush=True)
    print("flags: " + json.dumps(flags), flush=True)
    json.dump({"flags": flags, "steps": ARGS.steps, "size": ARGS.size, "arch": arch,
               "call_index": ARGS.call, "recorded": done,
               "token_dit_sdpa_stats": list(T.B2_TOKEN_DIT_SDPA_STATS)},
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    T.cleanup()


if __name__ == "__main__":
    ARGS = _args()
    TOKENS = ARGS.tokens
    OUT = ARGS.out or f"/tmp/b2z2_bytefloor_trace_{TOKENS}.json"
    (main_difftx if ARGS.layer == "difftx" else main_pairformer)()
