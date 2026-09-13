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

Usage:  TT_VISIBLE_DEVICES=<n> TT_BIO_LEASE_CARDS=<n> python3 trace_block.py [tokens] [out.json]
Opens one device, builds one PairformerLayer with the Boltz-2 trunk geometry, runs it twice
(the first call compiles) and records the second.
"""
import json
import os
import sys
import traceback

import torch

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/b2z2_bytefloor_trace_{TOKENS}.json"

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
    n = 0
    mods = [("ttnn", ttnn)]
    for sub in ("experimental", "operations"):
        m = getattr(ttnn, sub, None)
        if m is not None:
            mods.append((f"ttnn.{sub}", m))
            for a in dir(m):
                s = getattr(m, a, None)
                if type(s).__name__ == "module" and not a.startswith("_"):
                    mods.append((f"ttnn.{sub}.{a}", s))
    seen = set()
    for pfx, m in mods:
        if id(m) in seen:
            continue
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
                    setattr(m, a, _wrap(f"{pfx}.{a}", o))
                    n += 1
                except Exception:
                    pass
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


def main():
    from tt_bio import tenstorrent as tt
    from tt_bio import reference as ref
    dev = tt.get_device()
    arch = str(ttnn.get_arch_name()) if hasattr(ttnn, "get_arch_name") else "?"
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

    rows, bufs = ledger(REC)
    elig = {}
    for k in ("qkv_heads", "gate_proj", "out_proj"):
        st = getattr(__import__("tt_bio.triatt_qkv", fromlist=["x"]), "STATS", None)
        if isinstance(st, dict):
            elig = dict(st)
    json.dump({"arch": arch, "tokens": S, "n_ops": len(rows), "rows": rows, "buffers": bufs,
               "triatt_qkv_stats": elig}, open(OUT, "w"))
    print(f"{len(rows)} ops, {len(bufs)} buffers -> {OUT}", flush=True)


main()
