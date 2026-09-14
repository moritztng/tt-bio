"""Rank the Pairformer block's single-use DRAM intermediates by the bytes fusion would delete.

`perf/b2z2_byte_floor/CENSUS.md` established that 58.8 % of the block's traffic is single-use DRAM
intermediates -- 63 allocations written by one program and read by exactly one other -- and then
stopped, because "delete them" is program fusion and the Pairformer megakernel lost 2.9 %. That
verdict is about fusing EVERYTHING. It says nothing about which individual pairs are worth fusing,
and nothing in the archive ranks them. This does.

One allocation is a candidate when, in program order, it is written exactly once and read exactly
once, with the read after the write and nothing else touching it. Fusing that producer into that
consumer deletes the write and the read: 2 x its bytes off the DRAM ledger.

THE RULES ARE NOT RE-DERIVED. `split_io`, `NON_OPS` and `VIEWS` are imported out of
`perf/b2z2_byte_floor/census.py` so there is one copy of the arity/gated/in-place rule in the repo
and this table cannot drift from the census it extends (`verification-instrument-drift-is-shared-
code-drift`). census.py runs `main()` at import, so it is loaded with that call stripped.

FEASIBILITY is a separate column and it is the honest half. A byte count says what fusion would
save; it does not say the fusion is available. The classification is by the consumer's access
pattern, because that is what decides whether the consumer can take the producer's tiles in the
order the producer makes them:

  EPILOGUE  consumer is elementwise/unary/binary over the same shape -> the producer writes its
            output tile and the consumer's math runs on it in DST before it is ever packed out.
  PROLOGUE  producer is elementwise and consumer is a matmul reading it as an operand -> the
            consumer's reader does the producer's math on the tile it just loaded.
  REORDER   consumer transposes, permutes, reshapes across tile boundaries, or reduces over an axis
            the producer does not stream -- the tile order does not match and fusion needs a
            materialised intermediate anyway. NOT free.
  UNKNOWN   not classifiable from the op code alone. Reported, never counted as available.

Only EPILOGUE and PROLOGUE bytes are summed into the available prize. REORDER and UNKNOWN are
printed so the next pass can attack the classification rather than trust it.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CENSUS = os.path.join(ROOT, "perf", "b2z2_byte_floor", "census.py")


def load_census_rules():
    """census.py with its trailing `main()` call stripped, so importing it does not run it."""
    src = open(CENSUS).read()
    assert src.rstrip().endswith("main()"), "census.py no longer ends in main(); check the strip"
    src = src.rstrip()[: -len("main()")]
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("census_rules", loader=None))
    mod.__file__ = CENSUS
    exec(compile(src, CENSUS, "exec"), mod.__dict__)
    return mod


C = load_census_rules()

# Consumer op -> how it can take its operand. Anything absent is UNKNOWN and is not counted.
ELEMENTWISE = {
    "ttnn.add", "ttnn.add_", "ttnn.multiply", "ttnn.multiply_", "ttnn.subtract", "ttnn.sub",
    "ttnn.div", "ttnn.divide", "ttnn.sigmoid", "ttnn.silu", "ttnn.gelu", "ttnn.relu",
    "ttnn.softmax", "ttnn.layer_norm", "ttnn.rms_norm", "ttnn.mul", "ttnn.neg", "ttnn.exp",
    "ttnn.tanh", "ttnn.clamp", "ttnn.where", "ttnn.typecast", "ttnn.copy", "ttnn.assign",
}
MATMUL = {"ttnn.matmul", "ttnn.linear", "ttnn.bmm"}
REORDER = {"ttnn.transpose", "ttnn.permute", "ttnn.concat", "ttnn.split", "ttnn.slice",
           "ttnn.pad", "ttnn.repeat", "ttnn.repeat_interleave", "ttnn.tilize", "ttnn.untilize",
           "ttnn.sum", "ttnn.mean", "ttnn.max", "ttnn.min", "ttnn.argmax",
           "ttnn.to_memory_config", "ttnn.reshape",
           "ttnn.experimental.nlp_create_qkv_heads", "experimental.nlp_create_qkv_heads",
           "ttnn.transformer.scaled_dot_product_attention"}


# `ttnn.generic_op` is four different kernels in tt-bio and the op code alone cannot tell them
# apart, which left 1218.4 MB unclassified on the first run. Resolve it by the file that issues the
# call -- the same key `census.py:GENERIC_ARITY` already uses, so the two agree by construction.
GENERIC_KIND = {"mm_generic.py": "matmul", "sdpa_generic.py": "attention",
                "reblock_permute.py": "reorder", "trimul_tail.py": "elementwise"}


def kind(op: str, site: str) -> str:
    """The access pattern class of one program: matmul | attention | elementwise | reorder."""
    if op == "ttnn.generic_op":
        return GENERIC_KIND.get(site.split(":")[0], "unknown")
    if op in REORDER:
        return "reorder"
    if op in MATMUL:
        return "matmul"
    if op in ELEMENTWISE:
        return "elementwise"
    return "unknown"


def classify(prod_op: str, cons_op: str, prod_site: str = "", cons_site: str = "") -> str:
    p, c = kind(prod_op, prod_site), kind(cons_op, cons_site)
    if "unknown" in (p, c):
        return "UNKNOWN"
    if "reorder" in (p, c):
        # a channel move or a transpose does not stream its input in its output's tile order
        return "REORDER"
    if c == "elementwise":
        return "EPILOGUE"                    # consumer's math runs in DST before the pack
    if c in ("matmul", "attention") and p == "elementwise":
        return "PROLOGUE"                    # consumer's reader does the producer's math on load
    if c == "attention" and p == "matmul":
        # q/k/v projection feeding SDPA. Q streams once and fuses; K and V are re-read once per
        # query block, so folding their projection in trades DRAM traffic for recompute.
        return "ATTN-QKV"
    if c == "matmul" and p in ("matmul", "attention"):
        return "REORDER"                     # the second reduces over the first's output axis
    return "UNKNOWN"


def short(owner: str) -> str:
    """`tenstorrent.py:5595:__call__` -> `tenstorrent.py:5595`."""
    p = owner.split(":")
    return ":".join(p[:2]) if len(p) > 1 else owner


def site(r) -> str:
    """The model function that issued the program, from the tt_bio call chain.

    The trace's `owner` carries a LINE NUMBER from the commit it was captured at, which no longer
    resolves against the tip. The call chain's function names do, so the table is keyed on those
    and the line number is kept only as a tie-break.
    """
    fns = [t.split(":", 1)[1] for t in r["tags"]]
    fns = [f for f in fns if f != "__call__"]
    return fns[-1] if fns else short(r["owner"])


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "perf", "b2z2_byte_floor", "out", "trace_512_wh_c10.json.gz")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fusion_pairs.json")
    tr = json.load(gzip.open(src, "rt"))
    rows, bufs = tr["rows"], tr["buffers"]

    ev = defaultdict(list)
    for i, r in enumerate(rows):
        if r["op"] in C.NON_OPS:
            continue
        reads, writes = C.split_io(r)
        if r["op"] in C.VIEWS and set(writes) & {b for b, _ in reads}:
            continue
        for b, f in reads:
            ev[b].append((i, "r", f))
        for b in writes:
            ev[b].append((i, "w", 1.0))

    cands, dram_total = [], 0.0
    for b, es in ev.items():
        v = bufs[b]
        nb = v["bytes"]
        es = sorted(es)
        n_w = sum(1 for _, k, _ in es if k == "w")
        n_r = sum(1 for _, k, _ in es if k == "r")
        if v["where"] == "DRAM":
            dram_total += sum(f for _, k, f in es if k == "r") * nb + n_w * nb
        if v["where"] != "DRAM" or n_w != 1 or n_r != 1:
            continue
        (wi, _, _), (ri, _, rf) = (es[0], es[1]) if es[0][1] == "w" else (es[1], es[0])
        if wi > ri:                      # read before write: not a producer/consumer pair
            continue
        p, c = rows[wi], rows[ri]
        cands.append({
            "buf": b, "bytes": nb, "shape": v["shape"],
            "producer": p["op"], "producer_site": site(p),
            "consumer": c["op"], "consumer_site": site(c),
            "read_frac": rf,
            "class": classify(p["op"], c["op"], short(p["owner"]), short(c["owner"])),
            "producer_line": short(p["owner"]), "consumer_line": short(c["owner"]),
            "removable_b": nb * (1.0 + rf),   # the write, plus however much of it is read
        })

    by_pair = defaultdict(lambda: {"n": 0, "removable_b": 0.0, "bytes": 0.0})
    for x in cands:
        k = (x["producer_site"], x["producer"], x["consumer_site"], x["consumer"], x["class"])
        g = by_pair[k]
        g["n"] += 1
        g["removable_b"] += x["removable_b"]
        g["bytes"] += x["bytes"]
    ranked = sorted(({"producer_site": k[0], "producer": k[1], "consumer_site": k[2],
                      "consumer": k[3], "class": k[4], **v} for k, v in by_pair.items()),
                    key=lambda x: -x["removable_b"])

    avail = sum(x["removable_b"] for x in ranked if x["class"] in ("EPILOGUE", "PROLOGUE"))
    attn = sum(x["removable_b"] for x in ranked if x["class"] == "ATTN-QKV")
    blocked = sum(x["removable_b"] for x in ranked if x["class"] == "REORDER")
    unknown = sum(x["removable_b"] for x in ranked if x["class"] == "UNKNOWN")
    tot = avail + attn + blocked + unknown

    print("=" * 108)
    print("SINGLE-USE DRAM INTERMEDIATES OF ONE PairformerLayer, BY PRODUCER->CONSUMER PAIR")
    print(f"512 tokens, {tr['arch']}, {os.path.basename(src)}; rules imported from census.py")
    print("=" * 108)
    print(f"  block DRAM traffic (this trace)      {dram_total/1e6:9.1f} MB")
    print(f"  single-use intermediates             {len(cands):4d} allocations, "
          f"{tot/1e6:.1f} MB of read+write")
    print(f"    fusable as epilogue/prologue       {avail/1e6:9.1f} MB   "
          f"= {100*avail/dram_total:.2f} % of the block")
    print(f"    qkv projection into attention      {attn/1e6:9.1f} MB   "
          f"= {100*attn/dram_total:.2f} % of the block, part free part recompute")
    print(f"    blocked by tile reorder            {blocked/1e6:9.1f} MB")
    print(f"    unclassified                       {unknown/1e6:9.1f} MB")
    print()
    print(f"{'removable MB':>12} {'n':>4}  {'class':<9} {'producer':<44} -> {'consumer':<44}")
    for x in ranked:
        if x["removable_b"] / 1e6 < 1.0:
            continue
        print(f"{x['removable_b']/1e6:>12.1f} {x['n']:>4}  {x['class']:<9} "
              f"{x['producer'].replace('ttnn.','')+' @'+x['producer_site']:<44} -> "
              f"{x['consumer'].replace('ttnn.','')+' @'+x['consumer_site']:<44}")

    json.dump({"src": src, "arch": tr["arch"], "tokens": tr["tokens"],
               "block_dram_b": dram_total, "n_candidates": len(cands),
               "available_b": avail, "attn_qkv_b": attn, "blocked_b": blocked, "unknown_b": unknown,
               "pairs": ranked, "allocations": sorted(cands, key=lambda x: -x["removable_b"])},
              open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
