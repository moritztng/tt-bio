"""How many z-sized passes our PairformerLayer makes, per sub-unit, in their unit.

Anthropic denominate their fusion argument in passes over the full pair tensor: one unit is a full
read or a full write of `[N,N,c_z]`, and the counts they publish are 12 (starting node) / 14
(ending node) for the stock triangle-attention epilogue against 3-4 for their kernel
(`kernels/fpf_triatt_epi/epilogue.py:28-31`), and 18 units over 7 launches for stock cuEq trimul
against 14 over 5 for theirs (`kernels/fpf_trimul/trimul.py:233-241`). At 512 aa, c_z=128, bf16 one
unit is 512*512*128*2 = 67,108,864 B.

This counts the same quantity on our side off the buffer-keyed ledger `capture.py` arms on one
trunk PairformerLayer call inside a real fold.

METHOD, and the two places it can go wrong:

  BUFFER, NOT TENSOR. Every event is keyed by (device address, generation), so the reshapes and
  slices ttnn hands a fresh tensor id over one allocation are one buffer, not many. Keying on the
  tensor id inflated this same block by 1.83x once already
  (`ttnn-graph-byte-count-must-dedupe-buffer-not-tensor-id`).

  EVENTS, NOT LIFETIMES. A pass is one read or one write, so a buffer read three times is three
  passes. That is the opposite of `census.py`'s redundancy rule, which counts a re-read as
  redundant rather than as traffic; both are right for their own question. The ARITY / GATED rules
  are shared with it and imported rather than restated -- `generic_op` takes its output buffers in
  the operand list, and `reblock_permute_gated` reads half of its input.

Byte-weighted, as theirs is: a pass over a buffer that is not exactly z-sized counts
bytes / 67,108,864, so the trimul's `[N,N,4*c_z]` fused projection is 4 units and the attention's
32-channel bias is 0.25.

Usage:  python3 perf/anthro_zpass/zpass.py [block_512.json.gz] [--json out.json]
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "perf", "b2z2_byte_floor"))
from census import NON_OPS, VIEWS, split_io  # noqa: E402

# Their epilogue boundary, ours. Their kernel starts at the attention output and ends at the
# residual: sigmoid, o*g, the reshape copy, linear_o, the residual add and (ending node only) the
# transpose. These are the tt_bio functions that do those things.
EPILOGUE_FNS = {"gate_and_project", "out_proj"}
CORE_FNS = {"attend", "_attend_heads", "_tri_att_sdpa", "sdpa", "_softmax_over_tokens"}
# `_pair_transpose` runs on BOTH sides of the ending node: once to read z with the token axes
# swapped (their prologue does that with a stride swap) and once to put the result back (their
# "ending transpose 2"). Position decides which, so it is resolved against the core, not by name.
TRANSPOSE_FNS = {"_pair_transpose", "_pair_transpose_impl"}


def subunit(tags):
    """(sub-unit, stage) for one program, from the unit stack `capture.py` prepends."""
    u = [t[2:] for t in tags if t.startswith("u:")]
    try:
        i = next(i for i, x in enumerate(u) if x.startswith("PairformerLayer#"))
    except StopIteration:
        return "outside-block", ""
    return (u[i + 1] if len(u) > i + 1 else "residual"), ""


def stage(tags, owner, sub, seen_core):
    """prologue / core / epilogue for a triangle-attention program.

    Matched against the WHOLE tt_bio call chain, not the innermost frame: the output projection's
    innermost frame is `mm_generic.py:generic_minimal_matmul`, which is shared with the prologue's
    fused qkv matmul, so an owner-only test files the epilogue's matmul under the prologue.
    """
    if not sub.startswith("TriangleAttention"):
        return ""
    fns = {x.split(":")[-1] for x in tags if not x.startswith("u:")}
    fns.add(owner.rsplit(":", 1)[-1])
    if fns & TRANSPOSE_FNS:
        return "epilogue" if seen_core else "prologue"
    if fns & EPILOGUE_FNS:
        return "epilogue"
    if fns & CORE_FNS:
        return "core"
    return "prologue"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace", nargs="?",
                   default=os.path.join(HERE, "out", "c1", "block_512.json.gz"))
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--c-z", type=int, default=128)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    tr = json.load(gzip.open(a.trace, "rt"))
    rows, bufs = tr["rows"], tr["buffers"]
    Z = a.tokens * a.tokens * a.c_z * 2
    pair_axes = str(a.tokens)

    acc = defaultdict(lambda: Counter())
    core_seen = defaultdict(bool)
    launches = Counter()
    big = Counter()
    for r in rows:
        if r["op"] in NON_OPS:
            continue
        reads, writes = split_io(r)
        if r["op"] in VIEWS and set(writes) & {b for b, _ in reads}:
            continue
        sub, _ = subunit(r["tags"])
        st = stage(r["tags"], r["owner"], sub, core_seen[sub])
        if st == "core":
            core_seen[sub] = True
        key = (sub, st)
        launches[key] += 1
        for b, f in reads:
            v = bufs[b]
            pairish = v["shape"].split("x").count(pair_axes) >= 2
            acc[key][f"{v['where'].lower()}_rd"] += f * v["bytes"] / Z
            if pairish:
                acc[key][f"{v['where'].lower()}_rd_pair"] += f * v["bytes"] / Z
        for b in writes:
            v = bufs[b]
            pairish = v["shape"].split("x").count(pair_axes) >= 2
            acc[key][f"{v['where'].lower()}_wr"] += v["bytes"] / Z
            if pairish:
                acc[key][f"{v['where'].lower()}_wr_pair"] += v["bytes"] / Z
        for b in {x for x, _ in reads} | set(writes):
            if bufs[b]["bytes"] >= Z:
                big[bufs[b]["shape"]] += 1

    order = sorted(acc, key=lambda k: -(acc[k]["dram_rd"] + acc[k]["dram_wr"]))
    print("=" * 100)
    print(f"Z-SIZED PASSES, one trunk PairformerLayer at {a.tokens} aa, c_z={a.c_z}, bf16 "
          f"(1 unit = {Z:,} B) — {tr.get('arch')}, block call {tr.get('block_call')}")
    print("=" * 100)
    print(f"{'sub-unit':<26}{'stage':<10}{'prog':>5}{'DRAM rd':>9}{'DRAM wr':>9}{'DRAM':>8}"
          f"{'of which pair':>14}{'L1 rd':>8}{'L1 wr':>8}")
    tot = Counter()
    for k in order:
        v = acc[k]
        d = v["dram_rd"] + v["dram_wr"]
        pr = v["dram_rd_pair"] + v["dram_wr_pair"]
        print(f"{k[0]:<26}{k[1]:<10}{launches[k]:>5}{v['dram_rd']:>9.2f}{v['dram_wr']:>9.2f}"
              f"{d:>8.2f}{pr:>14.2f}{v['l1_rd']:>8.2f}{v['l1_wr']:>8.2f}")
        for f in v:
            tot[f] += v[f]
        tot["prog"] += launches[k]
    print("-" * 100)
    d = tot["dram_rd"] + tot["dram_wr"]
    print(f"{'BLOCK TOTAL':<26}{'':<10}{tot['prog']:>5}{tot['dram_rd']:>9.2f}{tot['dram_wr']:>9.2f}"
          f"{d:>8.2f}{tot['dram_rd_pair'] + tot['dram_wr_pair']:>14.2f}"
          f"{tot['l1_rd']:>8.2f}{tot['l1_wr']:>8.2f}")
    print(f"\nDRAM bytes implied: {d * Z / 1e9:.4f} GB per block "
          f"(L1 {(tot['l1_rd'] + tot['l1_wr']) * Z / 1e9:.4f} GB)")

    # The two comparisons that make the counts mean something.
    def agg(pred):
        s = Counter()
        n = 0
        for k, v in acc.items():
            if pred(k):
                for f in v:
                    s[f] += v[f]
                n += launches[k]
        return s["dram_rd"] + s["dram_wr"], n

    tm0, l0 = agg(lambda k: k[0] == "TriangleMultiplication#0")
    tm1, l1 = agg(lambda k: k[0] == "TriangleMultiplication#1")
    res, lres = agg(lambda k: k[0] == "residual")
    print("\nAGAINST THEIR NUMBERS")
    print(f"  trimul outgoing        {tm0:6.2f} units over {l0:3d} launches   "
          f"(theirs: 14 over 5; stock cuEq 18 over 7)")
    print(f"  trimul incoming        {tm1:6.2f} units over {l1:3d} launches")
    for tag, node in (("TriangleAttention#0", "starting"), ("TriangleAttention#1", "ending")):
        for st, their in (("prologue", None), ("core", None), ("epilogue", "12 / 14 stock, 3-4 theirs")):
            u, n = agg(lambda k, t=tag, s=st: k == (t, s))
            extra = f"   (theirs: {their})" if their else ""
            print(f"  triatt {node:<8} {st:<9}{u:6.2f} units over {n:3d} launches{extra}")
    print(f"  residual chain         {res:6.2f} units over {lres:3d} launches   "
          f"(their epilogue absorbs this: 3 of their stock 12)")

    if a.json:
        json.dump({"tokens": a.tokens, "c_z": a.c_z, "z_unit_bytes": Z, "arch": tr.get("arch"),
                   "block_call": tr.get("block_call"),
                   "per_subunit": {f"{k[0]}|{k[1]}": dict(v, prog=launches[k])
                                   for k, v in acc.items()},
                   "total": dict(tot), "big_buffers": dict(big)},
                  open(a.json, "w"), indent=1)
        print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
