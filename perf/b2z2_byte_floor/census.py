"""The Pairformer block's bytes, by site, with an OWNER and a REDUNDANCY column.

Inputs, both committed:
  art/ops_perf_blocksum_qb2c0.csv.gz  the Blackhole capture the campaign's 8.0493 GB comes from
  out/trace_512_wh_c10.json.gz        this row's buffer-keyed trace of the same block (WH card 10)

The BH capture supplies the bytes in the currency every other row quotes. The trace supplies what a
shape-keyed capture cannot have: which `tt_bio` call site issued each program, and which device
ALLOCATION each operand lives in. A byte is redundant when two programs read the same allocation
with no write in between, so one read would have served both.

Three rules keep this from inventing redundancy that is not there, each of which cost a wrong
number before it was written down:

  ARITY      `ttnn.generic_op` is called `[*inputs, *outputs]` everywhere in tt-bio, so the
             pre-allocated output buffers sit in the input list. Counting them as reads turns the
             fused qkv kernel's own q/k/v into 201 MB of phantom re-reads. Arities are read off the
             call sites: mm_generic.py 2, sdpa_generic.py 4, reblock_permute.py 1, trimul_tail.py 4.
  GATED      `reblock_permute_gated`'s reader walks two of its input's four channel slices
             (`b2z2-byte-axis-reopened`, verified against the kernel's reader). Two gated calls over
             one fused projection are ONE full read between them, not two.
  IN-PLACE   the residual chain is `z = ttnn.add_(z, z_update)`, which keeps the buffer and changes
             its contents. Five sub-units reading "the same allocation" are reading five different
             tensors. Redundancy is therefore counted over RUNS of consecutive reads with no
             intervening write, never over a buffer's lifetime.
"""
import gzip
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

NON_OPS = {"ttnn.deallocate", "ttnn.allocate_tensor_on_device", "ttnn.to_torch",
           "ttnn.from_torch", "ttnn.synchronize_device", "ttnn.chunk"}
VIEWS = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.to_layout",
         "ttnn.to_memory_config", "ttnn.reallocate"}
# generic_op operand-list arity, by the file that issues the call. Read off the call sites.
GENERIC_ARITY = {"mm_generic.py": 2, "sdpa_generic.py": 4, "reblock_permute.py": 1,
                 "trimul_tail.py": 4}
GATED_FRACTION = 0.5


def split_io(r):
    """(reads, writes) as (buffer, fraction) lists, with generic_op's output slots moved across."""
    ins, outs = list(dict.fromkeys(r["in"])), list(dict.fromkeys(r["out"]))
    if r["op"] == "ttnn.generic_op":
        f = r["owner"].split(":")[0]
        n = GENERIC_ARITY.get(f, len(ins))
        for b in ins[n:]:
            if b not in outs:
                outs.append(b)
        ins = ins[:n]
    frac = GATED_FRACTION if "reblock_permute_gated" in r["owner"] else 1.0
    outs_s = set(outs)
    return [(b, frac) for b in ins if b not in outs_s], outs


def unit(r):
    """The model sub-unit that owns the program, from the tt_bio call chain."""
    fns = [t for t in r["tags"]]
    for t in reversed(fns):
        f, n = t.split(":", 1)
        if n in ("__call__",) and f == "tenstorrent.py":
            continue
    # the chain is outermost-first; the second tenstorrent.py:__call__ is the sub-unit
    calls = [i for i, t in enumerate(fns) if t == "tenstorrent.py:__call__"]
    return fns[calls[1]] if len(calls) > 1 else (fns[0] if fns else "-")


def main():
    # argv: the trace to read and where to write the per-buffer ledger. Defaults are this row's
    # own arm, so `python3 census.py` still reproduces the published table.
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "trace_512_wh_c10.json.gz")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out", "census.json")
    tr = json.load(gzip.open(src, "rt"))
    rows, bufs = tr["rows"], tr["buffers"]

    # per-buffer event stream, in program order
    ev = defaultdict(list)          # buf -> [(i, 'r'|'w', frac)]
    for i, r in enumerate(rows):
        if r["op"] in NON_OPS:
            continue
        reads, writes = split_io(r)
        if r["op"] in VIEWS and set(writes) & {b for b, _ in reads}:
            continue
        for b, f in reads:
            ev[b].append((i, "r", f))
        for b in writes:
            ev[b].append((i, "w", 1.0))

    per_buf, dram_rd, l1_rd, dram_wr, l1_wr, red = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for b, es in ev.items():
        v = bufs[b]
        nb = v["bytes"]
        runs, cur = [], []
        for i, kind, f in sorted(es):
            if kind == "w":
                if cur:
                    runs.append(cur)
                    cur = []
            else:
                cur.append((i, f))
        if cur:
            runs.append(cur)
        rd_eff = sum(f for run in runs for _, f in run)
        wr_n = sum(1 for _, k, _ in es if k == "w")
        rdb = rd_eff * nb
        redb = sum(max(0.0, sum(f for _, f in run) - 1.0) for run in runs) * nb
        if v["where"] == "DRAM":
            dram_rd += rdb
            dram_wr += wr_n * nb
            red += redb
        else:
            l1_rd += rdb
            l1_wr += wr_n * nb
        per_buf.append({
            "buf": b, "bytes": nb, "where": v["where"], "shape": v["shape"],
            "reads": rd_eff, "writes": wr_n, "redundant_b": redb,
            "runs": [[f"{rows[i]['op'].replace('ttnn.','')}@{rows[i]['owner']}" for i, _ in run]
                     for run in runs if len(run) > 1],
            "owner": unit(rows[es[0][0]]),
        })
    per_buf.sort(key=lambda x: -x["redundant_b"])

    tot = dram_rd + dram_wr
    print("=" * 96)
    print("BUFFER-KEYED BYTE LEDGER — one PairformerLayer, 512 tokens, WH card 10, "
          f"commit {os.popen('git rev-parse --short HEAD').read().strip()}")
    print("=" * 96)
    print(f"  DRAM read              {dram_rd/1e6:9.1f} MB      (BH census: 4710.8)")
    print(f"  DRAM write             {dram_wr/1e6:9.1f} MB      (BH census: 3338.5)")
    print(f"  DRAM total             {tot/1e9:9.4f} GB      (BH census: 8.0493)")
    print(f"  L1 read / write        {l1_rd/1e6:9.1f} / {l1_wr/1e6:.1f} MB")
    print(f"  REDUNDANT DRAM reads   {red/1e6:9.1f} MB  = {100*red/tot:.2f} % of the block's traffic")
    print()
    print(f"{'allocation':<13}{'MB':>7}{'loc':>5}{'rd':>5}{'wr':>4}  {'shape':<16}{'redund MB':>10}"
          f"  {'owner':<22}")
    for x in per_buf:
        if x["redundant_b"] <= 0:
            continue
        print(f"{x['buf']:<13}{x['bytes']/1e6:>7.1f}{x['where']:>5}{x['reads']:>5.1f}"
              f"{x['writes']:>4}  {x['shape']:<16}{x['redundant_b']/1e6:>10.1f}  {x['owner']:<22}")
        for run in x["runs"]:
            print(f"{'':>18}shared read: " + " | ".join(run))
    json.dump({"per_buf": per_buf,
               "totals": {"dram_rd": dram_rd, "dram_wr": dram_wr, "l1_rd": l1_rd,
                          "l1_wr": l1_wr, "redundant_dram_rd": red}},
              open(dst, "w"), indent=1)


main()
