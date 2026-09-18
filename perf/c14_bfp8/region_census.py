#!/usr/bin/env python3
"""Which pair-scale buffers a bfp8 region could actually cover, read off a real capture.

The question this answers is the one the row lives or dies on, and it needs no card: for every
pair-scale DRAM buffer one trunk `PairformerLayer` moves at 512 aa, can its PRODUCER emit bfp8 and
can EVERY consumer take a bfp8 operand? A buffer whose producer cannot emit the format needs a
`typecast` (1.53125 Z at this key against the 0.9375 Z a narrowed pass saves, i.e. a debit), and a
buffer with one refusing consumer needs a cast back, which is the same debit on the other side.

Op dtype coverage is read from the INSTALLED ttnn wheel docs, not from memory, and printed with the
table so the basis is visible:

  * `ttnn.layer_norm`  -- takes BFLOAT8_B input but has NO output-dtype argument: "Output tensor
    will be in TILE layout and have the same dtype as the input_tensor". So it can only PASS the
    format through, never introduce it.
  * `ttnn.multiply` / `ttnn.multiply_` -- BFLOAT16, FLOAT32, UINT16. **No BFLOAT8_B at all.**
  * `ttnn.add` / `ttnn.add_`, `ttnn.silu`, `ttnn.linear`, `ttnn.matmul`,
    `ttnn.experimental.minimal_matmul` -- BFLOAT8_B supported, and the matmuls take an output
    `dtype=`, so they can introduce the format for free.
  * our own fused kernels (triatt_sdpa, triatt_qkv, trimul_tail, mm_dualnoc) decline non-bf16 on
    OUR line only; their CB page sizes come from `tile_bytes(dtype)` and their kernels take tile
    bytes from `get_tile_size(cb)`, so they are widenable. Marked `ours` so the table separates
    what we can fix from what we cannot.
"""
import argparse, collections, gzip, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b2x_difflayer"))
from itemize import itemize  # noqa: E402

Z = 67108864
B8 = 0.53125                      # 1088 B a tile against bf16 2048

# can this op EMIT bfp8 for the buffer it allocates?
EMIT = {
    "ttnn.linear": "yes", "ttnn.matmul": "yes",
    "ttnn.experimental.minimal_matmul": "yes",
    "ttnn.layer_norm": "no-out-dtype",          # output dtype == input dtype, upstream
    "ttnn.multiply": "no-kernel", "ttnn.multiply_": "no-kernel",
    "ttnn.add": "yes", "ttnn.add_": "yes", "ttnn.silu": "yes",
    "ttnn.clone": "yes", "ttnn.typecast": "yes",
    "ttnn.allocate_tensor_on_device": "ours",   # our fused kernels own dest, hardcoded bf16
    "ttnn.concat": "inherits", "ttnn.permute": "lossy",    # transpose_wh re-derives exponents
    "ttnn.transpose": "lossy", "ttnn.reallocate": "inherits",
    "ttnn.to_memory_config": "inherits", "ttnn.experimental.view": "inherits",
    # our four fused kernels all dispatch through ttnn.generic_op, so this is the label our own
    # bf16 gates wear in a capture. Every one of their CB page sizes comes from tile_bytes(dtype)
    # and every tile size in their kernels from get_tile_size(cb), so they are ours to widen.
    "ttnn.generic_op": "ours",
    "ttnn.reshape": "inherits", "ttnn.unsqueeze": "inherits", "ttnn.squeeze": "inherits",
    # a last-axis slice of a large DRAM tensor already wedges on Blackhole at bf16; at c_z = 128
    # each chunk is exactly one tile wide, and a bfp8 tile cannot be split at all without
    # re-deriving its shared exponents.
    "ttnn.chunk": "risky", "ttnn.slice": "risky",
}
# can this op CONSUME a bfp8 operand?
TAKE = dict(EMIT)
TAKE.update({"ttnn.layer_norm": "yes",                     # bfp8 input is supported
             "ttnn.allocate_tensor_on_device": "ours"})
GOOD = {"yes", "inherits"}


def load(p):
    d = json.load(gzip.open(p, "rt") if str(p).endswith(".gz") else open(p))
    return d["nodes"] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--min-mb", type=float, default=60.0)
    ap.add_argument("--calls", type=int, default=264, help="trunk PairformerLayer calls a fold")
    ap.add_argument("--rate", type=float, default=364.84, help="GB/s, and say where it came from")
    ap.add_argument("--out", default=str(HERE / "region_census.json"))
    a = ap.parse_args()

    ops, rows = itemize({"nodes": load(a.capture)})
    dram = [r for r in rows if "DRAM" in r["kind"] and r["size"] >= a.min_mb * 1e6]

    table, tot = [], collections.Counter()
    for r in sorted(dram, key=lambda r: -r["size"]):
        prod = r["alloc_op"] or "(pre-existing)"
        emit = EMIT.get(prod, "unknown") if r["alloc_op"] else "pre-existing"
        takes = [(n, TAKE.get(n, "unknown")) for n in r["consumer_names"]]
        blockers = sorted({v for _, v in takes if v not in GOOD})
        ok = emit in GOOD and not blockers
        # a narrowed buffer deletes 0.46875 of its write and of every read of it
        passes = (1 if r["alloc_op"] else 0) + max(len(r["consumer_names"]), 1 if not r["alloc_op"] else 0)
        deleted = r["size"] * passes * (1 - B8)
        row = {"buffer": r["buffer"], "mb": r["size"] / 1e6, "producer": prod, "emit": emit,
               "consumers": r["consumer_names"], "consumer_verdicts": [v for _, v in takes],
               "blockers": blockers, "eligible": ok, "passes": passes,
               "deleted_Z": deleted / Z}
        table.append(row)
        # the pair tensor z itself is pre-existing and is the accumulator: narrowing it IS the
        # arm b2x already measured at 0.949x and 1.4965 A, so it is excluded from every region
        # here rather than quietly carried in a total.
        bucket = ("accumulator_excluded_Z" if not r["alloc_op"]
                  else "eligible_today_Z" if ok
                  # fixable by us ONLY if every refusal on the buffer is one of our own gates
                  else "ours_to_fix_Z" if (emit in GOOD or emit == "ours")
                                          and set(blockers) <= {"ours"}
                  else "upstream_blocked_Z")
        row["bucket"] = bucket
        tot["all_Z"] += deleted / Z
        tot[bucket] += deleted / Z

    fold_gb = lambda z: z * a.calls * Z / 1e9                                    # noqa: E731
    out = {"capture": a.capture, "min_mb": a.min_mb, "calls": a.calls,
           "rate_gbs": a.rate, "buffers": table,
           "totals_Z_per_layer": dict(tot),
           "fold_gb": {k: fold_gb(v) for k, v in tot.items()},
           "fold_seconds_at_rate": {k: fold_gb(v) / a.rate for k, v in tot.items()}}
    Path(a.out).write_text(json.dumps(out, indent=2))

    print("%-6s %-34s %-13s %-13s %s" % ("MB", "producer", "emit", "deleted Z", "blockers"))
    for r in table:
        print("%6.1f %-34s %-13s %-13.4f %s  <- %s" % (
            r["mb"], r["producer"], r["emit"], r["deleted_Z"],
            ",".join(r["blockers"]) or "-", ",".join(r["consumers"]) or "(none)"))
    print()
    for k, v in tot.items():
        print("%-22s %8.3f Z/layer  %9.1f GB/fold  %7.4f s at %.2f GB/s"
              % (k, v, fold_gb(v), fold_gb(v) / a.rate, a.rate))


if __name__ == "__main__":
    main()
