#!/usr/bin/env python3
"""Per-site DRAM byte ledger for one PairformerLayer, deduped on BUFFER ADDRESS not tensor id.

Reads a ttnn graph capture of one settled call (perf/b2x-baseline-attrib/baseline_attrib.py
--phases attrib writes them) and answers the question c14-byte-deletion was given: at each of the
five z residuals, where does the update tensor live, and how many full passes over the 67.109 MB
pair tensor does the site cost today. A pass is the unit that converts to seconds: one deleted pass
at every trunk PairformerLayer call is 264 x 67.109 MB = 17.717 GB, 0.0418 s at the add_ class own
in-fold 423.6 GB/s.

Byte convention is perf/b2x_difflayer/real_traffic.py own, whose itemize() this reuses, so the
totals are comparable with ROOF_RESIDUAL.md rather than being a second instrument: one write per
buffer the op allocated, one read per DRAM buffer it consumes, an in-place op charged a read and a
rewrite of its destination, and a metadata view (reshape/squeeze/unsqueeze/deallocate) charged
nothing. ttnn hands a reshape a fresh tensor id over the same buffer, which is why the dedupe basis
is the buffer node.
"""
import argparse, collections, gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2x_difflayer"))
from itemize import itemize  # noqa: E402

Z = 67108864
NO_TRAFFIC = {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze", "ttnn.deallocate"}


def load(p):
    nodes = json.load(gzip.open(p, "rt") if str(p).endswith(".gz") else open(p))
    return nodes["nodes"] if isinstance(nodes, dict) else nodes


def ledger(nodes):
    ops, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if "DRAM" in r["kind"]]
    alloc_bytes = collections.defaultdict(int)
    for r in dram:
        if r["alloc_op_i"] is not None:
            alloc_bytes[r["alloc_op_i"]] += r["size"]

    def moves(i):
        return ops[i]["name"] not in NO_TRAFFIC and (alloc_bytes[i] > 0
                                                     or ops[i]["name"].endswith("_"))
    w = collections.defaultdict(int)
    rd = collections.defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if moves(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc_bytes[i] == 0:
                w[i] += r["size"]                      # in-place: rewrites what it read
        if not readers and r["alloc_op_i"] is not None:
            rd[r["alloc_op_i"]] += r["size"]           # device-op scratch has no visible reader
    return ops, rows, w, rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pair-min-mb", type=float, default=60.0)
    a = ap.parse_args()
    nodes = load(a.capture)
    ops, rows, w, rd = ledger(nodes)

    by_op = collections.Counter()
    calls = collections.Counter()
    for i, o in enumerate(ops):
        by_op[o["name"]] += w[i] + rd[i]
        calls[o["name"]] += 1
    tot = sum(by_op.values())

    cons = collections.defaultdict(list)
    alloc = collections.defaultdict(list)
    for r in rows:
        for i in r["consumers"]:
            cons[i].append(r)
        if r["alloc_op_i"] is not None:
            alloc[r["alloc_op_i"]].append(r)

    lim = a.pair_min_mb * 1e6
    sites, boundaries = [], []
    for i, o in enumerate(ops):
        ins = [r for r in cons[i] if r["size"] >= lim]
        outs = [r for r in alloc[i] if r["size"] >= lim]
        if not ins and not outs:
            continue
        row = {"op_i": i, "op": o["name"],
               "in": [[r["kind"].split("::")[-1], r["size"], r["alloc_op"]] for r in ins],
               "out": [[r["kind"].split("::")[-1], r["size"]] for r in outs],
               "dram_MB": round((w[i] + rd[i]) / 1e6, 3)}
        boundaries.append(row)
        if o["name"] == "ttnn.add_" and len(ins) >= 2:
            kinds = sorted(r["kind"].split("::")[-1] for r in ins[:2])
            sites.append({**row, "update_in": [k for k in kinds if k != "DRAM"] or ["DRAM"]})

    print("one PairformerLayer, DRAM bytes deduped on buffer address. Z = %.3f MB" % (Z / 1e6))
    for n, b in by_op.most_common(12):
        print("  %-36s %4d calls %9.1f MB %6.2f Z %5.1f%%"
              % (n, calls[n], b / 1e6, b / Z, 100 * b / tot))
    print("  TOTAL %d ops %.1f MB = %.2f Z" % (len(ops), tot / 1e6, tot / Z))
    print("\nresidual sites (pair-sized in-place add_), and where the update lives:")
    for s in sites:
        print("  op %-4d update in %-4s allocated by %-32s site cost %.1f MB = %.2f Z"
              % (s["op_i"], s["update_in"][0],
                 next((x[2] for x in s["in"] if x[0] != "DRAM"), "?"),
                 s["dram_MB"], s["dram_MB"] * 1e6 / Z))
    out = {"capture": a.capture, "Z_bytes": Z, "n_ops": len(ops),
           "total_dram_MB": round(tot / 1e6, 3), "total_Z": round(tot / Z, 4),
           "by_op_MB": {k: round(v / 1e6, 3) for k, v in by_op.most_common()},
           "calls_by_op": dict(calls), "residual_sites": sites, "pair_boundaries": boundaries}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
