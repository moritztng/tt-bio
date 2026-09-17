"""Reduce a traced fold graph into fusable pairs, and price each one in bytes then seconds.

A pair is fusable only if the producer's result has EXACTLY ONE consumer in the executed graph
and that consumer is the op we would fuse into. On a pre-norm residual trunk the sum of
``add_(z, update)`` is read both by the next sub-layer's norm and by the next in-place add, so
it has two consumers and there is nothing to fuse; that is the case a source grep cannot see.

Pricing. Fusing a producer into its consumer deletes the producer's output write and the
consumer's read of it: 2 x the producer's output bytes per call. Dedupe is on
(buffer address, version), never on tensor id, because ttnn reuses freed addresses.

Seconds come from the rate the class actually runs at, not from a nominal roof: keys the
c10-fold-census measured at or above 97 % of the 442.9 GB/s DRAM roof convert at the roof,
every other key converts at its own measured GB/s.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PRODUCER_MUL = {"multiply", "multiply_"}
CONSUMER_ADD = {"add", "add_"}
PRODUCER_ADD = {"add", "add_"}
CONSUMER_NORM = {"layer_norm"}

DRAM_ROOF_GBS = 442.8767243360721   # c10-fold-census sweep2, same chip, same recorded clock
AT_ROOF_PCT = 97.0                  # keys at/above this run at the roof, not at their own rate


def load_census(path: Path):
    """Per-key measured rate and call count, keyed (arm, out-shape-string)."""
    b = json.loads(path.read_text())
    out = {}
    for r in b["keys"]:
        shape = "x".join(str(int(d)) for d in r["out"])
        out[(r["arm"], shape)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t = json.loads(a.trace.read_text())
    sites = {int(k): v for k, v in t["sites"].items()}
    shapes = {int(k): v for k, v in t["shapes"].items()}
    recs = t["records"]
    census = load_census(a.census)

    # def-use over (addr, version). A record's writes define; a record's reads use.
    defs = {}                      # (addr, ver) -> record index
    uses = defaultdict(list)       # (addr, ver) -> [record indices]
    for i, (seq, op, site, reads, writes) in enumerate(recs):
        for ad, ver, sid, nb in writes:
            defs[(ad, ver)] = i
        for ad, ver, sid, nb in set((r[0], r[1]) for r in reads) if reads else ():
            uses[(ad, ver)].append(i)
    # reads were deduped above on (addr, ver); recover shape/bytes per read from the record
    read_meta = {}
    for i, (seq, op, site, reads, writes) in enumerate(recs):
        for ad, ver, sid, nb in reads:
            read_meta[(i, ad, ver)] = (sid, nb)

    op_counts = defaultdict(int)
    for seq, op, site, reads, writes in recs:
        op_counts[op] += 1

    # For an in-place producer the output buffer is the input buffer at the NEXT version, so
    # a self-read (same addr, previous version) is the op's own input and is not a consumer.
    cand = defaultdict(lambda: {"calls": 0, "bytes_per_call": 0, "shape": "", "n_consumers": None})
    rejected = defaultdict(lambda: {"calls": 0, "reason_hist": defaultdict(int)})

    for key, di in defs.items():
        seq, op, site, reads, writes = recs[di]
        consumers = uses.get(key, [])
        # the defining record itself never counts as its own consumer
        consumers = [c for c in consumers if c != di]
        w = next((x for x in writes if (x[0], x[1]) == key), None)
        if w is None:
            continue
        nb = w[3]
        shape = shapes.get(w[2], "?")

        for fam, prods, cons in (("mul_into_add", PRODUCER_MUL, CONSUMER_ADD),
                                 ("add_into_norm", PRODUCER_ADD, CONSUMER_NORM)):
            if op not in prods:
                continue
            ck = (fam, sites.get(site, "?"), shape)
            if len(consumers) != 1:
                rejected[ck]["calls"] += 1
                rejected[ck]["reason_hist"]["%d_consumers" % len(consumers)] += 1
                continue
            cop = recs[consumers[0]][1]
            if cop not in cons:
                rejected[ck]["calls"] += 1
                rejected[ck]["reason_hist"]["sole_consumer_is_%s" % cop] += 1
                continue
            ck2 = (fam, sites.get(site, "?"), sites.get(recs[consumers[0]][2], "?"), shape)
            e = cand[ck2]
            e["calls"] += 1
            e["bytes_per_call"] = 2 * nb
            e["shape"] = shape

    def price(shape, nb_per_fold, arm):
        """seconds deleted, and the rate used, for bytes removed from a class."""
        base = shape.split("|")[0]
        row = census.get((arm, base))
        if row is not None and row["pct_of_dram_roof"] >= AT_ROOF_PCT:
            return nb_per_fold / (DRAM_ROOF_GBS * 1e9), "roof %.1f GB/s" % DRAM_ROOF_GBS
        if row is not None and row["GBs"] > 0:
            return nb_per_fold / (row["GBs"] * 1e9), "key %.1f GB/s" % row["GBs"]
        return None, "unpriced (key not in census)"

    rows = []
    for (fam, psite, csite, shape), e in sorted(cand.items(), key=lambda x: -x[1]["calls"]):
        arm = "multiply_" if fam == "mul_into_add" else "add_"
        total = e["calls"] * e["bytes_per_call"]
        s, rate = price(shape, total, arm)
        rows.append({"family": fam, "producer_site": psite, "consumer_site": csite,
                     "shape": shape, "calls": e["calls"],
                     "bytes_per_call": e["bytes_per_call"], "bytes_per_fold": total,
                     "seconds": s, "rate_basis": rate})
    rej = []
    for (fam, psite, shape), e in sorted(rejected.items(), key=lambda x: -x[1]["calls"]):
        rej.append({"family": fam, "producer_site": psite, "shape": shape, "calls": e["calls"],
                    "reasons": dict(e["reason_hist"])})

    res = {"trace": str(a.trace), "n_ops": len(recs), "op_counts": dict(op_counts),
           "dram_roof_GBs": DRAM_ROOF_GBS, "at_roof_pct": AT_ROOF_PCT,
           "fusable": rows, "rejected": rej,
           "total_bytes_per_fold": sum(r["bytes_per_fold"] for r in rows),
           "total_seconds": sum(r["seconds"] for r in rows if r["seconds"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))

    print("ops traced: %d" % len(recs))
    print("\nFUSABLE (producer result has exactly one consumer, and it is the fuse target)")
    print("%-14s %8s %14s %12s %10s  %-30s -> %s"
          % ("family", "calls", "B/fold", "s", "rate", "producer", "consumer"))
    for r in rows:
        print("%-14s %8d %14d %12s %10s  %-30s -> %s"
              % (r["family"], r["calls"], r["bytes_per_fold"],
                 ("%.4f" % r["seconds"]) if r["seconds"] else "-",
                 r["rate_basis"].split()[0], r["producer_site"][-30:], r["consumer_site"][-30:]))
    print("\nTOTAL %d B/fold, %.4f s" % (res["total_bytes_per_fold"], res["total_seconds"]))
    print("\nREJECTED (top 25 by calls)")
    for r in rej[:25]:
        print("%-14s %8d %-46s %s"
              % (r["family"], r["calls"], r["producer_site"][-46:], r["reasons"]))
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
