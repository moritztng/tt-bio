#!/usr/bin/env python3
"""Reduce a traced fold graph into fusable pairs, and price each one in bytes then seconds.

A pair is fusable only if the producer\x27s result has EXACTLY ONE consumer in the executed graph
and that consumer is the op we would fuse into. On a pre-norm residual trunk the sum of
``add_(z, update)`` is read both by the next sub-layer\x27s norm and by the next in-place add, so
it has two consumers and there is nothing to fuse; that is the case a source grep cannot see.

Three things this gets right, each in the direction that refuses to invent a fusion:

* ``deallocate`` reads a tensor but moves no bytes, so it is not a consumer. Counting it would
  hide real single-consumer producers.
* A view op (reshape/squeeze/unsqueeze/to_layout/to_memory_config/clone) that hands back the SAME
  buffer address has not consumed the value, it has renamed it. Consumer counting follows through
  such a rename, so a producer read by one reshape that five ops then read counts as five.
* Dataflow is keyed on (buffer address, version), never on tensor id, because ttnn reuses freed
  addresses.
* Only DRAM bytes count. The shipped ``_PAIR_PROJ_L1_OUT`` lever already lands the trimul's output
  projection in L1, so the ``multiply_`` that reads it and the residual ``add_`` that reads the
  product move no DRAM bytes at all. Deleting an L1 write and pricing it at the DRAM roof is how
  a lever gets invented. Residency is recorded per operand by the tracer, not assumed here.

Pricing. Fusing a producer into its consumer deletes the producer\x27s output write and the
consumer\x27s read of it: 2 x the producer\x27s output bytes per call. Keys the c10-fold-census
measured at or above 97 % of the 442.9 GB/s DRAM roof convert deleted bytes at the roof, every
other key at its own measured GB/s.
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

NON_CONSUMING = {"deallocate"}
VIEW_OPS = {"reshape", "squeeze", "unsqueeze", "to_layout", "to_memory_config", "clone"}

DRAM_ROOF_GBS = 442.8767243360721   # c10-fold-census sweep2, same chip, same recorded clock
AT_ROOF_PCT = 97.0


def load_census(path: Path):
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

    defs = {}                      # (addr, ver) -> record index
    uses = defaultdict(list)       # (addr, ver) -> [record indices], deallocate excluded
    for i, r in enumerate(recs):
        op, reads, writes = r[1], r[3], r[4]
        for w0 in writes:
            ad, ver = w0[0], w0[1]
            defs[(ad, ver)] = i
        if op in NON_CONSUMING:
            continue
        for ad, ver in {(x[0], x[1]) for x in reads}:
            uses[(ad, ver)].append(i)

    def renames(i, key):
        """If record i is a view op that handed back key\x27s own buffer, the value it produced."""
        op, reads, writes = recs[i][1], recs[i][3], recs[i][4]
        if op not in VIEW_OPS:
            return None
        for w0 in writes:
            ad, ver = w0[0], w0[1]
            if ad == key[0]:
                return (ad, ver)
        return None

    def consumers_of(key, depth=0):
        """Records that actually read key\x27s bytes, following buffer renames."""
        out = []
        for c in uses.get(key, ()):
            nk = renames(c, key) if depth < 4 else None
            if nk is not None and nk != key:
                out.extend(consumers_of(nk, depth + 1))
            else:
                out.append(c)
        return out

    op_counts = defaultdict(int)
    for r in recs:
        op_counts[r[1]] += 1

    cand = defaultdict(lambda: {"calls": 0, "bytes_per_call": 0, "shape": "", "dram": 0,
                                "l1_calls": 0, "l1_bytes": 0})
    rejected = defaultdict(lambda: {"calls": 0, "reason_hist": defaultdict(int)})
    # P1 evidence: consumer-count histogram for every producer family, per shape
    hist = defaultdict(lambda: defaultdict(int))

    for key, di in defs.items():
        op, site, reads, writes = recs[di][1], recs[di][2], recs[di][3], recs[di][4]
        if op not in (PRODUCER_MUL | PRODUCER_ADD):
            continue
        w = next((x for x in writes if (x[0], x[1]) == key), None)
        if w is None:
            continue
        nb, shape = w[3], shapes.get(w[2], "?")
        dram = w[4] if len(w) > 4 else -1
        cons = [c for c in consumers_of(key) if c != di]
        cops = sorted({recs[c][1] for c in cons})
        hist[(op, shape)]["%d:%s" % (len(cons), ",".join(cops) if cops else "-")] += 1

        for fam, prods, ctgt in (("mul_into_add", PRODUCER_MUL, CONSUMER_ADD),
                                 ("add_into_norm", PRODUCER_ADD, CONSUMER_NORM)):
            if op not in prods:
                continue
            ck = (fam, sites.get(site, "?"), shape)
            if len(cons) != 1:
                rejected[ck]["calls"] += 1
                rejected[ck]["reason_hist"]["%d_consumers" % len(cons)] += 1
                continue
            cop = recs[cons[0]][1]
            if cop not in ctgt:
                rejected[ck]["calls"] += 1
                rejected[ck]["reason_hist"]["sole_consumer_is_%s" % cop] += 1
                continue
            ck2 = (fam, sites.get(site, "?"), sites.get(recs[cons[0]][2], "?"), shape)
            e = cand[ck2]
            e["shape"] = shape
            e["dram"] = dram
            if dram == 0:
                # the value never reaches DRAM, so the fusion deletes no DRAM byte
                e["l1_calls"] += 1
                e["l1_bytes"] += 2 * nb
            else:
                e["calls"] += 1
                e["bytes_per_call"] = 2 * nb

    def price(shape, nb_per_fold, arm):
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
        sec, rate = price(shape, total, arm)
        rows.append({"family": fam, "producer_site": psite, "consumer_site": csite,
                     "shape": shape, "calls": e["calls"],
                     "bytes_per_call": e["bytes_per_call"], "bytes_per_fold": total,
                     "seconds": sec, "rate_basis": rate,
                     "l1_calls": e["l1_calls"], "l1_bytes_not_counted": e["l1_bytes"]})
    rej = []
    for (fam, psite, shape), e in sorted(rejected.items(), key=lambda x: -x[1]["calls"]):
        rej.append({"family": fam, "producer_site": psite, "shape": shape, "calls": e["calls"],
                    "reasons": dict(e["reason_hist"])})
    hrows = []
    for (op, shape), h in sorted(hist.items(), key=lambda x: -sum(x[1].values())):
        hrows.append({"op": op, "shape": shape, "defs": sum(h.values()),
                      "consumer_hist": dict(sorted(h.items(), key=lambda x: -x[1]))})

    res = {"trace": str(a.trace), "n_ops": len(recs), "op_counts": dict(op_counts),
           "dram_roof_GBs": DRAM_ROOF_GBS, "at_roof_pct": AT_ROOF_PCT,
           "fusable": rows, "rejected": rej, "consumer_hist": hrows,
           "total_bytes_per_fold": sum(r["bytes_per_fold"] for r in rows),
           "total_seconds": sum(r["seconds"] for r in rows if r["seconds"])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))

    print("ops traced: %d" % len(recs))
    print("\nFUSABLE (producer result has exactly one consumer, and it is the fuse target)")
    print("%-14s %8s %14s %10s %8s %8s  %-30s -> %s"
          % ("family", "dram_c", "DRAM_B/fold", "s", "rate", "l1_c", "producer", "consumer"))
    for r in rows:
        print("%-14s %8d %14d %10s %8s %8d  %-30s -> %s"
              % (r["family"], r["calls"], r["bytes_per_fold"],
                 ("%.4f" % r["seconds"]) if r["seconds"] else "-",
                 r["rate_basis"].split()[0], r["l1_calls"],
                 r["producer_site"][-30:], r["consumer_site"][-30:]))
    print("\nTOTAL %d B/fold, %.4f s" % (res["total_bytes_per_fold"], res["total_seconds"]))
    print("\nCONSUMER-COUNT HISTOGRAM (every add_/multiply_ def, by output shape)")
    for h in hrows[:18]:
        print("%-11s %-26s defs=%-7d %s"
              % (h["op"], h["shape"], h["defs"],
                 "  ".join("%s x%d" % (k, v) for k, v in list(h["consumer_hist"].items())[:4])))
    print("\nREJECTED (top 20 by calls)")
    for r in rej[:20]:
        print("%-14s %8d %-44s %s"
              % (r["family"], r["calls"], r["producer_site"][-44:], r["reasons"]))
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
