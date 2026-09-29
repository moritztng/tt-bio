#!/usr/bin/env python3
"""bcx-p10-l1fuse leg 1: rank the chain census by written-then-immediately-reread bytes.

Per block = K2 - K1 on every counter, per stack, so everything outside the blocks cancels.
Per round = 48 evo blocks + 4 extra-MSA blocks, the round's own multiplicity.

Device seconds come off the same records, by the campaign's own subtraction

    device(tag, verb) = median_reps(sync wall) - median_reps(free wall) - lambda * calls

clamped at zero per key, which is what `bcx-p10-calls` does and why A_verb >= A_family. An
edge's device seconds are the producer's and the consumer's own seconds, prorated by the share
of that verb's calls the edge accounts for; the columns are printed separately because the two
ends are not additive when one op feeds several.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#: The composed round's block multiplicity, the same one every census in this campaign uses.
MULT = {"evo": 48, "extra": 4}


def _median_delta(recs, field, stack, mode):
    """median over reps of (K=2 minus K=1) for every key of `field`."""
    per = collections.defaultdict(dict)
    for r in recs:
        if r["stack"] != stack or r["mode"] != mode:
            continue
        for k, v in r.get(field, {}).items():
            per[k].setdefault(r["rep"], {})[r["K"]] = v
    out = {}
    for k, byrep in per.items():
        ds = [d.get(2, 0) - d.get(1, 0) for d in byrep.values() if 2 in d or 1 in d]
        if ds:
            out[k] = statistics.median(ds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "perf/bcx_p10_l1fuse/out/chains_E.json"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "perf/bcx_p10_l1fuse/out/chains_rank.json"))
    args = ap.parse_args()

    blob = json.loads(pathlib.Path(args.json).read_text())
    recs = blob["records"]
    lam = blob["sync_floor_s"]["median"]

    edges = collections.Counter()          # per round
    ebytes = collections.Counter()
    eadj = collections.Counter()
    egap = collections.Counter()
    dev_verb = collections.Counter()       # (tag, verb) -> device s per round
    calls_verb = collections.Counter()
    per_stack = {}

    for stack, mult in MULT.items():
        e = _median_delta(recs, "edges", stack, "sync")
        b = _median_delta(recs, "edge_bytes", stack, "sync")
        a = _median_delta(recs, "edge_adj", stack, "sync")
        g = _median_delta(recs, "edge_gap", stack, "sync")
        sw = _median_delta(recs, "verb_wall", stack, "sync")
        fw = _median_delta(recs, "verb_wall", stack, "free")
        vc = _median_delta(recs, "verb_calls", stack, "sync")
        per_stack[stack] = {"edge_keys": len(e), "edge_instances": sum(e.values()),
                            "verb_keys": len(sw)}
        for k, v in e.items():
            edges[k] += v * mult
        for k, v in b.items():
            ebytes[k] += v * mult
        for k, v in a.items():
            eadj[k] += v * mult
        for k, v in g.items():
            egap[k] += v * mult
        for k in set(sw) | set(fw):
            d = sw.get(k, 0.0) - fw.get(k, 0.0) - lam * vc.get(k, 0.0)
            dev_verb[k] += max(0.0, d) * mult
            calls_verb[k] += vc.get(k, 0.0) * mult

    rows = []
    for k, n in edges.items():
        pverb, ptag, cverb, ctag, shape, dt, buf = k.split("||")
        pk, ck = f"{ptag}||{pverb}", f"{ctag}||{cverb}"
        # tag is stack|block|dir|family; the block index is the K-difference's own bookkeeping
        # and is not part of what the edge IS, so it is dropped from the printed site.
        pst, pdir, pfam = ptag.split("|")[0], ptag.split("|")[2], ptag.split("|")[3]
        cst, cdir, cfam = ctag.split("|")[0], ctag.split("|")[2], ctag.split("|")[3]
        pshare = (n / calls_verb[pk]) if calls_verb.get(pk) else 0.0
        cshare = (n / calls_verb[ck]) if calls_verb.get(ck) else 0.0
        rows.append({
            "producer": pverb, "producer_site": ptag,
            "consumer": cverb, "consumer_site": ctag,
            "dir": pdir if pdir == cdir else f"{pdir}->{cdir}",
            "stack": pst, "producer_family": pfam, "consumer_family": cfam,
            "shape": shape, "dtype": dt, "buffer": buf,
            "n_round": round(n, 1),
            "gb_round": round(ebytes[k] / 1e9, 4),
            "adjacent_frac": round(eadj[k] / n, 3) if n else 0.0,
            "mean_gap": round(egap[k] / n, 2) if n else 0.0,
            "dev_s_producer": round(dev_verb.get(pk, 0.0) * min(1.0, pshare), 4),
            "dev_s_consumer": round(dev_verb.get(ck, 0.0) * min(1.0, cshare), 4),
        })
    rows.sort(key=lambda r: -r["gb_round"])

    out = {"source": args.json, "lambda_s": lam, "mult": MULT, "per_stack": per_stack,
           "armed": blob.get("armed"), "stamp": blob.get("stamp"),
           "edge_gb_round_total": round(sum(ebytes.values()) / 1e9, 3),
           "dram_edge_gb_round": round(
               sum(v for k, v in ebytes.items() if k.split("||")[6] == "DRAM") / 1e9, 3),
           "rows": rows}
    print(f"lambda {lam*1e6:.1f} us   edges/round {sum(edges.values()):.0f}   "
          f"bytes/round {out['edge_gb_round_total']:.2f} GB "
          f"(DRAM {out['dram_edge_gb_round']:.2f} GB)")
    hdr = (f"{'GB/rd':>7} {'n/rd':>7} {'adj':>5} {'gap':>6} {'devP':>7} {'devC':>7} {'dir':>4} "
           f"{'buf':>4} {'shape':<20} {'producer -> consumer'}")

    def show(rs, title):
        print("\n" + title)
        print(hdr)
        for r in rs:
            print(f"{r['gb_round']:7.3f} {r['n_round']:7.0f} {r['adjacent_frac']:5.2f} "
                  f"{r['mean_gap']:6.1f} {r['dev_s_producer']:7.4f} {r['dev_s_consumer']:7.4f} "
                  f"{r['dir']:>4} {r['buffer'][:4]:>4} {r['shape']+' '+r['dtype'][:4]:<20} "
                  f"{r['producer_family']}/{r['producer']} -> "
                  f"{r['consumer_family']}/{r['consumer']}")

    show(rows[:args.top], "ALL EDGES, by bytes written then re-read per round")
    bwd = [r for r in rows if r["dir"] == "bwd" and r["buffer"] == "DRAM"]
    fwd = [r for r in rows if r["dir"] == "fwd" and r["buffer"] == "DRAM"]
    show(bwd[:args.top], "BACKWARD DRAM edges -- no tape holds these, so no evict tax")
    show(fwd[:args.top], "FORWARD DRAM edges -- `Tensor.evict` taxes any the backward reads")
    out["bwd_dram_gb_round"] = round(sum(r["gb_round"] for r in bwd), 3)
    out["fwd_dram_gb_round"] = round(sum(r["gb_round"] for r in fwd), 3)
    print(f"\nDRAM edge bytes/round: backward {out['bwd_dram_gb_round']:.2f} GB, "
          f"forward {out['fwd_dram_gb_round']:.2f} GB")
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
