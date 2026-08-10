#!/usr/bin/env python3
"""Blast-radius scan: every contracting op in a captured Pairformer block, with the LOGICAL length
of its contracted axis next to the PADDED one.

Reads a `pf_block_ops.py` capture and reports, per (op, site, operand signature), whether the
contracted axis is logically tile-aligned. That is the axis Q13 says costs 1.56x when it is not.
"""
import argparse, collections, json

CON = {"linear", "matmul", "minimal_matmul", "scaled_dot_product_attention"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    a = ap.parse_args()
    d = json.load(open(a.ops))
    agg = collections.defaultdict(lambda: {"n": 0, "s": 0.0, "drop": 0})
    for r in d["records"]:
        if r["op"] not in CON:
            continue
        ins = r["in"] or []
        if not ins:
            continue
        in0 = ins[0]
        in1 = ins[1] if len(ins) > 1 else None
        out = r.get("out")
        key = (r["op"], r["site"],
               tuple(in0["shape"]), tuple(in0["logical"]),
               tuple(in1["shape"]) if in1 else None, tuple(in1["logical"]) if in1 else None,
               tuple(out["shape"]) if out else None, tuple(out["logical"]) if out else None,
               in0["buf"], out["buf"] if out else None)
        e = agg[key]
        e["n"] += 1
        if r["s"] is None:
            e["drop"] += 1
        else:
            e["s"] += r["s"]
    rows = []
    for k, v in agg.items():
        op, site, ip, il, wp, wl, op_, ol, ibuf, obuf = k
        rows.append((v["s"] * 1e3, v["n"], v["drop"], op, site, ip, il, wp, wl, op_, ol, ibuf, obuf))
    rows.sort(reverse=True)
    print("block wall %.3f ms, %d records" % (d["block_wall_s"] * 1e3, len(d["records"])))
    print("%8s %4s %4s  %-34s %-26s %-22s %-22s %-12s %s"
          % ("ms/blk", "n", "drop", "op@site", "in0 padded/logical", "in1 padded/logical",
             "out padded/logical", "buf", "contracted axis"))
    for s, n, dr, op, site, ip, il, wp, wl, o, ol, ibuf, obuf in rows:
        kpad, klog = ip[-1], il[-1]
        tag = "K=%d logical, %d padded  ***UNALIGNED***" % (klog, kpad) if klog != kpad \
            else "K=%d aligned" % klog
        print("%8.3f %4d %4d  %-34s %-26s %-22s %-22s %-12s %s"
              % (s, n, dr, "%s@%s" % (op, site),
                 "%s/%s" % (list(ip), list(il)),
                 "%s/%s" % (list(wp), list(wl)) if wp else "-",
                 "%s/%s" % (list(o), list(ol)) if o else "-",
                 "%s->%s" % (ibuf, obuf), tag))


if __name__ == "__main__":
    main()
