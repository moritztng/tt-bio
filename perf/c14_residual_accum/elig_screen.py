#!/usr/bin/env python3
"""Which adds in the executed boltz-2 path could take ttnn's residual_input_tensor, by firing.

`ttnn.layer_norm(x, residual_input_tensor=r)` computes `layer_norm(x + r)` in one kernel and
returns ONE tensor, the normed output. The sum is never materialised. So the fusion is legal at
exactly the adds whose result feeds nothing but a norm -- the contract tt_bio/eltwise_fusion.py:97
already states. That is a graph property, not a source property: an add whose destination buffer is
read again later, or is the in-place target of the next add, is ineligible however the source reads.

So this screens ttnn graph captures of real 512 aa ops, not source. For every add / add_ it finds
the buffer the add leaves its sum in, lists every op that consumes that buffer afterwards, and calls
the site eligible only when that list is exactly one norm. Buffer address is the dedupe basis, not
tensor id, because ttnn hands a reshape a fresh tensor id over the same buffer.
"""
import argparse, collections, gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b2x_difflayer"))
from itemize import itemize  # noqa: E402

Z = 67108864
ADDS = {"ttnn.add", "ttnn.add_"}
NORMS = {"ttnn.layer_norm", "ttnn.rms_norm"}


def load(p):
    n = json.load(gzip.open(p, "rt") if str(p).endswith(".gz") else open(p))
    return n["nodes"] if isinstance(n, dict) else n


def screen(path, pair_min_mb=0.0):
    ops, rows = itemize({"nodes": load(path)})
    by_buf = {r["buffer"]: r for r in rows}
    # every buffer an op reads, so an in-place add's destination can be recovered
    reads = collections.defaultdict(list)
    for r in rows:
        for i in r["consumers"]:
            reads[i].append(r)
    allocs = collections.defaultdict(list)
    for r in rows:
        if r["alloc_op_i"] is not None:
            allocs[r["alloc_op_i"]].append(r)

    sites = []
    for i, o in enumerate(ops):
        if o["name"] not in ADDS:
            continue
        # where the sum lands: an out-of-place add allocates it, an in-place add_ rewrites the
        # largest buffer it read (its destination operand).
        outs = allocs[i]
        inplace = o["name"].endswith("_")
        if inplace and not outs:
            cand = sorted(reads[i], key=lambda r: -r["size"])
            outs = cand[:1]
        if not outs:
            continue
        dest = max(outs, key=lambda r: r["size"])
        if dest["size"] < pair_min_mb * 1e6:
            continue
        after = [(j, ops[j]["name"]) for j in dest["consumers"] if j > i]
        # the in-place add itself shows up as a consumer of its own destination; drop it
        after = [(j, n) for j, n in after if j != i]
        # LIVE-OUT: a buffer the capture did not allocate came from outside it and can be read
        # outside it, so the capture cannot see all of the sums consumers and eligibility is NOT
        # decidable here. Every z residual is an in-place add_ on the pre-existing input buffer,
        # and the last one in a PairformerLayer capture reads as a single-consumer norm purely
        # because the next layers norm and add_ are past the capture boundary. Conservatively
        # ineligible, and reported separately so the artifact is visible rather than counted.
        live_out = dest["alloc_op_i"] is None
        elig = (not live_out) and len(after) == 1 and after[0][1] in NORMS
        sites.append({"op_i": i, "op": o["name"], "dest_kind": dest["kind"],
                      "dest_MB": round(dest["size"] / 1e6, 3),
                      "dest_Z": round(dest["size"] / Z, 3),
                      "consumers_after": after, "n_after": len(after),
                      "live_out": live_out, "eligible": elig})
    return ops, sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--pair-min-mb", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    report = {}
    tot_e = tot_s = 0
    for c in a.captures:
        ops, sites = screen(c, a.pair_min_mb)
        if not sites:
            continue
        name = Path(c).name
        report[name] = sites
        e = sum(s["eligible"] for s in sites)
        tot_e += e
        tot_s += len(sites)
        print("\n%s -- %d top-level ops, %d add sites, %d ELIGIBLE" % (name, len(ops), len(sites), e))
        for s in sites:
            tag = "ELIGIBLE" if s["eligible"] else ("live-out" if s["live_out"] else "no")
            cons = ", ".join("%d:%s" % (j, n.replace("ttnn.", "")) for j, n in s["consumers_after"]) or "(none)"
            print("  op %-4d %-10s dest %-4s %8.2f MB = %5.2f Z  after: %-58s  %s"
                  % (s["op_i"], s["op"].replace("ttnn.", ""), s["dest_kind"],
                     s["dest_MB"], s["dest_Z"], cons[:58], tag))
    lo = sum(s["live_out"] for v in report.values() for s in v)
    print("\nTOTAL %d add sites over %d captures, %d eligible for residual_input_tensor, "
          "%d live-out (undecidable inside the capture, counted ineligible)"
          % (tot_s, len(report), tot_e, lo))
    if a.out:
        a.out.write_text(json.dumps({"total_sites": tot_s, "total_eligible": tot_e,
                                     "pair_min_mb": a.pair_min_mb, "by_capture": report}, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
