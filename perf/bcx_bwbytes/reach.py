#!/usr/bin/env python3
"""bcx-bwbytes: the trunk backward's bytes attributed to the change that would remove them.

`bcx-bytes` measured 20.3 of 40.0 GB per Evoformer block as avoidable and sorted them by REASON
-- typecast, layout, padding, fusible. A reason is not a change. This re-reads that row's own
traces through its own census and sorts the same bytes by the NAMED site that issues them, so
the row can say how much of the 2x is reachable and by editing what.

Three things about the instrument, and all three cut against the answer looking good:

  * it counts DEPTH-0 ttnn calls, so a kernel's internal traffic is invisible. `ttnn.sum(dim=0)`
    permutes its operand before it reduces, and the census charges it one read where the card
    moves three. Every figure here is a LOWER bound on DRAM traffic.
  * a lever that replaces one kernel with a faster kernel over the same operands removes no
    bytes. The reblock permute backward is exactly that and it appears here worth zero.
  * the segment called "the backward" contains the checkpoint's recompute of the forward. Those
    bytes are split out rather than counted as backward work, because the edit that removes them
    is a memory decision and not a kernel.

Card-free: it reads `perf/bcx_bytes/trace_*.json` and computes.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perf.bcx_bytes.bytes import ITEM, census, summarize                # noqa: E402

TR = ROOT / "perf" / "bcx_bytes"
OUT = pathlib.Path(__file__).resolve().parent

#: A triangle attention's score tensor is [tokens, heads, tokens, tokens] and it is by a wide
#: margin the largest thing either stack touches: 268 MB in fp32 at n=256, against 33.6 MB for a
#: whole pair track. Rows are tagged by whether an operand of theirs has the score's rank-4 shape.
def _is_score(sig, n):
    return f"[{n}, 4, {n}, {n}]" in sig


def _ranges(path, names, rev):
    """(first, last) line of each named top-level def, so a `bw` frame can be attributed to the
    op whose closure it is.

    Read at `rev`, not from the working tree, and that is the whole point of the argument. A
    trace's frame line numbers are line numbers in the file AS IT WAS WHEN THE TRACE RAN. This
    row added 52 lines to `autograd.py` above `layer_norm`, and reading today's file put every
    closure in the wrong function -- silently, since the buckets still add to the same total.
    """
    import subprocess
    src = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    starts = [(i + 1, l[4:].split("(")[0]) for i, l in enumerate(src) if l.startswith("def ")]
    out = {}
    for k, (line, name) in enumerate(starts):
        if name in names:
            end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(src)
            out[name] = (line, end)
    return out


CLOSURES = {}


def _closure_of(site):
    """Which autograd.py op owns a `...:bw` / `...:<lambda>` frame, by line range."""
    if not site or not site.startswith("autograd.py:"):
        return None
    try:
        line = int(site.split(":")[1])
    except ValueError:
        return None
    for name, (a, b) in CLOSURES.items():
        if a <= line <= b:
            return name
    return None


def bucket(r, n):
    """The named site whose edit removes this row's bytes."""
    if r["view"]:
        return None
    site = r["tape"] or "?"
    op = r["op"]
    # `node` is set only while a tape closure runs, so a row without one was issued by the
    # checkpoint re-running the forward inside the backward window.
    if not r["node"]:
        return "0 recompute: the checkpoint re-running the forward"
    if "add_grad" in site:
        return "1 add_grad fan-in: two gradients widened to fp32 and summed"
    if ("softmax_bw_inner" in site or "_v_softmax" in site
            or _closure_of(site) in ("softmax", "triangle_attention")
            or _is_score(r["sig"], n) and op in (
            "multiply", "subtract", "add", "sum", "div", "divide", "typecast", "softmax")):
        return "2 softmax backward on the score tensor"
    if _closure_of(site) in ("layer_norm", "_taped_layer_norm") or op == "layer_norm":
        return "3 layer-norm backward"
    if op in ("permute", "transpose", "to_layout", "to_memory_config", "concat", "reshape",
              "nlp_concat_heads", "nlp_create_qkv_heads", "clone"):
        return "4 layout moves"
    if op in ("matmul", "linear", "minimal_matmul") or "bmm" in site or "_v_matmul" in site:
        return "5 the arithmetic itself"
    if op == "typecast":
        return "6 other typecasts"
    return "7 residual eltwise"


def _b(d, elem=None):
    """DRAM bytes of one operand descriptor, optionally as if it were `elem` bytes wide."""
    import numpy as np
    if d["buf"] != "DRAM":
        return 0.0
    w = elem if elem is not None else ITEM.get(d["dtype"], 2)
    return float(np.prod(d["padded"])) * w


def predict(ops, rows, n):
    """What each lever removes, computed op by op from the same trace. A MODEL, not a reading.

    Written down before the card pass so the measurement can contradict it. Two of the four
    levers are predicted at zero or near it and that is the useful half of the exercise:

      perm    bit-identical index reordering over the same operands. Removes NO bytes. It was
              routed on device milliseconds, which is a different and legitimate question.
      tree    unpredictable from this instrument. `ttnn.sum(dim=0)` permutes its operand before
              it reduces and the census only sees the depth-0 call, so the baseline it would be
              compared against is itself wrong here. Left unpredicted rather than guessed.
      smbf16  every fp32 operand of the softmax backward at half width, minus the cost of
              narrowing y and the cotangent once per call. Calls are counted as the `subtract`
              ops in the bucket: the expression has exactly one, `g - inner`.
      fanin   the widening typecasts `census.why` attributes to fan-in, in full.
    """
    idx = {r["i"]: r for r in rows}
    smb = [r for r in rows if r.get("_bucket", "").startswith("2 ")]
    fan = sum(r["typecast"] for r in rows
              if (r["why"] or "").startswith("fan-in") and not r["view"])
    now = sum(r["moved"] for r in smb)
    after = 0.0
    for r in smb:
        o = ops[r["i"]]
        io = [d for d in o["ins"] + o["outs"]]
        after += sum(_b(d, 2 if d["dtype"] == "FLOAT32" else None) for d in io)
    calls = sum(1 for r in smb if r["op"] == "subtract")
    # one narrowing per operand per call: read the fp32 tensor, write the bf16 one
    score_fp32 = max((_b(d) for r in smb for d in ops[r["i"]]["ins"]
                      if d["dtype"] == "FLOAT32"), default=0.0)
    narrow = 2 * calls * (score_fp32 * 1.5)
    return {"smbf16": {"bucket_now_GB": round(now / 1e9, 3),
                       "bucket_at_bf16_GB": round(after / 1e9, 3),
                       "narrowing_cost_GB": round(narrow / 1e9, 3),
                       "calls": calls,
                       "net_removed_GB": round((now - after - narrow) / 1e9, 3)},
            "fanin": {"net_removed_GB": round(fan / 1e9, 3)},
            "perm": {"net_removed_GB": 0.0,
                     "why": "a faster kernel over the same operands moves the same bytes"},
            "tree": {"net_removed_GB": None,
                     "why": "ttnn.sum(dim=0) permutes before it reduces and the census cannot "
                            "see that, so the baseline this would be measured against is wrong"}}


def scale(a, b, ns, at):
    """Per-bucket power law from two measured sizes, evaluated at the sizes that matter.

    Two points give one exponent and no error bar, so this is a MODEL and it is labelled one.
    What makes it worth having anyway is that the exponents come out interpretable rather than
    fitted: the score tensor is [n, heads, n, n] and the pair track is [1, n, n, c], so a bucket
    should land on n^3 or n^2 and it is a real check on the bucketing when one does not.

    The n=128 traces are the `bwd` arm, not `bwd-fix`. That is not a mismatch: `bcx-bytes`
    measured the fix as identity at n=128 -- "at n=128 the plan's block (330 and 165 rows) already
    covers all 128 rows, so nothing is chunked there" -- so the two points are the same tree.
    """
    import math
    lo, hi = ns
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k, {}).get("GB", 0.0), b.get(k, {}).get("GB", 0.0)
        if x <= 0 or y <= 0:
            continue
        e = math.log(y / x) / math.log(hi / lo)
        out[k] = {"GB_lo": x, "GB_hi": y, "exponent": round(e, 3),
                  "at": {str(n): round(y * (n / hi) ** e, 3) for n in at}}
    return out


def cmd_scale(args):
    """The reach map at the size BindCraft 2 actually runs, from the two sizes that were traced."""
    import json as _json
    blob = {}
    for stack, lo_t, hi_t in (("evo", "trace_bwd_evo_n128", "trace_bwd-fix_evo_n256"),
                              ("extra", "trace_bwd_extra_n128", "trace_bwd-fix_extra_n256")):
        recs = {}
        for tag, name, n in (("lo", lo_t, 128), ("hi", hi_t, 256)):
            t = _json.loads((TR / f"{name}.json").read_text())
            rows = census(t["bwd"])
            g = collections.defaultdict(float)
            for r in rows:
                bk = bucket(r, int(t["n"]))
                r["_bucket"] = bk or ""
                if bk:
                    g[bk] += r["moved"]
            recs[tag] = ({k: {"GB": round(v / 1e9, 3)} for k, v in g.items()},
                         predict(t["bwd"], rows, int(t["n"])))
        at = [int(x) for x in args.at.split(",")]
        sc = scale(recs["lo"][0], recs["hi"][0], (128, 256), at)
        lev = {}
        for k in ("smbf16", "fanin"):
            x = recs["lo"][1][k]["net_removed_GB"]
            y = recs["hi"][1][k]["net_removed_GB"]
            lev[k] = scale({k: {"GB": x}}, {k: {"GB": y}}, (128, 256), at)[k]
        blob[stack] = {"by_site": sc, "levers": lev,
                       "totals": {str(n): round(sum(v["at"][str(n)] for v in sc.values()), 3)
                                  for n in at}}
        print(f"\n{stack}: bytes per block backward, per-bucket power law from n=128 and n=256")
        print(f"   {'bucket':52s} {'exp':>5s} " + " ".join(f"{('n=%d' % n):>9s}" for n in at))
        for k, v in sorted(sc.items()):
            print(f"   {k[:52]:52s} {v['exponent']:5.2f} "
                  + " ".join(f"{v['at'][str(n)]:9.3f}" for n in at))
        print(f"   {'TOTAL':52s} {'':5s} "
              + " ".join(f"{blob[stack]['totals'][str(n)]:9.3f}" for n in at))
        for k, v in lev.items():
            print(f"   lever {k:46s} {v['exponent']:5.2f} "
                  + " ".join(f"{v['at'][str(n)]:9.3f}" for n in at))
        for n in at:
            st = sum(lev[k]["at"][str(n)] for k in lev)
            tot = blob[stack]["totals"][str(n)]
            print(f"   -> at n={n}: the precision stack removes {st:.3f} of {tot:.3f} GB, "
                  f"{st / tot * 100:.1f} %")
    pathlib.Path(args.out).write_text(_json.dumps(blob, indent=1))
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="trace_bwd-fix_evo_n256,trace_bwd-fix_extra_n256")
    ap.add_argument("--scale", action="store_true",
                    help="per-bucket power law from the two traced sizes, evaluated at --at")
    ap.add_argument("--at", default="224,256,512,1536",
                    help="224 is what a BindCraft 2 round runs (211 tokens bucketed); 1536 is the "
                         "MGX size target")
    ap.add_argument("--out", default=str(OUT / "reach.json"))
    ap.add_argument("--src", default="a5fa47832",
                    help="the revision the traces were RECORDED on -- their frame line numbers "
                         "are line numbers in that tree, not in this one")
    args = ap.parse_args()

    global CLOSURES
    CLOSURES = _ranges("tt_bio/autograd.py",
                       {"layer_norm", "_taped_layer_norm", "triangle_attention", "softmax"},
                       args.src)
    if args.scale:
        if args.out.endswith("reach.json"):
            args.out = args.out.replace("reach.json", "reach_scale.json")
        return cmd_scale(args)
    blob = {}
    for name in args.traces.split(","):
        t = json.loads((TR / f"{name}.json").read_text())
        n = int(t["n"])
        rows = census(t["bwd"])
        s = summarize(rows)
        g = collections.defaultdict(lambda: [0, 0.0])
        for r in rows:
            b = bucket(r, n)
            r["_bucket"] = b or ""
            if b is None:
                continue
            g[b][0] += 1
            g[b][1] += r["moved"]
        moved = sum(r["moved"] for r in rows if not r["view"])
        score_fp32 = sum(r["fp32"] for r in rows
                         if not r["view"] and _is_score(r["sig"], n))
        rec = {"trace": name, "n": n, "stack": t["stack"], "depth0_ops": len(t["bwd"]),
               "moved_GB": round(moved / 1e9, 3),
               "avoidable_GB_bcx_bytes_reasons": round(s["avoidable_MB"] / 1e3, 3),
               "fp32_excess_GB": round(s["fp32_excess_MB"] / 1e3, 3),
               "fp32_excess_on_score_GB": round(score_fp32 / 1e9, 3),
               "by_site": {k: {"ops": v[0], "GB": round(v[1] / 1e9, 3),
                               "share": round(v[1] / moved, 4)}
                           for k, v in sorted(g.items())}}
        rec["src_rev"] = args.src
        rec["predicted"] = predict(t["bwd"], rows, n)
        rec["closure_lines"] = {k: list(v) for k, v in CLOSURES.items()}
        blob[name] = rec
        print(f"\n{name}  {rec['moved_GB']} GB over {rec['depth0_ops']} depth-0 ops")
        for k, v in sorted(rec["by_site"].items(), key=lambda x: -x[1]["GB"]):
            print(f"   {v['GB']:7.3f} GB  {v['share'] * 100:5.1f} %  n={v['ops']:4d}  {k}")
        print(f"   {rec['fp32_excess_on_score_GB']:7.3f} GB          of the above is the half of "
              f"an fp32 SCORE operand a bf16 one would not move")
        print("   PREDICTED, a model and not a reading:")
        for k, v in rec["predicted"].items():
            net = v["net_removed_GB"]
            print(f"     {k:8s} {('%7.3f GB' % net) if net is not None else '      n/a'}"
                  f"  {v.get('why', '')}")

    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
