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
              routed on device milliseconds, which is a different and legitimate question -- and
              on that question it is worth 5.9 ms per Evoformer block backward at n=256
              (`bcx-bytes` `psum_prof_arms.json`, 131.9 -> 126.0 ms) and NOTHING at BC2's 224,
              because `eligible_back`'s DRAM window opens at 256 and a BC2 round is 211 tokens
              bucketed to 224. Whether the window could be widened is already answered on disk:
              `perf/trix_layout/back_onepass_qb1c0.json` sweeps the same kernel by N at 9 reps
              against an A/A floor of 1.0002-1.0039 and reads 0.9918 / 0.9903 at N=256, 1.0086 /
              1.0092 at 288, 1.0135 / 1.0172 at 320, 1.0011 / 1.0003 at 352 and 1.0045 / 1.0018
              at 512. It LOSES at the bottom of the band, peaks at 1.7 %, and is inside the A/A
              floor at 352 and above. The mechanism the gate's own docstring gives -- below the
              window there are fewer work groups than cores, so the per-call cost is not
              amortised -- says 224 is further into the losing half, and no measurement exists
              below 256 on either direction. So this lever is closed for BC2 unless someone
              measures 224 and it opens.
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
    # THREE casts per call, not two, and the third is the one that is easy to miss.
    #   narrow y   read the fp32 tensor, write the bf16 one   1.5 passes
    #   narrow g   same                                        1.5
    #   widen dx   `backward()` normalises every gradient to its VALUE's dtype before firing
    #              that value's closure (`autograd.py:621-622`), and the softmax's input is the
    #              fp32 score tensor, so the bf16 dx is cast straight back up. 1.5
    # An earlier version of this model counted two and overstated the lever by a third of its
    # own cast bill. The cast-back is not removable by narrowing earlier -- only by taking the
    # CONSUMER to bf16 as well, which is a different and larger change.
    score_fp32 = max((_b(d) for r in smb for d in ops[r["i"]]["ins"]
                      if d["dtype"] == "FLOAT32"), default=0.0)
    narrow = 3 * calls * (score_fp32 * 1.5)
    return {"smbf16": {"bucket_now_GB": round(now / 1e9, 3),
                       "bucket_at_bf16_GB": round(after / 1e9, 3),
                       "narrowing_cost_GB": round(narrow / 1e9, 3),
                       "casts_per_call": 3, "calls": calls,
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
    # SORTED, and it is not cosmetic: the artifact is committed, so a run-to-run key reordering
    # makes `git diff` unable to tell a changed number from a shuffled dict. Python randomises
    # string hashing per process, so an unsorted set iteration produced exactly that -- 91
    # insertions and 91 deletions with every value identical.
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, {}).get("GB", 0.0), b.get(k, {}).get("GB", 0.0)
        if x <= 0 or y <= 0:
            continue
        e = math.log(y / x) / math.log(hi / lo)
        out[k] = {"GB_lo": x, "GB_hi": y, "exponent": round(e, 3),
                  "at": {str(n): round(y * (n / hi) ** e, 3) for n in at}}
    return out


def seconds(evo_gb, extra_gb, args):
    """Bytes into seconds, and the ceiling that puts on every byte lever on this page.

    Three measured inputs, none of them this row's:

      424.7 GB/s   the Blackhole DRAM roof `bcx-intensity` measured, and the roof it showed every
                   part of the gradient step is bound by -- machine balance 272.4 FLOP/byte
                   against the most arithmetically intense part's AI of 47.7
      131.9 ms     device kernels per AF2 Evoformer block backward at n=256, and 115.9 for
                   extra-MSA, `bcx-bytes` `psum_prof_arms.json`, Tracy build, AICLK 1350
      4 + 48       the blocks in one checkpointed BC2 gradient step

    The first thing the division says is the important one, and it is a bound rather than an
    estimate. The census counts depth-0 calls only, so its byte figure is a LOWER bound on DRAM
    traffic; dividing a lower bound on bytes by a measured time gives a LOWER bound on achieved
    bandwidth. If that already sits near the roof there is no efficiency win hiding under these
    kernels, and the most any byte lever can return is the share of bytes it removes.
    """
    roof = args.roof
    out = {"roof_GB_s": roof, "note": "achieved is a LOWER bound: the census sees depth-0 only"}
    for name, gb, ms in (("evo", evo_gb, args.evo_ms), ("extra", extra_gb, args.extra_ms)):
        ach = gb / (ms / 1e3)
        out[name] = {"GB_n256": round(gb, 3), "device_ms_n256": ms,
                     "achieved_GB_s_lower_bound": round(ach, 1),
                     "share_of_roof_lower_bound": round(ach / roof, 3)}
    return out


def cmd_staleness(args):
    """How much of this census sits in ops whose IMPLEMENTATION main has since changed.

    The traces were taken on `bcx-bytes`' tree, which predates main's `_tree_sum`/`TREE_REDUCE`
    and `_permute_back`/`REBLOCK_PERMUTE_BW` — both of which are LIVE on main. An earlier version
    of `DISPOSITION.md` asserted the reach map was unaffected by that. Asserting is not checking.

    Two classes, and they are not the same kind of affected:

      permutes   `_permute_back` swaps one kernel for another over the SAME operands, so the byte
                 count does not move at all. Only the kernel does.
      leading-axis sums  `_tree_sum` replaces one `ttnn.sum` with a tree of adds. That DOES move
                 the census, and upward, while real DRAM traffic stays the same or falls — the
                 census sees the tree's depth-0 adds but not `ttnn.sum`'s nested permute. It is
                 the instrument inversion this file's header already warns about.

    `softmax_bw_inner`'s reduces are NOT in either class and it matters: they are LAST-dim, and
    `_reduce_to` gates the tree on `ax < len(gs) - 2`. A first cut of this check counted them and
    reported 7.2 % affected where the true figure is 1.4 %.
    """
    import json as _json
    t = _json.loads((TR / f"{args.traces.split(',')[0]}.json").read_text())
    n = int(t["n"])
    rows = census(t["bwd"])
    tot = moved = 0.0
    byclass = collections.defaultdict(float)
    for r in rows:
        if r["view"]:
            continue
        moved += r["moved"]
        site = r["tape"] or ""
        if r["op"] == "sum" and ("_reduce_to" in site or "_sum_leading" in site):
            byclass["leading_sum_census_moves"] += r["moved"]
        elif r["op"] == "permute":
            byclass["permute_same_bytes"] += r["moved"]
    out = {"trace": args.traces.split(",")[0], "n": n, "moved_GB": round(moved / 1e9, 3),
           "classes": {k: round(v / 1e9, 3) for k, v in byclass.items()}}
    shift = byclass["leading_sum_census_moves"]
    out["census_would_shift_GB"] = round(shift / 1e9, 3)
    out["census_would_shift_share"] = round(shift / moved, 4)
    print(f"{out['trace']}  {out['moved_GB']} GB")
    for k, v in out["classes"].items():
        print(f"   {v:7.3f} GB  {k}")
    print(f"   -> the census total would shift by {out['census_would_shift_GB']} GB "
          f"= {out['census_would_shift_share'] * 100:.1f} % on main's tree, and UPWARD, "
          f"while DRAM traffic does not rise")
    pathlib.Path(args.out.replace("reach.json", "staleness.json")).write_text(
        _json.dumps(out, indent=1, sort_keys=True))
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
    sec = seconds(blob["evo"]["totals"]["256"], blob["extra"]["totals"]["256"], args)
    blob["seconds"] = sec
    print("\nbytes into seconds, against bcx-intensity's measured DRAM roof")
    for k in ("evo", "extra"):
        v = sec[k]
        print(f"   {k:6s} {v['GB_n256']:7.3f} GB / {v['device_ms_n256']:.1f} ms = "
              f">= {v['achieved_GB_s_lower_bound']:.1f} GB/s, "
              f">= {v['share_of_roof_lower_bound'] * 100:.1f} % of the {sec['roof_GB_s']} GB/s roof")
    print("   so there is no efficiency win under these kernels: a byte lever returns at most the")
    print("   share of bytes it removes, and no more.")

    n = str(args.step_n)
    ev = sum(blob["evo"]["levers"][k]["at"][n] for k in blob["evo"]["levers"])
    ex = sum(blob["extra"]["levers"][k]["at"][n] for k in blob["extra"]["levers"])
    evt, ext = blob["evo"]["totals"][n], blob["extra"]["totals"][n]
    dev_ms = (args.evo_blocks * args.evo_ms * (evt / blob["evo"]["totals"]["256"])
              + args.extra_blocks * args.extra_ms * (ext / blob["extra"]["totals"]["256"]))
    cut = (args.evo_blocks * args.evo_ms * (ev / blob["evo"]["totals"]["256"])
           + args.extra_blocks * args.extra_ms * (ex / blob["extra"]["totals"]["256"]))
    blob["step"] = {"n": args.step_n, "blocks": [args.extra_blocks, args.evo_blocks],
                    "device_ms_modelled": round(dev_ms, 1), "removed_ms": round(cut, 1),
                    "device_speedup": round(dev_ms / (dev_ms - cut), 4)}
    print(f"\none {args.extra_blocks}+{args.evo_blocks} gradient step at n={args.step_n}: "
          f"{dev_ms:.0f} ms of device, of which the precision stack removes {cut:.0f} ms "
          f"= {dev_ms / (dev_ms - cut):.3f}x on the DEVICE side")
    print("   and end to end, as a function of the device share of the round -- NOT inherited,")
    print("   because the two shares on the record were measured on two different trees.")
    print("   QUOTE perf/bcx_bwbytes/power.py INSTEAD for the headline: the sweep below applies")
    print("   the lever to ALL of the round's device time, but these levers touch only the")
    print("   BACKWARD, which is 67.7 % of what the card does per round. power.py uses the")
    print("   measured 0.726 device share and that 67.7 %, and reads lower and right.")
    for share in [float(x) for x in args.shares.split(",")]:
        r = 1.0 / ((1 - share) + share / (dev_ms / (dev_ms - cut)))
        blob.setdefault("round", {})[f"{share:.2f}"] = round(r, 4)
        print(f"     device {share * 100:4.1f} % of the round  ->  {r:.3f}x")

    # The ceiling of the kernel programme, on the two buckets worth a kernel. Pass counts of the
    # bucket's own tensor, read off the shipped source, not guesses. Both are LOWER bounds on the
    # traffic for the same reason the census is: `ttnn.sum(dim=0)` permutes before it reduces and
    # the count below charges it one read.
    #
    # SOFTMAX BACKWARD, score-sized tensor, `autograd.softmax_bw_dx`:
    #   multiply(g,y) 2r+1w, sum(.,-1) 1r, sum(y,-1) 1r, subtract(g,inner) 1r+1w,
    #   multiply(y,.) 2r+1w  = 10, all fp32; the divide is O(n^2) and negligible.
    #   One fused kernel reads y, reads g, writes dx: 3 passes, 1.5 fp32-equivalents in bf16, and
    #   unlike the smbf16 lever it pays no narrowing cast because the narrowing is inside.
    #
    # LAYER-NORM BACKWARD, activation-sized tensor, `autograd.layer_norm`'s closure at
    # autograd.py:1146-1163. Twenty reads and ten writes:
    #   mean(x) 1r; subtract(x,mean) 1r+1w; multiply(c,c) 2r+1w; mean(.) 1r;
    #   multiply(c,rstd) 1r+1w; multiply(g,norm) 2r+1w; _sum_leading(.) 1r; _sum_leading(g) 1r;
    #   multiply(g,gamma) 1r+1w; mean(dnorm) 1r; multiply(dnorm,norm) 2r+1w; mean(.) 1r;
    #   subtract(dnorm,dn_mean) 1r+1w; multiply(norm,dn_norm_mean) 1r+1w; subtract(.,.) 2r+1w;
    #   multiply(dx,rstd) 1r+1w   = 30.
    #   `ttml::metal::layernorm_bw` returns {dx, dgamma, dbeta} and computes the per-row
    #   dgamma/dbeta partials IN the kernel, leaving one reduction over rows: 3 passes plus a
    #   small reduce, so 4 is the conservative figure used here. The mean/rstd it needs are
    #   already recomputed by our own backward, which is the blocker that ruled out the moreh
    #   route and does not bind here. It is Route B -- a tt-metal source build, two copies of the
    #   runtime in one process -- so the cost is real and is NOT a kernel problem.
    # The fused figures below are what the KERNEL costs. A kernel that writes bf16 -- which
    # `moreh_softmax_backward` does, its output dtype following its input -- pays one more thing
    # on the way out: `backward()` normalises every gradient to its VALUE's dtype before that
    # value's closure fires (`autograd.py:621-622`), and the softmax's input is the fp32 score
    # tensor, so a bf16 dx is read back (0.5) and written as fp32 (1.0), 1.5 more passes. A
    # HAND-WRITTEN kernel can simply write fp32 and skip it, which is a design choice worth
    # stating before anyone writes one. So Route A's ceiling is strictly lower than a custom
    # kernel's at the same arithmetic, and `fused_softmax_wheel` below is the honest figure for
    # the moreh route.
    KERNELS = {"2 ": ("fused_softmax", 10, 1.5), "3 ": ("fused_layernorm", 30, 4.0)}
    WHEEL = {"fused_softmax": 1.5 + 1.5}
    fused, detail = {}, {}
    for stack in ("evo", "extra"):
        b = blob[stack]["by_site"]
        tot = {str(x): 0.0 for x in at}
        for prefix, (label, now_p, fused_p) in KERNELS.items():
            v = next((vv for k, vv in b.items() if k.startswith(prefix)), None)
            if not v:
                continue
            got = {str(x): round(v["at"][str(x)] * (1 - fused_p / now_p), 3) for x in at}
            rec = {"passes_now": now_p, "passes_fused": fused_p, "removed_GB": got}
            if label in WHEEL:
                wp = WHEEL[label]
                rec["passes_wheel_incl_widen_dx"] = wp
                rec["removed_GB_wheel"] = {str(x): round(v["at"][str(x)] * (1 - wp / now_p), 3)
                                           for x in at}
            detail.setdefault(stack, {})[label] = rec
            for x in at:
                tot[str(x)] += got[str(x)]
        fused[stack] = {k: round(v, 3) for k, v in tot.items()}
    blob["fused_kernel_ceiling"] = {"per_kernel": detail, "removed_GB_total": fused}
    for stack in ("evo", "extra"):
        for label, d in detail.get(stack, {}).items():
            print(f"   {stack:6s} {label:16s} {d['passes_now']:.0f} passes -> {d['passes_fused']}"
                  f"  removes {d['removed_GB'][n]:7.3f} GB at n={n}"
                  + (f"   | wheel (bf16 out, +widen dx) -> {d['passes_wheel_incl_widen_dx']}"
                     f" removes {d['removed_GB_wheel'][n]:.3f} GB"
                     if "removed_GB_wheel" in d else ""))
    ev2 = ev + fused["evo"][n]
    ex2 = ex + fused["extra"][n]
    cut2 = (args.evo_blocks * args.evo_ms * (ev2 / blob["evo"]["totals"]["256"])
            + args.extra_blocks * args.extra_ms * (ex2 / blob["extra"]["totals"]["256"]))
    blob["step"]["with_fused_kernels"] = {
        "removed_ms": round(cut2, 1), "device_speedup": round(dev_ms / (dev_ms - cut2), 4)}
    print(f"\nand with BOTH fusable buckets given a real kernel, pass counts read off the source:")
    print(f"     removes {cut2:.0f} of {dev_ms:.0f} ms = {dev_ms / (dev_ms - cut2):.3f}x "
          f"on the DEVICE side at n={args.step_n}")
    for share in [float(x) for x in args.shares.split(",")]:
        r = 1.0 / ((1 - share) + share / (dev_ms / (dev_ms - cut2)))
        blob.setdefault("round_fused", {})[f"{share:.2f}"] = round(r, 4)
        print(f"     device {share * 100:4.1f} % of the round  ->  {r:.3f}x")

    pathlib.Path(args.out).write_text(_json.dumps(blob, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="trace_bwd-fix_evo_n256,trace_bwd-fix_extra_n256")
    ap.add_argument("--staleness", action="store_true",
                    help="how much of the census sits in ops main has since reimplemented")
    ap.add_argument("--scale", action="store_true",
                    help="per-bucket power law from the two traced sizes, evaluated at --at")
    ap.add_argument("--roof", type=float, default=424.7,
                    help="measured Blackhole DRAM roof, bcx-intensity")
    ap.add_argument("--evo-ms", type=float, default=131.9,
                    help="device ms per Evoformer block backward at n=256, bcx-bytes psum")
    ap.add_argument("--extra-ms", type=float, default=115.9)
    ap.add_argument("--evo-blocks", type=int, default=48)
    ap.add_argument("--extra-blocks", type=int, default=4)
    ap.add_argument("--step-n", type=int, default=224)
    ap.add_argument("--shares", default="0.20,0.40,0.70",
                    help="device share of the round to evaluate the end-to-end ratio at; the two "
                         "on the record (bcx-round 20.1 %%, bcx-tmplseam ~70 %%) were measured on "
                         "two different trees, so both are shown and neither is adopted")
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
    if args.staleness:
        cmd_staleness(args)
        return
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

    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
