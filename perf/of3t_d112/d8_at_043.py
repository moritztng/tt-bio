#!/usr/bin/env python3
"""D8's published tables, rescored against the 0.4.3 reference D8's own caveat asks for.

D8's pass-90 entry states its own gap: "the tb-off arm equalises on 0.5.0's side. The correct
configuration is our shipped flag against a 0.4.3 reference, which `of3t-rebase` produces."
`of3t-rebase` produced it and filed `score_arms_043.json`; D112 then pruned that row's worktree,
and the result never reached the defect it answers -- D8's entry still carries the 0.5.0 table
and the unanswered caveat. The arms themselves were committed before the prune, so the
substitution needs no card and no re-run.

Three regimes, scored with `perf/of3t_orchestrator/revision/d8_vs_endnode.py` unchanged (A14
zero-reference exclusion at 1e-12, per-tensor bar 5.0e-02, median bar 2.0e-02, over-bar share by
squared ref_norm):

  CROP64  D8's pass-90 six-arm table. Reproduces `score_arms_043.json` independently.
  N384    the full 384-token window at block 0, which `score_arms_043.json` left unscored
          (its `stacks` entry reads `against_0.4.3: null`).
  SUBMOD  D8's pass-47 decomposition of block 0 by sub-module, the table pass 85 builds its
          start/end asymmetry on. This is the one D8 calls "the sharpest statement the campaign
          has about D8".

COMPARABILITY is checked, not assumed. An arm pair is scored only if the crop, the checkpoint and
`scale_pair_bias` agree and the two references disagree on revision, so what moves is the
reference and the captured boundary it brings with it -- never the flag. The pair-bias flag split
between the two runs, and `of3t-rebase` quarantined its own mis-pinned arms for exactly that
reason; `arms/mispinned_spb_on/` holds copies here under a name that says so, because the
orchestrator's `block47/` directory carries the same six bytes under a name that does not.
"""
import json
import sys
from pathlib import Path

import numpy as np

BAR, MBAR, ZERO_REF = 5.0e-2, 2.0e-2, 1e-12
A = Path(__file__).parent / "arms"
OUT = Path(__file__).parent

CROP64 = [
    ("block 0  SHIPPED", "050_instrument_a_bundle_block0_r0_crop64.json",
     "instrument_a_bundle_043spb_block0_crop64_tbshipped.json"),
    ("block 0  tb-off ", "050_instrument_a_bundle_block0_r0_crop64_tboff.json",
     "instrument_a_bundle_043spb_block0_crop64_tboff.json"),
    ("block 23 SHIPPED", "050_instrument_a_bundle_block23_r0_crop64_tbshipped.json",
     "instrument_a_bundle_043spb_block23_crop64_tbshipped.json"),
    ("block 23 tb-off ", "050_instrument_a_bundle_block23_r0_crop64_tboff.json",
     "instrument_a_bundle_043spb_block23_crop64_tboff.json"),
    ("block 47 SHIPPED", "050_instrument_a_bundle_block47_r0_crop64_tbshipped.json",
     "instrument_a_bundle_043spb_block47_crop64_tbshipped.json"),
    ("block 47 tb-off ", "050_instrument_a_bundle_block47_r0_crop64_tboff.json",
     "instrument_a_bundle_043spb_block47_crop64_tboff.json"),
]
N384 = [("block 0  SHIPPED", "050_instrument_a_bundle_block0_r0.json",
         "instrument_a_bundle_043full_block0_n384_tbshipped.json")]

#: pass 47's grouping, in its own order. `pair_stack, the rest` is the remainder.
GROUPS = ["tri_att_end", "attn_pair_bias", "tri_att_start", "single_transition"]


def score(d):
    kept = [r for r in d["per_parameter"] if r["ref_norm"] >= ZERO_REF]
    return stats(kept)


def stats(kept):
    med = float(np.median(np.array([r["rel_l2"] for r in kept])))
    over = [r for r in kept if r["rel_l2"] > BAR]
    w = max(kept, key=lambda r: r["rel_l2"])
    sq = sum(r["ref_norm"] ** 2 for r in kept) or 1.0
    return {"n": len(kept), "median": med, "median_inside_bar": med <= MBAR,
            "over_bar": len(over),
            "over_bar_norm_share": sum(r["ref_norm"] ** 2 for r in over) / sq,
            "worst": w["rel_l2"], "worst_tensor": w["their_tensor"]}


def replay(d):
    """`block_grad_vs_bundle.worst_rel`: the instrument recomputing this block's gradients from
    the captured boundary and comparing them to the bundle it is about to score against. Any
    value but ~0 says the probe is not the reference's own boundary, and every magnitude taken
    on it inherits that."""
    bg = d["capture"].get("block_grad_vs_bundle", {})
    if "worst_rel" not in bg:
        bg = next(iter(bg.values())) if bg else {}
    return bg.get("worst_rel")


def rev(d):
    return d["bundle"].get("upstream_revision") or "0.5.0"


def check(a, b):
    why = []
    if (a.get("crop") or 0) != (b.get("crop") or 0):
        why.append(f"crop {a.get('crop')} vs {b.get('crop')}")
    if Path(a["checkpoint"]).name != Path(b["checkpoint"]).name:
        why.append("checkpoint")
    for k in ("scale_pair_bias", "transpose_bias", "fp32_softmax"):
        if a["shipped_config"].get(k) != b["shipped_config"].get(k):
            why.append(f"{k} {a['shipped_config'].get(k)} vs {b['shipped_config'].get(k)}")
    if a["probe"]["tokens"] != b["probe"]["tokens"]:
        why.append("probe.tokens")
    if rev(a) == rev(b):
        why.append(f"both references are {rev(a)} -- nothing is substituted")
    return why


def table(title, pairs, submod=False):
    print(f"\n=== {title} ===")
    hdr = (f"{'arm':17s} {'ref':>5s} {'n':>4s} {'median':>8s} {'':4s} {'over-bar':>9s} "
           f"{'share':>7s} {'replay':>8s}  worst")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tag, f050, f043 in pairs:
        a, b = json.load(open(A / f050)), json.load(open(A / f043))
        why = check(a, b)
        if why:
            print(f"{tag}: SKIPPED -- {'; '.join(why)}")
            rows.append({"arm": tag.strip(), "skipped": why})
            continue
        rec = {"arm": tag.strip(), "scale_pair_bias": a["shipped_config"]["scale_pair_bias"],
               "transpose_bias": a["shipped_config"]["transpose_bias"],
               "tokens": a["probe"]["tokens"], "crop": a.get("crop") or 0}
        for d in (a, b):
            s = score(d)
            print(f"{tag} {rev(d):>5s} {s['n']:4d} {s['median']:8.4f} "
                  f"{'PASS' if s['median_inside_bar'] else 'FAIL':4s} "
                  f"{s['over_bar']:4d}/{s['n']:<4d} {100 * s['over_bar_norm_share']:6.1f}% "
                  f"{replay(d):8.4f}  {s['worst']:.4f} @ {s['worst_tensor'].split('blocks.')[-1]}")
            rec[rev(d)] = {**s, "replay_worst_rel": replay(d),
                           "reference_sha256": d["bundle"].get("sha256"),
                           "reference_file": d["bundle"].get("file")}
        ra, rb = rec["0.5.0"], rec["0.4.3"]
        rec["median_ratio_043_over_050"] = rb["median"] / ra["median"] if ra["median"] else None
        rec["verdict_flip"] = ra["median_inside_bar"] != rb["median_inside_bar"]
        rec["worst_tensor_moves"] = ra["worst_tensor"] != rb["worst_tensor"]
        rows.append(rec)
        if submod:
            rec["by_submodule"] = submodule(a, b)
        print()
    return rows


def submodule(a, b):
    """Pass 47's grouping, recomputed on both references over the same tensor set."""
    out = {}
    ka = {r["their_tensor"]: r for r in a["per_parameter"] if r["ref_norm"] >= ZERO_REF}
    kb = {r["their_tensor"]: r for r in b["per_parameter"] if r["ref_norm"] >= ZERO_REF}
    shared = sorted(set(ka) & set(kb))
    named = set()
    for g in GROUPS:
        sel = [t for t in shared if f".{g}." in t]
        named |= set(sel)
        if sel:
            out[g] = {"0.5.0": stats([ka[t] for t in sel]), "0.4.3": stats([kb[t] for t in sel])}
    rest = [t for t in shared if t not in named]
    if rest:
        out["pair_stack, the rest"] = {"0.5.0": stats([ka[t] for t in rest]),
                                       "0.4.3": stats([kb[t] for t in rest])}
    out["_scope"] = {"n_shared": len(shared), "note": "tensors present and non-zero-reference "
                                                      "on BOTH sides, so the grouping compares "
                                                      "the same set at both revisions"}
    return out


def main():
    res = {"bars": {"per_tensor": BAR, "median": MBAR}, "a14_zero_ref": ZERO_REF,
           "scoring": "of3t-orchestrator revision/d8_vs_endnode.py, unchanged",
           "crop64": table("CROP64 -- D8's pass-90 six-arm table", CROP64),
           "n384": table("N384 -- the full 384-token window at block 0", N384, submod=True)}

    print("\n=== D8's pass-47 sub-module decomposition of block 0, at both references ===")
    sm = res["n384"][0]["by_submodule"]
    print(f"{'group':22s} {'n':>3s} {'med 0.5.0':>10s} {'med 0.4.3':>10s} {'ratio':>7s} "
          f"{'over-bar 0.5.0':>15s} {'over-bar 0.4.3':>15s}")
    for g, v in sm.items():
        if g.startswith("_"):
            continue
        x, y = v["0.5.0"], v["0.4.3"]
        print(f"{g:22s} {x['n']:3d} {x['median']:10.4f} {y['median']:10.4f} "
              f"{y['median'] / x['median']:7.3f} {x['over_bar']:8d}/{x['n']:<6d} "
              f"{y['over_bar']:8d}/{y['n']:<6d}")
    # pass 85's start/end asymmetry, the statistic it published
    for stat in ("median", "worst"):
        e, s = sm["tri_att_end"], sm["tri_att_start"]
        print(f"pass-85 end/start on {stat:6s}: 0.5.0 {e['0.5.0'][stat] / s['0.5.0'][stat]:.2f}x"
              f"   0.4.3 {e['0.4.3'][stat] / s['0.4.3'][stat]:.2f}x")
        res.setdefault("pass85_end_over_start", {})[stat] = {
            "0.5.0": e["0.5.0"][stat] / s["0.5.0"][stat],
            "0.4.3": e["0.4.3"][stat] / s["0.4.3"][stat]}

    json.dump(res, open(OUT / "D8_AT_043.json", "w"), indent=1)
    flips = [r["arm"] for r in res["crop64"] + res["n384"] if r.get("verdict_flip")]
    moves = [r["arm"] for r in res["crop64"] + res["n384"] if r.get("worst_tensor_moves")]
    print(f"\nmedian-bar verdict flips: {flips or 'none'}")
    print(f"worst-tensor identity moves: {moves or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
