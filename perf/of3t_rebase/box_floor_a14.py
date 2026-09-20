#!/usr/bin/env python3
"""Put the reference's own box-to-box floor and the trunk arms on the SAME tensor set.

`replay_vs_r0.json` measures upstream against upstream -- the same revision, the same batch, the
same replayed draws, a CPU replay scored against the A100 tape -- so it is the floor any trunk
arm sits on. Published, it reads median 5.707e-02 at block 0, which is ABOVE what the `tb-off`
arm reads there (1.15e-02). A comparison cannot be tighter than the floor under it, so one of
the two is measured on a set the other is not.

It is the set. The floor keeps all 57 tensors per block; the arms apply PROTOCOL A14 and drop
the ones whose reference norm is below 1e-12, where a relative L2 is a division by nothing. The
floor's worst tensor at all three blocks is `attn_pair_bias.layer_norm_z.bias`, whose reference
norm is 1.089e-19 -- seven orders under the A14 floor, and exactly the tensor A14 removes.

So this re-reads the floor with A14 applied, from the arms' own per-tensor reference norms, and
reports both numbers side by side. Nothing is recomputed from a model; it is arithmetic over two
artifacts already in the branch.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

A14_FLOOR = 1e-12
BAR = 5.0e-02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path,
                    default=Path("perf/of3t_gradients/replay_vs_r0.json"))
    ap.add_argument("--arms", nargs="+", type=Path,
                    default=[Path(f"perf/of3t_gradients/"
                                  f"instrument_a_bundle_block{b}_r0_crop64_tboff.json")
                             for b in (0, 23, 47)])
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_rebase/box_floor_a14.json"))
    a = ap.parse_args()

    floor = json.loads(a.floor.read_text())["refs"]["republished_r0"]
    per_tensor = floor["per_tensor"]

    # The arms carry the reference norm per tensor, which is what A14 thresholds on.
    ref_norm: dict[str, float] = {}
    for f in a.arms:
        for row in json.loads(f.read_text())["per_parameter"]:
            ref_norm[row["their_tensor"]] = float(row["ref_norm"])

    rows, excluded = {}, []
    for name, rel in per_tensor.items():
        blk = name.split("pairformer_stack.blocks.")[1].split(".")[0]
        rn = ref_norm.get(name)
        keep = rn is None or rn >= A14_FLOOR
        if not keep:
            excluded.append({"tensor": name, "ref_norm": rn, "rel_l2_in_floor": rel})
        rows.setdefault(blk, {"all": [], "a14": []})["all"].append((name, rel))
        if keep:
            rows[blk]["a14"].append((name, rel))

    def stat(pairs):
        v = [r for _, r in pairs]
        w = max(pairs, key=lambda p: p[1]) if pairs else (None, None)
        return {"n": len(v), "median": statistics.median(v) if v else None,
                "over_bar_5e-2": sum(1 for x in v if x > BAR),
                "worst": w[1], "worst_tensor": w[0]}

    rep = {
        "instrument": "the reference's own box-to-box floor, with and without PROTOCOL A14",
        "floor_source": str(a.floor),
        "floor_is": "upstream CPU replay against the upstream A100 tape: same revision, same "
                    "batch, same replayed draws. What moves is the box.",
        "reference_sha256": floor["sha256"],
        "a14_floor_on_reference_norm": A14_FLOOR,
        "ref_norm_source": [str(f) for f in a.arms],
        "by_block": {b: {"all_tensors": stat(v["all"]), "a14_applied": stat(v["a14"])}
                     for b, v in sorted(rows.items(), key=lambda kv: int(kv[0]))},
        "excluded_by_a14": sorted(excluded, key=lambda e: -e["rel_l2_in_floor"]),
    }
    # The real like-for-like: the floor restricted to the arm's OWN tensor set, not merely to
    # A14. The arms compare 52 of the 57 the floor carries -- the five `attn_pair_bias.mha.*`
    # and the degenerate `layer_norm_z.bias` are not in their bijection -- so even with A14
    # applied the two are scored over different tensors until this is done.
    matched = {}
    for f in a.arms:
        arm = json.loads(f.read_text())
        blk = str(arm["block"])
        names = [r["their_tensor"] for r in arm["per_parameter"]
                 if float(r["ref_norm"]) >= A14_FLOOR]
        arm_rel = [float(r["rel_l2"]) for r in arm["per_parameter"]
                   if float(r["ref_norm"]) >= A14_FLOOR]
        fl = [(n, per_tensor[n]) for n in names if n in per_tensor]
        matched[blk] = {
            "arm_file": f.name,
            "n_arm": len(names), "n_floor_on_arm_set": len(fl),
            "arm_tensors_absent_from_floor": [n for n in names if n not in per_tensor],
            "floor_tensors_absent_from_arm":
                sorted(n for n in per_tensor
                       if n.startswith(f"pairformer_stack.blocks.{blk}.")
                       and n not in set(names)),
            "floor_on_arm_set": stat(fl),
            "arm": stat(list(zip(names, arm_rel))),
        }
    rep["set_matched"] = matched
    rep["reading"] = (
        "With the sets identical the arm is TIGHTER than the floor at blocks 0 and 23, which a "
        "floor cannot be. The two contrasts are not independent: the floor is "
        "(upstream on CPU - reference on A100) and the arm is (ours on CPU - reference on "
        "A100), and both carry the same CPU-side displacement, so they share a term and "
        "partially cancel. The published box figure is therefore NOT a floor for these arms. "
        "The floor an arm actually sits on is (ours - upstream) with the box held fixed, which "
        "is what a bundle rebuilt on the same host as the capture gives for free.")

    allp = [p for v in rows.values() for p in v["all"]]
    a14p = [p for v in rows.values() for p in v["a14"]]
    rep["overall"] = {"all_tensors": stat(allp), "a14_applied": stat(a14p)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")

    print(f"{'block':>6} {'n':>4} {'median':>11} {'over bar':>9} {'worst':>10}   rule")
    for b, v in rep["by_block"].items():
        for rule, s in (("as published", v["all_tensors"]), ("A14 applied", v["a14_applied"])):
            print(f"{b:>6} {s['n']:>4} {s['median']:>11.4e} "
                  f"{s['over_bar_5e-2']:>4}/{s['n']:<4} {s['worst']:>10.4g}   {rule}")
    for rule, s in (("as published", rep["overall"]["all_tensors"]),
                    ("A14 applied", rep["overall"]["a14_applied"])):
        print(f"{'all':>6} {s['n']:>4} {s['median']:>11.4e} "
              f"{s['over_bar_5e-2']:>4}/{s['n']:<4} {s['worst']:>10.4g}   {rule}")
    print(f"\nexcluded by A14 ({len(excluded)}):")
    for e in rep["excluded_by_a14"]:
        print(f"  {e['tensor']}  ref_norm {e['ref_norm']:.3e}  "
              f"reads {e['rel_l2_in_floor']:.4g} in the floor")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
