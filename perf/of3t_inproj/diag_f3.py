#!/usr/bin/env python3
"""of3t-inproj F3 root cause: every TriangleMultiplication's in-projection cache after backward.

    diag_f3.py --batch B.pt --draws draws.pt --out F.json

Runs fullstep64's step (trainfwd_run, draws replayed) and, after the backward, reports per trimul
every `_gp_cache` / `_gp_gout_cache` entry: its key, whether the first walk (the one that
registers leaves) saw it, whether it is a registered leaf now, and whether it holds a gradient.
Also the norm_in gradients, which reach a trimul only through its in-projection's dx.
"""
import json
import os
import sys
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    get = lambda k: argv[argv.index(k) + 1]  # noqa: E731
    out = Path(get("--out"))
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[0:0] = [os.path.join(os.getcwd(), "perf/of3t_fullstep64"),
                     os.path.join(os.getcwd(), "perf/of3t_trainfwd")]
    import torch
    from draws import Draws
    from tt_bio import autograd as ag
    from tt_bio import openfold3_fold
    from tt_bio.tenstorrent import TriangleMultiplication, walk_device_weights
    from tt_bio.train.openfold3 import OpenFold3Forward
    import trainfwd_run

    replay = torch.load(get("--draws"), weights_only=False)["torch_randn"]
    orig = openfold3_fold.OpenFold3._gen_rollout
    mism = []

    def gen_rollout(self, *a, **k):
        with Draws(replay) as rec:
            got = orig(self, *a, **k)
        mism.extend(rec.mismatch)
        return got
    openfold3_fold.OpenFold3._gen_rollout = gen_rollout

    first = {}
    orig_params = OpenFold3Forward.parameters

    def parameters(self):
        got = orig_params(self)
        if not first:
            first["ids"] = {id(t) for t in got.values()}
            first["n"] = len(got)
            first["paths"] = set(got)
        first["model"] = self.model
        return got
    OpenFold3Forward.parameters = parameters

    report = {}
    orig_snap = trainfwd_run.grad_snapshot

    def snap(params):
        m = first["model"]
        rows = []
        seen = set()

        def trimuls(obj, prefix="", depth=0):
            if depth > 12 or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, TriangleMultiplication):
                yield prefix, obj
                return
            items = (obj.items() if isinstance(obj, dict) else enumerate(obj)
                     if isinstance(obj, (list, tuple)) else
                     vars(obj).items() if hasattr(obj, "__dict__") and not isinstance(obj, type)
                     else ())
            for k, v in items:
                if isinstance(v, (str, bytes, int, float, torch.Tensor)) or v is None:
                    continue
                yield from trimuls(v, f"{prefix}{k}.", depth + 1)

        def leaf(t):
            p = ag.parameter_for(t)
            return {"in_first_walk": id(t) in first["ids"], "leaf": p is not None,
                    "grad": None if p is None or p.grad is None else float(
                        (ttnn.to_torch(p.grad).double() ** 2).sum())}
        import ttnn
        for path, tm in trimuls(m):
            ent = {"gp_cache": {str(k): [leaf(t) for t in v] for k, v in tm._gp_cache.items()},
                   "gp_gout_cache": {str(k): leaf(v) for k, v in tm._gp_gout_cache.items()},
                   "in_norm_weight": leaf(tm.in_norm_weight),
                   "g_out_weight": leaf(tm.g_out_weight)}
            rows.append({"trimul": path, **ent})
        report["trimuls"] = rows
        report["first_walk_n"] = first["n"]
        report["walk_after_n"] = len(params)
        late = sorted(k for k in params if k not in first["paths"])
        report["late"] = [{"path": k, **leaf(params[k])} for k in late]
        return orig_snap(params)
    trainfwd_run.grad_snapshot = snap

    sys.argv = ["trainfwd_run.py", "--arm", "full", "--out", str(out) + ".arm.json",
                "--batch", get("--batch")]
    rc = trainfwd_run.main()
    rows = report.get("trimuls", [])
    summ = {"n_trimul": len(rows), "first_walk_n": report.get("first_walk_n"),
            "walk_after_n": report.get("walk_after_n"), "draw_mismatch": len(mism),
            "cache_entries": sum(len(v) for r in rows for v in r["gp_cache"].values())
            + sum(len(r["gp_gout_cache"]) for r in rows),
            "cache_in_first_walk": sum(e["in_first_walk"] for r in rows
                                       for v in r["gp_cache"].values() for e in v)
            + sum(e["in_first_walk"] for r in rows for e in r["gp_gout_cache"].values()),
            "cache_leaf": sum(e["leaf"] for r in rows for v in r["gp_cache"].values() for e in v)
            + sum(e["leaf"] for r in rows for e in r["gp_gout_cache"].values()),
            "cache_with_grad": sum(e["grad"] is not None for r in rows
                                   for v in r["gp_cache"].values() for e in v)
            + sum(e["grad"] is not None for r in rows for e in r["gp_gout_cache"].values()),
            "norm_in_with_grad": sum(r["in_norm_weight"]["grad"] is not None for r in rows),
            "g_out_with_grad": sum(r["g_out_weight"]["grad"] is not None for r in rows)}
    late = report.get("late", [])
    summ["late_n"] = len(late)
    summ["late_leaf"] = sum(e["leaf"] for e in late)
    summ["late_with_grad"] = sum(e["grad"] is not None for e in late)
    out.write_text(json.dumps({"summary": summ, "late": late, "trimuls": rows}, indent=1) + "\n")
    print("DIAG " + json.dumps(summ), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
