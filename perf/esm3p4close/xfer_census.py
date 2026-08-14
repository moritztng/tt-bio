#!/usr/bin/env python3
"""L-B: census the host<->device boundary of a 512 aa ESMFold2 fold, by CALL SITE.

esmfold2-to-4x.md measured 242 crossings costing 2.908 s but never asked how many of them are
structurally necessary. 242 is a count of calls, not of sites. This aggregates every
TorchWrapper._to_torch / ._from_torch by the frame that called it, so each site can be classified
by hand as a pure view/elementwise (deletable bit-exactly) or a reduction/contraction (not
deletable without changing the sum order).

Runs a warm fold, then a clean fold whose wall is the reference, then the instrumented fold, so
the instrument cost is stated rather than assumed.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "model": a.model, "size": a.size,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, state = B.build_fold(a.model, ROOT / (".msa_ab512_%d" % a.size), tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]

    print("warm", flush=True)
    res["warm_s"] = round(one_fold()[0], 3)
    print("clean", flush=True)
    clean_s, m = one_fold()
    res["clean_s"] = round(clean_s, 3)
    res["plddt"] = m.get("plddt")

    sites = defaultdict(lambda: {"n": 0, "s": 0.0, "elems": 0, "shapes": set()})
    orig_to, orig_from = T.TorchWrapper._to_torch, T.TorchWrapper._from_torch

    def site_of():
        f = sys._getframe(2)
        return "%s:%d" % (f.f_code.co_filename.split("tt_bio/")[-1], f.f_lineno)

    def wrap(fn, direction):
        def inner(self, x, *args, **kw):
            key = "%s %s" % (direction, site_of())
            t0 = time.perf_counter()
            out = fn(self, x, *args, **kw)
            dt = time.perf_counter() - t0
            e = sites[key]
            e["n"] += 1
            e["s"] += dt
            shp = tuple(int(v) for v in x.shape)
            e["shapes"].add(str(shp))
            n = 1
            for v in shp:
                n *= v
            e["elems"] += n
            return out
        return inner

    T.TorchWrapper._to_torch = wrap(orig_to, "to_torch")
    T.TorchWrapper._from_torch = wrap(orig_from, "from_torch")
    print("instrumented", flush=True)
    inst_s, m2 = one_fold()
    T.TorchWrapper._to_torch, T.TorchWrapper._from_torch = orig_to, orig_from

    res["instrumented_s"] = round(inst_s, 3)
    res["instrument_overhead_pct"] = round(100.0 * (inst_s - clean_s) / clean_s, 3)
    res["plddt_instrumented"] = m2.get("plddt")
    rows = [{"site": k, "n": v["n"], "s": round(v["s"], 4),
             "mb_fp32": round(v["elems"] * 4 / 1e6, 1), "shapes": sorted(v["shapes"])[:4]}
            for k, v in sites.items()]
    rows.sort(key=lambda r: -r["s"])
    res["sites"] = rows
    res["total_calls"] = sum(r["n"] for r in rows)
    res["total_s"] = round(sum(r["s"] for r in rows), 3)
    a.out.write_text(json.dumps(res, indent=1))
    print("clean %.3fs  instrumented %.3fs (+%.2f%%)  crossings %d costing %.3fs"
          % (clean_s, inst_s, res["instrument_overhead_pct"], res["total_calls"], res["total_s"]))
    for r in rows[:25]:
        print("  %-58s n=%-4d %7.3fs %9.1f MB" % (r["site"], r["n"], r["s"], r["mb_fp32"]))


if __name__ == "__main__":
    main()
