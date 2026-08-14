#!/usr/bin/env python3
"""Screen the device-resident z lever: measure, per wrapper INSTANCE, what the crossings around
`parcae_coda` actually cost, and what dtype each side is in.

The coarse census keyed by `file:line` merged three `FoldingTrunk` instances (lm_encoder,
parcae_coda, the confidence head's trunk) into one row of 3 calls, so the parcae_coda share had to
be pro-rated. This keys on the instance as well, using the block count to tell them apart, and
records the device-side dtype of every crossing -- which is what decides whether skipping the round
trip is bit-exact by construction (bf16 -> fp32 -> bf16 is lossless) or a narrowing cast that has
to be reproduced explicitly.

One warm fold, one clean fold (the reference wall), one instrumented fold, same process.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "esmfold2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "esmfold2")

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "size": a.size,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, state = B.build_fold("esmfold2", ROOT / (".msa_ab512_%d" % a.size), tgt, a3m)
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]

    res["warm_s"] = round(one_fold()[0], 3)
    clean_s, m = one_fold()
    res["clean_s"] = round(clean_s, 3)
    res["plddt"] = m.get("plddt")

    sites = defaultdict(lambda: {"n": 0, "s": 0.0, "shapes": set(), "dtypes": set()})
    orig_to, orig_from = T.TorchWrapper._to_torch, T.TorchWrapper._from_torch

    def who(self):
        cls = type(self).__name__
        mod = getattr(self, "module", None)
        blocks = getattr(mod, "blocks", None)
        nb = len(blocks) if blocks is not None else None
        return "%s%s" % (cls, "" if nb is None else "[%d blocks]" % nb)

    def site_of():
        f = sys._getframe(2)
        return "%s:%d" % (f.f_code.co_filename.split("tt_bio/")[-1], f.f_lineno)

    def wrap(fn, direction):
        def inner(self, x, *args, **kw):
            key = "%s %-28s %s" % (direction, who(self), site_of())
            t0 = time.perf_counter()
            out = fn(self, x, *args, **kw)
            dt = time.perf_counter() - t0
            e = sites[key]
            e["n"] += 1
            e["s"] += dt
            e["shapes"].add(str(tuple(int(v) for v in x.shape)))
            # device side of the crossing: the input for a download, the output for an upload
            dev_t = x if direction == "to_torch" else out
            e["dtypes"].add(str(getattr(dev_t, "dtype", "?")))
            return out
        return inner

    T.TorchWrapper._to_torch = wrap(orig_to, "to_torch")
    T.TorchWrapper._from_torch = wrap(orig_from, "from_torch")
    inst_s, m2 = one_fold()
    T.TorchWrapper._to_torch, T.TorchWrapper._from_torch = orig_to, orig_from

    res["instrumented_s"] = round(inst_s, 3)
    res["instrument_overhead_pct"] = round(100.0 * (inst_s - clean_s) / clean_s, 3)
    res["plddt_instrumented"] = m2.get("plddt")
    rows = [{"site": k, "n": v["n"], "s": round(v["s"], 4),
             "shapes": sorted(v["shapes"])[:3], "dtypes": sorted(v["dtypes"])}
            for k, v in sites.items()]
    rows.sort(key=lambda r: -r["s"])
    res["sites"] = rows
    res["total_calls"] = sum(r["n"] for r in rows)
    res["total_s"] = round(sum(r["s"] for r in rows), 3)
    a.out.write_text(json.dumps(res, indent=1))
    print("clean %.3fs instrumented %.3fs (+%.2f%%) crossings %d costing %.3fs"
          % (clean_s, inst_s, res["instrument_overhead_pct"], res["total_calls"], res["total_s"]))
    for r in rows[:26]:
        print("  %-72s n=%-4d %7.3fs %-14s %s"
              % (r["site"], r["n"], r["s"], ",".join(r["dtypes"]), r["shapes"][0]))


if __name__ == "__main__":
    main()
