#!/usr/bin/env python3
"""In-fold site wall for the batched-matmul per_core_M rule, shipped vs occupancy-first.

One process, one device context, one `build_fold` per target; the arm is a module-global flip plus
an `lru_cache` clear between folds, so the arms share weights, MSA cache and program cache. The
instrument is the SITE wall: `batched_matmul` synchronised on both sides and summed per shape
class over its real executions in the fold, not a per-call figure multiplied by a census. The fold
wall is reported too but it cannot resolve these deltas and is not the headline.

`batched_matmul` is imported into several model namespaces, so the wrapper is bound into every
module that holds a reference, not just `tt_bio.tenstorrent` -- rebinding only the definition site
would score a silent no-op.

Arms run cur / occ / cur, so the run carries its own A/A floor on the same fixture and the same
device context.
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
STATE = {"dev": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--arms", default="cur,occ,cur")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recycles", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    dev = T.get_device()
    STATE["dev"] = dev
    T._configure_active_compute_grid(dev)
    ORIG = T.batched_matmul

    def spy(x, y, compute_kernel_config=None, dtype=None):
        sa, sb = tuple(x.shape), tuple(y.shape)
        key = f"{'x'.join(str(int(d)) for d in sa)}@{'x'.join(str(int(d)) for d in sb)}|{x.dtype}"
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = ORIG(x, y, compute_kernel_config=compute_kernel_config, dtype=dtype)
        ttnn.synchronize_device(dev)
        w = WALL[key]
        w["n"] += 1
        w["s"] += time.perf_counter() - t0
        return out

    bound = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("tt_bio"):
            continue
        if getattr(mod, "batched_matmul", None) is ORIG:
            setattr(mod, "batched_matmul", spy)
            bound.append(name)
    T.batched_matmul = spy
    print(f"[ab] wrapper bound in {bound}", flush=True)

    if a.recycles:
        B.RECYCLING_STEPS = a.recycles
    if a.steps:
        B.SAMPLING_STEPS = a.steps
    fixdir = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, _st = B.build_fold(
        a.model, Path(f"/tmp/bmm-ab-{a.model}-{a.size}"),
        fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m")

    out = {"model": a.model, "size": a.size, "hardware": meta.get("hardware"),
           "grid": list(T.COMPUTE_GRID_MAIN),
           "recycles": B.RECYCLING_STEPS, "steps": B.SAMPLING_STEPS, "arms": []}

    def set_arm(arm):
        T._BATCHED_MATMUL_SELECT = "occ" if arm == "occ" else "blocks32"
        T._batched_matmul_search.cache_clear()

    # cold fold on the shipped arm: warms every kernel for both arms' shapes is not automatic, so
    # each arm's first fold also compiles its own configs. Run one cold fold per arm value first.
    for arm in sorted(set(a.arms.split(","))):
        set_arm(arm)
        WALL.clear()
        t0 = time.perf_counter()
        one_fold()
        print(f"[ab] cold {arm} {time.perf_counter()-t0:.1f}s", flush=True)

    for i, arm in enumerate(a.arms.split(",")):
        set_arm(arm)
        WALL.clear()
        s, m = one_fold()
        sites = {k: {"n": v["n"], "ms": round(v["s"] * 1e3, 2),
                     "ms_per_call": round(v["s"] * 1e3 / v["n"], 5)}
                 for k, v in sorted(WALL.items())}
        total = round(sum(v["ms"] for v in sites.values()), 2)
        cifs = sorted(Path(meta["struct_dir"]).glob("*.cif"))
        sha = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else None
        rec = {"i": i, "arm": arm, "fold_s": round(s, 3), "site_wall_ms": total,
               "plddt": m.get("plddt"), "cif_sha256_16": sha, "sites": sites}
        out["arms"].append(rec)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"[ab] {i} {arm:4s} fold {s:8.3f}s site_wall {total:9.2f} ms "
              f"plddt {m.get('plddt')} cif {sha}", flush=True)

    a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
