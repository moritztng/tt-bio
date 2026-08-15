#!/usr/bin/env python3
"""Fold-level A/B for esmfold2-to-4x-per-dollar.

`--fast` is a LOAD-time property (weights load as bfloat8_b), so one process is one arm.
The driver interleaves processes instead of arms, and the per-process spread is the A/A
floor; a cross-process pair of the SAME arm is the honest error bar for the delta.

Every timed fold is the shipped `predict_one` boundary, the same one the page publishes.
The output CIF of every fold is kept so the accuracy cost of a non-bit-exact arm can be
measured afterwards rather than asserted.
"""
import argparse, hashlib, json, os, shutil, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--grid", default=None, help="e.g. 13x10; overrides the pinned 11x10 main grid")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--cifdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.esmc as EC
    import tt_bio.reblock_permute as RP
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__

    # L-G: COMPUTE_GRID_MAIN / CORE_GRID_MAIN are module constants pinned to 11x10, and
    # _pair_proj_program_config reads COMPUTE_GRID_MAIN rather than the device. On a 13x10
    # card that leaves 20 of 130 cores unused by construction. The constant is imported
    # BY VALUE into every consumer module, so each one has to be rebound before the model
    # is built, and the program-config cache has to be dropped.
    if a.grid:
        gx, gy = (int(v) for v in a.grid.lower().split("x"))
        import importlib
        T.COMPUTE_GRID_MAIN = (gx, gy)
        T.CORE_GRID_MAIN = __import__("ttnn").CoreGrid(y=gy, x=gx)
        T._pair_proj_program_config.cache_clear()
        for mod in ("esmfold2", "esmc", "protenix", "esmfold2_runtime"):
            try:
                m = importlib.import_module("tt_bio." + mod)
            except Exception:
                continue
            if hasattr(m, "CORE_GRID_MAIN"):
                m.CORE_GRID_MAIN = T.CORE_GRID_MAIN
            if hasattr(m, "COMPUTE_GRID_MAIN"):
                m.COMPUTE_GRID_MAIN = T.COMPUTE_GRID_MAIN
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "size": a.size,
           "fast": a.fast, "tag": a.tag, "grid_override": a.grid,
           "git_head": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "rounds": a.rounds, "runs": []}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, state = B.build_fold(a.model, ROOT / (".msa_ab512_%d" % a.size), tgt, a3m,
                                         fast=a.fast)
    res["load_s"] = meta.get("load_s")
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    struct_dir = Path(meta["struct_dir"])
    cifdir = a.cifdir
    if cifdir:
        cifdir.mkdir(parents=True, exist_ok=True)

    def run(label):
        RP.STATS_GATED[0] = RP.STATS_GATED[1] = 0
        EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
        fold_s, m = one_fold()
        cif = sha_dir(struct_dir)
        if cifdir:
            for p in sorted(struct_dir.glob("*.cif")):
                shutil.copyfile(p, cifdir / ("%s_%s_%s" % (a.tag, label, p.name)))
        row = {"label": label, "fast": a.fast, "fold_s": round(fold_s, 3),
               "plddt": m.get("plddt"), "cif": cif,
               "e6_served": RP.STATS_GATED[0], "e6_declined": RP.STATS_GATED[1],
               "l1_fc1_stats": list(EC.L1_FC1_STATS),
               "l1_out_refused": len(T._L1_OUT_REFUSED),
               "loadavg": open("/proc/loadavg").read().split()[0]}
        res["runs"].append(row)
        a.out.write_text(json.dumps(res, indent=1))
        print("  %-10s %8.3fs plddt=%s e6=%d/%d l1fc1=%d/%d refused=%d cif=%s load=%s"
              % (label, fold_s, m.get("plddt"), RP.STATS_GATED[0], RP.STATS_GATED[1],
                 EC.L1_FC1_STATS[0], EC.L1_FC1_STATS[1], row["l1_out_refused"],
                 list(cif.values())[0] if cif else "-", row["loadavg"]), flush=True)

    print("=== %s %d aa fast=%s rec=%s steps=%s head=%s grid=%s load=%.1fs ==="
          % (a.model, a.size, a.fast, B.RECYCLING_STEPS, B.SAMPLING_STEPS,
             res["git_head"], res["grid"], res["load_s"] or -1), flush=True)
    run("warm")
    v = []
    for r in range(a.rounds):
        run("r%d" % r)
        v.append(res["runs"][-1]["fold_s"])
    res["summary"] = {"n": len(v), "median": round(st.median(v), 3), "min": min(v),
                      "max": max(v), "spread": round(max(v) - min(v), 3)}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
