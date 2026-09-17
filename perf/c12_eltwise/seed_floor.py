#!/usr/bin/env python3
"""The seed-scatter floor for THIS fixture, on THIS metric, with the shipped code on both sides.

The fused-conditioning arm moves cdk2x2_512 by 12.56 A all-atom. That number is only
interpretable against what re-running the SAME code with a different seed does, measured the
same way on the same fixture. A floor quoted from another protocol is not a control.

Everything here runs with TT_BIO_FUSE_COND_MULADD off, so the only thing that differs between
folds is the seed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", default="512")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=Path, default=ROOT / "perf/c12_eltwise/runs/seed_floor")
    a = ap.parse_args()

    import tt_baseline as B
    from tt_bio import eltwise_fusion as EF
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    EF.FUSE_COND_MULADD = False
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / (".msa_s512_%s_%s" % (a.model, a.size)),
        fixdir / ("cdk2x2_%s.yaml" % a.size), fixdir / ("cdk2x2_%s.a3m" % a.size))
    struct_dir = Path(meta["struct_dir"])
    job_cfg = meta["job_cfg"]
    print("job_cfg seed key present: %s (value %r)"
          % ("seed" in job_cfg, job_cfg.get("seed")), flush=True)

    one_fold()   # cold
    a.out.mkdir(parents=True, exist_ok=True)
    for sd in [int(x) for x in a.seeds.split(",")]:
        job_cfg["seed"] = sd
        t, _m = one_fold()
        cifs = sorted(struct_dir.glob("*.cif"))
        d = a.out / ("seed%d" % sd)
        d.mkdir(exist_ok=True)
        for c in cifs:
            (d / c.name).write_bytes(c.read_bytes())
        print("seed %d  %8.3f s  -> %s" % (sd, t, d), flush=True)


if __name__ == "__main__":
    main()
