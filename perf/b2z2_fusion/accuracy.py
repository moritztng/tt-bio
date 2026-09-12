#!/usr/bin/env python3
"""The accuracy gate for the two Transition levers: cdk2x2_298 control AND the 512 aa fold.

Neither lever is bit-exact, so neither is a mechanical change and neither can be quoted off a chunk
PCC. The campaign's own history is the reason: three wave-2 levers passed everything upstream and
died at 512 aa, and one of them (`transition`) passed 298 aa at 0.266 A while 512 aa moved 13.21 A.
So both sizes, every time, and >0.60 A all-atom is a kill.

Writes one directory per (size, arm, run) holding that fold's CIF, in the layout
`perf/other512/cif_rmsd.py` reads: `<size>_<arm>_<run>`. `base` is folded TWICE so the base/base
pair is an A/A floor that must come back 0.000000 -- if it does not, the instrument is broken and
no other number in the run means anything.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--arms", default="base,base,swiglu,usilu")
    ap.add_argument("--grid", default="11x8")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.transition_swiglu as TS
    import tt_baseline as B
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    import fold_ab_multi as FAM

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        FAM.patch_boltz2_cfg()

    TS.GRID = None if a.grid == "main" else tuple(int(v) for v in a.grid.split("x"))
    served = [0, 0]
    orig_fs = TS.fused_swiglu

    def counted(*args, **kw):
        out = orig_fs(*args, **kw)
        served[0 if out is not None else 1] += 1
        return out

    TS.fused_swiglu = counted

    def set_arm(name):
        T._UNFUSED_SILU = (name == "usilu")
        TS.set_enabled(name.startswith("swiglu"))
        # swiglu_s1 is the same kernel with sigmoid at bf16 precision; see fold_ab.py.
        TS.SILU_MODE = TS.SILU_HOIST = 1 if name == "swiglu_s1" else 0
        TS.MUL_BATCH = 4
        TS.REJECTS.clear()
        served[0] = served[1] = 0

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename, "model": a.model,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "fused_grid": list(TS.GRID) if TS.GRID else "main",
           "block_config": {str(k): list(v) for k, v in TS.BLOCK_KEYS.items()},
           "mul_mode": TS.MUL_MODE, "runs": []}
    a.cifdir.mkdir(parents=True, exist_ok=True)

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / ("cdk2x2_%d.yaml" % size)
        a3m = a.fixdir / ("cdk2x2_%d.a3m" % size)
        set_arm("base")
        one_fold, meta, state = B.build_fold(a.model, ROOT / (".msa_b2z2fr_%d" % size), tgt, a3m)
        struct_dir = Path(meta["struct_dir"])
        print("=== %s %d aa: cold ===" % (a.model, size), flush=True)
        cold_s, cold_m = one_fold()
        print("  cold %.2fs plddt=%s" % (cold_s, cold_m.get("plddt")), flush=True)

        run_ix = {}
        for arm in a.arms.split(","):
            set_arm(arm)
            fold_s, m = one_fold()
            ix = run_ix[arm] = run_ix.get(arm, -1) + 1
            dest = a.cifdir / ("%d_%s_%d" % (size, arm, ix))
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for p in sorted(struct_dir.glob("*")):
                if p.is_file():
                    shutil.copy2(p, dest / p.name)
            rec = {"size": size, "arm": arm, "run": ix, "fold_s": round(fold_s, 3),
                   "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
                   "cif_sha256": sha_dir(struct_dir), "cif_dir": str(dest),
                   "swiglu_served": served[0], "swiglu_declined": served[1],
                   "unfused_silu": T._UNFUSED_SILU,
                   "silu_mode": TS.SILU_MODE, "silu_hoist": TS.SILU_HOIST}
            res["runs"].append(rec)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
            print("  %d %-7s run%d  fold %7.3fs  plddt %s  sha %s  swiglu served %d declined %d"
                  % (size, arm, ix, fold_s, m.get("plddt"),
                     list(rec["cif_sha256"].values()), served[0], served[1]), flush=True)

    a.out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
