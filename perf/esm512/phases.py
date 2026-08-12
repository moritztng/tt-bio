#!/usr/bin/env python3
"""Split the 5.16 s the 512 aa decomposition could not attribute.

`tt_baseline.build_fold(instrument=True)` refuses anything but protenix-v2, and its three phases
are protenix's. ESMFold2's `_predict_esmfold2_one` has its own three: `_read_protein_chains` +
`resolve_msa` (host featurization), `fold_complex` (the model), `_write_structure` (the CIF
write). Timing those says whether the unattributed 11.3 % is host work that is out of scope or
device work that outranks the whole diffusion head.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

PH = {}


def wrap(mod, name, key):
    f = getattr(mod, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            PH[key] = PH.get(key, 0.0) + time.perf_counter() - t
    setattr(mod, name, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "esmfold2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "esmfold2")

    import tt_bio.main as M
    import tt_bio.esmfold2_runtime as ER
    wrap(M, "_read_protein_chains", "host:read_chains")
    wrap(ER, "resolve_msa", "host:resolve_msa")
    wrap(ER, "fold_complex", "fold_complex")
    wrap(M, "_write_structure", "host:write_cif")

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("esmfold2", ROOT / f".msa_om512_{a.size}", tgt, a3m)
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "size": a.size, "runs": []}
    for i in range(a.folds):
        PH.clear()
        fold_s, m = one_fold()
        row = {"i": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "phases": {k: round(v, 3) for k, v in sorted(PH.items(), key=lambda kv: -kv[1])}}
        row["outside_fold_complex_s"] = round(fold_s - PH.get("fold_complex", 0.0), 3)
        R["runs"].append(row)
        a.out.write_text(json.dumps(R, indent=1))
        print(f"  fold{i} {fold_s:.3f}s  " + "  ".join(
            f"{k}={v:.3f}" for k, v in row["phases"].items())
            + f"  outside_fold_complex={row['outside_fold_complex_s']:.3f}", flush=True)
    a.out.write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
