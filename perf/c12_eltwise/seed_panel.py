#!/usr/bin/env python3
"""Does the conditioning fold cost real quality, or is it re-sampling? Paired by seed, n=8.

The three-seed screen left this unresolved: the lever moved cdk2x2_512 by 12.2-12.6 A against a
seed band of 5.35-9.90 A, and dropped mean plDDT to 84.47-84.56 against a three-sample band of
85.32-86.51. Three samples cannot separate a systematic cost from the tail of a wide
distribution.

The design that can: fold BOTH arms at the SAME 8 seeds, in one session, alternating arm by seed
so any drift is shared. Then

  * per-seed delta   mean plDDT(on, seed k) - mean plDDT(off, seed k), which is paired and so
    carries none of the between-seed variance. If the fold costs quality, every delta is negative.
  * off-arm spread   the between-seed plDDT scatter, which is the bar those deltas are read
    against.
  * RMSD is recorded but is NOT the deciding number. The trajectory saturates on any one-ULP
    perturbation, so an RMSD between arms measures which basin was picked, not whether the answer
    got worse. plDDT carries no frame.
"""
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

PL, X, Y, Z = 17, 10, 11, 12


def read_cif(p):
    import torch
    pl, xyz = [], []
    for line in Path(p).read_text().splitlines():
        f = line.split()
        if len(f) > 18 and f[0] in ("ATOM", "HETATM"):
            pl.append(float(f[PL]))
            xyz.append([float(f[X]), float(f[Y]), float(f[Z])])
    return torch.tensor(pl, dtype=torch.float64), torch.tensor(xyz, dtype=torch.float64)


def kabsch_rmsd(P, Q):
    import torch
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = torch.linalg.svd(Pc.T @ Qc)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype))
    return float(torch.sqrt((((Vt.T @ D @ U.T @ Pc.T).T - Qc) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", default="512")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "perf/c12_eltwise/runs/seed_panel.json")
    a = ap.parse_args()

    import tt_baseline as B
    from tt_bio import eltwise_fusion as EF
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
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

    EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = False
    one_fold()   # cold, off arm
    EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = True
    one_fold()   # cold, on arm

    rows = []
    for sd in range(a.seeds):
        job_cfg["seed"] = sd
        rec = {"seed": sd}
        got = {}
        for arm, flag in (("off", False), ("on", True)):
            EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = flag
            t, _m = one_fold()
            cif = sorted(struct_dir.glob("*.cif"))[0]
            pl, xyz = read_cif(cif)
            got[arm] = (pl, xyz)
            rec[arm + "_plddt"] = float(pl.mean())
            rec[arm + "_s"] = t
        rec["delta_plddt"] = rec["on_plddt"] - rec["off_plddt"]
        rec["rmsd_A"] = kabsch_rmsd(got["off"][1], got["on"][1])
        rows.append(rec)
        print("seed %d  off %.3f  on %.3f  delta %+.3f  rmsd %6.3f A"
              % (sd, rec["off_plddt"], rec["on_plddt"], rec["delta_plddt"], rec["rmsd_A"]),
              flush=True)
    EF.FUSE_COND_MULADD = EF.FUSE_ATTN_GATE_ADD = False

    d = [r["delta_plddt"] for r in rows]
    off = [r["off_plddt"] for r in rows]
    neg = sum(1 for x in d if x < 0)
    res = {"model": a.model, "size": a.size, "seeds": a.seeds, "rows": rows,
           "delta_plddt": {"mean": st.mean(d), "median": st.median(d), "min": min(d),
                           "max": max(d),
                           "stdev": st.stdev(d) if len(d) > 1 else 0.0,
                           "n_negative": neg, "n": len(d)},
           "off_plddt_between_seed": {"mean": st.mean(off), "min": min(off), "max": max(off),
                                      "stdev": st.stdev(off) if len(off) > 1 else 0.0},
           "rmsd_A": {"mean": st.mean([r["rmsd_A"] for r in rows]),
                      "min": min(r["rmsd_A"] for r in rows),
                      "max": max(r["rmsd_A"] for r in rows)}}
    a.out.write_text(json.dumps(res, indent=1))
    print("\npaired delta plDDT: mean %+.4f  median %+.4f  stdev %.4f  range %+.3f..%+.3f  "
          "negative %d/%d" % (res["delta_plddt"]["mean"], res["delta_plddt"]["median"],
                              res["delta_plddt"]["stdev"], min(d), max(d), neg, len(d)))
    print("off-arm between-seed plDDT: mean %.4f stdev %.4f range %.3f..%.3f"
          % (res["off_plddt_between_seed"]["mean"], res["off_plddt_between_seed"]["stdev"],
             min(off), max(off)))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
