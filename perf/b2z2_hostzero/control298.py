#!/usr/bin/env python3
"""cdk2x2 structural control, and a paired A/B, for one or more TT_BIO_* levers (``--flag``).

The device conditioning is not bit-exact -- bf16 device math against fp32 torch, and the fused
bias stack on top -- so a CIF hash cannot score it. `cdk2x2_512` cannot score it either: it is
CDK2 fused to a truncated copy of itself and its unconstrained hinge saturates RMSD for any
change whatever the cause (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`).
`cdk2x2_298` is the same family with one real domain and no hinge, and the thresholds in
`state/answered/4649-decision.md` are stated against it: <=0.35 A pass, 0.35-0.60 A hold,
>0.60 A reject.

Default mode (``--reps 0``): three folds in one process on one device open -- base, base again,
and the arm. The repeated base gives the A/A floor in the same session, which is what makes the
arm's number readable: a nonzero A/A would mean the comparison is measuring the box, not the
change. Score with `perf/b2x-flag-levers/score298.py <cifdir> --ref base_0`, or, for a lever
whose output is a score rather than a coordinate, `perf/b2z2_confhead/score_conf.py`.

``--reps N`` instead runs a paired fold A/B: ``base arm arm base`` per rep, so the order reverses
inside the rep and a monotonic drift in the box cancels rather than accumulating in one arm. That
is 4N folds and 2N paired ratios, plus the same-arm adjacent pairs as this session's own A/A
floor. CIFs are kept for the first fold of each arm; every fold's CIF sha256 and pLDDT are
recorded, so an arm that is supposed to be structurally inert can be checked at every fold and
not just the one that was copied.

``--flag`` takes a comma-separated set, set and unset together, because a lever built on top of
another is approved as a stack or not at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
FIX = REPO / "perf" / "size512" / "fixtures"


def cif_sha(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*.cif")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=298)
    ap.add_argument("--flag", default="TT_BIO_DEVICE_CONDITIONING",
                    help="comma-separated; the whole set is one arm")
    ap.add_argument("--arm", default=None, help="tag for the on fold; defaults from --flag")
    ap.add_argument("--reps", type=int, default=0,
                    help="0 = the three-fold control; N > 0 = 4N paired folds, base arm arm base")
    a = ap.parse_args()
    flags = [f.strip() for f in a.flag.split(",") if f.strip()]
    arm = a.arm or flags[0].removeprefix("TT_BIO_").lower()

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    import tt_bio as _TB
    from tt_bio.main import _resolve_recycling_steps

    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    for f in flags:
        assert f not in os.environ, f"{f} is set in the environment; this script owns it"
    tgt, a3m = FIX / f"cdk2x2_{a.size}.yaml", FIX / f"cdk2x2_{a.size}.a3m"
    msa_dir = Path(__file__).resolve().parent / f".msa_{a.size}"
    one_fold, meta, _state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    out = {"doc": __doc__,
           "env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename, "size": a.size, "msa_dir": str(msa_dir),
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "tt_bio_file": _TB.__file__, "flags": flags, "reps": a.reps,
                   "loadavg_start": os.getloadavg(),
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
           "folds": []}
    a.cifdir.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def fold(tag: str, on: bool, keep_cif: bool):
        for f in flags:
            os.environ[f] = "1" if on else "0"
        t0 = time.perf_counter()
        _t, m = one_fold()
        wall = time.perf_counter() - t0
        sha = cif_sha(struct_dir)
        rec = {"tag": tag, "arm": arm if on else "base", "on": on,
               "wall_s": round(wall, 4), "plddt": m.get("plddt"), "cif_sha": sha,
               "loadavg": os.getloadavg()[0]}
        if keep_cif:
            dest = a.cifdir / f"{a.size}_{tag}"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(struct_dir, dest)
            rec["dir"] = dest.name
        out["folds"].append(rec)
        print("  %-12s on=%d  %.3f s  plddt %s  sha %s"
              % (tag, int(on), wall, m.get("plddt"), sha), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    # Two cold folds, one per arm, both discarded: each arm compiles its own programs and a
    # first-of-arm fold carries that compile. Warming only the base would hand the whole of it
    # to the arm.
    print("=== cold folds (discarded) ===", flush=True)
    for on in (False, True):
        for f in flags:
            os.environ[f] = "1" if on else "0"
        one_fold()

    if a.reps <= 0:
        for tag, on in (("base_0", False), ("base_1", False), (arm + "_0", True)):
            fold(tag, on, keep_cif=True)
    else:
        print("=== paired A/B, %d reps of base %s %s base ===" % (a.reps, arm, arm), flush=True)
        for r in range(a.reps):
            for i, on in enumerate((False, True, True, False)):
                tag = "%s_r%d_%d" % (arm if on else "base", r, i)
                fold(tag, on, keep_cif=(r == 0 and i in (0, 1)))
        base = [f["wall_s"] for f in out["folds"] if not f["on"]]
        armw = [f["wall_s"] for f in out["folds"] if f["on"]]
        # Paired inside the rep: fold 0 against fold 1 and fold 3 against fold 2, so each ratio
        # is two adjacent folds and the rep's own drift cancels.
        w = [f["wall_s"] for f in out["folds"]]
        pairs = []
        for r in range(a.reps):
            b0, a0, a1, b1 = w[4 * r:4 * r + 4]
            pairs += [b0 / a0, b1 / a1]
        # A/A floor from this session's same-arm adjacent pairs: the two arm folds inside a rep,
        # and the base fold that closes one rep against the base fold that opens the next.
        aa = [w[4 * r + 1] / w[4 * r + 2] for r in range(a.reps)]
        aa += [w[4 * r + 3] / w[4 * r + 4] for r in range(a.reps - 1)]
        out["analysis"] = {
            "n_pairs": len(pairs),
            "ratio_paired_median": round(st.median(pairs), 5),
            "ratio_global_median": round(st.median(base) / st.median(armw), 5),
            "pairs_positive": sum(p > 1.0 for p in pairs),
            "pairs": [round(p, 5) for p in pairs],
            "aa_floor_median": round(st.median(aa), 5),
            "aa_floor_max": round(max(max(x, 1 / x) for x in aa), 5),
            "base_median_s": round(st.median(base), 4),
            "arm_median_s": round(st.median(armw), 4),
            "delta_s": round(st.median(base) - st.median(armw), 4),
            "cif_shas": sorted({f["cif_sha"] for f in out["folds"]}),
            "plddt_values": sorted({f["plddt"] for f in out["folds"]}),
            "loadavg_end": os.getloadavg(),
        }
        a.out.write_text(json.dumps(out, indent=1))
        print(json.dumps(out["analysis"], indent=1), flush=True)
    for f in flags:
        os.environ.pop(f, None)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
