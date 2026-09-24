"""One model, one process: the same fold with `ops.linear`'s rows view off and on.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python perf/bcx_oplin/fold_ab.py \
        --model boltz2 --out perf/bcx_oplin/folds/boltz2_512

Folds through `scripts/gpu_vs_tt/tt_baseline.py`'s `build_fold` (the path the campaign's
absolutes come from: the model's default recycles, 200 sampling steps, 1 sample, the committed
alignment seeded into the MSA cache). The model loads once. Then:

  cold    arm on, seed 0, under a census of every `ops.linear` call that reaches `_via2d`
  warm    `--rounds` x (off, on) at seed 0, order alternating per round, wall + AICLK each
  floor   off and on at seed 1, the seed-to-seed floor measured in the same process

Structure deviation is CA RMSD after Kabsch superposition, every structure against the off arm
at seed 0. Writes `result.json` and each CIF beside it.
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import numpy as np

from common import Census, Clock, arm


def ca(path):
    import gemmi
    st = gemmi.read_structure(str(path))
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch for a in r
                     if a.name == "CA"])


def rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import tt_baseline as B
    one_fold, meta, _state = B.build_fold(a.model, Path(tempfile.mkdtemp(prefix="oplin-msa-")),
                                          a.target, a.a3m)
    job = meta["job_cfg"]
    folds = []

    def fold(on, seed, tag, census=None):
        job["seed"] = seed
        with arm(on):
            if census is not None:
                with census, Clock() as clk:
                    t, m = one_fold()
            else:
                with Clock() as clk:
                    t, m = one_fold()
        cif = sorted(Path(meta["struct_dir"]).glob("*.cif"))[0]
        dst = a.out / f"{tag}.cif"
        shutil.copy(cif, dst)
        row = dict(tag=tag, on=on, seed=seed, wall_s=round(t, 3), plddt=m.get("plddt"),
                   aiclk=clk.stats(), sha16=hashlib.sha256(dst.read_bytes()).hexdigest()[:16])
        folds.append(row)
        print(json.dumps(row), flush=True)

    census = Census()
    fold(True, 0, "cold_on", census)
    for r in range(a.rounds):
        for on in ((False, True) if r % 2 == 0 else (True, False)):
            fold(on, 0, f"s0_{'on' if on else 'off'}_{r}")
    fold(False, 1, "s1_off")
    fold(True, 1, "s1_on")

    ref = ca(a.out / "s0_off_0.cif")
    for row in folds:
        row["ca_rmsd_vs_off_s0"] = round(rmsd(ca(a.out / f"{row['tag']}.cif"), ref), 4)
    warm = lambda on: [f["wall_s"] for f in folds if f["tag"].startswith("s0_") and f["on"] == on]
    s1off = ca(a.out / "s1_off.cif")
    res = dict(
        model=a.model, target=str(a.target.relative_to(ROOT)), n_ca=len(ref),
        recycling_steps=meta["recycling_steps"], sampling_steps=B.SAMPLING_STEPS,
        deviation_A=next(f["ca_rmsd_vs_off_s0"] for f in folds if f["tag"] == "s0_on_0"),
        seed_floor_A=next(f["ca_rmsd_vs_off_s0"] for f in folds if f["tag"] == "s1_off"),
        seed1_on_vs_off_A=round(rmsd(ca(a.out / "s1_on.cif"), s1off), 4),
        off_repeat_identical=len({f["sha16"] for f in folds if f["tag"].startswith("s0_off")}) == 1,
        on_repeat_identical=len({f["sha16"] for f in folds
                                 if f["tag"].startswith("s0_on") or f["tag"] == "cold_on"}) == 1,
        warm_off_s=warm(False), warm_on_s=warm(True),
        speedup=round(statistics.median(warm(False)) / statistics.median(warm(True)), 4),
        census=census.summary(), folds=folds,
        card=dict(hardware=meta.get("hardware"), card_type=meta.get("card_type"),
                  sysfs_subsystem=meta.get("sysfs_subsystem"), grid=meta.get("grid")),
        host=os.uname().nodename, loadavg=os.getloadavg(),
    )
    (a.out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("folds", "census")}))


if __name__ == "__main__":
    main()
