"""One model, one process: the same fold with `ops.linear`'s rows view off and on.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python perf/bcx_oplin/fold_ab.py \
        --model boltz2 --out perf/bcx_oplin/folds/boltz2_512

Folds through `scripts/gpu_vs_tt/tt_baseline.py`'s `build_fold` (the path the campaign's
absolutes come from: the model's default recycles, 200 sampling steps, 1 sample, the committed
alignment seeded into the MSA cache). The model loads once. Then:

  cold    arm on, seed 0, under a census of every `ops.linear` call that reaches `_via2d`
  seeds   off and on at each of `--seeds` seeds, order alternating per seed
  timing  `--rounds - 1` more (on, off) pairs at seed 0; every warm fold is a timing sample

deviation: CA RMSD (Kabsch) of on against off at the same seed. seed floor: each arm at seed k
against the same arm at seed 0, in the same process. `--truth`: every fold against an
experimental structure, CA matched by residue number. Writes `result.json` and each CIF.
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


def ca(path, by_resnum=False):
    """CA coordinates of the first model, keyed by order, or by (chain, residue number)."""
    import gemmi
    st = gemmi.read_structure(str(path))
    atoms = [(ch, r, a) for ch in st[0] for r in ch for a in r
             if a.name == "CA" and a.altloc in ("\0", "A")]
    return {((ch.name, r.seqid.num) if by_resnum else k): (a.pos.x, a.pos.y, a.pos.z)
            for k, (ch, r, a) in enumerate(atoms)}


def rmsd(p, q):
    """CA RMSD after Kabsch over the keys both structures carry."""
    keys = sorted(set(p) & set(q))
    p = np.array([p[k] for k in keys]); q = np.array([q[k] for k in keys])
    p, q = p - p.mean(0), q - q.mean(0)
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def boltz2_cfg(model, B):
    """`conf_kwargs` as `tt-bio predict --model boltz2` builds it with every flag at its default."""
    if model != "boltz2":
        return None
    from tt_bio.main import _resolve_recycling_steps
    return dict(conf_kwargs=dict(
        predict_args={"recycling_steps": _resolve_recycling_steps(None, "boltz2"),
                      "sampling_steps": B.SAMPLING_STEPS,
                      "diffusion_samples": B.DIFFUSION_SAMPLES, "max_parallel_samples": 5},
        diffusion_process_args={
            "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
            "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
            "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
            "alignment_reverse_diff": True, "synchronize_sigmas": True},
        pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
        msa_args={"subsample_msa": False, "num_subsampled_msa": 1024,
                  "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                  "msa_dropout": 0.15, "z_dropout": 0.25, "pairwise_head_width": 32,
                  "pairwise_num_heads": 4, "activation_checkpointing": True},
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=True, use_tenstorrent=True, trace=False, diffusion_trace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "perf/size512/fixtures/cdk2x2_512.a3m")
    ap.add_argument("--seeds", type=int, default=3, help="seeds folded in both arms")
    ap.add_argument("--rounds", type=int, default=2,
                    help="timing pairs at seed 0, counting the seed-0 pair")
    ap.add_argument("--truth", type=Path,
                    help="experimental structure; CA matched by residue number")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import tt_baseline as B
    one_fold, meta, _state = B.build_fold(a.model, Path(tempfile.mkdtemp(prefix="oplin-msa-")),
                                          a.target, a.a3m, extra_cfg=boltz2_cfg(a.model, B))
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
    seeds = list(range(a.seeds))
    for k, seed in enumerate(seeds):
        for on in ((False, True) if k % 2 == 0 else (True, False)):
            fold(on, seed, f"s{seed}_{'on' if on else 'off'}")
    for r in range(1, a.rounds):  # extra timing pairs at seed 0
        for on in ((True, False) if r % 2 else (False, True)):
            fold(on, 0, f"s0_{'on' if on else 'off'}_t{r}")

    st = {f["tag"]: ca(a.out / f"{f['tag']}.cif") for f in folds}
    dev = {s_: round(rmsd(st[f"s{s_}_on"], st[f"s{s_}_off"]), 4) for s_ in seeds}
    floor_off = {s_: round(rmsd(st[f"s{s_}_off"], st["s0_off"]), 4) for s_ in seeds[1:]}
    floor_on = {s_: round(rmsd(st[f"s{s_}_on"], st["s0_on"]), 4) for s_ in seeds[1:]}
    graded = {f"s{s_}_{arm_}" for s_ in seeds for arm_ in ("off", "on")}
    truth = None
    if a.truth:
        t = ca(a.truth, by_resnum=True)
        chain = sorted({c for c, _n in t})[0]
        t = {n: v for (c, n), v in t.items() if c == chain}
        for f in folds:
            pred = {n: v for (_c, n), v in ca(a.out / f"{f['tag']}.cif", True).items()}
            f["ca_rmsd_vs_truth"] = round(rmsd(pred, t), 4)
        by = lambda on: [f["ca_rmsd_vs_truth"] for f in folds if f["tag"] in graded and f["on"] == on]
        truth = dict(file=str(a.truth), off=by(False), on=by(True),
                     off_mean=round(float(np.mean(by(False))), 4),
                     on_mean=round(float(np.mean(by(True))), 4))
    warm = lambda on: [f["wall_s"] for f in folds if f["tag"] != "cold_on" and f["on"] == on]
    shas = lambda on, s_: {f["sha16"] for f in folds
                           if f["on"] == on and f["seed"] == s_ and f["tag"] != "cold_on"}
    res = dict(
        model=a.model, target=str(a.target), n_ca=len(st["s0_off"]),
        recycling_steps=meta["recycling_steps"], sampling_steps=B.SAMPLING_STEPS, seeds=seeds,
        deviation_A=dev, seed_floor_off_A=floor_off, seed_floor_on_A=floor_on, truth=truth,
        same_seed_repeat_identical=all(len(shas(on, 0)) == 1 for on in (False, True)),
        bit_identical_arms=all(shas(False, s_) == shas(True, s_) for s_ in seeds),
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
