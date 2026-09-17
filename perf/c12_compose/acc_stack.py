#!/usr/bin/env python3
"""The STACK's own accuracy, against a seed floor measured on the same fixture in the same session.

Approve a stack as a stack or not at all: the two levers' individual readings are not addable, so
this scores `both` (hoist=1, silu=1) against `base` directly and never sums 0.4504 A onto anything.

What is measured here, and why each piece is needed:

    lever      base vs both at the same seed        the deviation the stack costs
    floor      base vs base at two different seeds  the scatter the sampler produces by itself
    A/A        base vs base at the SAME seed        the control -- if this is not 0.0000 A the
                                                    paired read above is not the lever

The fixture matters more than the metric here. cdk2x2_298 is what the 0.35/0.60 A bar is written
against. cdk2x2_512 is chimeric: its unconstrained hinge saturates whole-molecule RMSD for any
non-bit-exact change (`c12-fused-eltwise-at-pin` read 5.35/9.90/7.95 A on it against an A/A of
0.0000 A, and `c12-unfused-silu-bh` read 12.69 A whole-molecule against 1.44 A hinge-free for a
change worth 0.45 A on the 298 aa fixture). So the 512 read is reported and plDDT, which carries no
frame and so cannot be confounded by which basin the trajectory chose, decides there.

    acc_stack.py --out <json> --cifs <dir> --size 298 --seeds 0,1 --arms base,both
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(REPO / "perf" / "other512"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_flag_levers as AB  # noqa: E402
import clk  # noqa: E402

AA_PASS, AA_HOLD = 0.35, 0.60          # the 4649 bar, all-atom, cdk2x2_298
ARMS = {"base": (False, False), "silu": (False, True),
        "hoist": (True, False), "both": (True, True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, required=True)
    ap.add_argument("--size", type=int, default=298)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--arms", default="base,both")
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--aa-repeat", action="store_true", default=True,
                    help="fold base twice at the first seed: the 0.0000 A control")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        assert a in ARMS, f"unknown arm {a!r}"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from cif_rmsd import kabsch_rmsd, read_atoms
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    for env, attr in (("TT_BIO_DIT_COND_HOIST", "_B2_DIT_COND_HOIST"),
                      ("TT_BIO_UNFUSED_SILU", "_UNFUSED_SILU")):
        assert env not in os.environ, f"{env} may not be pinned; the arm is set in-process"
        assert getattr(T, attr) is False, f"{attr} must ship off for base to be the default"

    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    clk.force(args.mhz, nodes)
    args.cifs.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="c12-compose-acc-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fx = AB.FIX / f"cdk2x2_{args.size}.yaml"
    AB._seed_msa(fx, (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)

    plan = [(s, a, "") for s in seeds for a in arms]
    if args.aa_repeat:
        plan.append((seeds[0], "base", "_aa"))

    folds = {}
    for seed, arm, suffix in plan:
        AB.SEED = seed
        cfg = AB.build_cfg(msa_dir, struct_dir)
        assert cfg["seed"] == seed, "build_cfg did not pick up the seed"
        _ensure_local_artifacts(cfg)
        state = _WorkerState("tenstorrent")
        state.load_model(cfg)
        state.bind_run(f"c12-compose-acc-s{seed}-{arm}{suffix}", cfg)
        T._B2_DIT_COND_HOIST, T._UNFUSED_SILU = ARMS[arm]
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        smp = clk.Sampler(nodes[0])
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(fx, cfg)
        ttnn.synchronize_device(dev)
        wall = round(time.perf_counter() - t, 3)
        aiclk = smp.stop()
        cif = sorted(struct_dir.rglob("*.cif"))[0]
        tag = f"s{seed}_{arm}{suffix}"
        dst = args.cifs / f"{args.size}_{tag}.cif"
        shutil.copy2(cif, dst)
        folds[tag] = {
            "seed": seed, "arm": arm, "hoist": ARMS[arm][0], "silu": ARMS[arm][1],
            "wall_s": wall, "aiclk": aiclk, "cif": dst.name,
            "loadavg": [round(x, 2) for x in os.getloadavg()],
            "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6),
            "digest": hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}
        print(f"  {tag:12s} {wall:8.3f}s  clk {aiclk['min']}-{aiclk['max']}  "
              f"plddt={folds[tag]['plddt']}  sha={folds[tag]['digest']}", flush=True)
    clk.release()

    xyz, keys = {}, None
    for tag, f in folds.items():
        k, x = read_atoms(args.cifs / f["cif"])
        if keys is None:
            keys = k
        assert k == keys, f"atom identity differs in {tag}"
        xyz[tag] = x
    ca = [i for i, k in enumerate(keys) if k[2] == "CA"]

    def rms(a, b):
        return round(kabsch_rmsd(xyz[a], xyz[b]), 5)

    def rms_ca(a, b):
        return round(kabsch_rmsd(xyz[a][ca], xyz[b][ca]), 5)

    lever = [{"seed": s, "pair": f"s{s}_base vs s{s}_both",
              "all_atom_A": rms(f"s{s}_base", f"s{s}_both"),
              "ca_A": rms_ca(f"s{s}_base", f"s{s}_both"),
              "plddt_base": folds[f"s{s}_base"]["plddt"],
              "plddt_both": folds[f"s{s}_both"]["plddt"],
              "plddt_delta": round(folds[f"s{s}_both"]["plddt"]
                                   - folds[f"s{s}_base"]["plddt"], 6)}
             for s in seeds if f"s{s}_both" in folds]
    floor = [{"pair": f"base s{a} vs base s{b}",
              "all_atom_A": rms(f"s{a}_base", f"s{b}_base"),
              "ca_A": rms_ca(f"s{a}_base", f"s{b}_base"),
              "plddt_delta": round(folds[f"s{b}_base"]["plddt"]
                                   - folds[f"s{a}_base"]["plddt"], 6)}
             for a, b in itertools.combinations(seeds, 2)]
    aa = None
    if f"s{seeds[0]}_base_aa" in folds:
        a, b = f"s{seeds[0]}_base", f"s{seeds[0]}_base_aa"
        aa = {"pair": "base vs base, same seed", "all_atom_A": rms(a, b), "ca_A": rms_ca(a, b),
              "digests_identical": folds[a]["digest"] == folds[b]["digest"],
              "plddt_delta": round(folds[b]["plddt"] - folds[a]["plddt"], 6)}

    worst = max((x["all_atom_A"] for x in lever), default=None)
    res = {
        "doc": __doc__, "size": args.size, "seeds": seeds, "arms": arms,
        "host": socket.gethostname(), "card_node": nodes[0], "clock_forced_mhz": args.mhz,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bar": {"pass": AA_PASS, "hold": AA_HOLD,
                "basis": "the 4649 bar, all-atom, cdk2x2_298; NOT applicable whole-molecule on "
                         "the chimeric cdk2x2_512"},
        "folds": folds,
        "all_pairs": {f"{a} vs {b}": {"all_atom_A": rms(a, b), "ca_A": rms_ca(a, b)}
                      for a, b in itertools.combinations(sorted(xyz), 2)},
        "stack_paired_per_seed": lever, "seed_floor": floor, "aa_control": aa,
        "worst_stack_all_atom_A": worst,
        "verdict": None if worst is None else
                   ("PASS" if worst <= AA_PASS else "HOLD" if worst <= AA_HOLD else "REJECT"),
        "stack_over_seed_floor": (round(worst / floor[0]["all_atom_A"], 3)
                                  if worst is not None and floor and floor[0]["all_atom_A"]
                                  else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("\n  STACK paired effect, per seed (all-atom / CA / dplDDT):")
    for x in lever:
        print(f"    s{x['seed']}  {x['all_atom_A']:8.4f} / {x['ca_A']:8.4f} A / "
              f"{x['plddt_delta']:+.6f}")
    print("  seed floor, base against base at another seed:")
    for x in floor:
        print(f"    {x['pair']:22s} {x['all_atom_A']:8.4f} / {x['ca_A']:8.4f} A")
    if aa:
        print(f"  A/A control: {aa['all_atom_A']:.4f} A all-atom, digests identical="
              f"{aa['digests_identical']}")
    print(f"\n  worst stack {worst} A against the {AA_PASS}/{AA_HOLD} bar -> {res['verdict']}, "
          f"{res['stack_over_seed_floor']}x the seed floor")
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
