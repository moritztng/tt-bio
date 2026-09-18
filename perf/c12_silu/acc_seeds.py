#!/usr/bin/env python3
"""The lever's accuracy against the seed scatter of the same fixture, both measured here.

The 512 aa paired read is on cdk2x2_512, the chimeric fixture whose unconstrained hinge saturates
RMSD for any non-bit-exact change, and the 0.35/0.60 A thresholds are written against cdk2x2_298.
So this scores on cdk2x2_298, and it scores the control that a single paired read cannot supply:
ship at seed 0 against ship at seed 1 IS the seed floor on this fixture, and the lever's deviation
only means something beside it.

Four folds in one process on one device open, so seed 0 also re-runs and its digest proves the
fold is deterministic across processes -- without that, a cross-session RMSD is not a paired read.

    acc_seeds.py --out <json> --cifs <dir> --size 298 --seeds 0,1
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

import ab_flag_levers as AB                                                   # noqa: E402
import clk                                                                   # noqa: E402

AA_PASS, AA_HOLD = 0.35, 0.60          # the 4649 bar, as score298.py states it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, required=True)
    ap.add_argument("--size", type=int, default=298)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--mhz", type=int, default=1350)
    args = ap.parse_args()
    assert "TT_BIO_UNFUSED_SILU" not in os.environ, "the arm is set in-process, not pinned by env"
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from cif_rmsd import kabsch_rmsd, read_atoms
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    assert TT._UNFUSED_SILU is False, "the shipped default must be off"

    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    clk.force(args.mhz, nodes)
    args.cifs.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="c12-silu-acc-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / f"cdk2x2_{args.size}.yaml",
                 (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)

    folds = {}
    for seed, arm in itertools.product(seeds, ("ship", "silu")):
        AB.SEED = seed
        cfg = AB.build_cfg(msa_dir, struct_dir)
        assert cfg["seed"] == seed, "build_cfg did not pick up the seed"
        _ensure_local_artifacts(cfg)
        state = _WorkerState("tenstorrent")
        state.load_model(cfg)
        state.bind_run(f"c12-silu-acc-s{seed}-{arm}", cfg)
        TT._UNFUSED_SILU = arm == "silu"
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        t = time.perf_counter()
        state.predict_one(AB.FIX / f"cdk2x2_{args.size}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = round(time.perf_counter() - t, 3)
        cif = sorted(struct_dir.rglob("*.cif"))[0]
        dst = args.cifs / f"{args.size}_s{seed}_{arm}.cif"
        shutil.copy(cif, dst)
        tag = f"s{seed}_{arm}"
        folds[tag] = {"seed": seed, "arm": arm, "wall_s": wall, "cif": dst.name,
                      "digest": hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}
        print(f"  {tag:10s} {wall:8.3f}s  sha={folds[tag]['digest']}", flush=True)
    clk.release()

    xyz, keys = {}, None
    for tag, f in folds.items():
        k, x = read_atoms(args.cifs / f["cif"])
        if keys is None:
            keys = k
        assert k == keys, f"atom identity differs in {tag}"
        xyz[tag] = x

    def rms(x, y):
        return round(kabsch_rmsd(xyz[x], xyz[y]), 5)

    ca = [i for i, k in enumerate(keys) if k[2] == "CA"]

    def rms_ca(x, y):
        return round(kabsch_rmsd(xyz[x][ca], xyz[y][ca]), 5)

    pairs = {}
    for a, b in itertools.combinations(sorted(xyz), 2):
        pairs[f"{a} vs {b}"] = {"all_atom_A": rms(a, b), "ca_A": rms_ca(a, b)}

    lever = [(f"s{s}", rms(f"s{s}_ship", f"s{s}_silu"), rms_ca(f"s{s}_ship", f"s{s}_silu"))
             for s in seeds]
    floor = [(f"ship s{a} vs s{b}", rms(f"s{a}_ship", f"s{b}_ship"),
              rms_ca(f"s{a}_ship", f"s{b}_ship"))
             for a, b in itertools.combinations(seeds, 2)]
    worst = max(v for _, v, _ in lever)
    res = {
        "doc": __doc__, "size": args.size, "seeds": seeds,
        "host": socket.gethostname(), "card_node": nodes[0], "clock_forced_mhz": args.mhz,
        "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "bar": {"pass": AA_PASS, "hold": AA_HOLD, "basis": "the 4649 bar, all-atom, cdk2x2_298"},
        "folds": folds, "all_pairs": pairs,
        "lever_paired_per_seed": [{"seed": s, "all_atom_A": v, "ca_A": c} for s, v, c in lever],
        "seed_floor": [{"pair": s, "all_atom_A": v, "ca_A": c} for s, v, c in floor],
        "worst_lever_all_atom_A": worst,
        "verdict": "PASS" if worst <= AA_PASS else ("HOLD" if worst <= AA_HOLD else "REJECT"),
        "lever_over_seed_floor": (round(worst / floor[0][1], 3) if floor and floor[0][1] else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("\n  paired lever effect, per seed (all-atom / CA):")
    for s, v, c in lever:
        print(f"    {s:4s} {v:8.4f} / {c:8.4f} A")
    print("  seed floor, ship against ship (all-atom / CA):")
    for s, v, c in floor:
        print(f"    {s:18s} {v:8.4f} / {c:8.4f} A")
    print(f"\n  worst lever {worst:.4f} A against the 0.35/0.60 bar -> {res['verdict']}, "
          f"and {res['lever_over_seed_floor']}x the seed floor")
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
