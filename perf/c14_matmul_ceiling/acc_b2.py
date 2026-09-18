#!/usr/bin/env python3
"""Boltz-2's side of the same question: does `TT_BIO_MM_SHORT_M_BW` cost accuracy at 512 aa?

The flag is not bit-exact by construction -- a wider `in0_block_w` accumulates the contraction in
a different order under `packer_l1_acc` -- so it owes a structural reading. This scores it the way
`state/b2z2-union-land.md:66` showed is the only screen with power on this class: **CA-lDDT
against the experimental structure**, 1HCL. A structural-RMSD-against-seed-scatter reading cleared
`TT_BIO_UNFUSED_SILU` on Protenix-v2 while CA-lDDT against the native separated the arms
completely, so the native reading is the decisive one and the seed floor is reported beside it,
not instead of it.

cdk2x2_512 is CDK2 followed by its own residues 1-214, so both pseudo-domains have a native
answer (`GT_SEGMENTS`, perf/fused_sdpa/of3_score_ref.py). Whole-molecule RMSD is NOT reported as a
verdict: the fixture's hinge saturates it for any non-bit-exact change.

    acc_b2.py --out <json> --cifs <dir> --size 512 --seeds 0,1,2,3 --arms base,bw
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "perf" / "b2x-flag-levers"), str(REPO / "perf" / "other512"),
                str(REPO / "perf" / "fused_sdpa"), str(REPO / "perf" / "c12_compose")]

import ab_flag_levers as AB          # noqa: E402
import clk                           # noqa: E402

ARMS = {"base": False, "bw": True}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--arms", default="base,bw")
    ap.add_argument("--mhz", type=int, default=1350)
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
    from of3_score_ref import GT_SEGMENTS, ca_map
    from basin_lddt import lddt_per_residue
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    assert "TT_BIO_MM_SHORT_M_BW" not in os.environ, "the arm is set in-process, not pinned"
    assert T._MM_SHORT_M_BW is False, "the lever must ship OFF for base to be the default"
    GT = REPO / "perf" / "fused_sdpa" / "cifs" / "1hcl.cif"

    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    clk.force(args.mhz, nodes)
    args.cifs.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="c14-accbw-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fx = AB.FIX / f"cdk2x2_{args.size}.yaml"
    AB._seed_msa(fx, (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)

    # interleaved, arm order reversed on odd seeds
    plan = [(s, a, "") for s in seeds for a in (arms if s % 2 == 0 else arms[::-1])]
    plan.append((seeds[0], "base", "_aa"))

    folds = {}
    for seed, arm, suffix in plan:
        AB.SEED = seed
        cfg = AB.build_cfg(msa_dir, struct_dir)
        assert cfg["seed"] == seed, "build_cfg did not pick up the seed"
        _ensure_local_artifacts(cfg)
        state = _WorkerState("tenstorrent")
        state.load_model(cfg)
        state.bind_run(f"c14-accbw-s{seed}-{arm}{suffix}", cfg)
        T._MM_SHORT_M_BW = ARMS[arm]
        T.MM_SHORT_M_STATS[0] = T.MM_SHORT_M_STATS[1] = 0
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
            "seed": seed, "arm": arm, "bw": ARMS[arm], "wall_s": wall, "aiclk": aiclk,
            "cif": dst.name, "loadavg": [round(x, 2) for x in os.getloadavg()],
            "served": int(T.MM_SHORT_M_STATS[0]), "declined": int(T.MM_SHORT_M_STATS[1]),
            "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6),
            "digest": hashlib.sha256(dst.read_bytes()).hexdigest()[:16]}
        print(f"  {tag:14s} {wall:8.3f}s  clk {aiclk['min']}-{aiclk['max']}  "
              f"plddt={folds[tag]['plddt']}  served={folds[tag]['served']} "
              f"declined={folds[tag]['declined']}  sha={folds[tag]['digest']}", flush=True)
    clk.release()

    # --- native CA-lDDT, per pseudo-domain, against 1HCL -------------------------------
    gt = ca_map(GT)
    seg = GT_SEGMENTS[args.size]
    native = {}
    for tag, f in folds.items():
        pm = ca_map(args.cifs / f["cif"])
        row = {}
        for name, pairs in seg.items():
            keys = [(p, g) for p, g in pairs if p in pm and g in gt]
            P = np.array([pm[p][1] for p, _ in keys])
            G = np.array([gt[g][1] for _, g in keys])
            _per, glob = lddt_per_residue(P, G)
            row[name] = {"n": len(keys), "ca_lddt_gt": float(glob),
                         "ca_rmsd_gt": round(kabsch_rmsd(P, G), 5)}
        native[tag] = row
    doms = list(seg)

    xyz, keys = {}, None
    for tag, f in folds.items():
        k, x = read_atoms(args.cifs / f["cif"])
        keys = keys or k
        assert k == keys, f"atom identity differs in {tag}"
        xyz[tag] = x
    ca = [i for i, k in enumerate(keys) if k[2] == "CA"]

    def rms(a, b, sel=None):
        A, B = (xyz[a], xyz[b]) if sel is None else (xyz[a][sel], xyz[b][sel])
        return round(kabsch_rmsd(A, B), 5)

    summary = {}
    for dom in doms:
        b = [native[f"s{s}_base"][dom]["ca_lddt_gt"] for s in seeds]
        t = [native[f"s{s}_{arms[1]}"][dom]["ca_lddt_gt"] for s in seeds]
        d = [y - x for x, y in zip(b, t)]
        md, sd = st.mean(d), (st.stdev(d) if len(d) > 1 else 0.0)
        ci = T95[len(d) - 1] * sd / len(d) ** 0.5 if len(d) > 1 else float("inf")
        summary[dom] = {
            "base_mean": st.mean(b), "base_sd": st.stdev(b) if len(b) > 1 else 0.0,
            "arm_mean": st.mean(t), "arm_sd": st.stdev(t) if len(t) > 1 else 0.0,
            "paired_delta_mean": md, "paired_ci95": ci, "resolved": abs(md) > ci,
            "rank_separated": bool(max(b) < min(t) or max(t) < min(b)),
            "base_seed_spread": max(b) - min(b),
            "aa_delta": (native[f"s{seeds[0]}_base_aa"][dom]["ca_lddt_gt"]
                         - native[f"s{seeds[0]}_base"][dom]["ca_lddt_gt"])}

    aa_tag = f"s{seeds[0]}_base_aa"
    res = {
        "doc": __doc__, "size": args.size, "seeds": seeds, "arms": arms, "gt": GT.name,
        "host": socket.gethostname(), "card_node": nodes[0], "clock_forced_mhz": args.mhz,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "folds": folds, "native": native, "native_summary": summary,
        "arm_vs_base_rmsd": [{"seed": s, "all_atom_A": rms(f"s{s}_base", f"s{s}_{arms[1]}"),
                              "ca_A": rms(f"s{s}_base", f"s{s}_{arms[1]}", ca),
                              "plddt_delta": round(folds[f"s{s}_{arms[1]}"]["plddt"]
                                                   - folds[f"s{s}_base"]["plddt"], 6)}
                             for s in seeds],
        "seed_floor_rmsd": [{"pair": f"base s{a} vs base s{b}",
                             "all_atom_A": rms(f"s{a}_base", f"s{b}_base"),
                             "ca_A": rms(f"s{a}_base", f"s{b}_base", ca)}
                            for a, b in itertools.combinations(seeds, 2)],
        "aa_control": {"all_atom_A": rms(f"s{seeds[0]}_base", aa_tag),
                       "digests_identical": folds[aa_tag]["digest"]
                       == folds[f"s{seeds[0]}_base"]["digest"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))

    print(f"\n  CA-lDDT against {GT.name}, per pseudo-domain")
    for dom in doms:
        s = summary[dom]
        print(f"   {dom}")
        print("     base  " + "  ".join(
            f"s{x} {native[f's{x}_base'][dom]['ca_lddt_gt']:.6f}" for x in seeds)
            + f"   mean {s['base_mean']:.6f}")
        print(f"     {arms[1]:>4}  " + "  ".join(
            f"s{x} {native[f's{x}_{arms[1]}'][dom]['ca_lddt_gt']:.6f}" for x in seeds)
            + f"   mean {s['arm_mean']:.6f}")
        print(f"     paired {s['paired_delta_mean']:+.6f}  CI95 +/-{s['paired_ci95']:.6f}  "
              f"resolved {s['resolved']}  rank-sep {s['rank_separated']}  "
              f"seed spread {s['base_seed_spread']:.6f}  A/A {s['aa_delta']:+.6f}")
    print("\n  arm vs base, same seed (all-atom / CA / dplDDT):")
    for x in res["arm_vs_base_rmsd"]:
        print(f"    s{x['seed']}  {x['all_atom_A']:8.4f} / {x['ca_A']:8.4f} A / "
              f"{x['plddt_delta']:+.6f}")
    print("  seed floor, base against base at another seed:")
    for x in res["seed_floor_rmsd"]:
        print(f"    {x['pair']:22s} {x['all_atom_A']:8.4f} / {x['ca_A']:8.4f} A")
    print(f"  A/A control {res['aa_control']['all_atom_A']:.4f} A, digests identical="
          f"{res['aa_control']['digests_identical']}")
    ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
