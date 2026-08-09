#!/usr/bin/env python3
"""Fold-level parity for the trimul output-projection arm at 298 aa.

The op-level A/B says minimal_matmul on the two trimul output projections is 1.117x per trimul
and 1.0384x on the Pairformer block, but it is not bit-exact and one block amplifies the
deviation to PCC 0.99888 on a random input. A random pair tensor drives the gates and the
softmax somewhere the model never goes, so that number cannot decide anything. This folds the
real 298 aa target (CDK2, examples/prot300.yaml, the committed 35-sequence alignment) on both
arms at production config and compares structures.

The floor it is measured against is the model's own run-to-run nondeterminism: `--repeat` folds
per arm at the same seed, so the base-vs-base spread is the noise the arm has to beat to be
called a difference. Reports all-atom RMSD after Kabsch, plus pLDDT.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:perfwar-trimul-kernel \
        python3 -u perf/trimul_kernel/w2_fold_parity.py --repeat 3
"""

import argparse
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))


def cif_coords(path: Path):
    """(atom_key, xyz) for every ATOM record, in file order."""
    keys, xyz = [], []
    hdr = None
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            hdr = hdr or []
            hdr.append(line.strip().split(".")[1])
            continue
        if hdr and (line.startswith("ATOM") or line.startswith("HETATM")):
            f = line.split()
            c = dict(zip(hdr, f))
            keys.append((c.get("label_asym_id"), c.get("label_seq_id"), c.get("label_atom_id")))
            xyz.append((float(c["Cartn_x"]), float(c["Cartn_y"]), float(c["Cartn_z"])))
    return keys, np.asarray(xyz, dtype=np.float64)


def kabsch_rmsd(a, b):
    """All-atom RMSD after optimal rigid superposition."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3, help="warm folds per arm")
    ap.add_argument("--target", default=str(REPO / "examples" / "prot300.yaml"))
    ap.add_argument("--msa-a3m",
                    default=str(REPO / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    keep = Path(tempfile.mkdtemp(prefix="w2fold-cifs-"))
    print(f"cifs -> {keep}", flush=True)

    import tt_baseline
    import tt_bio.tenstorrent as T

    T._TRIMUL_OUT_MOVE_DRAM = False
    work = Path(tempfile.mkdtemp(prefix="w2fold-"))
    # ONE model load; the arm is a module-level flag read at call time inside the trimul, so
    # both arms fold through the same weights, the same features and the same MSA cache. That
    # makes the arm the only difference between the two sets of structures.
    one_fold, meta, _state = tt_baseline.build_fold(
        "protenix-v2", work / "msa", Path(args.target), Path(args.msa_a3m))
    struct_dir = Path(meta["struct_dir"])
    print(f"loaded in {meta['load_s']}s, n_msa={meta['n_msa']}, "
          f"region={meta['timed_region']}", flush=True)

    def fold(tag, mm_out, cold=False):
        T._TRIMUL_MM_OUT = mm_out
        _s, m = one_fold()
        cifs = sorted(struct_dir.rglob("*.cif"))
        assert cifs, f"{tag}: fold wrote no cif into {struct_dir}"
        dst = keep / f"{tag}.cif"
        shutil.copyfile(cifs[-1], dst)
        print(f"  {tag} mm_out={mm_out} {_s:.2f}s plddt={m.get('plddt')} "
              f"msa={m.get('msa')} n_tokens={m.get('n_tokens')}"
              f"{' (cold, discarded)' if cold else ''}", flush=True)
        assert m.get("msa"), f"{tag}: fold ran without an MSA"
        return dict(fold=tag, cold=cold, plddt=m.get("plddt"), cif=str(dst))

    arms = {"base": [], "mm_out": []}
    # One cold fold per arm absorbs that arm's first-kernel compile; neither is compared.
    fold("cold_base", False, cold=True)
    fold("cold_mm_out", True, cold=True)
    for k in range(args.repeat):
        arms["base"].append(fold(f"base_{k}", False))
    for k in range(args.repeat):
        arms["mm_out"].append(fold(f"mm_out_{k}", True))

    # Warm folds only; the cold fold is a different code path (first-kernel compile).
    coords, ref_keys = {}, None
    for name, folds in arms.items():
        for f in folds:
            if f["cold"]:
                continue
            k, x = cif_coords(Path(f["cif"]))
            ref_keys = ref_keys or k
            assert k == ref_keys, f"{f['cif']} atom order differs from the reference"
            coords[(name, f["fold"])] = x

    def spread(pairs):
        return [round(kabsch_rmsd(coords[p], coords[q]), 4) for p, q in pairs]

    base = [k for k in coords if k[0] == "base"]
    mm = [k for k in coords if k[0] == "mm_out"]
    within_base = spread(list(itertools.combinations(base, 2)))
    within_mm = spread(list(itertools.combinations(mm, 2)))
    across = spread([(b, m) for b in base for m in mm])

    res = dict(
        n_atoms=len(ref_keys),
        plddt=dict((n, [f["plddt"] for f in fs if not f["cold"]]) for n, fs in arms.items()),
        rmsd_within_base=within_base, rmsd_within_mm_out=within_mm, rmsd_across=across,
        noise_floor_max=round(max(within_base + within_mm), 4) if within_base else None,
        across_max=round(max(across), 4) if across else None,
        cif_dir=str(keep),
    )
    print(json.dumps(res, indent=2), flush=True)
    if res["noise_floor_max"] is not None:
        verdict = ("INSIDE the noise floor" if res["across_max"] <= res["noise_floor_max"]
                   else "OUTSIDE the noise floor")
        print(f"VERDICT: across-arm RMSD max {res['across_max']} A vs run-to-run floor "
              f"{res['noise_floor_max']} A -> {verdict}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
