#!/usr/bin/env python3
"""Fold a real Fv with ABodyBuilder3's released weights and score it with B1's instrument.

This is the end-to-end check the port exists for, and it validates four things at once that are
otherwise only checked in pieces: that the released checkpoint loads into our module tree, that our
featurisation reproduces theirs, that the forward is right on TRAINED weights rather than random
ones, and that the output path lands a number the instrument can compare.

It is self-validating in one specific way. `residue_index` is not in the released output -- it comes
from their dataset -- so it is taken here from the written PDB numbering (heavy 1.., light 501..),
which is what their own writer emits and what makes the cross-chain relative position saturate the
+-64 clamp that `edge_chain_feature` then disambiguates. If that guess were wrong the pair features
would be wrong for every residue and the fold would be visibly bad, so a result near their own
prediction's RMSD is evidence for the choice rather than something the choice was fitted to.

Reported next to ours, from the same instrument: their refined structure (the published number) and
their unrefined one, because ours is unrefined and that is the comparison that is apples to apples.

`--device` runs the same structures through the card instead of the reference, which turns the
port's 0.004 A rmsd against the reference into a number in the same units as their published table:
the RMSD of a device-folded structure against their truth, scored by their instrument.

Run: PYTHONPATH=$PWD python3 scripts/abb3_port/fold_gate.py <output-dir> [--n 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tt_bio.abodybuilder3_output import (atom14_from_frames, device_outputs_to_host, score_fv,
                                          write_fv_pdb)
from tt_bio.abodybuilder3_reference import load_abb3_model, single_and_pair_features
from tt_bio.af2_data import RESTYPE_ATOM14_MASK
from tt_bio._vendor.esm.utils import residue_constants as _rc

RESTYPE_INDEX = {_rc.restype_1to3[aa]: i for i, aa in enumerate(_rc.restypes)}


def features_from_pdb(path: Path):
    """`(aatype, is_heavy, residue_index)` from a written Fv, one entry per CA."""
    aatype, is_heavy, residue_index = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        aatype.append(RESTYPE_INDEX.get(line[17:20].strip(), len(_rc.restypes)))
        is_heavy.append(1 if line[21] == "H" else 0)
        residue_index.append(int(line[22:26]))
    return (torch.tensor(aatype), torch.tensor(is_heavy), torch.tensor(residue_index))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--variant", default="base-loss")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", action="store_true", help="fold on the card, not the reference")
    args = ap.parse_args()

    ckpt = torch.load(args.root / args.variant / "best_second_stage.ckpt", map_location="cpu",
                      weights_only=False)
    model = load_abb3_model(ckpt["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {args.variant} epoch {ckpt['epoch']} step {ckpt['global_step']}, "
          f"{n_params:,} parameters, folding on {'the card' if args.device else 'the reference'}")

    device_model = dev = None
    if args.device:
        from tt_bio.abodybuilder3 import DeviceABB3, to_device_fp32
        from tt_bio.abodybuilder3_reference import ABB3Config
        from tt_bio.tenstorrent import get_device
        dev = get_device()
        device_model = DeviceABB3(ckpt["state_dict"], ABB3Config(use_plddt=False),
                                  to_device=to_device_fp32)

    names = sorted(p.stem for p in (args.root / args.variant / "plddt").glob("*.pt"))[: args.n]
    rows = []
    for name in names:
        regions = list(torch.load(args.root / args.variant / "plddt" / f"{name}.pt",
                                  weights_only=False)["region"])
        true_pdb = args.root / args.variant / "true" / f"{name}.pdb"
        aatype, is_heavy, residue_index = features_from_pdb(true_pdb)
        if len(aatype) != len(regions):
            print(f"  skip {name}: {len(aatype)} residues against {len(regions)} regions")
            continue
        single, pair = single_and_pair_features(aatype.unsqueeze(0), is_heavy.unsqueeze(0),
                                                residue_index.unsqueeze(0))
        if args.device:
            import ttnn
            from tt_bio.abodybuilder3 import to_device_fp32
            mask = torch.ones(1, len(aatype))
            square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
            out = device_model(to_device_fp32(single), to_device_fp32(pair),
                               to_device_fp32(square), to_device_fp32(1e7 * (square - 1.0)))
            host = device_outputs_to_host(out, 7, taped=False)
            atom14 = atom14_from_frames(host["frames"], host["angles"],
                                        aatype.unsqueeze(0))[0]
        else:
            with torch.no_grad():
                out = model(single, pair, aatype.unsqueeze(0), torch.ones(1, len(aatype)))
            atom14 = out["positions"][-1][0]
        atom_mask = torch.as_tensor(RESTYPE_ATOM14_MASK)[aatype]
        ours = write_fv_pdb(Path("/tmp") / f"abb3_ours_{name}.pdb", aatype, atom14, atom_mask,
                            int(is_heavy.sum()))
        scored = {"ours": score_fv(true_pdb, ours, regions)}
        for label, sub in (("theirs refined", "refine"), ("theirs unrefined", "pred_unfixed")):
            path = args.root / args.variant / sub / f"{name}.pdb"
            if path.exists():
                scored[label] = score_fv(true_pdb, path, regions)
        rows.append((name, scored))
        parts = "  ".join(f"{k} {v['rmsd_cdrh3']:.3f}" for k, v in scored.items())
        print(f"  {name:<16} CDR-H3: {parts}")

    if not rows:
        print("\nFAIL: nothing scored")
        return 1
    ours = torch.tensor([r[1]["ours"]["rmsd_cdrh3"] for r in rows])
    theirs = torch.tensor([r[1]["theirs unrefined"]["rmsd_cdrh3"] for r in rows
                           if "theirs unrefined" in r[1]])
    print(f"\n  mean CDR-H3 over {len(ours)}: ours {ours.mean():.3f} A"
          + (f", theirs unrefined {theirs.mean():.3f} A" if len(theirs) else ""))
    fwh = torch.tensor([r[1]["ours"]["rmsd_fwh"] for r in rows])
    print(f"  mean Fw-H  over {len(fwh)}: ours {fwh.mean():.3f} A")
    # A wrong featurisation or a wrong weight mapping does not produce a plausible antibody: the
    # framework is the rigid part every ABodyBuilder3 variant gets to ~0.65 A, so a framework
    # above 2 A means something structural is wrong rather than the model being imperfect.
    ok = bool(fwh.mean() < 2.0)
    print(f"\n{'PASS' if ok else 'FAIL'} (bar: mean Fw-H < 2.0 A, the structural-sanity bar)")
    if dev is not None:
        import ttnn
        ttnn.close_device(dev)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
