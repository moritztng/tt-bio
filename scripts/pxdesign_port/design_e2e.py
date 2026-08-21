#!/usr/bin/env python3
"""Generation-only end-to-end for PXDesign-d on a captured anchor.

Drives `tt_bio.pxdesign.model.ProtenixDesign` from `parity_artifacts/pdl1_protenix05_noH/
ref_design_inputs.pt` -- the model-ready input dict captured from the upstream featurizer
itself (`capture_ref_design_f.py`). tt-bio has no CIF parser for Protenix features, and the
design-specific arithmetic on top of that atom array is already gated bit-exact
(`parity_gate.py`), so composing the two halves against an upstream-produced input is the
honest first end-to-end rather than a stub.

    TT_VISIBLE_DEVICES=1 python3 scripts/pxdesign_port/design_e2e.py --n_step 20
    TT_VISIBLE_DEVICES=1 python3 scripts/pxdesign_port/design_e2e.py --determinism

`--score_coords` scores a saved coordinate tensor instead of generating one, so the upstream
CPU reference (`upstream_ref.py --stage traj`) goes through these exact metrics rather than
through a second implementation of them. Needs no device.

    python3 scripts/pxdesign_port/design_e2e.py --score_coords /tmp/ref_traj_s0.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# The shipped PD-L1 CIF carries explicit hydrogens and PXDesign's CIF path has no
# hydrogen filter, so 61 of the target's 116 residues parse as unresolved and get
# conditioned on at the origin (see strip_cif_hydrogens.py). Default to the capture
# taken from the hydrogen-stripped copy; `--art` selects another.
ART = REPO / "scripts" / "pxdesign_port" / "parity_artifacts" / "pdl1_protenix05_noH"
CKPT = Path("~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt").expanduser()

# tt-bio's atom encoder concatenates the one-hots with the float channels, so the stored
# uint8 has to come back as float; the index features stay integer.
_FLOAT = ("ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask",
          "restype", "hotspot", "deletion_mean")
_LONG = ("ref_space_uid", "atom_to_token_idx", "asym_id", "residue_index", "entity_id",
         "sym_id", "token_index", "conditional_templ")


def load_design_inputs(path=None) -> dict:
    """The captured input dict, in the dtypes tt_bio.protenix expects."""
    import torch
    raw = torch.load(path or (ART / "ref_design_inputs.pt"), weights_only=False)
    feats = {}
    for k, v in raw.items():
        feats[k] = v.float() if k in _FLOAT else (v.long() if k in _LONG else v)
    return feats


def build(diffusion_fp32=None):
    from tt_bio.pxdesign.model import ProtenixDesign
    return ProtenixDesign.load_from_checkpoint(str(CKPT))


def _kabsch_rmsd(a, b):
    """RMSD after optimal rigid superposition of `a` onto `b` (host fp64, exact SVD --
    ttnn-host-kabsch: the alignment is a scoring step, not a device op)."""
    import torch
    a = a.double() - a.double().mean(0)
    b = b.double() - b.double().mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return float((a @ r - b).pow(2).sum(-1).mean().sqrt())


def target_reproduction(coords, feats, art=None):
    """The end-to-end correctness signal for the conditioning path.

    PXDesign is handed a 64-bin distogram of the target and nothing else about its
    coordinates, so a correct generator reproduces the target's own fold in the sample while
    the binder is free. Scored as Kabsch RMSD of the generated distogram-representative
    atoms of the CONDITIONED tokens against the target coordinates upstream fed to
    `get_condition_template_feature` -- the same tensor the featurizer gate scores against.
    A broken conditioning path (wrong embedding row, wrong bin edges, leaked placeholder)
    lands in the tens of angstroms here; nothing else in the pipeline complains.

    Also reports the closest binder-to-target atom distance: a design that never touches its
    target is a failure the RMSD cannot see."""
    import torch
    art = Path(art) if art else ART
    ref = torch.load(art / "ref_condition_inputs.pt", weights_only=False)
    gate = torch.load(art / "ref_design_f.pt", weights_only=False)
    disto = gate["distogram_rep_atom_mask"].bool()
    conditioned = torch.tensor([r != "xpb" for r in ref["res_name"]]) & ref["is_resolved"].bool()
    out = []
    a2t = feats["atom_to_token_idx"].long()
    binder_tok = feats["restype"].argmax(-1) == 32
    at_binder = binder_tok[a2t]
    for s in range(coords.shape[0]):
        rep = coords[s][disto]                       # (N_token, 3), one per token
        d = torch.cdist(coords[s][at_binder], coords[s][~at_binder])
        out.append({"target_rmsd": _kabsch_rmsd(rep[conditioned], ref["coord"][conditioned]),
                    "n_scored": int(conditioned.sum()),
                    "min_binder_target_dist": float(d.min()),
                    "n_contacts_below_5A": int((d < 5.0).any(-1).sum())})
    return out


def _stats(coords, feats):
    """Radius of gyration overall and per half, plus the binder's own extent. A collapsed
    or exploded diffusion shows up here before any structural metric does."""
    import torch
    a2t = feats["atom_to_token_idx"].long()
    binder = torch.zeros(int(a2t.max()) + 1, dtype=torch.bool)
    binder[feats["restype"].argmax(-1) == 32] = True          # 32 == xpb, the binder placeholder
    at_binder = binder[a2t]

    def rg(x):
        return float((x - x.mean(0)).pow(2).sum(-1).mean().sqrt()) if len(x) else float("nan")
    out = []
    for s in range(coords.shape[0]):
        x = coords[s]
        out.append({"rg_all": rg(x), "rg_target": rg(x[~at_binder]), "rg_binder": rg(x[at_binder])})
    return out, int(at_binder.sum()), int((~at_binder).sum())


def score_only(args, feats):
    """Score coordinates produced elsewhere, through the same two functions the device path
    uses. An upstream reference and the port are then never compared through two different
    implementations of the same metric."""
    import torch
    blob = torch.load(args.score_coords, weights_only=False)
    coords = blob["coords"] if isinstance(blob, dict) else blob
    coords = coords.reshape(-1, coords.shape[-2], coords.shape[-1]).float()
    st, n_b, n_t = _stats(coords, feats)
    repro = target_reproduction(coords, feats, art=args.art)
    rec = {"source": args.score_coords, "art": str(args.art or ART), "coords_shape": list(coords.shape),
           "binder_atoms": n_b, "target_atoms": n_t, "stats": st,
           "target_reproduction": repro,
           "finite": bool(torch.isfinite(coords).all())}
    print(f"[score] {args.score_coords}  shape={rec['coords_shape']}  "
          f"binder/target atoms {n_b}/{n_t}  finite={rec['finite']}", flush=True)
    for sv, r in zip(st, repro):
        print(f"[score]   Rg all {sv['rg_all']:.2f} A | target {sv['rg_target']:.2f} A | "
              f"binder {sv['rg_binder']:.2f} A", flush=True)
        print(f"[score]   target reproduction RMSD {r['target_rmsd']:.2f} A over "
              f"{r['n_scored']} conditioned tokens | closest binder-target atom "
              f"{r['min_binder_target_dist']:.2f} A | {r['n_contacts_below_5A']} binder "
              f"atoms within 5 A", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1))
        print(f"[score] -> {args.out}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_step", type=int, default=400)
    ap.add_argument("--n_sample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--determinism", action="store_true",
                    help="two solo runs at the same seed in one process; report the floor")
    ap.add_argument("--art", default=None,
                    help="capture directory to read the input dict and the scoring reference "
                         "from; defaults to parity_artifacts/pdl1_protenix05_noH")
    ap.add_argument("--score_coords", default=None,
                    help="score a saved (n_sample, N_atom, 3) tensor, or a dict with a "
                         "'coords' key, instead of generating one; no device needed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))

    import torch
    art = Path(args.art) if args.art else ART
    feats = load_design_inputs(art / "ref_design_inputs.pt")
    if args.score_coords:
        return score_only(args, feats)
    NT = int(feats["atom_to_token_idx"].max()) + 1
    print(f"[e2e] {Path(args.art or ART).name}: {NT} tokens, {feats['ref_pos'].shape[0]} atoms, "
          f"n_step={args.n_step} n_sample={args.n_sample} seed={args.seed}", flush=True)

    t0 = time.time()
    model = build()
    print(f"[e2e] model built in {time.time() - t0:.1f}s "
          f"(c_z={model.C_Z}, DiT blocks={model.diffusion.n_dit_blocks if hasattr(model.diffusion, 'n_dit_blocks') else '?'})",
          flush=True)

    rec = {"n_token": NT, "n_atom": int(feats["ref_pos"].shape[0]), "n_step": args.n_step,
           "n_sample": args.n_sample, "seed": args.seed}
    t0 = time.time()
    coords = model.design(feats, n_step=args.n_step, n_sample=args.n_sample, seed=args.seed)
    rec["seconds"] = round(time.time() - t0, 2)
    import hashlib
    rec["coords_sha16"] = hashlib.sha256(
        coords.contiguous().numpy().tobytes()).hexdigest()[:16]
    st, n_b, n_t = _stats(coords, feats)
    repro = target_reproduction(coords, feats, art=args.art)
    rec.update({"art": str(args.art or ART), "coords_shape": list(coords.shape), "binder_atoms": n_b, "target_atoms": n_t,
                "stats": st, "target_reproduction": repro,
                "finite": bool(torch.isfinite(coords).all())})
    print(f"[e2e] {rec['seconds']}s  shape={rec['coords_shape']}  "
          f"binder/target atoms {n_b}/{n_t}  finite={rec['finite']}  "
          f"coords_sha16={rec['coords_sha16']}", flush=True)
    for s, r in zip(st, repro):
        print(f"[e2e]   Rg all {s['rg_all']:.2f} A | target {s['rg_target']:.2f} A | "
              f"binder {s['rg_binder']:.2f} A", flush=True)
        print(f"[e2e]   target reproduction RMSD {r['target_rmsd']:.2f} A over "
              f"{r['n_scored']} conditioned tokens | closest binder-target atom "
              f"{r['min_binder_target_dist']:.2f} A | {r['n_contacts_below_5A']} binder "
              f"atoms within 5 A", flush=True)

    if args.determinism:
        c2 = model.design(feats, n_step=args.n_step, n_sample=args.n_sample, seed=args.seed)
        d = (coords - c2).abs()
        rec["determinism"] = {"bit_exact": bool(torch.equal(coords, c2)),
                              "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
        print(f"[e2e] determinism (same seed, same process): "
              f"bit_exact={rec['determinism']['bit_exact']} "
              f"max|d|={rec['determinism']['max_abs_diff']:.3e} "
              f"mean|d|={rec['determinism']['mean_abs_diff']:.3e}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1))
        print(f"[e2e] -> {args.out}", flush=True)
    return rec


if __name__ == "__main__":
    main()
