"""Multi-step trajectory parity for the p22 dict-form `unindex` case: same
real IAI_protein.pdb + contig "A1-10,20,A31-40" as verify_trajectory_unindex.py,
but `unindex` is now a dict with a RANGE key and an atom-name-selector value:
`{"A100-101": "CB,CA"}` -- TRP100 (14 real atoms) and VAL101 (7 real atoms)
both become unindexed, tied (range key), each subsetted to just its own
CB/CA (real reference mechanism: the dict VALUE directly subsets which real
atoms enter the unindexed token, via `InputSelection.get_tokens`, independent
of `select_fixed_atoms` -- see `_unindex_dict_atom_names`).

Real regression check this fixture exists for: TRP100/VAL101 keep only 2
real atoms each (not their full residue) -- if range-key expansion or the
dict-value atom subsetting were silently wrong (e.g. matched nothing, or kept
every atom), this would either crash (component not found) or produce a much
larger L / wrong `is_motif_atom_with_fixed_coord` count, both caught below.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_trajectory_unindex_dictform.py [num_timesteps]
"""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_dm_ref import RFD3DiffusionModuleRef, EDMSamplerRef, SharedDraws

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
UNINDEX = {"A100-101": "CB,CA"}
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def rmsd(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt())


def main(n_ts):
    torch.manual_seed(0)
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG, "unindex": UNINDEX})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    I = f["restype"].shape[0]
    n_unindexed_tok = int(f["is_motif_token_unindexed"].sum())
    print(f"[setup] real IAI-protein + dict-form range-key unindex f: I={I} L={L} "
          f"({n_unindexed_tok} unindexed tokens, {int(is_motif.sum())} fixed atoms)")
    assert n_unindexed_tok == 2, f"expected 2 unindexed tokens (TRP100, VAL101), got {n_unindexed_tok}"

    atom_to_token = f["atom_to_token_map"]
    unindexed_atom_counts = [int((atom_to_token == ti).sum())
                              for ti in range(I) if f["is_motif_token_unindexed"][ti]]
    print(f"[setup] unindexed token atom counts: {unindexed_atom_counts} (expect [2, 2] -- CB,CA only)")
    assert unindexed_atom_counts == [2, 2], \
        f"dict-value atom subsetting failed: expected 2 atoms (CB,CA) per unindexed token, got {unindexed_atom_counts}"

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)

    dev_ti = build_token_initializer(ti_weights)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    init = {k: v.float() for k, v in init.items()}

    coord = f["motif_pos"].unsqueeze(0)
    sampler = EDMSamplerRef(num_timesteps=n_ts)

    ref = RFD3DiffusionModuleRef().eval()
    m, u = ref.load_state_dict(dm_weights, strict=False)
    print(f"[ref] weights missing={len(m)} unexpected={len(u)}")
    with torch.no_grad():
        X_ref, traj_ref = sampler.sample(ref, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    dm = build_diffusion_module(dm_weights)
    X_tt, traj_tt = sampler.sample(dm, 1, L, coord, f, init, SharedDraws(seed=42), is_motif)

    assert len(traj_ref) == len(traj_tt), (len(traj_ref), len(traj_tt))
    print(f"[traj] {len(traj_ref)} steps; per-step X_L parity (device vs vendored-torch reference):")
    worst = 1.0
    for i, (r, t) in enumerate(zip(traj_ref, traj_tt)):
        p = pcc(t["X_L"], r["X_L"]); rr = rmsd(t["X_L"], r["X_L"])
        worst = min(worst, p)
        print(f"  step {i}: t_hat={float(r['t_hat']):.3f}  PCC={p:.6f}  RMSD={rr:.4f}")
    pf = pcc(X_tt, X_ref); rf = rmsd(X_tt, X_ref)
    motif_seed_ok_ref = bool(torch.equal(X_ref[0, is_motif], coord[0, is_motif]))
    motif_seed_ok_tt = bool(torch.equal(X_tt[0, is_motif], coord[0, is_motif]))
    print(f"[final] X_L PCC={pf:.6f}  RMSD={rf:.4f}  (ref std {X_ref.std().item():.4f})  worst-step PCC={worst:.6f}")
    print(f"[motif] final fixed-atom coords == seeded motif_pos: ref={motif_seed_ok_ref} device={motif_seed_ok_tt}")
    if worst < 0.97 or pf < 0.97:
        raise AssertionError(f"trajectory parity failed on the dict-form-unindex input: "
                              f"worst-step PCC={worst:.6f} final PCC={pf:.6f}")
    if not (motif_seed_ok_ref and motif_seed_ok_tt):
        raise AssertionError("fixed atoms moved away from their seeded ground-truth position")
    print("TRAJECTORY PARITY OK on the dict-form range-key unindex input (per-step PCC >= 0.97, "
          "fixed atoms verified locked, dict-value atom subsetting verified 2/2 tokens, on both backends)")


if __name__ == "__main__":
    n_ts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_ts)
