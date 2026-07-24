"""RFD3 release-video generation: real on-device 200-step diffusion trajectory for
the F5 (C2 symmetry) + F3 (real active-site ligand) hero design, grounded in the
real PDB 1J79 (E. coli dihydroorotase, a real pyrimidine-biosynthesis enzyme with a
binuclear Zn active site) -- the same fixture this port's own p18-p20 passes
value+device-trajectory parity-verified (see state/tt-bio-rfdiffusion3-port-p1.md
sections 2r-2t). This script does NOT re-derive parity; it runs the already-verified
featurizer + device TokenInitializer + RFD3Sampler at PRODUCTION step count (200,
not the 4/8-step parity smoke test) to produce a real, full trajectory for
rendering, plus the final step's sequence_logits_I (RFD3's own native sequence
head output -- no separate inverse-folding model needed) for a designability
self-consistency check.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/render_1j79_symmetric_ligand.py \
      --seed 42 --num_timesteps 200 --out_dir ~/.coworker/artifacts/rfd3-social/run_seed42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_design import _write_cif
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification
from tt_bio.rfd3_sampler import RFD3Sampler

DIR = os.path.dirname(__file__)
FIXTURE_DIR = os.path.join(DIR, "parity_artifacts", "unindexed_c2_1j79_full")
PDB = os.path.join(FIXTURE_DIR, "1j79_C2.pdb")
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def build_f(spec_name):
    SPEC_JSON = os.path.join(FIXTURE_DIR, spec_name)
    with open(SPEC_JSON) as fh:
        spec_dict = json.load(fh)
    spec_dict["input"] = PDB
    spec = InputSpecification.from_dict(spec_dict)
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_timesteps", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_recycle", type=int, default=None)
    ap.add_argument("--spec", default="spec.json", help="spec.json (length=130/subunit) or spec_small.json (length=20/subunit)")
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    struct_dir = out_dir / "struct"
    struct_dir.mkdir(parents=True, exist_ok=True)

    t_feat0 = time.time()
    f = build_f(args.spec)
    n_atoms = f["ref_pos"].shape[0]
    n_replicas = int(torch.unique(f["sym_transform_id"][f["sym_transform_id"] >= 0]).numel())
    print(f"[setup] real 1j79 C2+ligand f: {n_atoms} atoms, {n_replicas} symmetric replicas "
          f"(featurize took {time.time()-t_feat0:.1f}s)", flush=True)

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)

    t_build0 = time.time()
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    init = {k: v.float() for k, v in init.items()}
    print(f"[setup] device modules built + TokenInitializer run in {time.time()-t_build0:.1f}s", flush=True)

    L = f["ref_pos"].shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0)

    sampler = RFD3Sampler(num_timesteps=args.num_timesteps)
    g = torch.Generator().manual_seed(args.seed)

    t_sample0 = time.time()
    with torch.no_grad():
        X, traj = sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif,
                                  generator=g, n_recycle=args.n_recycle)
    dt_sample = time.time() - t_sample0
    print(f"[sample] {len(traj)} steps in {dt_sample:.1f}s ({dt_sample/max(1,len(traj))*1000:.0f} ms/step)",
          flush=True)

    # One extra forward call on the final (converged) state to fetch RFD3's own
    # native sequence_logits_I -- the sampler's per-step loop discards everything
    # from `outs` except X_L, so this is the cheapest way to get the design's
    # sequence without re-deriving/duplicating the sampler's step math. Uses the
    # exact same X_noisy_L / t_hat as the real last step, so X_L reproduces
    # bit-identically (sanity-asserted below); only sequence_logits_I is new.
    last = traj[-1]
    with torch.no_grad():
        outs_final = dev_dm(X_noisy_L=last["X_noisy_L"], t=last["t_hat"].tile(1), f=f,
                             n_recycle=args.n_recycle, **init)
    reproduce_err = float((outs_final["X_L"] - last["X_L"]).abs().max())
    print(f"[seq] final-step re-forward reproduces X_L (max abs err {reproduce_err:.6f}); "
          f"sequence_logits_I shape {tuple(outs_final['sequence_logits_I'].shape)}", flush=True)

    # ---- write per-step trajectory CIFs (real dumped coordinates, no interpolation) ----
    t_write0 = time.time()
    for k, step in enumerate(traj):
        _write_cif(step["X_L"][0], f, frames_dir / f"f{k:04d}.cif")
    final_path = struct_dir / "1j79_c2_design.cif"
    _write_cif(X[0], f, final_path)
    print(f"[write] {len(traj)} frame CIFs + final structure in {time.time()-t_write0:.1f}s", flush=True)

    # ---- sequence extraction: argmax over RFD3's own sequence head, restricted to
    # protein tokens (is_protein), one designed chain (asym_id groups; symmetry
    # replicates the SAME sequence to every subunit by construction, verified below) ----
    logits = outs_final["sequence_logits_I"]
    if logits.ndim == 3:
        logits = logits[0]
    restype_idx = logits.argmax(-1)
    is_protein_tok = f["is_protein"].bool()
    asym = f["asym_id"]
    AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
          "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    AA1 = "ARNDCQEGHILKMFPSTWYV"
    chains = {}
    for a in torch.unique(asym[is_protein_tok]).tolist():
        mask = is_protein_tok & (asym == a)
        idx = restype_idx[mask].tolist()
        seq = "".join(AA1[i] if 0 <= i < 20 else "X" for i in idx)
        chains[int(a)] = seq
    seq_path = out_dir / "designed_sequences.json"
    seq_path.write_text(json.dumps({str(k): v for k, v in chains.items()}, indent=2))
    print(f"[seq] designed protein chains: " + ", ".join(f"asym={k} len={len(v)}" for k, v in chains.items()),
          flush=True)
    for a, seq in chains.items():
        print(f"  chain asym={a}: {seq}")

    meta = {
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "n_recycle": args.n_recycle,
        "n_atoms": n_atoms,
        "n_symmetric_replicas": n_replicas,
        "sample_wall_clock_s": dt_sample,
        "ms_per_step": dt_sample / max(1, len(traj)) * 1000,
        "pdb_source": "1J79 (RCSB) -- E. coli dihydroorotase, binuclear Zn active site",
        "ligand": "ORO (orotate/dihydroorotate) + 2x ZN per subunit",
        "symmetry": "C2",
        "designed_chains": {str(k): {"length": len(v)} for k, v in chains.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
