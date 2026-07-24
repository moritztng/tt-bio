"""RFD3 design runtime: assemble the on-device TokenInitializer + DiffusionModule
+ RFD3Sampler into a `tt-bio design` run that writes a CIF per design.

This is the user-facing entry point for RFD3 (a *design* model, not a folder):
it parses an InputSpecification (JSON/YAML) via :mod:`tt_bio.rfd3_input`,
validates it, runs the on-device diffusion sampler, and writes the designed
structure to disk.

Status (p12): the host featurizer (:mod:`tt_bio.rfd3_featurize`) is
value-parity-verified (43/43 `f` keys bit-exact vs a real reference capture,
see ``scripts/rfd3_port/parity_artifacts/``) for the protein-binder (F1) /
motif-scaffolding (F6) case. ``--from_pdb`` runs the real end-to-end path
(featurize → on-device TokenInitializer → sampler → CIF) without a captured
golden for the features; ``--golden_dir`` is still required for the device
ckpt weights (both paths need it). The fixed-motif atoms are seeded at their
real (centered) ground-truth position via ``f["motif_pos"]`` — the sampler
never moves them, so this is what actually appears in the output structure.
Full end-to-end design accuracy (device output vs an independent reference
RFD3 sampler run on the same real input) has not been separately measured
this pass; the earlier device-vs-reference sampler parity numbers (p5/p6)
used a placeholder zero motif seed on both sides, so they remain valid as
parity claims but don't cover this real-seed path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .rfd3 import build_diffusion_module, build_token_initializer
from .rfd3_input import InputSpecification, parse_contig
from .rfd3_sampler import RFD3Sampler


@dataclass
class DesignResult:
    spec_id: str
    out_path: Path
    final_pcc_vs_ref: float | None  # only set when a reference DM run is paired
    n_atoms: int


def extract_rfd3_weights(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    """Split a raw RFD3 training checkpoint into the two weight files tt-bio
    actually loads (TokenInitializer + DiffusionModule submodules, prefix
    stripped). Shared by scripts/rfd3_port/extract_weights.py (manual/dev use)
    and tt_bio.main's auto-downloader (real users)."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=False)
    model_sd = ck["model"]

    def extract(submodule):
        for prefix in (f"{submodule}.", f"model.{submodule}."):
            sub = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
            if sub:
                return sub, prefix
        return {}, None

    for sub in ("token_initializer", "diffusion_module"):
        weights, prefix = extract(sub)
        out_path = out_dir / f"{sub}.real_weights.pt"
        torch.save({k: v.detach().cpu() for k, v in weights.items()}, out_path)
        meta = {"n_keys": len(weights), "prefix": prefix,
                "keys": sorted(weights.keys()),
                "shapes": {k: [list(v.shape), str(v.dtype)] for k, v in weights.items()}}
        (out_dir / f"{sub}.real_weights.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[extract] {sub}: {len(weights)} tensors (prefix '{prefix}') -> {out_path}", flush=True)
    return out_dir


def _load_golden_f(cap_dir: str) -> dict:
    """Reconstruct the `f` feature dict from a captured golden (the parity
    fixture). Mirrors scripts/rfd3_port/verify_sampler.reconstruct_f."""
    import glob
    cap = Path(cap_dir)
    f = {}
    for p in glob.glob(str(cap / "token_initializer.in_f_*.pt")):
        k = Path(p).name[len("token_initializer.in_f_"):-3]
        t = torch.load(p, map_location="cpu", weights_only=True)
        if t.is_floating_point() and t.dtype != torch.float32:
            t = t.float()
        f[k] = t
    return f


def _write_cif(coords, f, out_path: Path, b_factors=None):
    """Write the designed structure as mmCIF via biotite, reconstructed from the
    feature dict (same approach as _write_protenix_structure; RFD3's `f` shares
    the AF3-family feature keys)."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as _pdbx

    a2t = f["atom_to_token_map"].tolist()
    # ref_element: [N_atom, 128] one-hot over element, index 0 = unknown/padding,
    # index N (N>=1) = atomic number N directly (see rfd3_featurize.py's
    # `ref_element[i, _ELEMENT_TO_ATOMIC_NUMBER[elem]] = 1.0` -- NOT "index =
    # atomic number - 1" as a stale comment here previously claimed; the extra
    # "+1" below used to shift every element by one row, e.g. real N/C/O/Zn were
    # written as O/N/F/Ga -- cosmetically silent (coordinates/atom names were
    # unaffected) but broke any downstream `elem Zn`-style element-based
    # selection (PyMOL, BoltzGen's mmCIF parser, etc).
    z_idx = f["ref_element"].argmax(-1).tolist() if f["ref_element"].ndim == 2 else f["ref_element"].tolist()
    from tt_bio.data import const
    z2sym = getattr(const, "atomic_num_to_element", None) or {z: s for s, z in const.element_to_atomic_num.items()}
    rt = f["restype"].argmax(-1) if f["restype"].ndim == 2 else f["restype"]
    rt = rt.tolist()
    # ref_atom_name_chars: [N_atom, 256] = [N_atom, 4, 64] one-hot over 4 chars (idx -> chr(idx+32)).
    anc = f["ref_atom_name_chars"]
    if anc.ndim == 2 and anc.shape[-1] == 256:
        anc = anc.reshape(anc.shape[0], 4, 64)
    name_idx = anc.argmax(-1).tolist()  # [N_atom, 4]
    names = ["".join(chr(c + 32) for c in chars).strip() for chars in name_idx]
    asym = f["asym_id"].tolist(); resid = f["residue_index"].tolist()

    arr = struc.AtomArray(coords.shape[0])
    arr.coord = coords.numpy().astype("float32")
    arr.add_annotation("occupancy", float); arr.occupancy[:] = 1.0
    arr.add_annotation("b_factor", float)
    if b_factors is not None:
        arr.b_factor[:] = b_factors.numpy().astype("float32")
    for i in range(coords.shape[0]):
        t = a2t[i]
        arr.chain_id[i] = _chain_label(int(asym[t]))
        arr.res_id[i] = int(resid[t])
        arr.atom_name[i] = names[i]
        z = int(z_idx[i])
        arr.element[i] = z2sym.get(z, "C")
        arr.res_name[i] = "LIG" if rt[t] == 20 else _resname(rt[t])
    cf = _pdbx.CIFFile(); _pdbx.set_structure(cf, arr); cf.write(str(out_path))


def _chain_label(asym: int) -> str:
    # asym_id is 0-based (parity-verified vs a real reference capture, p11/p12:
    # a single chain -> asym_id all 0). Negative values (none in the featurizer's
    # output) fall back to a placeholder label.
    if asym < 0:
        return "Z"
    if asym < 26:
        return chr(ord("A") + asym)
    return chr(ord("A") + asym // 26 - 1) + chr(ord("A") + asym % 26)


def _resname(rt_idx: int) -> str:
    # minimal 20-AA map (restype index order matches AF3/RFD3 standard 20 + UNK)
    names = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
             "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    if 0 <= rt_idx < 20:
        return names[rt_idx]
    return "UNK"


def run_design(
    specs: Mapping[str, Mapping],
    out_dir: str | Path,
    *,
    golden_dir: str | None = None,
    from_pdb: bool = False,
    num_timesteps: int = 4,
    seed: int = 42,
    partial_t: float | None = None,
    cfg_scale: float | None = None,
    fp32_residual: bool = False,
    device_visible: str = "0",
    verbose: bool = True,
) -> list[DesignResult]:
    """Run one on-device diffusion design per InputSpecification.

    Parameters
    ----------
    specs : {spec_id: spec_dict}
        The parsed JSON/YAML InputSpecification file (each top-level key is one
        design). Each spec is validated via :class:`InputSpecification`.
    out_dir : output directory (created if missing).
    golden_dir : path to the captured RFD3 device ckpt weights (always required —
        this holds the TokenInitializer/DiffusionModule weights regardless of
        the feature source).
    from_pdb : if True, build `f` from each spec's real `input` PDB + contig via
        :mod:`tt_bio.rfd3_featurize` (parity-verified for the F1/F6
        protein-binder/motif-scaffold case, see scripts/rfd3_port/parity_artifacts/).
        If False, fall back to the captured golden `f` bridge (p9).
    num_timesteps, seed, partial_t, cfg_scale, fp32_residual : sampler knobs.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if fp32_residual:
        os.environ["RFD3_FP32_RESIDUAL"] = "1"
    if golden_dir is None:
        raise ValueError("golden_dir is required (it holds the device ckpt weights)")

    results: list[DesignResult] = []
    cap = Path(golden_dir)
    dm_weights = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    ti_weights = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    sampler = RFD3Sampler(num_timesteps=num_timesteps)

    # golden-bridge path: one captured f + init shared across specs
    golden_f = None; golden_init = None; golden_L = None; golden_is_motif = None
    if not from_pdb:
        if not (cap / "token_initializer.out_Q_L_init.pt").exists():
            raise ValueError(
                f"{cap} has weights but no captured golden `f` bridge (it is an internal "
                "dev/test fixture, not publicly distributed). Pass --from_pdb to featurize "
                "from a real input PDB instead.")
        golden_f = _load_golden_f(str(cap))
        Q_L_init = torch.load(cap / "token_initializer.out_Q_L_init.pt", map_location="cpu", weights_only=True).float()
        C_L = torch.load(cap / "token_initializer.out_C_L.pt", map_location="cpu", weights_only=True).float()
        P_LL = torch.load(cap / "token_initializer.out_P_LL.pt", map_location="cpu", weights_only=True).float()
        S_I = torch.load(cap / "token_initializer.out_S_I.pt", map_location="cpu", weights_only=True).float()
        Z_II = torch.load(cap / "token_initializer.out_Z_II.pt", map_location="cpu", weights_only=True).float()
        golden_init = dict(Q_L_init=Q_L_init, C_L=C_L, P_LL=P_LL, S_I=S_I, Z_II=Z_II)
        golden_L = Q_L_init.shape[0]
        golden_is_motif = golden_f["is_motif_atom_with_fixed_coord"]

    for spec_id, raw in specs.items():
        spec = InputSpecification.from_dict(raw)
        spec.validate()
        if verbose:
            print(f"[design:{spec_id}] contig={spec.contig!r} length={spec.length!r} "
                  f"ligand={spec.ligand!r} partial_t={spec.partial_t} from_pdb={from_pdb}")
        sp_t = spec.partial_t if spec.partial_t is not None else partial_t

        if from_pdb:
            # real from-PDB path: featurize the spec's input PDB + contig, run
            # the on-device TokenInitializer on the ported f. NOT parity-gated.
            from .rfd3_featurize import featurize
            if spec.input is None:
                raise ValueError(f"spec {spec_id!r} has no `input` PDB (required for --from_pdb)")
            f = featurize(spec.input, spec)
            with torch.no_grad():
                init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
            L = init["Q_L_init"].shape[0]
            is_motif = f["is_motif_atom_with_fixed_coord"]
            f_used = f; init_used = init
        else:
            f_used = golden_f; init_used = golden_init
            L = golden_L; is_motif = golden_is_motif

        # Seed the trajectory's fixed-motif atoms at their real (centered)
        # ground-truth position — the sampler never adds noise or a delta
        # update at is_motif_fixed atoms (rfd3_sampler.RFD3Sampler.sample), so
        # whatever `coord` holds there is exactly what comes out. `motif_pos`
        # is zero everywhere else, so this is a correct seed for both the
        # designed (start-from-noise) and motif (start-at-truth) atoms.
        coord0 = f_used["motif_pos"].float().unsqueeze(0) if "motif_pos" in f_used else torch.zeros(1, L, 3)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            X, _ = sampler.sample(dev_dm, 1, L, coord0, f_used, init_used, is_motif,
                                  generator=g, partial_t=sp_t, cfg_scale=cfg_scale)
        out_path = out_dir / f"{spec_id}.cif"
        _write_cif(X[0], f_used, out_path)
        results.append(DesignResult(spec_id=spec_id, out_path=out_path,
                                    final_pcc_vs_ref=None, n_atoms=int(X.shape[1])))
        if verbose:
            print(f"[design:{spec_id}] wrote {out_path} ({X.shape[1]} atoms)")
    return results
