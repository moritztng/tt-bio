"""RFD3 design runtime: assemble the on-device TokenInitializer + DiffusionModule
+ RFD3Sampler into a `tt-bio design` run that writes a CIF per design.

This is the user-facing entry point for RFD3 (a *design* model, not a folder):
it parses an InputSpecification (JSON/YAML) via :mod:`tt_bio.rfd3.input`,
validates it, runs the on-device diffusion sampler, and writes the designed
structure to disk.

Status (p12): the host featurizer (:mod:`tt_bio.rfd3.featurize`) is
value-parity-verified (43/43 `f` keys bit-exact vs a real reference capture,
see ``scripts/rfd3_port/parity_artifacts/``) for the protein-binder (F1) /
motif-scaffolding (F6) case. ``--from_pdb`` runs the real end-to-end path
(featurize → on-device TokenInitializer → sampler → CIF) without a captured
golden for the features; ``--checkpoint`` is still required for the device
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
import pickle
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .model import (build_diffusion_module, build_token_initializer,
                    set_tune_matmul_for_atoms)
from .input import InputSpecification
from .sampler import RFD3Sampler


# What a design batch costs in device DRAM, measured per op with the allocator
# instrumented (scripts/rfd3_port/p25_dram_headroom.py) rather than assumed: the peak grows
# linearly in the batch and about L^1.7 in atoms, so two bounds cover both ends of the
# (batch, atoms) space on a 32 GiB card.
#   * The atom-pair budget binds on large designs. 8 designs x 3359 atoms -- the largest
#     fixture the port is measured on -- peaks at 7.0 GiB of 31.9, and the cost per atom
#     pair keeps falling as designs grow (80 B at 3359 atoms, 86 B at 2702).
#   * The design ceiling binds on small ones, where an L^2 budget alone is permissive:
#     the budget would allow 514 designs of 419 atoms, which peaks at 11.9 GiB.
# Both bounds only ever shrink `batch_size`, and batching is bit-exact (identical
# trajectories at every size, scripts/rfd3_port/verify_batch_trajectory_parity.py), so
# these are memory numbers with no accuracy tradeoff hiding behind them.
_BATCH_ATOM_PAIR_BUDGET = 8 * 3359 * 3359
_BATCH_DESIGN_CEILING = 512

# Both bounds above only ask whether a batch FITS. Neither asks whether it is fast, and above
# about 3000 atoms it is not: a batched design costs more per design than running the designs
# one at a time. Measured end to end, seconds per design at the shipped 200 timesteps, qb2,
# one Blackhole p150a, ttnn 0.68.0, under a run lock, two warm forwards after a discarded cold
# one, every design validated (perf/dsfix/rfd3_batch_e2e.py, rows in
# perf/dsfix/results/rfd3_batch_e2e.jsonl):
#
#     atoms      b=1       b=2      b=4      b=8    fastest measured
#      2299   24.971         -   22.253   21.885    b=8, 1.141x over b=1
#      2952   36.625         -        -   34.108    b=8, 1.074x over b=1
#      3844   59.967    64.890   59.975        -    b=1, 1.082x over b=2 (b=4 ties it)
#      6051  144.044   167.189        -        -    b=1, 1.161x over b=2
#
# Batching stops paying between 2952 and 3844 atoms, so the cap binds above 2952, the largest
# size where it was measured to still pay. Two things the table says that a monotone reading
# would miss: at 3844 the curve is not monotone in batch -- 4 ties 1 to 0.01 % while 2 is 8 %
# worse than both -- and at 6051 the atom-pair budget admits exactly 2, which is the worst arm
# there. The clamp was landing on the pothole.
#
# The cap only ever shrinks the batch, so it cannot OOM, and it cannot change a design: the
# device forward is bit-identical across batch size, trajectory PCC 1.000000 and maxabs 0 at
# 200 timesteps (scripts/rfd3_port/verify_batch_trajectory_parity.py).
#
# Decided on end-to-end seconds per design and nothing else. The marginal per-step
# differential in perf/dsfix/results/rfd3_tt.jsonl measures a different quantity, since it
# subtracts out every per-forward fixed cost by construction, and it gets the sign of this
# effect wrong at 6051 atoms: it says batch 2 wins by 1.13x where the wall says batch 1 wins
# by 1.16x. Do not re-decide this from per-step numbers.
_BATCH_SPEED_CAP_ABOVE_ATOMS = 2952
_BATCH_SPEED_CAP = 1


@dataclass
class DesignResult:
    spec_id: str
    design_idx: int  # 0-based index within this spec's --num_designs draws
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
    fixture)."""
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


def _write_cif(coords, f, out_path: Path, b_factors=None, pred_restype=None):
    """Write the designed structure as mmCIF via biotite, reconstructed from the
    feature dict (same approach as _write_protenix_structure; RFD3's `f` shares
    the AF3-family feature keys). `pred_restype` is the sequence head's per-token
    argmax from the final diffusion step ([I] int): designed protein tokens are
    named from it instead of staying GAP->UNK."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as _pdbx

    a2t = f["atom_to_token_map"].tolist()
    # ref_element: [N_atom, 128] one-hot over element, index 0 = unknown/padding,
    # index N (N>=1) = atomic number N directly (see featurize.py's
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
    gly_tok = set()
    if pred_restype is not None:
        # A designed position enters as DESIGNED_RESTYPE_IDX (GAP) and would leave
        # as UNK no matter what the model predicted; name it from the sequence head
        # instead. Protein tokens only: a ligand or nucleic token keeps its input
        # identity. GLY has no CB (its atom14 slot 4 is None), so a residue predicted
        # GLY drops that atom in the keep loop below.
        from .featurize import DESIGNED_RESTYPE_IDX, _RESTYPE_ORDER
        is_prot = f["is_protein"].tolist()
        pred = pred_restype.tolist()
        for t in range(len(rt)):
            if rt[t] == DESIGNED_RESTYPE_IDX and is_prot[t]:
                rt[t] = int(pred[t])
                if _RESTYPE_ORDER[rt[t]] == "GLY":
                    gly_tok.add(t)
    # ref_atom_name_chars: [N_atom, 256] = [N_atom, 4, 64] one-hot over 4 chars (idx -> chr(idx+32)).
    anc = f["ref_atom_name_chars"]
    if anc.ndim == 2 and anc.shape[-1] == 256:
        anc = anc.reshape(anc.shape[0], 4, 64)
    name_idx = anc.argmax(-1).tolist()  # [N_atom, 4]
    names = ["".join(chr(c + 32) for c in chars).strip() for chars in name_idx]
    asym = f["asym_id"].tolist(); resid = f["residue_index"].tolist()
    is_lig = f["is_ligand"].tolist() if "is_ligand" in f else None

    # Atom14 slots 5..13 always carry the generic "V{i}" template name, for a
    # designed residue's synthetic pad atom AND for a motif residue's real
    # side-chain atom alike (featurize.py:490). The delivered CIF used to ship
    # both verbatim: 964 of des_rfd3_binder's 2051 atoms were named V0..V8.
    # Only the pad atoms are meant to leave the network, and only they are
    # flagged by `is_virtual` -- a mask the feature dict has carried since
    # featurize.py:2263 and that this writer never read. So: drop the pads, and
    # give the real side-chain atoms their names back.
    keep, real_names = [], {}
    for i in range(coords.shape[0]):
        slot = _virtual_slot(names[i])
        if slot is not None and _is_virtual(f, i):
            continue                              # synthetic pad atom, not chemistry
        if a2t[i] in gly_tok and names[i] == "CB":
            continue                              # GLY has no CB
        keep.append(i)
        if slot is not None:
            # A real motif side-chain atom wearing the template name. The name
            # itself pins the atom14 slot (ATOM14_ATOM_NAMES[5 + i]), so the
            # dense scheme resolves it exactly, with no reliance on atom order:
            # all 374 such atoms across the three rfd3 protocols resolve, none
            # lands on a None slot.
            real_names[i] = _dense_atom_name(_resname(int(rt[a2t[i]])), slot)

    arr = struc.AtomArray(len(keep))
    arr.coord = coords.numpy().astype("float32")[keep]
    arr.add_annotation("occupancy", float); arr.occupancy[:] = 1.0
    arr.add_annotation("b_factor", float)
    if b_factors is not None:
        arr.b_factor[:] = b_factors.numpy().astype("float32")[keep]
    for j, i in enumerate(keep):
        t = a2t[i]
        name = real_names.get(i, names[i])
        arr.chain_id[j] = _chain_label(int(asym[t]))
        arr.res_id[j] = int(resid[t])
        arr.atom_name[j] = name
        arr.element[j] = _element_of(int(z_idx[i]), name, z2sym)
        arr.res_name[j] = _resname(int(rt[t]), is_ligand=bool(is_lig[t]) if is_lig else False)
    cf = _pdbx.CIFFile(); _pdbx.set_structure(cf, arr); cf.write(str(out_path))


def _virtual_slot(name: str) -> int | None:
    """atom14 slot index for a generic "V{i}" template name, else None."""
    if len(name) == 2 and name[0] == "V" and name[1].isdigit():
        return 5 + int(name[1])
    return None


def _is_virtual(f, i: int) -> bool:
    iv = f.get("is_virtual") if hasattr(f, "get") else None
    # Without the mask every V-named atom is treated as a pad, which is the old
    # all-or-nothing behaviour and still better than shipping "V3" to a user.
    return True if iv is None else bool(iv[i])


def _dense_atom_name(res_name: str, slot: int) -> str:
    from .featurize import _DENSE_ATOM14_SCHEME

    real = _DENSE_ATOM14_SCHEME.get(res_name, (None,) * 14)[slot]
    return real or f"V{slot - 5}"


def _element_of(z: int, name: str, z2sym) -> str:
    # ref_element is never filled for protein (featurize.py:32), so z is the
    # index-0 unknown row for every protein atom and the old `z2sym.get(z, "C")`
    # labelled backbone N and O as carbon. For z >= 1 (ligand/nucleic, where the
    # featurizer does fill it) keep the atomic number. For z == 0 fall back to
    # the mmCIF convention: the leading alphabetic character of the atom name,
    # which is exact for the N/CA/C/O/CB names a protein atom can carry here.
    if z >= 1:
        sym = z2sym.get(z)
        if sym:
            return sym
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "C"


def _chain_label(asym: int) -> str:
    # asym_id is 0-based (parity-verified vs a real reference capture, p11/p12:
    # a single chain -> asym_id all 0). Negative values (none in the featurizer's
    # output) fall back to a placeholder label.
    if asym < 0:
        return "Z"
    if asym < 26:
        return chr(ord("A") + asym)
    return chr(ord("A") + asym // 26 - 1) + chr(ord("A") + asym % 26)


def _resname(rt_idx: int, *, is_ligand: bool = False) -> str:
    # featurize._RESTYPE_ORDER is the real AF3 32-token vocabulary and every one
    # of its first 31 entries is already the mmCIF comp_id: 0-19 the 20 AA, 20
    # UNK, 21-25 RNA A/C/G/U/N, 26-30 DNA DA/DC/DG/DT/DN. The old 20-entry map
    # wrote every nucleotide as UNK (all 1466 residues of an rfd3-na-binder
    # design), and its `rt == 20 -> "LIG"` special case had the index wrong in
    # both directions: 20 is unknown-AA, while a ligand's CCD code is not in the
    # vocabulary at all and lands on the GAP slot with the designed residues.
    from .featurize import _RESTYPE_ORDER, DESIGNED_RESTYPE_IDX

    if rt_idx == DESIGNED_RESTYPE_IDX:
        return "LIG" if is_ligand else "UNK"
    if 0 <= rt_idx < len(_RESTYPE_ORDER):
        return _RESTYPE_ORDER[rt_idx]
    return "UNK"


def run_design(
    specs: Mapping[str, Mapping],
    out_dir: str | Path,
    *,
    checkpoint_dir: str | None = None,
    from_pdb: bool = False,
    num_timesteps: int = 4,
    seed: int = 42,
    partial_t: float | None = None,
    cfg_scale: float | None = None,
    fp32_residual: bool = False,
    num_designs: int = 1,
    batch_size: int = 8,
    devices: Sequence[int] | None = None,
    host_threads: int | None = None,
    device_visible: str = "0",
    verbose: bool = True,
) -> list[DesignResult]:
    """Run on-device diffusion designs for a set of InputSpecifications.

    Parameters
    ----------
    specs : {spec_id: spec_dict}
        The parsed JSON/YAML InputSpecification file (each top-level key is one
        design). Each spec is validated via :class:`InputSpecification`.
    out_dir : output directory (created if missing).
    checkpoint_dir : path to the extracted RFD3 device ckpt weights (always required —
        this holds the TokenInitializer/DiffusionModule weights regardless of
        the feature source).
    from_pdb : if True, build `f` from each spec's real `input` PDB + contig via
        :mod:`tt_bio.rfd3.featurize` (parity-verified for the F1/F6
        protein-binder/motif-scaffold case, see scripts/rfd3_port/parity_artifacts/).
        If False, fall back to the captured golden `f` bridge (p9).
    num_timesteps, seed, partial_t, cfg_scale, fp32_residual : sampler knobs.
    num_designs : number of independent designs to produce per spec (each with a
        different noise seed = ``seed + design_idx``). Output files are
        ``<spec_id>.cif`` when ``num_designs == 1`` (back-compat) else
        ``<spec_id>_<i>.cif``.
    batch_size : maximum number of designs from one spec evaluated in a single
        device forward. The runtime automatically shrinks the batch for larger
        atom counts. Each design has its own seeded RNG stream and the device
        forward is bit-identical across batch size, so a batched design matches
        its standalone run exactly (see
        scripts/rfd3_port/verify_batch_invariance.py and
        scripts/rfd3_port/verify_batch_trajectory_parity.py).
    devices : list of physical TT card ids to fan the (spec x design_idx) jobs
        across, one pinned subprocess per card (data-parallel, the same pattern
        ``tt-bio embed``/``predict`` use). With 0/1 device the run is in-process
        on this card.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if fp32_residual:
        os.environ["RFD3_FP32_RESIDUAL"] = "1"
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required (it holds the device ckpt weights)")

    # Build the flat job list: (spec_id, design_idx, seed). Each spec is parsed
    # and validated up front so a bad input fails fast before any device work.
    parsed: list[tuple[str, InputSpecification]] = []
    for spec_id, raw in specs.items():
        spec = InputSpecification.from_dict(raw)
        spec.validate()
        parsed.append((spec_id, spec))
    jobs = [(sid, i, seed + i) for sid, _ in parsed for i in range(num_designs)]

    if devices and len(devices) > 1:
        return _run_design_fanout(jobs, specs, out_dir, checkpoint_dir=checkpoint_dir,
                                 from_pdb=from_pdb, num_timesteps=num_timesteps,
                                 partial_t=partial_t, cfg_scale=cfg_scale,
                                 fp32_residual=fp32_residual, batch_size=batch_size,
                                 multi_designs=num_designs > 1, devices=devices,
                                 host_threads=host_threads,
                                 verbose=verbose)
    if devices and "TT_VISIBLE_DEVICES" not in os.environ:
        # Single card, in-process path: get_device() opens TT_BIO_LOGICAL_DEVICE_ID
        # (default 0), so without this --devices 2 would silently run on card 0.
        # An explicit TT_VISIBLE_DEVICES already pins visibility and wins.
        os.environ.setdefault("TT_BIO_LOGICAL_DEVICE_ID", str(devices[0]))
    return _run_design_jobs(jobs, specs, out_dir, checkpoint_dir=checkpoint_dir,
                            from_pdb=from_pdb, num_timesteps=num_timesteps,
                            partial_t=partial_t, cfg_scale=cfg_scale,
                            fp32_residual=fp32_residual, batch_size=batch_size,
                            multi_designs=num_designs > 1, verbose=verbose)


def _run_design_jobs(jobs, specs, out_dir, *, checkpoint_dir, from_pdb, num_timesteps,
                     partial_t, cfg_scale, fp32_residual, batch_size,
                     multi_designs,
                     verbose=True) -> list[DesignResult]:
    """In-process: load weights once, run every (spec_id, design_idx, seed) job
    sequentially on this card. Featurize + TokenInitializer run once per spec and
    are reused across that spec's design_idx draws (they don't depend on the
    noise seed); only the sampler re-runs per design."""
    cap = Path(checkpoint_dir)
    dm_weights = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    ti_weights = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    sampler = RFD3Sampler(num_timesteps=num_timesteps)

    # golden-bridge path: one captured f + init shared across specs/designs
    golden_f = None; golden_init = None; golden_L = None; golden_is_motif = None
    if not from_pdb:
        if not (cap / "token_initializer.out_Q_L_init.pt").exists():
            raise ValueError(
                f"{cap} has weights but no captured golden `f` bridge (it is an internal "
                "dev/test fixture, not publicly distributed). Pass --from_pdb to featurize "
                "from a real input PDB instead.")
        golden_f = _load_golden_f(str(cap))
        golden_init = dict(
            Q_L_init=torch.load(cap / "token_initializer.out_Q_L_init.pt", map_location="cpu", weights_only=True).float(),
            C_L=torch.load(cap / "token_initializer.out_C_L.pt", map_location="cpu", weights_only=True).float(),
            P_LL=torch.load(cap / "token_initializer.out_P_LL.pt", map_location="cpu", weights_only=True).float(),
            S_I=torch.load(cap / "token_initializer.out_S_I.pt", map_location="cpu", weights_only=True).float(),
            Z_II=torch.load(cap / "token_initializer.out_Z_II.pt", map_location="cpu", weights_only=True).float(),
        )
        golden_L = golden_init["Q_L_init"].shape[0]
        golden_is_motif = golden_f["is_motif_atom_with_fixed_coord"]

    # Cache per-spec featurize+init (reused across that spec's design_idx draws).
    spec_feat: dict[str, tuple] = {}
    results: list[DesignResult] = []
    grouped_jobs: dict[str, list[tuple[str, int, int]]] = {}
    for job in jobs:
        grouped_jobs.setdefault(job[0], []).append(job)
    for spec_id, spec_jobs in grouped_jobs.items():
        spec = InputSpecification.from_dict(specs[spec_id])
        if spec_id not in spec_feat:
            if verbose:
                print(f"[design:{spec_id}] contig={spec.contig!r} length={spec.length!r} "
                      f"ligand={spec.ligand!r} partial_t={spec.partial_t} from_pdb={from_pdb}")
            if from_pdb:
                from .featurize import featurize
                if spec.input is None:
                    raise ValueError(f"spec {spec_id!r} has no `input` PDB (required for --from_pdb)")
                f = featurize(spec.input, spec)
                with torch.no_grad():
                    init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
                L = init["Q_L_init"].shape[0]
                is_motif = f["is_motif_atom_with_fixed_coord"]
            else:
                f = golden_f; init = golden_init; L = golden_L; is_motif = golden_is_motif
            coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
            spec_feat[spec_id] = (f, init, L, is_motif, coord0)
        f_used, init_used, L, is_motif, coord0 = spec_feat[spec_id]
        sp_t = spec.partial_t if spec.partial_t is not None else partial_t
        set_tune_matmul_for_atoms(L)
        effective_batch = min(
            batch_size,
            _BATCH_DESIGN_CEILING,
            max(1, _BATCH_ATOM_PAIR_BUDGET // max(1, L * L)),
            _BATCH_SPEED_CAP if L > _BATCH_SPEED_CAP_ABOVE_ATOMS else batch_size,
        )
        for start in range(0, len(spec_jobs), effective_batch):
            chunk = spec_jobs[start : start + effective_batch]
            generators = [
                torch.Generator().manual_seed(this_seed)
                for _, _, this_seed in chunk
            ]
            with torch.no_grad():
                X, traj = sampler.sample(
                    dev_dm,
                    len(chunk),
                    L,
                    coord0,
                    f_used,
                    init_used,
                    is_motif,
                    generator=generators,
                    partial_t=sp_t,
                    cfg_scale=cfg_scale,
                )
            for offset, (_, design_idx, _) in enumerate(chunk):
                out_path = _design_out_path(
                    out_dir, spec_id, design_idx, multi=multi_designs
                )
                _write_cif(X[offset], f_used, out_path,
                           pred_restype=traj[-1]["sequence_restype_I"][offset])
                results.append(
                    DesignResult(
                        spec_id=spec_id,
                        design_idx=design_idx,
                        out_path=out_path,
                        final_pcc_vs_ref=None,
                        n_atoms=int(X.shape[1]),
                    )
                )
                if verbose:
                    print(
                        f"[design:{spec_id}#{design_idx}] wrote {out_path} "
                        f"({X.shape[1]} atoms, batch={len(chunk)})"
                    )
    return results


def _design_out_path(out_dir, spec_id, design_idx, *, multi: bool) -> Path:
    # <spec_id>_<i>.cif when num_designs>1 (multi-design per spec), else back-compat <spec_id>.cif.
    # Path() wrap: the fanout pickles out_dir as str for the shard subprocess, so
    # _run_design_jobs receives a str here (run_design converts to Path for the
    # in-process path, but _run_design_shard passes the pickled str through).
    return Path(out_dir) / (f"{spec_id}_{design_idx}.cif" if multi else f"{spec_id}.cif")


def _run_design_fanout(jobs, specs, out_dir, *, checkpoint_dir, from_pdb, num_timesteps,
                       partial_t, cfg_scale, fp32_residual, batch_size,
                       multi_designs, devices, verbose,
                       host_threads=None) -> list[DesignResult]:
    """Data-parallel fan-out: one pinned subprocess per physical card, sharding
    the (spec_id, design_idx) jobs round-robin. Reuses the embed/predict
    subprocess-per-card pattern. Each child runs ``_run_design_shard``."""
    from .. import runtime
    from ..main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    devices = list(devices)[:max(1, len(jobs))]
    # Without this every shard's torch/OMP/BLAS pools claim all host cores, so N
    # co-resident shards oversubscribe the host N-fold and 4-card aggregate throughput
    # lands *below* one card. Measured on qb1 (16 physical cores, 4 cards, 3 designs
    # each): 633s uncapped vs 122s capped, i.e. 0.68x vs 3.51x of linear.
    thread_cap = runtime.host_thread_cap_env(len(devices), host_threads)
    workdir = tempfile.mkdtemp(prefix="tt-bio-design-fanout-")
    try:
        handles = []
        for idx, dev in enumerate(devices):
            shard = jobs[idx::len(devices)]
            if not shard:
                continue
            in_path = os.path.join(workdir, f"shard{idx}.in.pkl")
            out_path = os.path.join(workdir, f"shard{idx}.out.pkl")
            log_path = os.path.join(workdir, f"shard{idx}.log")
            with open(in_path, "wb") as fp:
                pickle.dump(dict(jobs=shard, specs=dict(specs), out_dir=str(out_dir),
                                 checkpoint_dir=str(checkpoint_dir), from_pdb=from_pdb,
                                 num_timesteps=num_timesteps, partial_t=partial_t,
                                 cfg_scale=cfg_scale, fp32_residual=fp32_residual,
                                 batch_size=batch_size,
                                 multi_designs=multi_designs), fp)
            env = {**os.environ, **thread_cap, "TT_VISIBLE_DEVICES": str(dev),
                   "TT_BIO_LEASE_HOLDER": f"worker:design-fanout-{dev}"}
            if dev in _detect_p300_devices() and not env.get("TT_MESH_GRAPH_DESC_PATH"):
                mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
                if mgd:
                    env["TT_MESH_GRAPH_DESC_PATH"] = mgd
            logf = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys; from tt_bio.rfd3.design import _run_design_shard; "
                 "_run_design_shard(sys.argv[1], sys.argv[2])",
                 in_path, out_path], env=env, stdout=logf, stderr=subprocess.STDOUT)
            handles.append((proc, out_path, dev, log_path, logf))
        results = []
        for proc, out_path, dev, log_path, logf in handles:
            proc.wait(); logf.close()
            if proc.returncode != 0:
                tail = Path(log_path).read_text(errors="replace").splitlines()[-25:]
                raise RuntimeError(f"design shard on device {dev} failed (exit {proc.returncode}):\n"
                                   + "\n".join(tail))
            with open(out_path, "rb") as fp:
                results.extend(pickle.load(fp))
        # Reassemble in (spec_id, design_idx) order
        order = {(sid, di): i for i, (sid, di, _) in enumerate(jobs)}
        results.sort(key=lambda r: order.get((r.spec_id, r.design_idx), 0))
        return results
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_design_shard(in_path: str, out_path: str) -> None:
    """Subprocess entry: load the pickled shard, run its jobs in-process on this
    card, pickle the DesignResult list back."""
    from .. import runtime

    runtime.bind_host_threads()
    with open(in_path, "rb") as fp:
        cfg = pickle.load(fp)
    res = _run_design_jobs(cfg["jobs"], cfg["specs"], cfg["out_dir"], checkpoint_dir=cfg["checkpoint_dir"],
                           from_pdb=cfg["from_pdb"], num_timesteps=cfg["num_timesteps"],
                           partial_t=cfg["partial_t"], cfg_scale=cfg["cfg_scale"],
                           fp32_residual=cfg["fp32_residual"],
                           batch_size=cfg["batch_size"],
                           multi_designs=cfg["multi_designs"], verbose=False)
    with open(out_path, "wb") as fp:
        pickle.dump(res, fp)
