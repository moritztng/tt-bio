"""End-to-end ESMFold2 parity: ttnn on-device pipeline vs the vendored torch reference.

Both paths share the *same* featurization (``ESMFold2InputBuilder.prepare_input``)
and the *same* language-model hidden states (computed once with the ttnn ESMC-6B
and passed to both ``forward``s via ``lm_hidden_states=``). That isolates the
ESMFold2 neural port under test -- inputs embedder, relpos, folding trunk (48 or
24 blocks), parcae recurrence, diffusion structure head, distogram + confidence
heads -- from the separately-validated ESMC-6B port (tests/test_esmc.py) and from
featurization. The torch reference is ``ESMFold2Model`` left unpatched; the test
path is the same model after ``patch_esmfold2`` (every learnable submodule -> ttnn).

Both samplers consume CPU ``torch`` random draws from the same globally-seeded
stream, so matching seeds provide a paired, shared-draw comparison. Each protein
is also folded at several seeds on both backends, giving three coordinate-metric
distributions -- reference-vs-reference (R), device-vs-device (D) and
device-vs-reference (X) -- summarized with the same statistical core
(`pharma_parity.summarize` / `noise_floor_verdict`) the rest of the parity
benchmark uses. Parity holds when X sits within max(R, D); the paired diagonal
shows whether matching random draws collapses the residual.

Reported per protein:
  * plddt_pcc / plddt_mae   -- per-residue confidence, the metric ESMFold ranks on
  * distogram_pcc, ptm      -- sampler-independent (computed once, first seed)
  * distogram_rel_l2        -- ||tt-ref||/||ref|| on distogram_logits, gated by
                              DISTOGRAM_REL_L2_MAX: PCC is scale-blind, so the
                              double-symmetrize bug (head received z+zT and
                              symmetrized again) scored distogram_pcc 0.9996.
                              rel_l2 was 1.09 with that bug, 0.05 without it.
  * kabsch_rmsd, coord_dm_pcc R/D/X distributions across the sampler seeds

Usage:
  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=1 \
    /home/ttuser/tt-bio-dev/env/bin/python scripts/esmfold2_e2e_parity.py \
      [--checkpoint esmfold2|esmfold2-fast] [--fast] [--proteins trpcage,gb1] \
      [--steps 20] [--loops 3] [--seeds 0,1,2] [--out /tmp/x.json] [--fixture_dir <dir>]

``--checkpoint`` names the tt-bio model id, the same vocabulary ``tt-bio predict --model``
uses, and the HF repo comes from the ``tt_bio.weights`` registry. It selects which released
checkpoint is folded: ``esmfold2`` (48-block trunk, MSA encoder) or ``esmfold2-fast``
(24-block, no MSA encoder). It is a different axis from ``--fast``, which is bf8 device
precision and leaves the checkpoint alone. Trunk depth is never hardcoded here: both the
reference and the ttnn port read it from the checkpoint config (``_spec()`` in
tt_bio/esmfold2_runtime.py), so one harness covers both depths.

With ``--fixture_dir`` (requires exactly one --proteins entry) the run also dumps a
committed reference-fixture tree (opendde schema): ``meta.json`` +
``ref_fp32/seed<N>.npz`` reference outputs + ``seed<N>/{meta.json,results.json,
structures/<target>.cif,device_seed<N>.npz}`` device outputs, and asserts the executed
sampler step count (requested 100 -> 68 after the sigma_max=256 schedule clip).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import torch

from pharma_parity import noise_floor_verdict, summarize

# Representative single-domain proteins (no MSA): short, medium, medium-long, and a
# pharma-realistic length. Hen egg-white lysozyme (L129) is the model antigen in antibody
# drug-discovery assays (HyHEL10-class complexes) and extends the leg past the L76 ubiquitin
# ceiling toward the 150-450 aa range pharma targets actually live in.
PROTEINS = {
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",                                              # 20
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",             # 56
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 76
    "lysozyme": ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDG"
                 "RTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),        # 129
}

# A --proteins entry that is not in PROTEINS is looked up here: the fold fixtures the perf
# page's own cells are measured on. That lets the published target go through THIS harness
# instead of a second one, which is the only way a page number and a parity number can be
# compared without an unstated protocol difference between them.
PERF_FIXTURES = "perf/size512/fixtures"


def protein_sequence(name: str) -> str:
    """Sequence for a harness target: a PROTEINS key, or a perf fold fixture by name."""
    if name in PROTEINS:
        return PROTEINS[name]
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        PERF_FIXTURES, f"{name}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"unknown protein {name!r}: not in {sorted(PROTEINS)} and no {path}")
    with open(path) as f:
        doc = yaml.safe_load(f)
    seqs = [e["protein"]["sequence"] for e in doc["sequences"] if "protein" in e]
    if len(seqs) != 1:
        raise SystemExit(f"{path}: expected one protein chain, got {len(seqs)}")
    return seqs[0]


# forward() kwargs that prepare_input supplies (extras are dropped by name).
_FORWARD_KEYS = {
    "token_index", "residue_index", "asym_id", "sym_id", "entity_id", "mol_type",
    "res_type", "token_bonds", "token_attention_mask", "ref_pos", "ref_element",
    "ref_charge", "ref_atom_name_chars", "ref_space_uid", "atom_attention_mask",
    "atom_to_token", "distogram_atom_idx", "deletion_mean", "msa", "has_deletion",
    "deletion_value", "msa_attention_mask", "input_ids",
}


def pcc(a, b) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rel_l2(a, b) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return ((a - b).norm() / b.norm()).item()


# Hard bound on the distogram rel_l2 above: the only legitimate gap is bf16 device
# noise (0.053 measured on the trpcage leg); a re-symmetrizing head lands at ~0.9.
# 0.25 sits far from both.
DISTOGRAM_REL_L2_MAX = 0.25


# tt-bio model id -> released checkpoint. The HF repo is read from the weights registry
# rather than written out again here, so `--checkpoint esmfold2-fast` and
# `tt-bio predict --model esmfold2-fast` cannot drift apart.
CHECKPOINTS = ("esmfold2", "esmfold2-fast")


def checkpoint_repo(checkpoint: str) -> str:
    from tt_bio.weights import ARTIFACTS
    return ARTIFACTS[checkpoint].repo


def checkpoint_identity(repo: str, config) -> dict:
    """What was actually loaded: repo, weight sha256, trunk depth, MSA encoder.

    The sha256 is the hub's own blob name for model.safetensors, read off the resolved cache
    entry, so it names the bytes this run folded instead of a literal that goes stale when the
    repo is re-uploaded. Depth and MSA-encoder presence come from the checkpoint config."""
    sha = None
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo, "model.safetensors")
        if isinstance(hit, str):
            sha = os.path.basename(os.path.realpath(hit))
    except Exception:
        pass
    blocks = config.folding_trunk.n_layers
    return {"repo": repo, "sha256": sha, "trunk_blocks": blocks,
            "msa_encoder": bool(getattr(config.msa_encoder, "enabled", True)),
            "version": f"{repo} sha256:{sha} ({blocks}-block release trunk)"}


def plddt_seed_block(ref_means, tt_means, seeds) -> dict:
    """Mean plDDT per seed on both backends, with the two numbers a cross-backend plDDT
    difference has to be read against: the reference's own seed-to-seed spread, and the
    same-seed backend difference. A single-seed plDDT gap between two backends is
    uninterpretable without the first, because the two samplers draw independent noise and
    part of any gap is which diffusion basin each landed in rather than arithmetic."""
    span = lambda v: max(v) - min(v)
    deltas = [abs(a - b) for a, b in zip(tt_means, ref_means)]
    return {"seeds": list(seeds), "ref": ref_means, "tt": tt_means,
            "ref_seed_spread": span(ref_means), "tt_seed_spread": span(tt_means),
            "same_seed_abs_delta": deltas, "mean_abs_delta": sum(deltas) / len(deltas)}


def dist_matrix(x):  # x: [n,3] -> [n,n]
    return torch.cdist(x.float(), x.float())


def build_features(seq, seed, device):
    from tt_bio._vendor.esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
    builder = ESMFold2InputBuilder()
    feats, chain_infos = builder.prepare_input(spi, seed=seed, device=device)
    return feats, chain_infos, builder


def run_forward(model, feats, lm_hs, *, loops, steps, samples, seed=0):
    fwd = {k: v for k, v in feats.items() if k in _FORWARD_KEYS}
    torch.manual_seed(seed)  # seeds the (torch) diffusion sampler's global RNG
    with torch.no_grad():
        return model(**fwd, lm_hidden_states=lm_hs, num_loops=loops,
                     num_sampling_steps=steps, num_diffusion_samples=samples)


def kabsch_rmsd(a_coords, b_coords, atom_mask):
    """RMSD (Angstrom) of a_coords onto b_coords after weighted rigid alignment.

    Reduced over REAL atoms only (atom_mask). sample_atom_coords carries padding
    atom slots the model emits at arbitrary, run-varying positions ~10 A out; they
    are not part of the structure. Averaging squared deviation over them swamps the
    real-atom RMSD and manufactures a spurious device-vs-reference gap (it inflates
    the cross-backend term more than the same-backend floor).
    """
    import tt_bio.esmfold2 as E
    a = a_coords.float(); b = b_coords.float()
    aligned = E._weighted_rigid_align(a.unsqueeze(0), b.unsqueeze(0), atom_mask, atom_mask)[0]
    m = atom_mask[0] > 0.5
    return (aligned[m] - b[m]).pow(2).sum(-1).mean().sqrt().item()


def pair_metrics(a, b, atom_mask):
    ac = a["sample_atom_coords"][0].float()
    bc = b["sample_atom_coords"][0].float()
    m = atom_mask[0] > 0.5  # score real atoms only (see kabsch_rmsd docstring)
    return (kabsch_rmsd(ac, bc, atom_mask),
            pcc(dist_matrix(ac[m]), dist_matrix(bc[m])))


def compare_multiseed(ref_runs: dict, tt_runs: dict, atom_mask, seeds):
    """R/D/X distributions plus the shared-draw same-seed diagonal."""
    r_rmsd, r_pcc = [], []
    for s1, s2 in itertools.combinations(seeds, 2):
        rmsd, p = pair_metrics(ref_runs[s1], ref_runs[s2], atom_mask)
        r_rmsd.append(rmsd); r_pcc.append(1 - p)
    d_rmsd, d_pcc = [], []
    for s1, s2 in itertools.combinations(seeds, 2):
        rmsd, p = pair_metrics(tt_runs[s1], tt_runs[s2], atom_mask)
        d_rmsd.append(rmsd); d_pcc.append(1 - p)
    x_rmsd, x_pcc = [], []
    for s1, s2 in itertools.product(seeds, seeds):
        rmsd, p = pair_metrics(tt_runs[s1], ref_runs[s2], atom_mask)
        x_rmsd.append(rmsd); x_pcc.append(1 - p)
    paired_rmsd, paired_pcc = [], []
    for seed in seeds:
        rmsd, p = pair_metrics(tt_runs[seed], ref_runs[seed], atom_mask)
        paired_rmsd.append(rmsd); paired_pcc.append(1 - p)

    def with_paired(verdict, paired):
        verdict["same_seed_cross"] = summarize(paired)
        verdict["same_seed_over_all_cross"] = (
            verdict["same_seed_cross"]["mean"] / verdict["cross"]["mean"]
        )
        return verdict

    return {
        "kabsch_rmsd": with_paired(
            noise_floor_verdict(x_rmsd, r_rmsd, d_rmsd, "kabsch_rmsd"), paired_rmsd),
        "coord_dm_1mpcc": with_paired(
            noise_floor_verdict(x_pcc, r_pcc, d_pcc, "1-coord_dm_pcc"), paired_pcc),
    }


def executed_step_count(ref_model, requested, cap=256.0):
    """Denoise steps the sampler actually runs for a requested count, computed with the
    vendored reference's own schedule code: karras(requested) clipped to sigma<=cap with the
    cap re-prepended (identical logic in the ttnn sampler, tt_bio/esmfold2.py). Requested 100
    -> 68 executed; requested 68 -> 46 (never pass 68 literally)."""
    import torch.nn.functional as F
    sampler = next(m for m in ref_model.modules() if hasattr(m, "inference_noise_schedule"))
    sched = sampler.inference_noise_schedule(requested)
    sched = sched[sched <= cap]
    sched = F.pad(sched, (1, 0), value=cap)
    return len(sched) - 1


def _seed_result(name, seq, run, atom_mask):
    ptm = float(run["ptm"].float().mean())
    iptm_t = run.get("iptm")
    iptm = float(iptm_t.float().mean()) if iptm_t is not None else 0.0
    plddt = float(run["plddt"].float().mean())
    return {"id": name, "status": "ok", "plddt": plddt, "complex_plddt": plddt,
            "ptm": ptm, "iptm": iptm, "confidence_score": 0.8 * iptm + 0.2 * ptm,
            "n_residues": len(seq), "n_atoms": int(atom_mask.sum()), "samples": 1}


def dump_fixture(fixture_dir, name, seq, feats, chain_infos, builder, ref_runs, tt_runs,
                 atom_mask, seeds, args, parity, n_steps_code, n_steps_device, ckpt):
    """Dump the opendde-schema fixture tree: meta.json + ref_fp32/ + seed<N>/ device legs."""
    from pathlib import Path
    import numpy as np
    root = Path(fixture_dir)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"fixture dir {root} exists and is non-empty; refusing to overwrite")
    (root / "ref_fp32").mkdir(parents=True)

    def _npz(run):
        return dict(
            sample_atom_coords=run["sample_atom_coords"].float().cpu().numpy(),
            plddt=run["plddt"].float().cpu().numpy(),
            distogram_logits=run["distogram_logits"].float().cpu().numpy(),
            ptm=run["ptm"].float().cpu().numpy(),
            atom_mask=atom_mask.float().cpu().numpy(),
        )

    for s in seeds:
        np.savez(root / "ref_fp32" / f"seed{s}.npz", **_npz(ref_runs[s]))
    (root / "ref_fp32" / "results.json").write_text(json.dumps(
        [_seed_result(name, seq, ref_runs[s], atom_mask) for s in seeds], indent=2))
    (root / "ref_fp32" / "meta.json").write_text(json.dumps({
        "reference_impl": "tt-bio vendored torch reference (Biohub/transformers fork f9a5a37, CPU fp32)",
        "dtype": "fp32", "seeds": seeds,
        "note": "shared ESMC hidden states: computed once with the ttnn ESMC-6B and injected "
                "into both forwards (lm_hidden_states=), so this leg isolates the ESMFold2 port",
    }, indent=2))

    for s in seeds:
        d = root / f"seed{s}"
        (d / "structures").mkdir(parents=True)
        np.savez(d / f"device_seed{s}.npz", **_npz(tt_runs[s]))
        res = builder.decode(tt_runs[s], feats, chain_infos,
                             num_diffusion_samples=1, complex_id=name)
        if isinstance(res, list):
            res = res[0]
        (d / "structures" / f"{name}.cif").write_text(res.complex.to_mmcif())
        (d / "results.json").write_text(json.dumps(
            [_seed_result(name, seq, tt_runs[s], atom_mask)], indent=2))
        (d / "meta.json").write_text(json.dumps({
            "seed": s, "backend": "ttnn (Tenstorrent Blackhole), production sampler path",
            "shared_rng": "TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG=1 — device sampler draws "
                          "from the global CPU torch stream, seeded per seed before each "
                          "forward on both paths (paired shared-draw comparison)",
        }, indent=2))

    cmd = ("TT_VISIBLE_DEVICES=1 TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG=1 PYTHONPATH=$PWD "
           f"python3 scripts/esmfold2_e2e_parity.py --checkpoint {args.checkpoint} "
           f"--proteins {name} --loops {args.loops} "
           f"--steps {args.steps} --seeds {','.join(map(str, seeds))} "
           f"--fixture_dir {fixture_dir} --out <summary.json>")
    meta = {
        "command": cmd,
        "date": "2026-07-29",
        "model": args.checkpoint,
        "msa": "none (single-sequence; ESMFold2 is single-sequence by design, its MSA is optional)",
        "envelope": {
            "reference_commit": "Biohub/transformers fork f9a5a37 (vendored tt_bio/_vendor/esmfold2_hf)",
            "reference_impl": "tt-bio vendored torch reference (CPU fp32)",
            "reference_version": f"{ckpt['version']}, dtype fp32",
            "seeds": seeds,
            "settings": {
                "device_args": ["--single_sequence", "--recycling_steps", str(args.loops),
                                "--sampling_steps", str(args.steps), "--diffusion_samples", "1"],
                "model": args.checkpoint, "msa": "none", "target_id": name,
            },
        },
        "reference_commit": "Biohub/transformers fork f9a5a37 (vendored tt_bio/_vendor/esmfold2_hf)",
        "reference_impl": "tt-bio vendored torch reference (Biohub/transformers fork f9a5a37, CPU fp32)",
        "reference_version": ckpt["version"],
        "seeds": seeds,
        "settings": {
            "recycling_cycles": args.loops,
            "requested_steps": args.steps,
            "executed_steps": n_steps_code,
            "executed_steps_device_empirical": n_steps_device,
            "diffusion_samples": 1,
            "seeds": seeds,
            "single_sequence": True,
            "dtype": "fp32",
            "checkpoint": ckpt["version"],
            "target": f"{name} ({len(seq)} res)",
            "shared_rng": "TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG=1 (global CPU torch stream seeded per seed on both paths)",
            "lm_hidden_states": "shared: one ttnn ESMC-6B (biohub/ESMC-6B rev 45b0fa5d) forward injected into both paths",
            "rationale": "sample=1 isolates convergence (loops/steps) from best-of-N selection",
        },
        "parity": parity,
        "invalidation_rule": "Regenerate this fixture ONLY when the pinned reference "
                             "commit/checkpoint or the settings above change. For any other "
                             "change (device seeds, device code, release tag) the fixture is "
                             "reused as-is and only the device side re-runs.",
        "provenance": "Built 2026-07-29 on qb1 (Blackhole p150 card 1) at the ESMFold2 paper "
                      "benchmark protocol (10 loops, 100 requested = 68 executed steps, 1 "
                      "sample, seeds 0-2, single-sequence) — the D4a reference leg esmfold2 "
                      "never got. Parity numbers in the parity field are the output of this "
                      "harness's noise-floor verdict (X within max(R,D)).",
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote fixture tree to {root}", flush=True)


def conf_trunk_ab(tt_model, ref_model, feats, lm_hs, *, loops, steps, seed,
                  trunk_probe=False, z_sens=False, x_swap_pdb=None):
    """Paired A/B on the confidence head's 4-block pair trunk, device path only.

    Runs one device fold, capturing the confidence head's inputs (s_inputs, z, x_pred
    and the index tensors), then re-scores those SAME inputs twice: once with the ttnn
    pair trunk the port installs, once with the reference fp32 pair trunk from
    `ref_model`. Every other part of the head is already host fp32 on both legs, so the
    difference between the two mean plDDTs is exactly what the device pair trunk
    contributes, with zero fold-level sampler noise between the legs.

    No reference fold is needed, which is what makes this affordable at L512: a CPU
    fp32 reference fold of cdk2x2_512 takes ~47 min per seed.
    """
    head = tt_model.confidence_head
    dev_trunk, inner = head.folding_trunk, head.forward
    cap = {}

    def _capture(*a, **k):
        cap["a"], cap["k"] = a, k
        return inner(*a, **k)

    head.forward = _capture
    try:
        run = run_forward(tt_model, feats, lm_hs, loops=loops, steps=steps,
                          samples=1, seed=seed)
    finally:
        head.forward = inner
    if "k" not in cap:
        raise SystemExit("confidence head was never called: nothing to A/B")

    def _score(out):
        return {"plddt": float(out["plddt"].float().mean()),
                "ptm": float(out["ptm"].float().mean())}

    legs = {"fold": _score(run)}
    for tag, trunk in (("device_trunk", dev_trunk),
                       ("host_fp32_trunk", ref_model.confidence_head.folding_trunk)):
        head.folding_trunk = trunk
        try:
            legs[tag] = _score(inner(*cap["a"], **cap["k"]))
        finally:
            head.folding_trunk = dev_trunk
    legs["delta_plddt"] = legs["host_fp32_trunk"]["plddt"] - legs["device_trunk"]["plddt"]
    legs["delta_ptm"] = legs["host_fp32_trunk"]["ptm"] - legs["device_trunk"]["ptm"]
    z = cap["k"]["z"].float()
    legs["z_in"] = _pair_stats(z)
    if trunk_probe:
        legs["trunk_probe"] = trunk_bf16_probe(tt_model, ref_model, z)
    if z_sens:
        legs["z_sensitivity"] = z_plddt_sensitivity(head, inner, cap, dev_trunk)
        legs["x_pred_sensitivity"] = x_plddt_sensitivity(inner, cap)
    if x_swap_pdb:
        legs["x_swap"] = x_coord_swap(inner, cap,
                                      [q for q in x_swap_pdb.split(",") if q])
    return legs


def read_pdb_ca(path):
    """Representative-atom (CA) coordinates in residue order, as [1, n_res, 3]."""
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in open(path)
           if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)


def x_coord_swap(inner, cap, pdbs) -> dict:
    """Re-score the confidence head with another backend's coordinates substituted in.

    `x_pred` reaches the head only through `gather_rep_atom_coords` -> `cdist` -> 1.25 A
    distance bins, so writing foreign coordinates into the representative-atom slots swaps the
    entire coordinate channel and touches nothing else. `cdist` is invariant to rigid motion,
    so no superposition is needed and none is done: the substituted structure is compared to
    itself, exactly as the head would compare a native one.

    This is the experiment that replaces the extrapolation from a random-displacement slope. A
    reference fold would cost ~47 min per seed on CPU; a published reference structure costs
    nothing.
    """
    x0 = cap["k"]["x_pred"]
    rep = cap["k"]["distogram_atom_idx"].long()
    out = {"baseline": {"plddt": float(inner(*cap["a"], **cap["k"])["plddt"].float().mean())}}
    for path in pdbs:
        ca = read_pdb_ca(path)
        if ca.shape[1] != rep.shape[1]:
            raise SystemExit(f"{path}: {ca.shape[1]} CA atoms but {rep.shape[1]} tokens")
        xs = x0.float().clone()
        flat = xs.reshape(-1, xs.shape[-2], 3) if xs.ndim == 4 else xs
        flat[0].index_copy_(0, rep[0], ca[0].to(flat.dtype))
        res = inner(*cap["a"], **dict(cap["k"], x_pred=xs.to(x0.dtype)))
        out[path.split("/")[-1]] = {"plddt": float(res["plddt"].float().mean()),
                                    "ptm": float(res["ptm"].float().mean())}
    return out


def x_plddt_sensitivity(inner, cap, rms_a=(0.25, 0.5, 1.0, 2.0)) -> dict:
    """d(mean plDDT) / d(coordinate displacement), pair state and s_inputs held fixed.

    `x_pred` reaches the confidence head only through 1.25 A-wide binned representative-atom
    distances, so this reads how much confidence a given amount of structural difference costs.
    Displacements are isotropic Gaussian per atom, quoted as RMS Angstrom, which is the same
    unit the harness's Kabsch numbers are in.
    """
    x0 = cap["k"]["x_pred"]
    out = {}
    for r in rms_a:
        g = torch.Generator().manual_seed(0)
        d = torch.randn(x0.shape, generator=g, dtype=torch.float32) * (r / 3.0 ** 0.5)
        k = dict(cap["k"], x_pred=(x0.float() + d).to(x0.dtype))
        res = inner(*cap["a"], **k)
        out[f"rms_{r}A"] = {"plddt": float(res["plddt"].float().mean()),
                            "ptm": float(res["ptm"].float().mean())}
    return out


def z_plddt_sensitivity(head, inner, cap, dev_trunk, rels=(0.01, 0.03, 0.125)) -> dict:
    """d(mean plDDT) / d(relative error in the trunk pair state), same head, same x_pred.

    Perturbs the captured `z` multiplicatively (`z * (1 + r*u)`, `u ~ N(0,1)`), which is the
    shape bf16 arithmetic error actually has: relative, not additive. Everything else the head
    consumes is held fixed, so this reads the z -> plDDT channel on its own and answers whether
    a checkpoint's confidence readout amplifies pair-state error or absorbs it.

    The perturbation is a random direction, not the port's own error direction, so read the
    slope as a magnitude, not a prediction of the sign of any single run.
    """
    z0 = cap["k"]["z"]
    out = {}
    head.folding_trunk = dev_trunk
    try:
        for r in rels:
            g = torch.Generator().manual_seed(0)
            u = torch.randn(z0.shape, generator=g, dtype=torch.float32)
            zp = (z0.float() * (1.0 + r * u)).to(z0.dtype)
            k = dict(cap["k"], z=zp)
            res = inner(*cap["a"], **k)
            out[f"rel_{r}"] = {"z_rel_l2": rel_l2(zp, z0),
                               "plddt": float(res["plddt"].float().mean()),
                               "ptm": float(res["ptm"].float().mean())}
    finally:
        head.folding_trunk = dev_trunk
    return out


def _pair_stats(z) -> dict:
    """Magnitude profile of a pair tensor, and what bf16 storage costs at that scale.

    `ulp_at_absmax` is the bf16 spacing at the largest entry: a residual update smaller
    than half of it is lost outright when the pair state is stored in bf16.
    """
    import math
    a = z.float().abs().flatten()
    absmax = float(a.max())
    # torch.quantile caps at ~16M elements; a [1,512,512,256] pair is 67M, so read the
    # quantiles off a deterministic stride-4 subsample (absmax below is exact).
    sub = a[::4] if a.numel() > 16_000_000 else a
    q = [float(v) for v in sub.quantile(torch.tensor([0.5, 0.99, 0.9999]))]
    ulp = 2.0 ** (math.floor(math.log2(absmax)) - 7) if absmax > 0 else 0.0
    return {"rms": float(z.float().pow(2).mean().sqrt()), "absmax": absmax,
            "p50": q[0], "p99": q[1], "p9999": q[2],
            "frac_above_256": float((a > 256).float().mean()),
            "bf16_ulp_at_absmax": ulp}


def trunk_bf16_probe(tt_model, ref_model, z) -> dict:
    """One trunk pass over the same pair tensor, three ways, to price bf16 pair storage.

    * `device`: the ttnn trunk, whose pair state lives in bf16 between blocks.
    * `host_fp32`: the reference trunk, fp32 throughout (the parity target).
    * `host_bf16_state`: the reference trunk in fp32 arithmetic, but with the pair state
      rounded to bf16 after every block, i.e. the device's storage precision and nothing
      else.

    If `host_bf16_state` reproduces `device`'s error against `host_fp32`, the port's pair
    divergence is bf16 pair storage rather than any op's arithmetic.
    """
    ftw = tt_model.folding_trunk.m          # _Adapter.m -> E.FoldingTrunk (torch in/out)
    ref_trunk = ref_model.folding_trunk     # torch fp32 FoldingTrunk, same weights
    z = z.float()

    out_dev = ftw(z).float()
    out_ref = ref_trunk(z.clone()).float()

    cur = z.clone()
    for block in ref_trunk.blocks:
        cur = block(cur).float().to(torch.bfloat16).float()
    out_bf16 = cur

    return {"n_blocks": len(ref_trunk.blocks),
            "device_vs_host_fp32_rel_l2": rel_l2(out_dev, out_ref),
            "host_bf16_state_vs_host_fp32_rel_l2": rel_l2(out_bf16, out_ref),
            "device_vs_host_bf16_state_rel_l2": rel_l2(out_dev, out_bf16),
            "out_host_fp32": _pair_stats(out_ref),
            "out_device": _pair_stats(out_dev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--proteins", default="trpcage,gb1,ubiquitin",
                    help="comma-separated targets: PROTEINS keys, or a perf fold fixture "
                         "name such as cdk2x2_512")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated sampler seeds, run on both backends")
    ap.add_argument("--feature_seed", type=int, default=7, help="seed for featurization (not the sampler)")
    ap.add_argument("--checkpoint", default="esmfold2", choices=CHECKPOINTS,
                    help="which released checkpoint to fold: esmfold2 (48-block trunk) or "
                         "esmfold2-fast (24-block, no MSA encoder). Resolved to an HF repo "
                         "through tt_bio.weights. Not to be confused with --fast, which is "
                         "device precision and leaves the checkpoint alone.")
    ap.add_argument("--esmfold2_repo", default=None,
                    help="fold this HF repo instead of the one --checkpoint resolves to "
                         "(unreleased weights); provenance in the output still records it")
    ap.add_argument("--esmc_repo", default="biohub/ESMC-6B")
    ap.add_argument("--out", default="/tmp/ef2_parity/summary.json")
    ap.add_argument("--x_swap_pdb", default=None,
                    help="with --conf_ab, comma-separated PDBs whose CA coordinates replace the "
                         "device fold's in the confidence head, swapping the whole coordinate "
                         "channel against a published reference structure")
    ap.add_argument("--z_sens", action="store_true",
                    help="with --conf_ab, measure how far mean plDDT moves per unit of "
                         "relative error in the trunk pair state, holding x_pred fixed")
    ap.add_argument("--trunk_probe", action="store_true",
                    help="with --conf_ab, also price bf16 pair storage: one trunk pass over "
                         "the captured pair tensor on device, on host fp32, and on host fp32 "
                         "with the pair state rounded to bf16 between blocks")
    ap.add_argument("--conf_ab", action="store_true",
                    help="device-only paired A/B of the confidence head's pair trunk "
                         "(ttnn vs reference fp32) on identical captured inputs; skips "
                         "the reference fold, so no parity verdict is produced")
    ap.add_argument("--fixture_dir", default=None,
                    help="dump a committed fixture tree here (requires exactly one protein)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio import tenstorrent
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2_common import compute_lm_hidden_states
    from tt_bio.esmfold2_runtime import _ESMCAdapter, patch_esmfold2

    names = [n.strip() for n in args.proteins.split(",") if n.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.fixture_dir and len(names) != 1:
        raise SystemExit("--fixture_dir requires exactly one --proteins entry")

    # Empirical device-side executed-step counter: capture the sampler's own n_steps via
    # its progress hook on the first device forward.
    import tt_bio.esmfold2 as E
    _seen, _orig_report = {}, E.report_progress

    def _report(stage, step=0, total=0):
        if stage == "diffusion" and total and "n_steps" not in _seen:
            _seen["n_steps"] = total
        return _orig_report(stage, step, total)
    E.report_progress = _report

    # Fast mode must be set BEFORE the ESMC preload: otherwise the 6B LM loads in
    # default (non-fast) bf16 (~12.8 GB) and the later fast-mode trunk construction
    # OOMs on Wormhole. Production sets fast mode before any load; mirror it.
    tenstorrent.set_fast_mode(args.fast)
    # Shared ttnn ESMC-6B (loaded once): produces the LM hidden states fed to BOTH paths.
    esmc = _ESMCAdapter(args.esmc_repo, persistent=True)
    esmc.preload()

    # Torch reference model (unpatched). Real ESMFold2 weights, no CPU ESMC (we
    # inject shared LM states instead).
    repo = args.esmfold2_repo or checkpoint_repo(args.checkpoint)
    print(f"loading torch reference model ({args.checkpoint} -> {repo}) ...", flush=True)
    ref_model = ESMFold2Model.from_pretrained(repo, load_esmc=False).eval()
    ckpt = checkpoint_identity(repo, ref_model.config)
    print(f"checkpoint: {ckpt['version']}, msa_encoder={ckpt['msa_encoder']}", flush=True)

    n_steps_code = executed_step_count(ref_model, args.steps)
    print(f"requested steps={args.steps} -> executed={n_steps_code} (sigma_max=256 clip)",
          flush=True)

    # ttnn model: same weights, every submodule swapped to ttnn.
    print(f"loading ttnn model (fast={args.fast}) ...", flush=True)
    tt_model = ESMFold2Model.from_pretrained(repo, load_esmc=False).eval()
    patch_esmfold2(tt_model, esmc_repo=args.esmc_repo)
    tt_model._esmc = esmc  # reuse the already-loaded ESMC (LM states are passed in anyway)

    results = []
    for name in names:
        seq = protein_sequence(name)
        print(f"\n=== {name} (L={len(seq)}), seeds={seeds} ===", flush=True)
        feats, chain_infos, builder = build_features(seq, args.feature_seed, ref_model.device)
        lm_hs = compute_lm_hidden_states(
            esmc, feats["input_ids"], feats["asym_id"], feats["residue_index"],
            feats["mol_type"], feats["token_attention_mask"])
        atom_mask = feats["atom_attention_mask"].float()
        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0)

        if args.conf_ab:
            ab = conf_trunk_ab(tt_model, ref_model, feats, lm_hs, loops=args.loops,
                               steps=args.steps, seed=seeds[0],
                               trunk_probe=args.trunk_probe, z_sens=args.z_sens,
                               x_swap_pdb=args.x_swap_pdb)
            ab = dict(protein=name, L=len(seq), seed=seeds[0], checkpoint=args.checkpoint,
                      trunk_blocks=ckpt["trunk_blocks"], **ab)
            results.append(ab)
            print(json.dumps(ab, indent=2), flush=True)
            continue

        ref_runs, tt_runs = {}, {}
        for s in seeds:
            print(f"  ref seed={s} ...", flush=True)
            ref_runs[s] = run_forward(ref_model, feats, lm_hs, loops=args.loops, steps=args.steps, samples=1, seed=s)
            print(f"  device seed={s} ...", flush=True)
            tt_runs[s] = run_forward(tt_model, feats, lm_hs, loops=args.loops, steps=args.steps, samples=1, seed=s)

        base_ref, base_tt = ref_runs[seeds[0]], tt_runs[seeds[0]]
        verdicts = compare_multiseed(ref_runs, tt_runs, atom_mask, seeds)
        dg_rel = rel_l2(base_tt["distogram_logits"], base_ref["distogram_logits"])
        plddt_means = plddt_seed_block(
            [float(ref_runs[s]["plddt"].float().mean()) for s in seeds],
            [float(tt_runs[s]["plddt"].float().mean()) for s in seeds], seeds)
        m = dict(
            protein=name, L=len(seq), n_seeds=len(seeds),
            checkpoint=args.checkpoint, trunk_blocks=ckpt["trunk_blocks"],
            plddt_mean=plddt_means,
            plddt_pcc=pcc(base_tt["plddt"], base_ref["plddt"]),
            plddt_mae=(base_tt["plddt"].float() - base_ref["plddt"].float()).abs().mean().item(),
            distogram_pcc=pcc(base_tt["distogram_logits"], base_ref["distogram_logits"]),
            distogram_rel_l2=dg_rel,
            ptm_tt=float(base_tt["ptm"].mean()), ptm_ref=float(base_ref["ptm"].mean()),
            **verdicts,
        )
        results.append(m)
        print(json.dumps(m, indent=2), flush=True)
        assert dg_rel < DISTOGRAM_REL_L2_MAX, (
            f"{name}: distogram_rel_l2 {dg_rel:.4f} >= {DISTOGRAM_REL_L2_MAX}: "
            "distogram_logits is scale-wrong, not just noisy (a PCC-only anchor "
            "shipped the double-symmetrize bug at distogram_pcc 0.9996)")

        if args.fixture_dir:
            n_steps_device = _seen.get("n_steps")
            assert n_steps_device == n_steps_code, (
                f"device executed {n_steps_device} steps, schedule says {n_steps_code}")
            if args.steps == 100:
                assert n_steps_code == 68, f"requested 100 must execute 68, got {n_steps_code}"
            dump_fixture(args.fixture_dir, name, seq, feats, chain_infos, builder,
                         ref_runs, tt_runs, atom_mask, seeds, args, m,
                         n_steps_code, n_steps_device, ckpt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
