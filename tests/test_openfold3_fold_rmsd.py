"""OpenFold3 end-to-end ``fold()`` -> vs-ground-truth Kabsch Cα-RMSD merge gate
(docs/openfold3-port.md "Concretely what remains" item 4 -- the actual merge gate
for the whole OF3 port).

Runs the real device ``OpenFold3.fold()`` on the ubiquitin fixture (the target wired
into every OF3 component golden), Kabsch-aligns the predicted Cα positions vs the
experimental structure (1UBQ), and asserts the result is a real structure (finite,
protein-scale, sub-garbage RMSD) -- NOT a tight accuracy gate. The OF3 155k checkpoint
is undertrained and ubiquitin is folded single-sequence (no MSA), so the honest number
is mediocre; this test guards against wiring regressions (NaN / non-converged
rollout / coordinate-frame bug), not against the model's intrinsic accuracy.

Full production rollout (200 steps x 1 sample; ~10 s on one BH card). The 4-step
reduced rollout deliberately does NOT pass this gate -- it is a sampler-math PCC gate
(``test_openfold3_sample_diffusion``), not a structure: noise_schedule[0]=2560 needs
the full 200-step EDM denoise to converge to a protein-scale structure.

Golden / features: ~/of3_ref_out.pkl (real ubiquitin features from the P1 data
pipeline + reference embedder outputs, reused to avoid redundant data-pipeline work).
Ground truth: examples/ground_truth_structures/ubiquitin.pdb (1UBQ). Cα atom indices:
tests/fixtures/of3_ubiquitin_ca_mask.npy.
"""
import os, pickle, pytest, torch

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GT_PDB = os.path.join(_REPO, "examples/ground_truth_structures/ubiquitin.pdb")
_CA_MASK = os.path.join(_REPO, "tests/fixtures/of3_ubiquitin_ca_mask.npy")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CKPT) and os.path.exists(_GOLD) and os.path.exists(_GT_PDB)
         and os.path.exists(_CA_MASK)),
    reason="of3 ckpt / golden / 1UBQ ground truth / Cα mask missing")


def test_of3_fold_kabsch_rmsd():
    """Real end-to-end fold(): device trunk -> 200-step device SampleDiffusion ->
    Cα -> Kabsch vs 1UBQ. Asserts a real structure (finite, protein-scale,
    sub-garbage RMSD), not a tight accuracy floor."""
    import numpy as np
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_fold import OpenFold3, kabsch_rmsd, load_pdb_ca

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(_GOLD, "rb"))["intermediates"]
    tr = I["trunk_real"]
    te = I["template_embedder_real"]["feat"]
    me = I["msa_module_embedder_real"]["msa_feat"]
    cond = I["diffusion_conditioning_real"]
    xl = I["diffusion_module_xlout_real"]
    dec = I["diffusion_decoder_real"]
    at = I["input_embedder_atom_transformer_real"]
    n_atom, n_token, nb, NP = xl["n_atom"], xl["n_token"], xl["nb"], xl["NP"]
    dm_aux = dict(
        cl0=xl["cl0"], plm0=xl["plm0"], atom_mask=dec["atom_mask"],
        atom_to_token_index=dec["atom_to_token_index"],
        npe_q_indices=xl["npe_q_indices"], npe_k_indices=xl["npe_k_indices"],
        zij_mask=xl["zij_mask"], key_block_idxs=dec["key_block_idxs"],
        invalid_mask=dec["invalid_mask"], mask_trunked=dec["mask_trunked"],
        atom_to_token_mean=at["atom_to_token_mean"], nb=nb, NP=NP)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    model = OpenFold3(sd, ckc, num_cycles=tr["num_cycles"])

    (xl_final,) = model.fold(
        s_init=tr["s_init"], z_init=tr["z_init"], template_feat=te, msa_feat=me,
        s_input=tr["s_input"], si_input=cond["si_input"], relpos=cond["relpos"],
        token_mask=cond["token_mask"], dm_aux_host=dm_aux,
        n_atom=n_atom, n_token=n_token, no_rollout_steps=200, seed=1234, no_samples=1)

    assert torch.isfinite(xl_final).all(), "fold() produced non-finite coordinates"
    std = float(xl_final.std())
    # Protein-scale structure (Å): the 200-step rollout denoises noise_schedule[0]=2560
    # down to a structure. A non-converged rollout (e.g. 4 steps) sits at std~300.
    assert 1.0 < std < 50.0, f"fold() xl_final std {std:.2f} outside protein-scale (1,50) Å"

    ca_mask = np.load(_CA_MASK).astype(bool)
    assert len(ca_mask) == n_atom
    gt_ca = load_pdb_ca(_GT_PDB)
    pred_ca = xl_final[ca_mask].double()
    assert pred_ca.shape[0] == gt_ca.shape[0], \
        f"Cα count mismatch pred={pred_ca.shape[0]} gt={gt_ca.shape[0]}"
    rmsd = kabsch_rmsd(pred_ca, gt_ca)
    print(f"\nOF3 fold() end-to-end: 200-step rollout, 1 sample, seed 1234, "
          f"target ubiquitin(1UBQ): xl_final std={std:.3f} Å, "
          f"Kabsch Cα-RMSD = {rmsd:.3f} Å")
    # Sanity ceiling (well above the real ~11.5 Å): catches garbage/NaN/wiring
    # regressions without gating on the undertrained model's intrinsic accuracy.
    assert rmsd < 30.0, f"fold() Cα-RMSD {rmsd:.2f} Å above sanity ceiling 30 Å"
