"""P13/S1: reference OF3 trunk under the SEARCHED 1UBQ MSA (decisive experiment).

Every prior trunk PCC gate (P10: s_trunk 0.99910 / z_trunk 0.97097) was captured in the
single-sequence regime. The 9.45 A 1UBQ result comes from the searched 2734-row ColabFold
MSA, a regime no gate covers. This script runs the CPU reference trunk
(aqlaboratory/openfold-3 modules, /tmp/of3-ref shadowing the installed package) on the
EXACT inputs scripts/of3_fold_rmsd.py fed the device (asserted: raw_rows == 2734), twice:

  pass "fixed":    m computed ONCE from tt-bio's exact device draw
                   (make_openfold3_msa_features seed=0 -> 1024 rows), reused all 4 cycles
                   -- mirrors the device trunk's constant-m behaviour.
  pass "upstream": reference MSAModuleEmbedder re-called per cycle (fresh subsample draw
                   per cycle, upstream run_trunk behaviour, torch.manual_seed(0) sequence).

PCC(fixed, upstream) sizes the subsample-draw sensitivity of z_trunk/s_trunk -- the noise
floor the device PCC must be judged against (memory pcc-gate-bar-snr-limited-not-flat).

Saves ~/p13_s1_trunk_msa.pt with every tensor the device leg needs. Run with the CPU
reference venv:  /tmp/of3-venv/bin/python scripts/of3_trunk_msa_golden.py
"""
import os
import sys
import pickle

import numpy as np
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get(
    "TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
OUT = os.path.expanduser("~/p13_s1_trunk_msa.pt")
QUERY = os.path.join(REPO_ROOT, "tests/fixtures/of3_ubiquitin_query.json")
MSA_DIR = os.path.join(REPO_ROOT, ".artifacts/msa")
NUM_CYCLES = 4


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def to_batch(feat):
    return {k: v.unsqueeze(0) for k, v in feat.items() if torch.is_tensor(v)}


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom,
        MSAModuleEmbedder,
    )
    from openfold3.core.model.latent.pairformer import PairFormerStack
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    from openfold3.core.model.primitives import LayerNorm, Linear

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import (
        build_openfold3_features,
        make_openfold3_msa_features,
    )

    # Reproduce of3_fold_rmsd.py's exact RNG/featurization order so the searched-MSA
    # features are bit-identical to what the device fold consumed. The MSA cache is
    # warm, so resolve_openfold3_msas would only set main_msa_file_paths (no RNG);
    # do that directly and keep this venv free of tt_bio.main's rdkit/numba deps.
    import hashlib
    from pathlib import Path

    torch.manual_seed(0)
    np.random.seed(0)
    query = next(iter(InferenceQuerySet.from_json(QUERY).queries.values()))
    for chain in query.chains:
        if chain.molecule_type.name != "PROTEIN" or chain.main_msa_file_paths:
            continue
        seq_hash = hashlib.sha256(chain.sequence.encode()).hexdigest()[:16]
        chain.main_msa_file_paths = [
            Path(MSA_DIR) / "of3" / seq_hash / "colabfold_main.a3m"
        ]
    features = build_openfold3_features(query)
    raw_rows = int(features["msa"].shape[0])
    assert raw_rows == 2734, f"featurization drift vs the S0 fold: raw_rows={raw_rows}"
    print(f"features: raw_rows={raw_rows} n_tokens={int(features['token_mask'].shape[0])}")

    msa_feat_tt = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
    valid = torch.nonzero(features["msa_mask"].sum(dim=-1) > 0).flatten()
    gen = torch.Generator().manual_seed(0)
    sel = valid[torch.randperm(valid.numel(), generator=gen)[:1024]]
    msa_mask_sel = features["msa_mask"].index_select(0, sel)
    print(f"msa_feat: {tuple(msa_feat_tt.shape)} valid_rows={valid.numel()}")

    batch = to_batch(features)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    ie = InputEmbedderAllAtom(**C.architecture.input_embedder).eval()
    ie.load_state_dict(sub(sd, "input_embedder"), strict=True)
    torch.manual_seed(0)
    with torch.no_grad():
        s_input_ref, s_init, z_init = ie(batch=batch)
    print(f"input_embedder: s_input {tuple(s_input_ref.shape)} "
          f"s_init {tuple(s_init.shape)} z_init {tuple(z_init.shape)}")

    # Fixture cross-checks: (1) fixture ai vs real atom encoder, (2) the s_input the
    # device fold actually consumed vs the reference-computed one.
    I = pickle.load(open(GOLD, "rb"))["intermediates"]
    ai_fixture = I["input_embedder_atom_enc_real"]["out"][0].float()
    with torch.no_grad():
        ai_real, _, _, _ = ie.atom_attn_enc(batch=batch)
    ai_real = ai_real[0].float()
    print(f"fixture ai vs real atom encoder: pcc={pcc(ai_fixture, ai_real):.6f} "
          f"max_abs={float((ai_fixture - ai_real).abs().max()):.6f}")
    s_input_tt = torch.cat(
        [ai_fixture, features["restype"], features["profile"],
         features["deletion_mean"].unsqueeze(-1)], dim=-1)
    print(f"s_input fold-used vs reference: pcc={pcc(s_input_tt, s_input_ref[0]):.6f} "
          f"max_abs={float((s_input_tt - s_input_ref[0]).abs().max()):.6f}")

    me = MSAModuleEmbedder(**C.architecture.msa.msa_module_embedder).eval()
    me.load_state_dict(sub(sd, "msa_module_embedder"), strict=True)
    with torch.no_grad():
        m_fixed = me.linear_m(msa_feat_tt.unsqueeze(0).to(s_input_ref.dtype))
        m_fixed = m_fixed + me.linear_s_input(s_input_ref).unsqueeze(-3)
    print(f"m_fixed: {tuple(m_fixed.shape)} std={float(m_fixed.std()):.4f}")

    ln_z = LayerNorm(128).eval()
    ln_z.load_state_dict({k: sd["layer_norm_z." + k] for k in ("weight", "bias")})
    lin_z = Linear(128, 128, bias=False).eval()
    lin_z.load_state_dict({"weight": sd["linear_z.weight"]})
    ln_s = LayerNorm(384).eval()
    ln_s.load_state_dict({k: sd["layer_norm_s." + k] for k in ("weight", "bias")})
    lin_s = Linear(384, 384, bias=False).eval()
    lin_s.load_state_dict({"weight": sd["linear_s.weight"]})
    te = TemplateEmbedderAllAtom(config=C.architecture.template).eval()
    te.load_state_dict(sub(sd, "template_embedder"), strict=True)
    msa_stack = MSAModuleStack(**dict(C.architecture.msa.msa_module)).eval()
    msa_stack.load_state_dict(sub(sd, "msa_module"), strict=True)
    pairformer = PairFormerStack(**dict(C.architecture.pairformer)).eval()
    pairformer.load_state_dict(sub(sd, "pairformer_stack"), strict=True)

    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    single_mask_f = token_mask.to(z_init.dtype)
    pair_mask_f = pair_mask.to(z_init.dtype)

    def trunk_pass(mode):
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        if mode == "upstream":
            torch.manual_seed(0)
        cycles = []
        with torch.no_grad():
            for c in range(NUM_CYCLES):
                z = z_init + lin_z(ln_z(z))
                z = z + te(batch=batch, z=z, pair_mask=pair_mask)
                if mode == "fixed":
                    m, msk = m_fixed, msa_mask_sel.unsqueeze(0)
                else:
                    m, msk = me(batch=batch, s_input=s_input_ref)
                z = msa_stack(m, z, msa_mask=msk.to(z_init.dtype), pair_mask=pair_mask_f)
                z_after_msa = z.clone()
                s = s_init + lin_s(ln_s(s))
                s, z = pairformer(
                    s=s, z=z, single_mask=single_mask_f, pair_mask=pair_mask_f,
                    use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                    use_triton_triangle_kernels=False, use_lma=False,
                    inplace_safe=False, _mask_trans=True,
                )
                cycles.append({"z_after_msa": z_after_msa[0], "z_out": z[0].clone(),
                               "s_out": s[0].clone()})
                print(f"  [{mode}] cycle {c}: z_after_msa std={float(z_after_msa.std()):.3f} "
                      f"z_out std={float(z.std()):.3f} s_out std={float(s.std()):.3f}")
        return s[0], z[0], cycles

    print("pass fixed (device draw, constant m):")
    s_fixed, z_fixed, cyc_fixed = trunk_pass("fixed")
    print("pass upstream (per-cycle fresh subsample):")
    s_up, z_up, cyc_up = trunk_pass("upstream")

    tr = I["trunk_real"]
    print(f"RESULT regime: z_trunk std single-seq-fixture={float(tr['z_trunk'].std()):.3f} "
          f"searched-MSA-fixed={float(z_fixed.std()):.3f} "
          f"upstream={float(z_up.std()):.3f}")
    print(f"RESULT draw-sensitivity: pcc_z(fixed,upstream)={pcc(z_fixed, z_up):.6f} "
          f"pcc_s(fixed,upstream)={pcc(s_fixed, s_up):.6f}")

    torch.save(
        {
            "s_input_ref": s_input_ref[0],
            "s_input_tt": s_input_tt,
            "s_init": s_init[0],
            "z_init": z_init[0],
            "token_mask": token_mask[0],
            "msa_feat_tt": msa_feat_tt,
            "msa_mask_sel": msa_mask_sel,
            "m_fixed": m_fixed[0],
            "s_trunk_fixed": s_fixed,
            "z_trunk_fixed": z_fixed,
            "s_trunk_upstream": s_up,
            "z_trunk_upstream": z_up,
            "cycles_fixed": cyc_fixed,
            "cycles_upstream": cyc_up,
        },
        OUT,
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
