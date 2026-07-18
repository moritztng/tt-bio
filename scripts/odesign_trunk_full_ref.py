"""Pass-7 CPU-fp32 FULL-TRUNK reference: run ODesign's own trunk cycle
(get_pairformer_output logic) on the prot_binding_prot example (use_msa=False,
data_condition={data, constraint_distogram}, N_cycle=10) in fp32 on CPU, and dump
s_inputs / s_trunk / z_trunk (+ per-cycle s/z for drift localization). Verifies the
dumped s_trunk / z_trunk / s_inputs reproduce the golden pre (PCC ~0.999 -- the
bf16-GPU-vs-fp32-CPU gap, same magnitude as the pass-6 s_inputs 0.999314).

This establishes the rigorous parity baseline for the on-device trunk port
(scripts/odesign_trunk_full_parity.py). Same methodology as passes 1-6: CPU ref =
ODesign's own modules run fresh in fp32.

Instantiates ONLY the trunk modules (InputFeatureEmbedder, RelativePositionEncoding,
MSAModule, ConstraintTemplateEmbedder, PairformerStack, trunk-init linears +
layernorms) -- no hydra, no diffusion/head/invfold -- and replicates the
get_pairformer_output cycle (src/model/odesign.py L286-384) verbatim.

Pure torch, CPU, eval, no grad. Run from the ODesign repo root.
"""
import os, sys, json, pickle, time
os.chdir("/home/moritz/.coworker/scratch/odesign-ref/ODesign")
sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
os.environ.setdefault("LAYERNORM_TYPE", "")
os.environ["DATA_ROOT_DIR"] = "./data"   # CCD components.v20240608.cif symlinked here

import torch
import torch.nn.functional as F
from src.utils.inference.inference_utils import SampleDictToFeatures
from src.api.model_interface import PairFormerInput
from src.model.modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
from src.model.modules.pairformer import MSAModule, ConstraintTemplateEmbedder, PairformerStack
from src.model.modules.primitives import LinearNoBias
from src.utils.openfold_local.model.primitives import LayerNorm

JSON = "/home/moritz/.coworker/scratch/odesign-ref/ODesign/examples/protein_design/prot_binding_prot/odesign_input.json"
CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"
OUT = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_full_ref.pkl"

C_S, C_Z, C_S_INPUTS = 384, 128, 453


def pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def load_sd():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def main():
    torch.set_grad_enabled(False)
    sd = load_sd()

    # --- instantiate the trunk modules with the ODesign config params ---
    input_embedder = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    rpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=C_Z)
    msa_module = MSAModule(n_blocks=4, c_m=64, c_z=C_Z, c_s_inputs=C_S_INPUTS,
                           msa_configs={"enable": True, "strategy": "random",
                                         "sample_cutoff": {"test": 16384, "train": 16384},
                                         "min_size": {"test": 1, "train": 1}})
    cte = ConstraintTemplateEmbedder(n_blocks=2, c=64, c_z=C_Z)
    pairformer_stack = PairformerStack(n_blocks=48, n_heads=16, c_z=C_Z, c_s=C_S)
    # trunk-init linears + cycle layernorms/linears (top-level keys in the checkpoint)
    lin_sinit = LinearNoBias(in_features=C_S_INPUTS, out_features=C_S)
    lin_zinit1 = LinearNoBias(in_features=C_S, out_features=C_Z)
    lin_zinit2 = LinearNoBias(in_features=C_S, out_features=C_Z)
    lin_token_bond = LinearNoBias(in_features=1, out_features=C_Z)
    lin_z_cycle = LinearNoBias(in_features=C_Z, out_features=C_Z)
    ln_z_cycle = LayerNorm(C_Z)
    lin_s_cycle = LinearNoBias(in_features=C_S, out_features=C_S)
    ln_s = LayerNorm(C_S)

    # --- load weights (strict per-submodule) ---
    def load_sub(module, pfx):
        sub = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
        miss, unexp = module.load_state_dict(sub, strict=False)
        if miss or unexp:
            print(f"  {pfx} missing={len(miss)} unexpected={len(unexp)}")
            if miss: print("    miss:", miss[:5])
            if unexp: print("    unexp:", unexp[:5])
        return module
    load_sub(input_embedder, "input_embedder.")
    load_sub(rpe, "relative_position_encoding.")
    load_sub(msa_module, "msa_module.")
    load_sub(cte, "constraint_distogram_embedder.")
    load_sub(pairformer_stack, "pairformer_stack.")
    lin_sinit.weight.data = sd["linear_no_bias_sinit.weight"].clone()
    lin_zinit1.weight.data = sd["linear_no_bias_zinit1.weight"].clone()
    lin_zinit2.weight.data = sd["linear_no_bias_zinit2.weight"].clone()
    lin_token_bond.weight.data = sd["linear_no_bias_token_bond.weight"].clone()
    lin_z_cycle.weight.data = sd["linear_no_bias_z_cycle.weight"].clone()
    ln_z_cycle.weight.data = sd["layernorm_z_cycle.weight"].clone()
    ln_z_cycle.bias.data = sd["layernorm_z_cycle.bias"].clone()
    lin_s_cycle.weight.data = sd["linear_no_bias_s.weight"].clone()
    ln_s.weight.data = sd["layernorm_s.weight"].clone()
    ln_s.bias.data = sd["layernorm_s.bias"].clone()
    for m in [input_embedder, rpe, msa_module, cte, pairformer_stack]:
        m.eval()

    # --- rebuild the live feature_data via the data pipeline (faithful dtypes) ---
    samples = json.load(open(JSON))
    s = samples[0]
    print("sample name:", s.get("name"))
    s2f = SampleDictToFeatures(single_sample_dict=s,
                                data_condition={"data", "constraint_distogram"},
                                use_msa=False)
    feature_data, label_data, atom_array, token_array = s2f.get_feature_and_label()
    pfi = PairFormerInput.from_feature_data(feature_data)
    # move all tensors to cpu; keep integer/index dtypes (auto_type_convert set them),
    # upcast only the floating-point feature tensors to fp32 for the reference.
    for k in list(pfi.keys()):
        v = getattr(pfi, k, None)
        if torch.is_tensor(v):
            v = v.to("cpu")
            if v.is_floating_point():
                v = v.float()
            setattr(pfi, k, v)
    N = pfi.residue_index.shape[-1]
    print("N_token =", N)

    # --- replicate get_pairformer_output (src/model/odesign.py L286-384) in fp32 ---
    s_inputs = input_embedder(pfi, inplace_safe=True, chunk_size=4)
    s_init = lin_sinit(s_inputs)
    z_init = (lin_zinit1(s_init)[..., None, :] + lin_zinit2(s_init)[..., None, :, :])
    z_init = z_init + rpe(pfi)
    z_init = z_init + lin_token_bond(pfi.token_bonds.unsqueeze(-1))
    z = torch.zeros_like(z_init)
    s = torch.zeros_like(s_init)
    per_cycle = []
    t0 = time.time()
    for cyc in range(10):
        z = z_init + lin_z_cycle(ln_z_cycle(z))
        # constraint_distogram enabled -> z = z + CTE(input_data, z)
        z = z + cte(pfi, z, pair_mask=None, use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=False, use_lma=False,
                    inplace_safe=True, chunk_size=4)
        # MSA module: use_msa=False -> single-sequence self-MSA (set_default_msa_features),
        #   so MSAModule.forward RUNS (msa key present, dim=2). Faithful to the golden.
        z = msa_module(pfi, z, s_inputs, pair_mask=None, use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=False, use_lma=False,
                        inplace_safe=True, chunk_size=4)
        s = s_init + lin_s_cycle(ln_s(s))
        s, z = pairformer_stack(s, z, pair_mask=None, use_memory_efficient_kernel=False,
                                use_deepspeed_evo_attention=False, use_lma=False,
                                inplace_safe=True, chunk_size=4)
        per_cycle.append({"s": s.detach().clone(), "z": z.detach().clone()})
    print("trunk run (10 cycles, 48 blocks/cycle): %.1fs" % (time.time() - t0))
    s_trunk = s.float(); z_trunk = z.float(); s_inputs = s_inputs.float()
    print("s_inputs", tuple(s_inputs.shape), "s_trunk", tuple(s_trunk.shape),
          "z_trunk", tuple(z_trunk.shape), "cycles captured:", len(per_cycle))

    # --- verify vs golden pre ---
    pre = pickle.load(open(PRE, "rb"))
    print("\n--- vs golden pre (bf16-GPU) ---")
    for name, ref, dev in [("s_inputs", pre["s_inputs"].float(), s_inputs),
                          ("s_trunk", pre["s_trunk"].float(), s_trunk),
                          ("z_trunk", pre["z_trunk"].float(), z_trunk)]:
        p = pcc(ref, dev); m = float((ref - dev).abs().max())
        print("  %-9s PCC %.6f  maxerr %.3e" % (name, p, m))

    out = {"s_inputs": s_inputs, "s_trunk": s_trunk, "z_trunk": z_trunk, "per_cycle": per_cycle}
    pickle.dump(out, open(OUT, "wb"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
