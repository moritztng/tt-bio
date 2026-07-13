"""P6 bisect (CPU side): localize the MSA block-0 z amplification (thread 3).

P5 flagged the MSA block-0 z gap (device z_pcc=0.708) as a DISTINCT mechanism from the
pairformer's final-block cancellation: a single MSAModuleBlock takes z from std ~18 to
~270 (~15x) on this checkpoint's real ubiquitin input, while a pure-CPU bf16 control
tracks the same block to z_pcc=0.9998 (see tests/test_openfold3_msa.py docstring). So the
device loses real precision beyond bf16 rounding somewhere in this one block. This script
localizes WHERE the amplification happens (which sub-op) and where bf16 starts to depart,
by running the reference MSAModuleStack block-by-block AND block-0 sub-op-by-sub-op
(opm_first=True order: z+=OPM(m); m+=PWA(m,z); m+=transition(m); z=pair_stack(z)).

Writes ~/of3_msa_bisect.pkl:
  fp32:           per-block z (5 states: init + after each of 4 blocks)
  subops_b0:      block-0 z/m after each sub-op (opm, pwa, trans, pair_stack) -- fp32
  bf16_full:      per-block z_pcc of a full-bf16 stack (weights+acts bf16) vs fp32
  bf16_storage:   per-block z_pcc of an fp32-compute stack that ROUNDS z/m to bf16 between
                  blocks -- isolates inter-block storage rounding from bf16 compute
  bf16_subops_b0: per-sub-op z_pcc (and m_pcc) of a full-bf16 block-0 vs fp32 -- isolates
                  which sub-op's bf16 compute first departs from fp32
  inits:          m, z, msa_mask, pair_mask (for the device leg)

Run with the CPU reference venv:
    OF3_REF=/tmp/of3-ref TT_BIO_ROOT=<worktree> /tmp/of3-venv/bin/python scripts/of3_msa_bisect_cpu.py
"""
import os, sys, pickle
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.expanduser("~/.coworker/wt/tt-bio-openfold3-port-p6"))
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
QUERY_JSON = os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json")
OUT = os.path.expanduser("~/of3_msa_bisect.pkl")
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def bf16_rt(x):
    return x.to(torch.bfloat16).float()


def _kw():
    return dict(chunk_size=None, transition_ckpt_chunk_size=None,
                use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True)


def featurize():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom, MSAModuleEmbedder)
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet)
    from tt_bio.openfold3_data import build_openfold3_features

    qs = InferenceQuerySet.from_json(QUERY_JSON)
    query = next(iter(qs.queries.values()))
    feat = build_openfold3_features(query)
    batch = {k: v.unsqueeze(0) for k, v in feat.items() if torch.is_tensor(v)}
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    ie = InputEmbedderAllAtom(**C.architecture.input_embedder).eval()
    ie.load_state_dict(sub(sd, "input_embedder"), strict=True)
    with torch.no_grad():
        s_input, s_init, z_init = ie(batch=batch)
    me = MSAModuleEmbedder(**C.architecture.msa.msa_module_embedder).eval()
    me.load_state_dict(sub(sd, "msa_module_embedder"), strict=True)
    with torch.no_grad():
        m, msa_mask = me(batch=batch, s_input=s_input)
    single_mask = batch["token_mask"]
    pair_mask = single_mask[..., None] * single_mask[..., None, :]
    msa_mask_b = msa_mask.to(z_init.dtype)
    pair_mask_b = pair_mask.to(z_init.dtype)
    stack = MSAModuleStack(**dict(C.architecture.msa.msa_module)).eval()
    stack.load_state_dict(sub(sd, "msa_module"), strict=True)
    return stack, m, z_init, msa_mask_b, pair_mask_b, C


def run_stack_traj(stack, m0, z0, msa_mask, pair_mask, cast_bf16_weights, round_storage):
    """Run the 4-block MSAModuleStack block-by-block; return list of z after each block."""
    import copy
    st = copy.deepcopy(stack) if cast_bf16_weights else stack
    if cast_bf16_weights:
        st = st.to(torch.bfloat16)
        m0, z0 = m0.to(torch.bfloat16), z0.to(torch.bfloat16)
        msa_mask, pair_mask = msa_mask.to(torch.bfloat16), pair_mask.to(torch.bfloat16)
    with torch.no_grad():
        blocks = st._prep_blocks(m=m0, z=z0, **_kw(), msa_mask=msa_mask, pair_mask=pair_mask)
        m, z = m0, z0
        zs = []
        for b in blocks:
            m, z = b(m, z)
            if round_storage:
                m, z = bf16_rt(m), bf16_rt(z)
            zs.append(z.float().clone())
        return zs


def run_block0_subops(block, m0, z0, msa_mask, pair_mask, cast_bf16):
    """Run one MSAModuleBlock (opm_first=True, not last) sub-op-by-sub-op in fp32 (or bf16).
    Returns dict of (m, z) after each sub-op: opm, pwa, trans, pair_stack."""
    import copy
    if cast_bf16:
        block = copy.deepcopy(block).to(torch.bfloat16)
        m0, z0 = m0.to(torch.bfloat16), z0.to(torch.bfloat16)
        msa_mask, pair_mask = msa_mask.to(torch.bfloat16), pair_mask.to(torch.bfloat16)
    out = {}
    with torch.no_grad():
        m, z = m0, z0
        opm = block.outer_product_mean(m, mask=msa_mask, chunk_size=None, inplace_safe=False)
        z = z + opm
        out["opm"] = (m.float().clone(), z.float().clone())
        m = m + block.msa_att_row(m, z=z, mask=pair_mask, chunk_size=None)
        out["pwa"] = (m.float().clone(), z.float().clone())
        m = m + block.msa_transition(m, mask=msa_mask, chunk_size=None, ckpt_chunk_size=None)
        out["trans"] = (m.float().clone(), z.float().clone())
        z = block.pair_stack(z=z, pair_mask=pair_mask, chunk_size=None,
                             use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                             use_triton_triangle_kernels=False, use_lma=False,
                             inplace_safe=False, _mask_trans=True)
        out["pair_stack"] = (m.float().clone(), z.float().clone())
    return out


def main():
    stack, m, z_init, msa_mask, pair_mask, C = featurize()
    print("m", tuple(m.shape), "z_init", tuple(z_init.shape),
          "m std", float(m.std()), "z_init std", float(z_init.std()))

    # fp32 block-by-block trajectory
    with torch.no_grad():
        blocks = stack._prep_blocks(m=m, z=z_init, **_kw(), msa_mask=msa_mask, pair_mask=pair_mask)
        mm, z = m, z_init
        traj_z = []
        for b in blocks:
            mm, z = b(mm, z)
            traj_z.append(z.float().clone())
    print("fp32 per-block z std:", [round(float(t.std()), 2) for t in traj_z])

    # bf16 controls
    zs_full = run_stack_traj(stack, m.clone(), z_init.clone(), msa_mask, pair_mask,
                             cast_bf16_weights=True, round_storage=False)
    zs_store = run_stack_traj(stack, m.clone(), z_init.clone(), msa_mask, pair_mask,
                              cast_bf16_weights=False, round_storage=True)
    bf16_full = [pcc(zs_full[i], traj_z[i]) for i in range(len(traj_z))]
    bf16_storage = [pcc(zs_store[i], traj_z[i]) for i in range(len(traj_z))]
    print("bf16_full    z_pcc per block:", [round(x, 4) for x in bf16_full])
    print("bf16_storage z_pcc per block:", [round(x, 4) for x in bf16_storage])

    # block-0 sub-op bisect (fp32 + bf16)
    from openfold3.core.model.latent.msa_module import MSAModuleBlock
    import inspect
    MSA_PF = dict(C.architecture.msa.msa_module)
    params = set(inspect.signature(MSAModuleBlock.__init__).parameters) - {"self", "last_block"}
    blk0 = MSAModuleBlock(**{k: v for k, v in MSA_PF.items() if k in params}, last_block=False).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    blk0.load_state_dict(sub(sd, "msa_module.blocks.0"), strict=True)
    sub_fp32 = run_block0_subops(blk0, m.clone(), z_init.clone(), msa_mask, pair_mask, cast_bf16=False)
    sub_bf16 = run_block0_subops(blk0, m.clone(), z_init.clone(), msa_mask, pair_mask, cast_bf16=True)
    print("block-0 sub-op z std (fp32):",
          {k: round(float(v[1].std()), 2) for k, v in sub_fp32.items()})
    print("block-0 sub-op z_pcc (bf16 vs fp32):",
          {k: round(pcc(sub_bf16[k][1], sub_fp32[k][1]), 4) for k in sub_fp32})
    print("block-0 sub-op m_pcc (bf16 vs fp32):",
          {k: round(pcc(sub_bf16[k][0], sub_fp32[k][0]), 4) for k in sub_fp32})

    out = {
        "m": m[0].clone(), "z_init": z_init[0].clone(),
        "msa_mask": msa_mask[0].clone(), "pair_mask": pair_mask[0].clone(),
        "traj_z": [t[0].clone() for t in traj_z],
        "bf16_full": bf16_full, "bf16_storage": bf16_storage,
        "subops_b0_fp32": {k: (v[0][0].clone(), v[1][0].clone()) for k, v in sub_fp32.items()},
        "bf16_subops_b0": {k: (pcc(sub_bf16[k][1], sub_fp32[k][1]),
                               pcc(sub_bf16[k][0], sub_fp32[k][0])) for k in sub_fp32},
    }
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("wrote", OUT, "size", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
