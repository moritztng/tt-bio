"""Capture per-primitive fp32 + bf16 goldens WITHIN the OF3 MSA block-0 pair_stack.

The CPU bisect (of3_msa_bisect_cpu.py) localized the entire 15x z-magnitude jump
(std 18 -> 270) to the pair_stack as a whole, and showed a full-bf16 CPU stack tracks
it to z_pcc=0.9998 -- so the device's z_pcc=0.708 loss is device-pair_stack-specific.
This script breaks the pair_stack into its 5 PairformerLayer sub-ops
(tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, pair_transition) and captures z
after each, in BOTH fp32 and full-bf16, so the device leg can localize which sub-op's
device compute departs from the CPU baseline (and whether bf16 alone departs there too,
isolating a device-only vs bf16-general mechanism). Mirrors the reference PairBlock
forward order (base_blocks.py PairBlock.forward).

Writes ~/of3_msa_pair_stack_subops.pkl:
  z_in:           z fed to pair_stack (after OPM+PWA+transition), [N,N,c_z]
  pair_mask:      [N,N]
  fp32:           {sub_op: z_after} for the 5 sub-ops
  bf16:           {sub_op: z_after} for the 5 sub-ops (full-bf16 weights+acts)
  bf16_pcc:       {sub_op: pcc(bf16_z, fp32_z)} -- isolates bf16-alone departure per sub-op

Run with the CPU reference venv:
    OF3_REF=/tmp/of3-ref TT_BIO_ROOT=<worktree> /tmp/of3-venv/bin/python scripts/of3_msa_pair_stack_subop_golden.py
"""
import os, sys, pickle, copy
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.expanduser(
    "~/.coworker/wt/tt-bio-openfold3-msa-precision-gap"))
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
QUERY_JSON = os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json")
OUT = os.path.expanduser("~/of3_msa_pair_stack_subops.pkl")
sys.path.insert(0, OF3_REF)
sys.path.insert(0, REPO_ROOT)


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


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
    return m, z_init, msa_mask_b, pair_mask_b, C, sd


def run_pair_stack_subops(pair_block, z_in, pair_mask, cast_bf16):
    """Replicate PairBlock.forward sub-op-by-sub-op, capturing z after each.
    Order: tri_mul_out, tri_mul_in, tri_att_start, tri_att_end, pair_transition.
    Matches base_blocks.py PairBlock.forward + tri_mul_out_in + tri_att_start_end."""
    if cast_bf16:
        pb = copy.deepcopy(pair_block).to(torch.bfloat16)
        z = z_in.to(torch.bfloat16)
        pm = pair_mask.to(torch.bfloat16)
    else:
        pb = pair_block
        z = z_in
        pm = pair_mask
    out = {}
    with torch.no_grad():
        # tri_mul_out_in: z = z + tri_mul_out(z, mask); z = z + tri_mul_in(z, mask)
        upd = pb.tri_mul_out(z, mask=pm, inplace_safe=False, _add_with_inplace=True)
        z = z + upd
        out["tri_mul_out"] = z.float().clone()
        upd = pb.tri_mul_in(z, mask=pm, inplace_safe=False, _add_with_inplace=True)
        z = z + upd
        out["tri_mul_in"] = z.float().clone()
        # tri_att_start_end: z = z + tri_att_start(z, mask); transpose; z = z + tri_att_end(z, mask.T); transpose
        upd = pb.tri_att_start(z, mask=pm, chunk_size=None,
                               use_deepspeed_evo_attention=False,
                               use_cueq_triangle_kernels=False,
                               use_triton_triangle_kernels=False,
                               use_lma=False, inplace_safe=False)
        z = z + upd
        out["tri_att_start"] = z.float().clone()
        z = z.transpose(-2, -3)
        upd = pb.tri_att_end(z, mask=pm.transpose(-1, -2), chunk_size=None,
                             use_deepspeed_evo_attention=False,
                             use_cueq_triangle_kernels=False,
                             use_triton_triangle_kernels=False,
                             use_lma=False, inplace_safe=False)
        z = z + upd
        z = z.transpose(-2, -3)
        out["tri_att_end"] = z.float().clone()
        # pair_transition
        upd = pb.pair_transition(z, mask=pm, chunk_size=None)
        z = z + upd
        out["pair_transition"] = z.float().clone()
    return out


def main():
    m, z_init, msa_mask, pair_mask, C, sd = featurize()
    print("m", tuple(m.shape), "z_init", tuple(z_init.shape),
          "m std", float(m.std()), "z_init std", float(z_init.std()))

    # Run block-0 up to pair_stack input (opm_first=True: OPM, then PWA, then transition)
    from openfold3.core.model.latent.msa_module import MSAModuleBlock
    import inspect
    MSA_PF = dict(C.architecture.msa.msa_module)
    params = set(inspect.signature(MSAModuleBlock.__init__).parameters) - {"self", "last_block"}
    blk0 = MSAModuleBlock(**{k: v for k, v in MSA_PF.items() if k in params}, last_block=False).eval()
    blk0.load_state_dict(sub(sd, "msa_module.blocks.0"), strict=True)
    with torch.no_grad():
        opm = blk0.outer_product_mean(m, mask=msa_mask, chunk_size=None, inplace_safe=False)
        z = z_init + opm
        m = m + blk0.msa_att_row(m, z=z, mask=pair_mask, chunk_size=None)
        m = m + blk0.msa_transition(m, mask=msa_mask, chunk_size=None, ckpt_chunk_size=None)
        z_in = z.clone()
    print("z_in to pair_stack std:", float(z_in.std()), "shape", tuple(z_in.shape))

    pair_block = blk0.pair_stack
    fp32 = run_pair_stack_subops(pair_block, z_in, pair_mask, cast_bf16=False)
    bf16 = run_pair_stack_subops(pair_block, z_in, pair_mask, cast_bf16=True)
    print("fp32 sub-op z std:", {k: round(float(v.std()), 2) for k, v in fp32.items()})
    print("bf16 sub-op z_pcc vs fp32:",
          {k: round(pcc(bf16[k], fp32[k]), 5) for k in fp32})

    out = {
        "z_in": z_in[0].clone(),
        "pair_mask": pair_mask[0].clone(),
        "fp32": {k: v[0].clone() for k, v in fp32.items()},
        "bf16": {k: v[0].clone() for k, v in bf16.items()},
        "bf16_pcc": {k: pcc(bf16[k], fp32[k]) for k in fp32},
    }
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("wrote", OUT, "size", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
