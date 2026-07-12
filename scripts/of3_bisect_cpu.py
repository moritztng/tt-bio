"""P5 bisect (CPU side): capture the full 48-block reference trajectory + bf16 controls.

Feeds the REAL featurized ubiquitin example (P1 build_openfold3_features -> reference
InputEmbedderAllAtom) into the reference PairFormerStack, running block-by-block via the
stack's own _prep_blocks() (so we run EXACTLY the reference forward, just capturing every
intermediate). Writes ~/of3_bisect.pkl:

  fp32:            per-block (s, z) trajectory (49 states: init + after each of 48 blocks)
  masks/init:      single_mask, pair_mask, s_init, z_init  (for the device leg)
  bf16_full:       per-block z_pcc of a full-bf16 stack (weights+acts bf16) vs fp32
  bf16_storage:    per-block z_pcc of an fp32-compute stack that ROUNDS z (and s) to bf16
                   between blocks -- isolates inter-block storage rounding from bf16 compute

Run with the CPU reference venv:
    OF3_REF=/tmp/of3-ref /tmp/of3-venv/bin/python /tmp/of3_bisect_cpu.py
"""
import os, sys, pickle
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.expanduser("~/.coworker/wt/tt-bio-openfold3-port-p5"))
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
QUERY_JSON = os.path.join(OF3_REF, "examples/example_inference_inputs/query_ubiquitin.json")
OUT = os.path.expanduser("~/of3_bisect.pkl")
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


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.feature_embedders.input_embedders import InputEmbedderAllAtom
    from openfold3.core.model.latent.pairformer import PairFormerStack

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
    from tt_bio.openfold3_data import build_openfold3_features

    qs = InferenceQuerySet.from_json(QUERY_JSON)
    query = next(iter(qs.queries.values()))
    feat = build_openfold3_features(query)
    n = int(feat["token_mask"].shape[0])
    batch = {k: v.unsqueeze(0) for k, v in feat.items() if torch.is_tensor(v)}
    print("n_tokens", n)

    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    ie = InputEmbedderAllAtom(**C.architecture.input_embedder).eval()
    ie.load_state_dict(sub(sd, "input_embedder"), strict=True)
    with torch.no_grad():
        s_input, s_init, z_init = ie(batch=batch)
    print("s_init std", float(s_init.std()), "z_init std", float(z_init.std()))

    single_mask = batch["token_mask"].float()
    pair_mask = single_mask[..., None] * single_mask[..., None, :]

    stack = PairFormerStack(**dict(C.architecture.pairformer)).eval()
    stack.load_state_dict(sub(sd, "pairformer_stack"), strict=True)

    def run_traj(s0, z0, cast_bf16_weights, round_storage):
        # returns list of z after each block (fp32), running block-by-block.
        st = stack
        if cast_bf16_weights:
            st = PairFormerStack(**dict(C.architecture.pairformer)).eval()
            st.load_state_dict(sub(sd, "pairformer_stack"), strict=True)
            st = st.to(torch.bfloat16)
        with torch.no_grad():
            blocks = st._prep_blocks(
                s=s0, z=z0, single_mask=single_mask.to(s0.dtype),
                pair_mask=pair_mask.to(s0.dtype), chunk_size=None,
                use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False,
                _mask_trans=True,
            )
            s, z = s0, z0
            zs = []
            for b in blocks:
                s, z = b(s, z)
                if round_storage:
                    s, z = bf16_rt(s), bf16_rt(z)
                zs.append(z.float().clone())
            return zs

    # fp32 reference trajectory
    with torch.no_grad():
        blocks = stack._prep_blocks(
            s=s_init, z=z_init, single_mask=single_mask, pair_mask=pair_mask,
            chunk_size=None, use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False,
            use_lma=False, inplace_safe=False, _mask_trans=True,
        )
        s, z = s_init, z_init
        traj_s, traj_z = [], []
        for b in blocks:
            s, z = b(s, z)
            traj_s.append(s.float().clone()); traj_z.append(z.float().clone())
    print("fp32 final z std", float(traj_z[-1].std()))

    # bf16 controls
    zs_full = run_traj(s_init.to(torch.bfloat16), z_init.to(torch.bfloat16),
                       cast_bf16_weights=True, round_storage=False)
    zs_store = run_traj(s_init.clone(), z_init.clone(),
                        cast_bf16_weights=False, round_storage=True)
    bf16_full = [pcc(zs_full[i], traj_z[i]) for i in range(len(traj_z))]
    bf16_storage = [pcc(zs_store[i], traj_z[i]) for i in range(len(traj_z))]
    print("bf16_full   z_pcc  b0/b7/b15/b23/b31/b39/b47:",
          [round(bf16_full[i], 4) for i in (0, 7, 15, 23, 31, 39, 47)])
    print("bf16_storage z_pcc b0/b7/b15/b23/b31/b39/b47:",
          [round(bf16_storage[i], 4) for i in (0, 7, 15, 23, 31, 39, 47)])

    out = {
        "n": n,
        "s_init": s_init[0].clone(), "z_init": z_init[0].clone(),
        "single_mask": single_mask[0].clone(), "pair_mask": pair_mask[0].clone(),
        "traj_s": [t[0].clone() for t in traj_s],
        "traj_z": [t[0].clone() for t in traj_z],
        "bf16_full": bf16_full, "bf16_storage": bf16_storage,
    }
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("wrote", OUT, "size", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
