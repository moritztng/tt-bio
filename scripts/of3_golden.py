"""P0 OpenFold3 CPU golden-activation harness.

Instantiates real OpenFold3 reference submodules (weights from of3-p2-155k.pt) and
captures per-component (input -> output) tensor pairs to a pickle. These are the
ground truth every ported tt-bio device component is PCC-gated against (see
tests/test_openfold3_*.py), mirroring the Protenix golden method
(~/protenix_ref_out.pkl + test_protenix_trunk_pairformer.py).

Inputs are deterministic seeded tensors of the exact config shapes -- NOT full JSON
featurization. That is sufficient and standard for component parity: the device port
is fed the identical captured input, so the PCC compares the SAME math. Full-pipeline
real-input golden (data pipeline vendor) is P1, a later tick.

CAVEAT (tick 3): the N(0,1) seeded input is valid ONLY for single-component s-track
parity (block 0 s_pcc=0.99985). It is OFF the learned manifold, so the reference
trunk explodes over 48 blocks (out s std ~3.7e4 vs a real fold's ~1.8e2); at that
magnitude bf16 collapses the z-track with no device involved (pure-CPU fp32-vs-bf16
z_pcc=0.72). So the deep-stack gate is NOT meaningful here -- it needs real
input-embedder output (the P3 InputEmbedder port), exactly as Protenix's stack
gate uses real captured trunk I/O. See docs/openfold3-port.md status log.

Run with the CPU reference venv, NOT the tt-bio device env:
    /tmp/of3-venv/bin/python scripts/of3_golden.py
"""
import os, sys, pickle
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
CKPT = os.environ.get("OF3_CKPT", os.path.expanduser("~/of3-weights/of3-p2-155k.pt"))
OUT = os.environ.get("OF3_GOLD", os.path.expanduser("~/of3_ref_out.pkl"))
sys.path.insert(0, OF3_REF)

# AF3 pairformer config (of3_all_atom/config/model_config.py)
C_S, C_Z = 384, 128
PF = dict(c_s=C_S, c_z=C_Z, c_hidden_pair_bias=24, no_heads_pair_bias=16,
          c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4,
          transition_type="swiglu", transition_n=4, pair_dropout=0.25,
          fuse_projection_weights=False, inf=1e9)
N = 37  # small token count


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def main():
    from openfold3.core.model.latent.pairformer import PairFormerBlock, PairFormerStack
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single_mask = torch.ones(1, N)
    pair_mask = torch.ones(1, N, N)

    inter = {}

    # ---- single Pairformer block 0 ----
    blk = PairFormerBlock(**PF)
    blk.load_state_dict(sub(sd, "pairformer_stack.blocks.0"), strict=True)
    blk.eval()
    with torch.no_grad():
        so, zo = blk(s.clone(), z.clone(), single_mask, pair_mask)
    inter["pairformer_block0"] = {"in": (s[0].clone(), z[0].clone()),
                                  "out": (so[0].clone(), zo[0].clone())}
    print("block0:", so.shape, zo.shape, "z_out mean", float(zo.mean()))

    # ---- full 48-block stack ----
    stack = PairFormerStack(no_blocks=48, blocks_per_ckpt=None, **PF)
    stack.load_state_dict(sub(sd, "pairformer_stack"), strict=True)
    stack.eval()
    with torch.no_grad():
        ss, zs = stack(s.clone(), z.clone(), single_mask, pair_mask)
    inter["pairformer_stack"] = {"in": (s[0].clone(), z[0].clone()),
                                 "out": (ss[0].clone(), zs[0].clone())}
    print("stack:", ss.shape, zs.shape, "z_out mean", float(zs.mean()))

    with open(OUT, "wb") as f:
        pickle.dump({"intermediates": inter, "config": PF, "N": N}, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
