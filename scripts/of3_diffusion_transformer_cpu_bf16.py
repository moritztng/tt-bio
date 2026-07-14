"""CPU bf16 control for the OF3 DiT stack: cast the reference DiffusionTransformer to
bf16 and run it on the golden (a_in, s, z, mask) vs the fp32 reference trajectory. If
CPU-bf16 collapses to the same ~0.3 PCC as the device, the stack gate is an intrinsic
bf16-conditioning limit (not a device bug); if CPU-bf16 stays high, the device has a
real bug to hunt. Mirrors the P5 pairformer bisect discipline.

    /tmp/of3-venv/bin/python scripts/of3_diffusion_transformer_cpu_bf16.py
"""
import os, sys, pickle, torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
REPO_ROOT = os.environ.get("TT_BIO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OF3_REF); sys.path.insert(0, REPO_ROOT)
_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    if a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def sub(sd, prefix):
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def main():
    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    cfg = dict(C.architecture.diffusion_module.diffusion_transformer)
    cfg["blocks_per_ckpt"] = None
    dt = DiffusionTransformer(**cfg).eval()
    dt.load_state_dict(sub(sd, "diffusion_module.diffusion_transformer"), strict=True)

    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_transformer_real"]
    a_in, s, z, tok = g["a_in"], g["s"], g["z"], g["token_mask"]
    traj = g["a_traj"]

    def run(dtype):
        d = dt  # keep fp32 weights; autocast emulates device bf16 matmul-with-bf16-accum
        a = a_in.unsqueeze(0)
        with torch.no_grad(), torch.autocast("cpu", dtype=dtype, enabled=(dtype != torch.float32)):
            sout = d(a=a, s=s.unsqueeze(0), z=z.unsqueeze(0),
                     mask=tok.unsqueeze(0), _mask_trans=True)
        return sout[0].float()

    out_fp32 = run(torch.float32)
    out_bf16 = run(torch.bfloat16)
    print(f"fp32 stack vs ref traj[-1]: pcc={pcc(out_fp32, traj[-1].float()):.5f}  std={out_fp32.std():.3f} vs {traj[-1].std():.3f}")
    print(f"bf16 stack vs ref traj[-1]: pcc={pcc(out_bf16, traj[-1].float()):.5f}  std={out_bf16.std():.3f} vs {traj[-1].std():.3f}")
    print(f"bf16 stack vs fp32 stack:   pcc={pcc(out_bf16, out_fp32):.5f}")


if __name__ == "__main__":
    main()
