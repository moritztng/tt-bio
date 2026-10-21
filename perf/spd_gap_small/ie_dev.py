"""ie_dev probe: Boltz-2's input-embedder atom encoder on the card vs the fp32 host module.

Synthetic 512-token input (4128 atom slots, 4100 real, 8 atoms a token), Boltz-2 dims, random weights scaled
by fan-in. Reports the device output's relative error against the host fp32 module (whose own error is ~1e-6,
far below bf16), determinism over REPS calls, and wall time per call on each side.

    python perf/spd_gap_small/ie_dev.py --out r.json [--reps 3]
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import boltz2 as B  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

torch.manual_seed(0)
P, N, n_real = 512, 4128, 4100
ie = B.InputEmbedder(atom_s=128, atom_z=16, token_s=384, token_z=128, atoms_per_window_queries=32,
                     atoms_per_window_keys=128, atom_feature_dim=388, atom_encoder_depth=3,
                     atom_encoder_heads=4).eval()
with torch.no_grad():
    for name, prm in ie.named_parameters():
        if prm.dim() == 2:
            prm.normal_(std=1.0 / prm.shape[1] ** 0.5)
        else:
            prm.copy_(1.0 + 0.1 * torch.randn_like(prm) if name.endswith("weight") else 0.1 * torch.randn_like(prm))
tok = torch.clamp(torch.arange(N) // 8, max=P - 1)
a2t = torch.zeros(1, N, P)
a2t[0, torch.arange(n_real), tok[:n_real]] = 1.0
real = (torch.arange(N) < n_real).float().view(1, N)
feats = {"atom_pad_mask": real,
         "ref_pos": torch.randn(1, N, 3) * 3 * real.unsqueeze(-1),
         "ref_charge": torch.randint(-1, 2, (1, N)).float() * real,
         "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (1, N)), 128).float() * real.unsqueeze(-1),
         "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (1, N, 4)), 64).float()
         * real.view(1, N, 1, 1),
         "ref_space_uid": (torch.arange(N) // 8).view(1, N),
         "atom_to_token": a2t}


def host():
    q, c, p, to_keys = ie.atom_encoder(feats)
    a, _, _, _ = ie.atom_attention_encoder(feats=feats, q=q, c=c, atom_enc_bias=ie.atom_enc_proj_z(p),
                                           to_keys=to_keys)
    return a


with torch.no_grad():
    host()
    t = time.perf_counter()
    ref = host()
    host_s = time.perf_counter() - t

dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)   # TorchWrapper's, which msa_module carries in the fold
mod = T.InputAtomEncoderDevice(ie, ckc)


def rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


outs, secs = [], []
with torch.no_grad():
    for r in range(args.reps + 1):
        t = time.perf_counter()
        out = mod(feats)
        ttnn.synchronize_device(dev)
        secs.append(time.perf_counter() - t)
        outs.append(out)
real_tok = a2t[0].sum(0) > 0
rec = {"arch": str(dev.arch()), "shape": list(outs[-1].shape), "ref_shape": list(ref.shape),
       "rel_vs_host_fp32": [rel(o[0, real_tok], ref[0, real_tok]) for o in outs],
       "max_abs": float((outs[-1][0, real_tok] - ref[0, real_tok]).abs().max()),
       "ref_absmax": float(ref.abs().max()),
       "deterministic": all(torch.equal(outs[1], o) for o in outs[2:]),
       "finite": bool(torch.isfinite(outs[-1]).all()),
       "dev_s": secs, "host_s": host_s}
print(json.dumps(rec), flush=True)
json.dump(rec, open(args.out, "w"), indent=1)
