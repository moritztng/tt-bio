#!/usr/bin/env python3
"""D116, part 2: why a row sum of 0.9934 costs 22.52x, and why one-op rel_l2 cannot see it.

`dq_i = sum_j dlogits_ij k_j`. Split k into its mean over j and the rest:
`sum_j dlogits_ij k_j = sum_j dlogits_ij (k_j - kbar) + (sum_j dlogits_ij) kbar`.
The true dlogits has zero row sums, so the second term is absent from the gradient. Ours does
not, so it is present, and it is multiplied by `kbar` -- a vector the true gradient never
touches. That is the amplifier, and it is why a residual worth 0.4 % of ||dlogits|| can be
worth 100x of ||dq||.

Arm 3 is the new lever: keep the bf16 forward exactly as it ships and do the BACKWARD's two
reductions in fp32. Backward-only, so no inference result can move.
"""
import argparse, json, os, time

import os, pathlib, sys

# `python perf/of3t_d116/x.py` puts THIS file's directory on sys.path, not the repo root, so
# `import tt_bio` silently resolves to whatever is installed -- on this host the SHARED
# checkout /home/ttuser/tt-bio-dev. Measured the hard way: the first run of this script
# scored the shared tree and both arms came back bit-identical. Root first, then assert it.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT), (
    f"tt_bio came from {_tt_bio.__file__}, not {_ROOT}")

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.autograd import precise_config


def rel_l2(a, b):
    return float(torch.linalg.vector_norm((a - b).double()) / torch.linalg.vector_norm(b.double()))


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def to_dev(dev, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def case(dev, name, B, H, N, D, std, kmu, seed):
    torch.manual_seed(seed)
    logits = (torch.randn(B, H, N, N) * std).to(torch.bfloat16)
    g = torch.randn(B, H, N, N).to(torch.bfloat16)              # cotangent on the probs
    k = (torch.randn(B, H, N, D) + kmu * torch.randn(1, 1, 1, D)).to(torch.bfloat16)

    l_tt = to_dev(dev, logits)
    g_tt = to_dev(dev, g)
    y_tt = ttnn.softmax(l_tt, dim=-1)                            # the shipped forward, bf16
    y = ttnn.to_torch(y_tt).double()

    g64, k64 = g.double(), k.double()
    p = torch.softmax(logits.double(), dim=-1)

    dl_ref = p * (g64 - (g64 * p).sum(-1, keepdim=True))
    inner = (g64 * y).sum(-1, keepdim=True)
    dl_ship = y * (g64 - inner)                                  # main's rule, exact arithmetic
    dl_rn = y * (g64 - inner / y.sum(-1, keepdim=True))           # the repair, exact arithmetic

    # --- on device: shipped, renorm in bf16, renorm with the backward reductions in fp32 -----
    def dev_ship(dt):
        yy = ttnn.typecast(y_tt, dt) if dt != ttnn.bfloat16 else y_tt
        gg = ttnn.typecast(g_tt, dt) if dt != ttnn.bfloat16 else g_tt
        i = ttnn.sum(ttnn.multiply(gg, yy), dim=-1, keepdim=True,
                     compute_kernel_config=precise_config())
        return ttnn.to_torch(ttnn.multiply(yy, ttnn.subtract(gg, i))).double()

    def dev_rn(dt):
        yy = ttnn.typecast(y_tt, dt) if dt != ttnn.bfloat16 else y_tt
        gg = ttnn.typecast(g_tt, dt) if dt != ttnn.bfloat16 else g_tt
        i = ttnn.sum(ttnn.multiply(gg, yy), dim=-1, keepdim=True,
                     compute_kernel_config=precise_config())
        r = ttnn.sum(yy, dim=-1, keepdim=True, compute_kernel_config=precise_config())
        return ttnn.to_torch(ttnn.multiply(yy, ttnn.subtract(gg, ttnn.divide(i, r)))).double()

    dl_ship_dev = dev_ship(ttnn.bfloat16)
    dl_rn_dev = dev_rn(ttnn.bfloat16)
    dl_rn_dev32 = dev_rn(ttnn.float32)
    dl_ship_dev32 = dev_ship(ttnn.float32)

    kbar = k64.mean(-2, keepdim=True)
    common = float(torch.linalg.vector_norm(kbar) * (N ** 0.5)
                   / torch.linalg.vector_norm(k64 - kbar))

    arms = dict(shipped_rule_f64=dl_ship, renorm_rule_f64=dl_rn,
                shipped_device_bf16=dl_ship_dev, renorm_device_bf16=dl_rn_dev,
                shipped_device_fp32bw=dl_ship_dev32, renorm_device_fp32bw=dl_rn_dev32)
    dq_ref = dl_ref @ k64
    out = dict(case=name, shape=[B, H, N, D], std=std, kmu=kmu, seed=seed,
               k_common_ratio=common, arms={})
    for an, dl in arms.items():
        dq = dl @ k64
        rl_dl = rel_l2(dl, dl_ref)
        rl_dq = rel_l2(dq, dq_ref)
        rowsum = float(dl.sum(-1).pow(2).mean().sqrt())
        # the leak, written out: (row residual) x kbar, the term the true gradient lacks
        leak = dl.sum(-1, keepdim=True) * kbar
        out["arms"][an] = dict(
            dlogits_rel_l2=rl_dl, dq_rel_l2=rl_dq, amplification=rl_dq / rl_dl if rl_dl else None,
            dlogits_rowsum_rms=rowsum, dq_cos=cos(dq, dq_ref),
            leak_share_of_dq_err=float(torch.linalg.vector_norm(leak)
                                       / torch.linalg.vector_norm(dq - dq_ref)))
    out["reference"] = dict(dlogits_rowsum_rms=float(dl_ref.sum(-1).pow(2).mean().sqrt()),
                            dq_norm=float(torch.linalg.vector_norm(dq_ref)))
    # break control on the metric itself
    perm = torch.randperm(N)
    out["break_control"] = dict(dq_rel_l2=rel_l2((dl_ref @ k64)[..., perm, :], dq_ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_d116/amplify.json")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    dev = get_device()
    rows, t0 = [], time.time()
    # the trunk AttentionPairBias geometry: 16 heads, 384 tokens, 64 channels per head
    for kmu in (0.0, 0.5, 2.0, 5.0, 15.0):
        rows.append(case(dev, f"kmu{kmu}", 1, 16, 384, 64, 3.0, kmu, a.seed))
    for std in (1.0, 6.0, 12.0):
        rows.append(case(dev, f"std{std}_kmu2", 1, 16, 384, 64, std, 2.0, a.seed))
    rows.append(case(dev, "seed2_kmu2", 1, 16, 384, 64, 3.0, 2.0, a.seed + 1))
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=os.uname().nodename, seconds=round(time.time() - t0, 1), cases=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, out["seconds"], "s")


if __name__ == "__main__":
    main()
