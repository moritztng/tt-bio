"""atomp_dev determinism + accuracy probe: AtomPairDevice on fixed synthetic 512-token inputs.

apab (WH, 2a42df596) read -0.85 s fast / -0.86 s normal, but the "on" CIFs differ between warm reps while
"off" is identical every rep. This runs the device module REPS times on identical inputs, records every
intermediate (each _linear, _gather, layer_norm output) and reports the first stage whose output is not
bit-identical across runs, plus each stage's error against an fp64 torch reference of the same math.

    python perf/b2_atomp/probe.py --out probe.json [--reps 4]
"""
import argparse
import json
import sys
from functools import partial

import torch

sys.path.insert(0, ".")
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import boltz2 as B  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=4)
args = ap.parse_args()

torch.manual_seed(0)
P, N, W, H, AS, AZ, TZ = 512, 4128, 32, 128, 128, 16, 128
K = N // W
enc = B.AtomEncoder(atom_s=AS, atom_z=AZ, token_s=384, token_z=TZ, atoms_per_window_queries=W,
                    atoms_per_window_keys=H, atom_feature_dim=388).eval()
for prm in enc.parameters():
    torch.nn.init.normal_(prm, std=0.3)
n_real = 4100
tok = torch.clamp(torch.arange(N) // 8, max=P - 1)
a2t = torch.zeros(1, N, P)
a2t[0, torch.arange(n_real), tok[:n_real]] = 1.0
feats = {"atom_pad_mask": (torch.arange(N) < n_real).float().view(1, N),
         "ref_pos": torch.randn(1, N, 3) * 3,
         "ref_space_uid": (torch.arange(N) // 8).view(1, N),
         "atom_to_token": a2t}
c = torch.randn(1, N, AS)
z_in = torch.randn(1, P, P, TZ) * 0.5
wz = torch.randn(TZ, AZ) * 0.1
heads = 12
stack = lambda: [(torch.randn(heads, AZ) * 0.3, torch.randn(heads) * 0.1, None, 1e-5)][0]  # noqa: E731
enc_fused, dec_fused = stack(), stack()

dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)   # TorchWrapper's, which msa_module carries in the fold
mod = T.AtomPairDevice(enc, enc_fused, dec_fused, ckc)
wz_tt = ttnn.from_torch(wz, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
z_tt_in = ttnn.from_torch(z_in, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def z_to_p_dev():
    # As in the fold: z_to_p is a device linear's output, so its tile padding is whatever the op wrote.
    return ttnn.linear(z_tt_in, wz_tt, compute_kernel_config=ckc)


rec = []
orig_linear, orig_gather, orig_ln = mod._linear, mod._gather, T.ttnn.layer_norm


def tap(name, fn):
    def run(*a, **kw):
        out = fn(*a, **kw)
        rec.append((name, ttnn.to_torch(out).float()))
        return out
    return run


mod._linear = tap("linear", orig_linear)
mod._gather = tap("gather", orig_gather)
T.ttnn.layer_norm = tap("layer_norm", orig_ln)

runs = []
for r in range(args.reps):
    rec.clear()
    e, d, _ = mod(enc, feats, c, z_to_p_dev())
    rec.append(("enc", ttnn.to_torch(e).float()))
    rec.append(("dec", ttnn.to_torch(d).float()))
    ttnn.deallocate(e)
    ttnn.deallocate(d)
    runs.append(list(rec))
T.ttnn.layer_norm = orig_ln

# fp64 reference of the same math (bf16-rounded z_to_p input, as the card sees it).
with torch.no_grad():
    to_keys, K_, W_, H_, dd, dn, v = enc.windows(feats)
    f = torch.cat([dd * v, dn * v, v], -1).double().reshape(K * W, H, 5)
    w0 = torch.cat([enc.embed_atompair_ref_pos.weight, enc.embed_atompair_ref_dist.weight,
                    enc.embed_atompair_mask.weight], 1).double()
    p = f @ w0.t()
    zp = (z_in.bfloat16().double() @ wz.bfloat16().double())[0]
    tq, tk, keep = B.atom_window_tokens(a2t, to_keys, K, W, H)
    p = p + (zp[tq.view(K, W, 1), tk.view(K, 1, H)] * keep.view(K, W, H, 1).double()).reshape(K * W, H, AZ)
    cq = enc.c_to_p_trans_q(c[0]).double()
    ck = enc.c_to_p_trans_k(c[0]).double()
    atom = torch.arange(N, dtype=torch.float32).view(1, N, 1)
    key_atom = to_keys(atom).view(K, 1, H).expand(K, W, H).long()
    key_real = to_keys(torch.ones_like(atom)).view(K, 1, H).expand(K, W, H) > 0
    ck_pad = torch.cat([ck, torch.zeros(1, AZ, dtype=ck.dtype)])
    p = p + cq.view(K * W, 1, AZ) + ck_pad[torch.where(key_real, key_atom, N)].reshape(K * W, H, AZ)
    h = torch.relu(p)
    for i in (1, 3, 5):
        h = h @ enc.p_mlp[i].weight.double().t()
        if i < 5:
            h = torch.relu(h)
    p = p + h
    pn = torch.nn.functional.layer_norm(p, (AZ,), eps=1e-5)
    wcat = torch.cat([enc_fused[0], dec_fused[0]]).double()
    bcat = torch.cat([enc_fused[1], dec_fused[1]]).double()
    bias = (pn @ wcat.t() + bcat).reshape(K, W, H, 2 * heads).permute(0, 3, 1, 2)
    ref = {"layer_norm": pn, "enc": bias[:, :heads], "dec": bias[:, heads:]}


def rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


report = {"reps": args.reps, "stages": []}
for i, (name, t0) in enumerate(runs[0]):
    same = [bool(torch.equal(t0, run[i][1])) for run in runs[1:]]
    diff = [float((t0 - run[i][1]).abs().max()) for run in runs[1:]]
    row = {"i": i, "op": name, "shape": list(t0.shape), "bit_identical_to_run0": same,
           "max_abs_vs_run0": diff, "nan": bool(torch.isnan(t0).any()),
           "absmax": float(t0.abs().max())}
    if name in ref:
        row["rel_vs_f64"] = [rel(run[i][1], ref[name].reshape(run[i][1].shape)) for run in runs]
    report["stages"].append(row)
    print(json.dumps(row), flush=True)
first = next((s for s in report["stages"] if not all(s["bit_identical_to_run0"])), None)
report["first_nondeterministic"] = first
print("FIRST NONDETERMINISTIC:", json.dumps(first), flush=True)
json.dump(report, open(args.out, "w"), indent=1)
