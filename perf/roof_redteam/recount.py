#!/usr/bin/env python3
"""Independent re-derivation of the 512 aa Boltz-2 FLOP census, on three bases.

`perf/bioir_roofline/flops_bytes_512.json` was produced by running each real module under torch's
`FlopCounterMode`. This file does not run the modules. It enumerates every matmul in them from the
source, by shape, and sums 2*M*K*N. Agreeing with the census to five decimals is therefore an
independent check on the census and not a re-run of it.

Three bases:

  logical   2*M*K*N on the shapes the module declares. What FlopCounterMode counts.
  padded    every dimension rounded up to the 32x32 tile the Tensix matrix engine issues. What the
            device actually executes. The difference is the padding factor.
  true      logical, with the atom transformer's key window at its real 128 instead of the 32 the
            census script passed. The census file self-declares this undercount; this prices it.

No device, no torch. Run: python3 perf/roof_redteam/recount.py
"""
import json
from collections import defaultdict
from pathlib import Path

TILE = 32
ceil_t = lambda x: -(-x // TILE) * TILE


def mm(name, M, K, N, cnt=1):
    return dict(name=name, M=M, K=K, N=N, cnt=cnt)


def flops(ops, pad=False):
    f = ceil_t if pad else (lambda x: x)
    return sum(2 * f(o["M"]) * f(o["K"]) * f(o["N"]) * o["cnt"] for o in ops)


# --- the pair track: trimul x2, triangle attention x2, transition_z -----------------------
def pair_track(N, cz, tri_hw=32, tri_nh=4):
    L, NN, ch = [], N * N, tri_hw * tri_nh
    for d in ("out", "in"):
        L += [mm(f"trimul_{d}.p_in", NN, cz, 2 * cz),
              mm(f"trimul_{d}.g_in", NN, cz, 2 * cz),
              # einsum bikd,bjkd->bijd is a bmm batched over the cz channels
              mm(f"trimul_{d}.prod", N, N, N, cnt=cz),
              mm(f"trimul_{d}.p_out", NN, cz, cz),
              mm(f"trimul_{d}.g_out", NN, cz, cz)]
    for d in ("start", "end"):
        L += [mm(f"triatt_{d}.bias_lin", NN, cz, tri_nh),   # width tri_nh=4, pads to 32
              mm(f"triatt_{d}.q", NN, cz, ch),
              mm(f"triatt_{d}.k", NN, cz, ch),
              mm(f"triatt_{d}.v", NN, cz, ch),
              mm(f"triatt_{d}.qk", N, tri_hw, N, cnt=N * tri_nh),
              mm(f"triatt_{d}.av", N, N, tri_hw, cnt=N * tri_nh),
              mm(f"triatt_{d}.g", NN, cz, ch),
              mm(f"triatt_{d}.o", NN, ch, cz)]
    hz = 4 * cz
    L += [mm("trans_z.fc1", NN, cz, hz), mm("trans_z.fc2", NN, cz, hz),
          mm("trans_z.fc3", NN, hz, cz)]
    return L


def pairformer(N=512, cs=384, cz=128, heads=16):
    L = pair_track(N, cz)
    NN, hd, hs = N * N, cs // heads, 4 * cs
    L += [mm("apb.q", N, cs, cs), mm("apb.k", N, cs, cs), mm("apb.v", N, cs, cs),
          mm("apb.g", N, cs, cs),
          mm("apb.proj_z", NN, cz, heads),             # width heads=16, pads to 32
          mm("apb.qk", N, hd, N, cnt=heads),           # head_dim 24, pads to 32
          mm("apb.av", N, N, hd, cnt=heads),
          mm("apb.o", N, cs, cs),
          mm("trans_s.fc1", N, cs, hs), mm("trans_s.fc2", N, cs, hs),
          mm("trans_s.fc3", N, hs, cs)]
    return L


def msa(N=512, D=35, cm=64, cz=128, ch=32, nh=8):
    L, NN, DN, hm = pair_track(N, cz), N * N, D * N, nh * ch
    L += [mm("pwa.proj_m", DN, cm, hm), mm("pwa.proj_g", DN, cm, hm),
          mm("pwa.proj_z", NN, cz, nh),                # width nh=8, pads to 32
          mm("pwa.wavg", N, N, ch, cnt=nh * D),
          mm("pwa.proj_o", DN, hm, cm),
          mm("msa_trans.fc1", DN, cm, 4 * cm), mm("msa_trans.fc2", DN, cm, 4 * cm),
          mm("msa_trans.fc3", DN, 4 * cm, cm),
          mm("opm.proj_a", DN, cm, 32), mm("opm.proj_b", DN, cm, 32),
          mm("opm.outer", N * 32, D, N * 32),          # contraction is the MSA depth D=35 -> 64
          mm("opm.proj_o", NN, 32 * 32, cz)]
    return L


def dit_token(N=512, dim=768, heads=16, exp=2):
    hd, di = dim // heads, exp * dim
    return [mm("adaln.s_scale", N, dim, dim), mm("adaln.s_bias", N, dim, dim),
            mm("attn.q", N, dim, dim), mm("attn.k", N, dim, dim), mm("attn.v", N, dim, dim),
            mm("attn.g", N, dim, dim),
            mm("attn.qk", N, hd, N, cnt=heads),        # head_dim 48, pads to 64
            mm("attn.av", N, N, hd, cnt=heads),
            mm("attn.o", N, dim, dim), mm("out_proj", N, dim, dim),
            mm("tr.adaln.s_scale", N, dim, dim), mm("tr.adaln.s_bias", N, dim, dim),
            mm("tr.swish_gate", N, dim, 2 * di), mm("tr.a_to_b", N, dim, di),
            mm("tr.b_to_a", N, di, dim), mm("tr.out_proj", N, dim, dim)]


def dit_atom(NW=224, W=32, H=32, dim=128, heads=4, exp=2):
    """H is the key window. The census passed H=32; the model ships H=128."""
    hd, di, R = dim // heads, exp * dim, NW * W
    K = NW * H
    return [mm("adaln.s_scale", R, dim, dim), mm("adaln.s_bias", R, dim, dim),
            mm("attn.q", R, dim, dim),
            mm("attn.k", K, dim, dim), mm("attn.v", K, dim, dim),   # run on the KEY window
            mm("attn.g", R, dim, dim),
            mm("attn.qk", W, hd, H, cnt=NW * heads),
            mm("attn.av", W, H, hd, cnt=NW * heads),
            mm("attn.o", R, dim, dim), mm("out_proj", R, dim, dim),
            mm("tr.adaln.s_scale", R, dim, dim), mm("tr.adaln.s_bias", R, dim, dim),
            mm("tr.swish_gate", R, dim, 2 * di), mm("tr.a_to_b", R, dim, di),
            mm("tr.b_to_a", R, di, dim), mm("tr.out_proj", R, dim, dim)]


CENSUS = json.loads((Path(__file__).resolve().parents[1] /
                     "bioir_roofline" / "flops_bytes_512.json").read_text())
BY_PHASE = {r["phase"]: r for r in CENSUS["rows"]}

PHASES = [
    ("pairformer_block", pairformer(), pairformer(), 256),
    ("msa_block", msa(), msa(), 16),
    ("confidence_pairformer_block", pairformer(), pairformer(), 8),
    ("diffusion_token_layer", dit_token(), dit_token(), 4800),
    ("atom_transformer_layer", dit_atom(H=32), dit_atom(H=128), 1200),
]

print(f"{'phase':32s} {'census G':>10s} {'mine G':>10s} {'ratio':>8s} "
      f"{'padded G':>10s} {'pad x':>7s} {'true G':>10s} {'true x':>7s}")
tot = defaultdict(float)
for name, as_counted, as_shipped, calls in PHASES:
    c = BY_PHASE[name]["per_call_flops"]
    lg, pd = flops(as_counted), flops(as_counted, pad=True)
    tr, tr_pd = flops(as_shipped), flops(as_shipped, pad=True)
    print(f"{name:32s} {c/1e9:10.3f} {lg/1e9:10.3f} {lg/c:8.5f} "
          f"{pd/1e9:10.3f} {pd/lg:7.4f} {tr/1e9:10.3f} {tr/lg:7.4f}")
    tot["census"] += c * calls
    tot["logical"] += lg * calls
    tot["padded"] += pd * calls
    tot["true"] += tr * calls
    tot["true_padded"] += tr_pd * calls

print()
print(f"fold census   {CENSUS['fold_flops']/1e12:9.3f} TFLOP")
print(f"fold logical  {tot['logical']/1e12:9.3f} TFLOP   (mine / census = "
      f"{tot['logical']/CENSUS['fold_flops']:.5f})")
print(f"fold padded   {tot['padded']/1e12:9.3f} TFLOP   padding factor "
      f"{tot['padded']/tot['logical']:.4f}x")
print(f"fold true     {tot['true']/1e12:9.3f} TFLOP   key-window factor "
      f"{tot['true']/tot['logical']:.4f}x")
print(f"fold true+pad {tot['true_padded']/1e12:9.3f} TFLOP   combined "
      f"{tot['true_padded']/tot['logical']:.4f}x")

ROOF = 85.96e12
CELL = 17.34
print()
for k in ("census", "padded", "true", "true_padded"):
    v = tot[k] if k != "census" else CENSUS["fold_flops"]
    print(f"  compute floor, {k:12s} {v/ROOF:6.3f} s   = {100*v/ROOF/CELL:5.1f} % of the "
          f"{CELL} s cell")
