#!/usr/bin/env python3
"""Is the windowed atom-pair track bit-exact against the dense one it replaces?

The atom transformer RF3 feeds from the atom-level pair track is 32-query / 128-key
local attention, so of the L_atom^2 pairs the port used to build it reads L_atom x 128.
Every operation on that track is elementwise in the pair index or a matmul over the
channel axis only, so restricting it to the pairs that are read should be the same
arithmetic on the same elements. Should be: this scores it.

Arm A is a frozen copy of the pre-lever code -- the dense `pair_terms`, `windowed_bias`
and `_trunk_pair`, mixed into the shipped classes so everything else is literally the
same objects and the same weights. Arm B is the shipped path. Both arms run in ONE
process against ONE checkpoint load, and arm A runs twice, because a cross-run
comparison on Blackhole cannot tell a real divergence from run-to-run drift
(`protenix-v2-blackhole-nondeterminism-256aa`).

    A0 vs A1  -> is this fold deterministic on this card at all?
    A0 vs B   -> does the window change any number?

Arm A0 records the sampler's RNG stream and the other two arms replay it. Without that
the three arms denoise different noise and every comparison is meaningless -- measured:
arm A against itself came back with a max coordinate difference of 4641 A
(`diffusion-port-parity-shared-draws`).

Reported per rung on the final coordinates, the distogram and the trunk tensors, as an
exact-equality count, not a tolerance.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ arm A, frozen
# Copies of the three pre-lever bodies, kept here rather than behind a flag in the
# model: they are the thing being replaced, not an option the model still offers.

def dense_windowed_bias(p, ln0_w, ln0_b, to_b, mask, n_pad, ckc, device):
    import ttnn
    from tt_bio.rf3.atom_encoder import N_HEADS
    from tt_bio.rf3.atom_encoder_host import ATOM_KEYS, ATOM_WINDOW, PAD_LEFT
    b = ttnn.layer_norm(p, weight=ln0_w, bias=ln0_b, epsilon=1e-5,
                        compute_kernel_config=ckc)
    b = ttnn.linear(b, to_b, compute_kernel_config=ckc)
    b = ttnn.permute(b, (0, 3, 1, 2))
    k = n_pad // ATOM_WINDOW
    blocks = []
    for j in range(k):
        lo, hi = j * ATOM_WINDOW - PAD_LEFT, j * ATOM_WINDOW - PAD_LEFT + ATOM_KEYS
        lo_c, hi_c = max(lo, 0), min(hi, n_pad)
        piece = b[:, :, j * ATOM_WINDOW:(j + 1) * ATOM_WINDOW, lo_c:hi_c]
        pads = []
        if lo_c > lo:
            pads.append(ttnn.zeros((1, N_HEADS, ATOM_WINDOW, lo_c - lo),
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   dtype=piece.dtype))
        pads.append(piece)
        if hi > hi_c:
            pads.append(ttnn.zeros((1, N_HEADS, ATOM_WINDOW, hi - hi_c),
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   dtype=piece.dtype))
        blocks.append(ttnn.concat(pads, dim=-1) if len(pads) > 1 else piece)
    w = ttnn.concat(blocks, dim=0)
    w = ttnn.add(w, mask)
    return ttnn.reshape(w, (1, k * N_HEADS, ATOM_WINDOW, ATOM_KEYS))


class DensePair:
    """Mixed in ahead of the shipped class, so only these three bodies change."""

    def pair_terms(self, p, c, keys_indexing, n_pad):
        import ttnn
        rc = ttnn.relu(c)
        sl = ttnn.linear(rc, self.sl, compute_kernel_config=self.compute_kernel_config)
        sm = ttnn.linear(rc, self.sm, compute_kernel_config=self.compute_kernel_config)
        p = ttnn.add(p, ttnn.unsqueeze(sl, -2))
        p = ttnn.add(p, ttnn.unsqueeze(sm, -3))
        m = p
        for w in self.mlp:
            m = ttnn.linear(ttnn.relu(m), w,
                            compute_kernel_config=self.compute_kernel_config)
        return ttnn.add(p, m)

    def bias(self, p, i, mask, n_pad):
        return dense_windowed_bias(p, self.ln0_w[i], self.ln0_b[i], self.to_b[i], mask,
                                   n_pad, self.compute_kernel_config, self.device)

    def _trunk_pair(self, z, a2t, a2t_t, n_pad):
        import ttnn
        p = ttnn.layer_norm(z, weight=self.z_norm_w, bias=self.z_norm_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        p = ttnn.linear(p, self.z_w, compute_kernel_config=self.compute_kernel_config)
        _, i_tok, _, c_pair = p.shape
        l_atom = a2t.shape[1]
        g = ttnn.matmul(a2t, ttnn.reshape(p, (1, i_tok, i_tok * c_pair)),
                        compute_kernel_config=self.compute_kernel_config)
        g = ttnn.reshape(g, (1, l_atom, i_tok, c_pair))
        g = ttnn.permute(g, (0, 1, 3, 2))
        g = ttnn.reshape(g, (1, l_atom * c_pair, i_tok))
        g = ttnn.matmul(g, a2t_t, compute_kernel_config=self.compute_kernel_config)
        g = ttnn.reshape(g, (1, l_atom, c_pair, l_atom))
        return ttnn.permute(g, (0, 1, 3, 2))


class DenseBias:
    """The decoder only reads the pair track through the bias."""

    def bias(self, p, i, mask, n_pad):
        return dense_windowed_bias(p, self.ln0_w[i], self.ln0_b[i], self.to_b[i], mask,
                                   n_pad, self.compute_kernel_config, self.device)


def dense_host(host, f):
    """The pre-lever host tensors: dense `pair_in` / `pair_v`, transposed one-hot."""
    import ttnn
    from tt_bio.rf3.atom_encoder_host import pair_inputs
    from tt_bio.rf3.host import C_PAIR_IN, to_device

    L, Lp = host.n_atom, host.n_atom_padded
    device = host.single_in.device()
    p_raw, v_raw = pair_inputs(f, L)
    p_in = torch.zeros(1, Lp, Lp, C_PAIR_IN)
    p_in[0, :L, :L, :p_raw.shape[-1]] = p_raw
    v_in = torch.zeros(1, Lp, Lp, 1)
    v_in[0, :L, :L] = v_raw
    a2t = torch.Tensor(ttnn.to_torch(host.atom_to_token)).float()
    return dataclasses.replace(
        host, pair_in=to_device(p_in, device), pair_v=to_device(v_in, device),
        token_to_atom_win=to_device(a2t.transpose(1, 2).contiguous(), device))


def set_arm(tt, dense: bool):
    """Retype the three pair-track owners in place; weights and buffers untouched."""
    from tt_bio.rf3.atom_encoder import AtomAttentionEncoder
    from tt_bio.rf3.diffusion_atom_decoder import DiffusionAtomDecoder
    from tt_bio.rf3.diffusion_atom_encoder import DiffusionAtomEncoder

    pairs = [(tt.feature_initializer.encoder, AtomAttentionEncoder, DensePair),
             (tt.diffusion_module.encoder, DiffusionAtomEncoder, DensePair),
             (tt.diffusion_module.decoder, DiffusionAtomDecoder, DenseBias)]
    for obj, base, mixin in pairs:
        obj.__class__ = (type(f"Dense{base.__name__}", (mixin, base), {}) if dense
                         else base)


# ------------------------------------------------------------------------- scoring

def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    eq = torch.eq(a, b)
    d = (a - b).abs()
    den = a.abs().max().item() or 1.0
    return {"n": int(a.numel()), "exact": int(eq.sum()),
            "bit_exact": bool(eq.all()),
            "max_abs_diff": float(d.max()), "rel_max": float(d.max() / den)}


def one_fold(tt, host, n_recycles, num_steps, batch, draws):
    import ttnn
    s_inputs, s, z = tt.trunk(host, n_recycles)
    disto = torch.Tensor(ttnn.to_torch(tt.distogram_head(z))).float()
    tt.sampler.num_timesteps = num_steps
    coord = torch.zeros(batch, host.n_atom, 3)
    x_pred, _ = tt.sampler.sample(
        lambda x, t: tt.diffusion_module(host, x, t, s_inputs, s, z),
        coord, batch, draws=draws)
    return {"s_inputs": torch.Tensor(ttnn.to_torch(s_inputs)).float(),
            "z": torch.Tensor(ttnn.to_torch(z)).float(),
            "distogram": disto, "coords": x_pred}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=128)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=3)
    ap.add_argument("--diffusion_batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=args.n_recycles,
                   diffusion_batch_size=args.diffusion_batch_size, seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps, with_confidence=False)

    from tt_bio.rf3.sampler import Draws

    host_win = HostInputs.build(f, device)
    host_den = dense_host(host_win, f)
    runs, rec = {}, Draws()
    for name, dense in (("A0", True), ("B", False), ("A1", True)):
        set_arm(tt, dense)
        runs[name] = one_fold(tt, host_den if dense else host_win,
                              args.n_recycles, args.num_steps,
                              args.diffusion_batch_size,
                              rec if name == "A0" else Draws(rec.values))
        print(f"  ran {name} ({'dense' if dense else 'windowed'})", flush=True)
    set_arm(tt, False)

    keys = ("s_inputs", "z", "distogram", "coords")
    rep = {"aa": args.aa, "n_atom": host_win.n_atom,
           "n_atom_padded": host_win.n_atom_padded, "n_token": host_win.n_token,
           "n_recycles": args.n_recycles, "num_steps": args.num_steps,
           "dense_pairs": host_win.n_atom_padded ** 2,
           "windowed_pairs": host_win.n_atom_padded * 128,
           "A0_vs_A1": {k: compare(runs["A0"][k], runs["A1"][k]) for k in keys},
           "A0_vs_B": {k: compare(runs["A0"][k], runs["B"][k]) for k in keys}}
    rep["deterministic"] = all(v["bit_exact"] for v in rep["A0_vs_A1"].values())
    rep["bit_exact"] = all(v["bit_exact"] for v in rep["A0_vs_B"].values())
    print(json.dumps(rep, indent=2))
    print(f"\n{args.aa} aa: deterministic={rep['deterministic']}  "
          f"windowed==dense={rep['bit_exact']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
