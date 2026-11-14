"""RFD4-Proteina autoencoder on the device: encoder (structure -> 32-d local latent) and decoder
(latent + CA -> all-atom coordinates and 147-class residue logits).

Each half is the plain 12-block transformer `ae_fold` reduces the checkpoint to. The device runs the
blocks and the fused output head; the host builds the input features, projects them to the 768-d
stream in fp32, and does the head post-processing (`ae_fold.decode_heads` / `finish_decode`), so the
residue masking, argmax and atom masks are upstream's own arithmetic in fp32.

The input projection and the final LayerNorm + output head stay on the host. The head is N x 768 x 295,
and on the host it is exact; on the card it added a small common-mode logit error (every token's final
stream is nearly the same vector, so the result rounding repeats for every token).
The input projection because the card's matmul accumulates to about 2^-11 of the
summed |terms|, and the features mix absolute CA coordinates (nm) with a unit-scale 32-d latent: on
the device the coordinate terms swamp the latent's contribution. It is N x 214 x 768 (decoder) or
N x 1587 x 768 (encoder), milliseconds on the host, and the upload shrinks to 768 columns.

Precision: bf16 weights, an fp32 residual stream, fp32 attention scores with a HiFi4 fp32 softmax.
The token axis buckets to 32 (`token_axis`); padded and in-context keys carry upstream's additive
-1e4, so a valid row never reads a masked one and the stack needs no per-layer mask multiply (masked
rows stay finite and are dropped by the emit mask after the head).
"""

from __future__ import annotations

import torch
import ttnn

from ..tenstorrent import Module
from ..token_axis import TILE, bucketed_width
from . import ae_fold as AF

_PRECISE = dict(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)


def _ckc():
    return ttnn.WormholeComputeKernelConfig(**_PRECISE)


class AEHalf(Module):
    """One folded AE half (`ae_fold.fold(sd, part)`) resident on the device."""

    def __init__(self, folded: dict):
        super().__init__({}, _ckc())
        dev = self.device
        w = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.float().contiguous(), dtype=dt,  # noqa: E731
                                                       layout=ttnn.TILE_LAYOUT, device=dev)
        row = lambda t: w(t.reshape(1, -1), ttnn.float32)  # noqa: E731  (a linear's bias)
        vec = row  # a layer norm's affine, [1, D] fp32 like a bias
        self.ones = w(torch.ones(TILE, 1))  # the last 32-term reduction of `_row_mean`
        self.in_w = folded["in_w"].float().T.contiguous()  # host fp32, see the module docstring
        self.n_feat = self.in_w.shape[0]
        self.layers = []
        hw = lambda t: t.float().T.contiguous()  # noqa: E731  (host fp32 copy for the common rows)
        for L in folded["layers"]:
            q_w, k_w, v_w = L["qkv_w"].chunk(3)
            q_b, k_b, v_b = L["qkv_b"].chunk(3)
            w1a, w1g = L["w1"].chunk(2)  # upstream chunks [a | gate]; two projections keep each tile-aligned
            # Each linear runs twice over: the device takes the per-token part (x - c), the host the
            # token-common row c @ W + b in fp32, see `_split_common`.
            mats = {"q": (q_w, q_b), "k": (k_w, k_b), "v": (v_w, v_b), "g": (L["g_w"], L["g_b"]),
                    "o": (L["o_w"], L["o_b"]), "w1a": (w1a, None), "w1g": (w1g, None), "w2": (L["w2"], None)}
            layer = {
                "ln1_w": vec(L["ln1_w"]), "ln1_b": vec(L["ln1_b"]),
                "node_w": vec(L["node_w"]), "node_b": vec(L["node_b"]),
                "q_ln_w": vec(L["q_ln_w"]), "q_ln_b": vec(L["q_ln_b"]),
                "k_ln_w": vec(L["k_ln_w"]), "k_ln_b": vec(L["k_ln_b"]),
                "ln2_w": vec(L["ln2_w"]), "ln2_b": vec(L["ln2_b"]),
            }
            for n, (wt, bt) in mats.items():
                layer[n] = w(wt.T)
                layer[n + "_h"] = (hw(wt), None if bt is None else bt.float())
            self.layers.append(layer)
        self.head_w = folded["head_w"].float().T.contiguous()  # host fp32, see the module docstring
        self.head_b = folded["head_b"].float()

    def _lin32(self, x, w, b=None, activation=None):
        return self._lin(x, w, bias=b, dtype=ttnn.float32, activation=activation)

    def _row_mean(self, x):
        """Mean over the last (768) dim, [.., D] fp32 -> [.., 1], to fp32 rather than FPU accuracy.

        The card's FPU accumulates to about 2^-11 of the summed |terms| (`ttnn.mean`, `layer_norm` and
        a matmul against ones all read 5e-4..3e-3), while fp32 elementwise adds are exact. So the 24
        column tiles fold to one with elementwise adds, and only the last 32 terms go through a matmul,
        split into exact bf16 hi and lo parts."""
        d = x.shape[-1]
        while d > TILE and d % (2 * TILE) == 0:
            d //= 2
            x = ttnn.add(x[..., :d], x[..., d:])
        acc = x[..., :TILE]
        for i in range(1, d // TILE):
            acc = ttnn.add(acc, x[..., i * TILE:(i + 1) * TILE])
        hi = ttnn.typecast(acc, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(acc, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
        ckc = self.compute_kernel_config
        s = ttnn.add(ttnn.matmul(hi, self.ones, dtype=ttnn.float32, compute_kernel_config=ckc),
                     ttnn.matmul(lo, self.ones, dtype=ttnn.float32, compute_kernel_config=ckc))
        return ttnn.multiply(s, 1.0 / AF.D)

    def _ln(self, x, w=None, b=None):
        """LayerNorm with `_row_mean` statistics: `ttnn.layer_norm` on fp32 is off by ~2.5e-3 (Welford
        too), and it is the largest single loss in the stack (perf/rfd4/vae/debug_block.py)."""
        xc = ttnn.subtract(x, self._row_mean(x))
        y = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(self._row_mean(ttnn.multiply(xc, xc)), AF.LN_EPS)))
        if w is not None:
            y = ttnn.multiply(y, w)
        if b is not None:
            y = ttnn.add(y, b)
        return y

    def _split_common(self, x, kmean):
        """x -> (x - c on the device, c on the host) with c the mean attended row, [B, 1, D].

        Every token's stream is nearly the same vector (a real batch's final stream: token-common part
        27.50 of a 27.71 row norm), so a matmul over it rounds the same way for every token, and the
        card keeps 10 mantissa bits of a result: the error does not average out, it is a common-mode
        shift (logit25 - logit23 moved -0.14 on all 240 RNA tokens of real_rna_pk). W x = W (x - c) + W c
        holds for ANY c, so the device multiplies the small per-token part and the host adds W c in fp32."""
        c = ttnn.matmul(kmean, x, dtype=ttnn.float32, compute_kernel_config=self.compute_kernel_config)
        ch = ttnn.to_torch(c).float()
        xc = ttnn.subtract(x, c)
        ttnn.deallocate(c)
        return xc, ch

    def _row(self, ch, wb):
        """The host's token-common row c @ W (+ b), fp32, on the device as [B, 1, out]."""
        w, b = wb
        r = ch @ w if b is None else ch @ w + b
        return ttnn.from_torch(r.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _lin_cm(self, xc, ch, L, n, row=True):
        y = self._lin32(xc, L[n])
        return ttnn.add(y, self._row(ch, L[n + "_h"])) if row else y

    def _block(self, L, x, bias, kmean):
        ckc = self.compute_kernel_config
        h1 = self._ln(x, L["ln1_w"], L["ln1_b"])  # the folded adaLN
        h0 = self._ln(h1, L["node_w"], L["node_b"])
        ttnn.deallocate(h1)
        h, ch = self._split_common(h0, kmean)
        ttnn.deallocate(h0)
        # qk-LayerNorm runs on the full 768 inner dim, before the heads are split, so the three
        # projections stay separate and are re-packed for `_split_heads` -- the tile-aware split
        # (`nlp_create_qkv_heads`), never a reshape to [B, N, 12, 64]: 12 is not tile-aligned.
        q = self._ln(self._lin_cm(h, ch, L, "q"), L["q_ln_w"], L["q_ln_b"])
        k = self._ln(self._lin_cm(h, ch, L, "k"), L["k_ln_w"], L["k_ln_b"])
        # Subtract the mean attended key. Softmax is invariant to a per-row constant and q.(k - c) is
        # q.k - q.c for ANY c, so this is exact; it matters because the card's matmul keeps 10 mantissa
        # bits of its result. The keys share a large common part: on a real batch block 0's scores reach
        # |s| 824 with a row span of 5.2, so tf32 rounding alone moved them ~0.4; centred they reach 3.1.
        k = ttnn.subtract(k, ttnn.matmul(kmean, k, dtype=ttnn.float32, compute_kernel_config=ckc))
        # Softmax rows sum to 1, so p @ v = p @ (v - v_row) + v_row: attention averages only the
        # per-token part of v and the common row is added back after the heads merge.
        v = self._lin_cm(h, ch, L, "v", row=False)
        qkv = ttnn.concat([q, k, v], dim=-1)
        for t in (q, k, v):
            ttnn.deallocate(t)
        q, k, v = self._split_heads(qkv, AF.HEADS)
        kt = ttnn.permute(k, (0, 1, 3, 2))
        ttnn.deallocate(k)
        s = ttnn.matmul(q, kt, dtype=ttnn.float32, compute_kernel_config=ckc)
        ttnn.deallocate(q); ttnn.deallocate(kt)
        s = ttnn.add(ttnn.multiply(s, AF.DH ** -0.5), bias)
        p = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
        ttnn.deallocate(s)
        o = ttnn.matmul(p, v, dtype=ttnn.float32, compute_kernel_config=ckc)
        ttnn.deallocate(p); ttnn.deallocate(v)
        o = ttnn.add(self._merge_heads(o), self._row(ch, L["v_h"]))
        g = ttnn.sigmoid(self._lin_cm(h, ch, L, "g"))
        ttnn.deallocate(h)
        og0 = ttnn.multiply(o, g)
        ttnn.deallocate(o); ttnn.deallocate(g)
        og, cog = self._split_common(og0, kmean)
        ttnn.deallocate(og0)
        x2 = ttnn.add(x, self._lin_cm(og, cog, L, "o"))
        ttnn.deallocate(og)

        t0 = self._ln(x2, L["ln2_w"], L["ln2_b"])
        t, ct = self._split_common(t0, kmean)
        ttnn.deallocate(t0)
        u = self._lin_cm(t, ct, L, "w1a")
        gate = ttnn.silu(self._lin_cm(t, ct, L, "w1g"))
        ttnn.deallocate(t)
        ug0 = ttnn.multiply(u, gate)
        ttnn.deallocate(u); ttnn.deallocate(gate)
        ug, cug = self._split_common(ug0, kmean)
        ttnn.deallocate(ug0)
        out = ttnn.add(x2, self._lin_cm(ug, cug, L, "w2"))
        ttnn.deallocate(ug); ttnn.deallocate(x2)
        return out

    def __call__(self, feats: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
        """`feats` [B, N, n_feat], `attend` [B, N] bool -> raw head output [B, N, n_head] fp32 (host).

        Pads the token axis to the 32-bucket; padded tokens are masked keys, so the valid rows equal
        the unpadded forward."""
        B, N, F = feats.shape
        assert F == self.n_feat, (F, self.n_feat)
        Np = bucketed_width(N, TILE)
        f = torch.zeros(B, Np, AF.D, dtype=torch.float32)
        f[:, :N] = (feats.float() @ self.in_w) * attend[..., None]
        att = torch.zeros(B, Np, dtype=torch.bool)
        att[:, :N] = attend
        dev = self.device
        bias = AF.key_bias(att, torch.float32).expand(B, AF.HEADS, Np, Np).contiguous()
        bias = ttnn.from_torch(bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        kmean = (att / att.sum(-1, keepdim=True).clamp_min(1)).float()[:, None, :]  # [B, 1, Np]
        kmean = ttnn.from_torch(kmean, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        x = ttnn.from_torch(f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        for L in self.layers:
            x = self._block(L, x, bias, kmean)
        ttnn.deallocate(bias)
        ttnn.deallocate(kmean)
        out = ttnn.to_torch(x).float()[:, :N]
        ttnn.deallocate(x)
        return AF._ln(out) @ self.head_w + self.head_b


class RFD4AutoEncoder:
    """The sampler's view of the AE: `encode` a structure batch, `decode` a latent batch.

    Batches are upstream's (the keys `ae_fold.encoder_features` / `decoder_features` read), any
    number of designs along dim 0."""

    def __init__(self, state_dict: dict, parts=("encoder", "decoder")):
        self.tab = AF.tables()
        self.folded = {p: AF.fold(state_dict, p, torch.float32) for p in parts}
        self.dev = {p: AEHalf(self.folded[p]) for p in parts}
        #: Designs the decoder ran a forward on, counted per call (`DesignEngine.counted`).
        self.decoded = 0

    @classmethod
    def from_checkpoint(cls, path: str, parts=("encoder", "decoder")):
        # The released .ckpt is a Lightning pickle with an omegaconf config, as upstream loads it.
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        return cls(sd, parts)

    def encode(self, batch: dict, noise: torch.Tensor | None = None) -> dict:
        attend, emit = AF.ae_masks(batch)
        raw = self.dev["encoder"](AF.encoder_features(batch, self.folded["encoder"]["emb"]), attend)
        raw = raw * emit[..., None]
        mean, log_scale = raw.chunk(2, -1)
        z = mean if noise is None else mean + noise.float() * torch.exp(log_scale)
        return {"mean": mean, "log_scale": log_scale, "z_latent": z * emit[..., None], "emit_mask": emit}

    def decode_raw(self, batch: dict) -> dict:
        """`DecoderTransformer.forward`'s outputs."""
        attend, emit = AF.ae_masks(batch)
        feats = AF.decoder_features(batch, self.folded["decoder"]["emb"], self.tab["mask_token"])
        out = self.dev["decoder"](feats, attend)
        self.decoded += int(feats.shape[0])
        return AF.decode_heads(out, batch, emit, self.tab)

    def decode(self, batch: dict, required_atom_mask=None) -> dict:
        """`AutoEncoder.decode`: coordinates (nm), residue type and atom mask per token."""
        return AF.finish_decode(self.decode_raw(batch), batch, self.tab, required_atom_mask)
