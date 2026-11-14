"""OF3 AtomTransformer device port (P7).

OF3's ``atom_transformer`` is a ``DiffusionTransformer(cross_attention_mode=True)``:
3 blocks of windowed sequence-local atom attention (n_query=32, n_key=128, 4 heads,
head_dim=32). Each block: AdaLN-conditioned q/kv -> pair-biased windowed attention
with a query gate (``mha.linear_g``) -> ``sigmoid(linear_ada_out(s))`` output gate ->
residual -> AdaLN-conditioned SwiGLU transition (``swiglu.linear_a/b`` + ``linear_out``,
``sigmoid(linear_g(s))`` zero gate) -> residual. The pair ``plm`` is layer-normed once
at the top (``layer_norm_z``) and a per-block ``linear_z`` projects the pair bias.

The block topology is OF3-specific (``linear_ada_out`` output gate, ``mha.linear_g``
query gate, ``swiglu.linear_a/b`` + ``linear_out`` transition) and is NOT a key-remap
onto ``protenix.AtomTransformer`` -- this is a fresh device port. The shared ``AdaLN``
math (a = LN(a); a = sigmoid(Wg(LN_s(s))) * a + Ws(LN_s(s))) is identical, so the nine
conditioning submodules reuse ``tenstorrent.AdaLN`` via ``remap_of3_adaln``.

The mask-derived block gather (OF3 ``convert_single_rep_to_blocks``: centered key
windows with underflow/overflow shift) is precomputed on host (``key_block_idxs``,
``invalid_mask``, ``mask_trunked``) and replayed on device via ``ttnn.embedding`` --
the device re-blocks the evolving single ``a`` every block with the fixed gather
indices, so the port is reusable: the diffusion atom encoder/decoder use the same class
with ``a != s``. See ``scripts/of3_atom_transformer_golden.py`` for the golden and
``tests/test_openfold3_atom_transformer.py`` for the PCC gate.
"""
from __future__ import annotations

import torch
import ttnn

from . import tenstorrent as _T
from . import ops
from .tenstorrent import (Module, AdaLN, CORE_GRID_MAIN, _dtype, _cached, batched_matmul,
                          host_f64_softmax_site, site_softmax, softmax_ckc)
from .eltwise_fusion import gated_add, scale_add, mask_add, _taping
from .envflags import env_flag

# Where the superset bias is built. Host by default: the card path builds a bit-exact bias (superset_check.py) yet
# moves the Wormhole fold digest, unexplained so far (perf/spd_gap_of3/superset_fold_check.py);
# TT_BIO_OF3_SUPERSET_HOST=0 runs it.
_SUPERSET_HOST = env_flag("TT_BIO_OF3_SUPERSET_HOST", True)


def remap_of3_adaln(sd: dict) -> dict:
    """OF3 AdaLN keys -> tenstorrent.AdaLN keys.

    OF3 AdaLN: a = LayerNorm(a); g = sigmoid(linear_g(LayerNorm_s(s)));
    a = g * a + linear_s(LayerNorm_s(s)). ``LayerNorm(c_s, create_offset=False)`` is
    weight-only, matching tenstorrent.AdaLN's bias-free s layer norm; the a layer norm
    is weightless on both sides. ``linear_s`` is bias-free in the OF3 checkpoint, so
    s_bias (the shift term) carries no bias -- exactly tenstorrent.AdaLN's layout.
    """
    return {
        "s_norm.weight": sd["layer_norm_s.weight"],
        "s_scale.weight": sd["linear_g.weight"],
        "s_scale.bias": sd["linear_g.bias"],
        "s_bias.weight": sd["linear_s.weight"],
    }


def sample_offset_index(idx_tt, S: int, stride: int):
    """A gather index ``[1, n]`` -> ``[1, S*n]`` addressing a table of S stacked copies.

    The atom-level gathers read rows of a 2-D table; S samples stacked on the leading dim
    flatten to a table S times as tall, so copy k of the index is shifted by ``k * stride``
    rows. Built once per rollout from the device index (a few hundred KB), never per step."""
    idx = ttnn.to_torch(idx_tt).to(torch.int64)
    out = torch.cat([idx + k * stride for k in range(S)], dim=-1).to(torch.int32)
    return ttnn.from_torch(out, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=idx_tt.device())


def _sub(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class OF3AtomTransformer(Module):
    N_HEADS = 4
    HEAD_DIM = 32
    N_QUERY = 32
    N_KEY = 128

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self._act_dtype = _dtype(ttnn.bfloat16)
        self._w = {k: v for k, v in self.weights.data.items()}
        self._wc: dict = {}
        # OFF, measured rather than untried: the scores reach this site in bf16, where the
        # config buys 1.03x accuracy for 1.36x cost at its own [1,33,4,32,128]. bf16 storage is
        # the floor and the config has nowhere to accumulate under it.
        self._softmax_ckc = softmax_ckc("openfold3.atom_transformer")
        self._softmax_f64 = host_f64_softmax_site("openfold3.atom_transformer")
        self.ln_z_w = self._w_tt("layer_norm_z.weight", False)
        self.adaln_q = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.attention_pair_bias.layer_norm_a_q.")),
                  compute_kernel_config) for b in range(3)]
        self.adaln_k = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.attention_pair_bias.layer_norm_a_k.")),
                  compute_kernel_config) for b in range(3)]
        self.adaln_t = [
            AdaLN(False, remap_of3_adaln(_sub(self._w, f"blocks.{b}.conditioned_transition.layer_norm.")),
                  compute_kernel_config) for b in range(3)]
        # The windowed attention as one SDPA program over a superset key window, Protenix's
        # atom_sdpa32 recipe and the same switch (TT_BIO_ATOM_SDPA32). See `_superset`. OF3's atom
        # activations are bf16, so unlike Protenix the switch covers bf16 too: at atom_sdpa32's
        # exact config (HiFi4, exact exp, fp32 accumulation) by default, at atom_sdpa's under that
        # lever, as Protenix's bf16 site runs.
        from .protenix import _ATOM_SDPA32, _ATOM_SDPA32_CKC, _ATOM_SDPA_CKC
        self._sdpa32 = _ATOM_SDPA32
        self._sdpa32_ckc = (_ATOM_SDPA_CKC if _T.lever("atom_sdpa") and self._act_dtype == ttnn.bfloat16
                            else _ATOM_SDPA32_CKC)
        self._zeros: dict = {}
        # atom_k1 (Protenix's lever, same switch): the fp32 linears on k1_linear's program, one K
        # tile per dest pass on Wormhole, same formats.
        self._k1 = _T.lever("atom_k1") and self._act_dtype == ttnn.float32

    #: Key fragments of the three AdaLNs per block, which upload their own weights.
    _ADALN = (".layer_norm_a_q.", ".layer_norm_a_k.", ".conditioned_transition.layer_norm.")

    def materialize_device_weights(self) -> int:
        """Upload now every weight `__call__` uploads lazily through `_w_tt`; the count.

        For training only. A weight minted inside the first forward comes after the walk
        that registers parameters, so it trains as a constant (D256). Inference keeps the
        lazy upload. Everything in a block that the AdaLNs do not own is a `_lin` operand,
        matrices transposed and biases not: 14 tensors per block.
        """
        n = len(self._wc)
        for key, w in self._w.items():
            if key.startswith("blocks.") and not any(a in key for a in self._ADALN):
                self._w_tt(key, w.dim() == 2)
        return len(self._wc) - n

    def _w_tt(self, key, transpose=True):
        v = self._wc.get((key, transpose))
        if v is None:
            if key.endswith(".linear_kv.weight"):
                # The superset path's fused K|V projection, built on first use. Inference only:
                # its SDPA has no backward, so no training walk ever meets this weight.
                m = key[:-len("kv.weight")]
                w = torch.cat([self._w[m + "k.weight"], self._w[m + "v.weight"]], 0)
            else:
                w = self._w[key]
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.device, dtype=self._act_dtype)
            self._wc[(key, transpose)] = v
        return v

    def _lin(self, x, wkey, bkey=None, activation=None, memory_config=None):
        w = self._w_tt(wkey)
        bias = self._w_tt(bkey, False) if bkey else None
        # One K tile leaves the program no K blocking to choose: those keep the shared path, and a
        # tape takes ops.linear.
        if self._k1 and int(w.padded_shape[0]) > 32 and not _taping():
            return _T.k1_linear(x, w, bias, dtype=x.dtype, activation=activation,
                                compute_kernel_config=self.compute_kernel_config)
        return ops.linear(x, w, bias=bias, activation=activation, memory_config=memory_config,
                          compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)

    def _heads(self, x, n_blk, n_seq):
        # x: [S, n_blk, n_seq, 128] -> [S, n_blk, H, n_seq, dh]
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], n_blk, n_seq, self.N_HEADS, self.HEAD_DIM))
        x = ttnn.permute(x, (0, 1, 3, 2, 4))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _gather_keys(self, x, key_block_idxs_tt, valid_mask, nb):
        # x: [S, NP, 128] -> [S, nb, N_KEY, 128] via fixed host gather indices, which carry
        # the sample offsets when S > 1 (see `sample_offset_index`).
        S = x.shape[0]
        src = x
        if self._act_dtype != ttnn.bfloat16:
            src = ttnn.typecast(x, ttnn.bfloat16)
        x2d = ttnn.reshape(ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT), (S * x.shape[1], 128))
        if src is not x:
            ttnn.deallocate(src)
        xk = ttnn.embedding(key_block_idxs_tt, x2d, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xk = ttnn.reshape(xk, (S, nb, self.N_KEY, 128))
        xk = ttnn.to_layout(xk, ttnn.TILE_LAYOUT)
        if self._act_dtype != ttnn.bfloat16:
            xk = ttnn.typecast(xk, self._act_dtype)
        return ttnn.multiply(xk, valid_mask)

    def _invariants(self, s, z, key_block_idxs_tt, valid_mask, nb):
        """Everything in a block that is a pure function of (s, z).

        In the sampler rollout ``s`` is the atom conditioning ``cl`` and ``z`` the blocked
        pair ``plm``; both are loop invariants, so this whole group is computed once per
        rollout instead of once per step. The sampler owns the cache, so a cached tensor
        must never be deallocated by a consumer."""
        nq, nk, H = self.N_QUERY, self.N_KEY, self.N_HEADS
        z_ln = ttnn.layer_norm(z, weight=self.ln_z_w, epsilon=1e-5,
                               compute_kernel_config=self.compute_kernel_config)
        z_bias = []
        for b in range(3):
            zb = self._lin(z_ln, f"blocks.{b}.attention_pair_bias.linear_z.weight")
            zb = ttnn.to_layout(zb, ttnn.ROW_MAJOR_LAYOUT)
            zb = ttnn.permute(ttnn.reshape(zb, (1, nb, nq, nk, H)), (0, 1, 4, 2, 3))
            z_bias.append(ttnn.to_layout(zb, ttnn.TILE_LAYOUT))
        ttnn.deallocate(z_ln)
        # s blocks are fixed across blocks (s = cl does not evolve).
        s_q = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
        s_q = ttnn.reshape(s_q, (1, nb, nq, 128))
        s_q = ttnn.to_layout(s_q, ttnn.TILE_LAYOUT)
        s_k = self._gather_keys(s, key_block_idxs_tt, valid_mask, nb)
        ada_raw = [self._lin(s, f"blocks.{b}.attention_pair_bias.linear_ada_out.weight",
                             f"blocks.{b}.attention_pair_bias.linear_ada_out.bias")
                   for b in range(3)]
        cg_raw = [self._lin(s, f"blocks.{b}.conditioned_transition.linear_g.weight",
                            f"blocks.{b}.conditioned_transition.linear_g.bias")
                  for b in range(3)]
        # The AdaLN conditioning half reads s_q / s_k / s, all three invariant, so the nine
        # layer_norm + 2-linear groups are nine tensor pairs per rollout, not per call.
        ada_s = ()          # a cached None would break the sampler free walk
        if _T.ADALN_S_HOIST:
            ada_s = [(self._sig(self.adaln_q[b].s_terms(s_q)), self._sig(self.adaln_k[b].s_terms(s_k)),
                      self._sig(self.adaln_t[b].s_terms(s))) for b in range(3)]
        if self._sig_gates():
            ada_raw = [ttnn.sigmoid(g) for g in ada_raw]
            cg_raw = [ttnn.sigmoid(g) for g in cg_raw]
        return z_bias, s_q, s_k, ada_raw, cg_raw, ada_s

    def _sig_gates(self):
        """adaln_mod on an fp32 stack: every rollout-invariant gate and AdaLN scale is stored past
        its sigmoid, so each AdaLN is LN + one addcmul (`_SigTerms`), the attention residual one
        addcmul and the transition gate a plain multiply: 12 fewer ops and no per-call sigmoid in
        each of the rollout's 400 atom-transformer calls."""
        return self._act_dtype == ttnn.float32 and _T.lever("adaln_mod")

    def _gate_add(self, x, o, g):
        """``x + o * sigmoid(g)``, with `g` already past its sigmoid under `_sig_gates`. Frees `o`."""
        if self._sig_gates():
            out = gated_add(x, o, g)
        else:
            o2 = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            out = ttnn.add(x, o2)
            ttnn.deallocate(o2)
        ttnn.deallocate(o)
        return out

    def _sig(self, terms):
        if not self._sig_gates():
            return terms
        return _T._SigTerms((ttnn.sigmoid(terms[0]), terms[1]))

    def _transition(self, x, s, b, ada_s, cg_raw, atom_mask_col, mc=None):
        """AdaLN-conditioned SwiGLU transition with the sigmoid(linear_g(s)) zero gate, residual.
        `mc` (atom_l1): where the linears write; the elementwise ops follow their first input."""
        ct = f"blocks.{b}.conditioned_transition."
        an = self.adaln_t[b](x, s, s_terms=ada_s[b][2] if ada_s else None)
        b1 = self._lin(an, ct + "swiglu.linear_a.weight", activation="silu", memory_config=mc)
        b2 = self._lin(an, ct + "swiglu.linear_b.weight", memory_config=mc)
        ttnn.deallocate(an)
        bb = ttnn.multiply(b1, b2)
        ttnn.deallocate(b1); ttnn.deallocate(b2)
        out = self._lin(bb, ct + "linear_out.weight", memory_config=mc)
        ttnn.deallocate(bb)
        out = ttnn.multiply(out, cg_raw[b], **({} if self._sig_gates() else
                                               {"input_tensor_b_activations": [ttnn.UnaryOpType.SIGMOID]}))
        x = mask_add(x, out, atom_mask_col)
        ttnn.deallocate(out)
        return x

    #: Superset key window: block i reads frame rows [32 i, 32 i + W), i.e. atoms
    #: [32 i - lead, 32 i - lead + W), with W = lead + N_KEY. OF3 centres a 128-key window on the
    #: block and SHIFTS it back inside the real atoms at both ends (Protenix pads instead), so the
    #: last block's window ends at atom n_atom - 1: lead is 3 tiles when n_atom is a multiple of 32
    #: and 4 otherwise (`_superset` sizes it), and the frame carries lead zero rows in front and
    #: W - lead - N_QUERY behind. `_attend_superset` reads W off the bias.

    def _superset(self, z_bias, mask_bias, key_block_idxs_tt, s, nb):
        """Per block ``(bias, q_terms, k_terms)`` for the superset attention, or ``()`` when a
        key window is not a contiguous run of atoms inside the superset (then the gathered path
        runs). Built once per rollout (`_superset_host`, or `_superset_device` on the card): the bias of key ``j`` of
        block ``i`` moves to superset column ``ws_i - (32 i - lead) + j``, every other column
        is -1e9, and the whole is pre-multiplied by sqrt(dh) because the SDPA kernel scales the
        mask with the scores. The AdaLN conditioning terms are taken on the per-atom ``s``: the
        query and key rows are now per atom, not per window copy."""
        nq, nk, H, dh = self.N_QUERY, self.N_KEY, self.N_HEADS, self.HEAD_DIM
        idx = ttnn.to_torch(key_block_idxs_tt).reshape(nb, nk).long()
        ws = idx[:, 0]
        off = ws - torch.arange(nb) * nq                 # window start relative to the block
        if not torch.equal(idx, ws[:, None] + torch.arange(nk)) or int(off.max()) > 0:
            return ()
        lead = -(int(off.min()) // nq) * nq              # whole tiles holding the furthest start
        W = lead + nk
        off = off + lead
        out = []
        for b in range(3):
            bias = (self._superset_host(z_bias[b], mask_bias, off, W, nb) if _SUPERSET_HOST
                    else self._superset_device(z_bias[b], mask_bias, off, W, nb))
            terms = ((self._sig(self.adaln_q[b].s_terms(s)), self._sig(self.adaln_k[b].s_terms(s)))
                     if _T.ADALN_S_HOIST else ((), ()))   # () not None: the cache free walks it
            out.append((bias, *terms))
        return out

    def _superset_host(self, zb, mask_bias, off, W, nb):
        """One block's superset bias built on the host (the reference for `_superset_device`)."""
        nq, nk, H, dh = self.N_QUERY, self.N_KEY, self.N_HEADS, self.HEAD_DIM
        col = (off[:, None] + torch.arange(nk))[:, None, None, :].expand(nb, H, nq, nk)
        zb = (ttnn.to_torch(zb).float().reshape(nb, H, nq, nk)
              + ttnn.to_torch(mask_bias).float().reshape(nb, 1, nq, nk))
        full = torch.full((nb, H, nq, W), -1e9).scatter_(-1, col, zb)
        full = full.clamp(min=-1e4) * dh ** 0.5
        return _T.upload(full.permute(1, 0, 2, 3).reshape(1, H * nb, nq, W), self._act_dtype, self.device)

    def _superset_device(self, zb, mask_bias, off, W, nb):
        """`_superset_host` on the card, in fp32 like the host, so the bias never crosses the bus.

        Blocks sharing a window offset are contiguous runs (the interior plus a few at each end),
        so each run is one slice with its keys placed by a row-major concat between two constant
        fills; the offsets need not be tile multiples."""
        nq, nk, H, dh = self.N_QUERY, self.N_KEY, self.N_HEADS, self.HEAD_DIM
        dev, f32, RM = self.device, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT
        # mask_bias is 0 or -1e9, so the host's (zb + mb).clamp(-1e4) * sqrt(dh) is zb * sqrt(dh)
        # where the mask is 0 and the fill elsewhere. Written that way the only arithmetic is one
        # fp32 multiply: an fp32 add on the FPU rounds its operands to tf32 even when adding 0.
        fill = float(torch.tensor(-1e4, dtype=torch.float32) * dh ** 0.5)
        x = ttnn.multiply(ttnn.clamp(ttnn.typecast(ttnn.reshape(zb, (nb, H, nq, nk)), f32),
                                     min=-1e4, max=3e38), dh ** 0.5)
        m = ttnn.repeat(ttnn.typecast(ttnn.reshape(mask_bias, (nb, 1, nq, nk)), f32), (1, H, 1, 1))
        x = ttnn.where(m, fill, x)
        ttnn.deallocate(m)
        x = ttnn.permute(ttnn.to_layout(x, RM), (1, 0, 2, 3))          # [H, nb, nq, nk]
        offs = off.tolist()
        runs, i0 = [], 0
        for i in range(1, nb + 1):
            if i == nb or offs[i] != offs[i0]:
                runs.append((i0, i, offs[i0])); i0 = i
        parts = []
        for i0, i1, o in runs:
            seg = [ttnn.slice(x, (0, i0, 0, 0), (H, i1, nq, nk))]
            if o:
                seg.insert(0, ttnn.full((H, i1 - i0, nq, o), fill, dtype=f32, layout=RM, device=dev))
            if W - o - nk:
                seg.append(ttnn.full((H, i1 - i0, nq, W - o - nk), fill, dtype=f32, layout=RM, device=dev))
            parts.append(ttnn.concat(seg, dim=3) if len(seg) > 1 else seg[0])
        ttnn.deallocate(x)
        full = ttnn.concat(parts, dim=1) if len(parts) > 1 else parts[0]
        full = ttnn.to_layout(ttnn.reshape(full, (1, H * nb, nq, W)), ttnn.TILE_LAYOUT)
        return ttnn.typecast(full, self._act_dtype) if self._act_dtype != f32 else full

    def _attend_superset(self, a_qn, a_kn, apb, sup_b, NP, nb, mc=None):
        """softmax(q k^T * scale + bias) v over the superset window, one SDPA program.

        Q, K and V are per atom ``[S, NP, 128]``; one zero-led frame carries all three, split by
        ``nlp_create_qkv_heads``, and the kernel reads block i's queries and its key window
        straight from the frame (``kv_window``), so no window is ever materialised. `mc`
        (atom_l1) places the projections, the frame, the heads and the output; default DRAM."""
        from . import sdpa_generic as SG
        nq, H, dh = self.N_QUERY, self.N_HEADS, self.HEAD_DIM
        W = int(sup_b.shape[-1]); L = W - self.N_KEY
        S = a_qn.shape[0]
        dram = mc or ttnn.DRAM_MEMORY_CONFIG
        Q = self._lin(a_qn, apb + "mha.linear_q.weight", apb + "mha.linear_q.bias", memory_config=mc)
        KV = self._lin(a_kn, apb + "mha.linear_kv.weight", memory_config=mc)
        x = ttnn.concat([Q, KV], dim=-1, memory_config=mc)
        ttnn.deallocate(Q); ttnn.deallocate(KV)
        # Zero rows on both ends (ttnn.pad cannot pad the front of a tiled tensor): L in front,
        # W - L - nq behind, so block i's window [32 i, 32 i + W) stays inside the frame.
        def zero(rows):
            z = self._zeros.get((S, x.dtype, rows))
            if z is None:
                z = self._zeros[(S, x.dtype, rows)] = ttnn.zeros(
                    (S, rows, 3 * H * dh), dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            return z
        T = W - L - nq
        x2 = ttnn.concat([zero(L), x, zero(T)], dim=1, memory_config=mc)   # [S, L + NP + T, 384]
        ttnn.deallocate(x)
        x, F = x2, L + NP + T
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(x, (S, 1, F, 3 * H * dh)), num_heads=H, num_kv_heads=H,
            transpose_k_heads=False, memory_config=dram)  # each [S, H, F, dh]
        ttnn.deallocate(x)
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([S, H * nb, nq, dh]), q.dtype,
                                           ttnn.TILE_LAYOUT, self.device, dram)
        g = self.device.compute_with_storage_grid_size()
        SG.sdpa(self.device, q, k, v, sup_b, o, nq, W, (g.x, g.y), self._sdpa32_ckc, dh ** -0.5,
                kv_window=(nb, W, L, nq))
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = ttnn.experimental.nlp_concat_heads(ttnn.reshape(o, (S, H, NP, dh)), memory_config=dram)
        return ttnn.reshape(o, (S, NP, H * dh))

    def __call__(self, a, s, z, atom_mask_col, key_block_idxs_tt, valid_mask,
                 mask_bias, n_atom, NP, nb, cache=None):
        """a: [S, NP, 128] device, S noised samples at once (padded to a multiple of N_QUERY;
        a_init = cl when rl=None); s: [1, NP, 128]; z: [1, nb, N_QUERY, N_KEY, 16] device
        (blocked pair = plm);
        atom_mask_col: [1, NP, 1] device; key_block_idxs_tt: [1, nb*N_KEY] uint32 device;
        valid_mask: [1, nb, N_KEY, 1] device; mask_bias: [1, nb, 1, N_QUERY, N_KEY] device.
        Returns [S, n_atom, 128].

        Every sample shares s, z and the masks, so the sample axis is the leading dim of the
        evolving ``a`` only: the per-block invariants stay ``[1, ...]`` and broadcast, and the
        key gather reads a sample-offset copy of the index (``sample_offset_index``). At S=1
        this runs exactly the single-sample ops."""
        scale = self.HEAD_DIM ** -0.5
        nq, nk, H, dh = self.N_QUERY, self.N_KEY, self.N_HEADS, self.HEAD_DIM

        z_bias, s_q, s_k, ada_raw, cg_raw, ada_s = _cached(
            cache, (id(self), "block_invariants"),
            lambda: self._invariants(s, z, key_block_idxs_tt, valid_mask, nb))

        S = a.shape[0]
        # The superset bias is built on the host, so a taped (training) forward keeps the
        # gathered path, whose pair bias stays on the tape.
        sup = (_cached(cache, (id(self), "superset"),
                       lambda: self._superset(z_bias, mask_bias, key_block_idxs_tt, s, nb))
               if self._sdpa32 and not _taping() else ())
        kidx = key_block_idxs_tt if S == 1 or sup else _cached(
            cache, (id(self), "key_idx", S),
            lambda: sample_offset_index(key_block_idxs_tt, S, NP))
        # atom_l1 (superset path): the stream and every per-step tensor in L1 where four copies of the
        # largest, the q/k/v frame, fit; the rollout invariants above stay in DRAM. Copied in, so the
        # caller's `a` is untouched.
        mc = _T._atom_l1((S, NP + int(sup[0][0].shape[-1]), 3 * H * dh), a.dtype) if sup else None
        x = ttnn.to_memory_config(a, mc) if mc else a
        for b in range(3):
            P = f"blocks.{b}."
            apb = P + "attention_pair_bias."
            if sup:
                bias, q_terms, k_terms = sup[b]
                a_qn = self.adaln_q[b](x, s, s_terms=q_terms or None)
                a_kn = self.adaln_k[b](x, s, s_terms=k_terms or None)
                o = self._attend_superset(a_qn, a_kn, apb, bias, NP, nb, mc)
                ttnn.deallocate(a_kn)
                o = ttnn.multiply(o, self._lin(a_qn, apb + "mha.linear_g.weight", memory_config=mc),
                                  input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(a_qn)
                o = self._lin(o, apb + "mha.linear_o.weight", memory_config=mc)
                x = self._gate_add(x, o, ada_raw[b])
                x = self._transition(x, s, b, ada_s, cg_raw, atom_mask_col, mc)
                continue
            x_q = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_q = ttnn.reshape(x_q, (S, nb, nq, 128))
            x_q = ttnn.to_layout(x_q, ttnn.TILE_LAYOUT)
            x_k = self._gather_keys(x, kidx, valid_mask, nb)

            a_qn = self.adaln_q[b](x_q, s_q, s_terms=ada_s[b][0] if ada_s else None)
            a_kn = self.adaln_k[b](x_k, s_k, s_terms=ada_s[b][1] if ada_s else None)
            Q = self._heads(self._lin(a_qn, apb + "mha.linear_q.weight", apb + "mha.linear_q.bias"), nb, nq)
            K = self._heads(self._lin(a_kn, apb + "mha.linear_k.weight"), nb, nk)
            V = self._heads(self._lin(a_kn, apb + "mha.linear_v.weight"), nb, nk)
            sc = batched_matmul(Q, ttnn.permute(K, (0, 1, 2, 4, 3)),
                                compute_kernel_config=self.compute_kernel_config)
            sc = scale_add(sc, scale, mask_bias)
            sc = ttnn.add(sc, z_bias[b])
            attn = site_softmax(sc, dim=-1, compute_kernel_config=self._softmax_ckc,
                                host_f64=self._softmax_f64)
            o = batched_matmul(attn, V, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(sc); ttnn.deallocate(attn)
            # o: [1,nb,H,Q,dh] -> [1,nb,Q,H,dh]; gate with sigmoid(linear_g(a_qn)).
            o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
            o = ttnn.permute(o, (0, 1, 3, 2, 4))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            g_raw = self._lin(a_qn, apb + "mha.linear_g.weight")
            g_raw = ttnn.to_layout(g_raw, ttnn.ROW_MAJOR_LAYOUT)
            g_raw = ttnn.reshape(g_raw, (S, nb, nq, H, dh))
            g_raw = ttnn.to_layout(g_raw, ttnn.TILE_LAYOUT)
            o = ttnn.multiply(o, g_raw, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
            o = ttnn.reshape(o, (S, nb, nq, 128))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            o = self._lin(o, apb + "mha.linear_o.weight")
            o = ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), (S, NP, 128))
            o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
            x = self._gate_add(x, o, ada_raw[b])

            ttnn.deallocate(a_qn); ttnn.deallocate(a_kn)
            x = self._transition(x, s, b, ada_s, cg_raw, atom_mask_col)

        if cache is None:
            # Uncached path owns the group; the cached one is freed by the sampler.
            from .openfold3_sample_diffusion import _free_cached
            _free_cached(sup)
            for t in (*z_bias, s_q, s_k, *ada_raw, *cg_raw,
                      *(v for g in ada_s for pair in g for v in pair)):
                ttnn.deallocate(t)
            if kidx is not key_block_idxs_tt:
                ttnn.deallocate(kidx)

        if mc:
            x, x_l1 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), x
            ttnn.deallocate(x_l1)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, [0, 0, 0], [S, n_atom, 128])
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
