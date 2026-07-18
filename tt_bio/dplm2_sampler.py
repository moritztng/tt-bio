"""DPLM-2 discrete-diffusion generation layer (pure torch, no ttnn).

This module holds the parts of DPLM-2 that are *not* the ESM backbone: the
joint aa/struct tokenizer (vocab wiring), the categorical-sampling utilities,
and the iterative discrete-denoising (reparameterized decoding) loop. It is
pure PyTorch on host — the only per-step heavy compute is one backbone
forward, which the caller supplies as `backbone_fn(input_ids) -> logits`.

Keeping the algorithm here (shared by the ttnn port in tt_bio.dplm2 and the
PyTorch reference in tests/dplm2_reference.py) means the diffusion loop is
written once and parity-gated once: the reference and the device port differ
only in the backbone, never in the decoding math. Faithful to
byprot.models.dplm2.dplm2.MultimodalDiffusionProteinLanguageModel.generate and
its helpers (initialize_output_tokens / forward_decoder / _reparam_decoding)
and byprot.models.utils (topk_masking / sample_from_categorical /
top_k_top_p_filtering).

The 3D-structure <-> struct-token VQ-VAE (airkingbd/struct_tokenizer: GVP
encoder + LFQ + ESMFold-variant decoder) is a separate, heavier model with its
own checkpoint and is NOT ported here; this module wires the token *vocabulary*
(aa + 8192 struct tokens + 4 special struct tokens) so the diffusion loop can
run on struct tokens. See docs/dplm2-port.md.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

# airkingbd/dplm2_150m vocab.txt layout (verified against the cached repo):
#   0  <cls_aa>      (aa_bos)        29  .
#   1  <pad>                          30  -
#   2  <eos_aa>      (aa_eos)        31  <null_1>
#   3  <unk_aa>      (aa_unk)        32  <mask_aa>     (aa_mask)
#   4..28  20 AAs + X,B,U,Z,O        33  <cls_struct>  (struct_bos)
#   33  <cls_struct>                 34  <eos_struct>  (struct_eos)
#   34  <eos_struct>                 35  <unk_struct>  (struct_unk)
#   35  <unk_struct>                 36..8227  struct tokens "0000".."8191"
#   36..8227  struct tokens           8228 <mask_struct> (struct_mask)
#   8228 <mask_struct>
AA_VOCAB_BOUND = 33  # id < 33 -> aa; id >= 33 -> struct (matches byprot)


def _load_vocab(vocab_file: str) -> list[str]:
    with open(vocab_file, "r") as f:
        return [line.strip() for line in f.read().splitlines()]


class DPLM2Tokenizer:
    """Joint aa/struct tokenizer for DPLM-2 (vocab wiring only).

    Loads the vocab.txt shipped with airkingbd/dplm2_150m and exposes the
    special-token ids DPLM-2's generation loop needs, plus aa-string and
    struct-token-string (de)coding. No 3D coords -- the VQ-VAE that maps 3D
    <-> struct tokens is a separate model (deferred; see docs/dplm2-port.md).
    """

    def __init__(self, vocab_file: str):
        self.all_tokens = _load_vocab(vocab_file)
        assert len(self.all_tokens) == 8229, len(self.all_tokens)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {t: i for i, t in enumerate(self.all_tokens)}

        self.aa_bos_id = self._token_to_id["<cls_aa>"]
        self.aa_eos_id = self._token_to_id["<eos_aa>"]
        self.aa_unk_id = self._token_to_id["<unk_aa>"]
        self.aa_mask_id = self._token_to_id["<mask_aa>"]
        self.struct_bos_id = self._token_to_id["<cls_struct>"]
        self.struct_eos_id = self._token_to_id["<eos_struct>"]
        self.struct_unk_id = self._token_to_id["<unk_struct>"]
        self.struct_mask_id = self._token_to_id["<mask_struct>"]
        self.pad_id = self._token_to_id["<pad>"]

        self.aa_type = 1
        self.struct_type = 0
        self.pad_type = 2

        # 20 standard AAs (single-letter) -> id; X/B/U/Z/O map too.
        self.aa_letter_to_id = {t: self._token_to_id[t] for t in
                                "ACDEFGHIKLMNPQRSTVWYXBUZO"}
        self.aa_id_to_letter = {i: t for t, i in self.aa_letter_to_id.items()}
        # struct token strings "0000".."8191" -> id (36..8227)
        self.struct_str_to_id = {t: self._token_to_id[t]
                                 for t in self.all_tokens
                                 if t.isdigit() and len(t) == 4}

    @classmethod
    def from_pretrained(cls, repo_id: str = "airkingbd/dplm2_150m",
                        cache_dir: Optional[str] = None) -> "DPLM2Tokenizer":
        from huggingface_hub import hf_hub_download
        vf = hf_hub_download(repo_id, "vocab.txt", cache_dir=cache_dir)
        return cls(vf)

    @property
    def special_token_list(self) -> list[int]:
        # matches MultimodalDiffusionProteinLanguageModel.special_token_list
        return [
            self.aa_bos_id, self.aa_eos_id, self.aa_mask_id,
            self.struct_bos_id, self.struct_eos_id, self.struct_mask_id,
            self.pad_id, self.aa_unk_id, self.struct_unk_id,
        ] + [self._token_to_id[x] for x in ("X", "B", "U", "Z", "O")]

    def get_modality_type(self, input_ids: torch.Tensor) -> torch.Tensor:
        """0=struct, 1=aa, 2=pad -- matches DPLM2.get_modality_type."""
        mask = input_ids.ne(self.pad_id)
        m = ((input_ids < AA_VOCAB_BOUND) & mask).int()
        m[~mask] = self.pad_type
        return m

    def encode_aa(self, seq: str, add_special: bool = True) -> list[int]:
        ids = [self.aa_letter_to_id[c] for c in seq]
        if add_special:
            ids = [self.aa_bos_id] + ids + [self.aa_eos_id]
        return ids

    def decode_aa(self, ids: torch.Tensor) -> str:
        out = []
        for i in ids.tolist():
            if i == self.aa_id_to_letter.get(i):
                pass
            if i in self.aa_id_to_letter:
                out.append(self.aa_id_to_letter[i])
        return "".join(out)

    def encode_struct(self, struct_strs: str) -> list[int]:
        # struct_strs is a concatenation of 4-digit codes, e.g. "000100020003"
        return [self.struct_str_to_id[struct_strs[i:i + 4]]
                for i in range(0, len(struct_strs), 4)]

    def build_joint(self, aa_seq: str, struct_str: str,
                    pad_to: Optional[int] = None) -> torch.Tensor:
        """Build a joint [1, L] token tensor: [struct_bos, struct..., struct_eos,
        aa_bos, aa..., aa_eos], left-aligned and padded to `pad_to`."""
        s = [self.struct_bos_id] + self.encode_struct(struct_str) + [self.struct_eos_id]
        a = [self.aa_bos_id] + self.encode_aa(aa_seq, add_special=False) + [self.aa_eos_id]
        ids = s + a
        if pad_to is not None and len(ids) < pad_to:
            ids = ids + [self.pad_id] * (pad_to - len(ids))
        return torch.tensor([ids], dtype=torch.long)


# --------------------------------------------------------------------------- #
# Categorical sampling utilities (ported from byprot.models.utils).
# --------------------------------------------------------------------------- #
def topk_masking(scores: torch.Tensor, cutoff_len: torch.Tensor,
                  stochastic: bool = False, temp: float = 1.0) -> torch.Tensor:
    """Mask of the `cutoff_len` lowest-scoring positions per row [B,N]."""
    if stochastic:
        g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        s = scores + temp * g
    else:
        s = scores
    sorted_vals = s.sort(-1)[0]
    cutoff = sorted_vals.gather(dim=-1, index=cutoff_len)
    return s < cutoff


def sample_from_categorical(logits: torch.Tensor, temperature: float = 1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(logits: torch.Tensor, temperature: float = 1.0,
                                       noise_scale: float = 1.0):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * g
    return sample_from_categorical(logits, temperature)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.95,
                          filter_value: float = float("-Inf")) -> torch.Tensor:
    """Nucleus (top-p) filtering, faithful to byprot.models.utils."""
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        thr = torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits = torch.where(logits < thr, torch.full_like(logits, filter_value), logits)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    sorted_logits = torch.where(remove, torch.full_like(sorted_logits, filter_value), sorted_logits)
    logits = torch.gather(sorted_logits, 1, sorted_idx.argsort(-1))
    return logits.reshape(ori_shape)


# --------------------------------------------------------------------------- #
# Discrete-diffusion generation loop (faithful to byprot DPLM2.generate).
# --------------------------------------------------------------------------- #
class DPLM2Sampler:
    """Iterative discrete-denoising generation for DPLM-2.

    `backbone_fn(input_ids) -> logits[B,L,vocab]` is the only device-side
    dependency; everything else (modality masking, sampling, reparameterized
    re-masking) runs on host. Mirrors
    MultimodalDiffusionProteinLanguageModel.generate / forward_decoder /
    _reparam_decoding so that a fp32-backbone run and a ttnn-backbone run share
    the exact same decoding math (parity differs only in backbone precision).
    """

    def __init__(self, tok: DPLM2Tokenizer,
                 backbone_fn: Callable[[torch.Tensor], torch.Tensor],
                 num_diffusion_timesteps: int = 500):
        self.tok = tok
        self.backbone_fn = backbone_fn
        self.num_timesteps = num_diffusion_timesteps

    def _non_special_mask(self, tokens, partial_masks=None):
        t = self.tok
        m = (tokens.ne(t.pad_id) & tokens.ne(t.aa_bos_id) & tokens.ne(t.aa_eos_id)
             & tokens.ne(t.struct_bos_id) & tokens.ne(t.struct_eos_id))
        if partial_masks is not None:
            m = m & ~partial_masks
        return m

    def _initialize_output_tokens(self, input_tokens, partial_masks=None):
        t = self.tok
        type_ids = t.get_modality_type(input_tokens)
        out_mask = self._non_special_mask(input_tokens, partial_masks)
        aa_pos = type_ids.eq(t.aa_type) & out_mask
        st_pos = type_ids.eq(t.struct_type) & out_mask
        out = input_tokens.masked_fill(aa_pos, t.aa_mask_id)
        out = out.masked_fill(st_pos, t.struct_mask_id)
        scores = torch.zeros_like(out, dtype=torch.float)
        return out, scores

    def _forward_decoder(self, tokens, scores, step, max_step, temperature,
                         sampling_strategy):
        t = self.tok
        out_masks = self._non_special_mask(tokens)
        logits = self.backbone_fn(tokens).float().log_softmax(dim=-1)
        type_ids = t.get_modality_type(tokens)
        aa_pos = type_ids.eq(t.aa_type) & out_masks
        st_pos = type_ids.eq(t.struct_type) & out_masks
        ix_aa = torch.where(aa_pos)
        ix_st = torch.where(st_pos)
        # aa positions may only predict aa tokens (<33); struct positions >=33.
        if ix_aa[0].numel():
            logits[ix_aa[0], ix_aa[1], AA_VOCAB_BOUND:] = float("-inf")
        if ix_st[0].numel():
            logits[ix_st[0], ix_st[1], :AA_VOCAB_BOUND] = float("-inf")
        logits[..., t.special_token_list] = float("-inf")
        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            sc, tk = logits.max(-1)
        elif sampling_strategy.startswith("annealing"):
            max_t, min_t = map(float, sampling_strategy.split("@")[1].split(":"))
            rate = 1 - step / max_step
            temperature = min_t + (max_t - min_t) * rate
            tk, sc = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == "gumbel_argmax":
            tk, sc = stochastic_sample_from_categorical(logits, temperature=0.0,
                                                       noise_scale=temperature)
        else:
            tk, sc = sample_from_categorical(logits, temperature=temperature)

        new_tokens = tokens.clone()
        new_scores = scores.clone()
        new_tokens.masked_scatter_(out_masks, tk[out_masks])
        new_scores.masked_scatter_(out_masks, sc[out_masks])
        return new_tokens, new_scores, out_masks

    def _reparam(self, prev_tokens, prev_scores, cur_tokens, cur_scores,
                 xt_neq_x0, type_ids, non_special, t, max_step, topk_mode):
        # reparam-uncond-<topk_mode>-linear: rate = 1 - t/max_step
        rate = 1 - t / max_step
        cutoff_len = non_special.sum(1, keepdim=True).type_as(cur_scores) * rate
        cutoff_len = cutoff_len.long()
        scores_for_topk = cur_scores.masked_fill(~non_special, 1000.0)
        if topk_mode.startswith("stochastic"):
            ns = float(topk_mode.replace("stochastic", ""))
            lowest_k = topk_masking(scores_for_topk, cutoff_len, stochastic=True,
                                    temp=ns * rate)
        else:  # "deterministic"
            lowest_k = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
        not_v1_t = lowest_k  # uncond
        not_v2_t = lowest_k

        out_tokens = prev_tokens.clone()
        out_scores = prev_scores.clone()
        aa_pos = type_ids.eq(self.tok.aa_type) & non_special
        st_pos = type_ids.eq(self.tok.struct_type) & non_special
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        # noise is the per-modality mask id; apply separately for aa/struct rows.
        out_tokens.masked_fill_(masked_to_noise & aa_pos, self.tok.aa_mask_id)
        out_tokens.masked_fill_(masked_to_noise & st_pos, self.tok.struct_mask_id)
        out_scores.masked_fill_(masked_to_noise, float("-inf"))

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        out_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        out_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        new_xt = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt, out_tokens, out_scores

    @torch.no_grad()
    def generate(self, input_tokens, max_iter=None, temperature=1.0,
                 partial_masks=None,
                 unmasking_strategy="stochastic1.0",
                 sampling_strategy="annealing@2.0:0.1"):
        """Run iterative unmasking. Returns (output_tokens[B,L], history list)."""
        if max_iter is None:
            max_iter = self.num_timesteps
        out_tokens, out_scores = self._initialize_output_tokens(input_tokens, partial_masks)
        type_ids = self.tok.get_modality_type(out_tokens)
        xt_neq_x0 = self._non_special_mask(out_tokens, partial_masks)
        # topk_mode parsed from unmasking_strategy ("stochastic<scale>" | "deterministic")
        topk_mode = unmasking_strategy  # "stochastic1.0" or "deterministic"
        history = [out_tokens.clone()]
        for step in range(max_iter):
            cur_tokens, cur_scores, _ = self._forward_decoder(
                out_tokens, out_scores, step, max_iter, temperature, sampling_strategy)
            non_special = self._non_special_mask(out_tokens, partial_masks)
            xt_neq_x0, out_tokens, out_scores = self._reparam(
                out_tokens, out_scores, cur_tokens, cur_scores,
                xt_neq_x0, type_ids, non_special, step + 1, max_iter, topk_mode)
            history.append(out_tokens.clone())
        return out_tokens, history
