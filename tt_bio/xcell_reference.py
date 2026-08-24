"""X-Cell in torch: the reference of record for the X-Cell port.

Reconstructed from the preprint's Appendix A "X-Cell Model Specification"
(bioRxiv 10.64898/2026.03.18.712807v1, Wang, Karimzadeh, Ravindra et al., Xaira Therapeutics).
Equation numbers in the comments below are that appendix's, so every module can be checked
against a published line rather than against taste.

**Why this file exists at all.** X-Cell has no public inference code and no public weights: as of
2026-08-24 the HuggingFace repo holds three files and no checkpoint, and upstream's
``src/xcell/model.py`` raises ``NotImplementedError`` from both ``from_pretrained`` and
``predict``. There is therefore no runnable reference to port against, so this transcription IS
the reference, in the sense of `pet_model_host.py` for PET-MAD: the ttnn port in
``tt_bio/xcell.py`` is scored against these modules, and nothing here imports anything from
Xaira. It lives next to ``af2_reference.py`` rather than under ``_vendor/`` on purpose --
``_vendor/`` is third-party code, and X-Cell's upstream is CC BY-NC-SA 4.0, so filing our own
clean-room implementation there would misrepresent where it came from.

**What is pinned by a published equation and what is an assumption.** The input encoding (A1-A5),
the six prior projections (A6-A11), the attention and block structure (A12-A22), the decoder and
tied head (A23-A26) and the 4-step inference schedule (A.4) are transcribed. The dimensions the
appendix does not state -- the prior-MLP hidden width, the two decoder hidden widths, the norm
epsilon -- are ``XCellConfig`` fields with documented defaults, never hardcodes, so a real
checkpoint can set them without editing a module.

**Mini is not the block the appendix specifies, and that is deliberate.** A12-A22 describe a
pre-norm RMSNorm/SwiGLU/QK-Norm block, but upstream's own FSDP config wraps *two* layer classes,
``TransformerEncoderLayer`` and ``ModernTransformerEncoderLayer``, and ``docs/model.md`` gives
X-Cell Mini as Post-LN with a 1x ReLU FFN. Mini is scGPT-initialized and scGPT is Post-LN with a
ReLU FFN, which is what upstream's ``strict=False`` load is accommodating. The parameter budget
agrees: a modern block at d=512 costs 4d^2 + 3*d*d_ff = 4.20M with d_ff=4d, so 12 of them are
~50M before the ~9.9M embedding, well past the published 55M, while a 1x-FFN Post-LN block costs
~1.57M and leaves room for the embedding, the priors and the cross-attention blocks. Both blocks
are implemented; ``XCellConfig.block`` picks one, and ``x-cell-mini`` picks ``legacy``.

Inference only. The auxiliary perturbation decoder (A27) and every training loss are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# The six cross-attention prior sources, as DATA. A row is a name, the published width of the
# per-gene vector, and where it comes from; the model instantiates one projection per row. Adding
# a source is a row, not a code path (A6-A11).
#
#   name      dim   source
PRIOR_SOURCES: tuple[tuple[str, int, str], ...] = (
    ("esm",    5120, "ESM-2 protein language model embedding"),
    ("string",  512, "STRING protein-protein interaction network embedding"),
    ("genept", 3072, "GenePT LLM gene representation"),
    ("depmap", 1150, "DepMap genetic dependency profile"),
    ("cp",      259, "JUMP-Cell Painting morphology PCA"),
)
# The sixth context token is the stop-gradient gene embedding of the perturbed gene (A11). It is
# not an external file, so it has no row above and no projection: it is already d-dimensional.
N_PRIOR_TOKENS = len(PRIOR_SOURCES) + 1

# The inference reveal ladder (A.4, "Inference (cumulative mode)"). The four steps are the four
# training reveal fractions applied cumulatively, and the last one reveals every gene.
REVEAL_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

VALUE_CLIP = 512.0  # A3 clips expression at 512 INSIDE the encoder, not in data prep


@dataclass
class XCellConfig:
    """One X-Cell variant. Every field the appendix does not pin carries its reasoning here."""

    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    vocab_size: int = 19_400        # |V| ~ 19,400 protein-coding genes + <cls>/<mask> + aliases
    d_ff: int | None = None         # None -> 1*d for `legacy`, 4*d for `modern` (A18/inline A18)
    block: str = "legacy"           # "legacy" (Post-LN, ReLU, scGPT-shaped) | "modern" (A12-A22)
    cross_attn_stride: int = 3      # cross-attention at layers l where l % stride == stride-1
    global_residual: bool = False   # A23 H' = H^(L) + H^(0). Ablated OFF: MAE 0.178 -> 0.163
    norm_eps: float = 1e-5          # unstated in the appendix
    prior_hidden: int | None = None  # None -> d_model. "two-layer with LeakyReLU" fixes the
    #                                 shape, not the width
    decoder_hidden: tuple[int, int] | None = None  # None -> (d_model, d_model). A25 gives three
    #                                                layers and LReLU, not the widths
    leaky_slope: float = 0.01       # A25 states alpha = 0.01
    max_genes: int = 4000           # G' <= 4000 (A.10.2). Sequence length is G' + 1 with CLS
    set_size: int = 64              # S, cells per set (A.10.1)

    def __post_init__(self) -> None:
        if self.block not in ("legacy", "modern"):
            raise ValueError(f"block must be 'legacy' or 'modern', got {self.block!r}")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if self.d_ff is None:
            self.d_ff = self.d_model if self.block == "legacy" else 4 * self.d_model
        if self.prior_hidden is None:
            self.prior_hidden = self.d_model
        if self.decoder_hidden is None:
            self.decoder_hidden = (self.d_model, self.d_model)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads  # A16

    @property
    def cross_attn_layers(self) -> tuple[int, ...]:
        """Which layer indices carry a cross-attention block.

        `l % 3 == 2` reproduces BOTH published counts: 4 of 12 for Mini (docs/model.md) and 11 of
        34 for Ultra (the preprint's TTA section says it bypasses "all 11 cross-attention
        blocks"). Two independent data points on one rule, so this is not a guess.
        """
        s = self.cross_attn_stride
        return tuple(l for l in range(self.n_layers) if l % s == s - 1)


# X-Cell Mini, the only variant with announced weights (55M, docs/model.md).
XCELL_MINI = XCellConfig()


class RMSNorm(nn.Module):
    """A17: x / sqrt(mean(x^2) + eps) * gamma, learnable scale, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class PriorProjection(nn.Module):
    """A6-A10: a two-layer LeakyReLU network with a LayerNorm output, per prior source."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, slope: float, eps: float):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out, eps=eps)
        self.slope = slope

    def forward(self, z: Tensor) -> Tensor:
        return self.norm(self.fc2(F.leaky_relu(self.fc1(z), self.slope)))


class ValueEncoder(nn.Module):
    """A3: phi(x) = LayerNorm(W2 ReLU(W1 clip(x, -inf, 512)) + b2).

    Dropout is identity at inference and is omitted. W1 is d x 1, so the scalar expression value
    is broadcast into d channels before the ReLU.
    """

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.fc1 = nn.Linear(1, d, bias=False)
        self.fc2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(max=VALUE_CLIP).unsqueeze(-1)
        return self.norm(self.fc2(F.relu(self.fc1(x))))


class InputEmbedding(nn.Module):
    """A2, A4, A5: gene identity + value + perturbation-mask, with a learned CLS prepended."""

    def __init__(self, cfg: XCellConfig):
        super().__init__()
        d = cfg.d_model
        self.gene = nn.Embedding(cfg.vocab_size, d)
        self.gene_norm = nn.LayerNorm(d, eps=cfg.norm_eps)   # A2
        self.value = ValueEncoder(d, cfg.norm_eps)           # A3
        self.mask = nn.Embedding(2, d)                       # A4
        self.mask_norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.cls = nn.Embedding(1, d)                        # a standalone learned vector

    def raw_gene_embedding(self, tokens: Tensor) -> Tensor:
        """The UN-normalized gene embedding, which is what the tied head uses (A26)."""
        return self.gene(tokens)

    def forward(self, values: Tensor, tokens: Tensor, pert_mask: Tensor) -> Tensor:
        # values, tokens, pert_mask: [N, G]. Returns [N, G+1, d].
        e = self.gene_norm(self.gene(tokens))
        h = e + self.value(values) + self.mask_norm(self.mask(pert_mask.long()))
        cls = self.cls.weight.expand(h.shape[0], 1, -1)
        return torch.cat([cls, h], dim=1)                    # A5, CLS gets no value/mask term


class SwiGLU(nn.Module):
    """A18: W_down(SiLU(W_gate x) * W_up x), no biases."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    n, s, _ = x.shape
    return x.view(n, s, n_heads, -1).transpose(1, 2)         # [N, H, S, d_head]


def _merge_heads(x: Tensor) -> Tensor:
    n, h, s, dh = x.shape
    return x.transpose(1, 2).reshape(n, s, h * dh)


class SelfAttention(nn.Module):
    """A12-A16. `qk_norm` adds the per-head RMSNorm on Q and K that the modern block uses."""

    def __init__(self, cfg: XCellConfig, *, qk_norm: bool, bias: bool):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.q = nn.Linear(d, d, bias=bias)
        self.k = nn.Linear(d, d, bias=bias)
        self.v = nn.Linear(d, d, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)
        # Per-head RMSNorm: it normalises over d_head, not over d (A12/A13).
        self.q_norm = RMSNorm(cfg.d_head, cfg.norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(cfg.d_head, cfg.norm_eps) if qk_norm else None

    def forward(self, x: Tensor) -> Tensor:
        q = _split_heads(self.q(x), self.n_heads)
        k = _split_heads(self.k(x), self.n_heads)
        v = _split_heads(self.v(x), self.n_heads)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        a = F.scaled_dot_product_attention(q, k, v)          # A15, scale 1/sqrt(d_head)
        return self.o(_merge_heads(a))


class CrossAttention(nn.Module):
    """A21: query from the gene representations, key/value from the 6-token perturbation context.

    `context_missing` is True where a prior source is absent for this perturbation, which is the
    `key_padding_mask` convention the appendix describes: missing sources are zero-imputed and
    then masked out so the model ignores them rather than attending to zeros.
    """

    def __init__(self, cfg: XCellConfig, *, bias: bool):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.q = nn.Linear(d, d, bias=bias)
        self.k = nn.Linear(d, d, bias=bias)
        self.v = nn.Linear(d, d, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)

    def forward(self, x: Tensor, context: Tensor, context_missing: Tensor | None = None) -> Tensor:
        q = _split_heads(self.q(x), self.n_heads)
        k = _split_heads(self.k(context), self.n_heads)
        v = _split_heads(self.v(context), self.n_heads)
        attn_mask = None
        if context_missing is not None:
            # [N, 1, 1, C] additive-equivalent boolean mask: True means "may attend".
            attn_mask = (~context_missing.bool())[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(_merge_heads(a))


class ModernBlock(nn.Module):
    """A19-A22: pre-RMSNorm self-attention + SwiGLU, and at cross layers a second pair."""

    def __init__(self, cfg: XCellConfig, *, cross: bool):
        super().__init__()
        d, eps = cfg.d_model, cfg.norm_eps
        self.attn_norm = RMSNorm(d, eps)
        self.attn = SelfAttention(cfg, qk_norm=True, bias=False)
        self.ffn_norm = RMSNorm(d, eps)
        self.ffn = SwiGLU(d, cfg.d_ff)
        if cross:
            self.cross_norm = RMSNorm(d, eps)                # A20
            self.cross = CrossAttention(cfg, bias=False)      # A21
            self.cross_ffn_norm = RMSNorm(d, eps)             # A22 is a SECOND SwiGLU,
            self.cross_ffn = SwiGLU(d, cfg.d_ff)              # distinct from A19's
        else:
            self.cross = None

    def forward(self, x: Tensor, context: Tensor | None, context_missing: Tensor | None) -> Tensor:
        x = x + self.attn(self.attn_norm(x))                 # A19
        x = x + self.ffn(self.ffn_norm(x))                   # A19
        if self.cross is not None and context is not None:
            x = x + self.cross(self.cross_norm(x), context, context_missing)   # A20-A21
            x = x + self.cross_ffn(self.cross_ffn_norm(x))                     # A22
        return x


class LegacyBlock(nn.Module):
    """The Post-LN, ReLU-FFN block X-Cell Mini inherits from scGPT (docs/model.md).

    Structurally `torch.nn.TransformerEncoderLayer(norm_first=False, activation='relu')`: the
    residual is added first and the LayerNorm follows, with biases throughout. Cross-attention at
    a cross layer follows the same Post-LN shape.
    """

    def __init__(self, cfg: XCellConfig, *, cross: bool):
        super().__init__()
        d, eps = cfg.d_model, cfg.norm_eps
        self.attn = SelfAttention(cfg, qk_norm=False, bias=True)
        self.attn_norm = nn.LayerNorm(d, eps=eps)
        self.fc1 = nn.Linear(d, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, d)
        self.ffn_norm = nn.LayerNorm(d, eps=eps)
        if cross:
            self.cross = CrossAttention(cfg, bias=True)
            self.cross_norm = nn.LayerNorm(d, eps=eps)
            self.cross_fc1 = nn.Linear(d, cfg.d_ff)
            self.cross_fc2 = nn.Linear(cfg.d_ff, d)
            self.cross_ffn_norm = nn.LayerNorm(d, eps=eps)
        else:
            self.cross = None

    def forward(self, x: Tensor, context: Tensor | None, context_missing: Tensor | None) -> Tensor:
        x = self.attn_norm(x + self.attn(x))
        x = self.ffn_norm(x + self.fc2(F.relu(self.fc1(x))))
        if self.cross is not None and context is not None:
            x = self.cross_norm(x + self.cross(x, context, context_missing))
            x = self.cross_ffn_norm(x + self.cross_fc2(F.relu(self.cross_fc1(x))))
        return x


class Decoder(nn.Module):
    """A24-A26: concat the cell embedding onto each gene, three LeakyReLU layers, tied projection."""

    def __init__(self, cfg: XCellConfig):
        super().__init__()
        h1, h2 = cfg.decoder_hidden
        self.fc1 = nn.Linear(2 * cfg.d_model, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, cfg.d_model)
        self.gene_bias = nn.Embedding(cfg.vocab_size, 1)     # b_g, a learned per-gene bias
        self.slope = cfg.leaky_slope
        self.scale = 1.0 / sqrt(cfg.d_model)                 # PaLM-style 1/sqrt(d)

    def forward(self, h_genes: Tensor, h_cls: Tensor, raw_gene_emb: Tensor,
                tokens: Tensor) -> Tensor:
        # A24: gene output first, CLS second.
        d = torch.cat([h_genes, h_cls.unsqueeze(1).expand_as(h_genes)], dim=-1)
        h = F.leaky_relu(self.fc1(d), self.slope)
        h = F.leaky_relu(self.fc2(h), self.slope)
        h = self.fc3(h)                                      # A25, no activation after W3
        # A26: tied projection through the RAW (un-normalized) gene embeddings.
        out = (h * raw_gene_emb).sum(-1) * self.scale
        return out + self.gene_bias(tokens).squeeze(-1)


class XCell(nn.Module):
    """One forward pass of X-Cell: partially revealed expression in, full prediction out.

    The cell-set axis is folded into batch exactly as the appendix's A.1.1 describes, so a call
    sees `N = B * S` independent sequences of length `G' + 1`.
    """

    def __init__(self, cfg: XCellConfig = XCELL_MINI):
        super().__init__()
        self.cfg = cfg
        self.embed = InputEmbedding(cfg)
        self.priors = nn.ModuleDict({
            name: PriorProjection(dim, cfg.prior_hidden, cfg.d_model,
                                  cfg.leaky_slope, cfg.norm_eps)
            for name, dim, _ in PRIOR_SOURCES
        })
        block_cls = LegacyBlock if cfg.block == "legacy" else ModernBlock
        cross = set(cfg.cross_attn_layers)
        self.blocks = nn.ModuleList(
            block_cls(cfg, cross=(l in cross)) for l in range(cfg.n_layers)
        )
        # "A final RMSNorm is applied after all L layers" (A.1.3). The legacy block already ends
        # on a LayerNorm, so a further norm there would be a second normalisation of the same
        # tensor and is not what scGPT does.
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.block == "modern" else None
        self.decoder = Decoder(cfg)

    def perturbation_context(self, priors: dict[str, Tensor], pert_token: Tensor) -> Tensor:
        """A6-A11: six d-dimensional context tokens, in the appendix's order."""
        toks = [self.priors[name](priors[name]) for name, _, _ in PRIOR_SOURCES]
        # A11: stop-gradient gene embedding of the perturbed gene, LayerNormed as in A2.
        gene = self.embed.gene_norm(self.embed.gene(pert_token)).detach()
        toks.append(gene)
        return torch.stack(toks, dim=1)                      # [N, 6, d]

    def forward(self, values: Tensor, tokens: Tensor, pert_mask: Tensor,
                priors: dict[str, Tensor], pert_token: Tensor,
                prior_missing: Tensor | None = None) -> Tensor:
        """
        values        [N, G] partially revealed expression, log1p CP10k
        tokens        [N, G] gene vocabulary indices
        pert_mask     [N, G] 1 where `values` holds revealed perturbed signal, else 0
        priors        name -> [N, dim] per the PRIOR_SOURCES table
        pert_token    [N] vocabulary index of the knocked-down gene
        prior_missing [N, 6] True where a source is absent (masked out of cross-attention)
        returns       [N, G] predicted perturbed expression
        """
        h0 = self.embed(values, tokens, pert_mask)
        context = self.perturbation_context(priors, pert_token)
        h = h0
        for block in self.blocks:
            h = block(h, context, prior_missing)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.cfg.global_residual:
            h = h + h0                                       # A23, off by default
        raw = self.embed.raw_gene_embedding(tokens)
        return self.decoder(h[:, 1:], h[:, 0], raw, tokens)


@torch.no_grad()
def predict(model: XCell, control: Tensor, tokens: Tensor, priors: dict[str, Tensor],
            pert_token: Tensor, prior_missing: Tensor | None = None,
            n_steps: int = 4, generator: torch.Generator | None = None) -> Tensor:
    """The 4-step cumulative diffusion refinement of A.4, "Inference (cumulative mode)".

    Ranks are drawn ONCE before the loop, so the reveal sets are nested and step `t` only touches
    genes whose rank falls between `alpha_{t-1}` and `alpha_t`. The final step reveals every gene
    (alpha_4 = 1.0). Returns the last forward pass's prediction, not the accumulated buffer.
    """
    if n_steps < 1 or n_steps > len(REVEAL_FRACTIONS):
        raise ValueError(f"n_steps must be 1..{len(REVEAL_FRACTIONS)}, got {n_steps}")
    n, g = control.shape
    x = control.clone()
    mask = torch.zeros_like(control, dtype=torch.long)
    ranks = torch.rand(n, g, device=control.device, generator=generator)
    prev = 0.0
    pred = None
    for alpha in REVEAL_FRACTIONS[:n_steps]:
        pred = model(x, tokens, mask, priors, pert_token, prior_missing)
        newly = (ranks <= alpha) & (ranks > prev)
        x = torch.where(newly, pred.to(x.dtype), x)
        mask = torch.where(newly, torch.ones_like(mask), mask)
        prev = alpha
    return pred
