"""AlphaFold2 (`model_1_ptm`) in torch: the reference of record for the AF2-IG port.

Transcribed from ColabDesign 1.1.3's vendored AlphaFold
(`colabdesign/af/alphafold/model/{modules,common_modules,all_atom,r3}.py`), which is what
PXDesign actually runs. It consumes `tt_bio.af2_weights.load_af2_state_dict` directly and
`tt_bio.af2_data`'s feature dict directly, so the ttnn port is scored against the same weights
and the same inputs as the reference.

**Precision follows AlphaFold's, which is the whole reason the per-block bar can be tight.**
`global_config.bfloat16` is on, so the trunk runs in bfloat16 while parameters stay float32 and
are cast on read (`utils.py:29-44`). Every module here casts its weight to the activation dtype
in `forward` rather than storing a bf16 copy, which is that behaviour and not an approximation of
it. LayerNorm is the exception AlphaFold makes and it matters: it upcasts bf16 to float32,
normalises with float32 scale and offset, and casts back (`common_modules.py:157-182`). Only the
fused triangle multiplication's two LayerNorms use the `mean(x^2) - mean(x)^2` variance
(`modules.py:913-923`); every other LayerNorm in the trunk uses `mean((x - mean)^2)`. The heads
run in float32 on the float32 trunk output, because `bfloat16_output` is False.

**The whole model is here.** The trunk (input embedder, recycling embedder, relpos, the
template stack, the 4 extra-MSA blocks, the 48 Evoformer blocks, `single_activations`), the
8-layer structure module, and the three heads. `load_af2_model` accounts for every parameter in
the checkpoint, so a typo in a key fails loudly rather than silently leaving a module at zero.
The structure module and the heads run in float32: `bfloat16_context` wraps the Evoformer only.

Two upstream details that read like bugs and are not:

* `Transition` expands its mask and then never uses it (`modules.py:277`). The transitions are
  genuinely unmasked.
* `GlobalAttention` takes a `bias` argument and immediately overwrites it with one derived from
  `q_mask` (`modules.py:449`). The MSA mask reaches it only through that path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tt_bio._vendor.esm.utils import residue_constants as _rc
from tt_bio.af2_data import (ATOM_ORDER, NUM_ATOM, RESTYPE_ATOM14_MASK,
                             RESTYPE_ATOM14_RIGID_GROUP_POSITIONS,
                             RESTYPE_ATOM14_TO_RIGID_GROUP,
                             RESTYPE_RIGID_GROUP_DEFAULT_FRAME)

C_M = 256
C_Z = 128
C_S = 384
C_EXTRA = 64
C_TEMPLATE = 64
NUM_EVOFORMER_BLOCKS = 48
NUM_EXTRA_MSA_BLOCKS = 4
NUM_TEMPLATE_BLOCKS = 2
MAX_RELATIVE_FEATURE = 32
# `embeddings_and_evoformer.prev_pos` -- 15 bins to 20.75 A, which is NOT the template
# distogram's 39 bins to 50.75 A. Both are in `af2ig_model_config.json`.
RECYCLE_DGRAM = (15, 3.25, 20.75)
TEMPLATE_DGRAM = (39, 3.25, 50.75)
PAE_BINS = 64
PAE_MAX_ERROR_BIN = 31.0
PLDDT_BINS = 50


def _chi_atom_indices() -> np.ndarray:
    """`all_atom.get_chi_atom_indices`, shape [21, 4, 4], zero-padded and with an UNK row.

    Read from the vendored ESM copy. Its `chi_angles_atoms`, `chi_angles_mask` and
    `chi_pi_periodic` are identical to AlphaFold's on every row, unlike the atom-existence masks
    (see `tt_bio.af2_data`), and the featurizer's test pins that.
    """
    table = []
    for letter in _rc.restypes:
        rows = [[ATOM_ORDER[a] for a in chi]
                for chi in _rc.chi_angles_atoms[_rc.restype_1to3[letter]]]
        rows += [[0, 0, 0, 0]] * (4 - len(rows))
        table.append(rows)
    table.append([[0, 0, 0, 0]] * 4)
    return np.asarray(table, dtype=np.int64)


CHI_ATOM_INDICES = _chi_atom_indices()
CHI_ANGLES_MASK = np.asarray(_rc.chi_angles_mask, dtype=np.float32)
CHI_PI_PERIODIC = np.asarray(_rc.chi_pi_periodic, dtype=np.float32)


# ---------------------------------------------------------------------------- primitives


class Linear(nn.Module):
    """AlphaFold's `common_modules.Linear`: haiku `[in, *out]` stored as torch `[out, in]`.

    The weight is cast to the activation dtype on every call, which is what AlphaFold's
    bfloat16 custom getter does.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


class Projection(nn.Module):
    """A bare `hk.get_parameter` projection. AlphaFold's attention q/k/v carry no bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class LayerNorm(nn.Module):
    """AlphaFold's LayerNorm: float32 math and float32 parameters even in a bfloat16 trunk."""

    def __init__(self, size: int, eps: float = 1e-5, fast_variance: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps
        self.fast_variance = fast_variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        mean = y.mean(-1, keepdim=True)
        if self.fast_variance:
            var = (y * y).mean(-1, keepdim=True) - mean * mean
        else:
            var = (y - mean).square().mean(-1, keepdim=True)
        out = (y - mean) * torch.rsqrt(var + self.eps) * self.weight.float() + self.bias.float()
        return out.to(dtype)


class Attention(nn.Module):
    """AlphaFold's `Attention` (`modules.py:312`). Subclassed by the four users of it.

    Subclassing rather than composing is deliberate: the checkpoint remap puts the attention
    projections at the same level as the wrapper's LayerNorms, so `msa_row_attn.linear_q.weight`
    and `msa_row_attn.layer_norm.weight` are siblings.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_head: int, key_dim: int, value_dim: int,
                 out_dim: int, gating: bool = True):
        super().__init__()
        self.num_head, self.key_dim, self.value_dim = num_head, key_dim, value_dim
        self.linear_q = Projection(q_dim, num_head * key_dim)
        self.linear_k = Projection(kv_dim, num_head * key_dim)
        self.linear_v = Projection(kv_dim, num_head * value_dim)
        if gating:
            self.linear_g = Linear(q_dim, num_head * value_dim)
        self.linear_o = Linear(num_head * value_dim, out_dim)

    def _attend(self, q_data: torch.Tensor, m_data: torch.Tensor, bias: torch.Tensor,
                nonbatched_bias: torch.Tensor | None = None) -> torch.Tensor:
        h, kd, vd = self.num_head, self.key_dim, self.value_dim
        batch, num_q = q_data.shape[0], q_data.shape[1]
        num_k = m_data.shape[1]
        q = self.linear_q(q_data).view(batch, num_q, h, kd) * kd ** -0.5
        k = self.linear_k(m_data).view(batch, num_k, h, kd)
        v = self.linear_v(m_data).view(batch, num_k, h, vd)
        logits = torch.einsum("bqhc,bkhc->bhqk", q, k) + bias
        if nonbatched_bias is not None:
            logits = logits + nonbatched_bias.unsqueeze(0)
        weights = torch.softmax(logits.clamp(-1e8, 1e8), dim=-1)
        out = torch.einsum("bhqk,bkhc->bqhc", weights, v)
        if hasattr(self, "linear_g"):
            out = out * torch.sigmoid(self.linear_g(q_data)).view(batch, num_q, h, vd)
        return self.linear_o(out.reshape(batch, num_q, h * vd))

    forward = _attend


class GlobalAttention(nn.Module):
    """AlphaFold's `GlobalAttention` (`modules.py:407`): one key head and one value head,
    shared across the query heads, with the query averaged over the masked axis."""

    def __init__(self, dim: int, num_head: int, key_dim: int, value_dim: int, out_dim: int):
        super().__init__()
        self.num_head, self.key_dim, self.value_dim = num_head, key_dim, value_dim
        self.linear_q = Projection(dim, num_head * key_dim)
        self.linear_k = Projection(dim, key_dim)
        self.linear_v = Projection(dim, value_dim)
        self.linear_g = Linear(dim, num_head * value_dim)
        self.linear_o = Linear(num_head * value_dim, out_dim)

    def _attend(self, q_data: torch.Tensor, m_data: torch.Tensor,
                q_mask: torch.Tensor) -> torch.Tensor:
        h, kd, vd = self.num_head, self.key_dim, self.value_dim
        batch, num_k = q_data.shape[0], q_data.shape[1]
        value = self.linear_v(m_data)
        # `utils.mask_mean` with its own epsilon; with an all-zero mask this is exactly zero.
        q_avg = (q_mask * q_data).sum(1) / (q_mask.sum(1) + 1e-10)
        q = self.linear_q(q_avg).view(batch, h, kd) * kd ** -0.5
        key = self.linear_k(m_data)
        bias = 1e9 * (q_mask[:, None, :, 0] - 1.0)
        weights = torch.softmax(torch.einsum("bhc,bkc->bhk", q, key) + bias, dim=-1)
        out = torch.einsum("bhk,bkc->bhc", weights, value)
        gate = torch.sigmoid(self.linear_g(q_data)).view(batch, num_k, h, vd)
        out = out.unsqueeze(1) * gate
        return self.linear_o(out.reshape(batch, num_k, h * vd))

    forward = _attend


class ReluTransition(nn.Module):
    """AlphaFold's `Transition`: LayerNorm, linear, ReLU, linear. Not SwiGLU, and unmasked."""

    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fc1 = Linear(dim, dim * factor)
        self.fc2 = Linear(dim * factor, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(self.norm(x))))


class TriangleMultiplication(nn.Module):
    """AlphaFold's fused `TriangleMultiplication` (`modules.py:1017`).

    `fuse_projection_weights` is True at every config site, so this is the branch that runs. The
    incoming direction takes `a` = AlphaFold's *right* and `b` its *left*, which is tt-bio's
    convention and the swap `tt_bio.af2_weights` applies to the concatenation.
    """

    def __init__(self, dim: int, hidden: int, ending: bool):
        super().__init__()
        self.hidden, self.ending = hidden, ending
        self.norm_in = LayerNorm(dim, fast_variance=True)
        self.p_in = Linear(dim, 2 * hidden)
        self.g_in = Linear(dim, 2 * hidden)
        self.norm_out = LayerNorm(hidden, fast_variance=True)
        self.p_out = Linear(hidden, dim)
        self.g_out = Linear(dim, dim)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(z)
        proj = mask.unsqueeze(-1).to(x.dtype) * self.p_in(x)
        proj = proj * torch.sigmoid(self.g_in(x))
        a, b = proj.split(self.hidden, dim=-1)
        equation = "kic,kjc->ijc" if self.ending else "ikc,jkc->ijc"
        act = self.p_out(self.norm_out(torch.einsum(equation, a, b)))
        return act * torch.sigmoid(self.g_out(x))


class TriangleAttention(Attention):
    """AlphaFold's `TriangleAttention`: one LayerNorm feeds both q/k/v and the pair bias."""

    def __init__(self, dim: int, num_head: int, head_dim: int, ending: bool):
        super().__init__(dim, dim, num_head, head_dim, head_dim, dim, gating=True)
        self.ending = ending
        self.layer_norm = LayerNorm(dim)
        self.linear = Projection(dim, num_head)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.ending:
            z, mask = z.transpose(-2, -3), mask.transpose(-1, -2)
        bias = (1e9 * (mask - 1.0)).to(z.dtype)[:, None, None, :]
        x = self.layer_norm(z)
        out = self._attend(x, x, bias, self.linear(x).permute(2, 0, 1))
        return out.transpose(-2, -3) if self.ending else out


class OuterProductMean(nn.Module):
    """AlphaFold's `OuterProductMean`. The output bias is inside the division by the pair norm,
    with epsilon 1e-3, which is tt-bio's `scale_bias=True`."""

    def __init__(self, c_m: int, hidden: int, c_z: int, eps: float = 1e-3):
        super().__init__()
        self.hidden, self.eps = hidden, eps
        self.norm = LayerNorm(c_m)
        self.proj_a = Linear(c_m, hidden)
        self.proj_b = Linear(c_m, hidden)
        self.proj_o = Linear(hidden * hidden, c_z)

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(msa.dtype)
        x = self.norm(msa)
        a = mask * self.proj_a(x)
        b = mask * self.proj_b(x)
        num_res = a.shape[1]
        outer = torch.einsum("sic,sje->ijce", a, b).reshape(num_res, num_res, -1)
        norm = torch.einsum("sic,sjc->ijc", mask, mask)
        return self.proj_o(outer) / (self.eps + norm)


class MsaRowAttentionWithPairBias(Attention):
    def __init__(self, c_m: int, c_z: int, num_head: int):
        super().__init__(c_m, c_m, num_head, c_m // num_head, c_m // num_head, c_m)
        self.layer_norm = LayerNorm(c_m)
        self.pair_norm = LayerNorm(c_z)
        self.linear = Projection(c_z, num_head)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor,
                pair: torch.Tensor) -> torch.Tensor:
        bias = (1e9 * (msa_mask - 1.0)).to(msa.dtype)[:, None, None, :]
        x = self.layer_norm(msa)
        return self._attend(x, x, bias, self.linear(self.pair_norm(pair)).permute(2, 0, 1))


class MsaColumnAttention(Attention):
    def __init__(self, c_m: int, num_head: int):
        super().__init__(c_m, c_m, num_head, c_m // num_head, c_m // num_head, c_m)
        self.layer_norm = LayerNorm(c_m)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        msa, msa_mask = msa.transpose(-2, -3), msa_mask.transpose(-1, -2)
        bias = (1e9 * (msa_mask - 1.0)).to(msa.dtype)[:, None, None, :]
        x = self.layer_norm(msa)
        return self._attend(x, x, bias).transpose(-2, -3)


class MsaColumnGlobalAttention(GlobalAttention):
    def __init__(self, c_m: int, num_head: int):
        super().__init__(c_m, num_head, c_m // num_head, c_m // num_head, c_m)
        self.layer_norm = LayerNorm(c_m)

    def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        msa, msa_mask = msa.transpose(-2, -3), msa_mask.transpose(-1, -2)
        x = self.layer_norm(msa)
        return self._attend(x, x, msa_mask.unsqueeze(-1).to(x.dtype)).transpose(-2, -3)


class PairBlock(nn.Module):
    """The pair track: two triangle multiplications, two triangle attentions, a transition.

    `evoformer_order=False` is the template pair stack, which runs the attentions *before* the
    multiplications (`modules.py:212-241` vs `modules.py:1330-1356`).
    """

    def __init__(self, c_z: int, hidden: int, num_head: int, head_dim: int, factor: int,
                 evoformer_order: bool = True):
        super().__init__()
        self.evoformer_order = evoformer_order
        self.tri_mul_out = TriangleMultiplication(c_z, hidden, ending=False)
        self.tri_mul_in = TriangleMultiplication(c_z, hidden, ending=True)
        self.tri_att_start = TriangleAttention(c_z, num_head, head_dim, ending=False)
        self.tri_att_end = TriangleAttention(c_z, num_head, head_dim, ending=True)
        self.pair_transition = ReluTransition(c_z, factor)

    def _pair_track(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        order = ((self.tri_mul_out, self.tri_mul_in, self.tri_att_start, self.tri_att_end)
                 if self.evoformer_order
                 else (self.tri_att_start, self.tri_att_end, self.tri_mul_out, self.tri_mul_in))
        for module in order:
            pair = pair + module(pair, pair_mask)
        return pair + self.pair_transition(pair)

    forward = _pair_track


class EvoformerBlock(PairBlock):
    """One `EvoformerIteration`. `outer_product_mean.first` is False, so the MSA track runs
    first and the outer product mean folds into the pair representation after it.

    Subclasses `PairBlock` for the same reason `TriangleAttention` subclasses `Attention`: the
    checkpoint remap puts the pair tracks parameters at the blocks own level, so
    `evoformer.0.tri_mul_out` and `evoformer.0.opm` are siblings.
    """

    def __init__(self, c_m: int, c_z: int, extra_msa: bool):
        super().__init__(c_z, 128, num_head=4, head_dim=32, factor=4)
        self.msa_row_attn = MsaRowAttentionWithPairBias(c_m, c_z, num_head=8)
        self.msa_col_attn = (MsaColumnGlobalAttention(c_m, 8) if extra_msa
                             else MsaColumnAttention(c_m, 8))
        self.msa_transition = ReluTransition(c_m, 4)
        self.opm = OuterProductMean(c_m, 32, c_z)

    def forward(self, msa, pair, msa_mask, pair_mask):
        msa = msa + self.msa_row_attn(msa, msa_mask, pair)
        msa = msa + self.msa_col_attn(msa, msa_mask)
        msa = msa + self.msa_transition(msa)
        pair = pair + self.opm(msa, msa_mask)
        return msa, self._pair_track(pair, pair_mask)


# ---------------------------------------------------------------------------- geometry


def dgram_from_positions(positions: torch.Tensor, num_bins: int, min_bin: float,
                         max_bin: float) -> torch.Tensor:
    """`modules.py::dgram_from_positions`. The last bin catches everything above `max_bin`."""
    lower = torch.linspace(min_bin, max_bin, num_bins, dtype=torch.float32).square()
    upper = torch.cat([lower[1:], torch.tensor([1e8])])
    delta = positions.float().unsqueeze(-2) - positions.float().unsqueeze(-3)
    dist2 = delta.square().sum(-1, keepdim=True)
    return ((dist2 > lower).float() * (dist2 < upper).float())


def pseudo_beta(aatype: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """`modules.py::pseudo_beta_fn` without the mask branch. Glycine takes CA, everything CB."""
    is_gly = (aatype == _rc.restype_order["G"]).unsqueeze(-1)
    return torch.where(is_gly, positions[..., ATOM_ORDER["CA"], :],
                       positions[..., ATOM_ORDER["CB"], :])


def _frame_local_coords(points: torch.Tensor) -> torch.Tensor:
    """The last of 4 atoms in the Gram-Schmidt frame of the first three.

    `r3.rigids_from_3_points` + `rigids_mul_vecs(invert_rigids(...))`, written out: the rotation's
    columns are (e0, e1, e2), so the local coordinates are the three dot products. `points` is
    `[..., 4, 3]` ordered (point on xy plane, point on negative x axis, origin, target).
    """
    p0, p1, origin, target = (points[..., i, :] for i in range(4))

    def normalize(v):
        return v / v.square().sum(-1, keepdim=True).add(1e-8).sqrt()

    e0 = normalize(origin - p1)
    v1 = p0 - origin
    e1 = normalize(v1 - (v1 * e0).sum(-1, keepdim=True) * e0)
    e2 = torch.cross(e0, e1, dim=-1)
    delta = target - origin
    return torch.stack([(e0 * delta).sum(-1), (e1 * delta).sum(-1), (e2 * delta).sum(-1)], -1)


def atom37_to_torsion_angles(aatype: torch.Tensor, positions: torch.Tensor,
                             mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """`all_atom.atom37_to_torsion_angles` with `placeholder_for_undefined=False`.

    Production passes `placeholder_for_undefined = not global_config.zero_init`, and `zero_init`
    is True, so undefined torsions stay at whatever the degenerate frame produced and are carried
    only by the mask. Batched over the template dimension.
    """
    aatype = aatype.clamp(max=20).long()
    num_templ, num_res = aatype.shape
    positions, mask = positions.float(), mask.float()

    zero_pos = torch.zeros(num_templ, 1, NUM_ATOM, 3)
    prev_pos = torch.cat([zero_pos, positions[:, :-1]], dim=1)
    prev_mask = torch.cat([torch.zeros(num_templ, 1, NUM_ATOM), mask[:, :-1]], dim=1)

    pre_omega = torch.cat([prev_pos[:, :, 1:3], positions[:, :, 0:2]], dim=-2)
    phi = torch.cat([prev_pos[:, :, 2:3], positions[:, :, 0:3]], dim=-2)
    psi = torch.cat([positions[:, :, 0:3], positions[:, :, 4:5]], dim=-2)
    pre_omega_mask = prev_mask[:, :, 1:3].prod(-1) * mask[:, :, 0:2].prod(-1)
    phi_mask = prev_mask[:, :, 2] * mask[:, :, 0:3].prod(-1)
    psi_mask = mask[:, :, 0:3].prod(-1) * mask[:, :, 4]

    chi_index = torch.from_numpy(CHI_ATOM_INDICES)[aatype]          # [T, R, 4, 4]
    gather = chi_index.reshape(num_templ, num_res, 16)
    chi_pos = torch.gather(positions, 2, gather[..., None].expand(-1, -1, -1, 3))
    chi_pos = chi_pos.reshape(num_templ, num_res, 4, 4, 3)
    chi_atom_mask = torch.gather(mask, 2, gather).reshape(num_templ, num_res, 4, 4).prod(-1)
    chi_mask = torch.from_numpy(CHI_ANGLES_MASK)[aatype] * chi_atom_mask

    atom_pos = torch.cat([pre_omega[:, :, None], phi[:, :, None], psi[:, :, None], chi_pos], 2)
    torsion_mask = torch.cat([pre_omega_mask[..., None], phi_mask[..., None],
                              psi_mask[..., None], chi_mask], dim=-1)

    local = _frame_local_coords(atom_pos)
    sin_cos = torch.stack([local[..., 2], local[..., 1]], dim=-1)
    sin_cos = sin_cos / sin_cos.square().sum(-1, keepdim=True).add(1e-8).sqrt()
    # psi was computed from the oxygen, so its sign is mirrored.
    sin_cos = sin_cos * torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[None, None, :, None]

    periodic = torch.from_numpy(CHI_PI_PERIODIC)[aatype]
    mirror = torch.cat([torch.ones(num_templ, num_res, 3), 1.0 - 2.0 * periodic], dim=-1)
    return {
        "torsion_angles_sin_cos": sin_cos,
        "alt_torsion_angles_sin_cos": sin_cos * mirror[..., None],
        "torsion_angles_mask": torsion_mask,
    }


# ---------------------------------------------------------------------------- template


class TemplateEmbedding(nn.Module):
    """`TemplateEmbedding` + `SingleTemplateEmbedding` + the torsion-angle MSA rows.

    The 88 input channels are 39 distogram bins, the pseudo-beta pair mask, the two tiled aatype
    one-hots at 22 each, three zero channels where the unit vector would go
    (`use_template_unit_vector` is False), and the backbone pair mask. The two masks are not the
    same array: the first comes from the pseudo-beta mask and the second from N/CA/C, and it is
    the second that multiplies the whole stack.
    """

    def __init__(self, c_z: int = C_Z, c_t: int = C_TEMPLATE,
                 num_blocks: int = NUM_TEMPLATE_BLOCKS):
        super().__init__()
        self.embedding2d = Linear(88, c_t)
        self.pair_stack = nn.ModuleList([
            PairBlock(c_t, 64, num_head=4, head_dim=16, factor=2, evoformer_order=False)
            for _ in range(num_blocks)])
        self.output_norm = LayerNorm(c_t)
        self.attn = Attention(c_z, c_t, num_head=4, key_dim=16, value_dim=16, out_dim=c_z,
                              gating=False)
        self.single_embedding = Linear(57, C_M)
        self.single_projection = Linear(C_M, C_M)

    def pair_representation(self, pair: torch.Tensor, feats: dict, mask_2d: torch.Tensor,
                            multichain_mask: torch.Tensor) -> torch.Tensor:
        dtype = pair.dtype
        num_res = pair.shape[0]
        out = []
        for t in range(feats["template_mask"].shape[0]):
            pb_mask = feats["template_pseudo_beta_mask"][t].float()
            mask_pb = pb_mask[:, None] * pb_mask[None, :] * multichain_mask
            dgram = dgram_from_positions(feats["template_pseudo_beta"][t], *TEMPLATE_DGRAM)
            aatype = F.one_hot(feats["template_aatype"][t].long(), 22).to(dtype)
            atom_mask = feats["template_all_atom_mask"][t].float()
            bb = atom_mask[:, ATOM_ORDER["N"]] * atom_mask[:, ATOM_ORDER["CA"]] \
                * atom_mask[:, ATOM_ORDER["C"]]
            mask_bb = (bb[:, None] * bb[None, :] * multichain_mask).to(dtype)
            act = torch.cat([
                (dgram * mask_pb[..., None]).to(dtype),
                mask_pb.to(dtype)[..., None],
                aatype[None].expand(num_res, -1, -1),
                aatype[:, None].expand(-1, num_res, -1),
                torch.zeros(num_res, num_res, 3, dtype=dtype),
                mask_bb[..., None],
            ], dim=-1) * mask_bb[..., None]
            act = self.embedding2d(act)
            for block in self.pair_stack:
                act = block(act, mask_2d)
            out.append(self.output_norm(act))
        return torch.stack(out, dim=0)

    def forward(self, pair: torch.Tensor, feats: dict, mask_2d: torch.Tensor,
                multichain_mask: torch.Tensor) -> torch.Tensor:
        template_repr = self.pair_representation(pair, feats, mask_2d, multichain_mask)
        num_res, c_z = pair.shape[0], pair.shape[-1]
        num_templ, c_t = template_repr.shape[0], template_repr.shape[-1]
        template_mask = feats["template_mask"].to(pair.dtype)
        flat_query = pair.reshape(num_res * num_res, 1, c_z)
        flat_templates = template_repr.permute(1, 2, 0, 3).reshape(num_res * num_res,
                                                                  num_templ, c_t)
        bias = 1e9 * (template_mask[None, None, None, :] - 1.0)
        embedding = self.attn(flat_query, flat_templates, bias).reshape(num_res, num_res, c_z)
        return embedding * (template_mask.sum() > 0).to(embedding.dtype)

    def torsion_rows(self, feats: dict, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The template rows appended to the MSA, and the mask row that goes with them.

        The mask is the psi angle's, because psi depends only on this residue's backbone.
        """
        ret = atom37_to_torsion_angles(feats["template_aatype"],
                                       feats["template_all_atom_positions"],
                                       feats["template_all_atom_mask"])
        num_templ, num_res = feats["template_aatype"].shape
        features = torch.cat([
            F.one_hot(feats["template_aatype"].long(), 22).float(),
            ret["torsion_angles_sin_cos"].reshape(num_templ, num_res, 14),
            ret["alt_torsion_angles_sin_cos"].reshape(num_templ, num_res, 14),
            ret["torsion_angles_mask"],
        ], dim=-1).to(dtype)
        rows = self.single_projection(F.relu(self.single_embedding(features)))
        return rows, ret["torsion_angles_mask"][:, :, 2].to(dtype)


# ---------------------------------------------------------------------- the structure module


def _quat_tables() -> tuple[np.ndarray, np.ndarray]:
    """`quat_affine.QUAT_TO_ROT` and `QUAT_MULTIPLY`, transcribed as written."""
    to_rot = np.zeros((4, 4, 3, 3), np.float32)
    to_rot[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    to_rot[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    to_rot[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    to_rot[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    to_rot[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
    to_rot[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]
    to_rot[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]
    to_rot[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]
    to_rot[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]
    to_rot[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]

    multiply = np.zeros((4, 4, 4), np.float32)
    multiply[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    multiply[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
    multiply[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]
    multiply[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]
    return to_rot, multiply


_TO_ROT, _MULTIPLY = _quat_tables()
QUAT_TO_ROT = torch.from_numpy(_TO_ROT.reshape(4, 4, 9))
QUAT_MULTIPLY_BY_VEC = torch.from_numpy(_MULTIPLY[:, 1:, :])

RIGID_GROUP_DEFAULT_FRAME = torch.from_numpy(RESTYPE_RIGID_GROUP_DEFAULT_FRAME)
RIGID_GROUP_POSITIONS = torch.from_numpy(RESTYPE_ATOM14_RIGID_GROUP_POSITIONS)
ATOM14_TO_RIGID_GROUP = torch.from_numpy(RESTYPE_ATOM14_TO_RIGID_GROUP)
ATOM14_MASK = torch.from_numpy(RESTYPE_ATOM14_MASK)


def quat_to_rot(quaternion: torch.Tensor) -> torch.Tensor:
    """A unit quaternion `[..., 4]` as a rotation matrix `[..., 3, 3]`."""
    flat = torch.einsum("abk,...a,...b->...k", QUAT_TO_ROT.to(quaternion.dtype),
                        quaternion, quaternion)
    return flat.unflatten(-1, (3, 3))


def quat_multiply_by_vec(quaternion: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """`quaternion` times the pure-vector quaternion `(0, vec)`."""
    return torch.einsum("abk,...a,...b->...k", QUAT_MULTIPLY_BY_VEC.to(quaternion.dtype),
                        quaternion, vec)


class QuatAffine:
    """`quat_affine.QuatAffine` with the three vector components in one trailing axis.

    Rotation is `[..., 3, 3]`, translation `[..., 3]`, and a point set is `[..., num_point, 3]`,
    which is AlphaFold's `extra_dims=1`: one frame per residue applied to that residue's points.
    Transcribed rather than adapted from `_vendor/openfold3`'s `Rigid`, whose conventions and
    normalisation policy differ enough that a mismatch would look like a plausible 0.99 PCC.
    """

    def __init__(self, quaternion: torch.Tensor, translation: torch.Tensor,
                 rotation: torch.Tensor | None = None, normalize: bool = True):
        if normalize:
            quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        self.quaternion = quaternion
        self.translation = translation
        self.rotation = quat_to_rot(quaternion) if rotation is None else rotation

    @classmethod
    def identity(cls, num_res: int, dtype: torch.dtype = torch.float32) -> "QuatAffine":
        """`folding.generate_new_affine`: the identity quaternion and a zero translation."""
        quaternion = torch.zeros(num_res, 4, dtype=dtype)
        quaternion[:, 0] = 1.0
        return cls(quaternion, torch.zeros(num_res, 3, dtype=dtype))

    def to_tensor(self) -> torch.Tensor:
        return torch.cat([self.quaternion, self.translation], dim=-1)

    def scale_translation(self, position_scale: float) -> "QuatAffine":
        return QuatAffine(self.quaternion, self.translation * position_scale,
                          rotation=self.rotation, normalize=False)

    def pre_compose(self, update: torch.Tensor) -> "QuatAffine":
        """Apply a 6-vector update: a pure-vector quaternion, then a local-frame translation.

        The translation update is rotated by the *old* rotation before it is added, and the new
        quaternion is renormalised unconditionally, which is `QuatAffine.__init__`'s default.
        """
        vector_update, trans_update = update.split(3, dim=-1)
        quaternion = self.quaternion + quat_multiply_by_vec(self.quaternion, vector_update)
        translation = self.translation + self.rotate(trans_update.unsqueeze(-2))[..., 0, :]
        return QuatAffine(quaternion, translation)

    def rotate(self, points: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ij,...nj->...ni", self.rotation, points)

    def apply_to_point(self, points: torch.Tensor) -> torch.Tensor:
        return self.rotate(points) + self.translation.unsqueeze(-2)

    def invert_point(self, points: torch.Tensor) -> torch.Tensor:
        centred = points - self.translation.unsqueeze(-2)
        return torch.einsum("...ji,...nj->...ni", self.rotation, centred)


def _compose(rot_a: torch.Tensor, trans_a: torch.Tensor,
             rot_b: torch.Tensor, trans_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`r3.rigids_mul_rigids`: apply b first, then a."""
    return rot_a @ rot_b, trans_a + torch.einsum("...ij,...j->...i", rot_a, trans_b)


class InvariantPointAttention(nn.Module):
    """AlphaFold's `InvariantPointAttention` (`folding.py:37`), Suppl. Alg. 22.

    Three logit terms of equal variance: scalar q.k, the squared distance between query and key
    points pushed into the global frame, and a projection of the pair representation. The value
    points come back into the query residue's local frame before the output projection.
    """

    def __init__(self, c_s: int = C_S, c_z: int = C_Z, num_head: int = 12,
                 num_scalar_qk: int = 16, num_scalar_v: int = 16,
                 num_point_qk: int = 4, num_point_v: int = 8):
        super().__init__()
        self.num_head = num_head
        self.num_scalar_qk, self.num_scalar_v = num_scalar_qk, num_scalar_v
        self.num_point_qk, self.num_point_v = num_point_qk, num_point_v
        self.q_scalar = Linear(c_s, num_head * num_scalar_qk)
        self.kv_scalar = Linear(c_s, num_head * (num_scalar_qk + num_scalar_v))
        self.q_point_local = Linear(c_s, num_head * 3 * num_point_qk)
        self.kv_point_local = Linear(c_s, num_head * 3 * (num_point_qk + num_point_v))
        self.attention_2d = Linear(c_z, num_head)
        self.point_weights = nn.Parameter(torch.zeros(num_head))
        self.output_projection = Linear(num_head * (num_scalar_v + 4 * num_point_v + c_z), c_s)

    @staticmethod
    def _as_points(flat: torch.Tensor) -> torch.Tensor:
        """`split(3, axis=-1)` then stack: x-block, y-block, z-block, not interleaved."""
        return torch.stack(flat.chunk(3, dim=-1), dim=-1)

    def forward(self, act: torch.Tensor, act_2d: torch.Tensor, mask: torch.Tensor,
                affine: QuatAffine) -> torch.Tensor:
        num_res = act.shape[0]
        h, sqk, sv = self.num_head, self.num_scalar_qk, self.num_scalar_v
        pqk, pv = self.num_point_qk, self.num_point_v

        q_scalar = self.q_scalar(act).view(num_res, h, sqk).transpose(0, 1)
        kv_scalar = self.kv_scalar(act).view(num_res, h, sqk + sv).transpose(0, 1)
        k_scalar, v_scalar = kv_scalar.split([sqk, sv], dim=-1)

        q_point = affine.apply_to_point(self._as_points(self.q_point_local(act)))
        q_point = q_point.view(num_res, h, pqk, 3).transpose(0, 1)
        kv_point = affine.apply_to_point(self._as_points(self.kv_point_local(act)))
        kv_point = kv_point.view(num_res, h, pqk + pv, 3).transpose(0, 1)
        k_point, v_point = kv_point.split([pqk, pv], dim=2)

        # Equal variance for each of the three logit terms; a point pair contributes 9/2.
        scalar_weights = (1.0 / (3 * max(sqk, 1))) ** 0.5
        point_weights = (1.0 / (3 * max(pqk, 1) * 4.5)) ** 0.5 * F.softplus(self.point_weights)

        dist2 = (q_point[:, :, None] - k_point[:, None]).square().sum(-1)
        logits = torch.einsum("hqc,hkc->hqk", scalar_weights * q_scalar, k_scalar)
        logits = logits - 0.5 * (point_weights[:, None, None, None] * dist2).sum(-1)
        logits = logits + (1.0 / 3) ** 0.5 * self.attention_2d(act_2d).permute(2, 0, 1)
        logits = logits - 1e5 * (1.0 - mask * mask.transpose(-1, -2))
        attn = torch.softmax(logits, dim=-1)

        scalar_out = torch.einsum("hqk,hkc->qhc", attn, v_scalar).reshape(num_res, h * sv)
        point_global = torch.einsum("hqk,hkpc->qhpc", attn, v_point)
        point_local = affine.invert_point(point_global.reshape(num_res, h * pv, 3))
        pair_out = torch.einsum("hij,ijc->ihc", attn, act_2d).reshape(num_res, -1)
        features = torch.cat([
            scalar_out,
            point_local[..., 0], point_local[..., 1], point_local[..., 2],
            (1e-8 + point_local.square().sum(-1)).sqrt(),
            pair_out,
        ], dim=-1)
        return self.output_projection(features)


def torsion_angles_to_frames(aatype: torch.Tensor, backb_rot: torch.Tensor,
                             backb_trans: torch.Tensor, angles: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """`all_atom.torsion_angles_to_frames`: the 8 rigid-group frames, in the global frame.

    A zero rotation is prepended for the backbone group, chi2 to chi4 chain onto the previous chi
    frame rather than onto the backbone, and the result is composed with `backb_to_global`.
    """
    default = RIGID_GROUP_DEFAULT_FRAME.to(angles.dtype)[aatype]
    num_res = aatype.shape[0]
    ones = torch.ones(num_res, 1, dtype=angles.dtype)
    zeros = torch.zeros(num_res, 1, dtype=angles.dtype)
    sin = torch.cat([zeros, angles[..., 0]], dim=-1)
    cos = torch.cat([ones, angles[..., 1]], dim=-1)
    zero, one = torch.zeros_like(sin), torch.ones_like(sin)
    rot_x = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos], -1).unflatten(-1, (3, 3))

    rot = list((default[..., :3, :3] @ rot_x).unbind(1))
    trans = list(default[..., :3, 3].unbind(1))
    for i in (5, 6, 7):
        rot[i], trans[i] = _compose(rot[i - 1], trans[i - 1], rot[i], trans[i])
    return _compose(backb_rot[:, None], backb_trans[:, None],
                    torch.stack(rot, 1), torch.stack(trans, 1))


def frames_to_atom14_positions(aatype: torch.Tensor, rot: torch.Tensor,
                               trans: torch.Tensor) -> torch.Tensor:
    """`all_atom.frames_and_literature_positions_to_atom14_pos`: one frame per atom14 slot."""
    group_mask = F.one_hot(ATOM14_TO_RIGID_GROUP[aatype], 8).to(rot.dtype)
    atom_rot = torch.einsum("rgij,rag->raij", rot, group_mask)
    atom_trans = torch.einsum("rgi,rag->rai", trans, group_mask)
    literature = RIGID_GROUP_POSITIONS.to(rot.dtype)[aatype]
    positions = torch.einsum("raij,raj->rai", atom_rot, literature) + atom_trans
    return positions * ATOM14_MASK.to(rot.dtype)[aatype].unsqueeze(-1)


def atom14_to_atom37(atom14: torch.Tensor, feats: dict) -> torch.Tensor:
    """`all_atom.atom14_to_atom37`, masked by `atom37_atom_exists` as production does."""
    index = feats["residx_atom37_to_atom14"].long().unsqueeze(-1).expand(-1, -1, 3)
    atom37 = torch.gather(atom14, 1, index)
    return atom37 * feats["atom37_atom_exists"].to(atom37.dtype).unsqueeze(-1)


class MultiRigidSidechain(nn.Module):
    """AlphaFold's `MultiRigidSidechain` (`folding.py:900`): 7 torsion angles, then atom14.

    The two input projections read `[act, initial_act]` in that order and are summed, which is
    the concatenation the checkpoint's two weight blocks encode.
    """

    def __init__(self, c_s: int = C_S, c_hidden: int = 128, num_residual_block: int = 2):
        super().__init__()
        self.input_projection = nn.ModuleList([Linear(c_s, c_hidden) for _ in range(2)])
        self.resblock = nn.ModuleList([
            nn.ModuleList([Linear(c_hidden, c_hidden) for _ in range(2)])
            for _ in range(num_residual_block)])
        self.angles = Linear(c_hidden, 14)

    def forward(self, affine: QuatAffine, representations: list[torch.Tensor],
                aatype: torch.Tensor) -> dict[str, torch.Tensor]:
        act = sum(projection(F.relu(x))
                  for projection, x in zip(self.input_projection, representations))
        for first, second in self.resblock:
            act = act + second(F.relu(first(F.relu(act))))
        unnormalized = self.angles(F.relu(act)).view(-1, 7, 2)
        angles = unnormalized / unnormalized.square().sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
        rot, trans = torsion_angles_to_frames(aatype, affine.rotation, affine.translation, angles)
        return {
            "angles_sin_cos": angles,
            "unnormalized_angles_sin_cos": unnormalized,
            "atom_pos": frames_to_atom14_positions(aatype, rot, trans),
        }


class AF2StructureModule(nn.Module):
    """`folding.StructureModule`: 8 weight-sharing IPA layers over an initially identity frame.

    Float32 throughout: `bfloat16_context` wraps `EmbeddingsAndEvoformer` only
    (`modules.py:1387`) and the trunk hands back float32 (`:1583`). The initial guess reaches the
    model through the recycling embedder, not through here, because `use_initial_atom_pos` is
    False in production and every residue starts at the identity quaternion.
    """

    def __init__(self, c_s: int = C_S, c_z: int = C_Z, num_layer: int = 8,
                 position_scale: float = 10.0):
        super().__init__()
        self.num_layer, self.position_scale = num_layer, position_scale
        self.single_norm = LayerNorm(c_s)
        self.pair_norm = LayerNorm(c_z)
        self.initial_projection = Linear(c_s, c_s)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.attention_norm = LayerNorm(c_s)
        self.transition = nn.ModuleList([Linear(c_s, c_s) for _ in range(3)])
        self.transition_norm = LayerNorm(c_s)
        self.affine_update = Linear(c_s, 6)
        self.sidechain = MultiRigidSidechain(c_s)

    def forward(self, single: torch.Tensor, pair: torch.Tensor, feats: dict) -> dict:
        act = self.single_norm(single)
        initial_act = act
        act = self.initial_projection(act)
        act_2d = self.pair_norm(pair)
        mask = feats["seq_mask"].to(act.dtype)[:, None]
        aatype = feats["aatype"].long()
        affine = QuatAffine.identity(act.shape[0], dtype=act.dtype)

        affines, sidechains = [], []
        for _ in range(self.num_layer):
            act = act + self.ipa(act, act_2d, mask, affine)
            act = self.attention_norm(act)
            residual = act
            for i, layer in enumerate(self.transition):
                act = layer(act)
                if i < len(self.transition) - 1:
                    act = F.relu(act)
            act = self.transition_norm(act + residual)
            affine = affine.pre_compose(self.affine_update(act))
            sidechains.append(self.sidechain(affine.scale_translation(self.position_scale),
                                             [act, initial_act], aatype))
            affines.append(affine.to_tensor())

        scale = torch.tensor([1.0] * 4 + [self.position_scale] * 3, dtype=act.dtype)
        traj = torch.stack(affines) * scale
        atom14 = sidechains[-1]["atom_pos"]
        # `sidechains/angles` and `sidechains/angles_sin_cos` are one array under two names.
        angles = torch.stack([sc["angles_sin_cos"] for sc in sidechains])
        return {
            "representations/structure_module": act,
            "traj": traj,
            "final_affines": traj[-1],
            "sidechains/angles": angles,
            "sidechains/angles_sin_cos": angles,
            "sidechains/unnormalized_angles_sin_cos":
                torch.stack([sc["unnormalized_angles_sin_cos"] for sc in sidechains]),
            "final_atom14_positions": atom14,
            "final_atom14_mask": feats["atom14_atom_exists"].to(act.dtype),
            "final_atom_positions": atom14_to_atom37(atom14, feats),
            "final_atom_mask": feats["atom37_atom_exists"].to(act.dtype),
        }


class PredictedLDDTHead(nn.Module):
    """`modules.PredictedLDDTHead`: 50 pLDDT bins off the last fold layer's activations."""

    def __init__(self, c_s: int = C_S, c_hidden: int = 128, num_bins: int = PLDDT_BINS):
        super().__init__()
        self.norm = LayerNorm(c_s)
        self.act = nn.ModuleList([Linear(c_s, c_hidden), Linear(c_hidden, c_hidden)])
        self.logits = Linear(c_hidden, num_bins)

    def forward(self, act: torch.Tensor) -> torch.Tensor:
        act = self.norm(act)
        for layer in self.act:
            act = F.relu(layer(act))
        return self.logits(act)


# ---------------------------------------------------------------------------- the trunk


class AF2Model(nn.Module):
    """The whole of `model_1_ptm`: the trunk, the structure module and the two confidence heads.

    One call is one recycling pass. ColabDesign's `recycle_mode="last"` loops in python rather
    than in graph (`design.py:147-205`), so the loop lives in `run_recycles` and `prev` is an
    argument.
    """

    def __init__(self, *, template: bool = True,
                 num_evoformer_blocks: int = NUM_EVOFORMER_BLOCKS,
                 num_extra_msa_blocks: int = NUM_EXTRA_MSA_BLOCKS,
                 trunk_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.trunk_dtype = trunk_dtype
        self.embed = nn.ModuleDict({
            "preprocess_1d": Linear(22, C_M),
            "preprocess_msa": Linear(49, C_M),
            "left_single": Linear(22, C_Z),
            "right_single": Linear(22, C_Z),
            "pair_activations": Linear(2 * MAX_RELATIVE_FEATURE + 1, C_Z),
            "extra_msa_activations": Linear(25, C_EXTRA),
        })
        self.recycle = nn.ModuleDict({
            "prev_pos_linear": Linear(RECYCLE_DGRAM[0], C_Z),
            "prev_msa_norm": LayerNorm(C_M),
            "prev_pair_norm": LayerNorm(C_Z),
        })
        self.template = TemplateEmbedding() if template else None
        self.extra_msa = nn.ModuleList(
            [EvoformerBlock(C_EXTRA, C_Z, extra_msa=True) for _ in range(num_extra_msa_blocks)])
        self.evoformer = nn.ModuleList(
            [EvoformerBlock(C_M, C_Z, extra_msa=False) for _ in range(num_evoformer_blocks)])
        self.single_activations = Linear(C_M, C_S)
        self.structure = AF2StructureModule()
        self.heads = nn.ModuleDict({"pae": nn.ModuleDict({"logits": Linear(C_Z, PAE_BINS)}),
                                    "plddt": PredictedLDDTHead()})

    def forward(self, feats: dict, prev: dict) -> dict:
        dtype = self.trunk_dtype
        target_feat = F.pad(feats["target_feat"].to(dtype), (1, 1))
        msa = (self.embed["preprocess_1d"](target_feat).unsqueeze(0)
               + self.embed["preprocess_msa"](feats["msa_feat"].to(dtype)))
        left = self.embed["left_single"](target_feat)
        right = self.embed["right_single"](target_feat)
        pair = left.unsqueeze(1) + right.unsqueeze(0)

        seq_mask = feats["seq_mask"].float()
        mask_2d = (seq_mask[:, None] * seq_mask[None, :]).to(dtype)

        dgram = dgram_from_positions(pseudo_beta(feats["aatype"], prev["prev_pos"]),
                                     *RECYCLE_DGRAM).to(dtype)
        pair = pair + self.recycle["prev_pos_linear"](dgram)
        msa = torch.cat([
            (msa[0] + self.recycle["prev_msa_norm"](prev["prev_msa_first_row"]).to(dtype))[None],
            msa[1:],
        ], dim=0)
        pair = pair + self.recycle["prev_pair_norm"](prev["prev_pair"]).to(dtype)

        offset = feats["residue_index"].long()[:, None] - feats["residue_index"].long()[None, :]
        rel_pos = F.one_hot((offset + MAX_RELATIVE_FEATURE).clamp(0, 2 * MAX_RELATIVE_FEATURE),
                            2 * MAX_RELATIVE_FEATURE + 1).to(dtype)
        pair = pair + self.embed["pair_activations"](rel_pos)

        if self.template is not None:
            asym = feats["asym_id"]
            same_chain = (asym[:, None] == asym[None, :])
            multichain = (same_chain if bool(feats["mask_template_interchain"])
                          else torch.ones_like(same_chain)).float()
            pair = pair + self.template(pair, feats, mask_2d, multichain)

        extra = self.embed["extra_msa_activations"](_extra_msa_feature(feats).to(dtype))
        extra_mask = feats["extra_msa_mask"].to(dtype)
        for block in self.extra_msa:
            extra, pair = block(extra, pair, extra_mask, mask_2d)

        msa_mask = feats["msa_mask"].to(dtype)
        if self.template is not None:
            rows, row_mask = self.template.torsion_rows(feats, dtype)
            msa = torch.cat([msa, rows], dim=0)
            msa_mask = torch.cat([msa_mask, row_mask], dim=0)

        for block in self.evoformer:
            msa, pair = block(msa, pair, msa_mask, mask_2d)

        single = self.single_activations(msa[0])
        num_sequences = feats["msa_feat"].shape[0]
        out = {
            "single": single.float(),
            "pair": pair.float(),
            "msa": msa[:num_sequences].float(),
            "msa_first_row": msa[0].float(),
        }
        out["pae_logits"] = self.heads["pae"]["logits"](out["pair"])
        out["pae_breaks"] = torch.linspace(0.0, PAE_MAX_ERROR_BIN, PAE_BINS - 1)
        out["structure"] = self.structure(out["single"], out["pair"], feats)
        out["plddt_logits"] = self.heads["plddt"](
            out["structure"]["representations/structure_module"])
        return out


def _extra_msa_feature(feats: dict) -> torch.Tensor:
    """`modules.py::create_extra_msa_feature`: a 23-wide one-hot plus the two deletion channels."""
    one_hot = F.one_hot(feats["extra_msa"].long(), 23).float()
    return torch.cat([one_hot,
                      feats["extra_has_deletion"].float().unsqueeze(-1),
                      feats["extra_deletion_value"].float().unsqueeze(-1)], dim=-1)


def load_af2_model(state_dict: dict[str, torch.Tensor], *, template: bool = True,
                   **kwargs) -> AF2Model:
    """Build an `AF2Model` and load a remapped checkpoint into it.

    Every parameter in the checkpoint has a home here, so nothing may be missing and nothing may
    be left over. The one exception is the template stack under `template=False`: the monomer
    stage runs the model_3_ptm config and ColabDesign drops those parameters at load
    (`af/model.py:112-120`) from the same params_model_1_ptm.npz.
    """
    model = AF2Model(template=template, **kwargs)
    allowed = () if template else ("template.",)
    wanted = set(model.state_dict())
    consumed = {k: v for k, v in state_dict.items() if k in wanted}
    leftover = [k for k in state_dict if k not in wanted and not k.startswith(allowed)]
    if leftover:
        raise AssertionError(f"{len(leftover)} checkpoint keys have no home in AF2Model: "
                             f"{sorted(leftover)[:12]}")
    missing = sorted(wanted - set(consumed))
    if missing:
        raise AssertionError(f"{len(missing)} AF2Model parameters are not in the checkpoint: "
                             f"{missing[:12]}")
    model.load_state_dict(consumed, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_recycles(model: AF2Model, feats: dict, prev: dict, num_recycles: int = 3) -> list[dict]:
    """`num_recycles + 1` forward passes with `prev` threaded through (`design.py:147-205`).

    Returns every pass's output dict, not just the last, because the reference taps score the
    first and the fourth. The first `prev` is `af2_data.initial_recycle_state`, which already
    carries the initial guess; nothing else changes between passes.
    """
    outputs = []
    for _ in range(num_recycles + 1):
        out = model(feats, prev)
        outputs.append(out)
        prev = {"prev_msa_first_row": out["msa_first_row"],
                "prev_pair": out["pair"],
                "prev_pos": out["structure"]["final_atom_positions"]}
    return outputs
