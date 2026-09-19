"""ABodyBuilder3 in torch: the reference of record for the device port.

ABodyBuilder3 is an antibody Fv predictor with no trunk at all. Its whole model is OpenFold's
structure module driven from two hand-built features -- a 23-channel one-hot single (21 amino
acid types plus heavy/light chain) and a 132-channel pair (a clamped relative-position one-hot
plus a 3-channel chain-pair one-hot) -- so this file is the 8 IPA blocks, the backbone update,
the angle resnet and the pLDDT head, and nothing else. Config from upstream `params.yaml`
(`model:`) with `c_z` built in `stages/train.py:42-45`: `2 * rel_pos_dim + 1 + 3 = 132`.

Transcribed from `Exscientia/ABodyBuilder3` (Apache-2.0)
`src/abodybuilder3/openfold/model/structure_module.py` at `use_original_sm=True`, which is what
`params.yaml` sets and what every released checkpoint was trained with. Where upstream carries a
branch the released recipe never takes (`use_original_sm=False`, `inplace_safe`,
`_offload_inference`, the fp16 autocast path) the branch is not here; the taken branch is.

**The frame algebra is `af2_reference`'s, not a second copy.** `QuatAffine.pre_compose` is
`Rigid.compose_q_update_vec` op for op -- pure-vector quaternion update, renormalise, translation
update rotated by the OLD rotation -- and `apply_to_point`/`invert_point` are `Rigid.apply`/
`invert_apply` at `extra_dims=1`. `scripts/abb3_port/reference_gate.py` proves that numerically
against upstream's own `rigid_utils` in float64 rather than asserting it. The torsion and atom14
functions here are `af2_reference`'s two, batched over a leading sample axis, and the gate scores
them against the unbatched originals as well as against upstream.

Every module and parameter is named as upstream names it, so `load_abb3_model` is a prefix strip
and nothing else, and a missing or spare key fails loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .af2_data import (RESTYPE_ATOM14_MASK, RESTYPE_ATOM14_RIGID_GROUP_POSITIONS,
                       RESTYPE_ATOM14_TO_RIGID_GROUP, RESTYPE_RIGID_GROUP_DEFAULT_FRAME)
from .af2_reference import Linear, QuatAffine, _compose

#: Upstream `params.yaml` `model:`, with `c_z` as `stages/train.py` computes it and `use_plddt`
#: as `lightning_module.py:95` derives it (`loss.plddt.weight > 0`). The released `base-loss`
#: checkpoint is `use_plddt=False` and has 7,111,515 parameters; the pLDDT head is 111,922 more.
CONFIG = dict(
    c_s=23, embed_dim=128, c_z=132, c_ipa=16, c_resnet=256, no_heads_ipa=12, no_qk_points=4,
    no_v_points=8, dropout_rate=0.1, no_blocks=8, no_transition_layers=1, no_resnet_blocks=2,
    no_angles=7, trans_scale_factor=1, epsilon=1.0e-7, inf=1.0e7,
)
REL_POS_DIM = 64
PLDDT_BINS = 50
PLDDT_HIDDEN = 256


@dataclass(frozen=True)
class ABB3Config:
    c_s: int = 23
    embed_dim: int = 128
    c_z: int = 132
    c_ipa: int = 16
    c_resnet: int = 256
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: float = 1.0
    epsilon: float = 1.0e-7
    inf: float = 1.0e7
    use_plddt: bool = False


class LayerNorm(nn.Module):
    """`openfold.model.primitives.LayerNorm`: `F.layer_norm` with eps 1e-5, dtype preserved.

    Deliberately NOT `af2_reference.LayerNorm`, which upcasts to float32 unconditionally because
    AF2 runs a bfloat16 trunk. That upcast would silently truncate this file's float64 reference
    to float32 and cap every error in the port's parity table at float32 noise.
    """

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)


def _as_points(flat: torch.Tensor) -> torch.Tensor:
    """`split(3, dim=-1)` then stack: an x-block, a y-block and a z-block, not interleaved."""
    return torch.stack(flat.chunk(3, dim=-1), dim=-1)


class InvariantPointAttention(nn.Module):
    """Alg. 22, `structure_module.py:178` at `ipa_bias=True`.

    Three logit terms of equal variance -- scalar q.k, the squared distance between query and key
    points in the global frame, and a projection of the pair representation -- and an output that
    concatenates the scalar context, the value points brought back into the query residue's local
    frame, their norms and the pair context.
    """

    def __init__(self, cfg: ABB3Config):
        super().__init__()
        c_s, h, c = cfg.embed_dim, cfg.no_heads_ipa, cfg.c_ipa
        self.cfg = cfg
        self.linear_q = Linear(c_s, h * c)
        self.linear_kv = Linear(c_s, 2 * h * c)
        self.linear_q_points = Linear(c_s, h * cfg.no_qk_points * 3)
        self.linear_kv_points = Linear(c_s, h * (cfg.no_qk_points + cfg.no_v_points) * 3)
        self.linear_b = Linear(c_s, h)
        #: Allocated at zero and filled by `abb3_init.initialise_` with upstream's
        #: `ipa_point_weights_init_` (structure_module.py:243-244), which is not zero.
        self.head_weights = nn.Parameter(torch.zeros(h))
        self.linear_out = Linear(h * (c_s + c + cfg.no_v_points * 4), c_s, init="final")

    def forward(self, s: torch.Tensor, z: torch.Tensor, affine: QuatAffine,
                mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h, c, pq, pv = cfg.no_heads_ipa, cfg.c_ipa, cfg.no_qk_points, cfg.no_v_points
        lead = s.shape[:-1]

        q = self.linear_q(s).view(*lead, h, c)
        kv = self.linear_kv(s).view(*lead, h, 2 * c)
        k, v = kv.split(c, dim=-1)

        q_pts = affine.apply_to_point(_as_points(self.linear_q_points(s))).view(*lead, h, pq, 3)
        kv_pts = affine.apply_to_point(_as_points(self.linear_kv_points(s)))
        kv_pts = kv_pts.view(*lead, h, pq + pv, 3)
        k_pts, v_pts = kv_pts.split([pq, pv], dim=-2)

        # [*, H, N, N], scalar term scaled so each of the three terms has unit variance.
        logits = torch.einsum("...qhc,...khc->...hqk", q, k) * math.sqrt(1.0 / (3 * c))
        logits = logits + math.sqrt(1.0 / 3) * self.linear_b(z).movedim(-1, -3)
        dist2 = (q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)).square().sum(-1).sum(-1)
        head_weights = F.softplus(self.head_weights) * math.sqrt(1.0 / (3 * (pq * 9.0 / 2)))
        logits = logits - 0.5 * (dist2 * head_weights).movedim(-1, -3)
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        logits = logits + cfg.inf * (square_mask - 1).unsqueeze(-3)
        attn = torch.softmax(logits, dim=-1)

        o = torch.einsum("...hqk,...khc->...qhc", attn, v).reshape(*lead, h * c)
        o_pt = torch.einsum("...hqk,...khpc->...qhpc", attn, v_pts)
        o_pt = affine.invert_point(o_pt.reshape(*lead, h * pv, 3)).view(*lead, h, pv, 3)
        o_pt_norm = (o_pt.square().sum(-1) + cfg.epsilon).sqrt().reshape(*lead, h * pv)
        o_pt = o_pt.reshape(*lead, h * pv, 3)
        o_pair = torch.einsum("...hqk,...qkc->...qhc", attn, z).reshape(*lead, h * z.shape[-1])
        return self.linear_out(torch.cat(
            [o, o_pt[..., 0], o_pt[..., 1], o_pt[..., 2], o_pt_norm, o_pair], dim=-1))


class StructureModuleTransition(nn.Module):
    """`structure_module.py:466` at `no_transition_layers=1`: a 3-linear ReLU residual, then LN."""

    def __init__(self, cfg: ABB3Config):
        super().__init__()
        c = cfg.embed_dim
        self.layers = nn.ModuleList([_TransitionLayer(c) for _ in range(cfg.no_transition_layers)])
        self.layer_norm = LayerNorm(c)

    def forward(self, s: torch.Tensor, dropout=None) -> torch.Tensor:
        for layer in self.layers:
            s = layer(s)
        if dropout is not None:
            s = dropout(s)
        return self.layer_norm(s)


class _TransitionLayer(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.linear_1 = Linear(c, 2 * c, init="relu")
        self.linear_2 = Linear(2 * c, 2 * c, init="relu")
        self.linear_3 = Linear(2 * c, c, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s + self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(s)))))


class AngleResnet(nn.Module):
    """Alg. 20 lines 11-14, `structure_module.py:82` at `use_original_sm=True`.

    `use_original_sm` is the difference between ABodyBuilder3's own variant and AlphaFold's: the
    original projects `s` and `s_initial` separately and adds them, and its resnet block has two
    linears rather than three. Both released recipes set it, so only that branch is here.
    """

    def __init__(self, cfg: ABB3Config):
        super().__init__()
        c_in, c = cfg.embed_dim, cfg.c_resnet
        self.eps = cfg.epsilon
        self.linear_in = Linear(c_in, c)
        self.linear_initial = Linear(c_in, c)
        self.layers = nn.ModuleList([_AngleResnetBlock(c) for _ in range(cfg.no_resnet_blocks)])
        self.linear_out = Linear(c, cfg.no_angles * 2)

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.linear_in(F.relu(s)) + self.linear_initial(F.relu(s_initial))
        for layer in self.layers:
            s = layer(s)
        s = self.linear_out(F.relu(s)).unflatten(-1, (-1, 2))
        norm = s.square().sum(-1, keepdim=True).clamp(min=self.eps).sqrt()
        return s, s / norm


class _AngleResnetBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.linear_2 = Linear(c, c, init="relu")
        self.linear_3 = Linear(c, c, init="final")

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return a + self.linear_3(F.relu(self.linear_2(F.relu(a))))


class BackboneUpdate(nn.Module):
    """Alg. 23 line 10: the 6-vector that pre-composes onto the running frame."""

    def __init__(self, cfg: ABB3Config):
        super().__init__()
        self.linear = Linear(cfg.embed_dim, 6, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


class PerResidueLDDTCaPredictor(nn.Module):
    """`heads.py:66`: LN, two ReLU linears, 50 bins. Absent from the `base-loss` checkpoint."""

    def __init__(self, cfg: ABB3Config):
        super().__init__()
        self.layer_norm = LayerNorm(cfg.embed_dim)
        self.linear_1 = Linear(cfg.embed_dim, PLDDT_HIDDEN, init="relu")
        self.linear_2 = Linear(PLDDT_HIDDEN, PLDDT_HIDDEN, init="relu")
        self.linear_3 = Linear(PLDDT_HIDDEN, PLDDT_BINS, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = self.layer_norm(s)
        return self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(s)))))


def torsion_angles_to_frames(aatype: torch.Tensor, backb_rot: torch.Tensor,
                             backb_trans: torch.Tensor, angles: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """`af2_reference.torsion_angles_to_frames`, batched over a leading sample axis.

    Identical arithmetic: a zero rotation prepended for the backbone group, chi2 to chi4 chained
    onto the previous chi frame rather than onto the backbone, the result composed with
    `backb_to_global`. The only change is that the group axis is -3 instead of 1, because there
    is a sample axis in front of the residue axis now. Scored against the unbatched original in
    float64 by `scripts/abb3_port/reference_gate.py`.
    """
    default = torch.as_tensor(RESTYPE_RIGID_GROUP_DEFAULT_FRAME, dtype=angles.dtype,
                              device=angles.device)[aatype]
    lead = aatype.shape
    ones = torch.ones(*lead, 1, dtype=angles.dtype, device=angles.device)
    zeros = torch.zeros(*lead, 1, dtype=angles.dtype, device=angles.device)
    sin = torch.cat([zeros, angles[..., 0]], dim=-1)
    cos = torch.cat([ones, angles[..., 1]], dim=-1)
    zero, one = torch.zeros_like(sin), torch.ones_like(sin)
    rot_x = torch.stack([one, zero, zero, zero, cos, -sin, zero, sin, cos],
                        dim=-1).unflatten(-1, (3, 3))

    rot = list((default[..., :3, :3] @ rot_x).unbind(-3))
    trans = list(default[..., :3, 3].unbind(-2))
    for i in (5, 6, 7):
        rot[i], trans[i] = _compose(rot[i - 1], trans[i - 1], rot[i], trans[i])
    return _compose(backb_rot.unsqueeze(-3), backb_trans.unsqueeze(-2),
                    torch.stack(rot, dim=-3), torch.stack(trans, dim=-2))


def frames_to_atom14_positions(aatype: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor
                               ) -> torch.Tensor:
    """`af2_reference.frames_to_atom14_positions`, batched. One of the 8 frames per atom14 slot."""
    dev, dt = rot.device, rot.dtype
    group = torch.as_tensor(RESTYPE_ATOM14_TO_RIGID_GROUP, device=dev)[aatype]
    group_mask = F.one_hot(group, 8).to(dt)
    atom_rot = torch.einsum("...gij,...ag->...aij", rot, group_mask)
    atom_trans = torch.einsum("...gi,...ag->...ai", trans, group_mask)
    literature = torch.as_tensor(RESTYPE_ATOM14_RIGID_GROUP_POSITIONS, dtype=dt,
                                 device=dev)[aatype]
    positions = torch.einsum("...aij,...aj->...ai", atom_rot, literature) + atom_trans
    mask = torch.as_tensor(RESTYPE_ATOM14_MASK, dtype=dt, device=dev)[aatype]
    return positions * mask.unsqueeze(-1)


def frames_to_tensor_4x4(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """`Rigid.to_tensor_4x4`: the homogeneous form the sidechain FAPE term consumes."""
    out = torch.zeros(*rot.shape[:-2], 4, 4, dtype=rot.dtype, device=rot.device)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    out[..., 3, 3] = 1.0
    return out


class ABB3StructureModule(nn.Module):
    """The whole model: `linear_in_*`, then `no_blocks` of IPA / transition / update / angles.

    The pair representation is rebuilt every block, because its last channel is the pairwise
    distance between the current frames' translations (`structure_module.py:888`) and the frames
    move. Only that one channel moves: `linear_in_edge` runs once on the input features and its
    127 channels are shared by all 8 blocks.
    """

    def __init__(self, cfg: ABB3Config = ABB3Config()):
        super().__init__()
        self.cfg = cfg
        self.linear_in_node = Linear(cfg.c_s, cfg.embed_dim)
        self.linear_in_edge = Linear(cfg.c_z, cfg.embed_dim - 1)
        self.ipa_layers = nn.ModuleList([InvariantPointAttention(cfg) for _ in range(cfg.no_blocks)])
        self.layer_norm_ipa_layers = nn.ModuleList(
            [LayerNorm(cfg.embed_dim) for _ in range(cfg.no_blocks)])
        self.transition_layers = nn.ModuleList(
            [StructureModuleTransition(cfg) for _ in range(cfg.no_blocks)])
        self.bb_update_layers = nn.ModuleList([BackboneUpdate(cfg) for _ in range(cfg.no_blocks)])
        self.angle_resnet_layers = nn.ModuleList([AngleResnet(cfg) for _ in range(cfg.no_blocks)])
        if cfg.use_plddt:
            self.plddt = PerResidueLDDTCaPredictor(cfg)

    def forward(self, single: torch.Tensor, pair: torch.Tensor, aatype: torch.Tensor,
                mask: torch.Tensor, dropout=None) -> dict:
        cfg = self.cfg
        z_initial = self.linear_in_edge(pair)
        s = self.linear_in_node(single)
        s_initial = s

        quat = torch.zeros(*s.shape[:-1], 4, dtype=s.dtype, device=s.device)
        quat[..., 0] = 1.0
        affine = QuatAffine(quat, torch.zeros_like(quat[..., :3]))

        square_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1)
        out: dict[str, list] = {k: [] for k in
                                ("frames", "sidechain_frames", "unnormalized_angles", "angles",
                                 "positions", "states")}
        for i in range(cfg.no_blocks):
            trans = affine.translation
            # `torch.linalg.vector_norm` and not `sum().sqrt()`: the diagonal of this map is
            # exactly zero and is NOT masked away, and sqrt has an infinite derivative there
            # while torch defines the 2-norm backward as zero at the origin. Same values, and
            # the difference between a trainable model and NaN at the first step.
            dist = torch.linalg.vector_norm(trans.unsqueeze(-2) - trans.unsqueeze(-3), dim=-1)
            z = torch.cat([z_initial, dist.unsqueeze(-1) * square_mask], dim=-1)

            s = s + self.ipa_layers[i](s, z, affine, mask)
            if dropout is not None:
                s = dropout(s)
            s = self.layer_norm_ipa_layers[i](s)
            s = self.transition_layers[i](s, dropout=dropout)

            affine = affine.pre_compose(self.bb_update_layers[i](s))
            scaled = affine.scale_translation(cfg.trans_scale_factor)
            unnormalized_angles, angles = self.angle_resnet_layers[i](s, s_initial)
            rot, trans_g = torsion_angles_to_frames(aatype, scaled.rotation, scaled.translation,
                                                    angles)
            out["frames"].append(scaled.to_tensor())
            out["sidechain_frames"].append(frames_to_tensor_4x4(rot, trans_g))
            out["unnormalized_angles"].append(unnormalized_angles)
            out["angles"].append(angles)
            out["positions"].append(frames_to_atom14_positions(aatype, rot, trans_g))
            out["states"].append(s)

        stacked = {k: torch.stack(v) for k, v in out.items()}
        stacked["single"] = s
        if cfg.use_plddt:
            stacked["plddt"] = self.plddt(s)
        return stacked


def load_abb3_model(state_dict: dict, *, use_plddt: bool | None = None) -> ABB3StructureModule:
    """Load a released ABodyBuilder3 checkpoint. Keys are upstream's, under a `model.` prefix.

    `use_plddt` is inferred from the checkpoint rather than configured: `base-loss` has no pLDDT
    head, `plddt-loss` has one, and asking the caller to know which would be a way to load half a
    model silently.
    """
    weights = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    if not weights:
        weights = dict(state_dict)
    if use_plddt is None:
        use_plddt = any(k.startswith("plddt.") for k in weights)
    model = ABB3StructureModule(ABB3Config(use_plddt=use_plddt))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise KeyError(f"checkpoint does not match the module tree: "
                       f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}")
    return model.eval()


def single_and_pair_features(aatype: torch.Tensor, is_heavy: torch.Tensor,
                             residue_index: torch.Tensor, *, rel_pos_dim: int = REL_POS_DIM,
                             dtype: torch.dtype = torch.float32
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """`ABDataset.single_and_double_from_datapoint` at the released settings.

    Single is a 21-way amino-acid one-hot beside a 2-way chain one-hot. Pair is a 3-way one-hot
    of the chain pair (light/light 1, heavy/heavy 2, mixed 0) in front of a relative-position
    one-hot clamped to +-64, so 3 + 129 = 132 channels.
    """
    single = torch.cat([F.one_hot(aatype.long(), 21), F.one_hot(is_heavy.long(), 2)], dim=-1)
    rel = residue_index.unsqueeze(-2) - residue_index.unsqueeze(-1)
    rel = rel.clamp(-rel_pos_dim, rel_pos_dim) + rel_pos_dim
    pair = F.one_hot(rel.long(), 2 * rel_pos_dim + 1)
    heavy = is_heavy.long()
    chain = 2 * (heavy.unsqueeze(-1) * heavy.unsqueeze(-2)) + \
        ((1 - heavy).unsqueeze(-1) * (1 - heavy).unsqueeze(-2))
    pair = torch.cat([F.one_hot(chain, 3), pair], dim=-1)
    return single.to(dtype), pair.to(dtype)
