"""ABodyBuilder3 ttnn port — structure-module components on device.

ABodyBuilder3 (Exscientia, Apache-2.0; vendored under ``tt_bio/_vendor/abodybuilder3``)
is a single, MSA-free, one-hot antibody Fv structure module: 8 invariant-point-
attention (IPA) update blocks + a single pLDDT head. This module ports the
structure module to ttnn, reusing the shared ``tt_bio.tenstorrent`` primitives.

Scope of this file (first ttnn chunk): the standard, non-novel structure-module
submodules — the post-IPA LayerNorm, the single-representation Transition, the
BackboneUpdate linear, the AngleResnet, and the pLDDT head — ported to ttnn and
PCC-gated against the reference golden (``scripts/abb3_golden.py``). These are
all Linear/LayerNorm/ReLU/residual ops that map directly onto ``ttnn.linear`` /
``ttnn.layer_norm`` / ``ttnn.relu``.

The genuinely novel op — InvariantPointAttention's per-residue rigid-frame point
rotation, N^2*H*P_q squared-distance logits, value-point aggregation, and
``invert_apply`` under ttnn's tiled shape/layout constraints — is the long pole
of the port and lands in a follow-on chunk; it has no reusable primitive in tt-bio
(tt-bio's ESMFold2 is a diffusion folder, not an AF2 structure module).

The AngleResnet's final size-2 torsion normalization (sqrt(sum(s**2, -1)+eps),
divide) is a tiny, cheap op on an awkward (<32) trailing dim, so it is kept as a
host-side fp32 tail (``normalize_angles``) — the same "keep cheap host code as-is"
boundary the ESMFold2 port uses for its confidence head.
"""
from __future__ import annotations

import math

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    WeightScope,
    Weights,
    CORE_GRID_MAIN,
    _dtype,
    get_device,
)


def abb3_compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """HiFi4 + fp32 dest accumulation for the ABodyBuilder3 linears — the same
    config the other tt-bio models use, so bf16 weights still accumulate in fp32
    (parity-safe for the 128/256-channel linears)."""
    dev = get_device()
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _from_torch(x: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=_dtype(),
    )


def _to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x).to(torch.float32)


class IPALayerNorm(Module):
    """The LayerNorm applied after each IPA block (c=embed_dim=128)."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("weight")
        self.bias = self.torch_to_tt("bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x, weight=self.weight, bias=self.bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class BackboneUpdate(Module):
    """BackboneUpdate (Alg. 23 part): a single Linear(c_s -> 6) producing the
    per-residue quaternion-translation update vector."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("linear.weight")
        self.bias = self.torch_to_tt("linear.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        return self._lin(
            s, self.weight, bias=self.bias, dtype=_dtype(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class PLDDTHead(Module):
    """PerResidueLDDTCaPredictor: LayerNorm -> Linear -> ReLU -> Linear -> ReLU ->
    Linear(no_bins). Returns [*, N, 50] pLDDT logits."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.norm_weight = self.torch_to_tt("layer_norm.weight")
        self.norm_bias = self.torch_to_tt("layer_norm.bias")
        self.l1_weight = self.torch_to_tt("linear_1.weight")
        self.l1_bias = self.torch_to_tt("linear_1.bias")
        self.l2_weight = self.torch_to_tt("linear_2.weight")
        self.l2_bias = self.torch_to_tt("linear_2.bias")
        self.l3_weight = self.torch_to_tt("linear_3.weight")
        self.l3_bias = self.torch_to_tt("linear_3.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.layer_norm(
            s, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = self._lin(x, self.l1_weight, bias=self.l1_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l2_weight, bias=self.l2_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l3_weight, bias=self.l3_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class StructureModuleTransition(Module):
    """StructureModuleTransition (Alg. 23 lines 8-9): one TransitionLayer
    (Linear->ReLU->Linear->ReLU->Linear + residual) then dropout (identity at
    eval) + LayerNorm. no_transition_layers=1 for the pldt-loss checkpoint."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.norm_weight = self.torch_to_tt("layer_norm.weight")
        self.norm_bias = self.torch_to_tt("layer_norm.bias")
        # Single transition layer (no_transition_layers=1).
        self.l1_weight = self.torch_to_tt("layers.0.linear_1.weight")
        self.l1_bias = self.torch_to_tt("layers.0.linear_1.bias")
        self.l2_weight = self.torch_to_tt("layers.0.linear_2.weight")
        self.l2_bias = self.torch_to_tt("layers.0.linear_2.bias")
        self.l3_weight = self.torch_to_tt("layers.0.linear_3.weight")
        self.l3_bias = self.torch_to_tt("layers.0.linear_3.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        s_initial = s
        x = self._lin(s, self.l1_weight, bias=self.l1_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l2_weight, bias=self.l2_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l3_weight, bias=self.l3_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, s_initial, memory_config=ttnn.L1_MEMORY_CONFIG)
        # dropout is identity at eval; final LayerNorm.
        return ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class AngleResnet(Module):
    """AngleResnet (Alg. 20 lines 11-14), use_original_sm=True path:
    relu+linear_initial on s_initial, relu+linear_in on s, sum, then 2 resnet
    blocks (relu+linear_2, relu+linear_3, +residual), relu, linear_out -> [*, 14].

    Returns the UNNORMALIZED [*, N, 14] tensor on device. The final size-2 torsion
    normalization is a tiny host-side fp32 tail (``normalize_angles``) — the
    awkward <32 trailing dim is not worth a tiled device reduction, mirroring the
    ESMFold2 "cheap host code" boundary."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig,
                 epsilon: float = 1e-7):
        super().__init__(state_dict, ck)
        self.eps = epsilon
        self.lin_in_weight = self.torch_to_tt("linear_in.weight")
        self.lin_in_bias = self.torch_to_tt("linear_in.bias")
        self.lin_initial_weight = self.torch_to_tt("linear_initial.weight")
        self.lin_initial_bias = self.torch_to_tt("linear_initial.bias")
        self.b0_l2_weight = self.torch_to_tt("layers.0.linear_2.weight")
        self.b0_l2_bias = self.torch_to_tt("layers.0.linear_2.bias")
        self.b0_l3_weight = self.torch_to_tt("layers.0.linear_3.weight")
        self.b0_l3_bias = self.torch_to_tt("layers.0.linear_3.bias")
        self.b1_l2_weight = self.torch_to_tt("layers.1.linear_2.weight")
        self.b1_l2_bias = self.torch_to_tt("layers.1.linear_2.bias")
        self.b1_l3_weight = self.torch_to_tt("layers.1.linear_3.weight")
        self.b1_l3_bias = self.torch_to_tt("layers.1.linear_3.bias")
        self.out_weight = self.torch_to_tt("linear_out.weight")
        self.out_bias = self.torch_to_tt("linear_out.bias")

    def _block(self, s, l2_w, l2_b, l3_w, l3_b):
        a = ttnn.relu(s, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = self._lin(a, l2_w, bias=l2_b, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        a = ttnn.relu(a, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = self._lin(a, l3_w, bias=l3_b, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.add(a, s, memory_config=ttnn.L1_MEMORY_CONFIG)

    def __call__(self, s: ttnn.Tensor, s_initial: ttnn.Tensor) -> ttnn.Tensor:
        si = ttnn.relu(s_initial, memory_config=ttnn.L1_MEMORY_CONFIG)
        si = self._lin(si, self.lin_initial_weight, bias=self.lin_initial_bias,
                       dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(s, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.lin_in_weight, bias=self.lin_in_bias,
                      dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, si, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._block(x, self.b0_l2_weight, self.b0_l2_bias,
                        self.b0_l3_weight, self.b0_l3_bias)
        x = self._block(x, self.b1_l2_weight, self.b1_l2_bias,
                        self.b1_l3_weight, self.b1_l3_bias)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.out_weight, bias=self.out_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return x  # [*, N, 14] unnormalized


def normalize_angles(unnorm: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Host-side fp32 tail for AngleResnet: reshape [*, N, 14] -> [*, N, 7, 2] and
    L2-normalize over the last (size-2) dim, matching the reference."""
    s = unnorm.view(unnorm.shape[:-1] + (-1, 2))
    norm_denom = torch.sqrt(
        torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=epsilon)
    )
    return s / norm_denom


class IPALayer(Module):
    """InvariantPointAttention (Alg. 22) on ttnn — the novel op of the port.

    8 of these form the structure-module block loop. Inputs (per block):
      s [1, N, embed_dim=128], z [1, N, N, embed_dim=128] (pair state, resident
      across blocks), r = rigid (rot_mats [1,N,3,3], trans [1,N,3]), mask [1,N].
    Reference output: single update delta [1, N, 128].

    ON-DEVICE STATUS (this is the port's ceiling):
      * On device (validated PCC 1.0 vs the reference internals,
        scripts/abb3_ipa_internals.py): the IPA *linear projections* — q, kv,
        qp (q-points), kvp (kv-points), and the pair-bias b. ``__call__`` returns
        these as a dict.
      * NOT on device (the ceiling): the IPA *attention* — both the scalar q.k
        and the point-attention. Both need subtile trailing-dim reshapes that ttnn
        stock ops cannot express on device:
          - scalar q.k needs a head reshape to [1,N,12,16] (head=12, head_dim=16,
            both subtile); ttnn.reshape re-tiles and scrambles the layout, and
            nlp_create_qkv_heads scrambles non-32 head_dim (the documented
            ttnn-tile-alignment hazard).
          - point-attention needs subtile point coords (3) / P_q,P_v (4,8)
            broadcast/sum/reshape and front-padding; ttnn.pad rejects
            front-padding of subtile dims on device, and subtile broadcast/sum
            over the 3/4/8 dims are unsupported.
        A full on-device IPA therefore needs a custom tt-metal point-attention
        kernel (kernel authoring is a separate domain, out of scope for this
        port). The attention math stays host-side fp32 in the structure-module
        loop; only the projections run on device here.
    """

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig,
                 c_hidden: int = 16, no_heads: int = 12,
                 no_qk_points: int = 4, no_v_points: int = 8,
                 eps: float = 1e-8, inf: float = 1e7):
        super().__init__(state_dict, ck)
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.eps = eps
        self.inf = inf
        self.lq_w = self.torch_to_tt("linear_q.weight"); self.lq_b = self.torch_to_tt("linear_q.bias")
        # linear_kv packs [k|v] per head (interleaved, not concatenated); load it as a single
        # projection and split on the host for the attention (the attention math is the
        # documented ceiling -- host-side fp32 -- so no benefit to splitting the weight).
        self.lkv_w = self.torch_to_tt("linear_kv.weight"); self.lkv_b = self.torch_to_tt("linear_kv.bias")
        self.lqp_w = self.torch_to_tt("linear_q_points.weight"); self.lqp_b = self.torch_to_tt("linear_q_points.bias")
        # linear_kv_points packs [k_pts|v_pts] per head (interleaved); load as a single projection.
        self.lkvp_w = self.torch_to_tt("linear_kv_points.weight"); self.lkvp_b = self.torch_to_tt("linear_kv_points.bias")
        self.lb_w = self.torch_to_tt("linear_b.weight"); self.lb_b = self.torch_to_tt("linear_b.bias")
        self.lo_w = self.torch_to_tt("linear_out.weight"); self.lo_b = self.torch_to_tt("linear_out.bias")
        hw = torch.nn.functional.softplus(self.weights["head_weights"]) * math.sqrt(
            1.0 / (3 * (no_qk_points * 9.0 / 2)))
        # per-(h,p_q) weight: hw[h] broadcast over p_q -> [H*P_q]
        self._hw = hw.view(no_heads, 1).expand(no_heads, no_qk_points).reshape(-1).contiguous()
        self._hw_tt = ttnn.from_torch(self._hw.view(1, 1, 1, -1), layout=ttnn.TILE_LAYOUT,
                                      device=self.device, dtype=_dtype())

    def _proj(self, w, b, s):
        return self._lin(s, w, bias=b, dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)

    def linear_out(self, cat):
        return self._lin(cat, self.lo_w, bias=self.lo_b, dtype=_dtype(),
                         memory_config=ttnn.L1_MEMORY_CONFIG)

    def __call__(self, s, z, rot_mats, trans, mask):
        N = s.shape[1]
        H, C = self.no_heads, self.c_hidden
        Pq, Pv = self.no_qk_points, self.no_v_points
        L1 = ttnn.L1_MEMORY_CONFIG
        ck = self.compute_kernel_config
        # --- scalar projections (the IPA linears; the attention is the documented ceiling) ---
        q = self._proj(self.lq_w, self.lq_b, s)     # [1,N,192]
        kv = self._proj(self.lkv_w, self.lkv_b, s)   # [1,N,384]  (packed [k|v] per head)
        qp = self._proj(self.lqp_w, self.lqp_b, s)   # [1,N,144]
        kvp = self._proj(self.lkvp_w, self.lkvp_b, s)  # [1,N,432]  (packed [k_pts|v_pts] per head)
        b = self._lin(z, self.lb_w, bias=self.lb_b, dtype=_dtype(), memory_config=L1)  # [1,N,N,12]
        # --- CEILING: the IPA attention (scalar q.k AND point) is NOT ported here.
        # Both attentions require subtile trailing-dim reshapes that ttnn stock ops
        # cannot express on device: the scalar q.k needs a head reshape to
        # [1,N,12,16] (head=12, head_dim=16 -- both subtile); ttnn.reshape re-tiles
        # and scrambles the layout, and nlp_create_qkv_heads scrambles non-32
        # head_dim (the documented ttnn-tile-alignment hazard). The point-attention
        # additionally needs subtile point coords (3) / P_q,P_v (4,8) broadcast/sum/reshape
        # and front-padding (ttnn.pad rejects front-padding of subtile dims on device).
        # A full on-device IPA therefore needs a custom tt-metal point-attention
        # kernel (kernel authoring is a separate domain, out of scope for this port).
        # What IS on device here: the IPA linear projections (q, k, v, q_pts,
        # kv_points, pair bias b) -- validated PCC 1.0 vs the reference internals
        # (scripts/abb3_ipa_internals.py). The attention math stays host-side
        # fp32 in the structure-module loop.
        return dict(q=q, kv=kv, qp=qp, kvp=kvp, b=b)



class StructureModuleTT(Module):
    """Hybrid on-device / host ABodyBuilder3 StructureModule (8 IPA blocks).

    ON DEVICE (bf16 weights, fp32 dest accumulation; PCC ~1.0 vs the reference
    golden, validated component-by-component in tests/test_abodybuilder3_*):
      * input embeddings (linear_in_node, linear_in_edge)
      * the IPA linear projections (q, kv, qp, kvp, pair bias b) + linear_out
      * the post-IPA LayerNorm, the Transition, BackboneUpdate, AngleResnet linears
      * the pLDDT head

    ON HOST fp32 (the documented ceiling -- the IPA attention needs a custom
    tt-metal point-attention kernel for a fully on-device port; the quaternion
    backbone compose and the torsion_angles_to_frames + atom14 reconstruction are
    rigid compositions with residue-constant lookup tables, kept on host like the
    ESMFold2 "cheap host code" boundary):
      * the IPA rigid-apply + scalar/point attention + value aggregation
      * the quaternion backbone compose (compose_q_update_vec)
      * torsion_angles_to_frames + frames_and_literature_positions_to_atom14_pos

    The host tail is the exact reference math (reused from the vendored
    StructureModule), so end-to-end parity is governed only by the bf16 device
    projections -- which are PCC 1.0, so Cα-RMSD vs the reference is ~0.
    """

    def __init__(self, state_dict, ck, config):
        super().__init__(state_dict, ck)
        self.cfg = dict(config)
        self.no_blocks = self.cfg["no_blocks"]
        self.embed_dim = self.cfg["embed_dim"]
        self.trans_scale_factor = self.cfg["trans_scale_factor"]
        self.epsilon = self.cfg["epsilon"]
        self.inf = self.cfg["inf"]
        self.rotation_propagation = self.cfg["rotation_propagation"]
        # input embeddings
        self.lin_in_node_w = self.torch_to_tt("linear_in_node.weight")
        self.lin_in_node_b = self.torch_to_tt("linear_in_node.bias")
        self.lin_in_edge_w = self.torch_to_tt("linear_in_edge.weight")
        self.lin_in_edge_b = self.torch_to_tt("linear_in_edge.bias")
        # per-block on-device modules
        self.ipa = [IPALayer(self.scope(f"ipa_layers.{i}"), ck) for i in range(self.no_blocks)]
        self.ln = [IPALayerNorm(self.scope(f"layer_norm_ipa_layers.{i}"), ck) for i in range(self.no_blocks)]
        self.trans = [StructureModuleTransition(self.scope(f"transition_layers.{i}"), ck) for i in range(self.no_blocks)]
        self.bb = [BackboneUpdate(self.scope(f"bb_update_layers.{i}"), ck) for i in range(self.no_blocks)]
        self.ang = [AngleResnet(self.scope(f"angle_resnet_layers.{i}"), ck, epsilon=self.epsilon) for i in range(self.no_blocks)]
        # pLDDT head
        self.plddt = PLDDTHead(self.scope("plddt"), ck)
        # residue constants (host) for atom14 reconstruction
        from tt_bio._vendor.abodybuilder3.openfold.np.residue_constants import (
            restype_atom14_mask, restype_atom14_rigid_group_positions,
            restype_atom14_to_rigid_group, restype_rigid_group_default_frame,
        )
        self.default_frames = torch.tensor(restype_rigid_group_default_frame, dtype=torch.float32)
        self.group_idx = torch.tensor(restype_atom14_to_rigid_group, dtype=torch.long)
        self.atom_mask = torch.tensor(restype_atom14_mask, dtype=torch.float32)
        self.lit_positions = torch.tensor(restype_atom14_rigid_group_positions, dtype=torch.float32)

    def _push(self, x):
        return _from_torch(x.contiguous().float())

    def _pull(self, x):
        return _to_torch(x)

    def _embed(self, single, pair):
        s = self._lin(self._push(single), self.lin_in_node_w, bias=self.lin_in_node_b,
                      dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        z0 = self._lin(self._push(pair), self.lin_in_edge_w, bias=self.lin_in_edge_b,
                       dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        return self._pull(s), self._pull(z0)

    def _ipa_hybrid(self, i, s_host, z_block, rigids, mask):
        """On-device projections -> host attention -> on-device linear_out -> delta."""
        from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid
        ipa = self.ipa[i]
        H, C = ipa.no_heads, ipa.c_hidden
        Pq, Pv = ipa.no_qk_points, ipa.no_v_points
        s_tt = self._push(s_host)
        z_tt = self._push(z_block)
        proj = ipa(s_tt, z_tt, None, None, None)  # projections only
        q = self._pull(proj["q"]).reshape(1, -1, H, C)
        kv = self._pull(proj["kv"]).reshape(1, -1, H, 2 * C)
        k, v = torch.split(kv, C, dim=-1)
        qp = self._pull(proj["qp"]).view(1, -1, 3, H * Pq).permute(0, 1, 3, 2)      # [1,N,48,3]
        kvp = self._pull(proj["kvp"]).view(1, -1, 3, H * (Pq + Pv)).permute(0, 1, 3, 2)  # [1,N,144,3]
        b = self._pull(proj["b"])  # [1,N,N,H]
        del s_tt, z_tt, proj
        # rigid-apply the points (host)
        r = rigids
        q_pts = r[..., None].apply(qp).view(1, -1, H, Pq, 3)
        kv_pts = r[..., None].apply(kvp).view(1, -1, H, Pq + Pv, 3)
        k_pts, v_pts = torch.split(kv_pts, [Pq, Pv], dim=-2)
        # scalar attention a = q.k * scale + sqrt(1/3) * b
        a = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))  # [1,H,N,N]
        a = a * math.sqrt(1.0 / (3 * C))
        a = a + math.sqrt(1.0 / 3) * b.permute(0, 3, 1, 2)
        # point-attention logits (host): ||q-k||^2 weighted
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)  # [1,N,N,H,Pq,3]
        pt_att = pt_att ** 2
        pt_att = sum(torch.unbind(pt_att, dim=-1))  # [1,N,N,H,Pq]
        hw = torch.nn.functional.softplus(ipa.weights["head_weights"]).view(1, 1, 1, -1, 1)
        hw = hw * math.sqrt(1.0 / (3 * (Pq * 9.0 / 2)))
        pt_att = pt_att * hw
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)  # [1,N,N,H]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        pt_att = pt_att.permute(0, 3, 1, 2)  # [1,H,N,N]
        a = a + pt_att + square_mask.unsqueeze(-3)
        attn = torch.softmax(a, dim=-1)  # [1,H,N,N]
        # value aggregation (verbatim reference math; v [1,N,H,C], v_pts [1,N,H,Pv,3])
        from tt_bio._vendor.abodybuilder3.openfold.utils.tensor_utils import (
            flatten_final_dims, permute_final_dims,
        )
        o = torch.matmul(attn, v.transpose(-2, -3).to(attn.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)  # [1,N,H*C]
        o_pt = torch.sum(
            (attn[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + ipa.eps), 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pair = torch.matmul(attn.transpose(-2, -3), z_block.to(dtype=attn.dtype))
        o_pair = flatten_final_dims(o_pair, 2)
        cat = torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1)  # [1,N,2112]
        # linear_out on device
        delta = self._pull(ipa.linear_out(self._push(cat)))
        return delta  # [1,N,128]

    def __call__(self, single, pair, aatype, mask=None):
        from tt_bio._vendor.abodybuilder3.openfold.utils.feats import (
            frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames,
        )
        from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid
        N = single.shape[1]
        if mask is None:
            mask = torch.ones(single.shape[:-1], dtype=single.dtype)
        s, z_initial = self._embed(single, pair)
        s_initial = s.clone()
        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, False, fmt="quat")
        outputs = []
        for i in range(self.no_blocks):
            trans = rigids.get_trans()
            pd = torch.norm(trans[:, :, None] - trans[:, None, :], dim=-1)
            pmask = mask[:, :, None] * mask[:, None, :]
            pd = (pd * pmask).unsqueeze(-1).to(z_initial.dtype)
            z = torch.cat((z_initial, pd), dim=-1)
            delta = self._ipa_hybrid(i, s, z, rigids, mask)
            s = s + delta
            s = self._pull(self.ln[i](self._push(s)))
            s = self._pull(self.trans[i](self._push(s)))
            bb = self._pull(self.bb[i](self._push(s)))
            rigids = rigids.compose_q_update_vec(bb)
            backb_to_global = Rigid(
                type(rigids.get_rots())(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            ).scale_translation(self.trans_scale_factor)
            unnorm = self._pull(self.ang[i](self._push(s), self._push(s_initial)))
            angles = normalize_angles(unnorm, epsilon=self.epsilon)
            all_frames = torsion_angles_to_frames(backb_to_global, angles, aatype, self.default_frames)
            pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames, aatype, self.default_frames, self.group_idx, self.atom_mask, self.lit_positions,
            )
            scaled = rigids.scale_translation(self.trans_scale_factor)
            outputs.append({
                "frames": scaled.to_tensor_7(), "angles": angles, "positions": pred_xyz, "states": s,
            })
            if not self.rotation_propagation:
                rigids = rigids.stop_rot_gradient()
        stacked = {k: torch.stack([o[k] for o in outputs], dim=0) for k in outputs[0]}
        plddt = self._pull(self.plddt(self._push(s)))
        stacked["single"] = s
        stacked["plddt"] = plddt
        return stacked


def predict_abodybuilder3(heavy: str, light: str, cache_dir: str,
                          out_pdb: str | None = None) -> dict:
    """Run the hybrid ABodyBuilder3 port on a paired heavy+light Fv and return
    atom37 coords + pLDDT (+ write a PDB if out_pdb is set).

    This is the tt-bio predict entry point for ``--model abodybuilder3``. The
    on-device pieces (embeddings, IPA projections + linear_out, LayerNorm,
    Transition, BackboneUpdate, AngleResnet linears, pLDDT head) run in bf16 with
    fp32 dest accumulation; the IPA attention + quaternion compose + atom14
    reconstruction run host-side fp32 (the documented ceiling). End-to-end parity
    vs the PyTorch reference: Cα-RMSD ~0.016 Å, pLDDT PCC ~0.99998 (6yio H0-L0)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
    from tt_bio._vendor.abodybuilder3.openfold.data.data_transforms import make_atom14_masks
    from tt_bio._vendor.abodybuilder3.openfold.utils.feats import atom14_to_atom37
    from tt_bio._vendor.abodybuilder3.openfold.np.protein import Protein, to_pdb
    from abodybuilder3_reference import string_to_input, compute_plddt

    inp = string_to_input(heavy, light, "cpu")
    single, pair, aatype = inp["single"], inp["pair"], inp["aatype"]
    mask = torch.ones(single.shape[:-1], dtype=single.dtype)
    sd = torch.load(ensure_abb3_weights(cache_dir), map_location="cpu", weights_only=True)
    ck = abb3_compute_kernel_config()
    model = StructureModuleTT(sd, ck, ABB3_CONFIG)
    with torch.no_grad():
        out = model(single, pair, aatype, mask)
    atom14 = out["positions"][-1, 0]
    batch = make_atom14_masks({"aatype": aatype.squeeze(0)})
    atom37 = atom14_to_atom37(atom14, batch)
    plddt = compute_plddt(out["plddt"][0])
    result = {"atom37": atom37.cpu(), "plddt": plddt.cpu(),
              "aatype": aatype.squeeze(0).cpu(), "is_heavy": inp["is_heavy"].cpu()}
    if out_pdb:
        import numpy as np
        aatype_np = aatype.squeeze(0).cpu().numpy().astype(int)
        chain_index = 1 - inp["is_heavy"].cpu().numpy().astype(int)
        atom_mask = batch["atom37_atom_exists"].cpu().numpy().astype(int)
        b_factors = (plddt.cpu().numpy()[:, None] * atom_mask).astype(atom_mask.dtype)
        protein = Protein(aatype=aatype_np, atom_positions=atom37.cpu().numpy(),
                          atom_mask=atom_mask, residue_index=np.arange(len(aatype_np)),
                          b_factors=b_factors, chain_index=chain_index)
        Path(out_pdb).write_text(to_pdb(protein))
    return result
