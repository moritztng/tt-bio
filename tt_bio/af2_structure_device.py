"""AlphaFold 2's structure module on the card, built from ABodyBuilder3's device modules.

`af2_reference.AF2StructureModule` runs in host torch between recycles, and the WH device profile
of AF2-IG shows the card idle ~380 ms at every recycle boundary while it does (state/spd BOARD,
2026-10-12). ABodyBuilder3's structure module is the same algorithm (openfold's IPA, transition,
quaternion backbone update and angle resnet) with different weight names, so this file is a weight
remap onto `abodybuilder3.DeviceIPA` and friends rather than a second IPA port.

What differs from ABodyBuilder3, and how it is bridged:

* one weight set shared by all 8 layers: the device modules are built once and called 8 times;
* the pair is the trunk's, unchanged across layers, so there is no per-layer distance channel;
* c_s is 384 and the transition is AF2's own 3-linear residual (openfold's `layers.0` naming);
* masked logits use AF2's 1e5, and the point norm AF2's 1e-8;
* translations are scaled by 10 only where AF2 scales them (sidechains and outputs).

The geometry tail (torsion angles to frames to atom14 to atom37) stays on the host, as it does in
ABodyBuilder3, and runs for the LAST layer only: AF2-IG reads the final atoms and the final single
representation, and the 7 earlier layers' sidechains are outputs nothing consumes at inference.
"""

from __future__ import annotations

import torch
import ttnn

from . import abodybuilder3_ops as ops
from .abodybuilder3 import (DeviceAngleResnet, DeviceBackboneUpdate, DeviceIPA,
                            DeviceTransition, Rigid, to_device_fp32)
from .abodybuilder3_reference import ABB3Config
from .af2_reference import (AF2StructureModule, QuatAffine, atom14_to_atom37,
                            frames_to_atom14_positions, torsion_angles_to_frames)


def af2_config(sm: AF2StructureModule) -> ABB3Config:
    ipa = sm.ipa
    return ABB3Config(c_s=int(sm.initial_projection.weight.shape[0]),
                      embed_dim=int(sm.pair_norm.weight.shape[0]),
                      c_ipa=ipa.num_scalar_qk, no_heads_ipa=ipa.num_head,
                      no_qk_points=ipa.num_point_qk, no_v_points=ipa.num_point_v,
                      no_blocks=sm.num_layer,
                      c_resnet=int(sm.sidechain.angles.weight.shape[1]),
                      no_resnet_blocks=len(sm.sidechain.resblock),
                      epsilon=1e-8, inf=1e5)


def _openfold_names(sm: AF2StructureModule) -> dict[str, dict[str, torch.Tensor]]:
    """AF2's structure-module parameters under the names the ABodyBuilder3 device modules read."""
    ipa, sc = sm.ipa, sm.sidechain
    lin = lambda m, name: {f"{name}.weight": m.weight.detach(), f"{name}.bias": m.bias.detach()}
    w_ipa = {**lin(ipa.q_scalar, "linear_q"), **lin(ipa.kv_scalar, "linear_kv"),
             **lin(ipa.q_point_local, "linear_q_points"),
             **lin(ipa.kv_point_local, "linear_kv_points"),
             **lin(ipa.attention_2d, "linear_b"), **lin(ipa.output_projection, "linear_out"),
             "head_weights": ipa.point_weights.detach()}
    w_tr = {**{k: v for i, layer in enumerate(sm.transition)
               for k, v in lin(layer, f"layers.0.linear_{i + 1}").items()},
            "layer_norm.weight": sm.transition_norm.weight.detach(),
            "layer_norm.bias": sm.transition_norm.bias.detach()}
    w_bb = lin(sm.affine_update, "linear")
    w_ang = {**lin(sc.input_projection[0], "linear_in"),
             **lin(sc.input_projection[1], "linear_initial"),
             **lin(sc.angles, "linear_out")}
    for i, (first, second) in enumerate(sc.resblock):
        w_ang.update({**lin(first, f"layers.{i}.linear_2"), **lin(second, f"layers.{i}.linear_3")})
    return {"ipa": w_ipa, "transition": w_tr, "bb": w_bb, "angles": w_ang}


class AF2DeviceStructure:
    """`AF2StructureModule.forward` with its 8 layers on the card, in float32."""

    def __init__(self, sm: AF2StructureModule, *, upload=to_device_fp32):
        self.cfg = cfg = af2_config(sm)
        self.position_scale = sm.position_scale
        self.upload = upload
        w = _openfold_names(sm)
        self.ipa = DeviceIPA(w["ipa"], cfg, to_device=upload)
        self.transition = DeviceTransition(w["transition"], cfg, to_device=upload)
        self.bb_update = DeviceBackboneUpdate(w["bb"], to_device=upload)
        self.angles = DeviceAngleResnet(w["angles"], cfg, to_device=upload)
        ln = lambda m: (upload(m.weight.detach()), upload(m.bias.detach()))
        self.single_norm, self.pair_norm = ln(sm.single_norm), ln(sm.pair_norm)
        self.attention_norm = ln(sm.attention_norm)
        self.w_init = upload(sm.initial_projection.weight.detach().t())
        self.b_init = upload(sm.initial_projection.bias.detach())

    def mask_bias(self, seq_mask: torch.Tensor, n_pad: int) -> torch.Tensor:
        """AF2's `-1e5 * (1 - m_i m_j)` as a `[1, 1, n_pad, n_pad]` float32 host tensor.

        Residues past the sequence are masked like any other, so padded keys get no weight and
        padded queries compute rows nobody reads.
        """
        m = torch.zeros(n_pad)
        m[:len(seq_mask)] = seq_mask.float()
        return (-self.cfg.inf * (1.0 - m[:, None] * m[None, :]))[None, None]

    def __call__(self, single, pair, feats: dict) -> dict:
        """`single` `[1, n_pad, c_s]` and `pair` `[1, n_pad, n_pad, c_z]` device tensors, float32.

        `n_pad` may exceed the sequence length (the trunk pads to a tile); `feats` is unpadded.
        """
        n_pad = int(single.shape[1])
        bias = self.upload(self.mask_bias(feats["seq_mask"], n_pad))
        one, zero = self.upload(torch.ones(1, n_pad)), self.upload(torch.zeros(1, n_pad))
        return self.finish(self.layers(single, pair, bias, one, zero), feats)

    def layers(self, single, pair, bias, one, zero) -> list:
        """The 8 layers and the angle resnet, device tensors in and out, nothing uploaded or
        downloaded, so a trace can capture it. `one` / `zero` `[1, n_pad]` seed the identity
        frame. Returns `[s, *quat, *trans, sin, cos, unnorm_sin, unnorm_cos]`."""
        s = ops.layer_norm(single, *self.single_norm)
        s_initial = s
        s = ops.linear(s, self.w_init, self.b_init)
        z = ops.layer_norm(pair, *self.pair_norm)
        rigid = Rigid(quat=[one, zero, zero, zero], trans=[zero, zero, zero])
        for _ in range(self.cfg.no_blocks):
            s = ops.add(s, self.ipa(s, z, rigid.frame(), bias))
            s = ops.layer_norm(s, *self.attention_norm)
            s = self.transition(s)
            rigid = rigid.compose_update(self.bb_update(s))
        (unnorm_sin, unnorm_cos), (sin, cos) = self.angles(s, s_initial)
        return [s, *rigid.quat, *rigid.trans, sin, cos, unnorm_sin, unnorm_cos]

    def finish(self, outs: list, feats: dict) -> dict:
        s, quat, trans, (sin, cos, unnorm_sin, unnorm_cos) = outs[0], outs[1:5], outs[5:8], outs[8:12]
        return self.geometry(Rigid(quat=list(quat), trans=list(trans)), s, (sin, cos),
                             (unnorm_sin, unnorm_cos), feats)

    def geometry(self, rigid: Rigid, s, angles, unnorm, feats: dict) -> dict:
        """Download the last layer and run AF2's geometry tail on the host in float32."""
        n, n_pad = len(feats["seq_mask"]), int(s.shape[1])
        rows = lambda t: ttnn.to_torch(t).float().reshape(n_pad, -1)[:n]
        quat = torch.cat([rows(q) for q in rigid.quat], -1)
        trans = torch.cat([rows(t) for t in rigid.trans], -1)
        affine = QuatAffine(quat, trans, normalize=False)
        # sin in channels 0-6 of one tile block, cos in the next (`DeviceAngleResnet`).
        pair_of = lambda sc: torch.stack([rows(x)[:, :self.cfg.no_angles] for x in sc], -1)
        sin_cos, unnorm_sin_cos = pair_of(angles), pair_of(unnorm)
        aatype = feats["aatype"].long()
        scaled = affine.scale_translation(self.position_scale)
        rot, tr = torsion_angles_to_frames(aatype, scaled.rotation, scaled.translation, sin_cos)
        atom14 = frames_to_atom14_positions(aatype, rot, tr)
        act = rows(s)
        scale = torch.tensor([1.0] * 4 + [self.position_scale] * 3)
        final = affine.to_tensor() * scale
        return {
            "representations/structure_module": act,
            "final_affines": final,
            "sidechains/angles_sin_cos": sin_cos[None],
            "sidechains/unnormalized_angles_sin_cos": unnorm_sin_cos[None],
            "final_atom14_positions": atom14,
            "final_atom14_mask": feats["atom14_atom_exists"].float(),
            "final_atom_positions": atom14_to_atom37(atom14, feats),
            "final_atom_mask": feats["atom37_atom_exists"].float(),
        }
