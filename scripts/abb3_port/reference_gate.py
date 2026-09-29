#!/usr/bin/env python3
"""Score `tt_bio/abodybuilder3_reference.py` against the two references it claims to agree with.

Three independent checks, all in float64, because the port's whole parity table is quoted
against this file and a reference that is itself wrong makes every number below it meaningless.

0. **Upstream's own precision floor, which the bar has to account for.** OpenFold's
   `Rotation.__init__` and `Rigid.__init__` cast to float32 unconditionally
   (`rigid_utils.py:315-318` and `:856`, both commented "force full precision"), because the
   model around them normally runs bf16. So upstream's frame algebra is float32 whatever dtype
   the weights are, and a float64-vs-float64 comparison against it stalls at 1e-6 on atom
   positions -- upstream's error, not ours. The model check therefore runs twice: once in
   float32, which is what upstream actually executes, and once in float64 with that floor lifted
   by a shim, which is what proves the two implementations are the same algorithm.

1. **Tables.** `tt_bio/af2_data.py`'s four rigid-group tables against upstream's own
   `residue_constants`. A mismatch here silently moves every atom position.
2. **The model.** Our module against upstream's vendored `StructureModule` at
   `use_original_sm=True`, same random weights, same inputs, every output key.
3. **The pieces.** Our IPA against `af2_reference.InvariantPointAttention` at matching dims, and
   our batched torsion/atom14 functions against `af2_reference`'s unbatched originals.

Usage: reference_gate.py --upstream ~/abb3_src/ABodyBuilder3/src [--tokens 48] [--batch 2]
Check 2 is skipped with a loud line if --upstream is not given.
"""
from __future__ import annotations

import argparse
import sys

import torch

from tt_bio import af2_reference as afr
from tt_bio.abodybuilder3_reference import (ABB3Config, ABB3StructureModule,
                                            InvariantPointAttention, frames_to_atom14_positions,
                                            single_and_pair_features, torsion_angles_to_frames)

DT = torch.float64


def _err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(DT) - b.to(DT)).abs().max().item()


def _report(rows: list[tuple[str, float, float]]) -> int:
    bad = 0
    for name, err, bar in rows:
        ok = err <= bar
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<44} {err:.3e}  (bar {bar:.0e})")
    return bad


def check_tables(upstream: str | None) -> int:
    if upstream is None:
        print("tables: SKIPPED, no --upstream")
        return 0
    from abodybuilder3.openfold.np import residue_constants as rc
    from tt_bio import af2_data as d
    pairs = [
        ("restype_rigid_group_default_frame", d.RESTYPE_RIGID_GROUP_DEFAULT_FRAME,
         rc.restype_rigid_group_default_frame),
        ("restype_atom14_to_rigid_group", d.RESTYPE_ATOM14_TO_RIGID_GROUP,
         rc.restype_atom14_to_rigid_group),
        ("restype_atom14_mask", d.RESTYPE_ATOM14_MASK, rc.restype_atom14_mask),
        ("restype_atom14_rigid_group_positions", d.RESTYPE_ATOM14_RIGID_GROUP_POSITIONS,
         rc.restype_atom14_rigid_group_positions),
    ]
    print("tables: tt_bio/af2_data.py vs upstream residue_constants")
    rows = []
    for name, ours, theirs in pairs:
        ours_t = torch.as_tensor(ours).to(DT)
        theirs_t = torch.as_tensor(theirs).to(DT)
        rows.append((name, _err(ours_t, theirs_t) if ours_t.shape == theirs_t.shape else float("inf"),
                     0.0))
    return _report(rows)


class _Fp32FloorLifted:
    """Run upstream's frame algebra in float64 by making its `torch.float32` mean float64.

    `rigid_utils` reaches its two hard casts through the module-global name `torch`, so binding
    that name to a proxy whose `float32` is `float64` lifts the floor without touching upstream's
    source or affecting anything else in the process. The alternative -- comparing at float32 --
    cannot separate "same algorithm" from "same algorithm plus a 1e-6 dtype policy".
    """

    def __init__(self, module):
        self.module = module
        self.saved = module.torch

    class _Shim:
        def __init__(self, real):
            self._real = real
            self.float32 = real.float64

        def __getattr__(self, name):
            return getattr(self._real, name)

    def __enter__(self):
        self.module.torch = self._Shim(self.saved)
        return self

    def __exit__(self, *exc):
        self.module.torch = self.saved
        return False


def check_model(upstream: str | None, n_tok: int, batch: int, seed: int) -> int:
    if upstream is None:
        print("model: SKIPPED, no --upstream")
        return 0
    from abodybuilder3.openfold.model.structure_module import StructureModule
    from abodybuilder3.openfold.utils import rigid_utils

    rows = []
    # The float32 leg is scored relative to the magnitude of the tensor it compares, because
    # `unnormalized_angles` is the un-normalised resnet output and is two orders of magnitude
    # bigger than anything else in the dict; a shared absolute bar would either pass everything
    # or fail that one row on float32 noise.
    # 1e-4 relative on the float32 leg, and it is a sanity check rather than the precision claim.
    # Both sides run the same arithmetic in a different op order, and the 8 blocks feed their own
    # frames back into the next block's pair features, so float32 roundoff at 1e-7 amplifies:
    # measured 2.0e-5 on the frames and 5.3e-5 on the unit-norm angles with N(0, 0.3) weights,
    # which are far wider than a trained checkpoint's. The float64 leg below is the proof.
    for dtype, lift, bar in ((torch.float32, False, 1e-4), (DT, True, 1e-10)):
        cfg = ABB3Config(use_plddt=True)
        torch.manual_seed(seed)
        ours = ABB3StructureModule(cfg).to(dtype)
        # A trained checkpoint has no zeros; the released init leaves `linear_out` and the
        # backbone update at zero, which would hide a sign error in either. Random everything.
        with torch.no_grad():
            for p in ours.parameters():
                p.normal_(0.0, 0.3)
        theirs = StructureModule(
            c_s=cfg.c_s, embed_dim=cfg.embed_dim, c_z=cfg.c_z, c_ipa=cfg.c_ipa,
            c_resnet=cfg.c_resnet, no_heads_ipa=cfg.no_heads_ipa, no_qk_points=cfg.no_qk_points,
            no_v_points=cfg.no_v_points, dropout_rate=cfg.dropout_rate, no_blocks=cfg.no_blocks,
            no_transition_layers=cfg.no_transition_layers,
            no_resnet_blocks=cfg.no_resnet_blocks, no_angles=cfg.no_angles,
            trans_scale_factor=cfg.trans_scale_factor, epsilon=cfg.epsilon, inf=cfg.inf,
            rotation_propagation=True, use_original_sm=True, use_plddt=True,
        ).to(dtype).eval()
        theirs.load_state_dict(ours.state_dict())

        g = torch.Generator().manual_seed(seed + 1)
        aatype = torch.randint(0, 21, (batch, n_tok), generator=g)
        is_heavy = (torch.arange(n_tok) < n_tok // 2).expand(batch, n_tok).clone()
        residue_index = torch.arange(n_tok).expand(batch, n_tok).clone()
        single, pair = single_and_pair_features(aatype, is_heavy, residue_index, dtype=dtype)
        mask = torch.ones(batch, n_tok, dtype=dtype)
        mask[-1, -3:] = 0.0  # a padded sample, so the mask path is exercised not assumed

        with torch.no_grad():
            out = ours(single, pair, aatype, mask)
            if lift:
                with _Fp32FloorLifted(rigid_utils):
                    ref = theirs({"single": single, "pair": pair}, aatype, mask)
            else:
                ref = theirs({"single": single, "pair": pair}, aatype, mask)
        tag = f"{'float64, floor lifted' if lift else 'float32, as shipped'}"
        print(f"model: ours vs upstream StructureModule, {tag}, batch {batch} x {n_tok} tokens")
        keys = ["frames", "sidechain_frames", "unnormalized_angles", "angles", "positions",
                "states", "single", "plddt"]
        for k in keys:
            scale = max(1.0, ref[k].abs().max().item()) if not lift else 1.0
            rows.append((f"{k} [{'f64' if lift else 'f32'}]", _err(out[k], ref[k]),
                         bar * scale))
    return _report(rows)


def check_pieces(seed: int, n_tok: int) -> int:
    cfg = ABB3Config(epsilon=1e-8, inf=1e5)  # af2_reference's two hardcoded constants
    torch.manual_seed(seed)
    ours = InvariantPointAttention(cfg).to(DT)
    with torch.no_grad():
        for p in ours.parameters():
            p.normal_(0.0, 0.3)
    theirs = afr.InvariantPointAttention(
        c_s=cfg.embed_dim, c_z=cfg.embed_dim, num_head=cfg.no_heads_ipa, num_scalar_qk=cfg.c_ipa,
        num_scalar_v=cfg.c_ipa, num_point_qk=cfg.no_qk_points, num_point_v=cfg.no_v_points).to(DT)
    with torch.no_grad():
        for dst, src in (("q_scalar", "linear_q"), ("kv_scalar", "linear_kv"),
                         ("q_point_local", "linear_q_points"),
                         ("kv_point_local", "linear_kv_points"), ("attention_2d", "linear_b"),
                         ("output_projection", "linear_out")):
            getattr(theirs, dst).weight.copy_(getattr(ours, src).weight)
            getattr(theirs, dst).bias.copy_(getattr(ours, src).bias)
        theirs.point_weights.copy_(ours.head_weights)

    g = torch.Generator().manual_seed(seed + 2)
    s = torch.randn(n_tok, cfg.embed_dim, generator=g, dtype=DT)
    z = torch.randn(n_tok, n_tok, cfg.embed_dim, generator=g, dtype=DT)
    quat = torch.randn(n_tok, 4, generator=g, dtype=DT)
    trans = torch.randn(n_tok, 3, generator=g, dtype=DT) * 8.0
    affine = afr.QuatAffine(quat, trans)
    mask = torch.ones(n_tok, dtype=DT)
    with torch.no_grad():
        mine = ours(s.unsqueeze(0), z.unsqueeze(0), affine, mask.unsqueeze(0)).squeeze(0)
        theirs_out = theirs(s, z, mask.unsqueeze(-1), affine)
    rows = [("IPA vs af2_reference.InvariantPointAttention", _err(mine, theirs_out), 1e-10)]

    aatype = torch.randint(0, 21, (n_tok,), generator=g)
    angles = torch.randn(n_tok, 7, 2, generator=g, dtype=DT)
    angles = angles / angles.square().sum(-1, keepdim=True).sqrt()
    rot, tr = torsion_angles_to_frames(aatype.unsqueeze(0), affine.rotation.unsqueeze(0),
                                       trans.unsqueeze(0), angles.unsqueeze(0))
    rot_r, tr_r = afr.torsion_angles_to_frames(aatype, affine.rotation, trans, angles)
    rows.append(("torsion_angles_to_frames rot vs unbatched", _err(rot[0], rot_r), 0.0))
    rows.append(("torsion_angles_to_frames trans vs unbatched", _err(tr[0], tr_r), 0.0))
    pos = frames_to_atom14_positions(aatype.unsqueeze(0), rot, tr)
    pos_r = afr.frames_to_atom14_positions(aatype, rot_r, tr_r)
    rows.append(("frames_to_atom14_positions vs unbatched", _err(pos[0], pos_r), 0.0))
    print("pieces: ours vs af2_reference, float64")
    return _report(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=None, help="path to ABodyBuilder3/src")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.upstream:
        sys.path.insert(0, args.upstream)
    torch.set_grad_enabled(False)
    bad = check_tables(args.upstream)
    bad += check_model(args.upstream, args.tokens, args.batch, args.seed)
    bad += check_pieces(args.seed, args.tokens)
    print(f"\n{'PASS' if bad == 0 else f'FAIL: {bad} rows'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
