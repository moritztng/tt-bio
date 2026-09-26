"""AlphaFold 2's multimer structure module on ttnn: IPA, the quaternion frame, the sidechain.

This is `folding_multimer.StructureModule`, the module the public BindCraft 2 configuration
runs, not the monomer `folding.StructureModule` that `af2_reference.AF2StructureModule`
transcribes. The two are different programs: q, k and v are projected separately here, the
summed logits are scaled by sqrt(1/3) AFTER the mask rather than the pair term alone being
scaled before it, and the frame is a rotation matrix built from a quaternion rather than a
quaternion carried through. BindCraft 2 also comments out the stop-gradient on the rotation
(`folding_multimer.py:467`), so the rotation cotangent runs through all eight layers.

Written in plain ttnn verbs so `tt_bio.taped_ttnn` supplies the backward. There is no second
implementation of anything here and no hand-written VJP: the module that runs under a tape is
the module that runs without one.

**Layouts, because the whole file is one convention.** Everything per-head is `[1, h, n, d]`
with the head axis as the matmul batch, which is why no tensor is ever split into heads: the
projections are batched matmuls against a weight the loader has already reshaped to
`[1, h, c_s, d]`, so the head axis is born where it is used. Everything per-residue is
`[1, 1, n, 1]`, and a rotation is nine of those rather than a `[n, 3, 3]`: a 3x3 on this
hardware is one tile holding nine numbers, so the matmul form buys nothing and costs a reshape
across a padded axis. The pair track is `[1, n, n, c_z]` and the single track `[1, 1, n, c_s]`.

**Why the distance term is a matmul.** The logit needs `sum_p |q_p - k_p|^2` for every residue
pair, which written out is a `[h, n, n, num_point_qk]` intermediate -- 12 million elements at
n=288, four times over, and every one of them read back. Expanded it is
`|Q|^2 + |K|^2 - 2 Q.K` over the 12 flattened components of a head's points, so it is one
`[1, h, n, 12] @ [1, h, 12, n]` matmul and two rank-one adds. The cancellation this trades for
is bounded: the terms are at most a few hundred Angstrom squared and float32 resolves them to
1e-5, four orders below what a logit carries into a softmax. It is NOT safe in bfloat16, which
is why `point_dtype` exists and defaults to float32 while the projections do not.
"""
from __future__ import annotations

import math

import numpy as np
import ttnn

from . import ops

C_S = 384
C_Z = 128
NUM_HEAD = 12
NUM_SCALAR_QK = 16
NUM_SCALAR_V = 16
NUM_POINT_QK = 4
NUM_POINT_V = 8
NUM_LAYER = 8
POSITION_SCALE = 20.0
SIDECHAIN_CHANNEL = 128
NUM_RESIDUAL_BLOCK = 2

#: haiku's LayerNorm default, and what `common_modules.LayerNorm` leaves it at.
LAYER_NORM_EPS = 1e-5

#: `InvariantPointAttention._dist_epsilon`, clipping the point norm from below. The clip is on
#: the SQUARE, so the floor is the square of this.
DIST_EPSILON = 1e-8

#: `l2_normalize`'s floor on the sum of squares of a (sin, cos) pair.
ANGLE_EPSILON = 1e-12

def compute_kernel_config(device) -> "ttnn.DeviceComputeKernelConfig":
    """HiFi4 with an fp32 accumulator, which is not a tuning choice here but the difference
    between a port and noise.

    ttnn's default fidelity truncates both matmul operands to bfloat16 whatever the tensors
    say, and this module amplifies: a float32 arm on the default config graded the IPA's
    scalar logits at rel 5.5e-3 against float64 -- bfloat16's error, from float32 inputs --
    and eight layers turned that into cos 0.49 on the frame. The same arm at HiFi4 grades the
    logits at 1.7e-5. Every matmul and every norm in this file carries it.
    """
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


_PREFIX = "alphafold/alphafold_iteration/structure_module/"
_FOLD = _PREFIX + "fold_iteration/"
_IPA = _FOLD + "invariant_point_attention/"
_SIDECHAIN = _FOLD + "rigid_sidechain/"


def _softplus(x: np.ndarray) -> np.ndarray:
    """`jnp.logaddexp(x, 0)`, which is what the module calls its softplus."""
    return np.logaddexp(x, np.zeros_like(x))


class StructureWeights:
    """The 46 checkpoint arrays, reshaped on host into the layouts the forward multiplies.

    Every reshape is here and none is in the forward. A point projection's `[c_s, h, 3*p]`
    weight becomes three `[1, h, c_s, p]` weights, one per coordinate, because
    `folding_multimer.PointProjection` splits its output into x, y and z blocks and a split of
    a 12-wide axis is not tile-aligned on device: cutting the WEIGHT costs nothing and cutting
    the activation costs a row-major round trip. The output projection is cut the same way, into
    the six feature groups its 2112 input channels are the concatenation of.
    """

    def __init__(self, params, device, *, dtype=ttnn.bfloat16, point_dtype=ttnn.float32):
        self.device = device
        self.dtype = dtype
        self.point_dtype = point_dtype
        raw = {k: np.asarray(params[k], dtype=np.float32) for k in params.files} \
            if hasattr(params, "files") else \
            {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self._raw = raw
        self._built: dict = {}
        self.build()

    # ------------------------------------------------------------------ host -> card

    def _up(self, array: np.ndarray, dtype=None) -> ttnn.Tensor:
        return ttnn.from_torch(
            __import__("torch").from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
            layout=ttnn.TILE_LAYOUT, device=self.device,
            dtype=self.dtype if dtype is None else dtype)

    def _get(self, key: str) -> np.ndarray:
        return self._raw[key]

    def _linear(self, scope: str, bias: bool = True):
        w = self._up(self._get(scope + "//weights"))
        b = self._up(self._get(scope + "//bias").reshape(1, 1, 1, -1)) if bias else None
        return w, b

    def _heads(self, scope: str, dtype=None) -> ttnn.Tensor:
        """`[c_s, h, d] -> [1, h, c_s, d]`: the head axis becomes the matmul batch."""
        w = self._get(scope + "//weights")
        return self._up(np.transpose(w, (1, 0, 2))[None], dtype=dtype)

    def _points(self, scope: str, num_point: int):
        """A point projection as three `[1, h, c_s, p]` weights and three `[1, h, 1, p]` biases.

        `PointProjection` writes `[h, 3*p]` and splits the LAST axis into three, so block 0 is x,
        block 1 is y and block 2 is z -- not an interleaving, which is the one way to get this
        wrong and still see a plausible number.
        """
        w = np.transpose(self._get(scope + "//weights"), (1, 0, 2))      # [h, c_s, 3p]
        b = self._get(scope + "//bias")                                  # [h, 3p]
        ws, bs = [], []
        for i in range(3):
            ws.append(self._up(w[None, :, :, i * num_point:(i + 1) * num_point],
                               dtype=self.point_dtype))
            bs.append(self._up(b[None, :, None, i * num_point:(i + 1) * num_point],
                               dtype=self.point_dtype))
        return ws, bs

    # ------------------------------------------------------------------ the set

    def build(self) -> None:
        b = self._built
        for name, scope, width in (("single_norm", _PREFIX + "single_layer_norm", C_S),
                                   ("pair_norm", _PREFIX + "pair_layer_norm", C_Z),
                                   ("attention_norm", _FOLD + "attention_layer_norm", C_S),
                                   ("transition_norm", _FOLD + "transition_layer_norm", C_S)):
            b[name] = (self._up(self._get(scope + "//scale").reshape(1, 1, 1, width)),
                       self._up(self._get(scope + "//offset").reshape(1, 1, 1, width)))

        b["initial_projection"] = self._linear(_PREFIX + "initial_projection")
        for i in range(3):
            b[f"transition_{i}"] = self._linear(
                _FOLD + "transition" + ("" if i == 0 else f"_{i}"))

        b["q_scalar"] = self._heads(_IPA + "q_scalar_projection")
        b["k_scalar"] = self._heads(_IPA + "k_scalar_projection")
        b["v_scalar"] = self._heads(_IPA + "v_scalar_projection")
        b["q_point"] = self._points(_IPA + "q_point_projection/point_projection", NUM_POINT_QK)
        b["k_point"] = self._points(_IPA + "k_point_projection/point_projection", NUM_POINT_QK)
        b["v_point"] = self._points(_IPA + "v_point_projection/point_projection", NUM_POINT_V)
        b["attention_2d"] = self._linear(_IPA + "attention_2d")

        # `point_weights = sqrt(1 / (num_point_qk * 4.5)) * softplus(trainable)`. The whole
        # expression is a constant of the checkpoint, so the softplus is evaluated on host once
        # rather than on every one of the eight layers. The -0.5 the logit carries is folded in
        # here too: the logit is `-0.5 * sum_p w * dist2`, and one sign on a [12] vector is the
        # same number as a multiply on a [1, 12, n, n] one.
        raw_pw = self._get(_IPA + "/trainable_point_weights")
        scale = math.sqrt(1.0 / (max(NUM_POINT_QK, 1) * 9.0 / 2.0))
        b["point_weights"] = self._up(
            (-0.5 * scale * _softplus(raw_pw)).reshape(1, NUM_HEAD, 1, 1),
            dtype=self.point_dtype)

        # The output projection, cut into the six groups its input is the concatenation of:
        # scalar (h*16), the three point coordinates (h*8 each), the point norms (h*8) and the
        # attention over the pair (h*128).
        w_out = self._get(_IPA + "output_projection//weights")            # [2112, c_s]
        groups, start = [], 0
        for width in (NUM_SCALAR_V, NUM_POINT_V, NUM_POINT_V, NUM_POINT_V, NUM_POINT_V, C_Z):
            block = w_out[start:start + NUM_HEAD * width]                 # head-major
            groups.append(self._up(block.reshape(NUM_HEAD, width, C_S)[None]))
            start += NUM_HEAD * width
        if start != w_out.shape[0]:
            raise ValueError(f"output projection is {w_out.shape[0]} wide, the six feature "
                             f"groups account for {start}")
        b["output_groups"] = groups
        b["output_bias"] = self._up(self._get(_IPA + "output_projection//bias")
                                    .reshape(1, 1, 1, C_S))

        # QuatRigid. Its six outputs are read one at a time and a width-1 slice of a six-wide
        # tile is not addressable, so the weight is cut into six `[c_s, 1]` columns and the bias
        # into six python floats -- an add of a scalar, not of a tensor.
        rigid_w = self._get(_FOLD + "quat_rigid/rigid//weights")
        rigid_b = self._get(_FOLD + "quat_rigid/rigid//bias")
        b["rigid_columns"] = [self._up(rigid_w[None, None, :, i:i + 1], dtype=self.point_dtype)
                              for i in range(6)]
        b["rigid_bias"] = [float(v) for v in rigid_b]

        b["sc_input"] = self._linear(_SIDECHAIN + "input_projection")
        b["sc_input_1"] = self._linear(_SIDECHAIN + "input_projection_1")
        for i in range(NUM_RESIDUAL_BLOCK):
            suffix = "" if i == 0 else f"_{i}"
            b[f"sc_res1_{i}"] = self._linear(_SIDECHAIN + "resblock1" + suffix)
            b[f"sc_res2_{i}"] = self._linear(_SIDECHAIN + "resblock2" + suffix)
        b["sc_angles"] = self._linear(_SIDECHAIN + "unnormalized_angles")

    def __getitem__(self, key):
        return self._built[key]


# ---------------------------------------------------------------------------- the frame

class Rigid:
    """A per-residue rigid transform: nine rotation components and three translations.

    Every component is `[1, 1, n, 1]`, which is the layout the point pipeline multiplies
    against: a `[1, h, n, p]` point tensor takes a `[1, 1, n, 1]` operand by broadcast on both
    the head axis and the point axis, so the rotation never has to be materialised per head.
    """

    __slots__ = ("rot", "trans")

    def __init__(self, rot, trans):
        self.rot = tuple(rot)           # xx xy xz yx yy yz zx zy zz
        self.trans = tuple(trans)       # x y z
        if len(self.rot) != 9 or len(self.trans) != 3:
            raise ValueError("a Rigid is nine rotation components and three translations")

    @classmethod
    def identity(cls, device, n, dtype):
        """Layer 0's frame. Ones and zeros, so it is the same number in every precision."""
        import torch
        one = torch.ones(1, 1, n, 1)
        zero = torch.zeros(1, 1, n, 1)

        def up(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

        return cls([up(one), up(zero), up(zero),
                    up(zero), up(one), up(zero),
                    up(zero), up(zero), up(one)],
                   [up(zero), up(zero), up(zero)])

    def apply(self, point):
        """`rotation @ point + translation`, on a 3-tuple of `[1, h, n, p]` tensors."""
        x, y, z = self.rotate(point)
        return (ttnn.add(x, self.trans[0]),
                ttnn.add(y, self.trans[1]),
                ttnn.add(z, self.trans[2]))

    def rotate(self, point):
        xx, xy, xz, yx, yy, yz, zx, zy, zz = self.rot
        px, py, pz = point
        return (_axpy(xx, px, xy, py, xz, pz),
                _axpy(yx, px, yy, py, yz, pz),
                _axpy(zx, px, zy, py, zz, pz))

    def apply_inverse(self, point):
        """`rotation^T @ (point - translation)`, which is what the IPA reads its values back
        through. The inverse of a rotation is its transpose, so this costs a reindex and no
        arithmetic beyond the subtraction."""
        xx, xy, xz, yx, yy, yz, zx, zy, zz = self.rot
        px = ttnn.subtract(point[0], self.trans[0])
        py = ttnn.subtract(point[1], self.trans[1])
        pz = ttnn.subtract(point[2], self.trans[2])
        return (_axpy(xx, px, yx, py, zx, pz),
                _axpy(xy, px, yy, py, zy, pz),
                _axpy(xz, px, yz, py, zz, pz))

    def compose(self, other: "Rigid") -> "Rigid":
        """`self @ other`, the frame update. `Rigid3Array.__matmul__`: the rotations multiply
        and the translation of `other` is carried through `self`."""
        a = self.rot
        b = other.rot
        rot = []
        for r in range(3):
            for c in range(3):
                rot.append(_axpy(a[3 * r + 0], b[0 + c],
                                 a[3 * r + 1], b[3 + c],
                                 a[3 * r + 2], b[6 + c]))
        return Rigid(rot, self.apply(other.trans))

    def to_array(self, scale: float = 1.0):
        """`scale_translation(scale).to_array()`, flattened to the twelve `[1, 1, n, 1]`
        tensors in the order `Rigid3Array.to_array` writes them: the rotation's three rows,
        each followed by that row's translation component."""
        t = [ttnn.multiply(c, scale) for c in self.trans] if scale != 1.0 else list(self.trans)
        return [self.rot[0], self.rot[1], self.rot[2], t[0],
                self.rot[3], self.rot[4], self.rot[5], t[1],
                self.rot[6], self.rot[7], self.rot[8], t[2]]


def _axpy(a, x, b, y, c, z):
    """`a*x + b*y + c*z`. Three multiplies and two adds, written once because the rotation
    apply, its inverse and the rotation compose are all this expression."""
    return ttnn.add(ttnn.add(ttnn.multiply(a, x), ttnn.multiply(b, y)), ttnn.multiply(c, z))


# ---------------------------------------------------------------------------- the module

class AF2MultimerStructureModule:
    """`folding_multimer.StructureModule`'s eight fold iterations, on the card.

    Call it with the trunk's `single` and `pair` and it returns the final activation, the
    per-layer frames and the per-layer unnormalised torsion angles. What it does NOT do is
    build atoms: `torsion_angles_to_frames` and
    `frames_and_literature_positions_to_atom14_pos` are `aatype` gathers of constant tables
    over an axis of 8 and 14, no matmul in either, and they are left where they are. Every
    differentiated matmul in the module is here.
    """

    def __init__(self, weights: StructureWeights, *, num_layer: int = NUM_LAYER):
        self.w = weights
        self.device = weights.device
        self.dtype = weights.dtype
        self.point_dtype = weights.point_dtype
        self.num_layer = num_layer
        self.ckc = compute_kernel_config(self.device)

    # -- helpers ------------------------------------------------------------------------

    def _cast(self, x, dtype):
        return x if x.dtype == dtype else ttnn.typecast(x, dtype)

    def _mm(self, a, b):
        return ttnn.matmul(a, b, compute_kernel_config=self.ckc)

    def _lin(self, x, w, bias=None):
        return ops.linear(x, w, bias=bias, compute_kernel_config=self.ckc)

    def _ln(self, x, weight, bias, *, epsilon):
        return ops.layer_norm(x, weight, bias, epsilon=epsilon,
                              compute_kernel_config=self.ckc)

    def _heads(self, act, w):
        """`[1, 1, n, c_s] @ [1, h, c_s, d] -> [1, h, n, d]`. The head axis is the matmul's
        batch and `act` broadcasts into it, so no tensor is ever split into heads."""
        return self._mm(act, w)

    def _point(self, act_p, ws, bs, rigid):
        """A `PointProjection`: three coordinates projected per head, then pushed through the
        residue's frame. Returns the global point as a 3-tuple of `[1, h, n, p]`."""
        local = [ttnn.add(self._mm(act_p, ws[i]), bs[i]) for i in range(3)]
        return rigid.apply(local), local

    # -- invariant point attention -------------------------------------------------------

    def ipa(self, act, act_2d, rigid, mask_bias, n):
        w = self.w
        act_p = self._cast(act, self.point_dtype)

        # The scalar term. `q_scalar *= sqrt(1/num_scalar_qk)` before the contraction, which
        # is one multiply on a [1,h,n,16] rather than on the [1,h,n,n] logit.
        q = ttnn.multiply(self._heads(act, w["q_scalar"]),
                          math.sqrt(1.0 / max(NUM_SCALAR_QK, 1)))
        k = self._heads(act, w["k_scalar"])
        logits = self._mm(q, ttnn.transpose(k, -2, -1))              # [1, h, n, n]
        logits = self._cast(logits, self.point_dtype)

        # The point term, as |Q|^2 + |K|^2 - 2 Q.K over the twelve flattened components of a
        # head's four points. `point_weights` already carries the -0.5 and the softplus.
        (qx, qy, qz), _ = self._point(act_p, *w["q_point"], rigid)
        (kx, ky, kz), _ = self._point(act_p, *w["k_point"], rigid)
        qp = ttnn.concat([qx, qy, qz], dim=-1)                          # [1, h, n, 12]
        kp = ttnn.concat([kx, ky, kz], dim=-1)
        qq = ttnn.sum(ttnn.multiply(qp, qp), dim=-1, keepdim=True)      # [1, h, n, 1]
        kk = ttnn.sum(ttnn.multiply(kp, kp), dim=-1, keepdim=True)
        dist2 = ttnn.add(ttnn.add(qq, ttnn.transpose(kk, -2, -1)),
                         ttnn.multiply(self._mm(qp, ttnn.transpose(kp, -2, -1)), -2.0))
        logits = ttnn.add(logits, ttnn.multiply(dist2, w["point_weights"]))

        # The pair term. `attention_2d` writes [1, n, n, h]; the head axis becomes the batch.
        a2d = self._lin(act_2d, w["attention_2d"][0], bias=w["attention_2d"][1])
        logits = ttnn.add(logits, self._cast(ttnn.permute(a2d, (0, 3, 1, 2)),
                                             self.point_dtype))

        # The mask, then sqrt(1/3) over the three logit terms -- AFTER the mask here, which is
        # where the multimer module differs from the monomer one.
        logits = ttnn.add(logits, mask_bias)
        logits = ttnn.multiply(logits, math.sqrt(1.0 / 3.0))
        attn = ttnn.softmax(logits, dim=-1)

        features = []
        # scalar values
        v = self._heads(act, w["v_scalar"])                             # [1, h, n, 16]
        attn_s = self._cast(attn, self.dtype)
        features.append(self._mm(attn_s, v))                         # [1, h, n, 16]

        # point values, read back into the query residue's own frame
        (vx, vy, vz), _ = self._point(act_p, *w["v_point"], rigid)
        attn_p = self._cast(attn, self.point_dtype)
        glob = tuple(self._mm(attn_p, c) for c in (vx, vy, vz))      # [1, h, n, 8]
        local = rigid.apply_inverse(glob)
        norm2 = ttnn.add(ttnn.add(ttnn.multiply(local[0], local[0]),
                                  ttnn.multiply(local[1], local[1])),
                         ttnn.multiply(local[2], local[2]))
        # `Vec3Array.norm(eps)` clips the SQUARE at eps^2 before the sqrt, which is what keeps
        # the backward finite on a residue whose points all landed on its own frame origin.
        norms = ttnn.sqrt(ttnn.clamp(norm2, min=DIST_EPSILON ** 2))
        features.extend(self._cast(t, self.dtype)
                        for t in (local[0], local[1], local[2], norms))

        # the attention over the pair track, contracted over the key residue
        pair_out = self._mm(ttnn.permute(attn_s, (0, 2, 1, 3)), act_2d)   # [1, n, h, c_z]
        features.append(ttnn.permute(pair_out, (0, 2, 1, 3)))                # [1, h, n, c_z]

        # The output projection, as six per-head matmuls summed instead of a concatenation of
        # six sub-tile-wide features followed by one. Same arithmetic, and no 2112-wide
        # intermediate that only exists to be read once.
        out = None
        for feat, wg in zip(features, w["output_groups"]):
            part = self._mm(feat, wg)                                # [1, h, n, c_s]
            out = part if out is None else ttnn.add(out, part)
        out = ttnn.sum(out, dim=1, keepdim=True)                        # [1, 1, n, c_s]
        return ttnn.add(out, w["output_bias"])

    # -- the frame update ----------------------------------------------------------------

    def quat_rigid(self, act):
        """`QuatRigid`: six columns, three of them a quaternion with w pinned to 1."""
        w = self.w
        act_p = self._cast(act, self.point_dtype)
        cols = [ttnn.add(self._mm(act_p, w["rigid_columns"][i]), w["rigid_bias"][i])
                for i in range(6)]
        qx, qy, qz = cols[0], cols[1], cols[2]
        # `from_quaternion(normalize=True)` takes rsqrt of `max(1e-6, w^2+x^2+y^2+z^2)` with
        # w == 1, so the argument is at least 1 and the clip is unreachable -- in the value and
        # in the gradient. Leaving it out removes an op and changes no number.
        s = ttnn.add(ttnn.add(ttnn.multiply(qx, qx), ttnn.multiply(qy, qy)),
                     ttnn.add(ttnn.multiply(qz, qz), 1.0))
        inv = ttnn.rsqrt(s)
        qw = inv
        qx = ttnn.multiply(qx, inv)
        qy = ttnn.multiply(qy, inv)
        qz = ttnn.multiply(qz, inv)

        xy, xz, yz = (ttnn.multiply(qx, qy), ttnn.multiply(qx, qz), ttnn.multiply(qy, qz))
        wx, wy, wz = (ttnn.multiply(qw, qx), ttnn.multiply(qw, qy), ttnn.multiply(qw, qz))
        xx2, yy2, zz2 = (ttnn.multiply(qx, qx), ttnn.multiply(qy, qy), ttnn.multiply(qz, qz))
        rot = [
            ttnn.add(ttnn.multiply(ttnn.add(yy2, zz2), -2.0), 1.0),
            ttnn.multiply(ttnn.subtract(xy, wz), 2.0),
            ttnn.multiply(ttnn.add(xz, wy), 2.0),
            ttnn.multiply(ttnn.add(xy, wz), 2.0),
            ttnn.add(ttnn.multiply(ttnn.add(xx2, zz2), -2.0), 1.0),
            ttnn.multiply(ttnn.subtract(yz, wx), 2.0),
            ttnn.multiply(ttnn.subtract(xz, wy), 2.0),
            ttnn.multiply(ttnn.add(yz, wx), 2.0),
            ttnn.add(ttnn.multiply(ttnn.add(xx2, yy2), -2.0), 1.0),
        ]
        return Rigid(rot, [cols[3], cols[4], cols[5]])

    # -- the sidechain -------------------------------------------------------------------

    def sidechain(self, act, initial_act):
        """`MultiRigidSidechain` up to the unnormalised angles.

        `l2_normalize` and the torsion frames want a 2-wide and an 8-wide axis; both are
        elementwise on `[n, 7, 2]` and neither is a matmul, so they stay on the host with the
        atom building they feed. Every Linear in the module is here."""
        w = self.w
        a = ttnn.add(self._lin(ttnn.relu(act), *w["sc_input"]),
                     self._lin(ttnn.relu(initial_act), *w["sc_input_1"]))
        for i in range(NUM_RESIDUAL_BLOCK):
            r = self._lin(ttnn.relu(a), *w[f"sc_res1_{i}"])
            r = self._lin(ttnn.relu(r), *w[f"sc_res2_{i}"])
            a = ttnn.add(a, r)
        return self._lin(ttnn.relu(a), *w["sc_angles"])                # [1, 1, n, 14]

    # -- one iteration -------------------------------------------------------------------

    def fold_iteration(self, act, act_2d, rigid, initial_act, mask_bias, n):
        w = self.w
        act = ttnn.add(act, self.ipa(act, act_2d, rigid, mask_bias, n))
        act = self._ln(act, *w["attention_norm"], epsilon=LAYER_NORM_EPS)
        residual = act
        for i in range(3):
            act = self._lin(act, *w[f"transition_{i}"])
            if i < 2:
                act = ttnn.relu(act)
        act = ttnn.add(act, residual)
        act = self._ln(act, *w["transition_norm"], epsilon=LAYER_NORM_EPS)
        rigid = rigid.compose(self.quat_rigid(act))
        # BindCraft 2 has the stop-gradient on the rotation commented out
        # (`folding_multimer.py:467`), so the frame carried to the next layer is the one the
        # cotangent flows back through. Nothing is detached here on purpose.
        return act, rigid, self.sidechain(act, initial_act)

    # -- the module ----------------------------------------------------------------------

    def __call__(self, single, pair, seq_mask, *, initial_rigid: "Rigid | None" = None):
        """`single` is `[1, 1, n, c_s]`, `pair` is `[1, n, n, c_z]`, `seq_mask` is
        `[1, 1, n, 1]`. Returns `(act, traj, angles)` where `traj` and `angles` hold one entry
        per layer."""
        w = self.w
        n = int(single.shape[-2])
        act = self._ln(single, *w["single_norm"], epsilon=LAYER_NORM_EPS)
        initial_act = act
        act = self._lin(act, *w["initial_projection"])
        act_2d = self._ln(pair, *w["pair_norm"], epsilon=LAYER_NORM_EPS)

        # `-1e5 * (1 - mask_i * mask_j)`, built once rather than per layer. It is a constant of
        # the batch, so it carries no gradient and the tape never sees the subtraction.
        mask_2d = self._mm(seq_mask, ttnn.transpose(seq_mask, -2, -1))   # [1, 1, n, n]
        mask_bias = ttnn.multiply(ttnn.subtract(mask_2d, 1.0), 1e5)
        mask_bias = self._cast(mask_bias, self.point_dtype)

        rigid = initial_rigid or Rigid.identity(self.device, n, self.point_dtype)
        traj, angles = [], []
        for _ in range(self.num_layer):
            act, rigid, unnorm = self.fold_iteration(
                act, act_2d, rigid, initial_act, mask_bias, n)
            traj.append(rigid.to_array(POSITION_SCALE))
            angles.append(unnorm)
        return act, traj, angles
