"""AlphaFold 2 multimer's structure module as a short chain of dense ops, so it can run on card.

BindCraft 2 differentiates a design through AlphaFold 2, and the structure module is the largest
piece of that step still on the host (0.27 s an iteration on pc, measured as a removal). Run on card
it reads the Evoformer's final pair where it already lives instead of a copy of it.

Every step of a FoldIteration is rewritten as an op ttnn has (matmul, batched matmul, multiply,
add, softmax, layer norm, concat, reshape, permute), with the bookkeeping folded into constant
matrices built once on the host:

  * The six IPA projections are ONE [384, 1152] matmul, columns [q | k | v | X | Y | Z], so moving
    the points into the global frame is one batched [3, 3] matmul per residue.
  * The point term is folded into the attention head: logits = q'.k' + bias' with
    q' = s3 [q/4, pw qp, 1, -pw/2 |qp|^2], k' = [k, kp, -pw/2 |kp|^2, 1], so the logits are AF2's
    exact logits and nothing is broadcast. q' and k' pad to 32 per head and v' to 64.
  * The quaternion update is a linear map to (1, qx, qy, qz), the ten quadratic monomials and a
    constant [10, 9] combination divided by |q|^2.

The backend `o` is the op set. `NP` runs it in float64 numpy, card-free, which is how the rewrite
is proven against BindCraft 2's own FoldIteration (`perf/bc4_headdev/sm_check.py`); `Card` runs it
on tt_bio.autograd's tape in fp32, so the backward is the tape's VJPs. Every add and multiply takes
equal shapes, because the tape's add and mul have no reducing backward.
"""
import numpy as np

H, C, PQ, PV, CS, CZ = 12, 16, 4, 8, 384, 128
LAYERS = 8
NP_ = 2 * PQ + PV                      # points per head across q, k, v: 16
DQ, DV = 32, 64                        # padded head widths of q'/k' and of v'
ROOT = "alphafold/alphafold_iteration/structure_module/"
SM = ROOT + "fold_iteration/"
IPA = SM + "invariant_point_attention/"
S3 = np.sqrt(1.0 / 3)


def pack(p):
    """Every constant the layer needs, float64, from the golden model's parameter dict."""
    W = lambda n: p[n + "//weights"]
    B = lambda n, shape: p[n + "//bias"] if n + "//bias" in p else np.zeros(shape)
    cols, bias = [], []
    for n in ("q", "k", "v"):                                       # scalar: [384, H*C], no bias
        cols.append(W(f"{IPA}{n}_scalar_projection").reshape(CS, H * C)); bias.append(np.zeros(H * C))
    pts = {n: (W(f"{IPA}{n}_point_projection/point_projection"), B(f"{IPA}{n}_point_projection/point_projection", None))
           for n in ("q", "k", "v")}
    for d in range(3):                                              # X, Y, Z blocks: [q pts | k pts | v pts]
        for n, P in (("q", PQ), ("k", PQ), ("v", PV)):
            w, b = pts[n]
            cols.append(w[:, :, d * P:(d + 1) * P].reshape(CS, H * P)); bias.append(b[:, d * P:(d + 1) * P].reshape(-1))
    cols.append(np.zeros((CS, 1))); bias.append(np.ones(1))          # a constant-1 column, q' slot 28
    k = {"w_proj": np.concatenate(cols, 1), "b_proj": np.concatenate(bias)}
    ONE = k["w_proj"].shape[1] - 1
    off = {"q": 0, "k": H * C, "v": 2 * H * C}
    po = {"q": 0, "k": H * PQ, "v": 2 * H * PQ}                     # offsets inside one X/Y/Z block
    XB = 3 * H * C                                                  # first X column
    blk = H * NP_                                                   # width of one coordinate block
    raw = p[IPA + "/trainable_point_weights"]
    pw = np.sqrt(1.0 / (PQ * 9.0 / 2)) * np.logaddexp(raw, 0.0)
    width = k["w_proj"].shape[1]

    def sel(dpad, parts):
        """[width, H*dpad]: head h's slot j takes column `parts[j](h)` times its scale."""
        m = np.zeros((width, H * dpad))
        for h in range(H):
            j = 0
            for col, scale in parts(h):
                m[col, h * dpad + j] = scale; j += 1
        return m

    def head_parts(n, P, scalar_scale, point_scale, one=False):
        def f(h):
            out = [(off[n] + h * C + c, scalar_scale) for c in range(C)]
            for d in range(3):
                out += [(XB + d * blk + po[n] + h * P + i, point_scale(h)) for i in range(P)]
            return out + ([(ONE, 1.0)] if one else [])
        return f
    k["sel_q"] = sel(DQ, head_parts("q", PQ, S3 * np.sqrt(1.0 / C), lambda h: S3 * pw[h], one=True))
    k["sel_k"] = sel(DQ, head_parts("k", PQ, 1.0, lambda h: 1.0))
    k["sel_v"] = sel(DV, head_parts("v", PV, 1.0, lambda h: 1.0))
    # sum |kp|^2 over a head's points, times -s3 pw/2, into k' slot 28 opposite q's constant 1: the
    # key-side bias rides inside the one logits matmul, and nothing is broadcast (the tape's add and
    # mul take equal shapes only).
    def sq_sel(n, slot):
        """[3 blk, H*DQ]: the squared global coordinates of head h's n-points, summed, times -s3 pw/2."""
        m = np.zeros((3 * blk, H * DQ))
        for d in range(3):
            for h in range(H):
                m[d * blk + po[n] + h * PQ: d * blk + po[n] + (h + 1) * PQ, h * DQ + slot] = -0.5 * S3 * pw[h]
        return m
    k["sel_k2"] = sq_sel("k", C + 3 * PQ)
    # and -s3 pw/2 |qp|^2 into q' slot 29 opposite a constant 1 in k': the logits are then the exact
    # AF2 logits, not shifted by a per-row constant of ~1e3. The card's fp32 row max (and so
    # ttnn.softmax) is off by up to 240 at that magnitude (NOTES "Pass 5"), and is exact near 0.
    for h in range(H):
        k["sel_k"][ONE, h * DQ + C + 3 * PQ + 1] = 1.0
    k["sel_q2"] = sq_sel("q", C + 3 * PQ + 1)
    k["w_2d"], k["b_2d"] = W(IPA + "attention_2d") * S3, B(IPA + "attention_2d", (H,)) * S3
    # o = attn @ v' is [H, N, 64]: per head [o_s (16) | ox (8) | oy (8) | oz (8) | pad]. After
    # unheads that is [N, H*64]; the output projection's rows are rearranged to read it in place,
    # with the inverse-rotated points and norms appended as their own row blocks.
    w_out = W(IPA + "output_projection")
    s_rows = w_out[:H * C]
    pt_rows = [w_out[H * C + d * H * PV: H * C + (d + 1) * H * PV] for d in range(3)]
    norm_rows = w_out[H * C + 3 * H * PV: H * C + 4 * H * PV]
    pair_rows = w_out[H * C + 4 * H * PV:]
    w_os = np.zeros((H * DV, CS))
    for h in range(H):
        w_os[h * DV:h * DV + C] = s_rows[h * C:(h + 1) * C]
    k["w_out_s"], k["w_out_pt"], k["w_out_norm"], k["w_out_pair"] = w_os, np.concatenate(pt_rows), norm_rows, pair_rows
    k["b_out"] = B(IPA + "output_projection", (CS,))
    ex = np.zeros((3, H * DV, H * PV))                              # pull ox/oy/oz out of the head layout
    for h in range(H):
        for d in range(3):
            for i in range(PV):
                ex[d, h * DV + C + d * PV + i, h * PV + i] = 1.0
    k["sel_opt"] = np.concatenate(list(ex), 1)                     # [H*DV, 3 H*PV], x | y | z
    k["sum3"] = np.tile(np.eye(H * PV), (3, 1))                     # [3 H*PV, H*PV]: x^2 + y^2 + z^2
    for n in ("attention_layer_norm", "transition_layer_norm"):
        k[n] = (p[SM + n + "//scale"], p[SM + n + "//offset"])
    for n in ("transition", "transition_1", "transition_2"):
        k[n] = (W(SM + n), B(SM + n, (CS,)))
    wq, bq = W(SM + "quat_rigid/rigid"), B(SM + "quat_rigid/rigid", (6,))
    k["w_quat"] = np.concatenate([np.zeros((CS, 1)), wq[:, :3]], 1)          # (1, qx, qy, qz)
    k["b_quat"] = np.concatenate([[1.0], bq[:3]])
    k["w_dt"], k["b_dt"] = wq[:, 3:6], bq[3:6]
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]         # ww wx wy wz xx xy xz yy yz zz
    k["mono_a"], k["mono_b"] = (np.eye(4)[:, [a for a, _ in pairs]], np.eye(4)[:, [b for _, b in pairs]])
    I = {pr: i for i, pr in enumerate(pairs)}
    c = np.zeros((10, 9)); n2 = np.zeros((10, 1))
    def put(r, terms):
        for coef, pr in terms:
            c[I[pr], r] += coef
    w_, x_, y_, z_ = 0, 1, 2, 3
    put(0, [(1, (w_, w_)), (1, (x_, x_)), (-1, (y_, y_)), (-1, (z_, z_))])     # 1 - 2(y2 + z2) on a unit q
    put(1, [(2, (x_, y_)), (-2, (w_, z_))]); put(2, [(2, (x_, z_)), (2, (w_, y_))])
    put(3, [(2, (x_, y_)), (2, (w_, z_))])
    put(4, [(1, (w_, w_)), (-1, (x_, x_)), (1, (y_, y_)), (-1, (z_, z_))])
    put(5, [(2, (y_, z_)), (-2, (w_, x_))]); put(6, [(2, (x_, z_)), (-2, (w_, y_))])
    put(7, [(2, (y_, z_)), (2, (w_, x_))])
    put(8, [(1, (w_, w_)), (-1, (x_, x_)), (-1, (y_, y_)), (1, (z_, z_))])
    for pr in ((w_, w_), (x_, x_), (y_, y_), (z_, z_)):
        n2[I[pr], 0] = 1.0
    k["quat_rot"], k["quat_n2"] = c, np.repeat(n2, 9, 1)            # |q|^2 already 9 wide: no broadcast
    # t repeated to width w per coordinate by one matmul: a broadcast add has no reducing backward
    # on the tape, a matmul's backward is the reduction.
    for w in (blk, H * PV):
        k[f"exp3_{w}"] = np.kron(np.eye(3), np.ones((1, w)))
    k["one_col"] = ONE
    k["pair_layer_norm"] = (p[ROOT + "pair_layer_norm//scale"], p[ROOT + "pair_layer_norm//offset"])
    return k


class NP:
    """float64 host backend: the op set the card backend has to supply. `xp` is numpy, or
    jax.numpy with x64 on, which makes the same chain differentiable for a float64 reference VJP."""

    def __init__(self, xp=np):
        self.xp = xp

    def put(self, x): return self.xp.asarray(x, self.xp.float64)
    def get(self, x): return np.asarray(x)
    def identity(self, n): return self.put(np.tile(np.eye(3).reshape(1, 9), (n, 1))), self.put(np.zeros((n, 3)))
    mm = bmm = staticmethod(lambda a, b: a @ b)
    linear = staticmethod(lambda x, w, b=None: x @ w + (0 if b is None else b))
    add = staticmethod(lambda a, b: a + b); sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    col = staticmethod(lambda x, i: x[..., i:i + 1])
    cols = staticmethod(lambda x, s, e: x[..., s:e])
    def concat(self, xs): return self.xp.concatenate(xs, -1)
    heads = staticmethod(lambda x, d: x.reshape(x.shape[0], H, d).transpose(1, 0, 2))   # [N, H*d] -> [H, N, d]
    unheads = staticmethod(lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1))       # [H, N, d] -> [N, H*d]
    bmm_t = staticmethod(lambda a, b: a @ b.swapaxes(-1, -2))
    permute = staticmethod(lambda x, perm: x.transpose(perm))
    flat = staticmethod(lambda x: x.reshape(x.shape[0], -1))
    reshape = staticmethod(lambda x, shape: x.reshape(shape))
    def relu(self, x): return self.xp.maximum(x, 0)
    def sqrt_eps(self, x, eps): return self.xp.sqrt(x + eps)
    recip = staticmethod(lambda x: 1.0 / x)
    def softmax(self, x):
        e = self.xp.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    def layer_norm(self, x, gb, eps=1e-5):
        mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / self.xp.sqrt(var + eps) * gb[0] + gb[1]


def layer(o, k, act, R, t, bias, z_rows, drop=None):
    """One FoldIteration. act [N, 384], R [N, 9], t [N, 3], bias [H, N, N] from `pair_bias`,
    z_rows = z as [N, N, 128] (batch = query residue), drop = None or dropout's two keep / keep_rate
    masks [N, 384] (drawn in JAX from the layer's key, see ipa_phases.check). Returns (act, R, t)."""
    n = act.shape[0]
    pr = o.linear(act, k["w_proj"], k["b_proj"])                         # [N, 1152]
    XB, blk = 3 * H * C, H * NP_
    R3 = o.reshape(R, [n, 3, 3])
    # global points R p + t: one batched [3, 3] x [3, blk] matmul per residue
    g = o.add(o.bmm(R3, o.reshape(o.cols(pr, XB, XB + 3 * blk), [n, 3, blk])),
              o.reshape(o.mm(t, k[f"exp3_{blk}"]), [n, 3, blk]))
    g = o.reshape(g, [n, 3 * blk])
    one = k["one_col"]
    full = o.concat([o.cols(pr, 0, XB), g, o.cols(pr, one, one + 1)])   # global points, constant 1
    g2 = o.mul(g, g)
    q = o.heads(o.add(o.mm(full, k["sel_q"]), o.mm(g2, k["sel_q2"])), DQ)   # [H, N, 32]
    kk = o.heads(o.add(o.mm(full, k["sel_k"]), o.mm(g2, k["sel_k2"])), DQ)
    v = o.heads(o.mm(full, k["sel_v"]), DV)                              # [H, N, 64]
    logits = o.add(o.bmm_t(q, kk), bias)
    attn = o.softmax(logits)                                             # [H, Nq, Nk]
    ov = o.unheads(o.bmm(attn, v))                                       # [N, H*64]
    # back to the local frame, R^T (o - t), one batched matmul again
    w = H * PV
    d = o.sub(o.reshape(o.mm(ov, k["sel_opt"]), [n, 3, w]), o.reshape(o.mm(t, k[f"exp3_{w}"]), [n, 3, w]))
    loc = o.reshape(o.bmm(o.permute(R3, (0, 2, 1)), d), [n, 3 * w])
    norms = o.sqrt_eps(o.mm(o.mul(loc, loc), k["sum3"]), 1e-16)
    o_pair = o.bmm(o.permute(attn, (1, 0, 2)), z_rows)                   # [N, H, 128]
    o_pair = o.flat(o_pair)
    ipa = o.add(o.add(o.add(o.linear(ov, k["w_out_s"], k["b_out"]), o.mm(loc, k["w_out_pt"])),
                      o.mm(norms, k["w_out_norm"])), o.mm(o_pair, k["w_out_pair"]))
    dr = (lambda x, i: x) if drop is None else (lambda x, i: o.mul(x, drop[i]))
    act = o.layer_norm(dr(o.add(act, ipa), 0), k["attention_layer_norm"])
    h = o.relu(o.linear(act, *k["transition"]))
    h = o.relu(o.linear(h, *k["transition_1"]))
    act = o.layer_norm(dr(o.add(act, o.linear(h, *k["transition_2"])), 1), k["transition_layer_norm"])
    qv = o.linear(act, k["w_quat"], k["b_quat"])                         # [N, 4] = (1, qx, qy, qz)
    mono = o.mul(o.mm(qv, k["mono_a"]), o.mm(qv, k["mono_b"]))           # [N, 10]
    r_u = o.mul(o.mm(mono, k["quat_rot"]), o.recip(o.mm(mono, k["quat_n2"])))
    dt = o.linear(act, k["w_dt"], k["b_dt"])                             # [N, 3] in the old frame
    t_new = o.add(o.reshape(o.bmm(R3, o.reshape(dt, [n, 3, 1])), [n, 3]), t)
    R_new = o.reshape(o.bmm(R3, o.reshape(r_u, [n, 3, 3])), [n, 9])
    return act, R_new, t_new


def device_bytes(padded):
    """Device bytes `Card` holds at its peak for one taped stack and its backward at `padded` tokens:
    the fp32 pair copy, key bias, dropout keeps, the tape and the backward's transient. They are
    live while the Evoformer's tape is, so the Evoformer's resident budget has to leave them room.
    Fitted on a Blackhole chip at 224/352/512/672 (0.75/1.37/2.32/3.50 GB, all within 0.01 GB);
    the per-token term is the [N, 3, w] frame tensors, each padded to a 32-row tile.
    `perf/bc4_stack/sm_bytes.py`."""
    return int(2.434e6 * padded + 4119 * padded ** 2)


def key_bias(mask):
    """-1e5 s3 on every masked key, [H, N, N].

    AF2 masks 1 - m_i m_j. On a real query row that is the same; on a padded one AF2's -1e5 shifts
    the whole row and cancels in the softmax, which needs a row max to be safe. Masking keys only
    keeps every row's logits small and leaves every real row's result unchanged; padded rows read
    nothing and their outputs are sliced away."""
    n = mask.shape[0]
    return np.broadcast_to(-1e5 * S3 * (1.0 - np.asarray(mask, np.float64))[None, None, :], (H, n, n))


def stack(o, k, act, z, kbias, drops=None, layers=LAYERS):
    """The structure module after its initial projection: pair layer norm, the pair bias (the same
    in all eight layers, so computed once) and eight FoldIterations from identity frames.

    act [N, 384] is `initial_projection(single_layer_norm(single))`, z the Evoformer's final pair
    [N, N, 128], kbias `key_bias(mask)` on the backend, drops None or eight pairs of keep / keep_rate
    masks [N, 384] drawn from the layers' own keys. Returns the final act and every layer's (R, t),
    R [N, 9] row-major, t [N, 3] in AF2's unscaled units."""
    zl = o.layer_norm(z, k["pair_layer_norm"])
    bias = o.add(o.permute(o.linear(zl, k["w_2d"], k["b_2d"]), (2, 0, 1)), kbias)
    R, t = o.identity(int(act.shape[0]))
    traj = []
    for i in range(layers):
        act, R, t = layer(o, k, act, R, t, bias, zl, None if drops is None else drops[i])
        traj.append((R, t))
    return act, traj


class Card:
    """The op set on tt_bio.autograd's tape, fp32 TILE tensors at HiFi4. Weights, masks and the
    identity frames are constants (BindCraft 2 differentiates the sequence, not the weights)."""

    def __init__(self, device):
        import ttnn
        from tt_bio import autograd as A
        self.t, self.A, self.device = ttnn, A, device
        self._identity = {}
        self.ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                    math_approx_mode=False, fp32_dest_acc_en=True,
                                                    packer_l1_acc=True)

    def raw(self, x):
        import torch
        return self.t.from_torch(torch.as_tensor(np.ascontiguousarray(x, np.float32)),
                                 dtype=self.t.float32, layout=self.t.TILE_LAYOUT, device=self.device)

    def put(self, x, grad=False):
        return self.A.Tensor(self.raw(x), requires_grad=grad)

    def identity(self, n):
        """Identity frames for n residues, uploaded once: a trace capture refuses a host write."""
        if n not in self._identity:
            self._identity[n] = (self.put(np.tile(np.eye(3).reshape(1, 9), (n, 1))), self.put(np.zeros((n, 3))))
        return self._identity[n]

    def upload(self, k):
        """`pack`'s constants on card; 1-D biases and layer-norm parameters as [1, w] rows."""
        def up(v):
            if isinstance(v, tuple):
                return tuple(up(x) for x in v)
            if isinstance(v, np.ndarray):
                return self.put(v[None] if v.ndim == 1 else v)
            return v
        return {name: up(v) for name, v in k.items()}

    def mm(self, a, b): return self.A.matmul(a, b)
    bmm = mm
    def bmm_t(self, a, b): return self.A.matmul(a, b, transpose_b=True)
    def linear(self, x, w, b=None): return self.A.linear(x, w, b)
    def add(self, a, b): return self.A.add(a, b)
    def mul(self, a, b): return self.A.mul(a, b)
    def sub(self, a, b): return self.A.add(a, self.A.scale(b, -1.0))
    def cols(self, x, s, e): return self.A.narrow(x, -1, s, e - s)
    def col(self, x, i): return self.cols(x, i, i + 1)
    def concat(self, xs): return self.A.concat(xs, dim=-1)
    def permute(self, x, perm): return self.A.permute(x, perm)
    def heads(self, x, d): return self.A.permute(self.A.reshape(x, [int(x.shape[0]), H, d]), (1, 0, 2))
    def unheads(self, x):
        h, n, d = (int(s) for s in x.shape)
        return self.A.reshape(self.A.permute(x, (1, 0, 2)), [n, h * d])
    def flat(self, x): return self.A.reshape(x, [int(x.shape[0]), -1])
    def reshape(self, x, shape): return self.A.reshape(x, shape)
    def relu(self, x): return self.A.relu(x)
    def softmax(self, x): return self.A.softmax(x, -1, config=self.ckc, refine=True)   # one-hot rows
    def layer_norm(self, x, gb, eps=1e-5): return self.A.layer_norm(x, gb[0], gb[1], eps=eps)

    def sqrt_eps(self, x, eps):
        """sqrt(x + eps); backward g / (2 out)."""
        t, out = self.t, self.t.sqrt(self.t.add(x.value, eps))
        def make():
            def bw(g):
                x.add_grad(t.multiply(g, t.reciprocal(t.multiply(out, 2.0))))
            return bw
        return self.A._tape(out, [x], make)

    def recip(self, x):
        """1 / x; backward -g out^2."""
        t, out = self.t, self.t.reciprocal(x.value)
        def make():
            def bw(g):
                x.add_grad(t.multiply(g, t.multiply(t.multiply(out, out), -1.0)))
            return bw
        return self.A._tape(out, [x], make)
