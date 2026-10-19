#!/usr/bin/env python3
"""One AF2-multimer structure-module layer as a short chain of dense ops, for the card.

`ipa_phases.py` is the float64 golden model in the kernel's phase order. This file is the same layer
rewritten so that every step is an op ttnn has (matmul, batched matmul, broadcast multiply, add,
softmax, layer norm, concat, reshape/permute), with the bookkeeping folded into constant matrices
built once on the host:

  * The six IPA projections are ONE [384, 1152] matmul. Its columns are [q | k | v | X | Y | Z],
    X/Y/Z holding the x, y, z coordinate of every q, k and v point in the same order, so moving the
    points into the global frame is nine broadcast multiplies against the rotation's columns.
  * The point term is folded into the attention head (see NOTES "Pass 2"):
        logits = q'.k' + bias',  q' = s3 [q/4, pw qp],  k' = [k, kp],
        bias'  = s3 (b_2d - pw/2 |kp|^2 - 1e5 (1 - m_i m_j)),
    the -pw/2 |qp|^2 row constant cancelling in the softmax. q' and k' are padded to 32 per head and
    v' = [v, vp] to 64, so each head is tile aligned. They are built from the projection by one
    constant "selection" matmul each, which also carries s3 and pw.
  * The quaternion update is a linear map to (1, qx, qy, qz), the ten quadratic monomials by two
    selection matmuls and a multiply, and the rotation entries as a constant [10, 9] combination
    divided by |q|^2. Composing two rotations is three products of selection matmuls.

`Backend` is the op set. `NP` runs it in float64 numpy, card-free, so `--check` proves the rewrite
against the golden model before a card is ever opened; the ttnn backend lives in `ipa_fused_tt.py`.
Calls per layer: 185, against 330 in the DeviceIPA chain; one attention and two
batched matmuls are the only [H, N, N]-sized ones. Nothing is broadcast: every add and multiply takes
equal shapes, so tt_bio.autograd's tape (whose add/mul have no reducing backward) can differentiate it.
usage: ipa_fused.py --params <params_model_1_multimer_v3.npz> --check [--n 224]
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipa_phases as G

H, C, PQ, PV, CS, CZ = G.H, G.C, G.PQK, G.PV, G.CS, G.CZ
NP_ = 2 * PQ + PV                      # points per head across q, k, v: 16
DQ, DV = 32, 64                        # padded head widths of q'/k' and of v'
IPA, SM = G.SM + "invariant_point_attention/", G.SM
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
    sk = np.zeros((H * PQ, H * DQ))
    for h in range(H):
        sk[h * PQ:(h + 1) * PQ, h * DQ + C + 3 * PQ] = -0.5 * S3 * pw[h]
    k["sel_kp2"], k["kp_cols"] = sk, (po["k"], po["k"] + H * PQ)
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
    k["w_out_s"], k["w_out_pt"], k["w_out_norm"], k["w_out_pair"] = w_os, pt_rows, norm_rows, pair_rows
    k["b_out"] = B(IPA + "output_projection", (CS,))
    ex = np.zeros((3, H * DV, H * PV))                              # pull ox/oy/oz out of the head layout
    for h in range(H):
        for d in range(3):
            for i in range(PV):
                ex[d, h * DV + C + d * PV + i, h * PV + i] = 1.0
    k["sel_opt"] = ex
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
    k["quat_rot"], k["quat_n2"] = c, n2
    # R_new[a, c] = sum_b R[a, b] r[b, c] on row-major [N, 9]: R @ PA[b] picks R[a, b] into slot (a, c),
    # r @ PB[b] picks r[b, c] into the same slot.
    PA, PB = np.zeros((3, 9, 9)), np.zeros((3, 9, 9))
    for b in range(3):
        for a in range(3):
            for cc in range(3):
                PA[b, 3 * a + b, 3 * a + cc] = 1.0
                PB[b, 3 * b + cc, 3 * a + cc] = 1.0
    k["compose_a"], k["compose_b"] = PA, PB
    # R's nine columns and t's three, each repeated to width w by one matmul: a broadcast multiply
    # has no reducing backward on the tape, a matmul's backward is the reduction.
    for w in (blk, H * PV):
        k[f"exp9_{w}"] = np.kron(np.eye(9), np.ones((1, w)))
        k[f"exp3_{w}"] = np.kron(np.eye(3), np.ones((1, w)))
    k["one_col"] = ONE
    return k


class NP:
    """float64 numpy backend: the op set the card backend has to supply."""
    def put(self, x): return np.asarray(x, np.float64)
    def get(self, x): return np.asarray(x)
    mm = staticmethod(lambda a, b: a @ b)
    linear = staticmethod(lambda x, w, b=None: x @ w + (0 if b is None else b))
    add = staticmethod(np.add); sub = staticmethod(np.subtract); mul = staticmethod(np.multiply)
    col = staticmethod(lambda x, i: x[..., i:i + 1])
    cols = staticmethod(lambda x, s, e: x[..., s:e])
    concat = staticmethod(lambda xs: np.concatenate(xs, -1))
    heads = staticmethod(lambda x, d: x.reshape(x.shape[0], H, d).transpose(1, 0, 2))   # [N, H*d] -> [H, N, d]
    unheads = staticmethod(lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1))       # [H, N, d] -> [N, H*d]
    bmm = staticmethod(lambda a, b: a @ b)
    bmm_t = staticmethod(lambda a, b: a @ np.swapaxes(b, -1, -2))
    permute = staticmethod(lambda x, perm: x.transpose(perm))
    unsq = staticmethod(lambda x: x[:, None, :])
    flat = staticmethod(lambda x: x.reshape(x.shape[0], -1))
    relu = staticmethod(lambda x: np.maximum(x, 0))
    sqrt_eps = staticmethod(lambda x, eps: np.sqrt(x + eps)); recip = staticmethod(lambda x: 1.0 / x)
    def softmax(self, x):
        e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
    def layer_norm(self, x, gb, eps=1e-5):
        mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * gb[0] + gb[1]


def pair_bias(o, k, z, mask):
    """bias' without the |kp|^2 term: s3 (z @ W_2d + b - 1e5 (1 - m_i m_j)), [H, N, N]. Same in all
    eight layers, so it is computed once per structure-module pass."""
    b = o.linear(z, k["w_2d"], k["b_2d"])                                # [N, N, H]
    return o.add(o.permute(b, (2, 0, 1)), o.put(-1e5 * S3 * (1.0 - np.outer(mask, mask))[None]))


def frame(o, rc, tc, X, Y, Z):
    """R p + t on coordinate blocks [N, w]; rc the nine [N, 1] columns of row-major R, tc t's three."""
    out = []
    for a in range(3):
        g = o.add(o.add(o.mul(X, rc[3 * a]), o.mul(Y, rc[3 * a + 1])), o.mul(Z, rc[3 * a + 2]))
        out.append(o.add(g, tc[a]))
    return out


def unframe(o, rc, tc, X, Y, Z):
    """R^T (p - t)."""
    d = [o.sub(X, tc[0]), o.sub(Y, tc[1]), o.sub(Z, tc[2])]
    return [o.add(o.add(o.mul(d[0], rc[b]), o.mul(d[1], rc[3 + b])), o.mul(d[2], rc[6 + b])) for b in range(3)]


def layer(o, k, act, R, t, bias, z_rows):
    """One FoldIteration. act [N, 384], R [N, 9], t [N, 3], bias [H, N, N] from `pair_bias`,
    z_rows = z as [N, N, 128] (batch = query residue). Returns (act, R, t)."""
    n = act.shape[0]
    pr = o.linear(act, k["w_proj"], k["b_proj"])                         # [N, 1152]
    XB, blk = 3 * H * C, H * NP_
    def wide(x, m, parts, w):
        x = o.mm(x, k[f"{m}_{w}"])
        return [o.cols(x, i * w, (i + 1) * w) for i in range(parts)]
    gX, gY, gZ = frame(o, wide(R, "exp9", 9, blk), wide(t, "exp3", 3, blk),
                       *[o.cols(pr, XB + d * blk, XB + (d + 1) * blk) for d in range(3)])
    one = k["one_col"]
    full = o.concat([o.cols(pr, 0, XB), gX, gY, gZ, o.cols(pr, one, one + 1)])   # global points, constant 1
    s, e = k["kp_cols"]
    kp2 = o.add(o.add(o.mul(o.cols(gX, s, e), o.cols(gX, s, e)), o.mul(o.cols(gY, s, e), o.cols(gY, s, e))),
                o.mul(o.cols(gZ, s, e), o.cols(gZ, s, e)))
    q = o.heads(o.mm(full, k["sel_q"]), DQ)                              # [H, N, 32]
    kk = o.heads(o.add(o.mm(full, k["sel_k"]), o.mm(kp2, k["sel_kp2"])), DQ)
    v = o.heads(o.mm(full, k["sel_v"]), DV)                              # [H, N, 64]
    logits = o.add(o.bmm_t(q, kk), bias)
    attn = o.softmax(logits)                                             # [H, Nq, Nk]
    ov = o.unheads(o.bmm(attn, v))                                       # [N, H*64]
    opx, opy, opz = unframe(o, wide(R, "exp9", 9, H * PV), wide(t, "exp3", 3, H * PV),
                            *[o.mm(ov, k["sel_opt"][d]) for d in range(3)])
    norms = o.sqrt_eps(o.add(o.add(o.mul(opx, opx), o.mul(opy, opy)), o.mul(opz, opz)), 1e-16)
    o_pair = o.bmm(o.permute(attn, (1, 0, 2)), z_rows)                   # [N, H, 128]
    o_pair = o.flat(o_pair)
    ipa = o.add(o.add(o.add(o.add(o.add(o.linear(ov, k["w_out_s"], k["b_out"]), o.mm(opx, k["w_out_pt"][0])),
                                  o.mm(opy, k["w_out_pt"][1])), o.mm(opz, k["w_out_pt"][2])),
                      o.mm(norms, k["w_out_norm"])), o.mm(o_pair, k["w_out_pair"]))
    act = o.layer_norm(o.add(act, ipa), k["attention_layer_norm"])
    h = o.relu(o.linear(act, *k["transition"]))
    h = o.relu(o.linear(h, *k["transition_1"]))
    act = o.layer_norm(o.add(act, o.linear(h, *k["transition_2"])), k["transition_layer_norm"])
    qv = o.linear(act, k["w_quat"], k["b_quat"])                         # [N, 4] = (1, qx, qy, qz)
    mono = o.mul(o.mm(qv, k["mono_a"]), o.mm(qv, k["mono_b"]))           # [N, 10]
    r_u = o.mul(o.mm(mono, k["quat_rot"]), o.recip(o.mm(mono, k["quat_n2"])))
    dt = o.linear(act, k["w_dt"], k["b_dt"])                             # [N, 3] in the old frame
    tX, tY, tZ = frame(o, [o.col(R, i) for i in range(9)], [o.col(t, a) for a in range(3)],
                       o.col(dt, 0), o.col(dt, 1), o.col(dt, 2))
    R_new = o.add(o.add(*[o.mul(o.mm(R, k["compose_a"][b]), o.mm(r_u, k["compose_b"][b])) for b in range(2)]),
                  o.mul(o.mm(R, k["compose_a"][2]), o.mm(r_u, k["compose_b"][2])))
    return act, R_new, o.concat([tX, tY, tZ])


def check(a):
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if "/structure_module/" in k}
    k = pack(p)
    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, CS)); z = rng.standard_normal((n, n, CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0])
    t = rng.standard_normal((n, 3)) * 5.0
    ref = G.layer(act, R, t, G.linear(z, p, G.SM + "invariant_point_attention/attention_2d"), z, mask, p)
    o = NP()
    got = layer(o, k, act, R.reshape(n, 9), t, pair_bias(o, k, z, mask), z)
    got = (got[0], got[1].reshape(n, 3, 3), got[2])
    worst = 0.0
    for name, x, y in zip(("act", "rotation", "translation"), ref, got):
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x)); worst = max(worst, rel)
        print(f"{name:12s} rel L2 vs golden float64: {rel:.2e}")
    print("PASS" if worst < 1e-10 else "FAIL", f"worst {worst:.2e}")
    return worst < 1e-10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a) else 1)
