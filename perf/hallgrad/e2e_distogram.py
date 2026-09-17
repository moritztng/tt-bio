#!/usr/bin/env python3
"""End to end: distogram loss -> gradient with respect to the sequence logits.

The chain, trunk-only, in the shapes Protenix v2 actually uses where they matter:

    logits [N, 21] -> softmax (soft one-hot sequence parametrisation)
                   -> single rep s [N, c_s]
                   -> pair init z[i,j] = lin_i(s_i) + lin_j(s_j)   [N, N, c_z]
                   -> pairformer block: trimul outgoing, trimul incoming,
                      triangle attention starting, triangle attention ending, transition
                   -> distogram head, ONE linear (bins, c_z)       [N, N, bins]
                   -> cross entropy against a target distogram

Two independent verifications, because either alone is weak:

  * The whole-chain logit gradient against a float64 torch reference of the SAME chain on the
    SAME rounded weights. Catches a wrong op, a wrong axis, a dropped term.
  * A DIRECTIONAL finite difference: step the logits along +/- the reported gradient and check
    the loss moves by the predicted 2*eps*||g||. Catches a gradient that is individually
    plausible per op but mis-wired between them, which the per-op gradchecks cannot see.

The loss layer's own gradient is analytic -- the seed handed to backward() is exactly
(softmax(d) - target)/M -- so neither check is measuring a cross-entropy approximation. The
loss VALUE is evaluated in float64 on host from the device's distogram logits, which keeps the
finite difference from being swamped by a bf16 reduction of a scalar.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

VOCAB = 21


def clocks_thread(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            for dev in json.loads(r.stdout).get("device_info", []):
                c = dev.get("telemetry", {}).get("aiclk")
                if c is not None:
                    out.append(int(c))
        except Exception:
            pass
        time.sleep(1.0)


def make_weights(rng, c_s, c_z, heads, head_dim, hidden, bins):
    """Every weight in (in, out) layout, which is what ttnn.linear and tt-bio both want."""
    def w(i, o, s=1.0):
        return rng.standard_normal((i, o)) * (s / np.sqrt(i))
    d = {
        "emb_s": w(VOCAB, c_s),
        "pair_i": w(c_s, c_z), "pair_j": w(c_s, c_z),
    }
    for tag in ("out", "in"):
        d[f"tm_{tag}_a"] = w(c_z, hidden)
        d[f"tm_{tag}_ag"] = w(c_z, hidden)
        d[f"tm_{tag}_b"] = w(c_z, hidden)
        d[f"tm_{tag}_bg"] = w(c_z, hidden)
        d[f"tm_{tag}_g"] = w(c_z, c_z)
        d[f"tm_{tag}_z"] = w(hidden, c_z)
        d[f"tm_{tag}_ln_g"] = np.ones((1, c_z))
        d[f"tm_{tag}_ln2_g"] = np.ones((1, hidden))
    for tag in ("start", "end"):
        d[f"ta_{tag}_q"] = w(c_z, heads * head_dim)
        d[f"ta_{tag}_k"] = w(c_z, heads * head_dim)
        d[f"ta_{tag}_v"] = w(c_z, heads * head_dim)
        d[f"ta_{tag}_g"] = w(c_z, heads * head_dim)
        d[f"ta_{tag}_b"] = w(c_z, heads)
        d[f"ta_{tag}_o"] = w(heads * head_dim, c_z)
        d[f"ta_{tag}_ln_g"] = np.ones((1, c_z))
    d["tr_ln_g"] = np.ones((1, c_z))
    d["tr_a"] = w(c_z, 4 * c_z)
    d["tr_b"] = w(c_z, 4 * c_z)
    d["tr_o"] = w(4 * c_z, c_z)
    d["dist_ln_g"] = np.ones((1, c_z))
    d["dist"] = w(c_z, bins)
    return d


def torch_chain(logits, W, cfg):
    """float64 reference for the whole chain. Mirrors tt_chain op for op."""
    heads, head_dim, hidden = cfg["heads"], cfg["head_dim"], cfg["hidden"]
    n = logits.shape[0]
    scale = head_dim ** -0.5

    def ln(x, g):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        return xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + 1e-6) * g

    p = torch.softmax(logits, dim=-1)
    s = p @ W["emb_s"]
    z = (s @ W["pair_i"]).unsqueeze(1) + (s @ W["pair_j"]).unsqueeze(0)

    for tag, incoming in (("out", False), ("in", True)):
        zn = ln(z, W[f"tm_{tag}_ln_g"])
        a = torch.sigmoid(zn @ W[f"tm_{tag}_ag"]) * (zn @ W[f"tm_{tag}_a"])
        b = torch.sigmoid(zn @ W[f"tm_{tag}_bg"]) * (zn @ W[f"tm_{tag}_b"])
        x = (torch.einsum("kic,kjc->ijc", a, b) if incoming
             else torch.einsum("ikc,jkc->ijc", a, b))
        x = ln(x, W[f"tm_{tag}_ln2_g"])
        gate = torch.sigmoid(zn @ W[f"tm_{tag}_g"])
        z = z + gate * (x @ W[f"tm_{tag}_z"])

    for tag in ("start", "end"):
        zin = z.permute(1, 0, 2) if tag == "end" else z
        zn = ln(zin, W[f"ta_{tag}_ln_g"])
        def heads_of(key):
            return (zn @ W[f"ta_{tag}_{key}"]).reshape(n, n, heads, head_dim).permute(0, 2, 1, 3)
        q, k, v, g = (heads_of(x) for x in ("q", "k", "v", "g"))
        bias = (zn @ W[f"ta_{tag}_b"]).permute(2, 0, 1).unsqueeze(0)
        o = torch.softmax(q @ k.transpose(-2, -1) * scale + bias, dim=-1) @ v
        o = o * torch.sigmoid(g)
        o = o.permute(0, 2, 1, 3).reshape(n, n, heads * head_dim)
        upd = o @ W[f"ta_{tag}_o"]
        z = z + (upd.permute(1, 0, 2) if tag == "end" else upd)

    zn = ln(z, W["tr_ln_g"])
    swi = torch.nn.functional.silu(zn @ W["tr_a"]) * (zn @ W["tr_b"])
    z = z + swi @ W["tr_o"]
    return ln(z, W["dist_ln_g"]) @ W["dist"]


def tt_chain(ag, ttnn, logits, W, cfg):
    """The same chain on device, on the tape. Every op here is gradchecked individually."""
    heads, head_dim, hidden = cfg["heads"], cfg["head_dim"], cfg["hidden"]
    n = int(logits.value.shape[0])
    scale = head_dim ** -0.5

    p = ag.softmax(logits, dim=-1)
    s = ag.linear(p, W["emb_s"])
    zi, zj = ag.linear(s, W["pair_i"]), ag.linear(s, W["pair_j"])
    c_z = int(zi.value.shape[-1])
    z = ag.add(_expand(ag, ttnn, zi, n, rows=True), _expand(ag, ttnn, zj, n, rows=False))

    if cfg.get("checkpoint"):
        # One checkpoint per pairformer block, which is the granularity the study called for.
        z = ag.checkpoint(lambda zz: _block(ag, zz, W, cfg, n, scale), z)
    else:
        z = _block(ag, z, W, cfg, n, scale)
    zn = ag.layer_norm(z, W["tr_ln_g"])
    ta = ag.linear(zn, W["tr_a"])
    swi = ag.mul(ag.mul(ta, ag.sigmoid(ta)), ag.linear(zn, W["tr_b"]))
    z = ag.add(z, ag.linear(swi, W["tr_o"]))
    return ag.linear(ag.layer_norm(z, W["dist_ln_g"]), W["dist"])


def _block(ag, z, W, cfg, n, scale):
    """trimul outgoing, trimul incoming, tri-attention starting, tri-attention ending."""
    heads, head_dim = cfg["heads"], cfg["head_dim"]
    for tag, incoming in (("out", False), ("in", True)):
        zn = ag.layer_norm(z, W[f"tm_{tag}_ln_g"])
        a = ag.mul(ag.sigmoid(ag.linear(zn, W[f"tm_{tag}_ag"])), ag.linear(zn, W[f"tm_{tag}_a"]))
        b = ag.mul(ag.sigmoid(ag.linear(zn, W[f"tm_{tag}_bg"])), ag.linear(zn, W[f"tm_{tag}_b"]))
        x = ag.layer_norm(ag.pair_contract(a, b, incoming=incoming), W[f"tm_{tag}_ln2_g"])
        gate = ag.sigmoid(ag.linear(zn, W[f"tm_{tag}_g"]))
        z = ag.add(z, ag.mul(gate, ag.linear(x, W[f"tm_{tag}_z"])))

    for tag in ("start", "end"):
        zin = ag.permute(z, (1, 0, 2)) if tag == "end" else z
        zn = ag.layer_norm(zin, W[f"ta_{tag}_ln_g"])

        def heads_of(key):
            h = ag.linear(zn, W[f"ta_{tag}_{key}"])
            return ag.permute(ag.reshape(h, [n, n, heads, head_dim]), (0, 2, 1, 3))
        q, k, v, g = (heads_of(x) for x in ("q", "k", "v", "g"))
        bias = ag.reshape(ag.permute(ag.linear(zn, W[f"ta_{tag}_b"]), (2, 0, 1)),
                          [1, heads, n, n])
        o = ag.triangle_attention(q, k, v, bias, scale=scale,
                                  chunk=cfg["chunk"], q_chunk=cfg["chunk"])
        o = ag.mul(o, ag.sigmoid(g))
        o = ag.reshape(ag.permute(o, (0, 2, 1, 3)), [n, n, heads * head_dim])
        upd = ag.linear(o, W[f"ta_{tag}_o"])
        z = ag.add(z, ag.permute(upd, (1, 0, 2)) if tag == "end" else upd)
    return z


def _expand(ag, ttnn, x, n, rows):
    """[N, c] -> [N, N, c], broadcasting along rows or columns. The pair init's outer sum.

    Done with reshape + a taped repeat rather than relying on ttnn eltwise broadcast, so the
    backward is an explicit reduction over the axis that was expanded.
    """
    nn_, c = int(x.value.shape[0]), int(x.value.shape[-1])
    shaped = ag.reshape(x, [nn_, 1, c] if rows else [1, nn_, c])
    reps = [1, n, 1] if rows else [n, 1, 1]
    out_v = ttnn.repeat(shaped.value, reps)
    axis = 1 if rows else 0

    def make(out):
        def bw():
            g = ttnn.sum(out.grad, dim=axis, keepdim=True)
            shaped.add_grad(g)
        return bw
    return ag._tape(out_v, [shaped], make)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--c-s", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--checkpoint", action="store_true",
                    help="recompute the pairformer block inside its own backward")
    ap.add_argument("--skip-reference", action="store_true",
                    help="skip the float64 chain (it is O(N^3) on CPU); keep the directional check")
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop, clocks = threading.Event(), []
    threading.Thread(target=clocks_thread, args=(stop, clocks), daemon=True).start()

    cfg = dict(heads=args.heads, head_dim=args.head_dim, hidden=args.c_z,
               chunk=args.chunk, checkpoint=args.checkpoint)
    rng = np.random.default_rng(args.seed)
    n, bins = args.n, args.bins
    Wnp = make_weights(rng, args.c_s, args.c_z, args.heads, args.head_dim, args.c_z, bins)
    logits_np = rng.standard_normal((n, VOCAB))
    target_np = rng.random((n, n, bins))
    target_np /= target_np.sum(-1, keepdims=True)

    # round to bf16 once; both the device and the reference see these exact values
    def bf(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)
    W64 = {k: bf(v).to(torch.float64) for k, v in Wnp.items()}
    logits64 = bf(logits_np).to(torch.float64)
    target64 = bf(target_np).to(torch.float64)
    M = n * n

    def loss_of(dist_logits64):
        lp = torch.log_softmax(dist_logits64, dim=-1)
        return float(-(target64 * lp).sum() / M)

    Wtt = {k: ag.Tensor(ttnn.from_torch(bf(v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=device)) for k, v in Wnp.items()}

    def device_forward(lg64, want_grad):
        lt = ag.Tensor(ttnn.from_torch(lg64.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device),
                       requires_grad=want_grad)
        d = tt_chain(ag, ttnn, lt, Wtt, cfg)
        d64 = ttnn.to_torch(d.value).to(torch.float64)
        return lt, d, d64

    t0 = time.perf_counter()
    lt, d, d64 = device_forward(logits64, True)
    ttnn.synchronize_device(device)
    t_fwd = time.perf_counter() - t0
    base_loss = loss_of(d64)

    # the loss layer's gradient, analytic: d(CE)/d(logits_d) = (softmax - target)/M
    seed64 = (torch.softmax(d64, dim=-1) - target64) / M
    t0 = time.perf_counter()
    d.backward(seed=ttnn.from_torch(seed64.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device))
    ttnn.synchronize_device(device)
    t_bwd = time.perf_counter() - t0
    if lt.grad is None:
        print("E2E FAIL: no gradient reached the sequence logits")
        return 1
    g_dev = ttnn.to_torch(lt.grad).to(torch.float64).numpy()

    print(f"# N={n} c_z={args.c_z} heads={args.heads} head_dim={args.head_dim} bins={bins} "
          f"chunk={args.chunk} checkpoint={args.checkpoint}")
    print(f"# one pairformer block: trimul out, trimul in, tri-attention start, "
          f"tri-attention end, transition")
    print(f"# loss {base_loss:.6f}   forward {t_fwd:.3f} s   backward {t_bwd:.3f} s   "
          f"bwd/fwd {t_bwd / t_fwd:.2f}")
    print(f"# |grad| = {np.linalg.norm(g_dev):.6e} over {g_dev.size} logits")

    ok = True
    if not args.skip_reference:
        lr = logits64.clone().requires_grad_(True)
        dl = torch_chain(lr, W64, cfg)
        (-(target64 * torch.log_softmax(dl, dim=-1)).sum() / M).backward()
        g_ref = lr.grad.detach().numpy()
        fwd_rel = float(np.linalg.norm(d64.numpy() - dl.detach().numpy())
                        / np.linalg.norm(dl.detach().numpy()))
        gd, gr = g_dev.ravel(), g_ref.ravel()
        rel = float(np.linalg.norm(gd - gr) / np.linalg.norm(gr))
        cos = float(gd @ gr / (np.linalg.norm(gd) * np.linalg.norm(gr)))
        print(f"\n# whole-chain FORWARD vs float64: rel_l2 {fwd_rel:.2e}")
        print(f"# whole-chain LOGIT GRADIENT vs float64: rel_l2 {rel:.2e}, cos {cos:.6f}")
        good = rel <= 5.0e-2 and cos >= 0.999
        print(f"# {'PASS' if good else 'FAIL'} (bar rel_l2 <= 5.0e-2, cos >= 0.999; looser than "
              f"the per-op 1.0e-2 because this is ~40 bf16 ops deep)")
        ok = ok and good

    # ---- directional finite difference on the reported gradient -------------
    gn = np.linalg.norm(g_dev)
    u = torch.from_numpy(g_dev / gn).to(torch.float64)
    print(f"\n# directional finite difference along the reported gradient.")
    print(f"# 'nominal' predicts 2*eps*|g|, which assumes the whole step is applied. 'applied'")
    print(f"# predicts g . (bf16(l+eps*u) - bf16(l-eps*u)), the step the device actually took:")
    print(f"# the logits are bf16 on device, so a step below bf16 spacing is silently rounded")
    print(f"# away. 'kept' is the fraction of the intended step that survived rounding.")
    print(f"{'eps':>8} {'kept':>7} {'measured dL':>13} {'nominal':>12} {'r_nom':>7} "
          f"{'applied':>12} {'r_app':>7}  verdict")
    eps_list = [args.eps] if args.eps else [1e-2, 3e-2, 1e-1, 3e-1]
    g_flat = torch.from_numpy(g_dev.ravel()).to(torch.float64)
    resolved = 0
    l_rounded = logits64.to(torch.bfloat16).to(torch.float64)
    for eps in eps_list:
        lp64, lm64 = logits64 + eps * u, logits64 - eps * u
        # what the device really saw, after its own bf16 rounding of the design variable
        step = ((lp64.to(torch.bfloat16).to(torch.float64)
                 - lm64.to(torch.bfloat16).to(torch.float64)).reshape(-1))
        kept = float(torch.linalg.vector_norm(step) / (2.0 * eps))
        _, _, dp = device_forward(lp64, False)
        _, _, dm = device_forward(lm64, False)
        meas = loss_of(dp) - loss_of(dm)
        nominal = 2.0 * eps * gn      # u = g/|g|, so the directional derivative is exactly |g|
        applied = float(g_flat @ step)
        r_nom = meas / nominal
        r_app = meas / applied if applied != 0 else float("nan")
        # Only eps values whose step actually reaches the device can test a gradient. `kept`
        # is measured from the rounding alone, independently of the loss, so gating on it is
        # not a moved goalpost -- below the floor the step's DIRECTION is destroyed too, which
        # is why correcting the magnitude via `applied` does not rescue eps=0.01.
        resolvable = kept >= 0.95
        good = 0.9 <= r_app <= 1.1
        verdict = ("PASS" if good else "FAIL") if resolvable else "below bf16 step floor"
        print(f"{eps:>8.3g} {kept:>7.3f} {meas:>13.5e} {nominal:>12.5e} {r_nom:>7.4f} "
              f"{applied:>12.5e} {r_app:>7.4f}  {verdict}")
        if resolvable:
            ok = ok and good
            resolved += 1
    stop.set()
    time.sleep(0.1)
    if clocks:
        print(f"\n# AICLK during: min {min(clocks)} max {max(clocks)} "
              f"median {int(statistics.median(clocks))} MHz over {len(clocks)} samples")
    else:
        print("\n# AICLK: NO SAMPLES -- timings unusable")
    if resolved == 0:
        ok = False
        print("\n# no eps cleared the bf16 step floor -- the directional check did not run")
    print(f"\nE2E {'PASS' if ok else 'FAIL'}: the distogram gradient "
          f"{'reaches the sequence logits and moves the loss as predicted' if ok else 'did not clear a check'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
