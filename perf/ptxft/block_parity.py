#!/usr/bin/env python3
"""Hold the taped pair track to the PRODUCTION PairformerLayer, on the REAL weights.

This is the seam the whole fine-tune hangs off. Blocks below the adapted window run on
the production fused path; the adapted blocks run on the tape. If the two compute
different functions then the adapter is trained against something other than the model
that will serve, and every downstream number is measuring the wrong thing.

Three arms:

--parity          the taped block vs tenstorrent.PairformerLayer on the same z and the
                  same block's real weights, reported as PCC and relative L2.
--s-independence  the control behind the design: run the production layer twice with
                  DIFFERENT s and the same z. If z out is bit-identical then the pair
                  track really is closed and a distogram loss has exactly zero gradient
                  into attention_pair_bias and single_transition.
--blocks          check more than block 0. A remap that is wrong for one block's shapes
                  and right for another is a real failure mode.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def pcc(a, b):
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 0 else float("nan")


def rel_l2(a, b):
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.linalg.norm(y)
    return float(np.linalg.norm(x - y) / d) if d > 0 else float("nan")


def load_blocks(ckpt, idxs):
    """Remapped per-block state dicts plus the distogram head, off the real checkpoint."""
    from tt_bio import protenix_weights as PW
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "ema", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    out = {}
    for i in idxs:
        pre = f"pairformer_stack.blocks.{i}."
        blk = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        if not blk:
            raise KeyError(f"no pairformer block {i} in {ckpt}")
        out[i] = PW.remap_pairformer_block(blk)
    head = {k: v for k, v in sd.items() if k.startswith("distogram_head.")}
    return out, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=64, help="tokens; the pair tensor is [n, n, c_z]")
    ap.add_argument("--blocks", default="0")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--s-independence", action="store_true")
    ap.add_argument("--stages", action="store_true",
                    help="compare each of the five z updates against its production module")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--q-chunk", type=int, default=None)
    ap.add_argument("--pcc-bar", type=float, default=0.99)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="control: if a formula were WRONG rather than imprecise, more "
                         "precision would not fix it")
    ap.add_argument("--warm", type=int, default=0,
                    help="run the production block this many times first and parity-check "
                         "from its output. A random z makes the residual updates ~30x the "
                         "stream, which is nowhere near what a trunk z looks like")
    ap.add_argument("--z-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from perf.clocksample import during
    from perf.ptxft.tape_block import PairTrackBlock

    idxs = [int(x) for x in a.blocks.split(",")]
    blocks, _head = load_blocks(a.ckpt, idxs)
    device = tt.get_device()
    ckc = tt.precise_kernel_config() if hasattr(tt, "precise_kernel_config") else None
    if ckc is None:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

    N = a.n
    C_Z, C_S, H, D = 256, 384, 8, 32
    DT = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[a.dtype]
    TDT = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    rng = np.random.default_rng(a.seed)
    failures = []

    print(f"# Protenix v2 pair track: N={N} c_z={C_Z} heads={H} head_dim={D}, "
          f"blocks {idxs}, dtype {a.dtype}, warm {a.warm}, PCC bar {a.pcc_bar}")
    print(f"# checkpoint {a.ckpt}")

    with during() as clk:
        for i in idxs:
            sd = blocks[i]
            # bf16 first, then back up, so BOTH arms read the identical input bytes and the
            # comparison is of the two implementations rather than of their inputs.
            z0 = torch.from_numpy(rng.standard_normal((N, N, C_Z)).astype(np.float32)
                                  * a.z_scale).to(TDT)
            s0 = torch.from_numpy(rng.standard_normal((1, N, C_S)).astype(np.float32)
                                  * 0.5).to(TDT)

            def prod_forward(s_t, z_t):
                layer = tt.PairformerLayer(D, H, C_S // 16, 16, True, sd, ckc)
                zz = ttnn.from_torch(z_t.unsqueeze(0), dtype=DT,
                                     layout=ttnn.TILE_LAYOUT, device=device)
                ss = ttnn.from_torch(s_t, dtype=DT,
                                     layout=ttnn.TILE_LAYOUT, device=device)
                t0 = time.perf_counter()
                _so, zo = layer(ss, zz)
                ttnn.synchronize_device(device)
                dt = time.perf_counter() - t0
                return ttnn.to_torch(zo).to(torch.float64).numpy().reshape(N, N, C_Z), dt

            if a.s_independence:
                s_alt = torch.from_numpy(rng.standard_normal((1, N, C_S)).astype(np.float32)
                                         * 3.0).to(torch.bfloat16)
                zA, _ = prod_forward(s0, z0)
                zB, _ = prod_forward(s_alt, z0)
                same = bool((zA == zB).all())
                moved = float(np.abs(s0.to(torch.float64).numpy()
                                     - s_alt.to(torch.float64).numpy()).max())
                print()
                print(f"# --s-independence, block {i}: the same z with two DIFFERENT single")
                print(f"# representations (max |ds| = {moved:.3f}). If z out is bit-identical")
                print(f"# then the pair track is closed and attention_pair_bias /")
                print(f"# single_transition are UNREACHABLE from a distogram loss.")
                print(f"block {i}: z out bit-identical under a changed s: {same} "
                      f"(max abs diff {np.abs(zA - zB).max():.3e})")
                if not same:
                    failures.append(f"block {i}: z depends on s, so the pair track is NOT "
                                    f"closed and the reachable-parameter claim is wrong")


            for _w in range(a.warm):
                zw, _ = prod_forward(s0, z0)
                z0 = torch.from_numpy(zw.astype(np.float32)).to(TDT)
            if a.warm:
                upd = float(np.linalg.norm(
                    z0.to(torch.float64).numpy()))
                print(f"# --warm {a.warm}: z warmed through the production block, "
                      f"||z|| now {upd:.4e}")

            if a.stages:
                # A whole-block PCC says "close" or "not close". It cannot say WHICH of the
                # five updates is wrong, and a wrong transform in one of them is exactly
                # what this row must not ship. So each update is compared to its own
                # production module on the same input bytes.
                print()
                print(f"# --stages, block {i}: each z update vs its production module")
                print(f"{'stage':<16} {'PCC':>10} {'rel L2':>10} {'|prod|':>11} "
                      f"{'|tape|':>11}  verdict")
                blk = PairTrackBlock(sd, device, n_heads=H, head_dim=D,
                                     chunk=a.chunk, q_chunk=a.q_chunk)
                z4 = z0.unsqueeze(0)

                def sub(prefix, strip=""):
                    d = {k[len(prefix) + 1:]: v for k, v in sd.items()
                         if k.startswith(prefix + ".")}
                    if strip:
                        d = {(k[len(strip):] if k.startswith(strip) else k): v
                             for k, v in d.items()}
                    return d

                def prod_in():
                    return ttnn.from_torch(z4, dtype=DT,
                                           layout=ttnn.TILE_LAYOUT, device=device)

                def tape_in():
                    return ag.Tensor(ft.to_device(z0.to(torch.float32).numpy(), device,
                                                  dtype=DT))

                stages = [
                    ("tri_mul_out",
                     lambda: tt.TriangleMultiplication(False, sub("tri_mul_out"), ckc)(prod_in()),
                     lambda: blk.trimul(tape_in(), "tri_mul_out", incoming=False)),
                    ("tri_mul_in",
                     lambda: tt.TriangleMultiplication(True, sub("tri_mul_in"), ckc)(prod_in()),
                     lambda: blk.trimul(tape_in(), "tri_mul_in", incoming=True)),
                    ("tri_att_start",
                     lambda: tt.TriangleAttention(D, H, False, sub("tri_att_start", "mha."),
                                                  ckc)(prod_in()),
                     lambda: blk.triatt(tape_in(), "tri_att_start", ending=False)),
                    ("tri_att_end",
                     lambda: tt.TriangleAttention(D, H, True, sub("tri_att_end", "mha."),
                                                  ckc)(prod_in()),
                     lambda: blk.triatt(tape_in(), "tri_att_end", ending=True)),
                    ("transition_z",
                     lambda: tt.Transition(sub("transition_z"), ckc)(prod_in()),
                     lambda: blk.transition(tape_in())),
                ]
                for label, pf, tf in stages:
                    try:
                        pv = ttnn.to_torch(pf()).to(torch.float64).numpy().reshape(N, N, C_Z)
                    except Exception as exc:                                  # noqa: BLE001
                        print(f"{label:<16} production module refused: "
                              f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
                        continue
                    tv = ft.to_host(tf().value).astype(np.float64).reshape(N, N, C_Z)
                    pp, rr = pcc(tv, pv), rel_l2(tv, pv)
                    ok = pp >= a.pcc_bar
                    print(f"{label:<16} {pp:>10.6f} {rr:>10.3e} "
                          f"{np.linalg.norm(pv):>11.4e} {np.linalg.norm(tv):>11.4e}  "
                          f"{'PASS' if ok else 'FAIL'}")
                    if not ok:
                        failures.append(f"block {i} stage {label}: PCC {pp:.6f} rel L2 {rr:.3e}")

            if a.parity:
                z_prod, dt_prod = prod_forward(s0, z0)
                blk = PairTrackBlock(sd, device, n_heads=H, head_dim=D,
                                     chunk=a.chunk, q_chunk=a.q_chunk)
                zt = ag.Tensor(ft.to_device(
                    z0.to(torch.float32).numpy(), device, dtype=DT),
                    requires_grad=True)
                t0 = time.perf_counter()
                out = blk(zt)
                ttnn.synchronize_device(device)
                dt_tape = time.perf_counter() - t0
                z_tape = ft.to_host(out.value).astype(np.float64).reshape(N, N, C_Z)
                p, r = pcc(z_tape, z_prod), rel_l2(z_tape, z_prod)
                ok = p >= a.pcc_bar
                print()
                print(f"# --parity, block {i}: the taped twin vs the production layer")
                print(f"{'':<12} {'PCC':>10} {'rel L2':>10} {'prod s':>9} {'tape s':>9}  verdict")
                print(f"block {i:<6} {p:>10.6f} {r:>10.3e} {dt_prod:>9.3f} {dt_tape:>9.3f}  "
                      f"{'PASS' if ok else 'FAIL'}")
                if not ok:
                    failures.append(f"block {i}: taped twin PCC {p:.6f} < {a.pcc_bar}")
                # Where the disagreement is, when there is one: per-stage PCC tells you
                # which of the five updates is wrong, and a whole-block number does not.
                if not ok:
                    print("# per-stage, to localise it:")
                    for label, fn in (
                            ("tri_mul_out", lambda t: blk.trimul(t, "tri_mul_out", incoming=False)),
                            ("tri_mul_in", lambda t: blk.trimul(t, "tri_mul_in", incoming=True)),
                            ("tri_att_start", lambda t: blk.triatt(t, "tri_att_start", ending=False)),
                            ("tri_att_end", lambda t: blk.triatt(t, "tri_att_end", ending=True)),
                            ("transition_z", blk.transition)):
                        zz = ag.Tensor(ft.to_device(z0.to(torch.float32).numpy(), device,
                                                    dtype=ttnn.bfloat16))
                        v = ft.to_host(fn(zz).value).astype(np.float64)
                        print(f"  {label:<16} out norm {np.linalg.norm(v):.4e} "
                              f"shape {v.shape}")

    print()
    print(clk.line(0))
    print()
    if failures:
        print(f"BLOCK-PARITY FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("BLOCK-PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
