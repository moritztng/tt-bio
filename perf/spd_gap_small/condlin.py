"""Device time and float64 error of the token DiT's per-layer conditioning linears (dit_cond_jit).

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/condlin.py --out condlin.jsonl

Boltz-2 --fast on Wormhole projects six [512, 768] x [768, 768] terms a layer a denoise step. cen7 (WH, 1000 MHz):
the four with bias + sigmoid read 42 us a call, the two without either read 28.6 us, so the epilogue costs ~13 us on
19200 calls a fold (~0.26 s). The arms split bias from sigmoid and try `kfold`: the bias as one more K row of the
weight, the activation carrying a ones column, so the matmul has no bias epilogue at all (K 768 -> 800). N 3072 is
the token attention's qkv projection (bias on q), 107 us a call on WH in cen7, same question. N 36864 is the cond
hoist under dit_shoist. `*_sep` arms run the sigmoid as a separate ttnn.sigmoid: on Blackhole (condbh, p300c 1350 MHz)
the bias costs ~1 us and the fused sigmoid 7-21 us, so the question there is the epilogue, not the bias.

Device time is a captured trace replayed back to back (op_probe.dev_time), median of 7 batches, AICLK of the opened
node. Error is against float64 with op_probe's wrong-pixel rule.
"""
import argparse, json, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import aiclk, dev_time, opened_nodes, wrong  # noqa: E402

M, K = 512, 768


def bench(dev, nodes, emit, N, seed=7, pl1=True):
    emit(ev="start", m=M, k=K, n=N, seed=seed, packer_l1_acc=pl1)
    torch.manual_seed(seed)
    A = torch.randn(1, M, K); W = torch.randn(K, N) / K ** 0.5; Bv = torch.randn(N) * 0.5
    R = A.double() @ W.double()
    refs = {"plain": R, "bias": R + Bv.double()}
    refs["sig"] = torch.sigmoid(R); refs["bias_sig"] = torch.sigmoid(refs["bias"])
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    ta, tw, tb = up(A), up(W), up(Bv.reshape(1, N))
    # kfold: [A | 1 | 0 x 31] x [W ; b ; 0 x 31], one extra K tile
    Ak = torch.cat([A, torch.ones(1, M, 1), torch.zeros(1, M, 31)], -1)
    Wk = torch.cat([W, Bv.reshape(1, N), torch.zeros(31, N)], 0)
    tak, twk = up(Ak), up(Wk)
    for acc in (False, True):
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=pl1)
        arms = [
            ("plain", "plain", lambda: ttnn.linear(ta, tw, compute_kernel_config=ckc)),
            ("bias", "bias", lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc)),
            ("sig", "sig", lambda: ttnn.linear(ta, tw, compute_kernel_config=ckc, activation="sigmoid")),
            ("bias_sig", "bias_sig", lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc,
                                                          activation="sigmoid")),
            ("kfold", "bias", lambda: ttnn.linear(tak, twk, compute_kernel_config=ckc)),
            ("kfold_sig", "bias_sig", lambda: ttnn.linear(tak, twk, compute_kernel_config=ckc,
                                                           activation="sigmoid")),
            # as the token DiT calls it (DitLinear.linear): the model's core grid, bias on, fp32 output
            ("bias_cg", "bias", lambda: ttnn.linear(ta, tw, bias=tb, compute_kernel_config=ckc,
                                                    core_grid=T.CORE_GRID_MAIN, dtype=ttnn.float32)),
            # the sigmoid as its own eltwise op over all cores instead of the matmul's epilogue
            ("sig_sep", "sig", lambda: ttnn.sigmoid(ttnn.linear(ta, tw, compute_kernel_config=ckc))),
            ("bias_sig_sep", "bias_sig", lambda: ttnn.sigmoid(ttnn.linear(ta, tw, bias=tb,
                                                                          compute_kernel_config=ckc))),
        ]
        for lab, ref, fn in arms:
            try:
                y = fn()
            except Exception as e:
                emit(ev="arm", n=N, acc=acc, arm=lab, error=str(e).splitlines()[0][:240])
                continue
            Y = ttnn.to_torch(y).double().reshape(R.shape); ttnn.deallocate(y)
            nb, mx, rel = wrong(refs[ref], Y)
            at = []
            if nb:  # where the wrong pixels sit: (row, col, f64, device), to tell a bad core from a bad program
                err = (Y - refs[ref]).abs().reshape(M, N)
                for i in err.flatten().topk(min(nb, 4)).indices.tolist():
                    r, c = divmod(i, N)
                    at.append([r, c, round(refs[ref].reshape(M, N)[r, c].item(), 4), round(Y.reshape(M, N)[r, c].item(), 4)])
            us, sp, ck = dev_time(dev, fn, nodes=nodes)
            emit(ev="arm", n=N, acc=acc, arm=lab, us=round(us, 1), spread=round(sp, 3), wrong=nb, at=at,
                 max_err=round(mx, 5), rel_rms=float(f"{rel:.3e}"), aiclk=ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    # 36864 = Boltz-2's hoisted DiT cond projection under dit_shoist (BH: 755 us with bias + sigmoid, 436 us without)
    ap.add_argument("--n", type=int, nargs="+", default=[768, 3072, 36864])
    # condwp (WH): one wrong pixel at (289, 2183) on four chips under fp32 acc at N 3072. A different seed tells an
    # input-dependent rounding event from a fixed position the program gets wrong; packer_l1_acc off isolates the packer.
    # condseed: seeds 8, 9 clean and seed 7 clean with the packer off, so it is input-dependent; many seeds give a rate per arm
    ap.add_argument("--seed", type=int, nargs="+", default=[7])
    ap.add_argument("--packer-l1-acc", type=int, choices=(0, 1), default=1)
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = T.get_device(trace="protenix")
    nodes = opened_nodes()
    log = a.out.open("a")

    def emit(**kw):
        kw.setdefault("aiclk", aiclk(nodes)); kw.update(arch=T.arch_name(), t=time.time())
        log.write(json.dumps(kw) + "\n"); log.flush(); print(json.dumps(kw), flush=True)

    for seed in a.seed:
        for N in a.n:
            bench(dev, nodes, emit, N, seed, bool(a.packer_l1_acc))
    emit(ev="end")


if __name__ == "__main__":
    main()
