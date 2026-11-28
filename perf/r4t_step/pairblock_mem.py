"""r4t-step: retained tape bytes and fwd+bwd time of ONE RFD4 PairUpdate block, taped on the production module.

Measures what MEM-PRED assumed (~44 pair-sized node outputs retained per composed block). Synthetic release-shaped
weights (dim 768, pair 384, tri 256), all tokens real, batch 1. Prints one JSON line.

Run: TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:r4t-step python3 perf/r4t_step/pairblock_mem.py --n 256
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))


def aiclk_sampler(card, out, stop):
    path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
    while not stop.is_set():
        try:
            out.append(int(open(path).read().split()[0]))
        except Exception:
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train import lora
    import rfd4_reference as REF

    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])

    def used():
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)

    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    x, z, _, _ = REF.block_inputs(1, a.n, 768, 384, 256, seed=1)
    mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                        TR.compute_kernel_config())

    def put(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    xt, zt = put(x), put(z)
    ag.install()
    params = lora.walked_weights(mod, None, mod, xt, zt, None, None, None)
    leaves = list(params.values()) if hasattr(params, "values") else list(params)
    pair_bytes = a.n * a.n * 384 * 2
    rows, clk, stop = [], [], threading.Event()
    threading.Thread(target=aiclk_sampler, args=(card, clk, stop), daemon=True).start()
    for r in range(a.reps):
        ttnn.synchronize_device(dev)
        base = used()
        t0 = time.time()
        with ag.tape():
            out = mod(xt, zt, None, None, None)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        retained = used() - base
        out_t = out if isinstance(out, ag.Tensor) else ag.Tensor(out)
        seed = ttnn.ones_like(out_t.value)
        ag.backward([out_t], [seed])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        after_bw = used() - base
        grads = [ttnn.to_torch(p.grad).float() for p in leaves if p.grad is not None]
        nz = sum(1 for g in grads if float(g.abs().sum()) > 0)
        gd = hashlib.sha256(b"".join(g.numpy().tobytes() for g in grads)).hexdigest()[:16]
        for p in leaves:
            p.grad = None
        ag.release_pins()
        del out, out_t, seed
        rows.append(dict(rep=r, fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4), retained_B=retained,
                         retained_pair_eq=round(retained / pair_bytes, 2), after_bw_B=after_bw,
                         params=len(leaves), with_grad=len(grads), nonzero_grad=nz, grad_digest=gd))
    stop.set()
    print(json.dumps(dict(n=a.n, card=card, aiclk_min=min(clk or [0]), aiclk_max=max(clk or [0]),
                          aiclk_samples=len(clk), rows=rows)))


if __name__ == "__main__":
    main()
