"""r4t-step: N real optimizer steps on RFD4's pair stack (K production PairUpdate blocks), taped, on one chip.

The pair track is 91-98 % of RFD4's FLOPs and the part of the trunk whose backward exists today (the 14 token
layers wait on the rms_norm VJP). Synthetic release-shaped weights, loss = <out, C> / numel with a fixed C, fresh
pair input per step (seeded by step). AdamW (tt_bio.train.optim), the G2 gate (parity.GradientCoverage (wk/r4t-accuracy)), the
checkpointer (tt_bio.train.checkpoint). Prints one JSON line per step and a summary.

  --save-at k     write a checkpoint after step k
  --resume PATH   load it into a fresh process and continue from its step: the next step must match bit for bit
  --detach-at j   cut the graph in front of block j: blocks < j get no gradient and the gate must refuse the run
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--save-at", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--detach-at", type=int, default=-1)
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/ckpt"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import numpy as np
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train import lora
    from tt_bio.train.parity import GradientCoverage, zero_or_missing
    from tt_bio.train.checkpoint import Checkpointer, load_adapter
    from tt_bio.train.optim import AdamW
    import rfd4_reference as REF

    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    clk, stop = [], threading.Event()

    def sample():
        path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
        while not stop.is_set():
            try:
                clk.append(int(open(path).read().split()[0]))
            except Exception:
                pass
            time.sleep(0.05)

    def used():
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)

    def put(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    class Stack:
        def __init__(self):
            self.blocks = []
            for k in range(a.blocks):
                w = REF.pair_update_weights(768, 384, 256, seed=100 + k, prefix="pu")
                self.blocks.append(TR.PairUpdate(T.WeightScope({n: v.to(torch.float32) for n, v in w.items()})
                                                 .child("pu"), TR.compute_kernel_config()))

        def __call__(self, x, z):
            for k, b in enumerate(self.blocks):
                if k == a.detach_at:
                    z = ttnn.clone(z.value if isinstance(z, ag.Tensor) else z)
                z = b(x, z, None, None, None)
            return z

    model = Stack()
    x0, z0, _, _ = REF.block_inputs(1, a.n, 768, 384, 256, seed=1)
    xt = put(x0)
    g = torch.Generator().manual_seed(7)
    C = torch.randn(1, a.n, a.n, 384, generator=g)
    numel = C.numel()
    seed_dev = put(C / numel)
    ag.install()
    params = lora.walked_weights(model, None, model, xt, put(z0))
    opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0)
    start = 0
    if a.resume:
        meta = load_adapter(a.resume, dict(params), dev, opt=opt)
        params.rebind()
        start = int(meta["step"])
    ck = Checkpointer(a.out, every=10**9, keep=10)
    cov = GradientCoverage(list(params), where="r4t-step pair stack: ")
    threading.Thread(target=sample, daemon=True).start()
    for step in range(start, a.steps):
        zin = put(REF.block_inputs(1, a.n, 768, 384, 256, seed=1000 + step)[1])
        ttnn.synchronize_device(dev)
        base = used()
        c0 = len(clk)
        t0 = time.time()
        with ag.tape():
            out = model(xt, zin)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        retained = used() - base
        out_t = out if isinstance(out, ag.Tensor) else ag.Tensor(out)
        loss = float((ttnn.to_torch(out_t.value).float() * C).sum() / numel)
        ag.backward([out_t], [seed_dev])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        peak_dev = used() - base
        grads = {n: (None if p.grad is None else ttnn.to_torch(p.grad).float().numpy()) for n, p in params.items()}
        dead = zero_or_missing(grads)
        opt.step()
        params.rebind()   # the module reads the handles the optimizer just wrote
        ttnn.synchronize_device(dev)
        t3 = time.time()
        for p in params.values():
            p.grad = None
        ag.release_pins()
        del out, out_t
        md = hashlib.sha256(b"".join(opt.master[n].tobytes() for n in sorted(opt.master))).hexdigest()[:16]
        s = clk[c0:] or clk[-1:]
        print(json.dumps(dict(step=step + 1, loss=loss, fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                              opt_s=round(t3 - t2, 4), step_s=round(t3 - t0, 4), retained_B=retained,
                              after_bwd_B=peak_dev, dead=len(dead), params=len(params),
                              master_digest=md, aiclk=[min(s or [0]), max(s or [0])],
                              host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))),
              flush=True)
        if not a.resume:
            cov.observe(dead)
        if a.save_at and step + 1 == a.save_at:
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics={"loss": loss}, force=True)))), flush=True)
    stop.set()


if __name__ == "__main__":
    main()
