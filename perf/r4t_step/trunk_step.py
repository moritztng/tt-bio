"""r4t-step: optimizer steps on RFD4-Proteina's real trunk (release weights, 14 layers + 7 pair updates), one chip.

Per step, as upstream's recipe runs it: `k` noise candidates through the production trunk grad-off at recycle depth
`d` (one batched inference call), then the winner's LAST pass re-run taped on the production methods
(`Trunk._recycle` + `Trunk.run_pass`, the same calls `Trunk.__call__` makes), earlier passes detached as upstream's
`recycle_grad: last`. Not yet bridged: the host glue (conditioning, init_repr, pair builder) and heads; their inputs
are seeded tensors and the loss seed lands on s (`<s, C>/numel`). Prints one JSON line per step.

  --save-at k / --resume PATH: checkpoint after step k / continue from it in a fresh process (bit-identical check)
  --detach-at j: cut the graph in front of trunk layer j (the G2 fail arm)
"""
import argparse
import hashlib
import json
import os
import resource
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--save-at", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--detach-at", type=int, default=-1)
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/tck"))
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import network as N, engine as E, trunk as TR
    from tt_bio.train import lora
    from tt_bio.train.checkpoint import Checkpointer, load_adapter
    from tt_bio.train.optim import AdamW
    from tt_bio.train.parity import GradientCoverage, zero_or_missing

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

    def put(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    trunk = E.load_trunk(glue, state, cfg)
    levers = trunk._levers
    n, D, P, C_ = a.n, cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(3)
    cond = torch.randn(1, n, C_, generator=g)
    Cs = torch.randn(1, n, D, generator=g)
    numel = Cs.numel()
    seed_s = put(Cs / numel)
    masks1 = TR.trunk_masks(torch.ones(1, n), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    masksk = TR.trunk_masks(torch.ones(a.k, n), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    cond1 = put(cond, trunk.dtype)
    condk = put(cond.expand(a.k, -1, -1), trunk.dtype)
    zd = trunk.pair_dtype(masks1)

    def inputs(step):
        gs = torch.Generator().manual_seed(1000 + step)
        s0 = torch.randn(a.k, n, D, generator=gs)
        z0 = torch.randn(a.k, n, n, P, generator=gs) * 0.5
        return s0, z0

    def last_pass(s_init, z_init, s_prev, z_prev):
        """`Trunk.__call__`'s final iteration, taped: recycle from the detached previous pass, then one pass."""
        low = trunk.dtype if zd == ttnn.float32 and trunk.dtype == ttnn.bfloat16 else None
        zb = None
        if s_prev is None:
            s_in, z_in = ttnn.clone(s_init), ttnn.clone(z_init)
        else:
            s_in = trunk._recycle(s_init, s_prev, trunk.s_norm, trunk.s_proj, trunk.residual_dtype)
            z_in = trunk._recycle(z_init, z_prev, trunk.z_norm, trunk.z_proj, zd, low=low)
            if low is not None:
                z_in, zb = z_in
        if a.detach_at >= 0:
            blk = trunk.blocks[a.detach_at]

            def cut(s, *r, **kw):
                return type(blk).__call__(blk, ttnn.clone(s.value if isinstance(s, ag.Tensor) else s), *r, **kw)
            trunk.blocks[a.detach_at] = cut
            try:
                return trunk.run_pass(s_in, z_in, cond1, masks1, zb=zb)
            finally:
                trunk.blocks[a.detach_at] = blk
        return trunk.run_pass(s_in, z_in, cond1, masks1, zb=zb)

    def one_example(step, taped):
        s0, z0 = inputs(step)
        # k candidates, production trunk, grad off, at depth d: the inference call the sampler makes
        sk, zk = trunk(put(s0), put(z0, zd), [condk] * a.depth, masksk)
        win = 0   # the objective picks the winner from the heads' loss; heads are not bridged yet (see doc)
        s_init, z_init = put(s0[win:win + 1]), put(z0[win:win + 1], zd)
        s_prev = z_prev = None
        if a.depth > 1:
            s_prev, z_prev = trunk(s_init, z_init, [cond1] * (a.depth - 1), masks1)
        ttnn.deallocate(sk)
        ttnn.deallocate(zk)
        with T.lever_scope(levers):
            if not taped:
                return last_pass(s_init, z_init, s_prev, z_prev)
            with ag.tape():
                return last_pass(s_init, z_init, s_prev, z_prev)

    ag.install()
    params = lora.walked_weights(lambda: one_example(0, False), None, trunk)
    names = list(params)
    cyc = {nm: "depth-1 draws feed a zero state into the cycle projection; trains at depth >= 2"
           for nm in names if "cycle" in nm and a.depth < 2}
    opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0)
    start = 0
    if a.resume:
        start = int(load_adapter(a.resume, dict(params), dev, opt=opt)["step"])
        params.rebind()
    ck = Checkpointer(a.out, every=10**9, keep=10)
    cov = GradientCoverage(names, exempt=cyc, where="r4t-step trunk: ")
    threading.Thread(target=sample, daemon=True).start()
    for step in range(start, a.steps):
        ttnn.synchronize_device(dev)
        base = used()
        c0 = len(clk)
        t0 = time.time()
        s, z = one_example(step, True)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        retained = used() - base
        st = s if isinstance(s, ag.Tensor) else ag.Tensor(s)
        loss = float((ttnn.to_torch(st.value).float().reshape(Cs.shape) * Cs).sum() / numel)
        ag.backward([st], [seed_s])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        grads = {nm: (None if p.grad is None else ttnn.to_torch(p.grad).float().numpy()) for nm, p in params.items()}
        dead = zero_or_missing(grads)
        opt.step()
        params.rebind()
        ttnn.synchronize_device(dev)
        t3 = time.time()
        for p in params.values():
            p.grad = None
        ag.release_pins()
        del s, z, st
        md = hashlib.sha256(b"".join(opt.master[k].tobytes() for k in sorted(opt.master))).hexdigest()[:16]
        sm = clk[c0:] or clk[-1:]
        print(json.dumps(dict(step=step + 1, loss=loss, fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                              opt_s=round(t3 - t2, 4), step_s=round(t3 - t0, 4), retained_B=retained,
                              dead=len(dead), params=len(params), n_elems=int(sum(v.size for v in opt.master.values())),
                              master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])], aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                              k=a.k, depth=a.depth, n=n,
                              host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))),
              flush=True)
        if not a.resume:
            cov.observe(dead)
        if a.save_at and step + 1 == a.save_at:
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics={"loss": loss}, force=True)))), flush=True)
    stop.set()


if __name__ == "__main__":
    main()
