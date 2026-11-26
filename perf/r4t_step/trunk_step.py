"""r4t-step: optimizer steps on RFD4-Proteina's real trunk (release weights, 14 layers + 7 pair updates), one chip.

Per step, as upstream's recipe runs it: `k` noise candidates through the production trunk grad-off at recycle depth
`d` (one batched inference call), then the winner's LAST pass re-run taped on the production methods
by the production `Trunk.__call__` under the tape, earlier passes no_grad as upstream's `recycle_grad: last`. Not yet bridged: the host glue (conditioning, init_repr, pair builder) and heads; their inputs
are seeded tensors and the loss seed lands on s (`<s, C>/numel`). Prints one JSON line per step.

  --save-at k / --resume PATH: checkpoint after step k / continue from it in a fresh process (bit-identical check)
  --detach-at j: cut the graph in front of trunk layer j (the G2 fail arm)
  --batch PATH: the real step on a batch upstream itself trained on (`perf/r4t/loss_parity_reference.py` dumps:
     every feature, x_1/x_t/t, k candidates, depth, upstream winners and losses). Host glue (conditioning,
     initial representations, pair builder) and heads run with grad on the host and join the tape through
     `train.bridge.Bridge`; the loss is `train.rfd4_objective.rfd4_loss`. The candidates' winner is chosen
     from our own heads, as upstream does, and printed beside upstream's.
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
    ap.add_argument("--batch", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/tck"))
    a = ap.parse_args()
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> prints every thread stack to the log
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import network as N, engine as E, trunk as TR
    from tt_bio.train.folds import Folds
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
    folds = Folds(state)
    trunk = E.load_trunk(glue, state, cfg)   # an unrecorded build for the lever set and dtypes; build() replaces it

    def build():
        """The serving trunk from the current upstream leaves, every upload recorded with its fold. Called
        again after each optimizer step: the device weights are derived, never stepped themselves."""
        nonlocal trunk
        if trunk is not None:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ttnn.deallocate(getattr(t, "value", t))
            ag.release_pins()
        folds.sources.clear()
        folds.consts.clear()
        with folds.recording() as sd:
            trunk = E.load_trunk(glue, sd, cfg)
            one_example(0, False)   # lazy uploads (trimul in-projections at this chunk width), on the taped route
        for _p, _o, _k, t in T.walk_device_weights(trunk):
            ag.parameter(t)

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

    def winner(s_init, z_init, conds=None, masks=None):
        """The production `Trunk.__call__` at depth d: under a tape every cycle but the last runs no_grad
        (`ops.recycle_region`, upstream's `recycle_grad: last`), so only the last pass is differentiated."""
        conds = [cond1] * a.depth if conds is None else conds
        masks = masks1 if masks is None else masks
        if a.detach_at < 0:
            return trunk(s_init, z_init, conds, masks)
        blk = trunk.blocks[a.detach_at]

        def cut(s, *r, **kw):
            return type(blk).__call__(blk, ttnn.clone(s.value if isinstance(s, ag.Tensor) else s), *r, **kw)
        trunk.blocks[a.detach_at] = cut
        try:
            return trunk(s_init, z_init, conds, masks)
        finally:
            trunk.blocks[a.detach_at] = blk

    def one_example(step, taped):
        s0, z0 = inputs(step)
        # Everything runs inside one tape so the walked (taped) weights meet taped verbs everywhere; the
        # k candidates are the sampler's batched inference call and stay grad-off.
        with T.lever_scope(levers), ag.tape():
            with ag.no_grad():
                sk, zk = trunk(put(s0), put(z0, zd), [condk] * a.depth, masksk)
            win = 0   # the objective picks the winner from the heads' loss; heads are not bridged yet (see doc)
            for t in (sk, zk):
                ttnn.deallocate(t.value if isinstance(t, ag.Tensor) else t)
            s_init, z_init = put(s0[win:win + 1]), put(z0[win:win + 1], zd)
            if not taped:
                with ag.no_grad():
                    return winner(s_init, z_init)
            return winner(s_init, z_init)

    real = None
    if a.batch:
        import sys
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t"))
        from loss_parity_reference import load_batch
        from tt_bio.train import rfd4_objective as RO
        from tt_bio.train.bridge import Bridge, host_params
        bt, meta, _rec, ref_loss = load_batch(a.batch)
        K, depth = int(bt["padding_mask"].shape[0]), max(1, int(meta["depth"]))
        glue.requires_grad_(True)

        def take(d, i):
            if isinstance(d, dict):
                return {k: take(v, i) for k, v in d.items()}
            return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d

        def npy(d):
            return {k: npy(v) for k, v in d.items()} if isinstance(d, dict) else (
                d.detach().numpy() if torch.is_tensor(d) else d)

        def conds_for(b):
            cs = [glue.conditioning(b, None)]
            for cyc in range(1, depth):
                cs.append(glue.conditioning(b, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
            return cs

        def flat(o):
            v = lambda m: next(iter(o[m].values()))
            out = {"v_bb_ca": v("bb_ca"), "v_local_latents": v("local_latents")}
            if "sequence" in o:
                out["sequence_logits"] = o["sequence"]["logits"]
            if "insertion" in o:
                out["insertion_logits"] = o["insertion"]["logits"]
            return out

        def tmask(b):
            return TR.trunk_masks(b["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)

        def real_example(taped):
            """Upstream's step on the dumped batch: k candidates grad-off through glue + trunk + heads, the
            winner by our own exploration loss, then the winner re-run with grad (host glue, taped trunk)."""
            mk = tmask(bt)
            with torch.no_grad():
                ck = conds_for(bt)
                s0, z0 = glue.initial_states(bt)
            with T.lever_scope(levers), ag.tape():
                with ag.no_grad():
                    sk, zk = trunk(put(s0), put(z0, trunk.pair_dtype(mk)), [put(c, trunk.dtype) for c in ck], mk)
                    sh = ttnn.to_torch(getattr(sk, "value", sk)).float().reshape(s0.shape)
                for t in (sk, zk):
                    ttnn.deallocate(getattr(t, "value", t))
                with torch.no_grad():
                    ok = {k: v.double().numpy() for k, v in flat(glue.heads(sh, bt)).items()}
                _t, bd, _s = RO.rfd4_loss({**npy(bt), "exploration_k": K}, ok, RO.WEIGHTS)
                win = int(bd["winners"][0])
                bw = take(bt, win)
                br = Bridge(dev, trunk.dtype)
                m1 = tmask(bw)
                if not taped:
                    with torch.no_grad(), ag.no_grad():
                        s_i, z_i = glue.initial_states(bw)
                        return winner(put(s_i), put(z_i, trunk.pair_dtype(m1)),
                                      [put(c, trunk.dtype) for c in conds_for(bw)], m1)
                cs = conds_for(bw)
                s_i, z_i = glue.initial_states(bw)
                s, z = winner(br.up(s_i, ttnn.float32), br.up(z_i, trunk.pair_dtype(m1)),
                              [br.up(c) for c in cs], m1)
            return s, z, dict(br=br, bw=bw, win=win, shape=s_i.shape)

        def real_loss(s, ctx):
            """rfd4_loss on our heads; its seeds are d total / d output, so <outputs, seeds> carries the
            gradient into the heads, back over the bridge into the tape and from its inputs into the glue."""
            br, bw = ctx["br"], ctx["bw"]
            sh = br.down(s if isinstance(s, ag.Tensor) else ag.Tensor(s, requires_grad=True)).reshape(ctx["shape"])
            o = flat(glue.heads(sh, bw))
            total, bd, seeds = RO.rfd4_loss({**npy(bw), "exploration_k": 1},
                                            {k: v.detach().double().numpy() for k, v in o.items()}, RO.WEIGHTS)
            sur = sum((o[k] * torch.from_numpy(np.asarray(g)).to(o[k].dtype)).sum() for k, g in seeds.items())
            br.backward(sur)
            terms = {k: v.get("value") for k, v in bd.items() if isinstance(v, dict)}
            ref = {k: float(ref_loss[k][ctx["win"]]) for k in terms if k in ref_loss}
            return total, terms, ref
        real = (real_example, real_loss)

    ag.install()
    build()
    unreached = set(folds.unreached())   # one autograd pass over every fold; never inside a per-leaf loop
    every = folds.params()
    params = {nm: p for nm, p in every.items() if nm not in unreached}
    if real:
        params.update(host_params(glue, "glue."))
    names = list(params)
    cyc = {nm: "depth-1 draws feed a zero state into the cycle projection; trains at depth >= 2"
           for nm in names if "cycle" in nm and a.depth < 2}
    opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0)
    start = 0
    if a.resume:
        start = int(load_adapter(a.resume, dict(params), dev, opt=opt)["step"])
        build()
    ck = Checkpointer(a.out, every=10**9, keep=10)
    cov = GradientCoverage(names, exempt=cyc, where="r4t-step trunk: ")
    threading.Thread(target=sample, daemon=True).start()
    for step in range(start, a.steps):
        ttnn.synchronize_device(dev)
        base = used()
        c0 = len(clk)
        t0 = time.time()
        extra = {}
        if real:
            s, z, ctx = real[0](True)
            ttnn.synchronize_device(dev)
            t1 = time.time()
            retained = used() - base
            loss, terms, ref = real[1](s, ctx)
            st = s
            extra = dict(winner=ctx["win"], upstream_winner=int(meta["winners"][0]), depth_upstream=depth,
                         terms=terms, upstream_terms=ref)
        else:
            s, z = one_example(step, True)
            ttnn.synchronize_device(dev)
            t1 = time.time()
            retained = used() - base
            st = s if isinstance(s, ag.Tensor) else ag.Tensor(s)
            loss = float((ttnn.to_torch(st.value).float().reshape(Cs.shape) * Cs).sum() / numel)
            ag.backward([st], [seed_s])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        idle = folds.tape_backward()   # device-weight gradients through each fold's VJP into the upstream leaves
        grads = {nm: (None if p.grad is None else p.grad.float().numpy()) for nm, p in params.items()}
        dead = zero_or_missing(grads)
        t2b = time.time()
        opt.step()
        for p in params.values():
            p.grad = None
        t2c = time.time()
        build()
        ttnn.synchronize_device(dev)
        t3 = time.time()
        del s, z, st
        md = hashlib.sha256(b"".join(opt.master[k].tobytes() for k in sorted(opt.master))).hexdigest()[:16]
        sm = clk[c0:] or clk[-1:]
        print(json.dumps(dict(step=step + 1, loss=loss, fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                              fold_bwd_s=round(t2b - t2, 4), opt_s=round(t2c - t2b, 4), rebuild_s=round(t3 - t2c, 4), idle_dev_w=len(idle), step_s=round(t3 - t0, 4), retained_B=retained,
                              dead=len(dead), dead_names=sorted(dead)[:6], params=len(params), n_elems=int(sum(v.size for v in opt.master.values())),
                              master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])], aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                              k=a.k, depth=a.depth, n=n, **extra,
                              host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))),
              flush=True)
        if not a.resume:
            cov.observe(dead)
        if a.save_at and step + 1 == a.save_at:
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics={"loss": loss}, force=True)))), flush=True)
    stop.set()


if __name__ == "__main__":
    main()
