"""r4t-step: optimizer steps on RFD4-Proteina's real trunk (release weights, 14 layers + 7 pair updates), one chip.

Per step, as upstream's recipe runs it: `k` noise candidates through the production trunk grad-off at recycle depth
`d` (one batched inference call), then the winner's LAST pass re-run taped on the production methods
by the production `Trunk.__call__` under the tape, earlier passes no_grad as upstream's `recycle_grad: last`. Not yet bridged: the host glue (conditioning, init_repr, pair builder) and heads; their inputs
are seeded tensors and the loss seed lands on s (`<s, C>/numel`). Prints one JSON line per step.

  --save-at k / --resume PATH: checkpoint after step k / continue from it in a fresh process (bit-identical check)
  --freeze SUB: the G2 fail arm. The first trained upstream leaf whose name contains SUB stops requiring grad
     before the build, so its fold uploads as a constant: one parameter detached from the graph, as a load-path
     `.detach()` bug would. (Cutting the token stream at a layer detaches nothing: the pair biases and z carry
     every earlier layer forward.)
  --batch P1,P2,..: the real step, step i on batch P[i % len], each a batch upstream itself trained on (`perf/r4t/loss_parity_reference.py` dumps:
     every feature, x_1/x_t/t, k candidates, depth, upstream winners and losses). Host glue (conditioning,
     initial representations, pair builder) and heads run with grad on the host and join the tape through
     `train.bridge.Bridge`; the loss is `train.rfd4_objective.rfd4_loss`. The candidates' winner is chosen
     from our own heads, as upstream does, and printed beside upstream's.
  --exempt FILE: G2 exemptions, one `name<TAB>upstream's reason` a line
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
    ap.add_argument("--freeze", default="")
    ap.add_argument("--batch", default="")
    ap.add_argument("--exempt", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/tck"))
    ap.add_argument("--cand-pair", choices=("host", "card"), default="card",
                    help="where the grad-off K-candidate pass builds z_init: host (upstream's builder) or the serving "
                         "card path (rfd4.pair.PairOnCard: host constant + bins, card assembly). The taped winner is always host.")
    ap.add_argument("--lnbw-weights", choices=("on", "off"), default="on",
                    help="LayerNorm backward with trainable gamma/beta in one fused kernel (tt_bio.lnbw weights form, "
                         "r4t-perf L2) or the composed ttnn path")
    ap.add_argument("--gate-bw", choices=("on", "off"), default="off",
                    help="backward of every o * sigmoid(g) gate as one kernel (tt_bio.gate_bw, r4t-perf G1) or the "
                         "seven-op composed path")
    a = ap.parse_args()
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> prints every thread stack to the log
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio import lnbw
    lnbw.WEIGHTS = a.lnbw_weights == "on"
    from tt_bio import gate_bw
    gate_bw.GATE_BW_FUSED = a.gate_bw == "on"
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
    card_pair = []   # r4t-perf P1 (--cand-pair card): the candidate pass builds its pair as serving does

    def pair_card():
        """This step's PairOnCard, or None. A split copies the builder's weights, which the optimizer moves, so
        each step takes a fresh one (the split refuses stale weights)."""
        if a.cand_pair != "card":
            return None
        from tt_bio.rfd4.pair import PairOnCard, PairSplit
        if card_pair:
            card_pair.pop().release()
        glue.pair_split = PairSplit(glue.pair_repr_builder)
        card_pair.append(PairOnCard(glue.pair_split, dev))
        return card_pair[0]
    trunk = E.load_trunk(glue, state, cfg, training=True)   # an unrecorded build for the lever set and dtypes; build() replaces it

    RB = {}

    # R4T_REBUILD_INLINE=1 (r4t-perf R1): a per-step rebuild skips the grad-off forward whose only job is to make
    # the lazy uploads happen under the recording. The recording stays open into the step's own forward instead,
    # and each lazy upload of a leaf-fed tensor is declared a parameter the moment it is made, before the taped
    # winner reads it. The first build and a resume build still run the forward: the unreached/orphan checks
    # after them need every weight to exist.
    inline = os.environ.get("R4T_REBUILD_INLINE", "1") == "1"   # default ON since ca9fa7ac4: digest-identical to the lazy-forward rebuild (rb/I4.jsonl)
    pack = os.environ.get("R4T_DEVICE_PACK", "0") == "1"   # r4t-perf R2: bfp8 weights packed on the card in the rebuild (Folds.recording device_pack)
    # The trimul in-projection is cached per chunk width on first use, and caches filled in the grad-off
    # candidate pass reach the taped winner as untracked constants (R1 first form: step-2 digest moved, every
    # linear_{a,b}_{p,g} grad dead). `train_in_proj` holds the two sources as registered device tensors and
    # cuts chunks per call, so nothing is cached. Implied by R4T_REBUILD_INLINE.
    in_proj = os.environ.get("R4T_TRAIN_IN_PROJ") == "1"   # train_in_proj rounds dW to bf16 (gcmp.txt); not implied

    def trimuls(root):
        seen, out, todo = set(), [], [root]
        while todo:
            o = todo.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "prewarm") and hasattr(o, "width_variants"):
                out.append(o)
            if isinstance(o, (list, tuple)):
                todo.extend(o)
            elif isinstance(o, dict):
                todo.extend(o.values())
            elif hasattr(o, "__dict__") and not isinstance(o, (torch.Tensor, ttnn.Tensor, type)):
                todo.extend(v for v in vars(o).values() if not isinstance(v, (torch.Tensor, ttnn.Tensor)))
        return out

    open_rec = []

    def walk_register():
        """Inline mode, between the grad-off candidates and the taped winner: register what the candidate pass
        cached (per-width uploads AND the device concats derived from them), as the stock rebuild's walk after
        its lazy forward does, before any taped read."""
        if open_rec:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ag.parameter(t)

    def close_rec():
        while open_rec:
            open_rec.pop()()

    def build(lazy=True):
        """The serving trunk from the current upstream leaves, every upload recorded with its fold. Called
        again after each optimizer step: the device weights are derived, never stepped themselves."""
        nonlocal trunk
        _rb0 = time.time()
        if trunk is not None:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ttnn.deallocate(getattr(t, "value", t))
            ag.release_pins()
            ag.forget_parameters()   # the old leaves point at freed handles; build() registers the new ones
        folds.sources.clear()
        folds.consts.clear()
        ttnn.synchronize_device(dev); _rb1 = time.time()
        close_rec()
        if lazy or not inline:
            with folds.recording(device_pack=pack) as sd:
                trunk = E.load_trunk(glue, sd, cfg, training=True)
                if in_proj:
                    RB["trimuls"] = len([m.train_in_proj() for m in trimuls(trunk)])
                ttnn.synchronize_device(dev); _rb2 = time.time()
                one_example(0, False)   # lazy uploads (trimul in-projections at this chunk width), on the taped route
                ttnn.synchronize_device(dev); _rb3 = time.time()
        else:
            rec = folds.recording(device_pack=pack)
            sd = rec.__enter__()
            trunk = E.load_trunk(glue, sd, cfg, training=True)
            if in_proj:
                RB["trimuls"] = len([m.train_in_proj() for m in trimuls(trunk)])
            # The trimul chunk width depends on the batch: the K-candidate pass fills its own width, and the
            # batch-1 winner would upload its in-projection INSIDE its taped verb, after the verb has resolved
            # its parents (R1 forms 1 and 3: 14 sources recorded, read, never on the tape). `prewarm` fills
            # exactly the entry `__call__` will use, so the winner width exists and is walked below.
            RB["prewarmed"] = len([m.prewarm(a.n, 1) for m in trimuls(trunk) if hasattr(m, "prewarm")])
            ttnn.synchronize_device(dev); _rb2 = _rb3 = time.time()
            recorded = ttnn.from_torch

            def declare(x, *a_, **k_):
                t_ = recorded(x, *a_, **k_)
                if folds.sources.get(id(t_), (None,))[0] is t_:
                    ag.parameter(t_)
                return t_

            ttnn.from_torch = declare

            def _close():
                ttnn.from_torch = recorded
                rec.__exit__(None, None, None)

            open_rec.append(_close)
        for _p, _o, _k, t in T.walk_device_weights(trunk):
            ag.parameter(t)
        RB.update(free_s=round(_rb1 - _rb0, 4), load_s=round(_rb2 - _rb1, 4), lazy_fwd_s=round(_rb3 - _rb2, 4),
                  register_s=round(time.time() - _rb3, 4))

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
        return trunk(s_init, z_init, conds, masks)

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
        batches = [(os.path.basename(f),) + load_batch(f) for f in a.batch.split(",")]
        cur = {}

        def use(step):
            nm, b, m, _rec, rl = batches[step % len(batches)]
            cur.update(name=nm, bt=b, meta=m, ref=rl, K=int(b["padding_mask"].shape[0]), depth=max(1, int(m["depth"])))
        use(0)
        glue.requires_grad_(True)

        def take(d, i):
            if isinstance(d, dict):
                return {k: take(v, i) for k, v in d.items()}
            return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == cur["K"] else d

        def npy(d):
            return {k: npy(v) for k, v in d.items()} if isinstance(d, dict) else (
                d.detach().numpy() if torch.is_tensor(d) else d)

        def conds_for(b):
            depth = cur["depth"]
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
            bt, K = cur["bt"], cur["K"]
            mk = tmask(bt)
            pc = pair_card()
            with torch.no_grad():
                ck = conds_for(bt)
                s0, z0 = glue.initial_states(bt, pair_constant=glue.pair_split.constant(bt) if pc else None)
            with T.lever_scope(levers), ag.tape():
                with ag.no_grad():
                    zk0 = pc(z0, 0) if pc else put(z0, trunk.pair_dtype(mk))
                    sk, zk = trunk(put(s0), zk0, [put(c, trunk.dtype) for c in ck], mk)
                    sh = ttnn.to_torch(getattr(sk, "value", sk)).float().reshape(s0.shape)
                walk_register()
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
            ref = {k: float(cur["ref"][k][ctx["win"]]) for k in terms if k in cur["ref"]}
            return total, terms, ref
        real = (real_example, real_loss)

    ag.install()
    build()
    unreached = set(folds.unreached())   # one autograd pass over every fold; never inside a per-leaf loop
    every = folds.params()
    params = {nm: p for nm, p in every.items() if nm not in unreached}
    if real:
        params.update(host_params(glue, "glue."))
    if a.freeze:
        fz = next(nm for nm in sorted(params) if a.freeze in nm)
        folds.leaves[fz].requires_grad_(False)
        build()
        print(json.dumps(dict(frozen=fz)), flush=True)
    names = list(params)
    cyc = {nm: "depth-1 draws feed a zero state into the cycle projection; trains at depth >= 2"
           for nm in names if "cycle" in nm and a.depth < 2}
    opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0)
    start = 0
    if a.resume:
        start = int(load_adapter(a.resume, dict(params), dev, opt=opt)["step"])
        build()
    ck = Checkpointer(a.out, every=10**9, keep=10)
    if a.exempt:
        for ln in open(a.exempt):
            if ln.strip() and not ln.startswith("#"):
                nm, why = ln.rstrip("\n").split("\t", 1)
                cyc[nm] = why
    cov = GradientCoverage(names, exempt=cyc, where="r4t-step trunk: ")
    threading.Thread(target=sample, daemon=True).start()
    for step in range(start, a.steps):
        ttnn.synchronize_device(dev)
        base = used()
        c0 = len(clk)
        t0 = time.time()
        extra = {}
        if real:
            use(step)
            s, z, ctx = real[0](True)
            close_rec()
            ttnn.synchronize_device(dev)
            t1 = time.time()
            retained = used() - base
            loss, terms, ref = real[1](s, ctx)
            ttnn.synchronize_device(dev)
            extra_mem = used()
            st = s
            extra = dict(dev_after_heads_B=extra_mem, batch=cur["name"], winner=ctx["win"], upstream_winner=int(cur["meta"]["winners"][0]), depth_upstream=cur["depth"],
                         terms=terms, upstream_terms=ref)
        else:
            s, z = one_example(step, True)
            close_rec()
            ttnn.synchronize_device(dev)
            t1 = time.time()
            retained = used() - base
            st = s if isinstance(s, ag.Tensor) else ag.Tensor(s)
            loss = float((ttnn.to_torch(st.value).float().reshape(Cs.shape) * Cs).sum() / numel)
            ag.backward([st], [seed_s])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        after_bwd = used()
        idle = folds.tape_backward()   # device-weight gradients through each fold's VJP into the upstream leaves
        grads = {nm: (None if p.grad is None else p.grad.float().numpy()) for nm, p in params.items()}
        dead = zero_or_missing(grads)
        if os.environ.get("R4T_DUMP_G") and step == start:
            import numpy as _np
            _np.savez(os.environ["R4T_DUMP_G"], **{nm.replace(".", "__"): g for nm, g in grads.items()
                                                    if g is not None and "pair_update_layers" in nm})
        t2b = time.time()
        opt.step()
        for p in params.values():
            p.grad = None
        t2c = time.time()
        build(lazy=False)
        ttnn.synchronize_device(dev)
        t3 = time.time()
        del s, z, st
        md = hashlib.sha256(b"".join(opt.master[k].tobytes() for k in sorted(opt.master))).hexdigest()[:16]
        sm = clk[c0:] or clk[-1:]
        print(json.dumps(dict(step=step + 1, loss=loss, fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                              fold_bwd_s=round(t2b - t2, 4), opt_s=round(t2c - t2b, 4), rebuild_s=round(t3 - t2c, 4), rebuild_split=dict(RB), idle_dev_w=len(idle), step_s=round(t3 - t0, 4), retained_B=retained,
                              dead=len(dead), dead_names=sorted(dead)[:20], dev_base_B=base, dev_after_bwd_B=after_bwd, params=len(params), n_elems=int(sum(v.size for v in opt.master.values())),
                              master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])], aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                              k=a.k, depth=a.depth, n=n, lnbw_reach=dict(lnbw.REACH), gate_bw=list(gate_bw.STATS), **extra,
                              host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))),
              flush=True)
        if a.save_at and step + 1 == a.save_at:   # before the guard, so a refusal still leaves the resume test its input
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics={"loss": loss}, force=True)))), flush=True)
        if not a.resume:
            cov.observe(dead)
    stop.set()


if __name__ == "__main__":
    main()
