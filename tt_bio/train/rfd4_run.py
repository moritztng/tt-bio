"""Optimizer steps on RFD4-Proteina's real trunk (release weights, 14 layers + 7 pair updates), one chip.

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
  --dp-cards C0,C1,..: data parallel, one rank per card (processes), gradients summed by the launcher's
     collective (`train.launcher.ProcessAxis`); --shard steps a ZeRO-1 slice per rank. Rank r's example at
     step i is index i*world + r, so a world-w step consumes w distinct examples; each rank's JSON lines go
     to OUT/rank<r>.log and its checkpoints under OUT/rank<r>/. Master digests must agree across ranks.

Moved here from `perf/r4t_step/trunk_step.py` (r4t D56), which is now a thin caller; `main(argv)`
takes the same flags and prints the same JSON lines.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import threading
import time

from tt_bio.envflags import env_flag

#: Parameters whose upstream gradient is identically zero on unconditional batches (D29).
#: Selected by ``--exempt default`` (used by Tier 0, which has no ``perf/`` tree in a wheel).
UPSTREAM_EXEMPT = {
    "glue.init_repr_factory.feat_creators.7.ss_embedding.weight":
        "upstream Embedding(4, 16, padding_idx=0) and every unconditional token is NONE "
        "(index 0): upstream gradient is an exact zero (r4t-accuracy D29)",
}


def _checkpoint_due(step: int, total_steps: int, save_every: int, save_at: int) -> tuple[bool, bool]:
    """Return ``(should_save, force)`` for the periodic checkpointer at 1-based ``step``.

    When ``--save-every`` is set, save on every multiple and on the final step (matching
    ``tt-bio train --checkpoint-every``'s contract), skipping if ``--save-at`` already wrote it.
    """
    if not save_every or (save_at and step == save_at):
        return False, False
    last = step == total_steps
    return (last or step % save_every == 0), last


def main(argv=None, on_step=None):
    """``on_step(row)`` is called with each step's row, the same dict that is printed. Tier 0
    (`rfd4_tier0`) hands it the status writer; a bare run hands nothing and only prints."""
    ap = argparse.ArgumentParser(prog="trunk_step")
    ap.add_argument("--n", type=int, default=256, help="token length; taken from --batch when one is given")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--save-at", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze", default="")
    ap.add_argument("--batch", default="")
    ap.add_argument("--exempt", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/tck"))
    ap.add_argument("--curve-dump", default="", help="the G3 curve leg: a directory of plan-step "
                    "inputs from `curve_upstream.py --dump-dir` (plan.json beside them). Step i "
                    "trains on step<i>.npz at ITS drawn depth, in order, no cycling.")
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (newest "
                    "two kept), so a long leg resumes with --resume instead of starting over")
    ap.add_argument("--keep", type=int, default=0, help="with --save-every: keep the newest N "
                    "checkpoints by step, not by loss (a resume ring)")
    ap.add_argument("--pin-spike", type=float, default=0.0, help="pin, never prune, the "
                    "checkpoint a step started from when its loss exceeds F x the median of the "
                    "previous 100 (>= 20 seen), so the spike replays from it")
    ap.add_argument("--curve-out", default="", help="with --curve-dump: the CurveLog to append to, "
                    "the file grade_parity.py --ours reads")
    ap.add_argument("--dp-cards", default="")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--dp-rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--dp-run", default="", help=argparse.SUPPRESS)
    ap.add_argument("--cand-pair", choices=("host", "card"), default="card",
                    help="where the grad-off K-candidate pass builds z_init: host (upstream's builder) or the serving "
                         "card path (rfd4.pair.PairOnCard: host constant + bins, card assembly). The taped winner is always host.")
    ap.add_argument("--cand-levers", choices=("training", "serving", "inference"), default="serving",
                    help="lever set of the grad-off K-candidate pass: the training set, or that set plus the dropped levers that "
                         "switch at run time (pair_bf16z, dit_sigf32, tok_recycle_mm; r4t-perf K2). The pass only picks the "
                         "winner, so the gradient is unchanged whenever the winner is; inference also lets the fused kernels without a tape entry "
                         "serve that pass (ops.NOGRAD_IS_INFERENCE, r4t-perf K4)")
    ap.add_argument("--lnbw-weights", choices=("on", "off"), default="on",
                    help="LayerNorm backward with trainable gamma/beta in one fused kernel (tt_bio.lnbw weights form, "
                         "r4t-perf L2) or the composed ttnn path")
    ap.add_argument("--grad-dump", default="", help="write every trained parameter's gradient of the first "
                    "step (host glue + upstream trunk leaves, after the fold VJPs) to PATH.npz with PATH.json "
                    "(winner, batch, loss, clock) and stop before the optimizer: the device side of "
                    "perf/r4t_tape/net_gradcheck.py")
    ap.add_argument("--arm", choices=("prod", "fp32-exact"), default="prod",
                    help="fp32-exact: the D49 control, float32 trunk with no levers and every contraction "
                         "under tt_bio.train.exact_fp32 (host float64, rounded once)")
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="with --grad-dump: the gross broken arm, block 0's Q slice dropped from its fold VJP")
    ap.add_argument("--attn-fp32", choices=("on", "off"), default="on",
                    help="q/k/v projections, q/k norms and attention in float32 (load_trunk attn_fp32). On by default: "
                         "with it off (bf16 attention) the RFD4 curve spikes at step 823 and fails G3; off is the opt-out")
    ap.add_argument("--gate-bw", choices=("on", "off"), default="on",
                    help="backward of every o * sigmoid(g) gate as one kernel (tt_bio.gate_bw, r4t-perf G1) or the "
                         "seven-op composed path")
    ap.add_argument("--cover-steps", type=int, default=0,
                    help="steps the no-gradient guard intersects over (default: the guard's own minimum). A run cycling "
                         "logged batches sets it to the cycle length: a head no early batch drives is not dead")
    ap.add_argument("--pcie-meter", choices=("off", "on", "sites"), default="off",
                    help="count host<->device bytes and transfers per step phase (perf/r4t_perf/pcie_meter.py); "
                         "sites adds the top call sites by bytes")
    ap.add_argument("--optimizer", choices=("auto", "muon", "adamw"), default="auto",
                    help="update rule: muon (upstream Muon + Adam, rfd4_objective.update_rule), "
                         "adamw (plain AdamW at 1e-4), or auto (muon for --curve-dump, adamw otherwise)")
    a = ap.parse_args(argv)
    if a.arm == "fp32-exact":
        os.environ["TT_BIO_LEVERS"] = "none"   # Trunk builds on mode_levers(), which reads this, not lever_scope
        a.lnbw_weights = "off"   # the fused LN-backward kernel is a perf lever; the control arm verified by r4t-tape has none
        a.gate_bw = "off"        # likewise the fused gate backward (r4t-perf G1)
        a.cand_levers = "training"   # and the serving levers in the candidate pass (r4t-perf K2)
    if a.curve_dump and a.dp_cards:
        raise SystemExit("--curve-dump is a single-card leg: every rank would append to one CurveLog")
    if a.dp_cards and a.dp_rank < 0:
        return launch(a, argv)
    world = len(a.dp_cards.split(",")) if a.dp_cards else 1
    rank = max(a.dp_rank, 0)
    if a.curve_dump:
        from tt_bio.train.curve import CurveLog, read_dump_dir
        cdir = os.path.expanduser(a.curve_dump)
        cplan, cfiles = read_dump_dir(cdir)   # refuses a dump that is not this plan's
        if not a.curve_out:
            raise SystemExit("--curve-out is required with --curve-dump")
        a.batch = ",".join(str(f) for f in cfiles)
        a.steps = min(a.steps, len(cfiles)) if a.steps != 6 else len(cfiles)
    if a.batch:
        # A real batch fixes the token length. The trimul prewarm below fills the in-projection entry at
        # a.n; at any other width the winner uploads it inside its taped verb and those weights get no
        # gradient (a 64-token crop under the 256 default lost 56 in-projection grads).
        import numpy as _np
        lens = {int(_np.load(f)["in/padding_mask"].shape[1]) for f in a.batch.split(",")}
        if len(lens) != 1:
            raise SystemExit(f"--batch files disagree on token length: {sorted(lens)}")
        a.n = lens.pop()
    if a.pcie_meter != "off":
        import pcie_meter as PM   # perf/r4t_perf; the perf/r4t_step caller puts it on sys.path
        PM.install(sites=a.pcie_meter == "sites")
    else:
        PM = None
    mark = (lambda name: PM.phase(name)) if PM else (lambda name: None)
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> prints every thread stack to the log
    os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
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
    import contextlib
    from tt_bio.train import exact_fp32 as EX
    TD = ttnn.float32 if a.arm == "fp32-exact" else None
    arm = contextlib.ExitStack()
    if TD is not None:   # the trimul follows the override; contractions go exact on the host
        arm.enter_context(T.device_dtype_override(ttnn.float32))
        arm.enter_context(EX.exact_fp32())

    dev = T.get_device()
    card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
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
    trunk = E.load_trunk(glue, state, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")   # an unrecorded build for the lever set and dtypes; build() replaces it

    RB = {}

    # R4T_REBUILD_INLINE=1 (r4t-perf R1): a per-step rebuild skips the grad-off forward whose only job is to make
    # the lazy uploads happen under the recording. The recording stays open into the step's own forward instead,
    # and each lazy upload of a leaf-fed tensor is declared a parameter the moment it is made, before the taped
    # winner reads it. The first build and a resume build still run the forward: the unreached/orphan checks
    # after them need every weight to exist.
    inline = env_flag("R4T_REBUILD_INLINE", True)   # default ON since ca9fa7ac4: digest-identical to the lazy-forward rebuild (rb/I4.jsonl)
    pack = env_flag("R4T_DEVICE_PACK", False)   # r4t-perf R2: bfp8 weights packed on the card in the rebuild (Folds.recording device_pack)
    # The trimul in-projection is cached per chunk width on first use, and caches filled in the grad-off
    # candidate pass reach the taped winner as untracked constants (R1 first form: step-2 digest moved, every
    # linear_{a,b}_{p,g} grad dead). `train_in_proj` holds the two sources as registered device tensors and
    # cuts chunks per call, so nothing is cached. Implied by R4T_REBUILD_INLINE.
    in_proj = env_flag("R4T_TRAIN_IN_PROJ", False)   # train_in_proj rounds dW to bf16 (gcmp.txt); not implied

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
                trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")
                if in_proj:
                    RB["trimuls"] = len([m.train_in_proj() for m in trimuls(trunk)])
                ttnn.synchronize_device(dev); _rb2 = time.time()
                one_example(0, False)   # lazy uploads (trimul in-projections at this chunk width), on the taped route
                ttnn.synchronize_device(dev); _rb3 = time.time()
        else:
            rec = folds.recording(device_pack=pack)
            sd = rec.__enter__()
            trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")
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
        import numpy as np
        from tt_bio.train.rfd4_data import load_batch
        from tt_bio.train import rfd4_objective as RO
        from tt_bio.train.bridge import Bridge, host_params
        batches = [(os.path.basename(f),) + load_batch(f) for f in a.batch.split(",")]
        cur = {}

        def use(step):
            nm, b, m, _rec, rl = batches[step if a.curve_dump else step % len(batches)]
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

        @contextlib.contextmanager
        def cand_scope():
            """--cand-levers serving: the candidate pass adds the run-time levers training drops; pair_bf16z also
            carries the pair stream in bf16, a dtype the trunk and its pair updates read from attributes."""
            if a.cand_levers == "training":
                yield
                return
            from tt_bio import ops as _ops
            nogi = _ops.NOGRAD_IS_INFERENCE
            _ops.NOGRAD_IS_INFERENCE = a.cand_levers == "inference" or nogi
            mods = [trunk] + list(trunk.pair_updates.values())
            saved = [(m, m.pair_residual_dtype if m is trunk else m.residual_dtype) for m in mods]
            trunk.pair_residual_dtype = ttnn.bfloat16
            for m in mods[1:]:
                m.residual_dtype = ttnn.bfloat16
            try:
                with T.lever_scope(levers | {"pair_bf16z", "dit_sigf32", "tok_recycle_mm"}):
                    yield
            finally:
                for m, d in saved:
                    setattr(m, "pair_residual_dtype" if m is trunk else "residual_dtype", d)
                _ops.NOGRAD_IS_INFERENCE = nogi

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
                with ag.no_grad(), cand_scope():
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
            ref = {k: float(cur["ref"][k][ctx["win"]]) for k in terms if k in cur["ref"]
                   and ctx["win"] < len(cur["ref"][k])}
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
    ax = None
    if world > 1:
        from tt_bio.train.launcher import ProcessAxis
        ax = ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=a.dp_run)
    use_muon = a.optimizer == "muon" or (a.optimizer == "auto" and bool(a.curve_dump))
    if use_muon:
        # Upstream's step: Muon + Adam, per-group clip 1.0 (rfd4_objective.update_rule). Plain
        # AdamW read 11 % above upstream from step 7 on the first graded leg; that gap was the optimizer.
        from tt_bio.train.rfd4_objective import update_rule
        opt = update_rule(dict(params), data_parallel=ax, shard=a.shard)
    else:
        opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0, data_parallel=ax, shard=a.shard)
    start = 0
    if a.resume:
        start = int(load_adapter(a.resume, dict(params), dev, opt=opt)["step"])
        build()
    ck = Checkpointer(a.out, every=a.save_every or 10**9, keep=a.keep or (2 if a.save_every else 10))
    seen = []   # losses this process logged, for --pin-spike
    if a.exempt == "default":
        for nm, why in UPSTREAM_EXEMPT.items():
            if nm in params:
                cyc[nm] = why
    elif a.exempt:
        for ln in open(a.exempt):
            if ln.strip() and not ln.startswith("#"):
                nm, why = ln.rstrip("\n").split("\t", 1)
                cyc[nm] = why
    cov = GradientCoverage(names, exempt=cyc, where="r4t-step trunk: ",
                           **({"steps": a.cover_steps} if a.cover_steps else {}))
    threading.Thread(target=sample, daemon=True).start()
    clog = None
    if a.curve_dump:
        from tt_bio.train.rfd4_objective import UPDATE_RULE as RO_RULE
        clog = CurveLog(os.path.expanduser(a.curve_out), cplan, "ours",
                        meta={"sha": os.environ.get("TT_BIO_SHA", ""), "card": card,
                              "dump_dir": cdir, "dtype": str(trunk.dtype), **RO_RULE,
                              "levers": sorted(levers) if isinstance(levers, (set, list, tuple))
                              else str(levers), "device": "tenstorrent", "k": a.k, "n": a.n},
                        start=start)
    for step in range(start, a.steps):
        ttnn.synchronize_device(dev)
        base = used()
        c0 = len(clk)
        t0 = time.time()
        mark("fwd")
        extra = {}
        if real:
            use(step * world + rank)
            s, z, ctx = real[0](True)
            close_rec()
            ttnn.synchronize_device(dev)
            t1 = time.time()
            mark("heads_bwd")
            retained = used() - base
            loss, terms, ref = real[1](s, ctx)
            ttnn.synchronize_device(dev)
            extra_mem = used()
            st = s
            extra = dict(dev_after_heads_B=extra_mem, batch=cur["name"], winner=ctx["win"], upstream_winner=(None if a.curve_dump else int(cur["meta"]["winners"][0])), depth_upstream=cur["depth"],
                         terms=terms, upstream_terms=ref)
        else:
            s, z = one_example(step * world + rank, True)
            close_rec()
            ttnn.synchronize_device(dev)
            t1 = time.time()
            mark("heads_bwd")
            retained = used() - base
            st = s if isinstance(s, ag.Tensor) else ag.Tensor(s)
            loss = float((ttnn.to_torch(st.value).float().reshape(Cs.shape) * Cs).sum() / numel)
            ag.backward([st], [seed_s])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        after_bwd = used()
        if a.grad_dump and a.brk:
            walked = {p: t for p, _o, _k, t in T.walk_device_weights(trunk)}
            cut = next(p for p in walked if p.endswith("q_w"))
            folds.sources.pop(id(walked[cut]))
            extra["dropped"] = cut
        mark("fold_bwd")
        idle = folds.tape_backward()   # device-weight gradients through each fold's VJP into the upstream leaves
        grads = {nm: (None if p.grad is None else p.grad.float().numpy()) for nm, p in params.items()}
        ag.release_pins()   # the backward is done; build() released them too, one rebuild later
        dead = zero_or_missing(grads)
        if ax:
            # G2 judges the gradient the optimizer steps on, the rank sum: a leaf that only another
            # rank's example reaches (a feature embedding its batch lacks) is alive. Dead = dead on every rank.
            order = sorted(grads)
            hits = ax.sum_scalars([float(nm not in dead) for nm in order])
            dead = {nm for nm, h in zip(order, hits) if h == 0.0}
        if os.environ.get("R4T_DUMP_G") and step == start:
            import numpy as _np
            _np.savez(os.environ["R4T_DUMP_G"], **{nm.replace(".", "__"): g for nm, g in grads.items()
                                                    if g is not None and "pair_update_layers" in nm})
        if a.grad_dump:
            import numpy as _np
            sm = clk[c0:] or clk[-1:]
            _np.savez(a.grad_dump + ".npz", **{nm: g for nm, g in grads.items() if g is not None})
            meta = dict(extra, loss=loss, arm=a.arm, attn_fp32=a.attn_fp32, levers=sorted(levers) if not isinstance(levers, str) else levers,
                        dtype=str(trunk.dtype), n=n, card=card, aiclk=[min(sm or [0]), max(sm or [0])],
                        exact={"%s/%s" % k: v for k, v in sorted(EX.EXACT_STATS.items())},
                        no_grad=sorted(nm for nm, g in grads.items() if g is None), sha=os.environ.get("TT_BIO_SHA", ""))
            open(a.grad_dump + ".json", "w").write(json.dumps(meta, indent=1, default=str))
            print(json.dumps(dict(grad_dump=a.grad_dump, n_grads=len(grads), loss=loss, aiclk=meta["aiclk"],
                                  exact=meta["exact"])), flush=True)
            stop.set()
            arm.close()
            return 0
        t2b = time.time()
        mark("opt")
        opt.step(replicas={nm: [g] for nm, g in grads.items() if g is not None}) if ax else opt.step()
        for p in params.values():
            p.grad = None
        t2c = time.time()
        mark("rebuild")
        build(lazy=False)
        ttnn.synchronize_device(dev)
        t3 = time.time()
        mark("digest")
        del s, z, st
        md = hashlib.sha256(b"".join(opt.master[k].tobytes() for k in sorted(opt.master))).hexdigest()[:16]
        sm = clk[c0:] or clk[-1:]
        gnorm = math.hypot(opt.last_grad_norm, getattr(opt, "last_grad_norm_muon", 0.0))
        row = dict(step=step + 1, rank=rank, world=world, loss=loss, lr=opt.last_lr, grad_norm=gnorm,
                   fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                   fold_bwd_s=round(t2b - t2, 4), opt_s=round(t2c - t2b, 4), rebuild_s=round(t3 - t2c, 4), rebuild_split=dict(RB), idle_dev_w=len(idle), step_s=round(t3 - t0, 4), retained_B=retained,
                   dead=len(dead), dead_names=sorted(dead)[:20], dev_base_B=base, dev_after_bwd_B=after_bwd, params=len(params), n_elems=int(sum(v.size for v in opt.master.values())),
                   master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])], aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                   k=a.k, depth=a.depth, n=n, lnbw_reach=dict(lnbw.REACH), gate_bw=list(gate_bw.STATS), kernels={k: list(v) for k, v in __import__("tt_bio.taped_ttnn", fromlist=["x"]).KERNEL_STATS.items()}, **extra,
                   **({"pcie": PM.take()} if PM else {}), host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))
        print(json.dumps(row), flush=True)
        if on_step is not None:
            on_step({**row, "s": row["step_s"]})
        if a.save_at and step + 1 == a.save_at:   # before the guard, so a refusal still leaves the resume test its input
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics={"loss": loss}, force=True)))), flush=True)
        if a.pin_spike:
            ref = sorted(seen[-100:])[len(seen[-100:]) // 2] if len(seen) >= 20 else None
            if ref is not None and loss > a.pin_spike * ref:
                print(json.dumps(dict(spike=step, loss=loss, median100=ref, example=cur["name"] if real else None,
                                      pinned=str(ck.pin(step)))), flush=True)
            seen.append(loss)
        due_save, force_save = _checkpoint_due(step + 1, a.steps, a.save_every, a.save_at)
        if due_save:   # --keep: a ring by step, so no metric to rank by
            print(json.dumps(dict(saved=str(ck.save(step + 1, opt, metrics=None if a.keep else {"loss": loss},
                                                    force=force_save)))), flush=True)
        if clog is not None:
            clog.record(step, loss, terms=extra.get("terms"), n_recycle=cur["depth"],
                        seconds=round(t3 - t0, 2), dead=len(dead), example=cur["name"],
                        aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0, master_digest=md)
        # G2 rides along, as on the upstream side: a depth-1 step trains neither cycle projection
        # (state/r4t/rows/accuracy.md), so counting it would refuse ~1 healthy run in 150.
        if not a.resume and (not real or cur["depth"] >= 2):
            cov.observe(dead)
    stop.set()
    if clog is not None:
        clog.close()


def launch(a, argv=None):
    """One process per card, each pinned to its own card and lease, re-running the same entry point; waits for all."""
    import subprocess
    import sys
    from pathlib import Path
    cards = a.dp_cards.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path("/dev/shm") / f"r4t-trunk-dp-{os.getpid()}"
    run.mkdir()
    holder = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:r4t-scale")
    skip = {"--dp-cards", "--out", "--dp-rank", "--dp-run"}
    fwd, it = [], iter(sys.argv[1:] if argv is None else argv)
    for t in it:   # forward every flag but the launcher's own
        if t.split("=")[0] in skip:
            if "=" not in t:
                next(it, None)
            continue
        fwd.append(t)
    entry = [os.path.abspath(sys.argv[0])] if sys.argv[0].endswith(".py") else ["-m", __name__]
    procs = []
    for r, c in enumerate(cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=c, TT_BIO_LEASE_CARDS=c, TT_BIO_LEASE_HOLDER=holder)
        cmd = [sys.executable, *entry, *fwd, "--dp-cards", a.dp_cards, "--dp-rank", str(r),
               "--dp-run", str(run), "--out", str(out / f"rank{r}")]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"rank{r}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    subprocess.run(["rm", "-rf", str(run)])
    print(json.dumps(dict(world=len(cards), cards=cards, rcs=rcs)), flush=True)
    return 0 if not any(rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
