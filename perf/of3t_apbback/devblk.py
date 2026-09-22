#!/usr/bin/env python3
"""ONE of our PairformerLayer on the card, on the capture's own operands, with per-op
host-float64 substitution of its backward.

The frame is the whole point. `block47_boundary.pt` holds block 47's input, the cotangent that
arrived at its output and its own 57 parameter gradients, all float64, all from the same model
run at crop 384. Upstream's own block re-run in float64 on that input with that cotangent
reproduces those gradients BIT-IDENTICALLY on the 16 `attn_pair_bias` and `single_transition`
tensors (`perf/of3t_apbback/BLK47_VALIDATION.json`). So the capture's own gradient is an exact,
self-consistent, frame-matched float64 reference for one block at crop 384, and a substitution
that recovers error against it is recovering real error rather than a frame difference.

SUBSTITUTION. `--sub NAME` replaces one op's backward, and only inside this block's
`AttentionPairBias.__call__` and `Transition.__call__`, with a host float64 computation on the
operands the card itself produced. The mechanism is the one `perf/of3t_bwdaccum/dev_cot.py`
already uses for teacher forcing: the taped output's `node.fn` is replaced after the verb has
run, so the forward is untouched and only the closure changes.

  none      the arm as shipped, with the interception installed but substituting nothing
  nulldev   THE NULL. Same interception, same scope detection, same node.fn replacement, and
            the replacement calls the op's own device closure. It must move nothing.
  ln        the LayerNorm backward
  linear    the four projections' dW, db and dx
  matmul    q@k^T and probs@v
  softmax   the attention softmax
  gate      multiply(o, g, [SIGMOID]) and the sigmoid
  eltwise   add/subtract/multiply/typecast and the shape ops
  all       every taped op inside the two sub-modules
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))

CKPT = "/home/ttuser/of3-weights/of3-p2-155k.pt"
PRE = "pairformer_stack.blocks."
CAP = "/home/ttuser/of3t_gradients/cap/block47_boundary.pt"

GROUPS = {
    "ln": {"layer_norm"},
    "linear": {"linear"},
    "matmul": {"matmul", "experimental.minimal_matmul"},
    "softmax": {"softmax", "softmax_in_place"},
    "gate": {"multiply"},   # multiply(o, g, [SIGMOID]) is one verb, not a separate sigmoid
    "eltwise": {"add", "add_", "subtract", "multiply", "multiply_", "divide", "silu", "relu"},
    "shape": {"concat", "permute", "reshape", "transpose", "unsqueeze"},
}


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=47)
    ap.add_argument("--arm", default="shipped", choices=("shipped", "flipped"))
    ap.add_argument("--sub", default="none")
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--record", default="")
    a = ap.parse_args()
    t0 = time.perf_counter()

    import torch, ttnn
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from tt_bio.train.lora import walked_weights
    import tt_bio.openfold3_trunk as OT
    import tt_bio
    _want = os.path.join(os.getcwd(), "tt_bio")
    if not os.path.dirname(os.path.abspath(tt_bio.__file__)).startswith(_want):
        raise SystemExit("tt_bio resolved to %s, not %s" % (tt_bio.__file__, _want))
    from dev_grad import apb_inverse
    from bijection_device import device_bijection

    ALL = set().union(*GROUPS.values())
    if a.sub == "softmax_devy":
        GROUPS["softmax_devy"] = GROUPS["softmax"]
    want = GROUPS.get(a.sub, set())
    if a.sub == "all":
        want = ALL
    elif a.sub.startswith("all-minus-"):
        # LEAVE-ONE-OUT. `all` tells you the ceiling and a single op tells you what that op is
        # worth alone; only leaving one out tells you whether the rest of the block can reach
        # the ceiling without it.
        drop = a.sub[len("all-minus-"):]
        if drop not in GROUPS:
            raise SystemExit("unknown group to leave out: %r" % drop)
        want = ALL - GROUPS[drop]
    SUBBING = a.sub not in ("none",)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    spy = {}

    class _Stop(Exception):
        pass

    def _spy(*ar, **kw):
        spy["dims"] = list(ar[1:5]); spy["transform_s"] = ar[5]
        spy["kwargs"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
                         for k, v in kw.items()}
        raise _Stop()

    real_pf = OT.Pairformer
    OT.Pairformer = _spy
    try:
        OT.OF3Trunk(sd, ckc)
    except _Stop:
        pass
    finally:
        OT.Pairformer = real_pf
    if "kwargs" not in spy:
        raise SystemExit("the spy never reached Pairformer")
    kw_cfg = dict(spy["kwargs"])
    if a.arm == "flipped":
        kw_cfg["scale_pair_bias"] = True

    cap = torch.load(a.cap, map_location="cpu", weights_only=False)
    s_in, z_in = cap["args"][0], cap["args"][1]
    sm = cap["kwargs"]["single_mask"].to(torch.float64)
    pm = cap["kwargs"]["pair_mask"].to(torch.float64)
    cot_s, cot_z = cap["cot"][0].to(torch.float64), cap["cot"][1].to(torch.float64)
    N = int(z_in.shape[1])

    # block K's weights, renamed to block 0 of a one-block stack
    pk = "%s%d." % (PRE, a.block)
    one = {("%s0." % PRE) + k[len(pk):]: v for k, v in sd.items() if k.startswith(pk)}
    flat = remap_pairformer_stack(one, prefix="pairformer_stack")
    mod = T.Pairformer(1, *spy["dims"], spy["transform_s"], flat, ckc, **kw_cfg)

    s_fp32 = bool(kw_cfg.get("s_fp32_residual", False))
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT, device=dev,
                                    dtype=s_dtype)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9

    # ---- the interception ---------------------------------------------------------------
    SCOPE = {"on": False}
    SEEN, SUBBED, FIRED = [], {"n": 0}, {"n": 0}
    MATCH = {"identity": 0, "shape": 0}
    t2h = lambda v: ttnn.to_torch(v).to(torch.float64)

    def h2t(x, like):
        """Host float64 gradient -> device, in FLOAT32, never in the operand's own dtype.

        Casting the substituted gradient back to the parent's dtype would re-quantize it: a
        weight is bfloat16 on the card, so an exact float64 dW handed back as bf16 is less
        accurate than the shipped path's, which accumulates dW in fp32 (`_taped_linear`'s dW
        rule asks `ttnn.matmul` for `dtype=ttnn.float32` for exactly this reason). Measured:
        with the bf16 cast in place the `linear` and `matmul` substitutions read -115 % and
        -172 % recovered, i.e. an EXACT backward scoring worse than the device's, which is the
        signature of the instrument and not of the arithmetic. `add_grad` checks shape, not
        dtype, and promotes to fp32 on the second contribution anyway.
        """
        return ttnn.from_torch(x.to(torch.float32).reshape([int(d) for d in like.shape]),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)

    HOST = {}

    def _f64(x, box):
        """Device value -> host float64, every tensor marked for autograd.

        `box` collects (underlying ttnn value, host copy) so a parent can be matched back by
        the VALUE it holds. Matching on the `ag.Tensor` object cannot work: the verb wraps a
        raw handle in a fresh `ag.Tensor` on the way in, so the object the tape keeps as a
        parent is not the object the call site passed.
        """
        if isinstance(x, ag.Tensor):
            h = t2h(x.value).requires_grad_(True)
            box.append((x.value, h))
            return h
        if isinstance(x, ttnn.Tensor):
            h = t2h(x).requires_grad_(True)
            box.append((x, h))
            return h
        if isinstance(x, (list, tuple)):
            return type(x)(_f64(v, box) for v in x)
        return x

    def _act(y, acts):
        """ttnn's fused unary activations, on the host. `ttnn.linear` names one as a STRING
        and the eltwise binaries as a list of UnaryOpType, so both spellings are handled; a
        name with no host equivalent raises rather than being dropped."""
        if acts is None:
            return y
        if isinstance(acts, str) or not isinstance(acts, (list, tuple)):
            acts = [acts]
        for a_ in acts:
            if isinstance(a_, (list, tuple)):
                a_ = a_[0]
            nm = str(a_).split(".")[-1].upper()
            if nm == "SIGMOID":
                y = torch.sigmoid(y)
            elif nm in ("SILU", "SWISH"):
                y = torch.nn.functional.silu(y)
            elif nm == "RELU":
                y = torch.relu(y)
            elif nm == "GELU":
                y = torch.nn.functional.gelu(y)
            else:
                raise SystemExit("no host float64 equivalent for fused activation %r" % nm)
        return y

    def _h_linear(A, K):
        y = A[0] @ A[1]
        b = K.get("bias", A[2] if len(A) > 2 else None)
        if b is not None:
            y = y + b
        return _act(y, K.get("activations") or K.get("activation"))

    def _h_layer_norm(A, K):
        x = A[0]
        w = K.get("weight", A[1] if len(A) > 1 else None)
        b = K.get("bias", A[2] if len(A) > 2 else None)
        eps = K.get("epsilon", 1e-5)
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        y = (x - mu) * torch.rsqrt(var + eps)
        if w is not None:
            y = y * w
        if b is not None:
            y = y + b
        return y

    def _h_matmul(A, K):
        a, b = A[0], A[1]
        if K.get("transpose_a"):
            a = a.transpose(-2, -1)
        if K.get("transpose_b"):
            b = b.transpose(-2, -1)
        y = a @ b
        bi = K.get("bias")
        if bi is not None:
            y = y + bi
        return _act(y, K.get("activations") or K.get("activation"))

    def _h_softmax(A, K):
        return torch.softmax(A[0], dim=K.get("dim", -1))

    def _h_binary(op):
        def f(A, K):
            a = _act(A[0], K.get("input_tensor_a_activations"))
            b = _act(A[1], K.get("input_tensor_b_activations"))
            return op(a, b)
        return f

    def _h_unsqueeze(A, K):
        return A[0].unsqueeze(K.get("dim", A[1] if len(A) > 1 else 0))

    def _h_reshape(A, K):
        sh = K.get("shape", A[1] if len(A) > 1 else None)
        sh = [int(d) for d in sh]
        return A[0].reshape(sh)

    def _h_permute(A, K):
        dims = K.get("dims", A[1] if len(A) > 1 else None)
        return A[0].permute([int(d) for d in dims])

    def _h_transpose(A, K):
        d0 = K.get("dim0", A[1] if len(A) > 1 else -2)
        d1 = K.get("dim1", A[2] if len(A) > 2 else -1)
        return A[0].transpose(int(d0), int(d1))

    def _h_concat(A, K):
        return torch.cat(list(A[0]), dim=K.get("dim", A[1] if len(A) > 1 else 0))

    HOST.update({
        "linear": _h_linear, "layer_norm": _h_layer_norm,
        "matmul": _h_matmul, "experimental.minimal_matmul": _h_matmul,
        "softmax": _h_softmax, "softmax_in_place": _h_softmax,
        "multiply": _h_binary(lambda a, b: a * b),
        "multiply_": _h_binary(lambda a, b: a * b),
        "add": _h_binary(lambda a, b: a + b), "add_": _h_binary(lambda a, b: a + b),
        "subtract": _h_binary(lambda a, b: a - b),
        "divide": _h_binary(lambda a, b: a / b),
        "sigmoid": lambda A, K: torch.sigmoid(A[0]),
        "silu": lambda A, K: torch.nn.functional.silu(A[0]),
        "relu": lambda A, K: torch.relu(A[0]),
        "unsqueeze": _h_unsqueeze, "reshape": _h_reshape, "permute": _h_permute,
        "transpose": _h_transpose, "concat": _h_concat,
    })

    def host_bw(name, args, kwargs, parents):
        """A float64 host replay of `name` on the operands the card itself produced.

        The replay is differentiated by torch, so the substituted gradient is the exact float64
        gradient of this op at this point rather than a more precise device kernel. A parent the
        replay cannot reach raises: a substitution that silently did not happen is the one
        failure this row cannot tolerate.
        """
        fn = HOST[name]
        box = []
        A = [_f64(x, box) for x in args]
        K = {k: _f64(v, box) for k, v in kwargs.items()}
        hs, hp = [], []
        used = set()
        for pnt in parents:
            hit = None
            for i, (v, h) in enumerate(box):
                if v is pnt.value and i not in used:
                    hit, ix = h, i
                    break
            if hit is None:
                # A verb may migrate an operand to DRAM on the way in, which REPLACES the
                # `ag.Tensor`'s `.value` object, so identity can miss where the operand is the
                # same numbers in a new handle. Shape and dtype disambiguate here; the count of
                # each kind is reported so a run cannot quietly be all fallback.
                for i, (v, h) in enumerate(box):
                    if i in used:
                        continue
                    if tuple(int(d) for d in v.shape) == tuple(int(d) for d in pnt.value.shape) \
                            and v.dtype == pnt.value.dtype:
                        hit, ix = h, i
                        MATCH["shape"] += 1
                        break
            else:
                MATCH["identity"] += 1
            if hit is None:
                raise SystemExit("host replay of %r cannot reach one of its taped parents "
                                 "(shape %s dtype %s); box holds %s. The substitution would be "
                                 "partial" % (name, [int(d) for d in pnt.value.shape],
                                              pnt.value.dtype,
                                              [([int(d) for d in v.shape], str(v.dtype))
                                               for v, _ in box]))
            used.add(ix)
            hs.append(hit)
            hp.append(pnt)
        y = fn(A, K)

        def bw(g):
            FIRED["n"] += 1
            gh = t2h(g).reshape(y.shape)
            gs = torch.autograd.grad(y, hs, gh, allow_unused=True, retain_graph=False)
            for pnt, gp in zip(hp, gs):
                if gp is not None and pnt.requires_grad:
                    pnt.add_grad(h2t(gp, pnt.value))
        return bw

    # A verb can build MORE THAN ONE tape node. `_taped_linear` with a fused activation runs
    # the linear unactivated, tapes it, then applies the activation as a second taped op and
    # returns that -- so the node the call site gets back is the ACTIVATION's, and patching
    # only that one would leave the linear's own backward on the device while claiming it had
    # been substituted. Hooking `_tape` collects every node a call creates, in order, and each
    # is given the replay that belongs to it.
    NEW = []
    _real_tape = ag._tape

    def _tape_hook(out_value, parents, make_fn, *ar, **kwv):
        t = _real_tape(out_value, parents, make_fn, *ar, **kwv)
        if getattr(t, "node", None) is not None:
            NEW.append(t)
        return t

    def _wrap_verb(name, real):
        def wrapper(shipped, args, kwargs):
            mark = len(NEW)
            try:
                out = real(shipped, args, kwargs)
                if not SCOPE["on"]:
                    return out
                created = list(NEW[mark:])
                if not created:
                    return out
                SEEN.append({"op": name, "nodes": len(created),
                             "parents": [[int(d) for d in p.value.shape]
                                         for p in created[0].node.parents],
                             "out": [int(d) for d in created[-1].value.shape],
                             "kwargs": sorted(kwargs)})
                if a.sub == "nulldev":
                    for t in created:
                        orig = t.node.fn

                        def same(g, orig=orig):
                            FIRED["n"] += 1
                            return orig(g)

                        t.node.fn = same
                        SUBBED["n"] += 1
                    return out
                if name not in want:
                    return out
                if name not in HOST:
                    raise SystemExit("no host float64 replay for taped verb %r; the substitution "
                                     "would silently not happen" % name)
                if a.sub == "softmax_devy":
                    # The softmax substitution above recomputes y in float64 from the logits,
                    # so it replaces the backward's ARITHMETIC and its OPERAND at once. This
                    # arm separates them: exact float64 backward, on the card's OWN bf16
                    # softmax output. `of3t-apbgrad` proved the operand is what carries the
                    # error and `TT_BIO_SOFTMAX_BW_RENORM` repairs its row sum; what is left
                    # after that repair is what this arm prices.
                    t = created[0]
                    yv = t.value
                    dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
                    xp = t.node.parents[0]

                    def bw_devy(g, yv=yv, dim=dim, xp=xp):
                        FIRED["n"] += 1
                        y = t2h(yv)
                        gh = t2h(g).reshape(y.shape)
                        inner = (gh * y).sum(dim=dim, keepdim=True)
                        rs = y.sum(dim=dim, keepdim=True)
                        dx = y * (gh - inner / rs)
                        if xp.requires_grad:
                            xp.add_grad(h2t(dx, xp.value))

                    t.node.fn = bw_devy
                    SUBBED["n"] += 1
                    return out
                act = kwargs.get("activation")
                core_kwargs = {k: v for k, v in kwargs.items() if k != "activation"}
                for i, t in enumerate(created):
                    if i == 0:
                        t.node.fn = host_bw(name, args, core_kwargs, t.node.parents)
                    else:
                        # the composed activation node: one parent, the pre-activation value
                        t.node.fn = host_bw(_ACT_VERB(act), [t.node.parents[0]], {},
                                            t.node.parents)
                    SUBBED["n"] += 1
                return out
            finally:
                # The hook must not RETAIN what it sees. Holding a reference to every taped
                # tensor keeps the whole block's activations alive through the checkpointed
                # recompute, and the block stops fitting: the first run died in
                # program.cpp:1052 with statically allocated circular buffers clashing with L1
                # buffers, which is the tape's own lifetime rule being defeated by the
                # instrument. The nodes stay reachable through the tape; only this list lets go.
                del NEW[mark:]

        return wrapper

    def _ACT_VERB(act):
        nm = str(act).split(".")[-1].lower()
        if nm in HOST:
            return nm
        raise SystemExit("no host float64 replay for the composed activation %r" % act)

    if SUBBING or a.record:
        ag._tape = _tape_hook
        if hasattr(tt, "_tape"):
            tt._tape = _tape_hook
        for nm in list(tt._VERBS):
            tt._VERBS[nm] = _wrap_verb(nm, tt._VERBS[nm])
        ag._TAPED = getattr(ag, "_TAPED", {})
        for nm in ("layer_norm", "linear"):
            if nm in tt._VERBS:
                ag._TAPED[nm] = tt._VERBS[nm]

    _apb = T.AttentionPairBias.__call__
    _tr = T.Transition.__call__

    def _scoped(real):
        def call(self, *ar, **kwv):
            prev = SCOPE["on"]
            SCOPE["on"] = True
            try:
                return real(self, *ar, **kwv)
            finally:
                SCOPE["on"] = prev
        return call

    T.AttentionPairBias.__call__ = _scoped(_apb)
    T.Transition.__call__ = _scoped(_tr)

    before = device_weights(mod)
    ours = walked_weights(lambda: mod(fts(s_in), ft(z_in), ft(pm), ft(attn), ft(attn)), None, mod)

    sa = ag.Tensor(fts(s_in), requires_grad=True)
    za = ag.Tensor(ft(z_in), requires_grad=True)
    with ag.tape():
        s_out, z_out = mod(sa, za, ft(pm), ft(attn), ft(attn))
    s_ours = ttnn.to_torch(s_out.value).to(torch.float64)
    ag.backward([s_out, z_out], [ft(cot_s), ft(cot_z)])

    # ---- placement, block 0 of the one-block stack back onto their names ------------------
    dev_all = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod).items()}
    atoms = {k[len("pairformer_stack.blocks.0."):]: v.detach().to(torch.float32)
             for k, v in one.items()}
    dev_j = {k[len("blocks.0."):]: v for k, v in dev_all.items() if k.startswith("blocks.0.")}
    bj = device_bijection(dev_j, atoms)
    grads = {n: (ttnn.to_torch(l.grad).to(torch.float64) if l.grad is not None else None)
             for n, l in ours.items()}
    apb = mod.blocks[0].attention_pair_bias
    H, d = int(apb.n_heads), int(apb.head_dim)
    D = int(getattr(apb, "padded_head_dim", d))
    c_s = int(atoms["attn_pair_bias.layer_norm_a.weight"].shape[0])
    FUSED = ["attn_pair_bias.mha.linear_q.weight", "attn_pair_bias.mha.linear_k.weight",
             "attn_pair_bias.mha.linear_v.weight", "attn_pair_bias.mha.linear_q.bias"]
    out_g, absent = {}, []
    for key in sorted(atoms):
        if key in bj["placements"]:
            chosen = None
            for pl in bj["placements"][key]:
                gd = grads.get("blocks.0." + pl["device_path"])
                if gd is not None:
                    chosen = (pl, gd); break
            if chosen is None:
                absent.append(key); continue
            pl, gd = chosen
            band = gd.narrow(pl["axis"], pl["start"], pl["length"])
            out_g[key] = (band.T if pl["layout"].endswith("transposed") else band).contiguous()
        elif key in FUSED:
            src = grads.get("blocks.0.attention_pair_bias." +
                            ("qkv_bias" if key.endswith("bias") else "qkv_weight"))
            if src is None:
                absent.append(key); continue
            out_g[key] = apb_inverse(src, key, H, d, D, c_s)
        elif key == "attn_pair_bias.linear_z.weight":
            src = grads.get("blocks.0.attention_pair_bias.z_weight")
            if src is None:
                absent.append(key); continue
            _ones = ttnn.from_torch(torch.ones(32, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                                    dtype=apb.z_weight.dtype)
            eff = float(ttnn.to_torch(ttnn.multiply_(_ones, float(apb._bias_scale)))
                        .to(torch.float64)[0, 0])
            out_g[key] = (src * eff).t().contiguous()
        else:
            absent.append(key)

    torch.save({"grads": out_g, "sub": a.sub, "arm": a.arm, "block": a.block,
                "s_out": s_ours, "config": kw_cfg}, a.out)
    rep = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename, "card": 0,
           "board": "p300c Blackhole", "block": a.block, "arm": a.arm, "sub": a.sub,
           "config": kw_cfg, "cap": a.cap, "cap_sha256": sha256_file(a.cap),
           "taped_ops_in_scope": len(SEEN), "substituted": SUBBED["n"],
           "substituted_backwards_fired": FIRED["n"],
           "operand_match": dict(MATCH),
           "placed": len(out_g), "absent": absent,
           "probe": {"s_in_norm": float(s_in.to(torch.float64).norm()),
                     "z_in_norm": float(z_in.to(torch.float64).norm()),
                     "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm()),
                     "s_out_norm": float(s_ours.norm())},
           "seconds": round(time.perf_counter() - t0, 1)}
    if a.record:
        seen_agg = {}
        for e in SEEN:
            seen_agg.setdefault(e["op"], {"n": 0, "example": e})["n"] += 1
        rep["ops_in_scope"] = {k: v["n"] for k, v in sorted(seen_agg.items())}
        json.dump({"ops": rep["ops_in_scope"], "detail": SEEN}, open(a.record, "w"), indent=2)
    json.dump(rep, open(a.report, "w"), indent=2)
    print(json.dumps({"sub": a.sub, "ops_in_scope": len(SEEN), "substituted": SUBBED["n"],
                      "fired": FIRED["n"], "placed": len(out_g), "absent": len(absent),
                      "s_out_norm": rep["probe"]["s_out_norm"],
                      "seconds": rep["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
