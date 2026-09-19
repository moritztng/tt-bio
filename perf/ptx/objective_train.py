#!/usr/bin/env python3
"""Does the COMPOSED objective train? Memorise one example from random init, on device.

`objective_check.py` proves the composition's value and gradient are right on the host.
That is not the same as proving it trains: the seeds still have to reach seven different
device outputs, one `backward` has to walk the union of their ancestors without replaying a
shared node twice, and the optimizer has to move weights the forward reads. ptxft's
distogram-only run is the thing to compare against -- it reached 0.00019 against a uniform
4.15888, 0.005 %, in 600 steps -- so `--terms distogram` runs THE SAME harness with seven
terms switched off and both numbers come out of one command.

WHAT THIS IS NOT. The trunk here is a small taped MLP, not Protenix's pairformer, because
the shipped fast pairformer routes 0 of its 29 linear/matmul/layer_norm calls through
`tt_bio.ops` and is therefore not differentiable today -- that is `ptx-fastpath`'s blocker,
not this row's. So this measures the objective, the tape's multi-root backward and the
optimizer; it does not measure whether the Protenix trunk trains. Every head IS the real
head shape the eight terms consume, and every number below comes from
`objectives.objective("af3")`, the shipped composition, called exactly as
`recipes.lora_finetune` calls it.

THE FREE CHECK, kept from ptxft: the distogram head is initialised to zeros, which is
upstream's own initialisation (`protenix/model/modules/head.py:40`). A zero head makes
every logit zero, so step 0 must read a uniform ln(64) on the distogram term. Not ln(64)
exactly: upstream divides by eps + sum(mask) with eps = 1e-6 (loss.py:626), so at M masked
pairs the value is ln(64) * M / (M + 1e-6), which at M = 1024 is 4.06e-09 below ln(64). The
check is against that expression, to 1e-12 relative -- rounding the eps away here would
hide exactly the class of bug the eps placement already caused once in this codebase.
Any other starting value means the head, the symmetrisation or the loss disagrees with
upstream, and it costs nothing to check.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

UNIFORM_CE = math.log(64.0)
AICLK = "/sys/class/tenstorrent/tenstorrent!{}/tt_aiclk"


def aiclk(card):
    try:
        return int(open(AICLK.format(card)).read().strip())
    except OSError:
        return 0


def target(a):
    """The one example to memorise: a fixed structure, and fixed features to predict it from.

    The features are random and frozen. That is the point of a memorisation test -- there is
    no generalisation available, so the only path to a low loss is that every gradient is
    right.
    """
    rng = np.random.default_rng(a.seed)
    N = a.tokens
    true = rng.normal(0, 8, (N, 3))
    tdist = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    cmask = np.ones(N, bool)
    dmask = np.ones((N, N))
    from tt_bio.train import losses as X
    batch = {
        "true_xyz": true, "coord_mask": cmask, "true_dist": tdist,
        "lddt_pair_mask": X.lddt_mask(tdist, dmask, np.zeros(N, bool)),
        "bond_mask": (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) == 1).astype(float),
        "frame_atom_index": np.stack([(np.arange(N) - 1) % N, np.arange(N),
                                      (np.arange(N) + 1) % N], 1),
    }
    feats = {"s0": rng.normal(0, 1, (N, a.channels)),
             "z0": rng.normal(0, 1, (N, N, a.channels))}
    return batch, feats, rng


class Model:
    """A taped MLP with the seven real head shapes. Every parameter is trainable.

    No LoRA: an adapter beside a random weight is adapting noise, which is ptxft's own
    reasoning for running the memorisation full-parameter.
    """

    def __init__(self, a, feats, device, rng):
        import ttnn
        from tt_bio import autograd as ag
        from tt_bio.train.tensors import to_device
        self.ag, self.ttnn, self.a = ag, ttnn, a
        N, C = a.tokens, a.channels
        self.dev = lambda x, g=False: ag.Tensor(
            to_device(np.ascontiguousarray(x, np.float32), device), requires_grad=g)
        # Glorot on fan_in, which is all a memorisation harness needs; the Protenix-faithful
        # `trunc_normal_init_` belongs with the Protenix trunk, and this is not it.
        g = lambda *s: rng.normal(0, 1.0 / math.sqrt(s[0]), s)
        self.p = {}
        def par(name, arr):
            self.p[name] = self.dev(arr, True)
            return self.p[name]
        self.s0, self.z0 = self.dev(feats["s0"]), self.dev(feats["z0"])
        self.blocks = []
        for i in range(a.depth):
            self.blocks.append({
                "sw1": par(f"s{i}.w1", g(C, C)), "sb1": par(f"s{i}.b1", np.zeros(C)),
                "sw2": par(f"s{i}.w2", g(C, C)), "sb2": par(f"s{i}.b2", np.zeros(C)),
                "zw1": par(f"z{i}.w1", g(C, C)), "zb1": par(f"z{i}.b1", np.zeros(C)),
                "zw2": par(f"z{i}.w2", g(C, C)), "zb2": par(f"z{i}.b2", np.zeros(C)),
            })
        # head.py:40 -- the distogram head is zeros, which is what puts step 0 at ln(64).
        par("head.disto.w", np.zeros((C, 64)))
        par("head.disto.b", np.zeros(64))
        for nm, out in (("pde", 64), ("pae", 64)):
            par(f"head.{nm}.w", g(C, out))
            par(f"head.{nm}.b", np.zeros(out))
        for nm, out in (("xyz", 3), ("plddt", 50), ("resolved", 2)):
            par(f"head.{nm}.w", g(C, out))
            par(f"head.{nm}.b", np.zeros(out))
        par("head.dist.a", g(C, C))
        par("head.dist.b", g(C, C))

    def __call__(self):
        ag, a = self.ag, self.a
        N = a.tokens
        s, z = self.s0, self.z0
        for b in self.blocks:
            s = ag.add(s, ag.linear(ag.relu(ag.linear(s, b["sw1"], b["sb1"])),
                                    b["sw2"], b["sb2"]))
            z = ag.add(z, ag.linear(ag.relu(ag.linear(z, b["zw1"], b["zb1"])),
                                    b["zw2"], b["zb2"]))
        P = self.p
        head = lambda x, nm, shape: ag.reshape(
            ag.linear(x, P[f"head.{nm}.w"], P[f"head.{nm}.b"]), shape)
        # A pair-shaped distance from the token track: matmul with transpose_b lands
        # [N, N] directly, so nothing here slices a sub-tile last axis.
        dist = ag.reshape(ag.matmul(ag.linear(s, P["head.dist.a"]),
                                    ag.linear(s, P["head.dist.b"]), transpose_b=True),
                          [1, N, N])
        return {
            "distogram_logits": ag.reshape(
                ag.linear(z, P["head.disto.w"], P["head.disto.b"]), [N, N, 64]),
            "pde_logits": head(z, "pde", [1, N, N, 64]),
            "pae_logits": head(z, "pae", [1, N, N, 64]),
            "pred_dist": dist,
            "pred_xyz": head(s, "xyz", [1, N, 3]),
            "plddt_logits": head(s, "plddt", [1, N, 50]),
            "resolved_logits": head(s, "resolved", [1, N, 2]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50,
                    help="steps of Protenix's own linear warmup (optim.af3_lr). Adam's first "
                         "update is ~lr per element whatever the gradient is")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--stage", default="finetune", choices=("pretrain", "finetune"),
                    help="finetune is the stage where all eight terms carry weight")
    ap.add_argument("--terms", default="all",
                    help="'all', or a comma-separated subset; the rest get weight 0. "
                         "'distogram' reproduces ptxft's distogram-only run in this harness")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.train import losses as X, objectives
    from tt_bio.train.optim import AdamW, af3_lr
    from tt_bio.train.tensors import to_device, to_host

    weights = dict(X.LOSS_WEIGHTS[a.stage])
    if a.terms != "all":
        keep = set(a.terms.split(","))
        unknown = keep - set(weights)
        if unknown:
            raise SystemExit(f"no such term(s): {sorted(unknown)}; terms are {sorted(weights)}")
        weights = {k: (v if k in keep else 0.0) for k, v in weights.items()}
    live = [k for k, v in weights.items() if v != 0.0]

    batch, feats, rng = target(a)
    device = ttnn.open_device(device_id=0)
    clocks = []
    try:
        ag.install()
        model = Model(a, feats, device, rng)
        row = objectives.objective("af3")
        opt = AdamW(model.p, lr=a.lr, clip_norm=10.0,
                    schedule=lambda st: af3_lr(st, a.lr, warmup_steps=a.warmup))
        print(f"# objective_train: memorise one {a.tokens}-token example from random init, "
              f"on the COMPOSED objective")
        print(f"# stage {a.stage}, live terms {len(live)} of 8: {', '.join(sorted(live))}")
        print(f"# depth {a.depth}, channels {a.channels}, lr {a.lr}, "
              f"{sum(int(np.prod(v.value.shape)) for v in model.p.values()):,} trainable "
              f"elements in {len(model.p)} tensors, seed {a.seed}, card {a.card}")
        print(f"# uniform distogram cross-entropy = ln(64) = {UNIFORM_CE:.15g}")
        hist, first, t0 = [], None, time.time()
        for step in range(a.steps):
            opt.zero_grad()
            out = model()
            host = {k: to_host(v.value).astype(np.float64) for k, v in out.items()}
            # Upstream rebuilds the pLDDT label from the CURRENT prediction under no_grad
            # every step (loss.py:1758-1766), so it is rebuilt here every step too. The
            # label moving is upstream's behaviour, not drift in the harness.
            ld, lw = X.atom_bespoke_lddt(host["pred_xyz"], batch["true_xyz"],
                                         np.zeros(a.tokens, bool), np.ones(a.tokens, bool),
                                         np.ones(a.tokens, bool))
            b = dict(batch, per_atom_lddt=ld, per_atom_weight=lw,
                     edm_scale=X.edm_scale(X.sample_noise_level(rng, (1,))))
            total, breakdown, seeds = row(b, host, weights=weights)
            # Lifted from recipes.lora_finetune: ONE backward over the union of the seeded
            # roots' ancestors, never one call per root.
            ag.backward([out[k] for k in seeds],
                        [to_device(np.ascontiguousarray(g, np.float32), device)
                         for g in seeds.values()])
            opt.step()
            clocks.append(aiclk(a.card))
            rec = {"step": step, "loss": total,
                   "terms": {k: v["value"] for k, v in breakdown.items()
                             if v["value"] is not None}}
            hist.append(rec)
            if first is None:
                first = rec
                dg = rec["terms"].get("distogram")
                if dg is not None:
                    M = float(np.sum(np.outer(batch["coord_mask"], batch["coord_mask"])))
                    want = UNIFORM_CE * M / (M + 1e-6)
                    r = abs(dg - want) / want
                    print(f"# zero-init check: step 0 distogram {dg:.15g}, expected "
                          f"ln(64)*M/(M+1e-6) at M={M:.0f} masked pairs = {want:.15g}, "
                          f"rel {r:.2e} (ln(64) itself is {UNIFORM_CE:.15g})"
                          f"{'' if r < 1e-12 else '   <-- ZERO-INIT CHECK FAILED'}")
            if step % a.every == 0 or step == a.steps - 1:
                print(f"step {step:>5}  loss {total:>12.6g}  " +
                      "  ".join(f"{k} {rec['terms'][k]:.4g}" for k in sorted(rec["terms"])))
        dt = time.time() - t0
    finally:
        ag.uninstall()
        ttnn.close_device(device)

    last = hist[-1]
    # The tail median, not the last step. mse and bond carry the EDM per-sample weight
    # (sigma^2 + 16^2) / (16 sigma)^2, and upstream RESAMPLES sigma every step
    # (generator.py:49-60), so one unlucky small sigma multiplies those two terms by
    # hundreds. That is the objective behaving as upstream specifies it, and it makes a
    # single final reading a coin flip rather than a result. Both are printed.
    tail = hist[max(1, int(0.9 * len(hist))):] or hist[-1:]
    med = lambda f: statistics.median([f(r) for r in tail])
    print(f"\n# {a.steps} steps in {dt:.1f} s ({1e3 * dt / a.steps:.0f} ms/step), AICLK "
          f"sampled DURING: min {min(clocks)} median {int(statistics.median(clocks))} "
          f"max {max(clocks)} MHz, card {a.card} (qb1, Blackhole p150a)")
    print(f"# 'final' is the median over the last {len(tail)} steps; 'last step' is the "
          f"single step {last['step']} reading")
    print(f"{'term':<14} {'step 0':>14} {'final':>14} {'ratio':>10} {'last step':>14}")
    finals = {}
    for k in sorted(first["terms"]):
        f0 = first["terms"][k]
        f1 = med(lambda r: r["terms"][k])
        finals[k] = f1
        print(f"{k:<14} {f0:>14.6g} {f1:>14.6g} "
              f"{(f0 / f1 if f1 else float('inf')):>10.1f}x {last['terms'][k]:>14.6g}")
    t1 = med(lambda r: r["loss"])
    print(f"{'TOTAL':<14} {first['loss']:>14.6g} {t1:>14.6g} "
          f"{(first['loss'] / t1 if t1 else float('inf')):>10.1f}x {last['loss']:>14.6g}")
    if "distogram" in finals:
        pct = 100.0 * finals["distogram"] / UNIFORM_CE
        print(f"\ndistogram: {finals['distogram']:.6g} against a uniform "
              f"{UNIFORM_CE:.6g} = {pct:.4f} % of uniform "
              f"(ptxft's distogram-only run: 0.00019, 0.005 %)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "weights": weights, "history": hist,
                       "seconds": dt, "aiclk": clocks}, f)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
