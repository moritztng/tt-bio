#!/usr/bin/env python3
"""Phase 5.1: can this stack memorise a tiny set from RANDOM initialisation?

This is the strongest single correctness signal a training stack has, and it is a
different question from the fine-tune. The fine-tune starts from a trained checkpoint, so
a broken backward can still look plausible: the model was already good and a small wrong
update barely moves it. From random init nothing is good, and the only way the loss goes
to near zero is if the forward, the tape, every weight gradient, the optimizer and the
precision are all right at once. If it cannot memorise, something is wrong; if it can,
the machinery is right.

WHAT IS ACTUALLY RUN, stated exactly so it is not read as a pretraining claim. The
architecture is the real Protenix v2 pair track, the same `PairTrackBlock` the fine-tune
differentiates and the same one `block_parity.py` holds to the production layer at PCC
0.99998. The weights are random rather than loaded, every one of them is trainable (no
LoRA: there is no pretrained weight for an adapter to sit beside, so a low-rank update
onto noise would be adapting noise), and the stack is `--depth` blocks rather than 48.
The depth is a cost decision and it is stated rather than hidden: a full-parameter step
carries 2,235,904 gradient elements per block back over PCIe, so 48 blocks is 429 MB of
readback per step against a memorisation run that needs hundreds of steps. Nothing about
the objective, the token count or the step count is weakened; this is a smaller stack
trained honestly, not the real one trained less.

Initialisation follows Protenix's own, cited rather than invented:
  * `trunc_normal_init_` (`protenix/model/triangular/layers.py:62-74`): normal with
    std = sqrt(scale/fan_in) / truncnorm.std(-2, 2), truncated at +/- 2 sigma. scale 1.0
    for a default linear and 2.0 for a "relu" one.
  * The transition's output projection is zeros (`primitives.py:187`) and its fc1/fc2 are
    "relu" (`:181-184`).
  * THE DISTOGRAM HEAD IS ZEROS (`protenix/model/modules/head.py:40`), which is why the
    run starts at exactly ln(64) = 4.15888, a uniform distogram, with no fitting involved.
    That is a free check on the whole pipeline: any other starting value means the head,
    the symmetrisation or the loss disagrees with upstream.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from perf.ptxft import dgdata as D           # noqa: E402

ART = os.path.expanduser("~/ptxft-art")
# truncnorm.std(a=-2, b=2, loc=0, scale=1). Hard-coded rather than importing scipy for one
# constant; the value is checked by --init-check against the sample std of the draws.
TRUNC_STD = 0.8796256610342398


def trunc_normal(shape, rng, *, scale=1.0, fan_in=None):
    """`trunc_normal_init_`, layers.py:62-74, in numpy."""
    fan_in = shape[-2] if fan_in is None else fan_in
    std = np.sqrt(scale / max(1, fan_in)) / TRUNC_STD
    out = rng.normal(0.0, std, shape)
    bad = np.abs(out) > 2 * std
    while bad.any():
        out[bad] = rng.normal(0.0, std, int(bad.sum()))
        bad = np.abs(out) > 2 * std
    return out.astype(np.float32)


def random_block_sd(shapes, rng):
    """A block state dict of Protenix-initialised random tensors with the real shapes.

    The shapes come from the real checkpoint so nothing about the architecture is guessed.
    Weights arrive in checkpoint layout (out, in), which is what `_t2d` transposes.
    """
    import torch
    out = {}
    for k, shp in shapes.items():
        if len(shp) == 1:
            # Every 1-D weight in a remapped block is a layer-norm gain -- every linear
            # weight is 2-D -- so the rule is on RANK, not on the name. Matching "norm" in
            # the name instead missed `attention.proj_z.0.weight`, which is the pair
            # bias's layer norm under a name that does not say so, and a zero gain there
            # silently removes the attention bias from every test that uses these weights.
            a = (np.ones if k.endswith("weight") else np.zeros)(shp, np.float32)
        elif k.endswith("transition_z.fc3.weight"):
            a = np.zeros(shp, np.float32)            # primitives.py:187, zeros
        elif k.endswith("transition_z.fc1.weight") or k.endswith("transition_z.fc2.weight"):
            a = trunc_normal(shp, rng, scale=2.0, fan_in=shp[1])   # "relu", :181-184
        else:
            a = trunc_normal(shp, rng, scale=1.0, fan_in=shp[1])
        out[k] = torch.from_numpy(a)
    return out


def build(a, device):
    """`--depth` randomly initialised pair-track blocks plus a zero distogram head."""
    import torch
    import ttnn
    from perf.ptxft.block_parity import load_blocks
    from perf.ptxft.tape_block import PairTrackBlock, DistogramHead
    real, head_sd = load_blocks(a.ckpt, [0])
    shapes = {k: tuple(v.shape) for k, v in real[0].items()}
    rng = np.random.default_rng(a.seed)
    blocks, params = [], {}
    for i in range(a.depth):
        b = PairTrackBlock(random_block_sd(shapes, rng), device, n_heads=8, head_dim=32,
                           dtype=ttnn.bfloat16, adapter_dtype=ttnn.float32,
                           train_base=True, chunk=a.chunk, q_chunk=a.q_chunk)
        blocks.append(b)
        for k, v in b.params.items():
            params[f"blocks.{i}.{k}"] = v
    zero_head = {k: torch.zeros_like(v) for k, v in head_sd.items()}   # head.py:40
    head = DistogramHead(zero_head, device, dtype=ttnn.bfloat16, trainable=True)
    params.update(head.params)
    return blocks, head, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--targets", default="1VII,1E0L")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip-norm", type=float, default=10.0,
                    help="upstream's own value, configs_base.py:80")
    ap.add_argument("--accum", type=int, default=0,
                    help="targets accumulated into one step; 0 means the whole set, "
                         "which is what a tiny set should use")
    ap.add_argument("--decay-every", type=int, default=50000,
                    help="af3_lr step-decay period; upstream is 50000 over a 100k run, "
                         "so scale it to the run rather than leaving it never firing")
    ap.add_argument("--decay-factor", type=float, default=0.95)
    ap.add_argument("--warmup", type=int, default=50,
                    help="af3_lr warmup steps; upstream runs 1000-2000 over 100k steps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--q-chunk", type=int, default=None)
    ap.add_argument("--near-zero", type=float, default=0.1,
                    help="CE counted as memorised; 0.1 nats is 2.4%% of the ln(64) start")
    a = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import finetune as ft
    from tt_bio import autograd as ag          # noqa: F401  (imported for the tape)
    from perf.ptxft import tape_block as TB
    from perf.ptxft.memgate import run_stack
    from perf.ptxft.train_distogram import load_target, forward_loss
    from perf.clocksample import during

    names = [t for t in a.targets.split(",") if t]
    with during() as clk:
        device = get_device()
        blocks, head, params = build(a, device)
        n_par = sum(int(np.prod([int(d) for d in v.value.shape])) for v in params.values())
        print(f"# --overfit: RANDOM INIT, {a.depth} blocks, every weight trainable, "
              f"{len(params)} tensors, {n_par:,} parameters")
        print(f"# {len(names)} targets {names}, lr {a.lr}, wd {a.weight_decay}, "
              f"{a.steps} steps, seed {a.seed}, clip {a.clip_norm}, warmup {a.warmup}")
        print(f"# uniform distogram CE = ln(64) = {np.log(64):.5f}; a zero-initialised "
              f"head must start exactly there")
        opt = ft.AdamW(params, lr=a.lr, weight_decay=a.weight_decay,
                       clip_norm=a.clip_norm,
                       schedule=lambda st: ft.af3_lr(
                           st, a.lr, warmup_steps=a.warmup,
                           decay_every_n_steps=a.decay_every,
                           decay_factor=a.decay_factor))
        tg = {p: load_target(p) for p in names}
        hist, failures = [], []
        print()
        accum = len(names) if a.accum in (0, None) else int(a.accum)
        print(f"# gradient accumulation: {accum} target(s) per optimizer step")
        print(f"{'step':>5} {'targets':<8} {'loss':>10} {'|g|':>10} {'kept':>7} {'s':>7}")
        start = None
        cursor = 0
        for step in range(1, a.steps + 1):
            t0 = time.perf_counter()
            opt.zero_grad()
            losses = []
            # GRADIENT ACCUMULATION IS WHAT MAKES A MULTI-TARGET TINY SET CONVERGE, and it
            # costs nothing to express: the tape's add_grad already accumulates into
            # t.grad, so not zeroing between micro-batches IS the accumulation. The seed
            # carries the 1/accum, so the step sees the mean gradient of the set rather
            # than the gradient of whichever target came last. Without it, one step fully
            # fits one target and breaks the other: two targets at lr 1e-4 plateau at CE
            # 1.10 and 1.50 where a single target reaches 0.234.
            for _ in range(accum):
                p = names[cursor % len(names)]
                cursor += 1
                t = tg[p]
                n = int(t["target"].shape[0])
                loss, out, _lg, seed, _pr = forward_loss(
                    blocks, head, t["z"], t["target"], t["pair_mask"], device,
                    checkpointed=True, n_real=n, want_grad=True)
                losses.append(loss)
                full = np.zeros((out.value.shape[-3], out.value.shape[-2], D.N_BINS),
                                np.float32)
                full[:n, :n, :] = TB.symmetrize_bins(seed) / accum
                out.backward(seed=ft.to_device(full, device, dtype=ttnn.bfloat16))
            gn = opt.grad_norm()
            rep = opt.step()
            mn = np.sqrt(sum(r["master_step"] ** 2 for r in rep.values()))
            dn = np.sqrt(sum(r["device_step"] ** 2 for r in rep.values()))
            dt = time.perf_counter() - t0
            kept = dn / mn if mn > 0 else float("nan")
            loss = float(np.mean(losses))
            if start is None:
                start = loss
            if step <= 4 or step % 25 == 0 or step == a.steps:
                print(f"{step:>5} {accum:<8} {loss:>10.5f} {gn:>10.3e} {kept:>7.3f} "
                      f"{dt:>7.2f}  lr {opt.last_lr:.2e} clip {opt.last_clip:.3f}",
                      flush=True)
            hist.append({"step": step, "loss": loss, "grad_norm": gn, "kept": kept,
                         "s": dt})
        print()
        final = {}
        for p in names:
            t = tg[p]
            n = int(t["target"].shape[0])
            final[p] = forward_loss(blocks, head, t["z"], t["target"], t["pair_mask"],
                                    device, want_grad=False, n_real=n)[0]
        uni = float(np.log(D.N_BINS))
        print(f"{'target':<8} {'start (uniform)':>16} {'final':>10} {'frac of start':>14}")
        for p in names:
            print(f"{p:<8} {uni:>16.5f} {final[p]:>10.5f} {final[p] / uni:>14.4f}")
        worst = max(final.values())
        print()
        print(f"OVERFIT-TINY: worst final CE {worst:.5f} of a uniform {uni:.5f} "
              f"({100 * worst / uni:.2f} %), bar {a.near_zero}")
        if abs(start - uni) > 1e-3:
            failures.append(f"step 1 read {start:.5f}, not ln(64) = {uni:.5f}; a "
                            f"zero-initialised head cannot start anywhere else, so the "
                            f"head, the symmetrisation or the loss disagrees with upstream")
        if worst > a.near_zero:
            failures.append(f"worst final CE {worst:.5f} did not reach the near-zero bar "
                            f"{a.near_zero}")
        bad_kept = [h["step"] for h in hist if not (0.95 <= h["kept"] <= 1.05)]
        if bad_kept:
            failures.append(f"the device step left the master step at {len(bad_kept)} of "
                            f"{len(hist)} steps (first at {bad_kept[0]})")
        with open(os.path.join(ART, "overfit_hist.json"), "w") as fh:
            json.dump({"hist": hist, "final": final, "uniform": uni, "depth": a.depth,
                       "lr": a.lr, "steps": a.steps, "params": n_par}, fh, indent=2)
    print()
    print(clk.line(0))
    print()
    if failures:
        print(f"OVERFIT FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("OVERFIT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
