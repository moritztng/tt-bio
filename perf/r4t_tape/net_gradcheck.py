"""The whole RFD4 trainable network's gradient (host glue + device trunk + host heads + rfd4_loss) against a
float64 reference that central differences validate first.

Device side: `perf/r4t_step/trunk_step.py --batch B --grad-dump PATH [--arm fp32-exact] [--break]` writes every
trained parameter's gradient of one real step (the production route: k candidates grad-off, our own winner, its
last recycle taped, the fold VJPs into the upstream leaves, the glue over the bridge). This script rebuilds that
step in float64 on the host: the same glue modules in float64, `tests/rfd4_reference.trunk` on the release `nn.`
weights with the early cycles frozen at the base point (upstream `recycle grad: last`), the same heads, the
registered `rfd4_loss`, the device's winner. Its gradient is torch autograd of that loss; a sample of tensors is
checked against central differences before anything is compared (D28 composed bar, `train.checks`).

    python3 perf/r4t_tape/net_gradcheck.py --dump OUT/fp32 --batch ~/r4t/step/batch00.npz [--scale-grad NAME]

--scale-grad: the subtle broken arm, one tensor's device gradient x1.10 (rel 0.10) before grading.
--save-ref / --ref: cache the float64 reference gradient (one run serves every arm of the same batch+winner).
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="trunk_step --grad-dump prefix (PATH.npz + PATH.json)")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--probe-tensors", type=int, default=12, help="tensors whose reference gradient is "
                    "checked by central differences (spread over glue, blocks, pair updates, cycle, heads)")
    ap.add_argument("--probe", type=int, default=1, help="coordinates per probed tensor")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale-grad", default="")
    ap.add_argument("--ref", default="", help="reuse a saved reference (npz from --save-ref)")
    ap.add_argument("--save-ref", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(here, "..", "r4t"), os.path.join(here, "..", "..", "tests")]
    import torch
    import rfd4_reference as REF
    from loss_parity_reference import load_batch
    from tt_bio.rfd4 import network as N, engine as E
    from tt_bio.train import checks as CK, rfd4_objective as RO

    meta = json.load(open(a.dump + ".json"))
    dz = np.load(a.dump + ".npz")
    got = {k: dz[k].astype(np.float64) for k in dz.files}
    win = int(meta["winner"])
    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config).double()
    glue.requires_grad_(True)
    bt, bmeta, _rec, _ = load_batch(a.batch, dtype=torch.float64)
    K = int(bt["padding_mask"].shape[0])
    depth = max(1, int(bmeta["depth"]))

    def take(d, i):
        if isinstance(d, dict):
            return {k: take(v, i) for k, v in d.items()}
        return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d

    def npy(d):
        return {k: npy(v) for k, v in d.items()} if isinstance(d, dict) else (
            d.detach().numpy() if torch.is_tensor(d) else d)

    bw = take(bt, win)
    mask = bw["padding_mask"].double()
    # trunk leaves: the upstream `nn.` tensors the device trunk reads, float64, named as the dump names them
    w = {}
    params = {}
    for k, v in state.items():
        if k.startswith("nn."):
            t = v.detach().double().clone().requires_grad_(True)
            w[k[3:]] = t
            for nm in (k, k[3:]):
                if nm in got:
                    params[nm] = t
    for nm, p in glue.named_parameters():
        if "glue." + nm in got:
            params["glue." + nm] = p
    missing = sorted(set(got) - set(params))
    if missing:
        raise SystemExit(f"device gradients with no reference tensor: {missing[:8]} ({len(missing)})")

    def conds():
        cs = [glue.conditioning(bw, None)]
        for cyc in range(1, depth):
            cs.append(glue.conditioning(bw, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
        return cs

    kw = dict(every_n=cfg["update_pair_repr_every_n"], n_heads=cfg["nheads"])
    with torch.no_grad():   # the early cycles, frozen at the base point (`grad: last` differentiates only the last)
        cs0 = conds()
        s0, z0 = glue.initial_states(bw)
        frozen = REF.trunk(w, s0, z0, cs0[:-1], mask, cfg["nlayers"], **kw) if depth > 1 else None

    def flat(o):
        v = lambda m: next(iter(o[m].values()))  # noqa: E731
        out = {"v_bb_ca": v("bb_ca"), "v_local_latents": v("local_latents")}
        if "sequence" in o:
            out["sequence_logits"] = o["sequence"]["logits"]
        if "insertion" in o:
            out["insertion_logits"] = o["insertion"]["logits"]
        return out

    def loss_fn():
        """Value = the registered loss; gradient = its seeds through the outputs (the step's own surrogate)."""
        s_i, z_i = glue.initial_states(bw)
        c = conds()[-1]
        s, _z = REF.trunk(w, s_i, z_i, [c], mask, cfg["nlayers"], state=frozen, **kw)
        o = flat(glue.heads(s, bw))
        total, _bd, seeds = RO.rfd4_loss({**npy(bw), "exploration_k": 1},
                                         {k: v.detach().double().numpy() for k, v in o.items()}, RO.WEIGHTS)
        sur = sum((o[k] * torch.from_numpy(np.asarray(g)).to(o[k].dtype)).sum() for k, g in seeds.items())
        return sur - sur.detach() + float(total)

    names = sorted(got)
    if a.ref:
        rz = np.load(a.ref)
        ref_grads = {k: rz[k] for k in rz.files}
        fd = json.load(open(a.ref + ".json"))
    else:
        with torch.enable_grad():
            L = loss_fn()
            gs = torch.autograd.grad(L, [params[k] for k in names], allow_unused=True)
        ref_grads = {k: (np.zeros_like(got[k]) if g is None else g.detach().numpy()) for k, g in zip(names, gs)}
        print(f"reference loss {float(L):.9e} (device {meta['loss']})", flush=True)
        # FD on a spread sample: the largest tensor of each kind, so every code path is differenced once
        kinds = {}
        for k in names:
            kind = ".".join(p for p in k.split(".") if not p.isdigit())
            if kind not in kinds or got[k].size > got[kinds[kind]].size:
                kinds[kind] = k
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.permutation(sorted(kinds.values()))[:a.probe_tensors].tolist())
        print("FD tensors", pick, flush=True)
        worst, probed, eligible = CK.fd_check(loss_fn, [params[k] for k in pick], n_probe=a.probe, seed=a.seed)
        fd = {"worst": worst, "probed": probed, "eligible": eligible, "tensors": pick}
        if a.save_ref:
            np.savez(a.save_ref, **ref_grads)
            json.dump(fd, open(a.save_ref + ".json", "w"))
    print(f"reference vs central differences: {fd['worst']:.2e} against {CK.FD_BAR:.0e} "
          f"({fd['probed']} coords in {len(fd['tensors'])} tensors)", flush=True)
    if a.scale_grad:
        got[a.scale_grad] = got[a.scale_grad] * 1.10
    rep = CK.GradcheckReport(name=f"rfd4-network/{meta['arm']}" + ("/broken" if meta.get("dropped") else "")
                             + (f"/x1.10:{a.scale_grad}" if a.scale_grad else ""), op_class="reduction",
                             dtype="float32" if "FLOAT32" in meta["dtype"].upper() else "bfloat16", bar_why="D28")
    rep.fd_worst, rep.fd_probed, rep.fd_eligible = fd["worst"], fd["probed"], fd["eligible"]
    rep.reference_verified = bool(fd["worst"] < CK.FD_BAR)
    for k in names:
        if np.abs(ref_grads[k]).max() == 0 and np.abs(got[k]).max() == 0:
            continue   # unread on this batch by both sides (e.g. a head the batch does not seed)
        rep.grads[k] = dict(CK.metrics(got[k], ref_grads[k]), pass_=True)
    comp = CK.composed_verdict(rep)
    zero_ref = sorted(k for k in names if np.abs(ref_grads[k]).max() == 0)
    res = {"arm": meta["arm"], "batch": meta.get("batch"), "winner": win, "dropped": meta.get("dropped"),
           "scale_grad": a.scale_grad, "aiclk": meta.get("aiclk"), "exact": meta.get("exact"), "fd": fd,
           "n_tensors": len(rep.grads), "zero_both": zero_ref, "composed": comp,
           "grads": {k: {q: float(g[q]) for q in ("rel_l2", "cos")} for k, g in rep.grads.items()}}
    worst = sorted(rep.grads, key=lambda k: -rep.grads[k]["rel_l2"])[:8]
    for k in worst:
        print(f"  {k}: rel_l2 {rep.grads[k]['rel_l2']:.3e} cos {rep.grads[k]['cos']:.7f}")
    print(json.dumps({k: res[k] for k in ("arm", "batch", "winner", "dropped", "scale_grad", "n_tensors", "composed")}
                     | {"fails": comp.get("fails", [])[:10]}), flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
    print("COMPOSED_PASS" if comp.get("pass") else "COMPOSED_FAIL")
    return 0 if comp.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
