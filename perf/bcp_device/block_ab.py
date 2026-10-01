#!/usr/bin/env python3
"""A device lever on the shipped Evoformer blocks: arms alternated in one process, reach counted.

K checkpointed blocks of `bindcraft2._Trunk` at the round's shape (n=288, 261 real, fold masks, two
MSA rows), taped forward + backward per step exactly as `_Trunk.evoformer(recompute=True)` runs
them, under `bindcraft2.fast_round()`. The lever is a module switch flipped live per arm; every arm
counts what the lever served and declined inside its own steps, and the arms' input gradients are
compared on identical operands. AICLK from the card's sysfs node during each step.
"""
import argparse, json, os, pathlib, sys, time, gc, collections
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402

LEVERS = {"lnbw": ("tt_bio.lnbw", "FUSED", "REACH")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz")
    ap.add_argument("--lever", default="lnbw")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--nreal", type=int, default=261)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import importlib
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    mod_name, flag, reach_name = LEVERS[a.lever]
    mod = importlib.import_module(mod_name)
    reach = getattr(mod, reach_name)
    tr = bindcraft2._Trunk(pathlib.Path(a.params))
    ag, T = tr.ag, tr.taped
    n, K = a.n, a.K
    torch.manual_seed(0)
    m0, z0 = torch.randn(2, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    wm, wz = torch.randn(m0.shape) * 1e-3, torch.randn(z0.shape) * 1e-3
    seq = torch.zeros(n); seq[:a.nreal] = 1
    msa_mask = tr.up(seq.expand(2, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blocks = tr.model.device_evoformer[:K]
    clock = Clock()

    def step(on):
        gc.collect()
        before = collections.Counter(reach)
        ml, zl = tr.leaf(m0), tr.leaf(z0)
        tr.sync(); t0 = time.time()
        with T.tape():
            m, z = ml, zl
            for blk in blocks:
                m, z = ag.checkpoint(lambda x, y, b=blk: b(x, y, msa_mask, *pm), m, z)
        tr.sync(); t1 = time.time()
        seeds = [tr.seed(wm, m), tr.seed(wz, z)]
        tr.sync(); t2 = time.time()
        ag.backward([m, z], seeds)
        tr.sync(); t3 = time.time()
        g = (tr.grad(ml, tuple(m0.shape)), tr.grad(zl, tuple(z0.shape)))
        ag.release_pins()
        del m, z, ml, zl, seeds
        d = collections.Counter(reach); d.subtract(before)
        return {"on": on, "fwd": t1 - t0, "bwd": t3 - t2, "aiclk": clock.window([(t0, t3)]),
                "reach": {k: v for k, v in d.items() if v}, "loadavg": os.getloadavg()[0]}, g

    rows, grads = [], {}
    with bindcraft2.fast_round() as armed:
        print(json.dumps({"pci": clock.pci, "armed": {k: str(v) for k, v in armed.items()}}), flush=True)
        for on in (False, True):
            setattr(mod, flag, on); step(on)
        for rep in range(a.reps):
            for on in ((False, True) if rep % 2 == 0 else (True, False)):
                setattr(mod, flag, on)
                r, g = step(on)
                r["rep"] = rep
                rows.append(r); grads.setdefault(on, g)
                print(json.dumps(r), flush=True)
    summary = {}
    for on in (False, True):
        b = sorted(r["bwd"] for r in rows if r["on"] == on)
        f = sorted(r["fwd"] for r in rows if r["on"] == on)
        summary["on" if on else "off"] = {"bwd_median": b[len(b) // 2], "fwd_median": f[len(f) // 2]}
    for name, i in (("dm", 0), ("dz", 1)):
        x, y = grads[True][i].double(), grads[False][i].double()
        summary[f"{name}_on_vs_off_rel_l2"] = float((x - y).norm() / y.norm())
    print(json.dumps(summary), flush=True)
    out = a.out or str(ROOT / f"perf/bcp_device/out/block_ab_{a.lever}.json")
    pathlib.Path(out).write_text(json.dumps({"args": vars(a), "rows": rows, "summary": summary}, indent=1))
    clock.stop()


if __name__ == "__main__":
    main()
