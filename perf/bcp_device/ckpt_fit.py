#!/usr/bin/env python3
"""bcp-device candidate 2: what one un-checkpointed Evoformer block costs and buys.

The shipped round checkpoints every block (`bindcraft2._Trunk.evoformer`, recompute=True): the
forward seam runs it untaped, the backward re-runs it taped. Leaving a block un-checkpointed
deletes that re-run and holds its tape in DRAM from the forward seam to the backward seam.

One process, one device open, the shipped `_Trunk` and its blocks. Each step runs K blocks taped,
the last `u` of them without `autograd.checkpoint`, then the backward. Arms alternate inside every
rep. Per arm: DRAM allocated after the forward (the tape the round would hold between seams),
synced forward and backward walls, AICLK sampled from the card's sysfs node during the step.
Slope over u is per block.
"""
import argparse, json, os, pathlib, sys, time, gc
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--nreal", type=int, default=261)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--us", default="0,4,8")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--fast", type=int, default=1, help="arm bindcraft2.fast_round(), as the round does")
    ap.add_argument("--out", default=str(ROOT / "perf/bcp_device/out/ckpt_fit.json"))
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path(a.params))
    ag, T, dev = tr.ag, tr.taped, tr.device

    def dram():
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

    n, K = a.n, a.K
    torch.manual_seed(0)
    m0 = torch.randn(a.depth, n, 256) * 0.5
    z0 = torch.randn(n, n, 128) * 0.5
    wm, wz = torch.randn(m0.shape) * 1e-3, torch.randn(z0.shape) * 1e-3
    seq = torch.zeros(n); seq[:a.nreal] = 1
    msa_mask = tr.up(seq.expand(a.depth, n).contiguous())
    pair_masks = af2_pair_masks(seq[:, None] * seq[None, :], dev)
    blocks = tr.model.device_evoformer[:K]
    clock = Clock()
    print(json.dumps({"pci": clock.pci, "n": n, "K": K, "dram_total_note": "allocated bytes"}), flush=True)

    def step(u):
        gc.collect()
        base = dram()
        ml, zl = tr.leaf(m0), tr.leaf(z0)
        tr.sync(); t0 = time.time()
        with T.tape():  # noqa
            m, z = ml, zl
            for i, blk in enumerate(blocks):
                if i < K - u:
                    m, z = ag.checkpoint(lambda x, y, b=blk: b(x, y, msa_mask, *pair_masks), m, z)
                else:
                    m, z = blk(m, z, msa_mask, *pair_masks)
        tr.sync(); t1 = time.time()
        held = dram() - base
        seeds = [tr.seed(wm, m), tr.seed(wz, z)]
        tr.sync(); t2 = time.time()
        ag.backward([m, z], seeds)
        tr.sync(); t3 = time.time()
        g = (tr.grad(ml, tuple(m0.shape)), tr.grad(zl, tuple(z0.shape)))
        ag.release_pins()
        del m, z, ml, zl, seeds
        gc.collect()
        return {"u": u, "fwd": t1 - t0, "bwd": t3 - t2, "held_after_fwd": held,
                "aiclk": clock.window([(t0, t3)]), "loadavg": os.getloadavg()[0]}, g

    us = [int(x) for x in a.us.split(",")]
    armed = bindcraft2.fast_round() if a.fast else None
    if armed is not None:
        print(json.dumps({"fast_round": {k: str(v) for k, v in armed.__enter__().items()}}), flush=True)
    for u in us:                                   # warm: program cache
        step(u)
    rows, grads = [], {}
    for rep in range(a.reps):
        order = us if rep % 2 == 0 else us[::-1]
        for u in order:
            r, g = step(u)
            r["rep"] = rep
            rows.append(r); grads.setdefault(u, g)
            print(json.dumps(r), flush=True)
    ref = grads[us[0]]
    for u in us[1:]:
        for name, x, y in (("dm", grads[u][0], ref[0]), ("dz", grads[u][1], ref[1])):
            rel = float((x - y).norm() / y.norm())
            print(json.dumps({"grad_vs_u0": u, name: rel, "equal": bool(torch.equal(x, y))}), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    clock.stop()


if __name__ == "__main__":
    main()
