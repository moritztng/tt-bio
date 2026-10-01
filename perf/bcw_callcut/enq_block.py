#!/usr/bin/env python3
"""Is the composed Evoformer block backward enqueue-bound? Per rep: host time until
`autograd.backward` returns (enqueue, nothing synced inside) against the wall to the card
finishing it (sync). K blocks chained, checkpointed, under fast_round, n=288."""
import argparse, json, pathlib, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--off", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import importlib, ttnn
    from tt_bio import bindcraft2
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal, depth = 288, 261, 2
    torch.manual_seed(0)
    m0, z0 = torch.randn(depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blocks = tr.model.device_evoformer[:a.blocks]
    off = [x for x in a.off.split(",") if x]
    res = []
    clk = pathlib.Path("/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")
    with bindcraft2.fast_round():
        for module, owner, attr, env, value in bindcraft2._FAST_ROUND:
            if attr in off or f"{module}.{attr}" in off:
                tgt = importlib.import_module(f"tt_bio.{module}")
                tgt = getattr(tgt, owner) if owner else tgt
                setattr(tgt, attr, "tri_att_sdpa_hifi,rne_add" if attr == "TAPED_KERNELS_DEFAULT" else False)
        for rep in range(a.reps + 1):
            m, z = tr.leaf(m0), tr.leaf(z0)
            mi, zi = m, z
            t0 = time.perf_counter()
            with T.tape():
                for blk in blocks:
                    m, z = ag.checkpoint(lambda x, y, b=blk: b(x, y, msa_mask, *pm), m, z)
            t1 = time.perf_counter()
            tr.sync()
            t2 = time.perf_counter()
            seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
            tr.sync()
            c0 = clk.read_text().strip()
            t3 = time.perf_counter()
            ag.backward([m, z], seeds)
            t4 = time.perf_counter()
            c1 = clk.read_text().strip()
            tr.sync()
            t5 = time.perf_counter()
            ag.release_pins()
            if rep:
                res.append({"fwd_enq": t1 - t0, "fwd_wall": t2 - t0, "bwd_enq": t4 - t3,
                            "bwd_wall": t5 - t3, "aiclk": [c0, c1]})
                print(json.dumps(res[-1]), flush=True)
    med = lambda k: sorted(r[k] for r in res)[len(res) // 2]
    summ = {k: med(k) for k in ("fwd_enq", "fwd_wall", "bwd_enq", "bwd_wall")}
    summ["blocks"], summ["off"] = a.blocks, off
    print(json.dumps(summ))
    pathlib.Path(a.out).write_text(json.dumps({"summary": summ, "reps": res}, indent=1))


if __name__ == "__main__":
    main()
