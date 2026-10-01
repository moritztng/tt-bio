#!/usr/bin/env python3
"""C9 at the block: K checkpointed Evoformer blocks under fast_round, n=288, the taped recompute's
qkv projection composed (minimal_matmul + nlp_create_qkv_heads, arm "composed") against the
head-major kernel taped through its `triatt_qkv_heads` entry (arm "gated"), alternated ABBA per rep
in one process. Per rep: backward enqueue and synced wall, AICLK before/after. Reach counted at
runtime per backward: nlp_create_qkv_heads calls and the entry's served/declined. Gradients of
both leaves compared bit for bit between arms on identical inputs and seeds."""
import os
import argparse, json, pathlib, statistics as st, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--card", default="2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2, autograd
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal, depth = 288, 261, 2
    torch.manual_seed(0)
    m0, z0 = torch.randn(depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    sm, sz = torch.randn(m0.shape) * 1e-3, torch.randn(z0.shape) * 1e-3
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blocks = tr.model.device_evoformer[:a.blocks]
    clk = pathlib.Path(f"/sys/class/tenstorrent/tenstorrent!{a.card}/tt_aiclk")
    from tt_bio import taped_ttnn
    reach = {"create_qkv_heads": 0, "entry_served": 0, "entry_declined": 0}
    o_cq = ttnn.experimental.nlp_create_qkv_heads

    def cq(*x, **k):
        reach["create_qkv_heads"] += 1
        return o_cq(*x, **k)

    ttnn.experimental.nlp_create_qkv_heads = cq
    res = {"composed": [], "gated": []}
    reach_by = {}
    grads = {}
    with bindcraft2.fast_round():
        order = ["composed", "gated", "gated", "composed"]
        for rep in range(a.reps + 2):
            arm = order[rep % 4]
            base = taped_ttnn.TAPED_KERNELS_DEFAULT
            os.environ["TT_BIO_TAPED_KERNELS"] = (base + ",triatt_qkv_heads" if arm == "gated"
                                                  else base)
            m, z = tr.leaf(m0), tr.leaf(z0)
            mi, zi = m, z
            with T.tape():
                for blk in blocks:
                    m, z = ag.checkpoint(lambda x, y, b=blk: b(x, y, msa_mask, *pm), m, z)
            seeds = [tr.seed(sm, m), tr.seed(sz, z)]
            tr.sync()
            for k in reach: reach[k] = 0
            st0 = list(taped_ttnn.KERNEL_STATS.get("triatt_qkv_heads", [0, 0]))
            c0 = int(clk.read_text().split()[0])
            t3 = time.perf_counter()
            ag.backward([m, z], seeds)
            t4 = time.perf_counter()
            c1 = int(clk.read_text().split()[0])
            tr.sync()
            t5 = time.perf_counter()
            st1 = taped_ttnn.KERNEL_STATS.get("triatt_qkv_heads", [0, 0])
            reach["entry_served"], reach["entry_declined"] = st1[0] - st0[0], st1[1] - st0[1]
            reach_by[arm] = dict(reach)
            if arm not in grads:
                grads[arm] = [ttnn.to_torch(t.grad) for t in (mi, zi)]
            ag.release_pins()
            if rep >= 2:
                res[arm].append({"bwd_enq": t4 - t3, "bwd_wall": t5 - t3, "aiclk": [c0, c1]})
                print(arm, json.dumps(res[arm][-1]), flush=True)
    summ = {arm: {k: round(st.median(r[k] for r in v), 5) for k in ("bwd_enq", "bwd_wall")}
            for arm, v in res.items()}
    clks = [c for v in res.values() for r in v for c in r["aiclk"]]
    eq = [bool(torch.equal(x, y)) for x, y in zip(grads["composed"], grads["gated"])]
    out = {"blocks": a.blocks, "summary": summ, "reach_per_backward": reach_by,
           "grads_bit_equal_dm_dz": eq, "aiclk": {"median": st.median(clks), "min": min(clks)},
           "reps": res}
    print(json.dumps({k: out[k] for k in ("summary", "reach_per_backward", "grads_bit_equal_dm_dz", "aiclk")}, indent=1))
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
