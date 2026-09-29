"""EXACT: the on-card expansion against the untouched float64 reference, on real SAbDab Fvs.

Exact equality, not a PCC: every value in either feature map is 0.0 or 1.0 and both are exact in
every float format, so a difference here would be wrong rather than imprecise.
"""
import statistics, sys, time
import torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/train-u-relpos-ondevice")
from tt_bio.tenstorrent import get_device
from tt_bio.abodybuilder3_reference import ABB3Config, single_and_pair_features
from tt_bio.abodybuilder3 import to_device_fp32, DeviceABB3
from tt_bio.train.abb3_dataset import SabdabFvs, resolve_split

ROOT = "/home/ttuser/abb3_data/data/structures/structures"
SPLIT = "/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv"
MICRO, TOKENS, NMB = 4, 256, 8


def main():
    dev = get_device()
    ids = resolve_split(SPLIT)["train"]
    cfg = ABB3Config(use_plddt=False, no_blocks=8)
    ds = SabdabFvs(ids, ROOT, cfg, dev, tokens=TOKENS)

    n_checked = 0
    for mb in range(NMB):
        idx = list(range(mb * MICRO, (mb + 1) * MICRO))
        hb = ds.host(idx)
        ref_single, ref_pair = single_and_pair_features(hb["aatype"], hb["is_heavy"],
                                                        hb["residue_index"])
        sample = ds.upload(hb)
        got_single = ttnn.to_torch(sample["single_d"]).reshape(ref_single.shape)
        got_pair = ttnn.to_torch(sample["pair_d"]).reshape(ref_pair.shape)
        s_eq = torch.equal(got_single, ref_single)
        p_eq = torch.equal(got_pair, ref_pair)
        print(f"micro-batch {mb} {hb['ids']}: single exact={s_eq} pair exact={p_eq} "
              f"mismatched={int((got_pair != ref_pair).sum()) + int((got_single != ref_single).sum())}")
        assert s_eq and p_eq, "NOT exact"
        n_checked += 1

        if mb == 0:
            # z_initial, the pair map's ONLY consumer, through the shipped linear.
            g = torch.Generator().manual_seed(7)
            w = torch.randn(cfg.c_z, cfg.embed_dim, generator=g) * 0.05
            b = torch.randn(cfg.embed_dim, generator=g) * 0.05
            wide_w = torch.zeros(cfg.embed_dim, cfg.c_z)
            state = {"linear_in_edge.weight": w.t().contiguous(),
                     "linear_in_edge.bias": b,
                     "linear_in_node.weight": torch.randn(cfg.embed_dim, cfg.c_s, generator=g),
                     "linear_in_node.bias": torch.randn(cfg.embed_dim, generator=g)}
            from tt_bio import abodybuilder3_ops as ops
            w_edge = to_device_fp32(torch.cat([w, torch.zeros(cfg.c_z, 0)], dim=1))
            b_edge = to_device_fp32(b)
            z_dev = ttnn.to_torch(ops.linear(sample["pair_d"], w_edge, b_edge))
            z_up = ttnn.to_torch(ops.linear(to_device_fp32(ref_pair), w_edge, b_edge))
            print(f"  z_initial exact={torch.equal(z_dev, z_up)} "
                  f"maxabs={(z_dev - z_up).abs().max().item():.6g}")
            assert torch.equal(z_dev, z_up), "z_initial not bit-identical"

    print(f"EXACT: {n_checked} micro-batches, {n_checked * MICRO} real Fvs, all bit-identical")

    # ---- cost of the two halves, warm
    idx = list(range(0, MICRO))
    hb = ds.host(idx)
    for _ in range(2):
        s = ds.upload(hb); ttnn.synchronize_device(dev); del s
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); s = ds.upload(hb); ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0); del s
    print(f"upload+expand per micro-batch: median {statistics.median(ts)*1e3:.2f} ms "
          f"min {min(ts)*1e3:.2f} ms")
    th = []
    for _ in range(5):
        t0 = time.perf_counter(); ds.host(idx); th.append(time.perf_counter() - t0)
    print(f"host build per micro-batch: median {statistics.median(th)*1e3:.2f} ms")


if __name__ == "__main__":
    main()
