"""`dit_fc123g` against the shipped `dit_shard` transition (Boltz-2 token DiT, 512 x 1): device time and error
against a float64 reference, under both compute configs the DiT can run (HiFi4 with fp32 dest, HiFi2 without).

    TT_VISIBLE_DEVICES=<chip> python perf/spd_gap_small/ditfc123.py --out ditfc123.jsonl

  shipped   T._dit_transition_sharded as it ships (three sharded linears, two sharded multiplies, b_to_a)
  fc123g    the same with the three hidden products and both multiplies as one fc12g pass (`w3`)
  hid       the fused hidden product alone (`_fc12g(products=3)`), against the reference before b_to_a
rel = ||out - ref|| / ||ref|| with ref computed in float64 from the same bf16 operands.
"""
import argparse, json, time, traceback

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from perf.spd_difflin.op_probe import dev_time, opened_nodes  # noqa: E402

N, D, H = 512, 768, 1536


def rel(t, ref):
    x = ttnn.to_torch(t).double().reshape(ref.shape)
    return float((x - ref).norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = T.get_device()
    nodes = opened_nodes()
    torch.manual_seed(0)
    bf = lambda t: t.bfloat16().double()  # noqa: E731
    a_h = bf(torch.randn(N, D))
    w_h = {k: bf(torch.randn(D, H) * .03) for k in ("sw", "gt", "ab")}
    w_h["ba"] = bf(torch.randn(H, D) * .03)
    hid_ref = torch.nn.functional.silu(a_h @ w_h["gt"]) * (a_h @ w_h["sw"]) * (a_h @ w_h["ab"])
    out_ref = hid_ref @ w_h["ba"]
    up = lambda t: ttnn.from_torch(t.reshape(1, 1, *t.shape).bfloat16() if t.dim() == 2 and t.shape[0] == N  # noqa: E731
                                   else t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a = up(a_h)
    W = {k: up(v) for k, v in w_h.items()}
    W["w3"] = up(T._fc12_interleave(w_h["gt"], w_h["sw"], w_h["ab"], gx=8))
    ckcs = {"hifi4_f32": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                          fp32_dest_acc_en=True, packer_l1_acc=True),
            "hifi2_bf16": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                           fp32_dest_acc_en=False, packer_l1_acc=True)}
    mt, kt, nt, ct = N // 32, D // 32, H // 32, D // 32
    gx, gy = T._transition_shard_grid(mt, nt, ct)
    pm, pn = mt // gy, nt // gx
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(cores, [pm * 32, pn * 32], ttnn.ShardOrientation.ROW_MAJOR))
    bw = T._transition_kb("fc1", kt)
    rows = [dict(arm="grid", arch=T.arch_name(), gx=gx, gy=gy, pm=pm, pn=pn, bw=bw, t=time.time())]
    print(json.dumps(rows[-1]), flush=True)
    for cname, ckc in ckcs.items():
        arms = {"shipped": (lambda: T._dit_transition_sharded(a, W["sw"], W["gt"], W["ab"], W["ba"], ckc), out_ref),
                "fc123g": (lambda: T._dit_transition_sharded(a, W["sw"], W["gt"], W["ab"], W["ba"], ckc, W["w3"]),
                           out_ref),
                "hid": (lambda: T._fc12g(a, W["w3"], mc, pm, pn, kt, gx, gy, ckc, ttnn.bfloat16, products=3,
                                         defines={"FC12G_MUL_HIFI4": "1"}), hid_ref)}
        for arm, (fn, ref) in arms.items():
            r = dict(arm=arm, ckc=cname, arch=T.arch_name(), t=time.time())
            try:
                y = fn()
                r["rel"] = rel(y, ref)
                ttnn.deallocate(y)
                us, spread, clk = dev_time(dev, fn, nodes=nodes)
                r.update(us=us, spread=spread, aiclk=clk)
            except Exception as e:  # noqa: BLE001 -- an arm that cannot run is a result, not a crash
                r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                traceback.print_exc()
            T._TRANSITION_SHARD_REFUSED.clear()
            r["latch"] = {k: dict(T.LATCH_STATS[k], why=T.LATCH_STATS[k]["why"][-1:]) for k in ("dit_shard", "dit_fc123g")}
            rows.append(r)
            print(json.dumps(r), flush=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
