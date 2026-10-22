#!/usr/bin/env python3
"""bc4-bet-move: the four data-movement sites of BindCraft 2's iteration, per call, at the shapes the u2
devprof census shows (`state/bc4/bets/unnamed/raw/u2-devprof.json.gz`).

Each case runs main's form and every candidate on the same input, checks bit-equality against torch's
own rearrangement, and times median of 7 bursts of 20 calls with one synchronize per burst. AICLK and
load1 are sampled after every burst.
    python perf/bc4_move/move_ab.py out.json
"""
import json, os, pathlib, statistics, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag, taped_ttnn as tt
    dev = T.get_device()
    card = int((os.environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    load1 = lambda: float(open("/proc/loadavg").read().split()[0])
    L1 = ttnn.L1_MEMORY_CONFIG

    def up(shape, mc=None, seed=0):
        t = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
        return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                  memory_config=mc or ttnn.DRAM_MEMORY_CONFIG)

    def two_t(x, a, b):
        mid = ttnn.transpose(x, *a)
        out = ttnn.transpose(mid, *b)
        ttnn.deallocate(mid)
        return out

    def join_main(g, ax, full):  # main's `_join_slices` for one part cut at 0
        return ttnn.concat([g, ag._zeros_like_along(g, ax, full - int(g.shape[ax]))], dim=ax)

    cases = []
    # Site: permute in autograd._backward. The joined OPM's VJP of permute(2,0,1): (1,2,0) on [I, c_z, S*D].
    for shape in [(224, 128, 64), (288, 128, 64)]:
        t, x = up(shape)
        cases.append((f"opm_bw_120_{shape[0]}", t.permute(1, 2, 0), {
            "main": lambda x=x: ttnn.permute(x, (1, 2, 0)),
            "two_t": lambda x=x: two_t(x, (0, 1), (1, 2)),
            "lib": lambda x=x: tt._permute_back(x, [1, 2, 0])}))
    # Site: af2._sum_rows_joined forward. P1 permute(2,0,1) on the [c_z, S*D, I] view, P2 (0,2,1) on [I, c_z, J].
    t, x = up((128, 64, 224))
    cases.append(("opm_fw_201_224", t.permute(2, 0, 1), {
        "main": lambda x=x: ttnn.permute(x, (2, 0, 1)),
        "two_t": lambda x=x: two_t(x, (1, 2), (0, 1))}))
    t, x = up((224, 128, 224))
    cases.append(("opm_fw_021_224", t.permute(0, 2, 1), {
        "main": lambda x=x: ttnn.permute(x, (0, 2, 1)),
        "transpose": lambda x=x: ttnn.transpose(x, 1, 2)}))
    # Site: tenstorrent._multiply, the channel chunk's move back (0,2,3,1) on [1, 64, 224, 224], L1 source.
    t, x = up((1, 64, 224, 224), L1)
    cases.append(("trimul_back_0231_224", t.permute(0, 2, 3, 1), {
        "main": lambda x=x: ttnn.permute(x, (0, 2, 3, 1), memory_config=L1),
        "two_t": lambda x=x: two_t(x, (1, 2), (2, 3))}))
    # Site: concat in autograd._join_slices. The subtile attention's o[..., :16] gradient back to 32.
    t, x = up((224, 4, 224, 16))
    cases.append(("join_tail_224x4x224_16to32", torch.cat([t, torch.zeros_like(t)], -1), {
        "main": lambda x=x: join_main(x, 3, 32),
        "lib": lambda x=x: ag._join_slices([([0, 0, 0, 0], [224, 4, 224, 16], x)], [224, 4, 224, 32])[0]}))
    # The join with a leading gap keeps its concat; with two parts and a tail it concats then pads.
    t2, x2 = up((224, 4, 224, 16), seed=1)
    cases.append(("join_two_parts_tail", torch.cat([t, t2, t.new_zeros(224, 4, 224, 32)], -1), {
        "lib": lambda x=x, x2=x2: ag._join_slices([([0, 0, 0, 0], [224, 4, 224, 16], x),
                                                    ([0, 0, 0, 16], [224, 4, 224, 32], x2)], [224, 4, 224, 64])[0]}))
    cases.append(("join_lead_gap", torch.cat([torch.zeros_like(t), t], -1), {
        "lib": lambda x=x: ag._join_slices([([0, 0, 0, 16], [224, 4, 224, 32], x)], [224, 4, 224, 32])[0]}))

    res = {"card": card, "sha": os.environ.get("SPD_TREE_SHA", ""), "cases": []}
    for name, ref, arms in cases:
        row = {"case": name, "bit_equal": {}}
        for k, f in list(arms.items()):
            try:
                row["bit_equal"][k] = bool(torch.equal(ttnn.to_torch(f()).reshape(ref.shape), ref))
            except Exception as e:
                row["bit_equal"][k] = f"raised {e!r}"[:300]
                del arms[k]
        ms, clk, ld = {k: [] for k in arms}, [], []
        for _ in range(7):
            for k, f in arms.items():
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(20):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(dev); ms[k].append((time.perf_counter() - t0) * 1e3 / 20)
                clk.append(aiclk()); ld.append(load1())
        for k in arms:
            row[k] = {"ms": round(statistics.median(ms[k]), 4), "spread_ms": round(max(ms[k]) - min(ms[k]), 4)}
        row["aiclk_median_min"] = [statistics.median(clk), min(clk)]
        row["load1_max"] = max(ld)
        res["cases"].append(row)
        print(json.dumps(row), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
