#!/usr/bin/env python3
"""bc4-bet-opm step 2: per-call device time of the masked OPM's joined sum, main's form against this
bet's, S=2, C=D=32, c_z=128, at BindCraft 2's token buckets. Same process, same operands, arms
interleaved burst by burst so a clock or load drift lands on both.

  main     the sequence on main (second product [I, c_z, S*D] @ [S*D, J], planned on 4 tiles of M)
  opm      `AF2MaskedOuterProductMean._sum_rows_joined` as on this branch ([1, I*c_z, S*D] view)
  opm_ng   the same view without `core_grid` (ttnn's simple planner), to show which plan won

Per call: median over bursts of 10 back-to-back calls, one synchronize per burst. Each arm also
reports bit-equality with `main` and relative L2 against a float64 einsum of the same bf16 inputs.

    TT_VISIBLE_DEVICES=<card> timeout 900 python perf/bc4_opm/opm_probe.py out.json [--ns 224,192,288]
"""
import argparse, json, os, pathlib, statistics, sys, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ns", default="224,192,288")
    p.add_argument("--bursts", type=int, default=9)
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as TT
    from tt_bio.af2 import AF2MaskedOuterProductMean
    from tt_bio.autograd import precise_config
    card = int((os.environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    dev = TT.get_device()
    cfg = precise_config()
    S, C, D, cz = 2, 32, 32, 128
    g = torch.Generator().manual_seed(0)
    sd = {"norm.weight": torch.ones(64), "norm.bias": torch.zeros(64),
          "proj_a.weight": torch.randn(C, 64, generator=g) * 0.1,
          "proj_b.weight": torch.randn(D, 64, generator=g) * 0.1,
          "proj_o.weight": torch.randn(cz, C * D, generator=g) * 0.05,
          "proj_o.bias": torch.randn(cz, generator=g)}
    opm = AF2MaskedOuterProductMean(sd, cfg, scale_bias=True)
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o64 = ttnn.to_torch(opm.o_weight).double()
    b64 = ttnn.to_torch(opm.o_bias).double().reshape(-1)

    def main_form(a, b, I, J):
        w = opm._rows_joined_weight(S, C, D)
        at = ttnn.reshape(ttnn.permute(a, (0, 2, 1)), (S * C, I))
        T = ttnn.matmul(w, at, compute_kernel_config=cfg, core_grid=TT.CORE_GRID_MAIN)
        Am = ttnn.permute(ttnn.reshape(T, (cz, S * D, I)), (2, 0, 1))
        bt = ttnn.reshape(ttnn.permute(b, (0, 2, 1)), (S * D, J))
        out = ttnn.permute(ttnn.matmul(Am, bt, compute_kernel_config=cfg), (0, 2, 1))
        out = ttnn.add(out, opm.o_bias)
        return ttnn.reshape(out, (1, *tuple(out.shape)))

    def flat_nogrid(a, b, I, J):
        w = opm._rows_joined_weight(S, C, D)
        at = ttnn.reshape(ttnn.permute(a, (0, 2, 1)), (S * C, I))
        T = ttnn.matmul(w, at, compute_kernel_config=cfg, core_grid=TT.CORE_GRID_MAIN)
        Am = ttnn.reshape(ttnn.permute(ttnn.reshape(T, (cz, S * D, I)), (2, 0, 1)), (1, I * cz, S * D))
        bt = ttnn.reshape(ttnn.permute(b, (0, 2, 1)), (S * D, J))
        out = ttnn.matmul(Am, bt, compute_kernel_config=cfg)
        out = ttnn.add(ttnn.permute(ttnn.reshape(out, (I, cz, J)), (0, 2, 1)), opm.o_bias)
        return ttnn.reshape(out, (1, *tuple(out.shape)))

    arms = {"main": main_form,
            "opm": lambda a, b, I, J: opm._sum_rows_joined(a, b, S, I, C, J, D),
            "opm_ng": flat_nogrid}
    res = {"card": card, "host": os.uname().nodename, "sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "loadavg_start": open("/proc/loadavg").read().split()[:3], "ns": {}}
    for n in (int(x) for x in A.ns.split(",")):
        I = J = n
        a_t = torch.randn(S, I, C, generator=g).to(torch.bfloat16)
        b_t = torch.randn(S, J, D, generator=g).to(torch.bfloat16)
        a, b = up(a_t), up(b_t)
        ref = torch.einsum("sic,sjd,cdk->ijk", a_t.double(), b_t.double(), o64.reshape(C, D, cz)) + b64
        outs = {k: ttnn.to_torch(f(a, b, I, J)).double().reshape(I, J, cz) for k, f in arms.items()}
        row = {k: {"bit_equal_main": bool(torch.equal(outs[k], outs["main"])),
                   "rel_f64": float((outs[k] - ref).norm() / ref.norm()), "ms": []} for k in arms}
        clocks = []
        for _ in range(A.bursts):
            for k, f in arms.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(10):
                    f(a, b, I, J)
                ttnn.synchronize_device(dev)
                row[k]["ms"].append((time.perf_counter() - t0) * 100)
                clocks.append(aiclk())
        for k in arms:
            ms = row[k]["ms"]
            row[k].update(ms_median=statistics.median(ms), spread_ms=max(ms) - min(ms))
        row["save_ms_per_call"] = row["main"]["ms_median"] - row["opm"]["ms_median"]
        row["aiclk_median"] = statistics.median(clocks)
        row["aiclk_min_max"] = [min(clocks), max(clocks)]
        res["ns"][str(n)] = row
        print(n, {k: (round(row[k]["ms_median"], 4), row[k]["bit_equal_main"], f'{row[k]["rel_f64"]:.2e}')
                  for k in arms}, row["aiclk_median"], flush=True)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    pathlib.Path(A.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
