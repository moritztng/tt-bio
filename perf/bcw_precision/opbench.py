#!/usr/bin/env python3
"""What one byte per element buys on the backward's own op families, on this card and this ttnn.

Each case is one ttnn op at a shape the BindCraft 2 backward issues at n=288 (pair track
[288*288, 128], triangle-multiplication batched matmul [128, 288, 288]), run in every dtype arm.
Arms rotate inside each repetition so drift lands on all of them; every timing is synchronised
before the clock starts and stops, warm (two discarded calls per arm first). AICLK is read from the
card's sysfs node every 50 ms by a separate process for the whole run. Reports the median per-call time
and its ratio to the first arm (bf16).
"""
import argparse, json, pathlib, statistics as stt, subprocess, sys, tempfile, time
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Clock:
    """AICLK from the card's sysfs node, sampled by a separate process: ttnn dispatch holds the
    GIL, so a sampler thread in this process starves (it took 3 samples in a 2 s case)."""

    def __init__(self, node, dt=0.05):
        self.f = tempfile.NamedTemporaryFile("w+", suffix=".aiclk")
        self.p = subprocess.Popen(["sh", "-c", f"while :; do cat '{node}/tt_aiclk'; sleep {dt}; done"],
                                  stdout=self.f, stderr=subprocess.DEVNULL)

    def stop(self):
        self.p.terminate()
        self.p.wait()
        self.f.seek(0)
        s = [int(x.split()[0]) for x in self.f if x.strip()]
        return {"median": stt.median(s), "min": min(s), "n": len(s), "under1200": sum(x < 1200 for x in s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="/sys/class/tenstorrent/tenstorrent!3")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of case names")
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    T = ttnn.TILE_LAYOUT
    DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
    torch.manual_seed(0)

    def up(shape, dt):
        return ttnn.from_torch(torch.randn(shape), dtype=DT[dt], layout=T, device=dev)

    hifi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=True)
    P = [1, 1, 288 * 288, 128]
    B = [1, 128, 288, 288]
    W = [1, 1, 128, 128]
    cases = {}

    def case(name, arms):
        cases[name] = arms

    # eltwise, the backward's largest byte family after matmul
    for op in ("multiply", "add", "subtract"):
        cases[op] = {dt: (lambda f=getattr(ttnn, op), x=up(P, dt), y=up(P, dt), d=DT[dt]: f(x, y, dtype=d), None)
                     for dt in ("bf16", "bfp8", "fp32")}
    # a narrow region always has a dtype seam: what does a mixed-operand binary cost?
    case("multiply_mixed", {
        "bf16*bf16->bf16": (lambda x=up(P, "bf16"), y=up(P, "bf16"): ttnn.multiply(x, y, dtype=ttnn.bfloat16), None),
        "bf16*bf16->bfp8": (lambda x=up(P, "bf16"), y=up(P, "bf16"): ttnn.multiply(x, y, dtype=ttnn.bfloat8_b), None),
        "bf16*bfp8->bfp8": (lambda x=up(P, "bf16"), y=up(P, "bfp8"): ttnn.multiply(x, y, dtype=ttnn.bfloat8_b), None),
        "bf16*bfp8->bf16": (lambda x=up(P, "bf16"), y=up(P, "bfp8"): ttnn.multiply(x, y, dtype=ttnn.bfloat16), None),
        "fp32*bf16->fp32": (lambda x=up(P, "fp32"), y=up(P, "bf16"): ttnn.multiply(x, y, dtype=ttnn.float32), None),
        "bfp8*bfp8->bfp8": (lambda x=up(P, "bfp8"), y=up(P, "bfp8"): ttnn.multiply(x, y, dtype=ttnn.bfloat8_b), None)})
    case("sigmoid_bw", {dt: (lambda g=up(P, dt), x=up(P, dt): ttnn.sigmoid_bw(g, x)[0], None)
                        for dt in ("bf16", "bfp8")})
    case("typecast_from_fp32", {dt: (lambda x=up(P, "fp32"), d=DT[dt]: ttnn.typecast(x, d), None)
                                for dt in ("bf16", "bfp8")})
    case("permute_pair", {dt: (lambda x=up([1, 288, 288, 128], dt): ttnn.permute(x, (0, 3, 1, 2)), None)
                          for dt in ("bf16", "bfp8")})
    case("layer_norm", {dt: (lambda x=up(P, dt), g=up([1, 1, 32, 128], "bf16"): ttnn.layer_norm(x), None)
                        for dt in ("bf16", "bfp8")})
    lin = {}
    for dt in ("bf16", "bfp8"):
        for wd in ("bf16", "bfp8"):
            for fid, cfg in (("hifi2", hifi), ("lofi", lofi)):
                lin[f"{dt}/w{wd}/{fid}"] = (lambda x=up(P, dt), w=up(W, wd), d=DT[dt], c=cfg:
                                            ttnn.matmul(x, w, dtype=d, compute_kernel_config=c), None)
    case("linear_128x128", lin)
    bmm = {}
    for dt in ("bf16", "bfp8"):
        for fid, cfg in (("hifi2", hifi), ("lofi", lofi)):
            bmm[f"{dt}/{fid}"] = (lambda x=up(B, dt), y=up(B, dt), d=DT[dt], c=cfg:
                                  ttnn.matmul(x, y, transpose_b=True, dtype=d, compute_kernel_config=c), None)
    case("bmm_trimul", bmm)

    if a.only:
        cases = {k: v for k, v in cases.items() if k in a.only.split(",")}
    clock = Clock(a.node)
    res = {}
    for name, arms in cases.items():
        times = {k: [] for k in arms}
        for k, (fn, _) in arms.items():
            for _ in range(2):
                o = fn()
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
        for rep in range(a.reps):
            for k, (fn, _) in arms.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    o = fn()
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                times[k].append((time.perf_counter() - t0) / a.iters)
        res[name] = {k: {"ms": stt.median(v) * 1e3, "spread_ms": [min(v) * 1e3, max(v) * 1e3]}
                     for k, v in times.items()}
        base = next(iter(res[name].values()))["ms"]
        for k, v in res[name].items():
            v["x_vs_first"] = base / v["ms"]
            print(f"{name:20s} {k:18s} {v['ms']:8.3f} ms  {v['x_vs_first']:5.2f}x  "
                  f"[{v['spread_ms'][0]:.3f}-{v['spread_ms'][1]:.3f}]", flush=True)
    res["aiclk"] = clock.stop()
    print("AICLK", res["aiclk"])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
