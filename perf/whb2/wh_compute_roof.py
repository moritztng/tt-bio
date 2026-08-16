"""The compute roof this part reaches, and the trunk's exact arithmetic against it.

Why this instead of a fourth graph capture: the byte model failed three times because ttnn.graph on
this wheel records neither per-op tensor traffic nor per-op device time for this model. FLOPs do not
need it. A matmul's arithmetic is 2*M*K*N from the shapes the caller passes, so a host-side hook on
ttnn.linear / ttnn.matmul / ttnn.experimental.minimal_matmul counts it exactly and depends on none of
what the capture got wrong.

Fidelity is why a generic "peak bf16" figure would be wrong here. The trunk runs HiFi4
(trunk_compute_kernel_config), which is four passes per tile against LoFi's one, so pricing the trunk
against a LoFi roof would flatter it about 4x. Both are measured and both are reported.

The region boundary is T.TrunkModule._iteration, the same one the byte model used. It is a Python
wrapper, so hooking it is unaffected by the capture defects. The second call is counted: warm, no
compile.
"""
import argparse, json, os, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roof-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))
    import torch, ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(tree), T.__file__

    device = T.get_device()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(grid), "cores": grid[0] * grid[1], "arch": T.arch_name(),
           "size_aa": a.size, "roof": {}, "trunk": {}}

    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if device.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)

    def roof_at(fid, N):
        ckc = kernel_cls(math_fidelity=fid, math_approx_mode=False,
                         fp32_dest_acc_en=True, packer_l1_acc=True)
        mk = lambda: ttnn.from_torch(
            torch.randn(N, N, dtype=torch.float32).to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x, w = mk(), mk()
        for _ in range(2):
            ttnn.deallocate(ttnn.matmul(x, w, compute_kernel_config=ckc))
        ttnn.synchronize_device(device)
        best = None
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(3):
                o = ttnn.matmul(x, w, compute_kernel_config=ckc)
            ttnn.synchronize_device(device)
            s = (time.perf_counter() - t0) / 3
            ttnn.deallocate(o)
            best = s if best is None else min(best, s)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return round(2.0 * N ** 3 / best / 1e12, 2)

    for name, fid in (("LoFi", ttnn.MathFidelity.LoFi), ("HiFi4", ttnn.MathFidelity.HiFi4)):
        out["roof"][name] = {str(N): roof_at(fid, N) for N in a.roof_sizes}
        out["roof"][name + "_best"] = max(out["roof"][name].values())
        print(f"[roof] {name}: {out['roof'][name]} best={out['roof'][name + '_best']} TFLOP/s",
              flush=True)

    acc = {"flop": 0, "calls": 0, "by": {}}
    active = {"on": False}

    def shape_flop(x, w):
        try:
            xs = [int(d) for d in x.padded_shape]
            ws = [int(d) for d in w.padded_shape]
        except Exception:
            return 0
        if len(xs) < 2 or len(ws) < 2:
            return 0
        k, n = xs[-1], ws[-1]
        m = 1
        for d in xs[:-1]:
            m *= d
        return 2 * m * k * n

    def wrap(mod, name, label):
        fn = getattr(mod, name, None)
        if fn is None:
            return

        def inner(*args, **kw):
            if active["on"] and len(args) >= 2:
                f = shape_flop(args[0], args[1])
                if f:
                    acc["flop"] += f
                    acc["calls"] += 1
                    b = acc["by"].setdefault(label, {"calls": 0, "flop": 0})
                    b["calls"] += 1
                    b["flop"] += f
            return fn(*args, **kw)

        setattr(mod, name, inner)

    wrap(ttnn, "linear", "ttnn.linear")
    wrap(ttnn, "matmul", "ttnn.matmul")
    wrap(ttnn.experimental, "minimal_matmul", "ttnn.experimental.minimal_matmul")

    import tt_baseline as B
    B.RECYCLING_STEPS, B.SAMPLING_STEPS = 1, 2
    msa_dir = tree / f".msa_flop_{a.size}"
    tgt = tree / f"perf/size512/fixtures/cdk2x2_{a.size}.yaml"
    a3m = tree / f"perf/size512/fixtures/cdk2x2_{a.size}.a3m"
    one_fold = B.build_fold("boltz2", msa_dir, tgt, a3m)[0]

    hits = {"n": 0}
    _orig = T.TrunkModule._iteration

    def _it(self, *ar, **kw):
        hits["n"] += 1
        if hits["n"] == 2:
            active["on"] = True
            try:
                return _orig(self, *ar, **kw)
            finally:
                active["on"] = False
        return _orig(self, *ar, **kw)

    T.TrunkModule._iteration = _it
    out["trunk"]["hooked"] = "TrunkModule._iteration, 2nd call"

    t0 = time.perf_counter()
    one_fold()
    out["trunk"]["fold_s"] = round(time.perf_counter() - t0, 3)
    out["trunk"].update(
        iterations_seen=hits["n"], calls=acc["calls"], tflop=round(acc["flop"] / 1e12, 4),
        by_op={k: {"calls": v["calls"], "tflop": round(v["flop"] / 1e12, 4)}
               for k, v in acc["by"].items()})
    print("[trunk]", json.dumps(out["trunk"]), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
