"""Device time of one ttnn call: a captured trace replayed back to back, so host dispatch is not in the number."""
import math, time
import torch, ttnn


class Bench:
    def __init__(self, dev, reps=10, target_s=0.004):
        self.dev, self.reps, self.target_s = dev, reps, target_s
        self.DT = dict(bf16=ttnn.bfloat16, b8=ttnn.bfloat8_b, b4=ttnn.bfloat4_b, f32=ttnn.float32)
        self.FID = dict(HiFi4=ttnn.MathFidelity.HiFi4, HiFi3=ttnn.MathFidelity.HiFi3,
                        HiFi2=ttnn.MathFidelity.HiFi2, LoFi=ttnn.MathFidelity.LoFi)

    def mk(self, spec, dtype=None, layout=None):
        """spec = ("T", shape, "BFLOAT16"|..., "L1"|"DRAM"); random values, interleaved."""
        _, shape, dt, buf = spec
        dt = dtype or getattr(ttnn, {"BFLOAT16": "bfloat16", "BFLOAT8_B": "bfloat8_b", "BFLOAT4_B": "bfloat4_b",
                                     "FLOAT32": "float32"}[dt])
        mc = ttnn.L1_MEMORY_CONFIG if buf == "L1" else ttnn.DRAM_MEMORY_CONFIG
        return ttnn.from_torch(torch.randn(shape) * 0.25, dtype=dt, layout=layout or ttnn.TILE_LAYOUT,
                               device=self.dev, memory_config=mc)

    def ckc(self, base, fid=None, acc=None):
        f = fid or (base.get("math_fidelity") if base.get("math_fidelity") in self.FID else "HiFi4")
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=self.FID[f], math_approx_mode=bool(base.get("math_approx_mode", True)),
            fp32_dest_acc_en=bool(base.get("fp32_dest_acc_en")) if acc is None else acc,
            packer_l1_acc=bool(base.get("packer_l1_acc", True)))

    @staticmethod
    def free(t):
        try: ttnn.deallocate(t)
        except Exception: pass

    def time(self, fn):
        dev = self.dev
        out = fn(); ttnn.synchronize_device(dev)
        finite = bool(torch.isfinite(ttnn.to_torch(out).float()).all())
        self.free(out)
        self.free(fn()); ttnn.synchronize_device(dev)            # second warm call
        mode, tid, o = "trace", None, None
        try:
            tid = ttnn.begin_trace_capture(dev, cq_id=0); o = fn(); ttnn.end_trace_capture(dev, tid, cq_id=0)
            run = lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        except Exception as e:
            mode, tid = f"eager({type(e).__name__})", None
            run = lambda: self.free(fn())
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); run(); ttnn.synchronize_device(dev)
        k = max(1, min(200, math.ceil(self.target_s / max(time.perf_counter() - t0, 1e-6))))
        per = []
        for _ in range(self.reps):
            t0 = time.perf_counter()
            for _ in range(k): run()
            ttnn.synchronize_device(dev); per.append((time.perf_counter() - t0) / k)
        if tid is not None:
            ttnn.release_trace(dev, tid); self.free(o)
        per.sort(); med = per[len(per) // 2]
        return dict(us=med * 1e6, spread=(per[-1] - per[0]) / med, mode=mode, k=k, finite=finite)
