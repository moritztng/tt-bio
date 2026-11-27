"""G4 PERF grade from a trunk_step log (protocol card D55): full step = step_s, steps 6..N, GPU side from
r4t-gpu-baseline's per-depth fit (0.131 + 0.022 x n_recycle s, B200 winner arm) at the SAME per-step depths.
Ratio = (32 S_bh / 160000) / (8 S_gpu / 562154) = 14.05 x S_bh / S_gpu, S = examples/s (1 example per step).
    perfgrade.py LOG [--first 6]"""
import json, sys, statistics as st
log = sys.argv[1]; first = int(sys.argv[3]) if len(sys.argv) > 3 else 6
R = [json.loads(l) for l in open(log) if l.startswith('{"step"')]
R = [r for r in R if r["step"] >= first]
assert R, "no steps"
n = len(R); bh = [r["step_s"] for r in R]; dep = [r["depth_upstream"] for r in R]
gpu = [0.131 + 0.022 * d for d in dep]
clk = [c for r in R for c in r["aiclk"]]
opt = [r["opt_s"] for r in R]
S_bh, S_gpu = n / sum(bh), n / sum(gpu)
k = 32 * 562154 / (8 * 160000)
out = dict(log=log, n=n, steps=[R[0]["step"], R[-1]["step"]], mean_depth=round(st.mean(dep), 3),
           bh_step_mean=round(st.mean(bh), 4), bh_step_median=round(st.median(bh), 4), bh_min=min(bh), bh_max=max(bh),
           fwd_mean=round(st.mean(r["fwd_s"] for r in R), 4), bwd_mean=round(st.mean(r["bwd_s"] for r in R), 4),
           opt_mean=round(st.mean(opt), 4), fold_bwd_mean=round(st.mean(r["fold_bwd_s"] for r in R), 4),
           rebuild_mean=round(st.mean(r["rebuild_s"] for r in R), 4),
           gpu_step_mean=round(st.mean(gpu), 4), S_bh=round(S_bh, 4), S_gpu=round(S_gpu, 3),
           ratio_per_chip=round(k * S_bh / S_gpu, 4), ratio_at_0p85=round(0.85 * k * S_bh / S_gpu, 4),
           # optimizer at ZeRO-1 share over 32 ranks (opt / 32), everything else as measured
           ratio_opt_sharded_per_chip=round(k * n / sum(b - o * (1 - 1 / 32) for b, o in zip(bh, opt)) / S_gpu, 4),
           aiclk=[min(clk), max(clk)], aiclk_below_1200=sum(c < 1200 for c in clk),
           digest_last=R[-1]["master_digest"], loss_finite=all(r["loss"] == r["loss"] for r in R))
print(json.dumps(out, indent=1))
