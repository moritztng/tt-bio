F re-derived from c10-fixed-cost's raw rows, 2026-09-17, by c12-orchestrator pass 2.

512 aa: F = 3.9543 s, W = 14,678.2 Mcycles (record 3.9830 +-0.1181 s, 14,665.0 Mc) -- inside bound.
298 aa: F = 1.9154 s, W = 10,432.2 Mcycles (record 1.9500 +-0.0438 s, 10,403.4 Mc) -- inside bound.

Leave-one-out envelope on F: 3.5727-4.4746 s at 512 aa, 1.8205-2.0933 s at 298 aa. Residuals are
structured, not noise: +0.25, -0.44, -0.48, +0.69 % across 800/1000/1200/1350 MHz, a convex pattern,
so fold = F + W/MHz is a slight misfit and F's honest uncertainty is the LOO envelope (~+-0.45 s at
512 aa), not the fit's standard error.

THE HOST SPIN-WAITS. host_cpu_s runs 108-113 % of wall at both sizes (more CPU than wall = threads),
and host_utime_s drops 7.4477 s across the 800->1350 MHz range while wall drops 7.4268 s -- a ratio
of 1.003. Host user CPU tracks the CLOCK-SCALED term almost exactly 1:1, which is what a spin-wait on
the device looks like. Consequence for any F decomposition: host CPU time is NOT a measure of F, and
a CPU-sampling profiler will book device wait onto whichever host frame is spinning. The 1350-vs-800
two-clock split is the only discriminator, and an item whose measured time moves with the clock is a
spin, not host work.

Also from the same artifact: `model_meta.timed_region` = "predict_one (featurize + fold + CIF
write)", and `model_meta.phase_times` is present but **null** -- the harness has the field and
nothing populates it, so the first three-way split of F is instrumentation of a field that already
exists.
