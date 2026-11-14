"""What t does --partial 0.4167 actually start the released recipe at?"""
from tt_bio.rfd4 import sampler as s

spec = s.recycle_strategy("medium/front_loaded")
print("nsteps", spec.nsteps, "modes", list(spec.data_modes))
start = s.resolve_partial_start(spec, partial_trajectory_fraction_steps=0.4166666666666667)
print("resolved t", start.t, "| remaining steps", start.nsteps, "| from_fraction", start.from_fraction)
ts, _ = s.sample_schedule(spec, spec.nsteps)
k = spec.nsteps - start.nsteps
for dm in spec.data_modes:
    v = [round(float(x), 4) for x in ts[dm]]
    print(f"{dm}: t[k={k}]={v[k]}  schedule={v}")
