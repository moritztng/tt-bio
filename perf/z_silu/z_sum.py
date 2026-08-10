import json, sys, glob, os
for f in sys.argv[1:]:
    if not os.path.exists(f): print("MISSING", f); continue
    d = json.load(open(f))
    print("%-22s %-5s fused=%8.3f bare=%7.3f silu=%8.3f us/tile=%.8f mean=%.9f load=%s" % (
        os.path.basename(f), d['arm'], d['fused_med_us'], d['bare_med_us'], d['silu_us'],
        d['silu_us_per_tile'], d['out_mean_abs'], d['load'][0]))
