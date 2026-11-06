# WH trunk flag stack (trsab) and its fast grade (gradewts), read 2026-11-06 07Z

Tree c02150635 (= main 2a6fc51db), bg_R3, seed 0, 6 designs, medians of designs 2,3,5,6. Host .108, AICLK 1000 MHz on
the run chips, host load 18 -> 33 across the arms (diffusion ms_step drifts with it; the trunk loop is device-bound).
All ten runs rc 0, CIF md5 deterministic within each arm.

| mode   | arm | levers on top of the shipped set                   | loop s | trunk s | md5        |
|--------|-----|----------------------------------------------------|--------|---------|------------|
| normal | b1  | -                                                  | 12.24  | 13.40   | 14916b5c6a |
| normal | s1  | trimul_sig6, trimul_t1w8                           | 12.06  | 13.21   | 5b63fb5fa0 |
| normal | b2  | -                                                  | 12.25  | 13.56   | 14916b5c6a |
| normal | s2  | trimul_sig6, trimul_t1w8                           | 12.06  | 13.23   | 5b63fb5fa0 |
| fast   | b1  | -                                                  | 11.92  | 13.01   | 914d232eed |
| fast   | s1  | trimul_sig6, triatt_ln_b8, trimul_t1w8             | 11.83  | 12.87   | 08c285fe62 |
| fast   | a1  | s + the six trlw levers that were dead alone       | 11.41  | 12.49   | 3b3f128a35 |
| fast   | b2  | -                                                  | 11.92  | 13.12   | 914d232eed |
| fast   | s2  | trimul_sig6, triatt_ln_b8, trimul_t1w8             | 11.99  | 13.29   | 08c285fe62 |
| fast   | a2  | s + the six                                        | 11.50  | 12.64   | 3b3f128a35 |

Pre-registered bar: keep s at loop <= b - 0.3 s in both pairs. s misses in both modes (normal -0.18/-0.19, fast
-0.09/+0.07): killed. a clears it in both fast pairs (-0.51/-0.42) and its md5 differs from s, so a lever that only acts
beside triatt_ln_b8 carries the gain. triatt_ln_b8g is that lever (bfp8 gate out of the fused LayerNorm; Boltz-2 and
OpenDDE ship the pair in WH fast). Next: trgab (b, g = s + triatt_ln_b8g, a) and gradewtg (g, seeds 0-4).

gradewts (fast s, seeds 0-4, vs TT_BIO_LEVERS=none wh-ee0-7, perf/spd/bg_unpaired.py): design_iptm +0.0036
[-0.0230,+0.0299], design_ptm +0.0238 [+0.0048,+0.0435], complex_plddt +0.0003 [-0.0056,+0.0063]. Passes the fast bar,
moot for s itself; it says the bfp8 q|k|v plus the two trimul flags cost no measurable design quality.
