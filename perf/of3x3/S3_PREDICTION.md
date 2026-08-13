# S3 prediction, written BEFORE the fold A/B (exec pass 1)

Measured inputs: op 55.3 ms/call over 488 calls; the tail at 12-row blocks is 1.0535 ms
interleaved and 0.6436 ms sharded (perf/of3x3/screen_l1_chain.py), 42.67 blocks per 512-row call.

    tail today       1.0535 x 42.67 = 44.95 ms/call
    tail sharded     0.6436 x 42.67 = 27.46 ms/call        -17.5 ms/call
    row-block loop   q/k/v slices (1.5 units) + concat of o (0.5 units) = 2 units
                     = 2.15 GB at 392 GB/s = +5.5 ms/call
    net                                       43.3 ms/call

    op    488 x 43.3 ms = 21.1 s   (from 26.980)
    fold  51.043 - 5.86 = 45.2 s   ratio 153.012/45.2 = 3.386x

Band, charging the loop overhead anywhere from 0 to 8 ms/call: fold 43.9-47.1 s, 3.25x-3.48x.
Pass mark is 51.004 s; the task's own target is below 50.00 s.
