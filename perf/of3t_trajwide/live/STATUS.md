# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T13:23:52Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped        7/20  RUNNING         69 s  
    shipped_aa2    0/20  not started   None s  
    permute        0/20  not started   None s  
    stale          0/20  not started   None s  
    norebind       0/20  not started   None s  
    zero           7/20  RUNNING        160 s  
    theirs         2/20  RUNNING        204 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    59900 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    60332 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    60724 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
    60857 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    61289 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
