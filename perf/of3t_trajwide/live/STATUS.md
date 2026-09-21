# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T13:11:42Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped        3/20  RUNNING          3 s  
    shipped_aa2    0/20  not started   None s  
    permute        0/20  not started   None s  
    stale          0/20  not started   None s  
    norebind       0/20  not started   None s  
    zero           3/20  RUNNING         27 s  
    theirs         0/20  starting       531 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): not yet read

Live `trajwide.py` processes:

    59900 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    60332 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    60724 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
    60857 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    61289 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
