# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T14:43:25Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       19/20  RUNNING        127 s  
    shipped_aa2    0/20  not started   None s  
    permute        0/20  not started   None s  
    stale          0/20  not started   None s  
    norebind       0/20  not started   None s  
    zero          19/20  RUNNING        155 s  
    theirs         6/20  RUNNING        483 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    26558 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    26568 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
    26605 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm zero --threads 3
    26606 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped --threads 3
    37142 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
