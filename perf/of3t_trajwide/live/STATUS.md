# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T14:46:27Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE       144 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2    0/20  starting       124 s  
    permute        0/20  not started   None s  
    stale          0/20  not started   None s  
    norebind       0/20  starting       146 s  
    zero          20/20  COMPLETE       166 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs         7/20  RUNNING        110 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    37142 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    88047 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm norebind --threads 3
    88113 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm norebind --threads 3
    88301 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
    88371 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
