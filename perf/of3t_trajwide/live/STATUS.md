# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T14:49:30Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE       327 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2    1/20  RUNNING        126 s  
    permute        0/20  not started   None s  
    stale          0/20  not started   None s  
    norebind       8/20  RUNNING         12 s  
    zero          20/20  COMPLETE       349 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs         7/20  RUNNING        293 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    37142 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    88047 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm norebind --threads 3
    88113 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm norebind --threads 3
    88301 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
    88371 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
