# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T16:05:26Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE      4883 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2   20/20  COMPLETE      1406 s  arm=shipped_aa2 rc=0 card=1 at=2026-09-21T15:41:59Z
    permute       11/20  RUNNING         77 s  
    stale         20/20  COMPLETE      1710 s  arm=stale rc=0 card=0 at=2026-09-21T15:36:56Z
    norebind      20/20  COMPLETE      4341 s  arm=norebind rc=0 card=0 at=2026-09-21T14:53:04Z
    zero          20/20  COMPLETE      4905 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs        17/20  RUNNING        227 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    37142 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    144368 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm permute --threads 3
    144435 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm permute --threads 3
