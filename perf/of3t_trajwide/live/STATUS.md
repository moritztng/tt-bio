# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T17:39:31Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE     10528 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2   20/20  COMPLETE      7052 s  arm=shipped_aa2 rc=0 card=1 at=2026-09-21T15:41:59Z
    permute       20/20  COMPLETE      4466 s  arm=permute rc=0 card=0 at=2026-09-21T16:25:05Z
    stale         20/20  COMPLETE      7355 s  arm=stale rc=0 card=0 at=2026-09-21T15:36:56Z
    norebind      20/20  COMPLETE      9987 s  arm=norebind rc=0 card=0 at=2026-09-21T14:53:04Z
    zero          20/20  COMPLETE     10550 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs        20/20  COMPLETE      4687 s  arm=theirs rc=0 side=theirs at=2026-09-21T16:21:24Z
    theirs_aa2    19/20  RUNNING        154 s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    163014 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs_aa2 --threads 10
