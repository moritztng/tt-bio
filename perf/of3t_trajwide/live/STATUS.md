# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T19:49:58Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE     18355 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2   20/20  COMPLETE     14878 s  arm=shipped_aa2 rc=0 card=1 at=2026-09-21T15:41:59Z
    permute       20/20  COMPLETE     12292 s  arm=permute rc=0 card=0 at=2026-09-21T16:25:05Z
    stale         20/20  COMPLETE     15181 s  arm=stale rc=0 card=0 at=2026-09-21T15:36:56Z
    norebind      20/20  COMPLETE     17813 s  arm=norebind rc=0 card=0 at=2026-09-21T14:53:04Z
    zero          20/20  COMPLETE     18377 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs        20/20  COMPLETE     12513 s  arm=theirs rc=0 side=theirs at=2026-09-21T16:21:24Z
    theirs_aa2    20/20  COMPLETE      7761 s  arm=theirs_aa2 rc=0 side=theirs at=2026-09-21T17:40:36Z

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    none
