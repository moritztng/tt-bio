# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T17:00:03Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE      8161 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2   20/20  COMPLETE      4684 s  arm=shipped_aa2 rc=0 card=1 at=2026-09-21T15:41:59Z
    permute       20/20  COMPLETE      2098 s  arm=permute rc=0 card=0 at=2026-09-21T16:25:05Z
    stale         20/20  COMPLETE      4987 s  arm=stale rc=0 card=0 at=2026-09-21T15:36:56Z
    norebind      20/20  COMPLETE      7619 s  arm=norebind rc=0 card=0 at=2026-09-21T14:53:04Z
    zero          20/20  COMPLETE      8182 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs        20/20  COMPLETE      2319 s  arm=theirs rc=0 side=theirs at=2026-09-21T16:21:24Z
    theirs_aa2    10/20  RUNNING        325 s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    163014 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs_aa2 --threads 10
    216348 bash -c cd /home/ttuser/.coworker/wt/of3t-trajwide && for A in zero norebind stale permute shipped_aa2; do echo "### $A"; timeout 900 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --score --arm $A --theirs-arm theirs --w0 own > /tmp/of3t/trajwide/score_$A.log 2>&1; echo "rc=$?"; done
    216349 timeout 900 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --score --arm zero --theirs-arm theirs --w0 own
    216350 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --score --arm zero --theirs-arm theirs --w0 own
