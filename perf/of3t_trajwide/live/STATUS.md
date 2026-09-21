# of3t-trajwide live run

Written by `recorder.py` at 2026-09-21T15:01:40Z.
Check liveness by the rung counter moving and `log_age_s` staying small, not by
the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.

    arm           rungs  status      log age  marker
    shipped       20/20  COMPLETE      1057 s  arm=shipped rc=0 card=1 at=2026-09-21T14:44:02Z
    shipped_aa2    6/20  RUNNING         63 s  
    permute        0/20  not started   None s  
    stale          3/20  RUNNING         40 s  
    norebind      20/20  COMPLETE       515 s  arm=norebind rc=0 card=0 at=2026-09-21T14:53:04Z
    zero          20/20  COMPLETE      1078 s  arm=zero rc=0 card=0 at=2026-09-21T14:43:41Z
    theirs         8/20  RUNNING        536 s  
    theirs_aa2     0/20  not started   None s  

Reference tree resolved in-process (D149): /home/ttuser/of3t_refprec/of3pkg043

Live `trajwide.py` processes:

    37142 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side theirs --arm theirs --threads 10 --aa-in-process
    88301 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
    88371 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm shipped_aa2 --threads 3
    101686 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm stale --threads 3
    101728 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --side ours --arm stale --threads 3
    117704 bash -c cd /home/ttuser/.coworker/wt/of3t-trajwide && nice -n 15 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --score --arm shipped --w0 own --threads 2 --out /tmp/of3t/trajwide/partial_shipped_k8.json 2>&1 | tail -70
    117705 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trajwide/trajwide.py --score --arm shipped --w0 own --threads 2 --out /tmp/of3t/trajwide/partial_shipped_k8.json
