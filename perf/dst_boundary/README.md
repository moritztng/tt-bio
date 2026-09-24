# Lease renewals under load, four Wormhole Galaxies

2026-09-24, `examples/many_hosts.py` driving the four dev Galaxies (172.16.102.103/104/107/108,
127 usable chips) through each one's loopback controller, engine b7087510e (heartbeat every 8 s).
`hb_sample.py` ran on each host and logged every distinct `lease_until` of the run's jobs;
`hb_reduce.py` turns those into gaps.

    python3 ../../examples/many_hosts.py inputs out --model esmfold2 --owner dst-boundary \
        --tt-bio '. ~/japanfold/env.sh && tt-bio' --host ubuntu@172.16.102.103:8767 ...

Result (`hb_summary.txt`): 8 jobs, 540 renewals, median gap 8.1 s, longest 14.19 s, longest
job held 1204.7 s. `jobs_ledger.txt` is the controllers' own record: all 8 `ok`, attempts 1.
Wall time for the whole run 21 min 19 s (`example_run.log`).
