# Why D10+D24 landed with the size-ladder arm red

The arm is red on `origin/main` itself. Both readings are protenix-v1, ten rungs, card p150a:

    branch  wk/land-standing @ 12fb56480   24 per-rung FAIL lines
    main    detached e1d887a54, no branch  24 per-rung FAIL lines

    md5 of the sorted `FAIL protenix-v1/<rung> ...` lines, both runs:
    444db3e22175380daa205ca5a534e4be

Identical. A gate arm that fails on the unmodified reference cannot discriminate the change.

The timing half is the only part that differs, and it goes the other way: main's control PASSES
it (exponent 512->768 reads 1.65 against a recorded 1.52, inside the +-0.50 band, and all ten
rungs reproduce the baseline within 8%), while the branch's 40644 s qb1 run read 2.07. That run
shared the host with `cov-ladder-p150a-p3`'s orphaned re-record chains at 746 %CPU. The tree
effect on the same rung, measured on a quiet qb2 with the clock sampled DURING every leg, is
+0.050 s, which is 0.17x the arm's own within-arm spread
(`../ladder_attrib/protenix_tree_ab_clocked.json`).

Files here:
- `ladder_main_control_e1d887a54_qb1_p150a.txt` — the control, `rc=1 wall=1618s`
- `pytest_mainctl_e1d887a54.txt` — the same 7 test failures the merged tree has, on main
- `pytest_ranking_tip.txt` — the ranking tests at the landed tip, 20 passed
- `fill_reasons.txt` — `--size-ladder-fill-reasons`: 1 of 56 carried, 55 need a human reason
