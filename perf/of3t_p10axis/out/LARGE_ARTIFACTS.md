# Seven arm artifacts kept in history, not in the tree

Each is 11.9-17.9 MB, over the 10 MB tracked-file limit in `tests/test_repo_size_guard.py`.
They were committed in 93577b599 and are unchanged since. Read one with
`git show 93577b599:perf/of3t_p10axis/out/<file>`.

| file | bytes | sha256 |
|---|---|---|
| arm_fixed_ab.json | 18754444 | cc4040c46502f598 |
| arm_fixed_f1.json | 14852823 | 284d4d977939467c |
| arm_floorrepro_384.json | 18754660 | 2002da61ffd76974 |
| arm_qb2repro_384.json | 17453886 | a5a73d8e0197df12 |
| arm_s1_384.json | 12504291 | 518b607eb421583d |
| arm_s2_384.json | 12504346 | 6b69256ea4607069 |
| arm_unfixed_ab.json | 17453755 | d8cf5efa833f1a45 |
