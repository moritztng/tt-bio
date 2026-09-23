#!/usr/bin/env python3
"""HEAD wins a conflict in a perf namespace the merging row does not own.

`sibling-perf-campaigns-need-namespaced-output-paths` is the campaign's ownership rule: a row
writes `perf/of3t_<its own row>/` and nothing else. So a conflict on `perf/of3t_rankunify/*`
while merging `of3t-apbback` is not a disagreement about anything -- it is a stale copy that
apbback's branch carried along from whatever `wk/of3t` looked like when it forked, meeting the
version main has since landed.

HEAD is the composition, which is rebuilt from `origin/main` every pass, so HEAD is the newer
and the shipped one. Taking it loses nothing: the owning row's branch still has its own copy,
and if the owning row is itself in the composition its merge writes the file directly rather
than through this rule.

Refuses with exit 2 unless the path is `perf/of3t_<something>/...` AND that something is not the
merging row. Anything under `tt_bio/`, `tests/`, `docs/` or `scripts/` is out of scope by
construction -- those are shared and a conflict there is real.

Usage: resolve_foreign_namespace_copy.py <file> <their-ref> <merging-row>  -> 0 resolved, 2 no.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

NS = re.compile(r"^perf/of3t_([A-Za-z0-9]+)/")


def resolve(path: pathlib.Path, theirs_ref: str, row: str) -> int:
    m = NS.match(path.as_posix())
    if not m:
        return 2
    owner = m.group(1)
    if owner.lower() == row.replace("-", "").lower():
        return 2                      # the row's own namespace: a real conflict, not a carry
    ours = subprocess.run(["git", "show", f":2:{path.as_posix()}"], capture_output=True)
    if ours.returncode != 0:
        print(f"  {path}: no HEAD side to take (added only by the row) -- not a stale carry",
              file=sys.stderr)
        return 2
    path.write_bytes(ours.stdout)
    subprocess.run(["git", "add", path.as_posix()], check=True)
    print(f"  {path.as_posix()}: HEAD kept -- perf/of3t_{owner}/ is not of3t-{row}'s namespace, "
          f"so this is a stale copy carried by its branch, not a disagreement")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(resolve(pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
