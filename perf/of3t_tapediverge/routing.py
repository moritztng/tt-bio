#!/usr/bin/env python3
"""D32's site table, re-derived from THIS tree instead of quoted.

D32 lists 21 `ops.taping()` branch points in 9 shipped modules. Line numbers move -- this
campaign already has a provenance note about exactly that -- so the count is re-taken by pattern
and every hit is classified, because a grep total counts the predicate's own definition, the
comments that discuss it and the helper that wraps it, and none of those route anything.

    routing.py [--out perf/of3t_tapediverge/ROUTING.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

BRANCH = re.compile(r"(?:^|[\s(])(?:not\s+)?(?:ops\.|_ops_l1\.)?_?taping\(\)")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def classify(path: str, lineno: int, text: str) -> str:
    s = text.strip()
    if s.startswith("#") or s.startswith('"') or s.startswith("*"):
        return "comment"
    if re.match(r"\s*def\s+_?taping\b", text):
        return "definition"
    if path.endswith("ops.py") and "def taping" in text:
        return "definition"
    if re.search(r"\b(if|and|or|while|return|=)\b", s) and BRANCH.search(s):
        # `return ops.taping()` inside the eltwise helper is the helper's BODY, not a branch:
        # its three callers are the branch points and counting it as well double-counts one.
        if s.startswith("return ") and path.endswith("eltwise_fusion.py"):
            return "helper_body"
        return "branch"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_tapediverge/ROUTING.json")
    a = ap.parse_args()

    out = subprocess.run(["grep", "-rn", "taping()", "tt_bio/", "--include=*.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        path, lineno, text = line.split(":", 2)
        hits.append({"file": path, "line": int(lineno), "kind": classify(path, int(lineno), text),
                     "text": text.strip()})

    branches = [h for h in hits if h["kind"] == "branch"]
    mods = sorted({h["file"] for h in branches})
    by_mod = {m: sorted(h["line"] for h in branches if h["file"] == m) for m in mods}

    # D32's published table, so the drift is reported rather than silently absorbed.
    D32 = {
        "tt_bio/triatt_qkv.py": [74, 179, 276, 390],
        "tt_bio/tenstorrent.py": [1126, 4129, 4595, 8531],
        "tt_bio/eltwise_fusion.py": [80, 104, 115],
        "tt_bio/reblock_permute.py": [373, 597, 877],
        "tt_bio/softmax_generic.py": [369, 514],
        "tt_bio/triatt_sdpa.py": [340, 486],
        "tt_bio/trimul_tail.py": [233],
        "tt_bio/mm_dualnoc.py": [87],
        "tt_bio/swiglu_fused.py": [94],
    }
    drift = {m: {"published": D32.get(m, []), "found": by_mod[m],
                 "moved": sorted(set(D32.get(m, [])) ^ set(by_mod[m]))}
             for m in mods}

    rep = {
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "raw_grep_hits": len(hits),
        "branch_points": len(branches),
        "modules": len(mods),
        "by_module": by_mod,
        "non_branch": [h for h in hits if h["kind"] != "branch"],
        "vs_D32_published": drift,
        "count_matches_D32": len(branches) == sum(len(v) for v in D32.values()),
        "modules_match_D32": sorted(mods) == sorted(D32),
    }
    (ROOT / a.out).write_text(json.dumps(rep, indent=1))
    print(f"branch points {len(branches)} in {len(mods)} modules "
          f"(raw grep {len(hits)}; {len(hits)-len(branches)} are comments, the predicate's own "
          f"definition or the eltwise helper body)")
    for m in mods:
        pub = D32.get(m, [])
        flag = "" if pub == by_mod[m] else f"   <- D32 published {pub}"
        print(f"  {m:28s} {by_mod[m]}{flag}")
    print(f"count matches D32: {rep['count_matches_D32']}; modules match: {rep['modules_match_D32']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
