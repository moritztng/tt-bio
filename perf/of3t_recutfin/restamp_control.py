#!/usr/bin/env python3
"""of3t-recutfin job 1, control: the restamped artifact against the one of3t-recut banked.

The stamp is additive by construction, so the claim to test is that it is additive IN FACT. Two
readers, because one of them could be wrong about its own arithmetic:

  BYTE      strip `injection` from the new document, re-serialise it the way model_scope.py
            serialises, and demand the bytes equal the banked file. This covers every key at
            once, including ones nobody thought to list.
  VALUE     walk both documents and compare every leaf by repr, so a float that differs in its
            last bit is a difference and not a rounding of the report.

Then R164's seven contract keys are read off the new file explicitly, six of them against the
banked values, because those are the ones the charter's clause dereferences and a control that
only reports an aggregate would not say which key moved.

The new file is installed over the banked one ONLY if every check passes. A difference is a
finding, not a fix.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# R164's contract, `perf/of3t_orchestrator/clausestatus/REPOINT_CONTRACT.json`
CONTRACT_KEYS = [
    "inputs.float64.sha256",
    "stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2",
    "stats.renorm_vs_FLOAT64.n_over_per_tensor_bar",
    "stats.UPSTREAM_BF16_vs_FLOAT64.n_over_per_tensor_bar",
    "coverage_total.pct_of_model_compared",
    "bars.A26_reachable_bar_vs_their_bf16",
    "injection.convention",
]
# the brief's table, and the banked digest of the correction the injected scope was driven by
EXPECT_SCOPES = {
    "renorm:diffusion": "not_injected",
    "renorm:input_embedder": "not_injected",
    "renorm:cond": "not_injected",
    "renorm:aux": "not_injected",
    "renorm:msa": "not_injected",
    "renorm:pairformer_stack": "graph-cut-external",
}
COTANGENTS = REPO / "perf/of3t_recut/COTANGENTS.json"


def dig(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def walk(a, b, path=""):
    """Every leaf, compared by repr. Returns a list of differences."""
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [{"path": path, "banked": repr(a)[:120], "new": repr(b)[:120],
                 "why": "different types"}]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append({"path": f"{path}.{k}", "banked": "<absent>", "new": repr(b[k])[:120]})
            elif k not in b:
                out.append({"path": f"{path}.{k}", "banked": repr(a[k])[:120], "new": "<absent>"})
            else:
                out += walk(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": path, "banked": f"list of {len(a)}", "new": f"list of {len(b)}"}]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += walk(x, y, f"{path}[{i}]")
        return out
    return [] if repr(a) == repr(b) else [{"path": path, "banked": repr(a), "new": repr(b)}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--banked", required=True, type=Path)
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()

    banked_text = args.banked.read_text()
    banked = json.loads(banked_text)
    new = json.loads(args.new.read_text())

    added = [k for k in new if k not in banked]
    removed = [k for k in banked if k not in new]
    stripped = {k: v for k, v in new.items() if k != "injection"}
    byte_identical = json.dumps(stripped, indent=2) == banked_text
    diffs = walk(banked, stripped)

    inj = new.get("injection")
    scopes = {k: v["convention"] for k, v in (inj or {}).get("by_scope", {}).items()}
    cot = json.loads(COTANGENTS.read_text())
    want_sha = cot["out"]["cot_external.pt"]["sha256"]
    got_sha = ((inj or {}).get("by_scope", {}).get("renorm:pairformer_stack", {})
               .get("correction", {}).get("sha256"))

    keys = []
    for k in CONTRACT_KEYS:
        found_n, vn = dig(new, k)
        found_b, vb = dig(banked, k)
        keys.append({"key": k, "present_in_new": found_n, "new": vn,
                     "present_in_banked": found_b, "banked": vb,
                     "unchanged": (found_b and repr(vn) == repr(vb)) or
                                  (not found_b and k == "injection.convention")})

    checks = {
        "only_key_added_is_injection": added == ["injection"],
        "no_key_removed": removed == [],
        "byte_identical_once_the_stamp_is_stripped": byte_identical,
        "no_leaf_differs_by_repr": diffs == [],
        "every_contract_key_present": all(k["present_in_new"] for k in keys),
        "the_six_existing_values_are_unchanged":
            all(k["unchanged"] for k in keys if k["key"] != "injection.convention"),
        "convention_reads_graph_cut_external":
            (inj or {}).get("convention") == "graph-cut-external",
        "per_scope_map_is_the_briefs_table": scopes == EXPECT_SCOPES,
        "exactly_one_scope_is_injected": (inj or {}).get("n_injected") == 1,
        "correction_digest_is_the_banked_cot_external": got_sha == want_sha,
    }
    ok = all(checks.values())

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "row": "of3t-recutfin",
        "device_involved": False,
        "why_no_aiclk": "CPU only; this compares two JSON documents",
        "banked": {"path": str(args.banked), "bytes": len(banked_text)},
        "new": {"path": str(args.new), "bytes": args.new.stat().st_size},
        "keys_added": added, "keys_removed": removed,
        "n_leaves_differing": len(diffs), "differences": diffs[:20],
        "CONTRACT_KEYS": keys,
        "injection": inj,
        "correction_digest": {"expected_from_COTANGENTS.json": want_sha, "computed_by_the_composer": got_sha},
        "checks": checks,
        "verdict": ("the stamp is additive: every value the banked artifact carried comes back "
                    "bit-identical and the only new key is `injection`"
                    if ok else "FAILED -- see `differences` and `checks`. A difference in a "
                               "banked value is a finding, not a fix: do NOT install."),
    }
    out = REPO / "perf/of3t_recutfin/RESTAMP_CONTROL.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "CONTRACT_KEYS"}, indent=1)[:4000])
    for k in keys:
        print(f"  {k['key']:58s} {k['new']!r} unchanged={k['unchanged']}")

    if ok and args.install:
        shutil.copyfile(args.new, args.banked)
        print(f"installed {args.new} -> {args.banked}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
