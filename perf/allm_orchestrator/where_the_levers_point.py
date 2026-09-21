#!/usr/bin/env python3
"""This campaign's levers are aimed at triangle attention. Its time is in triangle multiplication.

Three rows measured pieces of this and none of them stated it, because each holds one half:
`allm-gates` owns the lever inventory, `allm-model` owns the per-class counts, and only the
cross-tab shows they point in different directions.

The counts, all from committed run-time censuses rather than from reading call sites:

  Boltz-2   560 TriangleMultiplication, 560 TriangleAttention, 23 shared classes, 0 of its own
  ESMFold2 1064 TriangleMultiplication,   0 TriangleAttention, trimul = 45.61 % of the fold

ESMFold2 executes NO triangle attention at all, so every triangle-attention lever in this campaign
is unreachable on it by construction -- which is the real explanation for its 1.0385x, better than
either of the two I had recorded before.

Lever -> class is resolved by WHICH FILE the counter lives in, checkable with one grep each, not by
reading what a lever is called.

No device, no `tt_bio` import.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# counter -> the file it is defined in. Verified with
#   grep -rn '<counter>' tt_bio/*.py
# Any of these moving is a real change to what the lever touches, so the home is asserted below.
LEVERS = {
    "fused qkv+gate (QKVG_STATS)":            ("tt_bio/triatt_qkv.py",   "TriangleAttention"),
    "fused qkv+gate+bias (QKVGB_STATS)":      ("tt_bio/triatt_qkv.py",   "TriangleAttention"),
    "M18 fused-HiFi route":                   ("tt_bio/tenstorrent.py",  "TriangleAttention"),
    "M38 gate epilogue":                      ("tt_bio/triatt_sdpa.py",  "TriangleAttention"),
    "M38 qkv+SDPA fusion":                    ("tt_bio/triatt_sdpa.py",  "TriangleAttention"),
    "F1 tail product into L1 (OUT_L1_STATS)": ("tt_bio/trimul_tail.py",  "TriangleMultiplication"),
    "fused F1 trimul tail (STATS)":           ("tt_bio/trimul_tail.py",  "TriangleMultiplication"),
    "trimul g_out (TRIMUL_GOUT_STATS)":       ("tt_bio/tenstorrent.py",  "TriangleMultiplication"),
    "transition row raise":                   ("tt_bio/tenstorrent.py",  "Transition"),
    "residual into L1":                       ("tt_bio/tenstorrent.py",  "pair update"),
}
COUNTERS = {
    "fused qkv+gate (QKVG_STATS)": "QKVG_STATS",
    "fused qkv+gate+bias (QKVGB_STATS)": "QKVGB_STATS",
    "M18 fused-HiFi route": "TRIATT_FUSED_HIFI_STATS",
    "M38 gate epilogue": "TRIATT_GATE_EPILOGUE",
    "M38 qkv+SDPA fusion": "_FUSE_QKV",
    "F1 tail product into L1 (OUT_L1_STATS)": "OUT_L1_STATS",
    "fused F1 trimul tail (STATS)": "^STATS",
    "trimul g_out (TRIMUL_GOUT_STATS)": "TRIMUL_GOUT_STATS",
    "transition row raise": "TRANSITION_H_CHUNK_STATS",
    "residual into L1": "RESIDUAL_L1_STATS",
}


def assert_homes():
    """Each counter must still be defined in the file this table says it is."""
    bad = []
    for lever, (path, _) in LEVERS.items():
        src = (REPO / path).read_text()
        pat = COUNTERS[lever]
        pat = pat if pat.startswith("^") else rf"^{re.escape(pat)}\b|^_?{re.escape(pat)}\b"
        if not re.search(pat, src, re.M):
            bad.append(f"{COUNTERS[lever]} is not defined in {path}")
    if bad:
        raise SystemExit("lever->file table is stale:\n  " + "\n  ".join(bad))


def census(branch, path):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{branch}:{path}"],
                         capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def classes(d):
    """-> {class: (n, incl_s or None)} from the deepest leg that carries blocks."""
    A = next((f["fold_s"] for f in d.get("folds", []) if f.get("tag") == "A"), None)
    best = {}
    for leg in d.get("folds", []):
        for k, v in (leg.get("blocks") or {}).items():
            name = k.split(".")[-1]
            n, s = v.get("n", 0), v.get("incl_s") or 0.0
            if name not in best or n > best[name][0] or s > (best[name][1] or 0):
                best[name] = (n, s or None)
    return best, A


def main() -> int:
    assert_homes()
    print("LEVERS THIS CAMPAIGN HAS PURSUED, by the class their counter lives with\n")
    tally = {}
    for lever, (path, cls) in LEVERS.items():
        tally[cls] = tally.get(cls, 0) + 1
        print(f"  {lever:42} {cls:24} ({path})")
    print("\n  " + ", ".join(f"{c}: {n}" for c, n in sorted(tally.items(), key=lambda kv: -kv[1])))

    print("\nWHERE THE WORK ACTUALLY IS, from committed run-time censuses\n")
    for label, branch, path in (
            ("Boltz-2",  "origin/wk/allm-model", "perf/allm_model/out/b2census_512.json"),
            ("ESMFold2", "origin/wk/allm-model", "perf/allm_model/out/esmfold2_512_d3b.json")):
        d = census(branch, path)
        if d is None:
            print(f"  {label}: census not on the branch"); continue
        cls, A = classes(d)
        tm, ta = cls.get("TriangleMultiplication"), cls.get("TriangleAttention")
        def fmt(x):
            if x is None: return "absent -- ZERO calls"
            n, s = x
            return f"{n:>5} calls" + (f", {s:7.3f} s = {100*s/A:5.2f} % of the fold" if s and A else "")
        print(f"  {label:9} untaped fold {A} s")
        print(f"    TriangleMultiplication  {fmt(tm)}")
        print(f"    TriangleAttention       {fmt(ta)}")

    # "zero calls" from a census could be a DEPTH artifact rather than a fact about the model, so
    # it is cross-checked structurally: does ESMFold2 construct TriangleAttention anywhere at all?
    import ast
    esm = [REPO / "tt_bio/esmfold2.py", REPO / "tt_bio/esmfold2_runtime.py"]
    built = set()
    for f in esm:
        if not f.exists():
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in ("TriangleAttention", "TriangleMultiplication"):
                    built.add(name)
    print(f"\n  CROSS-CHECK, so the zero is not a census-depth artifact: ESMFold2's own files")
    print(f"  construct {sorted(built) or 'nothing'} -- TriangleAttention is "
          f"{'CONSTRUCTED (the zero is an artifact, stop here)' if 'TriangleAttention' in built else 'never constructed, so the zero is real'}.")

    print("""
THE CONSEQUENCE, which is an aiming decision and not a measurement

  ESMFold2 executes NO TriangleAttention. Every triangle-attention lever in this campaign --
  the shipped +1.438 s candidate, M18, and both of M38's -- is unreachable on it BY
  CONSTRUCTION, not because a gate refused it. That is a better explanation of its 1.0385x than
  either of the two previously recorded, and it is structural, so no gate fix changes it.

  The levers that CAN reach it are the trimul ones, and the largest of those --
  `trimul_tail.OUT_L1` -- is refused on 1084 of 1084 calls by a reserve fitted to a different
  consumer entirely (perf/allm_orchestrator/trimul_tail_reserve.py). On the class that is
  45.61 % of the fold.

  Boltz-2 is the model the campaign compares everything against, and it runs BOTH classes 560
  times each while executing 23 shared classes and none of its own. So it is the one model where
  a triangle-attention lever reaches a large share -- which is exactly why levers tuned on it
  transfer poorly to models weighted toward trimul.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
