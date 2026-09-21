#!/usr/bin/env python3
"""Independent audit of the derived `_MM_BLOCK` key, run by the landing row before merging.

Deliberately not `allm-orchestrator`'s script and not `allm-safety`'s. Both replay the keys a
run-time census COUNTED, which answers "what moved on the targets we folded". This asks the
strictly wider question the merge turns on: over the ENTIRE domain the rule can derive, which
keys resolve differently from `origin/main`, counted or not. A key no census target happened to
present is still a key a user can present.

Nothing is retyped. Both tables and the derivation come out of the trees by AST.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAIN = "origin/main"
SAFETY = "origin/wk/allm-safety"


def git_show(ref, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def node_of(src, name, kind):
    for n in ast.parse(src).body:
        if kind is ast.Assign and isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n
        if kind is ast.FunctionDef and isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise SystemExit(f"no {name} in source")


def merged_resolver():
    """`_mm_block_at` as the MERGED WORKING TREE holds it, exec'd from its own source."""
    src = (REPO / "tt_bio" / "tenstorrent.py").read_text()
    table = ast.literal_eval(node_of(src, "_MM_BLOCK", ast.Assign).value)
    ns = {"_MM_BLOCK": table, "_MM_FUSED_STATS": [0, 0], "_MM_FUSED_DERIVED": {}}
    for fn in ("_mm_fused_block", "_mm_block_at"):
        exec(compile(ast.get_source_segment(src, node_of(src, fn, ast.FunctionDef)),
                     f"<{fn}>", "exec"), ns)
    return table, ns["_mm_block_at"]


def main_table():
    return ast.literal_eval(node_of(git_show(MAIN, "tt_bio/tenstorrent.py"),
                                    "_MM_BLOCK", ast.Assign).value)


def counted_keys():
    """Every (kt, nt) the run-time census counted, per model, from allm-safety's artifacts."""
    out = {}
    paths = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", SAFETY,
                            "perf/allm_safety/out/"], capture_output=True, text=True,
                           check=True).stdout.split()
    for p in paths:
        m = re.search(r"mmkeys_(.+?)\.json$", p)
        if not m:
            continue
        per = {}
        for k, calls in json.loads(git_show(SAFETY, p)).get("keys", {}).items():
            reader, key, _status = k.split("|")
            if reader != "tenstorrent":
                continue
            kt, nt = (int(x) for x in key.split(","))
            per[(kt, nt)] = per.get((kt, nt), 0) + calls
        out[m.group(1)] = per
    return out


def main():
    mt = main_table()
    ct, at = merged_resolver()
    bad = []

    # 1. Every key main's table wrote down must resolve to the IDENTICAL value, whether the
    #    merged table still writes it down or derives it. Byte-for-byte, no tolerance.
    print("== 1. main's %d literals reproduce ==" % len(mt))
    for k, v in sorted(mt.items()):
        got = at(*k)
        how = "kept" if k in ct else "derived"
        ok = got == v
        bad += [] if ok else [f"{k}: main {v} -> merged {got}"]
        print(f"   {str(k):10s} {str(v):18s} {how:8s} {'OK' if ok else 'MISMATCH'}")

    # 2. The whole domain the rule can reach, enumerated rather than sampled. Any key that
    #    main left unconfigured and the merge configures is listed by name.
    kts = sorted({k for k, _ in ct})
    reach = {}
    for kt in kts:
        for nt in range(1, 200):
            got = at(kt, nt)
            if got is not None:
                reach[(kt, nt)] = got
    new = {k: v for k, v in reach.items() if k not in mt}
    gone = {k: mt[k] for k in mt if at(*k) is None}
    print(f"\n== 2. full reachable domain: {len(reach)} keys, {len(new)} newly configured ==")
    for k, v in sorted(new.items()):
        print(f"   NEW {k} -> {v}")
    print(f"   withdrawn vs main: {sorted(gone) or 'none'}")
    if gone:
        bad.append(f"main keys no longer resolve: {sorted(gone)}")

    # 3. Self-pairing must be gone: a width may not concatenate with itself.
    print("\n== 3. self-pair negative control ==")
    for kt in kts:
        for a in sorted({n for (k, n) in ct if k == kt}):
            for nt in (2 * a, 2 * a + 1):
                if (kt, nt) in mt or (kt, nt) in ct:
                    continue                       # a genuine registered/fused key, not a self-pair
                got = at(kt, nt)
                ok = got is None or any(nt in (x + y, x + y + 1)
                                        for x in {n for (k, n) in ct if k == kt}
                                        for y in {n for (k, n) in ct if k == kt} if x != y)
                print(f"   ({kt}, {nt}) self-paired from {a} -> {got} {'OK' if ok else 'SELF-PAIRED'}")
                bad += [] if ok else [f"self-pair survives at ({kt}, {nt})"]

    # 4. Replay what seven models actually presented at run time.
    print("\n== 4. run-time census replay, %s ==" % SAFETY)
    moved_total = 0
    for model, keys in sorted(counted_keys().items()):
        moved = {k: (mt.get(k), at(*k)) for k in keys if mt.get(k) != at(*k)}
        calls = sum(keys[k] for k in moved)
        moved_total += calls
        print(f"   {model:18s} {len(keys):2d} distinct keys, {sum(keys.values()):5d} calls; "
              f"moved {len(moved)} keys / {calls} calls")
        for k, (was, now) in sorted(moved.items()):
            print(f"        {k}: {was} -> {now}   ({keys[k]} calls)")
    print(f"   TOTAL calls whose config changes: {moved_total}")

    # 5. Unpresented new keys: reachable, never counted. Named, because they are the risk.
    presented = {k for keys in counted_keys().values() for k in keys}
    print("\n== 5. newly configured but never presented by any census target ==")
    for k in sorted(set(new) - presented):
        print(f"   {k} -> {new[k]}")

    print("\nVERDICT:", "FAIL " + "; ".join(bad) if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
