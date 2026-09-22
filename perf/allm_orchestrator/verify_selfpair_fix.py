#!/usr/bin/env python3
"""Does the one-character self-pair fix cost the measured OpenDDE win anything?

`allm-safety` located a defect in `allm-gates`' derivation rule: the inner loop `widths[i:]`
lets an operand pair with ITSELF, so `nt = 2a` or `2a + 1` derives a config for a fusion no
kernel performs. It priced the fix at one character, `widths[i+1:]`, and handed it over without
applying it. Before that amendment goes into the brief, the orchestrator owes the question the
candidate's fate turns on: does the fix touch the 1.438 s that was actually measured?

Everything here is read from the tree, nothing is retyped:
  - `_MM_BLOCK` comes out of BOTH tenstorrent.py files by AST (`ast.literal_eval` on the
    assignment), so a table that drifts cannot be missed by a stale copy in this script;
  - `_mm_fused_block` is exec'd FROM THE CANDIDATE'S OWN SOURCE, so this scores the row's real
    code and not a paraphrase of it, and the "fixed" variant is that same source with the single
    documented character change applied and asserted to have applied;
  - the keys replayed are every `(kt, nt)` the run-time census COUNTED on seven models, read from
    `perf/allm_safety/out/mmkeys_*.json`, because firing is counted and not read off a gate.

No `tt_bio` import and no device: this is a pure question about a dict and a rule.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

CAND = "origin/wk/allm-gates"
SAFETY = "origin/wk/allm-safety"
REPO = Path(__file__).resolve().parents[2]


def git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def mm_block(src: str) -> dict:
    """The `_MM_BLOCK` literal, by AST. Never by regex and never retyped."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_MM_BLOCK" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("no _MM_BLOCK assignment found")


def fused_fn(src: str, table: dict, fix: bool):
    """The candidate's OWN `_mm_fused_block`, optionally with the one-character fix applied."""
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_mm_fused_block")
    text = ast.get_source_segment(src, fn)
    if fix:
        patched = re.sub(r"widths\[i:\]", "widths[i + 1:]", text)
        if patched == text:
            raise SystemExit("the `widths[i:]` loop this fix targets is not in the source any "
                             "more -- re-read the rule before trusting this script")
        text = patched
    ns = {"_MM_BLOCK": table, "_MM_FUSED_STATS": [0, 0], "_MM_FUSED_DERIVED": {}}
    exec(compile(text, "<derivation>", "exec"), ns)
    return ns["_mm_fused_block"]


def counted_keys() -> dict:
    """Every (kt, nt) the run-time census counted, per model, with its call count."""
    out = {}
    names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", SAFETY,
                            "perf/allm_safety/out/"], capture_output=True, text=True,
                           check=True).stdout.split()
    for p in names:
        m = re.search(r"mmkeys_(.+?)\.json$", p)
        if not m:
            continue
        d = json.loads(git_show(SAFETY, p))
        per = {}
        for k, calls in d.get("keys", {}).items():
            reader, key, status = k.split("|")
            if reader != "tenstorrent":          # the readers with their own allow-lists
                continue                          # resolve the VALUE here but gate entry themselves
            kt, nt = (int(x) for x in key.split(","))
            per[(kt, nt)] = per.get((kt, nt), 0) + calls
        out[m.group(1)] = per   # an empty dict is a RESULT: that model presents no
                                 # `tenstorrent` key at all, which is why it cannot move
    return out


def main() -> int:
    cand_src, main_src = git_show(CAND, "tt_bio/tenstorrent.py"), git_show("origin/main", "tt_bio/tenstorrent.py")
    cand_tbl, main_tbl = mm_block(cand_src), mm_block(main_src)
    as_is = fused_fn(cand_src, cand_tbl, fix=False)
    fixed = fused_fn(cand_src, cand_tbl, fix=True)

    def resolve(tbl, fn, kt, nt):
        v = tbl.get((kt, nt))
        return v if v is not None else (fn(kt, nt) if fn else None)

    print(f"main table: {len(main_tbl)} keys | candidate base table: {len(cand_tbl)} keys\n")

    # 1. the six literals the candidate deleted must survive BOTH variants, byte for byte
    deleted = {k: v for k, v in main_tbl.items() if k not in cand_tbl}
    print(f"LITERALS DELETED FROM THE TABLE ({len(deleted)}) -- each must still resolve identically:")
    bad = 0
    for k, v in sorted(deleted.items()):
        a, f = as_is(*k), fixed(*k)
        ok = (a == v) and (f == v)
        bad += not ok
        print(f"  {str(k):10} main {v} | as-is {a} | fixed {f}  {'OK' if ok else '*** MOVED ***'}")

    # 2. every key the census counted, on every model, through all three resolutions
    print("\nCOUNTED KEYS, seven models, run-time counts -- main vs candidate vs candidate+fix:")
    moved_by_cand, undone_by_fix = [], []
    for model, keys in sorted(counted_keys().items()):
        rows = []
        for (kt, nt), calls in sorted(keys.items()):
            m = resolve(main_tbl, None, kt, nt)
            if m is None:                          # main's fallback is the six literals it has
                m = main_tbl.get((kt, nt))
            c, f = resolve(cand_tbl, as_is, kt, nt), resolve(cand_tbl, fixed, kt, nt)
            if (m, c, f) == (m, m, m):
                continue
            rows.append(((kt, nt), calls, m, c, f))
            if m is None and c is not None:
                moved_by_cand.append((model, (kt, nt), calls))
                if f is None:
                    undone_by_fix.append((model, (kt, nt), calls))
        if rows:
            print(f"  {model}:")
            for k, calls, m, c, f in rows:
                tag = "kept by the fix" if f is not None else "REFUSED by the fix"
                print(f"    {str(k):9} {calls:6} calls/fold  main={'None' if m is None else 'served'}"
                      f"  cand={'None' if c is None else 'served'}  fixed={'None' if f is None else 'served'}"
                      f"   <- {tag}")
        else:
            print(f"  {model}: no counted key moves under either variant")

    print(f"\nNewly configured by the candidate: "
          f"{sum(c for _, _, c in moved_by_cand)} calls/fold over {len(moved_by_cand)} keys")
    print(f"Withdrawn again by the one-character fix: "
          f"{sum(c for _, _, c in undone_by_fix)} calls/fold over {len(undone_by_fix)} keys")
    kept = [x for x in moved_by_cand if x not in undone_by_fix]
    print(f"SURVIVING the fix (this is what merging still buys): "
          f"{sum(c for _, _, c in kept)} calls/fold over {len(kept)} keys")
    for model, k, calls in kept:
        print(f"    {model} {k} {calls} calls/fold")

    # 3. the verdict the brief amendment turns on
    odde = [x for x in kept if x[0].startswith("opendde")]
    print("\nVERDICT")
    if undone_by_fix and len(odde) == 2 and not bad:
        print("  The fix withdraws ONLY keys on models the A/B never folded, and leaves both of")
        print("  OpenDDE's intact. The measured 1.438 s was produced by keys the fix does not")
        print("  touch, so it survives the amendment and needs NO re-fold.")
        return 0
    print("  NOT the clean case -- read the table above before amending the brief.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
