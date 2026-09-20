#!/usr/bin/env python3
"""Audit the exemption reasons a fresh p300c record left behind, and prove the other cards held.

Three things a re-record does to reasons, all of which have bitten this baseline before:

  IT LEAVES TODOs. A lever that is dark at a rung the previous entry had no reason for gets
  `TODO`, and a TODO is a red cell for the arm. Those need a judgement written by a human.

  IT CARRIES TEXT THE NEW NUMBERS CONTRADICT. `_size_ladder_fill_reasons` regenerates the
  evidence half of a reason only when the head is one of `SIZE_LADDER_EVIDENCE_HEADS`. An older
  reason opening "declines every call" matches nothing, so it travels verbatim onto an entry that
  now reads zero declines -- exactly the defect that function's own comment forbids. This lists
  every carried reason whose head the gate cannot regenerate, so none of them travels unread.

  IT CAN TOUCH ANOTHER CARD. A record pass writes one card's entry, and p150a and tt-galaxy-wh
  must come out byte-identical. Asserted here rather than eyeballed, the same check
  `perf/sizeladder_p300c/reasons_boltz2.py` carried.

Usage:  reasons.py [--ref origin/main] [--card p300c] <model> [<model> ...]
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rg = _load("rg", ROOT / "scripts" / "release_gate.py")
HEADS = tuple(rg.SIZE_LADDER_EVIDENCE_HEADS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--card", default="p300c")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    bad = 0
    for model in args.models:
        p = f"docs/size_ladder_baseline.d/{model}.json"
        new = json.loads((ROOT / p).read_text())
        old_raw = subprocess.run(["git", "show", f"{args.ref}:{p}"], cwd=ROOT,
                                 capture_output=True, text=True).stdout
        old = json.loads(old_raw) if old_raw.strip() else {}
        print("=" * 90)
        print(model)

        for card in sorted(new.get("cards", {})):
            if card == args.card:
                continue
            same = json.dumps(new["cards"][card], sort_keys=True) == \
                json.dumps((old.get("cards") or {}).get(card), sort_keys=True)
            print(f"  other card {card:<15} {'byte-identical' if same else 'CHANGED -- SCOPE LEAK'}")
            bad += 0 if same else 1

        levers = ((new.get("cards") or {}).get(args.card, {})
                  .get("models", {}).get(model, {}).get("levers") or {})
        todo, carried = [], []
        for rung in sorted(levers, key=int):
            for flag, e in sorted(levers[rung].items()):
                r = e.get("reason")
                if not r:
                    continue
                if r.startswith("TODO"):
                    todo.append((rung, flag, r))
                elif not r.startswith(HEADS):
                    carried.append((rung, flag, r))
        print(f"  {len(todo)} TODO reason(s) -- each is a red cell until a judgement is written")
        for rung, flag, r in todo:
            print(f"    TODO    {rung:>5} {flag:<30} {r[:80]}")
        print(f"  {len(carried)} carried reason(s) the gate cannot regenerate the numbers of")
        for rung, flag, r in carried:
            print(f"    STATIC  {rung:>5} {flag:<30} {r[:80]}")
        bad += len(todo)
    print("=" * 90)
    print(f"{bad} item(s) need a hand: TODOs to write, static heads to re-open, scope leaks to undo")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
