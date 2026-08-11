#!/usr/bin/env python3
"""Enumerate the rung-512 chunk-folds still missing, and emit the p33 refold task list.

Pass 50 pre-registered the final panel on the premise that folding was complete. It is not: 25
cells are missing, 24 of them chunk 7, and each one costs its target the whole 512 rung. They are
not unfolded tasks -- `fleet_results.jsonl` carries rc=124 (timeout, zero CIFs) records for them,
spread across 25 of the 32 chips, on targets that fold chunks 0-6 clean. The rig's transient
fold-failure rate at this rung is 8.9 pct and retries clear it; c7 holes concentrate at the end of
the queue because that is where the retry budget ran out, not because the work there is harder.
So a refold is expected to clear ~91 pct first attempt and effectively all of them with retries.

Task-list format matches p31/p32's tasks.txt: `<model> <target> <rung> <seed> <chunk> <chunks>`,
seed = base + 1000*chunk so the cells stay seed-nested and disjoint from every other chunk.

Refold with SRC=$H/deepn_src, the frozen 2026-08-02 engine (md5 ef0fe30fae32362de92bfe0d71dec076,
zero CONCAT_HOST_BYTES markers) -- NEVER deepn_src_oomfix. Four hole targets (9ve0 747, 9xth 665,
9t3r 663, 9t3s 663 residue tokens) sit above the Wormhole SEQ_LEN_MORE_CHUNKING gate of 640, so the
OOM-fixed engine takes a different code path for them and would mix two engines inside a
primary-panel 512 pool. Every other cell in the primary curve came off the frozen tree.

Read-only against the data tree. CPU only.
"""
import argparse
import json
import os
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
GALAXY = BASE / "galaxy"
PREFIX = {"opendde-abag": "opendde", "protenix-v2": "protenix",
          "boltz2": "boltz2", "esmfold2": "esmfold2"}
GALAXY_EXCLUDE = {"opendde-abag": {"9sbb"}}
P32_EXTENSION = {"opendde-abag": {"9i3p", "9j4c", "9ivj", "9q7y"},
                 "protenix-v2": {"9j4c"}, "esmfold2": {"9j4c"}}
P32_EXT_ON = os.environ.get("DEEPN_P32_EXT") == "1"
CHUNKS_512 = 8


def chunks_on_disk(model, rung):
    """target -> set of chunk indices present for (model, rung)."""
    out = {}
    root = GALAXY / PREFIX[model]
    for d in sorted(root.iterdir()):
        if not d.is_dir() or f"_n{rung}" not in d.name:
            continue
        target, rest = d.name.split("_n", 1)
        if not rest.startswith(f"{rung}"):
            continue
        if target in GALAXY_EXCLUDE.get(model, ()):
            continue
        if not P32_EXT_ON and target in P32_EXTENSION.get(model, ()):
            continue
        if "_c" not in rest:
            continue
        try:
            out.setdefault(target, set()).add(int(rest.split("_c")[1]))
        except ValueError:
            pass
    return out


def fleet_512():
    """Every rung-512 fold record, keyed (model, target, chunk). The fleet appends
    concurrently, so torn lines exist and are skipped."""
    out = {}
    p = GALAXY / "fleet_results.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("rung") == 512:
            out.setdefault((r.get("model"), r.get("target"), r.get("chunk")), []).append(r)
    return out


def seed_for(recs, model, target, chunk):
    """The seed this cell must use. Taken from the fleet's own record for the cell when one
    exists -- the failed attempts logged it, so it is authoritative rather than reconstructed.
    Otherwise derived from the target's chunk-0 seed by the ladder's base + 1000*chunk rule."""
    for r in recs.get((model, target, chunk), ()):
        if r.get("seed") is not None:
            return int(r["seed"]), "logged"
    for r in recs.get((model, target, 0), ()):
        if r.get("seed") is not None:
            return int(r["seed"]) + 1000 * chunk, "derived from c0"
    return None, "UNKNOWN -- do not launch this cell"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, help="write the task list here")
    ap.add_argument("--quiet", action="store_true", help="task lines only")
    args = ap.parse_args()

    recs = fleet_512()
    tasks, total, unknown = [], 0, 0
    for model in ("boltz2", "esmfold2", "opendde-abag", "protenix-v2"):
        present = chunks_on_disk(model, 512)
        # A target only counts as a hole if it reached rung 256 -- otherwise it is not in the
        # ladder at all and refolding one chunk of it buys nothing.
        at256 = {t for t, cs in chunks_on_disk(model, 256).items() if len(cs) >= 4}
        holes = {t: sorted(set(range(CHUNKS_512)) - cs)
                 for t, cs in present.items()
                 if t in at256 and len(cs) < CHUNKS_512}
        if not args.quiet:
            print(f"\n{model}: {len(holes)} target(s) short of a 512 pool")
        for target in sorted(holes):
            for c in holes[target]:
                seed, how = seed_for(recs, model, target, c)
                if seed is None:
                    unknown += 1
                else:
                    tasks.append(f"{model} {target} 512 {seed} {c} {CHUNKS_512}")
                    total += 1
                if not args.quiet:
                    hits = recs.get((model, target, c), ())
                    why = (", ".join(f"rc={r.get('rc')} {r.get('seconds')}s "
                                     f"cifs={r.get('cifs')}" for r in hits)
                           or "no fold record -- never attempted")
                    print(f"   {target:<6} c{c}  seed {seed} ({how})  [{why}]")

    if not args.quiet:
        print(f"\n{total} cell(s) to refold." +
              (f"  {unknown} cell(s) have no recoverable seed -- STOP, do not guess."
               if unknown else ""))
    if args.tasks:
        args.tasks.write_text("\n".join(tasks) + "\n")
        print(f"wrote {args.tasks}")
    else:
        for t in tasks:
            print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
