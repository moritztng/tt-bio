"""Turn rfd4-fair-gpu's two-seed reference arms into the seed floor in QUALITY.json.

The floor is what says a measured move is smaller than the noise the reference already has, so it
has to be measured on the arms the grade will be taken against: the same box, the same session and
the released recipe. That is `rfd4-fair-gpu`'s reference run, indexed at

    state/rfd4/evidence/reference-designs.json

with this shape (families may be a top-level mapping or sit under a "families" key):

    {"<feature family>": {
        "recipe":  "<what was run: preset, steps, precision>",
        "session": "<box and rental id>",
        "seeds":   {"<seed>": "<absolute path to that seed's output root>"}}}

A seed's value may also be a list of roots, because one arm's metrics are split across the
generation tree and the later `rfd4-evaluate` tree. Two seeds are the minimum; extra seeds are
recorded and the two lowest are used, so the floor does not move when a third is added later.

Run by `artifacts/rfd4-quality/tick.d/calibrate_floor.sh` every tick. It is idempotent: a family
whose floor is already in QUALITY.json from the same index entry is left alone.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import metrics as mspec  # noqa: E402
import quality  # noqa: E402

INDEX = pathlib.Path("/home/moritz/.coworker/state/rfd4/evidence/reference-designs.json")


def _say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _families(doc: dict) -> dict:
    inner = doc.get("families")
    return inner if isinstance(inner, dict) else {k: v for k, v in doc.items() if isinstance(v, dict)}


def main() -> int:
    if not INDEX.exists():
        return 0  # the reference run has not happened yet; nothing to say every ten minutes
    try:
        index = json.loads(INDEX.read_text())
    except ValueError as e:
        _say(f"{INDEX} is not JSON: {e}")
        return 1

    doc = quality.load_doc(quality.QUALITY_JSON)
    floors = doc.setdefault("seed_floor", {})
    wrote = 0
    for family, entry in _families(index).items():
        if family not in mspec.FEATURE_TAGS:
            _say(f"{family}: not a feature family the harness knows, skipped")
            continue
        seeds = entry.get("seeds") or {}
        if len(seeds) < 2:
            _say(f"{family}: {len(seeds)} seed arm(s) indexed, the floor needs 2")
            continue
        picked = sorted(seeds)[:2]
        arms = []
        for s in picked:
            roots = seeds[s] if isinstance(seeds[s], list) else [seeds[s]]
            missing = [r for r in roots if not pathlib.Path(r).exists()]
            if missing:
                _say(f"{family} seed {s}: indexed root does not exist: {missing[0]}")
                arms = []
                break
            arms.append(roots)
        if not arms:
            continue
        kind = entry.get("kind", "two-seed")
        if kind not in quality.FLOOR_KINDS:
            _say(f"{family}: index kind {kind!r} is not one of {quality.FLOOR_KINDS}, skipped")
            continue
        source = (f"{entry.get('recipe', 'recipe not recorded')} | {entry.get('session', 'session not recorded')}"
                  f" | {kind} arms {picked[0]}/{picked[1]}")
        if (floors.get(family) or {}).get("_source") == source:
            continue  # already calibrated from this exact entry
        a, b = quality.load_arm(arms[0]), quality.load_arm(arms[1])
        if min(a["n"], b["n"]) < mspec.N_MIN:
            _say(f"{family}: seed arms are {a['n']} and {b['n']} designs, the floor needs >= {mspec.N_MIN}")
            continue
        floors[family] = quality.calibrate_floor(family, a, b, source=source, kind=kind)
        doc.setdefault("paths", {}).setdefault(family, {})["seed_floor_arms"] = [
            quality._arm_summary(a), quality._arm_summary(b)]
        n_cols = sum(1 for k in floors[family] if not k.startswith("_"))
        _say(f"{family}: seed floor from {a['n']} vs {b['n']} designs, {n_cols} metrics -- {source}")
        wrote += 1

    if wrote:
        quality.write_doc(doc, quality.QUALITY_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
