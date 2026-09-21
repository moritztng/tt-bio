"""COVERAGE's two halves live in two artifacts, so the charter could read only one of them.

D179. GO condition COVERAGE has two clauses -- `union.*.covered` over upstream's eight
`LossWeights` terms, and `conditional_paths.*.covered` over eleven paths. It read
`perf/of3t_gradients/coverage_census.json`, which is dated 2026-09-19 and carries both keys but at
4 of 11 paths. `of3t-covpaths` then measured 9 of 11 and wrote `COVERAGE_UNION.json`, correctly
declining to re-emit a concluded row's artifact -- but in that file the key `union` holds the
ELEVEN PATHS, not the loss terms, and there is no `conditional_paths` key. So repointing the
charter at it would have graded the paths against the loss-term clause and read the other as
absent: a worse reading than the stale one.

This composes the two into one artifact in the census's schema. It measures nothing. Both sources
are pinned by sha256 and every entry records which file it came from, so the merge can be audited
line by line and regenerated rather than believed.

One entry is UPGRADED rather than copied, and it is called out here because it is the only
judgement in the file. The census reads `union.bond.covered = false` with `n_pairs_firing: 0`, for
the stated reason "no inter-token bond on 5nw3" -- true of 5nw3, which is the only target it
looked at. `of3t-covpaths` carries a bond reading on a different target: finetune_1/weighted-pdb,
4g5j, `bond_mask_nnz` 1, `loss_weight_bond` 4.0, `bond_loss` 1.2424831511452794e-03, with 3,924 of
4,170 parameters moving and 14.69 % of the squared gradient norm behind it. That is one firing
(stage, dataset) pair in exactly the shape the census's own schema records, so the term is marked
covered with `carried_by: ["finetune_1/weighted-pdb"]` and `n_pairs_firing: 1`. Dispute it by
disputing that evidence; it is cited, not asserted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CENSUS = "perf/of3t_gradients/coverage_census.json"
PATHS = "perf/of3t_covpaths/COVERAGE_UNION.json"
OUT = "perf/of3t_orchestrator/coverage/COVERAGE_MERGED.json"


def load(rel: str):
    p = ROOT / rel
    if not p.is_file():
        raise SystemExit(f"REFUSING: {rel} is absent from {ROOT}")
    raw = p.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> int:
    census, census_sha = load(CENSUS)
    paths, paths_sha = load(PATHS)

    cited = paths.get("census", {}).get("sha256")
    if cited and cited != census_sha:
        print(f"REFUSING: {PATHS} cites census sha256 {cited} but the census in this tree is "
              f"{census_sha}. One of them has moved, and merging them would compose two "
              f"different measurements into one artifact.")
        return 1

    if set(census.get("conditional_paths", {})) != set(paths.get("union", {})):
        print("REFUSING: the two files do not describe the same eleven paths")
        return 1

    union = {}
    for term, entry in census["union"].items():
        e = dict(entry)
        e["source"] = CENSUS
        union[term] = e

    bond = paths["union"].get("bond", {})
    if bond.get("covered") and not union["bond"]["covered"]:
        ev = bond.get("evidence", {})
        pair = f"{ev.get('stage')}/{ev.get('dataset')}"
        union["bond"] = {
            "covered": True, "carried_by": [pair], "n_pairs_firing": 1,
            "source": PATHS, "upgraded_by": "of3t-orchestrator, pass 328 (D179)",
            "upgraded_why": ("the census read this false on 5nw3, which has no inter-token bond; "
                             f"{PATHS} carries a firing pair on a different target"),
            "evidence": ev,
        }

    conditional = {}
    for name, entry in paths["union"].items():
        e = dict(entry)
        e["source"] = PATHS
        e["census_entry"] = census["conditional_paths"].get(name)
        conditional[name] = e

    n_u = sum(1 for v in union.values() if v.get("covered"))
    n_c = sum(1 for v in conditional.values() if v.get("covered"))
    out = {
        "instrument": "perf/of3t_orchestrator/coverage/merge_coverage.py",
        "what": ("COVERAGE's two halves in one artifact, in the census's schema. A composition of "
                 "two measurements, not a third measurement."),
        "owner": "of3t-orchestrator (D179)",
        "sources": [{"path": CENSUS, "sha256": census_sha, "provides": "union, the 8 loss terms"},
                    {"path": PATHS, "sha256": paths_sha,
                     "provides": "conditional_paths, the 11 paths"}],
        "rule": paths.get("rule"),
        "union": union,
        "conditional_paths": conditional,
        "summary": {"loss_terms_covered": n_u, "loss_terms": len(union),
                    "paths_covered": n_c, "paths": len(conditional),
                    "paths_uncovered": sorted(k for k, v in conditional.items()
                                              if not v.get("covered"))},
        "superseded": {"census_paths_covered": paths.get("census_conditional_paths_covered"),
                       "census_paths_total": paths.get("census_conditional_paths_total")},
    }
    (ROOT / OUT).write_text(json.dumps(out, indent=2) + "\n")
    print(f"loss terms {n_u} of {len(union)}, paths {n_c} of {len(conditional)}, "
          f"uncovered {out['summary']['paths_uncovered']}")
    print(f"wrote {ROOT / OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
