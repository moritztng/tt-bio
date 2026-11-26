"""Audit a :class:`CurvePlan`'s exemptions against the batches its ids actually yield.

A curve plan names the parameters its data cannot drive, and the G2 guard trusts those names.
An exemption carried over from a smaller id set is therefore a hole in the guard: the parameter
is exempt, the data does drive it, and a dead head in the graded run goes unreported.

This reads every id in the plan through upstream's own pipeline at the plan's crop and seed and
reports, per id: whether the batch carries the keys the exempted features read, how many tokens
are ATOMIZED (charge and element embeddings are multiplied by `is_atomized`, so a present key
with no atomized token still trains nothing) and the secondary-structure values (upstream builds
that table with `padding_idx=0`, so an all-NONE example gives an identically zero gradient
through the REAL branch, not the absent-key one).

    python perf/r4t/plan_audit.py perf/r4t/plans/g3_curve.json

Needs RFD4_REPO, DATA_PATH and HF_HOME, like the rest of perf/r4t.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# The exact parameter name each audited feature owns. Exact names, not a substring test:
# "feat_creators.1" is a prefix of "feat_creators.16", so matching by prefix would read an
# exemption for the charge table as an exemption for feat_creators.1 and pass a real hole.
PARAM_OF = {
    "charge": "init_repr_factory.feat_creators.16.embedding.weight",
    "element": "init_repr_factory.feat_creators.17.embedding.weight",
    "ss": "init_repr_factory.feat_creators.7.ss_embedding.weight",
}


def contradicted(exempt_names, drives):
    """The exemptions this id set contradicts: exempt, and driven by at least one of its ids."""
    exempt = set(map(str, exempt_names))
    return {name: ids for name, ids in drives.items() if ids and name in exempt}


def main() -> int:
    import torch  # noqa: F401  (upstream's stack needs it imported before hydra)

    repo = Path(os.environ["RFD4_REPO"]).expanduser()
    sys.path.insert(0, str(repo / "src"))
    from hydra import compose, initialize_config_dir
    from rfproteina.conditions import C_SEC
    from rfproteina.datapipes.schema.proteina import TokenType

    import curve_upstream as cu

    plan = json.loads(Path(sys.argv[1]).read_text())
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="training_proteina",
                      overrides=["experiment=tutorial/finetune_test_dataset", "+single=true",
                                 "+nolog=true"])
    ds, index = cu.dataset_index(cfg, 0)
    rows, ss_total = [], Counter()
    for eid in plan["example_ids"]:
        feats = cu.collate(ds, index[eid], plan["crop"], plan["exploration_k"],
                           seed=cu.example_seed(plan["seed"], eid))
        atomized = int((feats["token_type"] == TokenType.ATOMIZED).sum())
        ss_present = C_SEC.full_name in feats
        ss = feats[C_SEC.full_name].long() if ss_present else None
        if ss is not None:
            ss_total.update(ss.flatten().tolist())
        row = {"id": eid, "tokens": int(feats["padding_mask"][0].sum()), "atomized": atomized,
               "charge": "charge" in feats, "element": "element" in feats,
               "ss_present": ss_present, "ss_nonzero": int((ss > 0).sum()) if ss is not None else 0}
        rows.append(row)
        print(json.dumps(row), flush=True)

    drives = {
        PARAM_OF["charge"]: [r["id"] for r in rows if r["charge"] and r["atomized"] > 0],
        PARAM_OF["element"]: [r["id"] for r in rows if r["element"] and r["atomized"] > 0],
        PARAM_OF["ss"]: [r["id"] for r in rows if r["ss_nonzero"]],
    }
    print("SUMMARY", json.dumps({"ids": len(rows), "ss_values": dict(sorted(ss_total.items())),
                                 "driven_by": {k: len(v) for k, v in drives.items()},
                                 "drivers": drives}))
    bad = contradicted([e[0] for e in plan.get("exempt", [])], drives)
    if bad:
        for name, ids in sorted(bad.items()):
            print(f"FAIL: the plan exempts {name}, which {len(ids)} of its own ids drive "
                  f"(first: {ids[0]})")
        return 1
    print(f"PASS: all {len(plan.get('exempt', []))} exemptions name a parameter this id set "
          f"cannot drive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
