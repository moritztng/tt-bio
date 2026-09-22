#!/usr/bin/env python3
"""R2: OpenFold3's loss weights, extracted from their configs and checked against our table.

Their weights are not one table. They are a pydantic default (`LossWeights`,
`projects/of3_all_atom/config/dataset_config_components.py:164-172`) overridden PER DATASET
inside each stage yaml, and delivered per example as `batch["loss_weights"]`
(`runner.py:448`). Ours was `LOSS_WEIGHTS[stage][term]`, two stages deep, holding Protenix's
constants -- it could not express "this dataset in this stage zeroes the confidence terms"
at all.

This script is the provenance, and it is per entry rather than a comment over a table: it
reads their four shipped stage yamls and their pydantic defaults, composes base + per-dataset
override the way their config loader does, and then requires our table to equal the result
entry by entry. A weight in our table that upstream does not produce fails here, and so does
one upstream produces that we are missing. That is the only way a copied table stays honest
after the module it was copied from moves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from tt_bio.train.losses import LOSS_WEIGHTS, OF3_TERM_ALIASES, of3_loss_weights

YAMLS = Path("/home/ttuser/of3t_up/openfold3-0.5.0/examples/training_yamls")

# projects/of3_all_atom/config/dataset_config_components.py:164-172, read as text below so
# the numbers are not transcribed twice.
DEFAULTS_SRC = Path("/home/ttuser/of3t_up/venv/lib/python3.10/site-packages/openfold3/"
                    "projects/of3_all_atom/config/dataset_config_components.py")


def read_defaults() -> dict:
    """Parse `class LossWeights` out of their source rather than restating its numbers."""
    lines = DEFAULTS_SRC.read_text().splitlines()
    i = next(n for n, x in enumerate(lines) if x.startswith("class LossWeights"))
    out = {}
    for line in lines[i + 1:]:
        s = line.strip()
        if not s:
            break
        if ":" in s and "=" in s:
            name, rest = s.split(":", 1)
            out[name.strip()] = float(rest.split("=", 1)[1].strip())
    return out


def stage_table(path: Path, base: dict) -> dict:
    """(dataset -> composed weights, crop budget) for one of their stage yamls."""
    cfg = yaml.safe_load(path.read_text())
    out = {}
    for split, datasets in (cfg.get("dataset_configs") or {}).items():
        if split != "train":
            continue
        for name, d in (datasets or {}).items():
            c = (d or {}).get("config") or {}
            over = ((c.get("loss") or {}).get("loss_weights")
                    or c.get("loss_weights") or {})
            crop = (((c.get("crop") or {}).get("token_crop") or {}).get("token_budget"))
            # Their term names, translated to tt-bio's single vocabulary. The only
            # difference is experimentally_resolved -> resolved.
            w = {OF3_TERM_ALIASES.get(k, k): v for k, v in base.items()}
            w.update({OF3_TERM_ALIASES.get(k, k): float(v) for k, v in over.items()})
            out[name] = {"weights": w, "token_budget": crop,
                         "overrides": {k: float(v) for k, v in over.items()}}
    return out


def main() -> int:
    base = read_defaults()
    theirs = {}
    for y in sorted(YAMLS.glob("*.yml")):
        theirs[y.stem] = stage_table(y, base)

    report = {"instrument": "R2 -- OF3 loss weight table, rebuilt from upstream configs",
              "their_defaults": base, "source": str(DEFAULTS_SRC),
              "stages": theirs}

    # --- our table must reproduce it, entry by entry -----------------------------------
    mismatches = []
    for stage, datasets in theirs.items():
        for ds, info in datasets.items():
            try:
                got = of3_loss_weights(stage, ds)
            except Exception as e:
                mismatches.append({"stage": stage, "dataset": ds,
                                   "error": f"{type(e).__name__}: {e}"})
                continue
            want = info["weights"]
            if set(got) != set(want):
                mismatches.append({"stage": stage, "dataset": ds,
                                   "missing": sorted(set(want) - set(got)),
                                   "extra": sorted(set(got) - set(want))})
                continue
            bad = {k: [got[k], want[k]] for k in want if got[k] != want[k]}
            if bad:
                mismatches.append({"stage": stage, "dataset": ds, "differing": bad})

    n_entries = sum(len(v) for v in theirs.values())
    report["entries_checked"] = n_entries
    report["mismatches"] = mismatches

    # --- the Protenix table must not have moved ----------------------------------------
    protenix_intact = set(LOSS_WEIGHTS) >= {"pretrain", "finetune"}
    report["protenix_stages_intact"] = protenix_intact

    # --- negative control: the check must be able to fail ------------------------------
    probe_stage = sorted(theirs)[0]
    probe_ds = sorted(theirs[probe_stage])[0]
    want = dict(theirs[probe_stage][probe_ds]["weights"])
    term = sorted(want)[0]
    want[term] += 1e-3
    got = of3_loss_weights(probe_stage, probe_ds)
    caught = got[term] != want[term]
    report["negative_control"] = {
        "stage": probe_stage, "dataset": probe_ds, "term": term,
        "perturbation": "+1e-3 on the expected value",
        "detected": caught, "pass": caught}

    ok = not mismatches and protenix_intact and caught
    report["verdict"] = "PASS" if ok else "FAIL"
    out = Path(__file__).resolve().parent / "of3_loss_weights.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"their base defaults: {base}")
    for stage, datasets in theirs.items():
        overs = {d: i["overrides"] for d, i in datasets.items() if i["overrides"]}
        crops = sorted({i["token_budget"] for i in datasets.values()
                        if i["token_budget"] is not None})
        print(f"  {stage}: {len(datasets)} train datasets, {len(overs)} with overrides, "
              f"crop budget(s) {crops}")
    print(f"\nentries checked against our table: {n_entries}, mismatches: {len(mismatches)}")
    for m in mismatches[:8]:
        print(f"   {m}")
    print(f"[{'PASS' if caught else 'FAIL'}] negative control on "
          f"{probe_stage}/{probe_ds}/{term}")
    print(f"\nVERDICT: {report['verdict']}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
