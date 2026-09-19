#!/usr/bin/env python3
"""What one OpenFold3 training step actually does, read out of upstream rather than assumed.

This row has to report s/step against theirs, and "their s/step" is not one number: their four
published stage configs run four structurally different steps. Before anything is timed, this
reads the shape out of the upstream sdist so the comparison is against a stage that is named,
and so nobody has to trust a hand reading of their model.py.

The campaign's reference and reproduction target is **0.5.0** (PROTOCOL amendment A3); tt-bio's
production pin stays 0.4.3 and a bump is release-gated. Run against both and they agree: the
four training yamls are byte-identical across the two versions, and the architecture defaults
and grad scoping below come out the same. So the version amendment moves nothing this row
measures, which is worth having on the record rather than assumed.

What it extracts, and why each matters to a per-stage cost breakdown:

  token_budget            the crop. 384 / 640 / 768 / 768 across the four stages.
  batch_size              per rank. Their runner asserts 1 and every config states it.
  no_samples              how many noised structures `_train_diffusion` differentiates per
                          step. The config default is 48; two stages override it to 32. This
                          is the single largest term in a training step's cost and it has no
                          counterpart in the inference path.
  no_mini_rollout_steps   the mini rollout, run under no_grad, that feeds the confidence heads.
  num_recycles            SAMPLED per training step from U{0..num_recycles}, so the trunk's
                          cycle count is a random variable in training and fixed at inference.
  train_confidence_only   when true the trunk takes no gradient at all and `_train_diffusion`
                          is skipped, which makes that stage's step a different animal.

The three grad-scoping facts below are asserted against the source text rather than restated,
so this script fails if a future version changes them:

  1. only the FINAL trunk cycle carries a gradient (model.py `enable_grad = ... is_final_iter`)
  2. the mini rollout runs entirely under `torch.no_grad()`
  3. the recycle count is drawn per step from a synced generator when training

    python3 their_step_shape.py /path/to/openfold3-0.4.3 --out their_step_shape.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

STAGES = ("initial_training", "finetune_1", "finetune_2", "finetune_3")

#: Read off the config defaults; a stage that does not override one of these inherits it.
WANT = ("token_budget", "batch_size", "no_samples", "no_mini_rollout_steps",
        "no_mini_rollout_samples", "num_recycles", "train_confidence_only")


def scalars(text):
    """Every ``key: value`` in a yaml, as key -> sorted distinct values. No yaml dependency.

    A key can appear once per dataset block, so the distinct set is what matters: one value
    means the stage is uniform in it, more than one means the stage varies it per dataset and
    a single number for the stage would be wrong.
    """
    out = {}
    # `[ \t]` and not `\s`: `\s` matches a newline, which lets the leading `^\s*` swallow
    # line breaks and the trailing `\s*$` end the match on a LATER line. That silently drops
    # `batch_size` and reads the architecture default 48 for a stage that overrides
    # `no_samples` to 32 -- a wrong number that looks like a right one.
    for m in re.finditer(r"^[ \t]*([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*([^\n#]+?)[ \t]*$",
                         text, re.M):
        k, v = m.group(1), m.group(2).strip()
        if k not in WANT:
            continue
        try:
            v = ast.literal_eval(v.capitalize() if v in ("true", "false") else v)
        except (ValueError, SyntaxError):
            continue
        out.setdefault(k, set()).add(v)
    return {k: sorted(v) for k, v in out.items()}


def defaults(src):
    """The architecture defaults from ``config/model_config.py``, for keys no stage overrides."""
    text = (src / "openfold3/projects/of3_all_atom/config/model_config.py").read_text()
    out = {}
    for k in WANT:
        m = re.search(rf'"{k}"\s*:\s*([0-9]+|True|False)', text)
        if m:
            out[k] = ast.literal_eval(m.group(1))
    return out


def grad_scoping(src):
    """The three facts that decide which part of the step the tape has to cover.

    Asserted against the source text, not restated from a reading of it, so this fails loudly
    if a future upstream changes one.
    """
    model = (src / "openfold3/projects/of3_all_atom/model.py").read_text()
    facts = {
        "only_final_trunk_cycle_has_grad": bool(
            re.search(r"enable_grad\s*=\s*\(\s*is_grad_enabled\s*and\s*is_final_iter", model)),
        "mini_rollout_under_no_grad": bool(
            re.search(r"def _rollout\b.*?with \(\s*torch\.no_grad\(\)", model, re.S)),
        "recycles_sampled_per_training_step": bool(
            re.search(r"synced_generator\.integers\(\s*low=0,\s*high=self\.shared\.num_recycles",
                      model)),
        "train_diffusion_skipped_when_confidence_only": bool(
            re.search(r"if self\.training and not self\.settings\.train_confidence_only", model)),
    }
    runner = (src / "openfold3/projects/of3_all_atom/runner.py").read_text()
    facts["runner_asserts_per_rank_batch_1"] = bool(
        re.search(r'len\(batch\["pdb_id"\]\)\s*==\s*1', runner))
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="extracted openfold3 sdist root")
    ap.add_argument("--out", type=Path, default=Path("their_step_shape.json"))
    a = ap.parse_args()

    dflt = defaults(a.src)
    facts = grad_scoping(a.src)
    missing = [k for k, v in facts.items() if not v]
    rows = {}
    for st in STAGES:
        f = a.src / "examples/training_yamls" / f"{st}.yml"
        got = scalars(f.read_text())
        row = {}
        for k in WANT:
            if k in got:
                row[k] = got[k][0] if len(got[k]) == 1 else got[k]
                row.setdefault("_overridden", []).append(k)
            elif k in dflt:
                row[k] = dflt[k]
        rows[st] = row

    out = {"source": str(a.src), "architecture_defaults": dflt,
           "grad_scoping": facts, "facts_not_found": missing, "stages": rows}
    a.out.write_text(json.dumps(out, indent=1, default=str))

    cols = ("token_budget", "batch_size", "no_samples", "no_mini_rollout_steps",
            "num_recycles", "train_confidence_only")
    print(f"{'stage':<18}" + "".join(f"{c:>24}" for c in cols))
    for st, row in rows.items():
        print(f"{st:<18}" + "".join(f"{str(row.get(c, '-')):>24}" for c in cols))
    print()
    for k, v in facts.items():
        print(f"  {'OK ' if v else 'NOT FOUND'} {k}")
    if missing:
        print("\nA fact not found means upstream changed and the step shape below is stale.")
    print("WROTE", a.out)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
