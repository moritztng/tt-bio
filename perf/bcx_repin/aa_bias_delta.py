#!/usr/bin/env python3
"""What PR #17 actually changes, asked of each tree's own settings loader.

`upstream_delta.py` answered which FILES moved between `7a2dfdb` and `301efdd`. This asks
the question that decides whether re-pinning changes an arm: for a given example, what
amino-acid bias does BindCraft 2 resolve, and where does it apply it. The resolution is
not a transcription of the JSON -- `aa_bias` values are propensities, `resolve_amino_acid_bias`
takes their log and drops the non-positive ones into `omitted_amino_acids` -- so the answer
comes from each tree importing its OWN `bindcraft.settings`, in its own subprocess, because
two `bindcraft` packages cannot live in one interpreter.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
VENV = "/home/ttuser/bcx_e2e_venv/bin/python"

#: Run inside the child, with that tree first on sys.path.
PROBE = r"""
import json, os, sys
BC2 = os.environ["BCX_BC2"]
sys.path.insert(0, BC2)
from bindcraft.preflight import cleaned_campaign_settings
from bindcraft.settings import build_design_settings, read_settings, parse_setting_overrides
out = {}
for name in json.loads(os.environ["BCX_EXAMPLES"]):
    settings = cleaned_campaign_settings(read_settings(os.path.join(BC2, "examples", name),
                                                       parse_setting_overrides([])))
    binder = build_design_settings(settings).binder
    out[name] = {
        "resolved_amino_acid_bias": {k: round(v, 6) for k, v in binder.amino_acid_bias.items()},
        "omitted_amino_acids": binder.omitted_amino_acids,
        "min_monomer_plddt_final": settings.get("min_monomer_plddt_final"),
        "mutate_positions": settings.get("mutate_positions"),
        "design_models": settings.get("design_models"),
        "validation_model": settings.get("validation_model"),
    }
print(json.dumps(out))
"""

#: Every site that reads the bias, per tree. A site that exists on one tree and not the
#: other is the change; a site quoted in a brief and absent from both is the correction.
SITES = ("bindcraft/protein_preparation.py", "bindcraft/sequence_optimization.py",
         "bindcraft/af2.py", "bindcraft/proteinmpnn.py")


def probe(tree, examples):
    env = dict(os.environ, BCX_BC2=tree, BCX_EXAMPLES=json.dumps(examples))
    return json.loads(subprocess.run([VENV, "-c", PROBE], env=env, check=True,
                                     capture_output=True, text=True).stdout)


def sites(tree):
    got = {}
    for path in SITES:
        text = pathlib.Path(tree, path).read_text().splitlines()
        got[path] = [f"{i}: {line.strip()}" for i, line in enumerate(text, 1)
                     if "amino_acid_bias" in line and "def " not in line
                     and "__init__" not in line and not line.strip().startswith("#")]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", default="/home/ttuser/bcx_repin/bc2_7a2dfdb")
    ap.add_argument("--repin", default="/home/ttuser/bcx_repin/bc2")
    ap.add_argument("--examples", nargs="+", default=["pdl1.json", "pdl1_vhh.json"])
    ap.add_argument("--out", default=str(HERE / "aa_bias_delta.json"))
    args = ap.parse_args()

    trees = {"pinned": args.pinned, "repin": args.repin}
    head = {k: subprocess.check_output(["git", "-C", v, "rev-parse", "--short", "HEAD"],
                                       text=True).strip() for k, v in trees.items()}
    out = {"heads": head,
           "examples": {k: probe(v, args.examples) for k, v in trees.items()},
           "bias_sites": {k: sites(v) for k, v in trees.items()}}

    #: The one sentence the campaign plan turns on, computed rather than asserted.
    for name in args.examples:
        pin = out["examples"]["pinned"][name]
        rep = out["examples"]["repin"][name]
        out.setdefault("verdict", {})[name] = {
            "bias_empty_on_both": not pin["resolved_amino_acid_bias"] and not rep["resolved_amino_acid_bias"],
            "bias_changed": pin["resolved_amino_acid_bias"] != rep["resolved_amino_acid_bias"],
            "acceptance_filter_changed": pin["min_monomer_plddt_final"] != rep["min_monomer_plddt_final"],
            "mutate_positions_changed": pin["mutate_positions"] != rep["mutate_positions"],
        }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
