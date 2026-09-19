"""Both axes of each rung, read off the tensors the model is handed. No device.

The bar is in TOKENS, and a design rung reaches a token count two ways -- a long target with a
short binder or the other way round. So the count is not taken from the fixture's name: it is
`restype.shape[0]` in the 18-key dict `ProtenixDesign.design` eats, built by the same
`design_inputs_from_yaml` the shipped CLI calls.

The depth axis is settled here too, and by absence rather than by a comment: PXDesign's model
input has no alignment key at all. `MODEL_INPUT_KEYS` is the full contract, the YAML reader
REFUSES a `msa:` key by name rather than ignoring it, and the dict built from a real fixture is
checked against both. That is what `msa_rows=None` means for this model -- no depth axis exists,
not "the measurement happened to be single-sequence".

    TT_VISIBLE_DEVICES= python3 perf/pxdcov/tokens.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bhdesign"))

from ladder import pxdesign_fixture  # noqa: E402
from tt_bio.pxdesign.inputs import MODEL_INPUT_KEYS, design_inputs_from_yaml  # noqa: E402
from tt_bio.pxdesign.inputs import read_design_yaml  # noqa: E402

PX = ROOT / "perf" / "pxdesign" / "targets" / "laczc_1008.cif"
BIG = ROOT / "perf" / "bhdesign" / "targets" / "big_1831.cif"

#: (target residues, binder, target cif) -- the rungs this task ran.
RUNGS = [(960, 64, PX), (1008, 528, PX), (1456, 80, BIG)]

ALIGNMENT_KEYS = ("msa", "msa_feat", "deletion_matrix", "profile", "num_alignments")


def main() -> int:
    work = pathlib.Path(__file__).parent / "work_tokens"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for tres, binder, target in RUNGS:
        y = pxdesign_fixture(work, tres, target, binder)
        feats = design_inputs_from_yaml(y)
        got = sorted(feats)
        rows.append({
            "target_residues": tres, "binder": binder, "target_cif": target.name,
            "tokens": int(feats["restype"].shape[0]),
            "expected_tokens": tres + binder,
            "alignment_keys_present": [k for k in ALIGNMENT_KEYS if k in feats],
            "n_keys": len(got),
            "unexpected_keys": [k for k in got if k not in MODEL_INPUT_KEYS],
        })
    # The refusal is a second, independent witness that no alignment reaches this model: a
    # `msa:` in the spec is an error, not a key that is read and ignored.
    spec = work / "with_msa.yaml"
    spec.write_text('target:\n  file: "%s"\n  chains:\n    A:\n      crop: ["1-64"]\n'
                    "      msa: /tmp/some.a3m\nbinder_length: 16\n" % PX)
    try:
        read_design_yaml(spec)
        refused = ""
    except Exception as e:                      # noqa: BLE001 -- the message IS the evidence
        refused = f"{type(e).__name__}: {e}"

    report = {"rungs": rows, "msa_key_refused_with": refused,
              "model_input_keys": list(MODEL_INPUT_KEYS)}
    out = pathlib.Path(__file__).parent / "axes.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    bad = 0
    for r in rows:
        ok = r["tokens"] == r["expected_tokens"] and not r["alignment_keys_present"]
        bad += not ok
        print(f'{r["target_residues"]:5d} + {r["binder"]:<4d} -> {r["tokens"]:5d} tokens '
              f'(expected {r["expected_tokens"]}), alignment keys '
              f'{r["alignment_keys_present"] or "none"}, {r["n_keys"]} input keys  '
              f'{"OK" if ok else "MISMATCH"}')
    print(f"msa: in a spec is refused with -> {refused or 'NOTHING, which is a defect'}")
    return 1 if (bad or not refused) else 0


if __name__ == "__main__":
    raise SystemExit(main())
