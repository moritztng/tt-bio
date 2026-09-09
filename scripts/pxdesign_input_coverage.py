#!/usr/bin/env python3
"""Which parts of a PXDesign target YAML actually reach the model?

Builds the 18-key model input from a target YAML once per variation and diffs it
against the baseline. A variation that changes nothing is accepted by the reader and
never reaches the device.

Negative controls: hotspots, crop and binder_length must all move the input, or the
diff is blind and the script exits non-zero.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from tt_bio.pxdesign.inputs import design_inputs_from_yaml, read_design_yaml

FIXTURE = Path("tests/fixtures/pxdesign/PDL1.yaml")

BASE = {
    "target": {"file": "./5o45.cif.gz", "chains": {"A": {"crop": ["1-116"]}}},
    "binder_length": 80,
}


def spec(mutate):
    d = yaml.safe_load(yaml.safe_dump(BASE))
    mutate(d)
    return d


def build(d, tmp):
    p = tmp / "t.yaml"
    p.write_text(yaml.safe_dump(d))
    return design_inputs_from_yaml(p)


def diff(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        if k == "condition":
            continue
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            out.append(k + ":absent")
            continue
        x = x.float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y = y.float().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        if x.shape != y.shape:
            out.append("%s:shape %s vs %s" % (k, x.shape, y.shape))
        elif not np.array_equal(np.nan_to_num(x), np.nan_to_num(y)):
            out.append(k)
    return out


def set_chain(d, **props):
    d["target"]["chains"]["A"] = dict(d["target"]["chains"]["A"], **props)


CASES = [
    ("msa: (documented ignore)", lambda d: set_chain(d, msa="./msa/PDL1/0")),
    ("hotspot: singular typo",   lambda d: set_chain(d, hotspot=[40, 99, 107])),
    ("hotspots out of range",    lambda d: set_chain(d, hotspots=[9001, 9002])),
    ("hotspots partly in range", lambda d: set_chain(d, hotspots=[40, 9001])),
    ("crops: plural typo",       lambda d: set_chain(d, crops=["1-40"])),
    ("unknown top-level key",    lambda d: d.update(num_designs=7)),
    ("unknown chain prop",       lambda d: set_chain(d, symmetry="C3")),
    # --- negative controls ---
    ("NEG hotspots [40,99,107]", lambda d: set_chain(d, hotspots=[40, 99, 107])),
    ("NEG crop 1-116 -> 1-60",   lambda d: set_chain(d, crop=["1-60"])),
    ("NEG binder_length 80->96", lambda d: d.update(binder_length=96)),
]


def main():
    tmp = Path(tempfile.mkdtemp())
    # The fixture's own structure path is relative to the fixture, so run beside it.
    BASE["target"]["file"] = str((FIXTURE.parent / "5o45.cif.gz").resolve())
    base = build(BASE, tmp)
    n_hot = int(base["hotspot"].sum())
    print("baseline: %d tokens, hotspot channel sum=%d" % (base["restype"].shape[0], n_hot))
    fails = 0
    for label, mut in CASES:
        negative = label.startswith("NEG")
        try:
            d = diff(base, build(spec(mut), tmp))
        except (ValueError, FileNotFoundError, KeyError) as e:
            print("  REJECTED  %-28s %s: %s" % (label, type(e).__name__, str(e)[:66]))
            continue
        if d:
            print("  HONOURED  %-28s changes: %s" % (label, ", ".join(d[:5])))
        else:
            print("  IGNORED   %-28s zero model inputs change" % label)
            if negative:
                print("            ^^ NEGATIVE CONTROL DID NOT MOVE - instrument is blind")
                fails += 1
    # The reader's own view: every key it does not act on, in one message.
    d = spec(lambda x: set_chain(x, hotspot=[40], msa="m", symmetry="C3"))
    p = tmp / "r.yaml"
    p.write_text(yaml.safe_dump(d))
    try:
        print("  NOT REFUSED  read_design_yaml keeps: %s" % sorted(read_design_yaml(p)))
        fails += 1
    except ValueError as e:
        print("  REFUSED   every key the reader ignores, in one message:")
        print("            %s" % str(e).split(": ", 2)[-1])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
