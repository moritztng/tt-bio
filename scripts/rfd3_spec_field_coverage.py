#!/usr/bin/env python3
"""Which RFD3 InputSpecification fields actually reach the model?

Featurizes one spec once per field, with that field set to a plausible value, and
diffs the resulting feature dict against the no-field baseline. A field that changes
no feature is accepted by the parser and never reaches the device: the run happens,
the CIF is written, and the user gets an unconditioned design.

The negative control is built in: fields known to be honoured (contig, is_non_loopy,
per-atom select_fixed_atoms) must show a diff, otherwise the instrument is not
measuring anything and the script exits non-zero.
"""
import sys

import numpy as np
import torch

from tt_bio.rfd3.featurize import featurize
from tt_bio.rfd3.input import InputSpecification

STRUCT = "examples/ground_truth_structures/prot.cif"
BASE = {"input": STRUCT, "contig": "A140-189,70"}

CASES = [
    ("select_hotspots (str)",        {"select_hotspots": "A150-155"}),
    ("select_hotspots (bool)",       {"select_hotspots": True}),
    ("select_hotspots (dict)",       {"select_hotspots": {"A150-155": "ALL"}}),
    ("select_buried (protein)",      {"select_buried": "A150-155"}),
    ("select_partially_buried",      {"select_partially_buried": "A150-155"}),
    ("select_exposed (protein)",     {"select_exposed": "A150-155"}),
    ("select_hbond_donor",           {"select_hbond_donor": "A150-155"}),
    ("select_hbond_acceptor",        {"select_hbond_acceptor": "A150-155"}),
    ("redesign_motif_sidechains",    {"redesign_motif_sidechains": True}),
    ("ori_token",                    {"ori_token": [1.0, 2.0, 3.0]}),
    ("infer_ori_strategy=com",       {"infer_ori_strategy": "com"}),
    ("infer_ori_strategy=hotspots",  {"infer_ori_strategy": "hotspots"}),
    ("plddt_enhanced=False",         {"plddt_enhanced": False}),
    ("dialect=1",                    {"dialect": 1}),
    ("cif_parser_args",              {"cif_parser_args": {"cache_dir": "/tmp/nope"}}),
    ("select_unfixed_sequence=False",{"select_unfixed_sequence": False}),
    ("select_unfixed_sequence (str)", {"select_unfixed_sequence": "A150-155"}),
    ("unknown key (typo)",           {"select_hotspot": "A150-155"}),
    # --- negative controls: these MUST change the features ---
    ("NEG contig 70->60",            {"contig": "A140-189,60"}),
    ("NEG is_non_loopy=True",        {"is_non_loopy": True}),
    ("NEG select_fixed_atoms atoms", {"select_fixed_atoms": {"A150-155": "N,CA,C,O"}}),
    ("NEG select_fixed_atoms BKBN",  {"select_fixed_atoms": {"A150-155": "BKBN"}}),
]


def feats(extra):
    d = dict(BASE)
    d.update(extra)
    spec = InputSpecification.from_dict(d)
    spec.validate()
    return featurize(STRUCT, spec)


def diff(a, b):
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if k not in a or k not in b:
            out.append(k + ":absent")
            continue
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            x = x.float().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.float().cpu().numpy()
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            out.append("%s:shape %s vs %s" % (k, x.shape, y.shape))
        elif not np.array_equal(np.nan_to_num(x), np.nan_to_num(y)):
            out.append(k)
    return out


def main():
    base = feats({})
    hs = float(np.asarray(base["is_atom_level_hotspot"].float().cpu().numpy()).sum())
    print("baseline: %d features, is_atom_level_hotspot sum=%.1f, "
          "active_donor sum=%.1f, ref_atomwise_rasa sum=%.1f"
          % (len(base), hs,
             float(base["active_donor"].float().sum()),
             float(base["ref_atomwise_rasa"].float().sum())))
    fails = 0
    for label, extra in CASES:
        negative = label.startswith("NEG")
        try:
            d = diff(base, feats(extra))
        except NotImplementedError as e:
            print("  REFUSED   %-32s NotImplementedError: %s" % (label, str(e)[:70]))
            continue
        except (ValueError, TypeError) as e:
            print("  REJECTED  %-32s %s: %s" % (label, type(e).__name__, str(e)[:70]))
            continue
        if d:
            print("  HONOURED  %-32s changes %d feature(s): %s"
                  % (label, len(d), ", ".join(d[:6])))
        else:
            print("  IGNORED   %-32s zero features change" % label)
            if negative:
                print("            ^^ NEGATIVE CONTROL DID NOT MOVE - instrument is blind")
                fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
