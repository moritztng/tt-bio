#!/usr/bin/env python3
"""Is OF3's Transition the same code PTX measured at 0.83-0.85x under the tape? Yes, literally.

PTX's split is this row's prior: kernel-level optimizations survived differentiation (triangle
attention kept 1.19-1.29x) and L1-placement ones did not -- the Transition went to 0.83-0.85x at
256 tokens, "because the tape keeps what the forward frees". The brief says to check the
Transition first and to measure it here rather than inherit it. Measuring needs a card. Whether
it is the SAME CODE does not, and that is a different question with a different answer:

  * it decides whether PTX's number is a prior about our own module or an analogy across models;
  * it names the exact flag and line an A/B would move, before the card is free;
  * and if the answer is yes, a fix is a SHARED-PATH fix, which is the fleet's standing rule.

Every fact below is asserted against the working tree, so this fails loudly rather than going
stale the first time one of the four files moves. Run it anywhere; it opens no device and
imports no tt_bio module.

    python3 perf/of3t_perf/shared_transition.py --json perf/of3t_perf/shared_transition.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TT = REPO / "tt_bio"


def line_of(path: Path, pattern: str) -> tuple[int, str]:
    rx = re.compile(pattern)
    for i, ln in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if rx.search(ln):
            return i, ln.strip()
    raise AssertionError(f"{path.relative_to(REPO)}: nothing matches {pattern!r} -- the claim "
                         f"this file records has moved and must be re-read, not re-asserted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    ts, tr = TT / "tenstorrent.py", TT / "openfold3_trunk.py"
    f = {}

    # 1. One PairformerLayer class, and OF3's trunk reaches it through the shared Pairformer.
    f["pairformer_layer_class"] = line_of(ts, r"^class PairformerLayer\(Module\):")
    f["pairformer_builds_layers"] = line_of(ts, r"^\s+PairformerLayer\($")
    f["of3_trunk_builds_pairformer"] = line_of(tr, r"self\.pairformer = Pairformer\(")

    # 2. Protenix builds the same class, which is what makes PTX's number ours.
    f["protenix_builds_layer"] = line_of(TT / "protenix.py", r"PairformerLayer\(")

    # 3. The Transition lives inside that shared layer, on the pair track.
    f["transition_z_built"] = line_of(ts, r"self\.transition_z = Transition\(")
    f["transition_z_called"] = line_of(ts, r"z_update = self\.transition_z\(")

    # 4. The L1-placement lever, its default, and the free the tape declines.
    f["residual_l1_flag"] = line_of(ts, r'^_RESIDUAL_L1 = env_flag\("TT_BIO_RESIDUAL_L1"')
    f["residual_l1_at_transition"] = line_of(ts, r"if _RESIDUAL_L1 else None\)")
    f["tape_evicts_l1"] = line_of(TT / "autograd.py",
                                  r"buffer_type == ttnn\.BufferType\.L1:")

    out = {
        "question": "is OF3's Transition the module PTX measured under the tape",
        "answer": "yes -- the same PairformerLayer class object, not an analogous one",
        "facts": {k: {"file": str(p.relative_to(REPO)), "line": n, "text": t}
                  for k, (p, (n, t)) in {
                      "pairformer_layer_class": (ts, f["pairformer_layer_class"]),
                      "pairformer_builds_layers": (ts, f["pairformer_builds_layers"]),
                      "of3_trunk_builds_pairformer": (tr, f["of3_trunk_builds_pairformer"]),
                      "protenix_builds_layer": (TT / "protenix.py", f["protenix_builds_layer"]),
                      "transition_z_built": (ts, f["transition_z_built"]),
                      "transition_z_called": (ts, f["transition_z_called"]),
                      "residual_l1_flag": (ts, f["residual_l1_flag"]),
                      "residual_l1_at_transition": (ts, f["residual_l1_at_transition"]),
                      "tape_evicts_l1": (TT / "autograd.py", f["tape_evicts_l1"]),
                  }.items()},
        "default": "TT_BIO_RESIDUAL_L1 defaults TRUE, so the taped forward takes the eviction "
                   "by default and an A/B has a flag to move without touching a line of code",
        "mechanism": "under a tape an L1-resident activation is EVICTED to DRAM rather than "
                     "freed (autograd.Tensor.free), so the L1 placement buys the forward "
                     "nothing and costs the step one DRAM write plus one read per residual",
        "shape_note": "PTX measured 0.83-0.85x at a [1,256,256,128] pair track. OF3's smallest "
                      "training crop is 384 tokens, 2.25x the pair elements, so the prior is "
                      "about a SMALLER tensor than the one this row has to measure",
        "not_a_proposal": "PROTOCOL sequencing: no lever may be proposed or landed until "
                          "of3t-equivalence has passed. This records where to look, and the "
                          "A/B that answers it is a step-level one against an A/A floor.",
    }
    for k, v in out["facts"].items():
        print(f"  {k:<28} {v['file']}:{v['line']}")
    print(f"\n{out['answer']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
        print("WROTE", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
