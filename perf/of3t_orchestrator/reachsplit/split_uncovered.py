#!/usr/bin/env python3
"""Decompose the 4.8714 % of the squared gradient norm the parameter walk does not cover.

CPU only. No card. No new run. Reads `perf/of3t_rebind/REACH_SHARE.json`, which has carried the
decomposition since `of3t-rebind` pushed it and which nobody has read this way.

WHY
---
D127 was filed (by me, pass 225) as *"3.6438 % of the squared gradient norm is not device-resident
in this port at all"*, from two section headlines. Pass 227 corrected the framing: `of3t-rebind`'s
sentence was about the parameter WALK's path matching, not about silicon. This decomposes what the
walk actually misses, name by name, so the remaining question -- port, or bijection -- can be asked
per tensor instead of per section.

The artifact's own `top_uncovered_by_sq_norm` list answers it in fifteen rows.
"""
import collections
import json
import sys
from pathlib import Path

SRC = Path("/tmp/of3t/of3t-orchestrator/compose/perf/of3t_rebind/REACH_SHARE.json")
OUT = Path(__file__).with_name("UNCOVERED_SPLIT.json")


def section_of(name: str) -> str:
    """`diffusion_module.X...` groups by its second component; everything else by its first."""
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "diffusion_module" else parts[0]


def main() -> int:
    if not SRC.is_file():
        print(f"REFUSING: {SRC} not present -- run compose_verify.sh first.")
        return 2
    r = json.loads(SRC.read_text())
    tot = r["reference_sq_norm"]
    notcov = r["sq_norm_not_covered"]
    listed = r["top_uncovered_by_sq_norm"]

    rows, by = [], collections.Counter()
    acc = 0.0
    for name, sq in listed:
        acc += sq
        by[section_of(name)] += sq
        rows.append({"tensor": name, "sq_norm": sq, "pct_of_model": 100 * sq / tot})

    # The tail is what the artifact did not enumerate. Reported, not guessed at.
    tail_sq = notcov - acc
    tail_n = r["reference_names_not_covered_count"] - len(listed)

    res = {
        "what": "the squared-gradient-norm share the parameter walk does not fully cover, by tensor",
        "source": str(SRC),
        "reference_sq_norm": tot,
        "pct_fully_covered": r["pct_fully_covered"],
        "pct_not_covered": 100 * notcov / tot,
        "by_tensor_top15": rows,
        "by_section_top15": {s: {"sq_norm": v, "pct_of_model": 100 * v / tot}
                             for s, v in by.most_common()},
        "tail": {"n_names": tail_n, "sq_norm": tail_sq, "pct_of_model": 100 * tail_sq / tot,
                 "note": "not enumerated by the source artifact; reported as a lump rather than "
                         "attributed"},
        "readings": [
            "The whole 2.8431 % that D127 attributed to 'aux_heads output projections' is ONE "
            "tensor: aux_heads.distogram.linear.weight.",
            "diffusion_module.diffusion_transformer contributes 0.1661 % across attention_pair_bias "
            "mha linear_q/k/v in blocks 5, 8, 10 and 13. Those tensors certainly have device "
            "gradients -- the diffusion transformer is this campaign's most-measured scope -- so "
            "their being 'not covered' is a NAME-MATCHING failure, which settles the mechanism for "
            "the class rather than arguing it.",
            "diffusion_module.atom_attn_enc contributes 0.7493 % and appears in D127 not at all; it "
            "is most of the 1.2276 % the orchestrator flagged as unaccounted at pass 227.",
            "input_embedder's uncovered share is 0.7531 %, not the 0.8007 % D127 quoted: that "
            "figure was the whole section from a different instrument's table, and ~0.0476 % of it "
            "IS covered.",
        ],
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"reference_sq_norm {tot}")
    print(f"fully covered {r['pct_fully_covered']:.4f} %   not covered {100 * notcov / tot:.4f} %\n")
    for row in rows:
        print(f"  {row['pct_of_model']:7.4f} %  {row['tensor']}")
    print(f"\n  {100 * tail_sq / tot:7.4f} %  [tail: {tail_n} further names, not enumerated]")
    print("\nby section, over the enumerated fifteen:")
    for s, v in by.most_common():
        print(f"  {100 * v / tot:7.4f} %  {s}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
