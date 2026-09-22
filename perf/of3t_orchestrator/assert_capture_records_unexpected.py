#!/usr/bin/env python3
"""A capture's checkpoint provenance must record BOTH halves of the load (D141).

`load_state_dict(sd, strict=False)` returns `missing_keys` AND `unexpected_keys`. The shared
diffusion capture recorded only the first, so a checkpoint whose module has a different
architecture -- upstream 0.4.3 hoists the pair LayerNorm to the transformer level while
`of3-p2-155k.pt` stores 24 of them per block -- dropped 24 TRAINED tensors into `unexpected_keys`
and reported `n_missing 1, missing ['version_tensor']`. A clean-looking load.

`perf/of3t_gradients/capture_trunk_boundary.py` has recorded `n_unexpected` all along, so this is
not a new idea; it is the diffusion side never having had it. The check is therefore shaped as
"every capture report that claims to prove a load must prove both halves of it", not as a patch to
one script.

usage: assert_capture_records_unexpected.py [root]     (default: cwd, the composed tree)
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

#: The three that already exist, frozen at pass 255 as a RATCHET, not an exemption. Failing on
#: them would abort every compose until the captures are re-run, which costs reference-side hours
#: and belongs to whoever rebuilds the reference for D141 -- but a FOURTH must never ship blind,
#: and an entry that has since been fixed must be dropped, so the list can only shrink.
FROZEN = {
    "perf/of3t_diffusion/capture_diffusion_boundary.json",
    "perf/of3t_rebase/capture_diffusion_boundary_043.json",
    "perf/of3t_softgrad/capture_diffusion_boundary_043_REBUILT.json",
}


def offenders(root: pathlib.Path):
    """(path, why) for every artifact whose `checkpoint` block is blind to unexpected keys."""
    out = []
    for p in sorted(root.glob("perf/*/*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ck = d.get("checkpoint")
        if not isinstance(ck, dict):
            continue
        # A block that records neither half is not making a claim; one that records only the
        # missing half is, and it is the claim that is wrong.
        if "n_missing" in ck and "n_unexpected" not in ck:
            out.append((str(p.relative_to(root)),
                        f"records n_missing={ck.get('n_missing')} and no n_unexpected"))
    return out


def main() -> int:
    # Probe first: the check must fire on the shape it is written for, or its silence is
    # uninformative (A17).
    probe = offenders.__doc__ is not None
    synthetic = {"checkpoint": {"n_missing": 1, "missing": ["version_tensor"], "n_loaded": 4935}}
    fake = ROOT / "perf" / "_probe_d141" / "probe.json"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(json.dumps(synthetic))
    fired = [o for o in offenders(ROOT) if "_probe_d141" in o[0]]
    fake.unlink()
    try:
        fake.parent.rmdir()
    except OSError:
        pass
    if not fired:
        print("REFUSING: the D141 probe did not fire on a synthetic missing-only checkpoint "
              "block, so this check reads nothing and its silence means nothing")
        return 1

    seen = {o[0] for o in offenders(ROOT) if "_probe_d141" not in o[0]}
    new = sorted(seen - FROZEN)
    healed = sorted(FROZEN - seen)
    if new:
        print(f"{len(new)} NEW capture report(s) prove only half of their load (D141) -- "
              f"`unexpected_keys` is where 24 trained layer_norm_z tensors went:")
        for path in new:
            print(f"  {path}")
        print("  Record n_unexpected and unexpected beside the missing pair, or add the file to "
              "FROZEN with a reason.")
        return 1
    if healed:
        print("FROZEN still lists capture report(s) that now record both halves -- the ratchet "
              "only counts if it is tightened; drop them: " + ", ".join(healed))
        return 1
    print(f"ok    no capture report proves only half of its load beyond the {len(FROZEN)} frozen "
          f"at pass 255 (probe fires)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
