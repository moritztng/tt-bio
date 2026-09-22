#!/usr/bin/env python3
"""Consolidate the arms into one answer, including the triangle bound on the reading that
never existed -- what our port WOULD have read against a 0.5.0 reference.

The published A18 reading is our port against the 0.4.3 reference. The brief asked for it
against a 0.5.0 one. The two references are separated by a measured distance, so the second
reading is bounded by the triangle inequality without running the port again:

    | ||p - r043|| - ||r050 - r043|| |  <=  ||p - r050||  <=  ||p - r043|| + ||r050 - r043||

expressed throughout in units of ||r043||, which is the denominator the published figure uses.
"""
import json, torch
from pathlib import Path

R = Path("/home/moritz/.coworker/wt/of3t-auxheads043/perf/of3t_auxheads043")
O = Path("/tmp/of3t/auxheads043/out")

# our port's published A18 relative L2 against the 0.4.3 reference (of3t-auxheads,
# perf/of3t_auxheads/instrument_a_043_aux_heads.json), reproduced here as an input, not re-run.
PORT = {"distogram_logits": 2.8277e-03, "pde_logits": 8.7897e-02, "pae_logits": 1.8199e-01,
        "experimentally_resolved_logits": 3.3315e-01, "plddt_logits": 3.6515e-01}

a043 = torch.load(O / "a1_043_f64.pt", map_location="cpu", weights_only=False)
a050 = torch.load(O / "a1_050_f64.pt", map_location="cpu", weights_only=False)

rows = {}
for k, p in PORT.items():
    x, y = a043[k].double().flatten(), a050[k].double().flatten()
    n043 = float(x.norm())
    d = float((y - x).norm())
    rev = d / n043                      # ||r050 - r043|| / ||r043||
    rows[k] = {
        "port_vs_043_published": p,
        "revision_distance_in_units_of_043": rev,
        "port_vs_050_lower_bound": abs(p - rev),
        "port_vs_050_upper_bound": p + rev,
        "ratio_port_over_revision": p / rev if rev else None,
        "still_fails_A18_at_lower_bound": abs(p - rev) >= 5.0e-02,
    }

rep = {
    "what": "aux_heads A18 reading against 0.4.3 (published) and the bound on the 0.5.0 reading "
            "the brief assumed had been taken",
    "bar_per_tensor": 5.0e-02,
    "heads": rows,
}
(R / "forward_triangle_bound.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep, indent=1))
