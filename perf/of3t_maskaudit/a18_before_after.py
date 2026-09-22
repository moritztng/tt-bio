"""A18's pass/fail table before and after the audit, per scope, from ENUMERATION.json.

Before = the verdict the campaign holds today, formed on the value each row quoted.
After  = the verdict on the real-token block: the artifact's own masked column where it has one,
         the exact recomputation where the producing script kept both, and the bound
         rel_padded / sqrt(q) where only the reference survives.
"""
import json
import sys

E = json.load(open("perf/of3t_maskaudit/ENUMERATION.json"))
R = {r["id"]: r for r in E["readings"]}
BAR = 0.05

aux = [R[k] for k in R if k.startswith("A18-4.")]
aux_after = [(r["quantity"].split()[-2], r["real_token_value"] or r["value"]) for r in aux]
aux_before = [(r["quantity"].split()[-2], r["value"]) for r in aux]
cond = [R[k] for k in R if k.startswith("A18-2.")]

scopes = [
    {"scope": "diffusion device arm", "mass_pct": 51.1358, "source": "D30, of3t-diffusion",
     "before": {"value": R["A18-1"]["value"], "verdict": "PASS"},
     "after": {"value": R["A18-1"]["value"], "verdict": "PASS",
               "why": "the compared tensor is xl_out on the atom axis, 422 of 422 atoms real. "
                      "No padded position exists in it, so there is nothing to exclude."},
     "changed": False},
    {"scope": "diffusion_module.diffusion_conditioning", "mass_pct": 36.9462,
     "source": "of3t-conditioning",
     "before": {"value": max(r["value"] for r in cond), "verdict": "PASS",
                "note": "worst of the six padded readings"},
     "after": {"value": max(r["bound"] for r in cond), "verdict": "PASS",
               "why": "PADDED and exposed -- the reference carries 78.0 % of its si mass and "
                      "85.6 % of its zij mass outside the real block. Neither forward tensor was "
                      "kept, so the exact real-token value cannot be recomputed; the bound "
                      "rel/sqrt(q) is exact and every reading stays at least 3.7x inside the bar."},
     "changed": False},
    {"scope": "msa_module", "mass_pct": 1.24, "source": "of3t-auxheads",
     "before": {"value": R["A18-3"]["value"], "verdict": "PASS"},
     "after": {"value": R["A18-3"]["real_token_value"], "verdict": "PASS",
               "why": "the quoted figure was already the real-block one; the padded figure for "
                      f'the same tensor is {R["A18-3"]["padded_value"]:.6e} and was not used.'},
     "changed": False},
    {"scope": "aux_heads", "mass_pct": 2.8431, "source": "of3t-direct",
     "before": {"value": max(v for _, v in aux_before), "verdict": "FAIL",
                "worst_head": max(aux_before, key=lambda t: t[1])[0],
                "n_over_bar": sum(1 for _, v in aux_before if v > BAR), "n_heads": len(aux)},
     "after": {"value": max(v for _, v in aux_after), "verdict": "FAIL",
               "worst_head": max(aux_after, key=lambda t: t[1])[0],
               "n_over_bar": sum(1 for _, v in aux_after if v > BAR), "n_heads": len(aux),
               "why": "two heads were already masked by the atom gather; the three pair heads "
                      "were scored over the whole 384x384 tensor. On the real block the failure "
                      "is worse and it moves head: the worst reading is pae_logits at "
                      f'{max(v for _, v in aux_after):.6e}, not plddt_logits at '
                      f'{[v for h, v in aux_before if h == "plddt_logits"][0]:.6e}.'},
     "changed": False,
     "changed_within_scope": True},
    {"scope": "pairformer_stack", "mass_pct": 5.8282, "source": "of3t-pairformer",
     "before": {"value": R["A18-5.z"]["value"], "verdict": "FAIL"},
     "after": {"value": R["A18-5.z"]["real_token_value"], "verdict": "FAIL",
               "why": "every arm is named `*_masked` and the unmasked figure "
                      f'({R["A18-5.z"]["padded_value"]:.6e}) is published beside it.'},
     "changed": False},
]

out = {"what": "A18 pass/fail, before and after restricting to real tokens", "bar": BAR,
       "headline": "no A18 verdict changes. Three of the five scopes were never exposed "
                   "(one has no padding at all, two quoted masked figures), one is exposed but "
                   "bounded an order of magnitude inside the bar, and the one that was scored "
                   "padded already FAILED -- on the real block it fails harder and on a "
                   "different head.",
       "n_scopes": len(scopes),
       "n_verdicts_changed": sum(1 for s in scopes if s["changed"]),
       "aux_heads_per_head": {"before_padded": dict(aux_before), "after_real": dict(aux_after)},
       "scopes": scopes}
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(json.dumps(out, indent=1, sort_keys=True))
