"""mol1-constraint golden cases on upstream's 7PZB example (MSAs from the example; contacts from the 7PZB crystal,
label_seq_id numbering). Usage: make_cases.py <example input.json> <msa_root> <out.json>"""
import copy, json, sys
ex = json.load(open(sys.argv[1]))[0]
for s in ex["sequences"]:
    p = s.get("proteinChain")
    if p:
        for k in ("pairedMsaPath", "unpairedMsaPath"):
            p[k] = p[k].replace("./examples/", sys.argv[2].rstrip("/") + "/")
base = {k: v for k, v in ex.items() if k != "constraints"}
site = ex["constraints"]["site"]                                   # A 86, 95, 140 -> PCG G (upstream's example)
excl = {"source_chains": ["A"], "partner_chains": ["G"], "residue_ids": [["A", 6], ["A", 21], ["A", 36], ["A", 156]]}
pairs = [[["A", 71], ["B", 149]], [["A", 91], ["B", 138]], [["A", 131], ["B", 90]], [["A", 230], ["B", 69]]]
cases = {
    "c1_site": {"site": site},
    "c2_exclusion": {"exclusion": excl},
    "c3_contact": {"contact": {"pairs": pairs}},
    "c4_combined": {"site": site, "contact": {"pairs": pairs[:2]},
                    "exclusion": {"source_chains": ["A"], "partner_chains": ["H"], "residue_ids": [["A", 6], ["A", 21]]}},
    "c5_none": None,
    # A 88 is token 87; token 540 is DNA chain F's first nucleotide (A, B 244 each, C, D 19, E 14), ~55 A from A 88
    # in the crystal, so this asks for a contact the native fold does not have.
    "c6_token_pairs": {"contact": {"token_pairs": [[87, 540]]}},
    # c2 is weak evidence: its residues sit 17-23 A from the ligand in the crystal too, so it is satisfied by doing
    # nothing. c7 is the real EXCLUSION test -- the four chain-A residues an UNCONSTRAINED Mol-1 fold of this input
    # packs hardest against ligand G (2.57-2.96 A, measured on tt-bio BH 2026-11-11) are asked to leave. Upstream
    # must run it too: on tt-bio the fold does not move them (ROOT.md c7), and only upstream says whether that is
    # the model or our port.
    "c7_exclusion_strong": {"exclusion": {"source_chains": ["A"], "partner_chains": ["G"],
                                          "residue_ids": [["A", 85], ["A", 86], ["A", 88], ["A", 96]]}},
}
out = []
for name, cons in cases.items():
    t = copy.deepcopy(base); t["name"] = name
    if cons:
        t["constraints"] = cons
    out.append(t)
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(len(out), "cases")
