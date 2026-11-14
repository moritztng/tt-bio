"""Public inputs for the upstream-featurizer parity test: each case is (the `sequences` of an
upstream OpenDDE 1.2.0 JSON sample, tt-bio build_complex_features kwargs for the same input)."""
P1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
P2 = "GSHMDNVSLKLGEWLQRGN"
CASES = {
    "no_template": (
        [{"proteinChain": {"sequence": P1, "count": 1}}],
        dict(chains=[(P1, None, "protein")], chain_ids=["A"])),
    "multi_chain": (
        [{"proteinChain": {"sequence": P1, "count": 1}}, {"proteinChain": {"sequence": P2, "count": 1}},
         {"dnaSequence": {"sequence": "GATTACA", "count": 1}}, {"rnaSequence": {"sequence": "GGCUAA", "count": 1}}],
        dict(chains=[(P1, None, "protein"), (P2, None, "protein"), ("GATTACA", None, "dna"),
                     ("GGCUAA", None, "rna")], chain_ids=["A", "B", "C", "D"])),
    "glycan": (
        [{"proteinChain": {"sequence": P2, "count": 1}}, {"ligand": {"ligand": "CCD_NAG_NAG_BMA", "count": 1}}],
        dict(chains=[(P2, None, "protein"), ("CCD_NAG_NAG_BMA", None, "ligand")], chain_ids=["A", "B"])),
    "modified_residue": (
        [{"proteinChain": {"sequence": P2, "count": 1,
                           "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 5},
                                             {"ptmType": "CCD_MLY", "ptmPosition": 10}]}}],
        dict(chains=[(P2, None, "protein")], chain_ids=["A"],
             modifications=[[{"position": 5, "ccd": "SEP"}, {"position": 10, "ccd": "MLY"}]])),
    "smiles_ligand": (
        [{"proteinChain": {"sequence": P1, "count": 1}}, {"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 1}}],
        dict(chains=[(P1, None, "protein"), ("CC(=O)Oc1ccccc1C(=O)O", None, "ligand")], chain_ids=["A", "B"])),
    "modified_nucleotide": (
        [{"rnaSequence": {"sequence": "GGCUAAC", "count": 1,
                          "modifications": [{"modificationType": "CCD_PSU", "basePosition": 4}]}}],
        dict(chains=[("GGCUAAC", None, "rna")], chain_ids=["A"],
             modifications=[[{"position": 4, "ccd": "PSU"}]])),
}
CASES["cyclic_peptide"] = (
    [{"proteinChain": {"sequence": P2, "count": 1}}],
    dict(chains=[(P2, None, "protein")], chain_ids=["A"],
         bonds=[(("A", 1, "N"), ("A", len(P2), "C"))]))
COVALENT_BONDS = {"cyclic_peptide": [{"entity1": 1, "copy1": 1, "position1": 1, "atom1": "N",
                 "entity2": 1, "copy2": 1, "position2": len(P2), "atom2": "C"}]}
