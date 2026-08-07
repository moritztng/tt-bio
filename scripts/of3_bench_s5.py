"""S5 multi-target accuracy sweep: 8HEL / 7XI5 / 8JN0 from the OpenFold3 public
benchmark (monomer_protein.json), judged against upstream's own numbers for the same
155k checkpoint (state doc section 3, S5 table).

Per target: benchmark MSAs (uniref90/mgnify/uniprot/cfdb from the unsigned S3 bucket),
template cache npz when the query has one, the fixture-free device fold (200 steps x
5 samples, seed 1234), then sequence-aligned Ca-RMSD against the FoldBench reference
CIF plus pLDDT/pTM (directly comparable to the upstream table).

Run with the device env:
  PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_bench_s5.py [targets...]
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BENCH = os.path.expanduser("~/of3-bench")
TARGETS = {
    "8HEL": dict(msa="8hel_A", cif="8hel-assembly1.cif", upstream="TM 0.962 / pLDDT 72.8 / pTM 0.568"),
    "7XI5": dict(msa="7xi5_A", cif="7xi5-assembly1.cif", upstream="TM 0.976 / pLDDT 77.4 / pTM 0.598"),
    "8JN0": dict(msa="8jn0_A", cif="8jn0-assembly1.cif", upstream="TM 0.976 / pLDDT 79.2 / pTM 0.721"),
}


def aligned_ca_rmsd(pred_ca_xyz, pred_ca_seq, cif_path):
    """Ca-RMSD against the reference CIF, restricted to sequence-aligned positions.

    The FoldBench references resolve only part of the query (expression tags and
    disordered tails are missing), so a global sequence alignment picks the
    corresponding Ca pairs before the Kabsch fit.
    """
    import biotite.sequence as seq
    import biotite.sequence.align as align
    import biotite.structure.io.pdbx as pdbx

    from tt_bio.openfold3_fold import kabsch_rmsd

    f = pdbx.CIFFile.read(cif_path)
    arr = pdbx.get_structure(f, model=1)
    arr = arr[arr.chain_id == "A"]
    ca = arr[arr.atom_name == "CA"]
    # CIF residue sequence in res_id order.
    res_ids = np.unique(ca.res_id)
    cif_seq_1 = "".join(
        seq.ProteinSequence.convert_letter_3to1(ca[ca.res_id == r].res_name[0])
        for r in res_ids
    )
    q = seq.ProteinSequence(pred_ca_seq)
    r = seq.ProteinSequence(cif_seq_1)
    ali = align.align_optimal(
        q, r, align.SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1), max_number=1,
    )[0]
    pred_idx, cif_idx = [], []
    for qi_, ri_ in ali.trace:
        if qi_ >= 0 and ri_ >= 0:
            pred_idx.append(qi_)
            cif_idx.append(ri_)
    pred_sel = torch.from_numpy(np.array(pred_idx)).long()
    gt = torch.from_numpy(ca.coord[np.array(cif_idx)]).double()
    return kabsch_rmsd(pred_ca_xyz[pred_sel].double(), gt), len(pred_idx)


def main():
    targets = sys.argv[1:] or list(TARGETS)
    import ttnn

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features, make_openfold3_msa_features
    from tt_bio.openfold3_fold import OpenFold3
    from tt_bio.openfold3_host_prep import (
        derive_block_aux, derive_relpos, derive_template_feat, ref_atom_embed,
        run_input_atom_encoder,
    )
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import get_device

    mono = json.load(open("/tmp/monomer_protein.json"))["queries"]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    model = OpenFold3(sd, ckc, num_cycles=4)

    for name in targets:
        spec = TARGETS[name]
        entry = mono[name]
        chain = entry["chains"][0]
        chain["main_msa_file_paths"] = [os.path.join(BENCH, "benchmark_msas", spec["msa"])]
        if chain.get("template_alignment_file_path"):
            chain["template_alignment_file_path"] = os.path.join(
                BENCH, "benchmark_templates",
                os.path.basename(chain["template_alignment_file_path"]))
        qpath = f"/tmp/p13_query_{name}.json"
        with open(qpath, "w") as fh:
            json.dump({"queries": {name: entry}}, fh)

        torch.manual_seed(0)
        np.random.seed(0)
        query = next(iter(InferenceQuerySet.from_json(qpath).queries.values()))
        features = build_openfold3_features(query)
        msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
        aux = derive_block_aux(features)
        template_feat = derive_template_feat(features)
        relpos = derive_relpos(features)
        ai = run_input_atom_encoder(dev, ckc, sd, features, aux)
        s_input = torch.cat(
            [ai, features["restype"], features["profile"],
             features["deletion_mean"].unsqueeze(-1)], dim=-1)
        cl0, plm0 = ref_atom_embed(
            _sub(sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), features)
        dm_aux_host = dict(
            cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
            atom_to_token_index=aux["atom_to_token_index"],
            npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
            zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
            invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
            atom_to_token_mean=aux["atom_to_token_mean"], nb=aux["nb"], NP=aux["NP"])
        ca_mask = aux["ca_mask"]
        atom_to_token = aux["atom_to_token_index"].long()
        polymer_token = (features["is_protein"] | features["is_rna"] | features["is_dna"]).bool()
        confidence_aux = dict(
            representative_atom_indices=torch.from_numpy(np.flatnonzero(ca_mask.numpy())).long(),
            max_atom_per_token_mask=aux["max_atom_per_token_mask"],
            atom_array=features["atom_array"], asym_id=features["asym_id"],
            atom_to_token_index=atom_to_token, atom_mask=features["atom_mask"].bool(),
            polymer_mask=polymer_token[atom_to_token])

        raw_rows = int(features["msa"].shape[0])
        n_templates = int(template_feat["distogram"].shape[0])
        result = model.fold(
            template_feat=template_feat, msa_feat=msa_feat, s_input=s_input,
            relpos=relpos, token_bonds=features["token_bonds"],
            token_mask=features["token_mask"], dm_aux_host=dm_aux_host,
            n_atom=aux["n_atom"], n_token=aux["n_token"], no_rollout_steps=200,
            seed=1234, no_samples=5, confidence_aux_host=confidence_aux)

        # Query residue sequence at Ca positions (all protein tokens here).
        import biotite.sequence as _seq
        atom_array = features["atom_array"]
        ca_resnames = atom_array.res_name[ca_mask.numpy()]
        pred_ca_seq = "".join(
            _seq.ProteinSequence.convert_letter_3to1(r) for r in ca_resnames)
        rmsds, plddts, ptms = [], [], []
        for i, sample in enumerate(result.samples):
            rmsd, n_aln = aligned_ca_rmsd(
                sample[ca_mask], pred_ca_seq, os.path.join(BENCH, spec["cif"]))
            c = result.confidence[i]
            rmsds.append(rmsd)
            plddts.append(c["plddt"] * 100)
            ptms.append(c["ptm"])
            print(f"  {name} sample {i}: RMSD={rmsd:.4f} A (aligned {n_aln} Ca) "
                  f"pLDDT={c['plddt'] * 100:.2f} pTM={c['ptm']:.4f}")
        sel = result.best_index
        print(f"RESULT-{name}: selected={sel} selected_RMSD={rmsds[sel]:.4f} A "
              f"oracle_best={min(rmsds):.4f} A pLDDT(sel)={plddts[sel]:.2f} "
              f"pTM(sel)={ptms[sel]:.4f} msa_rows={raw_rows} templates={n_templates} "
              f"upstream[{spec['upstream']}]")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
