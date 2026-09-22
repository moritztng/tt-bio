"""Ca-RMSD of the OpenFold3 fold against deposited structures, both pair-bias arms.

One process = one arm = one card. Weights and device are loaded once and every target and seed
runs inside it, so the two arms differ in exactly one boolean and nothing else.

  ship  the trunk Pairformer as main builds it: scale_pair_bias=False for both attentions, so
        the token pair bias arrives at 1/sqrt(24) = 0.204 of the reference value.
  fix   scale_pair_bias=True, tri_att_scale_pair_bias=False -- the architecture's value at the
        token site, bit-identical wiring at the two triangle sites.

`ship` is produced by wrapping the trunk's `Pairformer` and forcing the old arguments back, not
by checking out a second tree: same file, same process, one boolean apart. The wrapper counts
its own calls and the run asserts the count, because a monkeypatch that silently fails to bind
produces two identical arms and an A/B that cannot fail.

Every target's reference is the DEPOSITED structure, and the query sequence is read off that
structure's observed CA residues, so model token i and reference CA i are the same residue by
construction and no alignment step can drift between arms.

    python3 perf/of3t_pairbias/fold_targets.py --msa-only            # host only, no device
    TT_VISIBLE_DEVICES=N python3 perf/of3t_pairbias/fold_targets.py --arm ship --seeds 1,2,3
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

# Three levels: this file is perf/of3t_pairbias/<name>.py. Two levels lands on perf/ and the
# env then resolves `tt_bio` through its PEP 660 editable finder, which points at the SHARED
# checkout -- a run that looks like it measured this branch and did not. The assert in main()
# is what catches it; this is what makes it pass.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CACHE = os.path.expanduser("~/of3t_d1_targets")

# pdb id, chain, what it is. All X-ray, all single-chain folds, all under 130 residues so a
# seed sweep is affordable. 1ubq is the target the pair-bias question was first measured on and
# is here to connect this table to that one.
TARGETS = {
    "1ubq": ("1ubq", "A", "ubiquitin, beta-grasp"),
    "1shg": ("1shg", "A", "alpha-spectrin SH3 domain, beta-barrel"),
    "1bni": ("1bni", "A", "barnase, alpha+beta"),
    "3chy": ("3chy", "A", "CheY, Rossmann-like alpha/beta"),
}

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "MSE": "M", "SEC": "U", "PYL": "O"}


def fetch(pdb_id):
    path = os.path.join(CACHE, f"{pdb_id}.cif")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["curl", "-sfL", "-o", path,
                        f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"], check=True)
    return path


def observed_chain(pdb_id, chain):
    """(one-letter sequence, [N,3] CA coords) over the residues the deposit actually resolves.

    Reading the sequence off the structure rather than off SEQRES is what makes token i and
    reference CA i the same residue without an alignment: a crystallographic gap becomes a gap
    in the query too, identically in both arms.
    """
    import gemmi
    st = gemmi.read_structure(fetch(pdb_id))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    ch = next(c for c in st[0] if c.name == chain)
    seq, xyz = [], []
    for r in ch:
        a = r.find_atom("CA", "*")
        if a is None or r.name not in AA3:
            continue
        seq.append(AA3[r.name])
        xyz.append([a.pos.x, a.pos.y, a.pos.z])
    return "".join(seq), torch.tensor(xyz, dtype=torch.float64)


def query_json(name, seq):
    path = os.path.join(CACHE, f"{name}_query.json")
    with open(path, "w") as f:
        json.dump({"queries": {name: {"chains": [
            {"molecule_type": "protein", "chain_ids": ["A"], "sequence": seq}]}}}, f, indent=1)
    return path


def build_features(name, seq, msa_dir, msa_server_url):
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet)
    from tt_bio.openfold3_data import (build_openfold3_features, make_openfold3_msa_features,
                                       resolve_openfold3_msas)
    q = next(iter(InferenceQuerySet.from_json(query_json(name, seq)).queries.values()))
    resolve_openfold3_msas(q, msa_dir, target_id=name, msa_server_url=msa_server_url)
    feats = build_openfold3_features(q)
    return feats, make_openfold3_msa_features(feats, max_sequences=1024, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("ship", "fix"))
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--seeds", default="1,11,21,31,41,51")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--msa-dir", default=os.path.join(CACHE, "msa"))
    ap.add_argument("--msa-server-url", default="https://api.colabfold.com")
    ap.add_argument("--msa-only", action="store_true")
    ap.add_argument("--repeat-first", action="store_true",
                    help="re-fold the first (target, seed) to record the A/A floor")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = [t for t in a.targets.split(",") if t]
    seeds = sorted(int(s) for s in a.seeds.split(",") if s)
    # `fold` draws sample i of a run from seed + i (openfold3_fold.py, `_gen_rollout`), so
    # consecutive seeds are SLIDING WINDOWS over one noise stream, not independent folds:
    # at 5 samples, seeds 1 and 2 share four of their five structures and their rank-0 pair can
    # be the same structure twice. A seed floor built that way is not a seed floor. Anything
    # calling itself a replicate has to be at least `--samples` apart.
    gaps = [y - x for x, y in zip(seeds, seeds[1:])]
    assert not gaps or min(gaps) >= a.samples, (
        f"seeds {seeds} are closer than {a.samples} samples apart: gaps {gaps}")
    torch.manual_seed(0)
    np.random.seed(0)

    prepared = {}
    for n in names:
        pdb_id, chain, blurb = TARGETS[n]
        seq, gt = observed_chain(pdb_id, chain)
        feats, msa = build_features(n, seq, a.msa_dir, a.msa_server_url)
        prepared[n] = dict(seq=seq, gt=gt, feats=feats, msa=msa, blurb=blurb,
                           pdb_id=pdb_id, chain=chain)
        print(f"{n}: {len(seq)} observed residues ({blurb}), MSA raw {int(feats['msa'].shape[0])} "
              f"rows -> {msa.shape[0]} selected", flush=True)
    if a.msa_only:
        return
    assert a.arm, "--arm is required unless --msa-only"

    import ttnn
    import tt_bio
    assert os.path.realpath(tt_bio.__file__).startswith(os.path.realpath(REPO)), tt_bio.__file__
    import tt_bio.openfold3_trunk as of3_trunk
    from tt_bio.openfold3_fold import OpenFold3, kabsch_rmsd
    from tt_bio.openfold3_host_prep import (derive_block_aux, derive_relpos, derive_template_feat,
                                            ref_atom_embed, run_input_atom_encoder)
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import get_device

    patched = [0]
    if a.arm == "ship":
        real = of3_trunk.Pairformer

        def shipped_pairformer(*args_, **kw):
            kw["scale_pair_bias"] = False
            kw.pop("tri_att_scale_pair_bias", None)
            patched[0] += 1
            return real(*args_, **kw)

        of3_trunk.Pairformer = shipped_pairformer

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    model = OpenFold3(sd, ckc, num_cycles=4)
    if a.arm == "ship":
        assert patched[0] == 1, f"trunk Pairformer wrapper fired {patched[0]} times, expected 1"
    pf0 = model.trunk.pairformer.blocks[0]
    wiring = dict(apb_bias_scale=float(pf0.attention_pair_bias._bias_scale),
                  tri_bias_scale=float(pf0.triangle_attention_start._bias_scale),
                  tri_scale=float(pf0.triangle_attention_start.scale))
    print(f"arm={a.arm} wiring {wiring}", flush=True)

    out = a.out or f"perf/of3t_pairbias/folds_{a.arm}.json"
    # Resume: a relaunch re-runs only the legs this arm has not recorded yet. The device work is
    # the whole cost here, so losing finished folds to a restart is the difference between a
    # table that lands and one that never does.
    rows = json.load(open(out))["rows"] if os.path.exists(out) else []
    done = {(r["target"], r["seed"], r["repeat"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} legs already recorded in {out}", flush=True)
    for n in names:
        p = prepared[n]
        feats, msa_feat = p["feats"], p["msa"]
        aux = derive_block_aux(feats)
        ai = run_input_atom_encoder(dev, ckc, sd, feats, aux)
        s_input = torch.cat([ai, feats["restype"], feats["profile"],
                             feats["deletion_mean"].unsqueeze(-1)], dim=-1)
        cl0, plm0 = ref_atom_embed(
            _sub(sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), feats)
        dm_aux_host = dict(
            cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
            atom_to_token_index=aux["atom_to_token_index"], npe_q_indices=aux["npe_q_indices"],
            npe_k_indices=aux["npe_k_indices"], zij_mask=aux["zij_mask"],
            key_block_idxs=aux["key_block_idxs"], invalid_mask=aux["invalid_mask"],
            mask_trunked=aux["mask_trunked"], atom_to_token_mean=aux["atom_to_token_mean"],
            nb=aux["nb"], NP=aux["NP"])
        ca_mask = aux["ca_mask"]
        atom_to_token = aux["atom_to_token_index"].long()
        polymer_token = (feats["is_protein"] | feats["is_rna"] | feats["is_dna"]).bool()
        confidence_aux = dict(
            representative_atom_indices=torch.from_numpy(
                np.flatnonzero(ca_mask.numpy())).long(),
            max_atom_per_token_mask=aux["max_atom_per_token_mask"],
            atom_array=feats["atom_array"], asym_id=feats["asym_id"],
            atom_to_token_index=atom_to_token, atom_mask=feats["atom_mask"].bool(),
            polymer_mask=polymer_token[atom_to_token])
        common = dict(template_feat=derive_template_feat(feats), msa_feat=msa_feat,
                      s_input=s_input, relpos=derive_relpos(feats),
                      token_bonds=feats["token_bonds"], token_mask=feats["token_mask"],
                      dm_aux_host=dm_aux_host, n_atom=aux["n_atom"], n_token=aux["n_token"],
                      no_rollout_steps=a.steps, no_samples=a.samples,
                      confidence_aux_host=confidence_aux)
        gt = p["gt"]
        assert int(ca_mask.sum()) == gt.shape[0], (int(ca_mask.sum()), gt.shape[0])

        legs = [(s, 0) for s in seeds]
        if a.repeat_first and n == names[0]:
            legs.append((seeds[0], 1))
        for seed, rep in legs:
            if (n, seed, rep) in done:
                continue
            t0 = time.time()
            r = model.fold(seed=seed, **common)
            ca = [s[ca_mask].double() for s in r.samples]
            assert all(torch.isfinite(c).all() for c in ca), f"non-finite sample {n} s{seed}"
            rm = [kabsch_rmsd(c, gt) for c in ca]
            c0 = r.confidence[r.best_index]
            row = dict(arm=a.arm, target=n, seed=seed, repeat=rep, n_ca=int(ca_mask.sum()),
                       rank0=float(rm[r.best_index]), best=float(min(rm)),
                       rmsd=[float(x) for x in rm], best_index=int(r.best_index),
                       ranking=float(c0["ranking_score"]), ptm=float(c0["ptm"]),
                       plddt=float(c0["plddt"]),
                       ca_xyz=[[[round(float(v), 4) for v in xyz] for xyz in c.tolist()]
                               for c in ca],
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(f"  {n} seed {seed}{' (repeat)' if rep else ''}: rank0 {row['rank0']:.4f} A  "
                  f"best {row['best']:.4f} A  pLDDT {row['plddt']:.4f}  pTM {row['ptm']:.4f}  "
                  f"{row['seconds']:.0f} s", flush=True)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(dict(arm=a.arm, card=os.environ.get("TT_VISIBLE_DEVICES"),
                           samples=a.samples, steps=a.steps, wiring=wiring,
                           targets={k: dict(pdb_id=v["pdb_id"], chain=v["chain"],
                                            blurb=v["blurb"], n_res=len(v["seq"]),
                                            msa_rows=int(v["msa"].shape[0]))
                                    for k, v in prepared.items()},
                           rows=rows), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
