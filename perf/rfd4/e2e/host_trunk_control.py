"""Does our sampler pair with upstream's when the trunk is not the card's?

Our 128-token design lands ~3.3 A from upstream's run of the same input and the same seed, while
upstream's own two runs of that seed land 0.15 A apart. Two things could do that and they want
opposite fixes:

1. **bf16.** The device trunk is bf16 where upstream's is fp32. Over 180 trunk passes of a chaotic
   flow that is enough to end somewhere else, and nothing is wrong -- the model is graded
   distributionally for exactly this reason.
2. **The pairing.** Our noise draws are not upstream's, because the RNG order or the sampler differs.
   That would be a bug, and no amount of distributional grading would excuse it.

This separates them without a card. It runs the same sampler against **upstream's own host fp32
trunk layers**, the ones `network_parity.py` already grades the wiring against, and compares the
sampled CA track -- not the decoded structure, so no autoencoder and no device are needed -- with
upstream's CIF. If this lands near upstream, cause 1 is the answer and the port is fine. If it still
lands near 3 A, cause 2 is the answer and the pairing is the bug.

    PYTHONPATH=<worktree>:<cli-lib> /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/host_trunk_control.py --spec <spec.yaml> --upstream <cif> --out <json>
"""
import argparse
import glob
import json

import numpy as np
import torch


def upstream_ca(path):
    """Chain B's CA coordinates in Angstrom, and chain A's, from upstream's design."""
    import biotite.structure.io.pdbx as pdbx

    arr = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    ca = arr[arr.atom_name == "CA"]
    return {c: ca[ca.chain_id == c].coord for c in np.unique(ca.chain_id)}


def _reading(features: str, rmsd: float) -> str:
    """What the measured RMSD says, which depends on whose features the run used."""
    if features == "upstream":
        return ("the sampler, the RNG order and the wiring pair: with the featurizer's randomized "
                "tensors held equal this reproduces upstream's design" if rmsd < 0.5 else
                "a divergence that survives equal features, so it is downstream of the featurizer")
    return ("bf16 on the card: the port is sound and the single-design Angstrom grade cannot see it"
            if rmsd < 1.0 else
            "the featurizer's randomized chain labels and internal-coordinate frame, which are not "
            "a function of the seed on upstream's side -- see state/rfd4/PAIR.md, and rerun with "
            "--features upstream to pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/home/moritz/rfd4-e2e-runs/tt/control")
    ap.add_argument("--features", choices=("ours", "upstream"), default="ours",
                    help="upstream: take the featurizer's ambient-stream draws (the chain and "
                         "molecule labels and the internal-coordinate frame) from an upstream "
                         "seam dump, which is the only way to hold them equal across the two "
                         "implementations -- see state/rfd4/PAIR.md")
    ap.add_argument("--upstream-seams",
                    default="/home/moritz/rfd4-pair-runs/upstream/seed0/seams.pt")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import sampler as S
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

    from network_parity import host_trunk, upstream_model  # same directory

    designs = rfd4.prepare_examples(a.spec, f"{a.work}/inputs")
    features, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    batch = E.preprocess(features)
    if a.features == "upstream":
        up_seams = torch.load(a.upstream_seams, weights_only=False)
        for key in ("chain_iid", "molecule_iid", "condition_InternalCoordinate_1_atom"):
            src = next(v for k, v in up_seams["batch"].items() if k == key or k.endswith("." + key))
            batch[key] = src.to(batch[key].dtype)
    n_tokens = batch["padding_mask"].shape[1]

    snapshot = glob.glob("/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/")[0]
    state, config = N.load_checkpoint(snapshot)
    model = upstream_model("nvidia/RFD4-Proteina")

    spec = S.recycle_strategy("thinking_medium")
    network = N.FlowNetwork(N.FlowGlue(state, config), host_trunk(model.nn), batch)
    chunked = S.ChunkedNetwork(network, spec, max_parallel_samples=1)
    torch.manual_seed(a.seed)                      # upstream's own entry point, as capture.py reseeds
    with torch.no_grad():
        out = S.generate(spec, chunked, n_designs=1, n_tokens=n_tokens, seed=a.seed,
                         rng_mode="upstream", emit_masks=compute_flow_emit_masks(batch),
                         groups=E._groups(batch), coord_cond_mask=batch.get("coord_cond_mask"))

    ours_ca = out.x["bb_ca"][0].double().numpy() * 10.0   # nm -> Angstrom
    up = upstream_ca(a.upstream)
    designed = min(up, key=lambda c: len(up[c]))
    n_binder = len(up[designed])
    # The designed chain is the first `n_binder` tokens: the contig puts the binder before the target.
    ours_binder, ours_target = ours_ca[:n_binder], ours_ca[n_binder:]
    up_binder = up[designed].astype(float)
    up_target = np.concatenate([up[c] for c in up if c != designed]).astype(float)

    def kabsch(p, q):
        pc, qc = p.mean(0), q.mean(0)
        u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1, 1, d]) @ u.T
        return r, qc - pc @ r.T

    r, t = kabsch(ours_target[:len(up_target)], up_target)
    rmsd = float(np.sqrt((((ours_binder @ r.T + t) - up_binder) ** 2).sum(1).mean()))
    tgt = float(np.sqrt((((ours_target[:len(up_target)] @ r.T + t) - up_target) ** 2).sum(1).mean()))
    res = {"seed": a.seed, "tokens": n_tokens, "features": a.features, "binder_residues": n_binder,
           "trunk": "upstream's own host fp32 layers", "sampler": "tt-bio, rng_mode=upstream",
           "binder_ca_rmsd_a": round(rmsd, 3), "target_ca_rmsd_a": round(tgt, 3),
           "device_run_rmsd_a": 3.315,
           "reading": _reading(a.features, rmsd)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
