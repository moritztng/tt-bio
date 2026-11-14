"""tt-bio's first sampler steps against upstream's, on upstream's own host fp32 trunk.

Second stage of the pairing bisect. `pair_tt_dump.py` showed the two sides agree on everything
before the noise except the randomized chain labels; this runs the real loop for a couple of
steps so the claim can be checked where it matters, on x after step 1 and step 2.

``--labels upstream`` substitutes upstream's ``chain_iid``/``molecule_iid`` (read out of
``seams.pt``) into our batch before anything else runs. That is the experiment, not a fix: it
isolates the labels from every other difference.

    PYTHONPATH=<worktree>:<worktree>/perf/rfd4/e2e /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/pair_tt_steps.py --labels upstream --steps 2 --out <dir>
"""
import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="/home/moritz/.coworker/state/rfd4/inputs/pdl1-128.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--labels", choices=("ours", "upstream"), default="ours")
    ap.add_argument("--upstream-seams", default="/home/moritz/rfd4-pair-runs/upstream/seed0/seams.pt")
    ap.add_argument("--out", default="/home/moritz/rfd4-pair-runs/tt/steps")
    ap.add_argument("--work", default="/home/moritz/rfd4-pair-runs/tt/inputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import sampler as S
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

    from network_parity import host_trunk, upstream_model  # same directory

    up = torch.load(a.upstream_seams, weights_only=False)

    designs = rfd4.prepare_examples(a.spec, a.work)
    features, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    batch = E.preprocess(features)
    if a.labels == "upstream":
        for key in ("chain_iid", "molecule_iid"):
            batch[key] = up["groups"][key].to(batch[key].dtype)
    n_tokens = batch["padding_mask"].shape[1]

    snapshot = glob.glob("/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/")[0]
    state, config = N.load_checkpoint(snapshot)
    model = upstream_model("nvidia/RFD4-Proteina")

    spec = S.recycle_strategy("thinking_medium")
    emit_masks = compute_flow_emit_masks(batch)
    groups = E._groups(batch)
    network = N.FlowNetwork(N.FlowGlue(state, config), host_trunk(model.nn), batch)
    chunked = S.ChunkedNetwork(network, spec, max_parallel_samples=1)

    rng = S.DesignRng(a.seed, 1, mode="upstream")
    torch.manual_seed(a.seed)
    with torch.no_grad():
        depths = S.recycle_depths(spec, spec.nsteps)
        ts, gt = S.sample_schedule(spec, spec.nsteps)
        x_0 = S.sample_noise(spec, n_tokens, rng, emit_masks=emit_masks,
                             coord_cond_mask=batch.get("coord_cond_mask"), groups=groups,
                             training=False)
        out = S.run_trajectory(spec, chunked, x=dict(x_0), ts=ts, gt=gt, nsteps=a.steps,
                               depths=depths[:a.steps], rng=rng, emit_masks=emit_masks,
                               coord_cond_mask=batch.get("coord_cond_mask"), groups=groups,
                               save_trajectory_every=1, dump_noisy_trajectories=True)

    dump = {"labels": a.labels, "x_0": x_0,
            **{f"x_{i + 1}": snap["x_t"] for i, snap in enumerate(out.trajectory)}}
    torch.save(dump, f"{a.out}/steps-{a.labels}.pt")

    report = {"labels": a.labels, "seed": a.seed, "steps": a.steps}
    for k in ["x_0"] + [f"x_{i + 1}" for i in range(a.steps)]:
        if k not in up or k not in dump:
            continue
        for mode in dump[k]:
            d = (up[k][mode].double() - dump[k][mode].double()).abs().max()
            report[f"{k}.{mode}.maxabs"] = float(d)
    json.dump(report, open(f"{a.out}/steps-{a.labels}.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
