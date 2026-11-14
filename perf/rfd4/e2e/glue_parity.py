"""Grade tt-bio's host glue against upstream's own modules on one real batch.

Runs in rfd4-ref's CPU fp32 env, which has upstream installed, so both sides are built from the
same released checkpoint and fed the same featurized example. The trunk is not involved: this
covers the feature factories, the conditioning transitions, the pair initialiser and the heads,
which are the only RFD4 modules no other row owns.

    PYTHONPATH=<worktree>:<cli-lib> /home/moritz/rfd4-ref-env/venv/bin/python -I \
        perf/rfd4/e2e/glue_parity.py --spec <spec.yaml> --out <json>
"""
import argparse
import glob
import json

import torch


def upstream_glue(sd, cfg, latent_dim):
    from rfproteina.nn.feature_factory import FeatureFactory
    from rfproteina.nn.modules.norm import set_norm_type
    from rfproteina.nn.modules.pair_rep_initial import PairReprBuilder
    from rfproteina.nn.modules.seq_transition_af3 import Transition

    kw = dict(cfg, latent_dim=latent_dim)
    set_norm_type(kw.get("norm_type"))
    mods = {
        "init_repr_factory": FeatureFactory(feats=kw["feats_seq"], dim_feats_out=kw["token_dim"],
                                            use_ln_out=False, mode="seq", **kw),
        "cond_factory": FeatureFactory(feats=kw["feats_cond_seq"], dim_feats_out=kw["dim_cond"],
                                       use_ln_out=False, mode="seq", **kw),
        "transition_c_1": Transition(kw["dim_cond"], expansion_factor=2),
        "transition_c_2": Transition(kw["dim_cond"], expansion_factor=2),
        "pair_repr_builder": PairReprBuilder(feats_repr=kw["feats_pair_repr"],
                                             feats_cond=kw["feats_pair_cond"],
                                             dim_feats_out=kw["pair_repr_dim"],
                                             dim_cond_pair=kw["dim_cond"], **kw),
    }
    for name, mod in mods.items():
        mod.load_state_dict({k[len(f"nn.{name}."):]: v for k, v in sd.items()
                             if k.startswith(f"nn.{name}.")}, strict=False)
        mod.eval().requires_grad_(False)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/home/moritz/rfd4-e2e-runs/tt/glue")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import network as N

    designs = rfd4.prepare_examples(a.spec, f"{a.work}/inputs")
    feats, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    snapshot = glob.glob("/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/")[0]
    sd, cfg = N.load_checkpoint(snapshot)
    ours = N.FlowGlue(sd, cfg)
    theirs = upstream_glue(sd, cfg["nn"], cfg["autoencoder_config"]["nn_ae"]["latent_z_dim"])

    n = feats["padding_mask"].shape[1]
    gen = torch.Generator().manual_seed(a.seed)
    batch = dict(feats)
    batch["coords_nm"] = feats["coords"] / 10.0
    batch["coord_mask"] = feats["atom37_mask"]
    batch["x_t"] = {"bb_ca": torch.randn(1, n, 3, generator=gen),
                    "local_latents": torch.randn(1, n, cfg["autoencoder_config"]["nn_ae"]["latent_z_dim"],
                                                 generator=gen)}
    batch["t"] = {"bb_ca": torch.full((1,), 0.37), "local_latents": torch.full((1,), 0.37)}
    mask = batch["padding_mask"]

    results = {}
    with torch.no_grad():
        s_ours, z_ours = ours.initial_states(batch)
        s_theirs = theirs["init_repr_factory"](batch) * mask[..., None]
        z_theirs = theirs["pair_repr_builder"](batch)
        c_ours = ours.conditioning(batch, None)
        c_theirs = theirs["transition_c_2"](
            theirs["transition_c_1"](theirs["cond_factory"]({**batch, "recycle_remaining": None}), mask), mask)
        c_ours_r = ours.conditioning(batch, 3)
        c_theirs_r = theirs["transition_c_2"](
            theirs["transition_c_1"](theirs["cond_factory"]({**batch, "recycle_remaining": 3}), mask), mask)
        pairs = {"s_init": (s_ours, s_theirs), "z_init": (z_ours, z_theirs),
                 "cond": (c_ours, c_theirs), "cond_recycle_remaining_3": (c_ours_r, c_theirs_r)}
        # The heads read the trunk's s; upstream's are plain Linears on the same tensor, so a
        # random s of the right shape exercises them exactly as a real one would.
        s_probe = torch.randn(1, n, cfg["nn"]["token_dim"], generator=gen)
        head_out = ours.heads(s_probe, batch)
        from rfproteina.model_masks import compute_flow_emit_masks
        emit = compute_flow_emit_masks(batch)
        ca = (sd["nn.ca_linear.1.weight"], sd["nn.ca_linear.0.weight"])
        for name, got, want in (
            ("head_bb_ca", head_out["bb_ca"]["v"],
             torch.nn.functional.linear(_rms(s_probe, ca[1]), ca[0]) * emit["bb_ca"][..., None]),
            ("head_local_latents", head_out["local_latents"]["v"],
             torch.nn.functional.linear(_rms(s_probe, sd["nn.local_latents_linear.0.weight"]),
                                        sd["nn.local_latents_linear.1.weight"]) * emit["local_latents"][..., None]),
            ("head_sequence", head_out["sequence"]["logits"],
             torch.nn.functional.linear(_rms(s_probe, sd["nn.sequence_readout.0.weight"]),
                                        sd["nn.sequence_readout.1.weight"]) * mask[..., None]),
        ):
            pairs[name] = (got, want)

    for name, (got, want) in pairs.items():
        d = (got.double() - want.double()).abs()
        results[name] = {"max_abs": float(d.max()), "rel": float(d.max() / want.double().abs().max()),
                         "exact": bool(torch.equal(got, want)), "shape": list(got.shape)}
    verdict = "PASS" if all(v["exact"] for v in results.values()) else "CHECK"
    out = {"spec": a.spec, "seed": a.seed, "tokens": int(mask.sum()), "padded_to": n,
           "modules": results, "verdict": verdict}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


def _rms(x, w):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, 1e-5)


if __name__ == "__main__":
    main()
