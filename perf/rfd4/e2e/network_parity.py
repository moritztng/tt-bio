"""Grade one whole network call against upstream's `LocalLatentsTransformer.forward`.

`glue_parity.py` covers the modules around the trunk one at a time. This covers the call the
sampler actually makes: the glue, the recycle loop and the heads, at a recycle depth above one,
where the conditioning is rebuilt per cycle from the remaining-pass count and the states come
back through the zero-initialised cycle projections.

It does not come out bit-exact, and the reason is measured rather than tolerated: upstream's pair
norm is a `CompiledRMSNorm`, ours the eager one, which puts z_init one fp32 ULP apart on identical
weights and input (s_init and the conditioning are bit-identical). Three recycles turn that into
2.5e-7 at the heads, four orders below the bf16 floor the device is graded on, so the bar here is
1e-5 relative.

The trunk is upstream's own layers on the host, so what is being graded is tt-bio's wiring
(`FlowNetwork`), not the device port: `tests/test_rfd4_trunk_parity.py` grades the device trunk
against the same weights, and the two together cover the network.

    PYTHONPATH=<worktree>:<cli-lib> /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/network_parity.py --spec <spec.yaml> --depth 3 --out <json>
"""
import argparse
import glob
import json

import torch


def upstream_model(snapshot):
    """Upstream's released model, built the way `rfd4.generate` builds it."""
    import rfd4_proteina as rfd4

    model = rfd4.Proteina.from_pretrained(snapshot)
    model.eval().requires_grad_(False)
    return model


def host_trunk(nn):
    """`(s_init, z_init, conds, mask) -> (s, z)` over upstream's own trunk layers.

    Mirrors `_run_trunk`: each cycle starts from the initial states plus a projection of the
    previous cycle's, and the first cycle's projection is exactly zero because rms(0) = 0.
    """
    def run(s_init, z_init, conds, mask):
        seq_mask = mask.bool()
        s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
        for cond in conds:
            # Upstream runs the projections on every cycle, including the first, where they are
            # exactly zero. The .contiguous() is upstream's: SDPA reads the pair bias from z, and
            # the layout it reads it in changes the last bits of the result.
            z_in = (z_init + nn.z_cycle_proj(nn.z_cycle_norm(z))).contiguous()
            s_in = s_init + nn.s_cycle_proj(nn.s_cycle_norm(s))
            s, z = s_in, z_in
            for i in range(nn.nlayers):
                s = nn.transformer_layers[i](s, z, cond, seq_mask)
                if nn.update_pair_repr and i < nn.nlayers - 1 and nn.pair_update_layers[i] is not None:
                    z = nn.pair_update_layers[i](s, z, seq_mask)
        return s, z

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3, help="recycle depth of the graded call")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/home/moritz/rfd4-e2e-runs/tt/network")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4 import network as N

    designs = rfd4.prepare_examples(a.spec, f"{a.work}/inputs")
    features, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    batch = E.preprocess(features)
    n = batch["padding_mask"].shape[1]

    snapshot = glob.glob("/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/*/")[0]
    state, config = N.load_checkpoint(snapshot)
    latent_dim = config["autoencoder_config"]["nn_ae"]["latent_z_dim"]
    model = upstream_model("nvidia/RFD4-Proteina")

    gen = torch.Generator().manual_seed(a.seed)
    x_t = {"bb_ca": torch.randn(1, n, 3, generator=gen),
           "local_latents": torch.randn(1, n, latent_dim, generator=gen)}
    t = {"bb_ca": 0.37, "local_latents": 0.37}

    with torch.no_grad():
        theirs = model.nn({**batch, "x_t": x_t,
                           "t": {k: torch.full((1,), v) for k, v in t.items()},
                           "n_recycle": a.depth, "condition_on_recycle_remaining": True})
        ours = N.FlowNetwork(N.FlowGlue(state, config), host_trunk(model.nn), batch)(
            x_t, t, recycle_depth=a.depth, step=0)

    results = {}
    for head, field in (("bb_ca", "v"), ("local_latents", "v"), ("sequence", "logits"),
                        ("insertion", "logits")):
        got, want = ours[head][field], theirs[head][field]
        d = (got.double() - want.double()).abs()
        results[f"{head}.{field}"] = {"max_abs": float(d.max()),
                                      "rel": float(d.max() / want.double().abs().max()),
                                      "exact": bool(torch.equal(got, want)),
                                      "shape": list(got.shape)}
    rel = max(v["rel"] for v in results.values())
    out = {"spec": a.spec, "seed": a.seed, "recycle_depth": a.depth, "tokens": n, "bar_rel": 1e-5,
           "trunk": "upstream host layers (the device trunk is graded by tests/test_rfd4_trunk_parity.py)",
           "heads": results,
           "verdict": "PASS" if rel < 1e-5 else "FAIL"}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
